"""The UNDER-budget half of the narration word budget.

Every assertion here fails on ``origin/main``: the budget was enforced as a
ceiling only, a malformed polish response was discarded in silence, the prompt
led with "polish … preserve the structure" even when the defect was length, and
the template was blind to the budget it was written against. Measured live
(orchestrator runs 01a0a136 / 01a09e7d) this service returned ~58% of its word
budget on every beat, and the orchestrator stretched the speech to a 0.80 rate
floor to fill the slot — audible as narration that "weirdly slows down".
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.generation.llm_client import LLMClient
from src.services.job_processor import JobProcessor, _word_budget_report
from src.services.narration.humanizer import (
    POLISH_OUTPUT_SCHEMA,
    NarrationHumanizer,
    PolishReport,
)
from src.services.narration.template_engine import NarrationTemplateEngine

MODULE = "src.services.job_processor"


def _parts() -> list[dict[str, Any]]:
    return [
        {"part_id": "clip-1", "scenario_name": "Open the dashboard", "text": "Short line one."},
        {"part_id": "clip-2", "scenario_name": "Create a project", "text": "Short line two."},
    ]


def _echo(parts: list[dict[str, Any]], words: int) -> str:
    """A faithful ``[Name]`` echo whose bodies are ``words`` words long."""
    return "\n\n".join(f"[{p['scenario_name']}]\n{' '.join(['word'] * words)}." for p in parts)


# ---------------------------------------------------------------------------
# 1b — a discarded polish response is VISIBLE, COUNTED and RETRIED
# ---------------------------------------------------------------------------


class TestMalformedResponseIsVisible:
    async def test_unfaithful_response_logs_retries_and_counts(self, caplog):
        client = AsyncMock(spec=LLMClient)
        client.generate = AsyncMock(return_value="Sure! Here is your polished script.")
        humanizer = NarrationHumanizer(client)

        with caplog.at_level(logging.WARNING):
            parts, report = await humanizer.polish_with_report(_parts(), "tutorial")

        # Retried rather than giving up on the first bad answer.
        assert client.generate.await_count == 2
        assert report == PolishReport(mode="polish", attempts=2, parse_failures=2, applied=False)
        # Template text kept — but no longer in silence.
        assert [p["text"] for p in parts] == [p["text"] for p in _parts()]
        discarded = [r.getMessage() for r in caplog.records if "DISCARDED" in r.getMessage()]
        assert len(discarded) == 2
        assert "Open the dashboard" in discarded[0]
        assert "Sure! Here is your polished script."[:20] in discarded[0]

    async def test_second_attempt_is_accepted(self):
        parts_in = _parts()
        client = AsyncMock(spec=LLMClient)
        client.generate = AsyncMock(side_effect=["garbage", _echo(parts_in, 9)])
        humanizer = NarrationHumanizer(client)

        parts, report = await humanizer.polish_with_report(parts_in, "tutorial")

        assert report.applied is True
        assert report.attempts == 2
        assert report.parse_failures == 1
        assert all(len(p["text"].split()) == 9 for p in parts)


# ---------------------------------------------------------------------------
# 1c — the prompt leads with LENGTH, and the call carries an output schema
# ---------------------------------------------------------------------------


class TestExpansionLedPrompt:
    async def test_short_section_makes_length_the_primary_instruction(self):
        client = AsyncMock(spec=LLMClient)
        client.generate = AsyncMock(return_value="")
        humanizer = NarrationHumanizer(client)

        await humanizer.polish_with_report(_parts(), "tutorial", word_budgets=[24, 24])

        prompt = client.generate.call_args.args[0]
        assert prompt.startswith("REWRITE")
        # The length requirement precedes the polish instruction, not the reverse.
        assert prompt.index("AS LONG AS ITS WORD BUDGET") < prompt.index("Secondarily")
        assert "ADD about" in prompt

    async def test_in_budget_sections_keep_the_polish_lead(self):
        parts = [{"scenario_name": "A", "text": "one two three four five"}]
        client = AsyncMock(spec=LLMClient)
        client.generate = AsyncMock(return_value="")
        humanizer = NarrationHumanizer(client)

        await humanizer.polish_with_report(parts, "tutorial", word_budgets=[5])

        assert client.generate.call_args.args[0].startswith("Polish the following")

    async def test_call_carries_the_output_schema(self):
        client = AsyncMock(spec=LLMClient)
        client.generate = AsyncMock(return_value="")
        humanizer = NarrationHumanizer(client)

        await humanizer.polish_with_report(_parts(), "tutorial")

        assert client.generate.call_args.kwargs["output_schema"] is POLISH_OUTPUT_SCHEMA

    async def test_schema_shaped_json_response_is_accepted(self):
        parts_in = _parts()
        payload = json.dumps(
            {
                "sections": [
                    {"scenario_name": p["scenario_name"], "text": f"Expanded {p['part_id']}."}
                    for p in parts_in
                ]
            }
        )
        client = AsyncMock(spec=LLMClient)
        client.generate = AsyncMock(return_value=payload)
        humanizer = NarrationHumanizer(client)

        parts, report = await humanizer.polish_with_report(parts_in, "tutorial")

        assert report.applied is True
        assert [p["text"] for p in parts] == ["Expanded clip-1.", "Expanded clip-2."]

    async def test_structured_response_naming_an_unknown_scenario_is_rejected(self):
        payload = json.dumps({"sections": [{"scenario_name": "Somewhere else", "text": "x"}]})
        client = AsyncMock(spec=LLMClient)
        client.generate = AsyncMock(return_value=payload)
        humanizer = NarrationHumanizer(client)

        parts, report = await humanizer.polish_with_report(_parts(), "tutorial")

        assert report.applied is False
        assert [p["text"] for p in parts] == [p["text"] for p in _parts()]


# ---------------------------------------------------------------------------
# 1a — a SHORT section is re-asked, and the residual is structured output
# ---------------------------------------------------------------------------


@pytest.fixture
def processor():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="mocked LLM response")
    with patch(f"{MODULE}.LLMClient", return_value=llm):
        yield JobProcessor()


class TestShortSectionsAreExpanded:
    async def test_under_budget_section_is_re_asked_and_grows(self, processor: JobProcessor):
        parts = [
            {"part_id": "clip-1", "scenario_name": "A", "text": "one two three", "word_count": 3},
            {
                "part_id": "clip-2",
                "scenario_name": "B",
                "text": " ".join(["w"] * 24),
                "word_count": 24,
            },
        ]
        expanded = [{"part_id": "clip-1", "scenario_name": "A", "text": " ".join(["x"] * 20)}]
        expand = AsyncMock(
            return_value=(expanded, PolishReport(mode="expand", attempts=1, applied=True))
        )

        with patch.object(processor._humanizer, "expand_to_budget", expand):
            out, report = await processor._expand_short_sections(
                parts, "tutorial", [24, 24], None, 150
            )

        # ONLY the short section was re-sent.
        sent_parts, _tone, sent_budgets = expand.call_args.args
        assert [p["part_id"] for p in sent_parts] == ["clip-1"]
        assert sent_budgets == [24]
        assert out[0]["word_count"] == 20
        assert out[1]["text"] == " ".join(["w"] * 24)  # untouched
        assert report is not None and report.applied is True

    async def test_a_shorter_re_ask_is_dropped(self, processor: JobProcessor):
        parts = [
            {
                "part_id": "clip-1",
                "scenario_name": "A",
                "text": "one two three four five",
                "word_count": 5,
            }
        ]
        expand = AsyncMock(
            return_value=(
                [{"part_id": "clip-1", "scenario_name": "A", "text": "two words"}],
                PolishReport(mode="expand", attempts=1, applied=True),
            )
        )

        with patch.object(processor._humanizer, "expand_to_budget", expand):
            out, _ = await processor._expand_short_sections(parts, "tutorial", [24], None, 150)

        assert out[0]["text"] == "one two three four five"

    async def test_no_short_section_makes_no_call(self, processor: JobProcessor):
        parts = [
            {
                "part_id": "clip-1",
                "scenario_name": "A",
                "text": " ".join(["w"] * 24),
                "word_count": 24,
            }
        ]
        expand = AsyncMock()

        with patch.object(processor._humanizer, "expand_to_budget", expand):
            out, report = await processor._expand_short_sections(parts, "tutorial", [24], None, 150)

        expand.assert_not_awaited()
        assert report is None
        assert out is parts


class TestReaskOutcomeIsLoggedPerSection:
    """One ``narration_reask_section`` INFO line per re-asked section — counts, no text."""

    async def test_replaced_and_dropped_sections_each_log_their_outcome(
        self, processor: JobProcessor, caplog
    ):
        parts = [
            {"part_id": "clip-1", "scenario_name": "A", "text": "one two three", "word_count": 3},
            {
                "part_id": "clip-2",
                "scenario_name": "B",
                "text": "alpha beta gamma delta",
                "word_count": 4,
            },
        ]
        expanded = [
            {"part_id": "clip-1", "scenario_name": "A", "text": " ".join(["x"] * 20)},
            {"part_id": "clip-2", "scenario_name": "B", "text": "secret words"},
        ]
        expand = AsyncMock(
            return_value=(expanded, PolishReport(mode="expand", attempts=1, applied=True))
        )

        with (
            patch.object(processor._humanizer, "expand_to_budget", expand),
            caplog.at_level(logging.INFO, logger=MODULE),
        ):
            out, _ = await processor._expand_short_sections(
                parts, "tutorial", [24, 24], None, 150, job_id="job-42"
            )

        lines = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("narration_reask_section")
        ]
        assert all(
            r.levelno == logging.INFO
            for r in caplog.records
            if r.getMessage().startswith("narration_reask_section")
        )
        assert lines == [
            "narration_reask_section job_id=job-42 section=clip-1 budget_words=24 "
            "before_words=3 after_words=20 replaced=true",
            "narration_reask_section job_id=job-42 section=clip-2 budget_words=24 "
            "before_words=4 after_words=2 replaced=false",
        ]
        # Behaviour unchanged: the longer answer landed, the shorter one was dropped.
        assert out[0]["word_count"] == 20
        assert out[1]["text"] == "alpha beta gamma delta"
        # Word counts only — the narration text never reaches the log.
        assert not any("secret" in line or "alpha" in line for line in lines)


class TestResidualShortfallIsStructured:
    def test_report_carries_per_section_shortfall(self):
        parts = [
            {"part_id": "clip-1", "scenario_name": "A", "word_count": 14},
            {"part_id": "clip-2", "scenario_name": "B", "word_count": 24},
        ]
        report = _word_budget_report(parts, [24, 24], None, None, None)

        assert report is not None
        assert report["budgeted_words"] == 48
        assert report["delivered_words"] == 38
        assert report["shortfall_words"] == 10
        assert report["sections_short"] == [
            {
                "part_id": "clip-1",
                "scenario_name": "A",
                "word_count": 14,
                "word_budget": 24,
                "shortfall_words": 10,
            }
        ]

    def test_no_budget_and_no_pass_reports_nothing(self):
        parts = [{"part_id": "clip-1", "scenario_name": "A", "word_count": 14}]
        assert _word_budget_report(parts, None, None, None, None) is None

    def test_pass_reports_are_carried(self):
        report = _word_budget_report(
            [{"part_id": "clip-1", "scenario_name": "A", "word_count": 14}],
            [24],
            None,
            PolishReport(mode="polish", attempts=2, parse_failures=1, applied=True),
            PolishReport(mode="expand", attempts=1, parse_failures=0, applied=True),
        )
        assert report is not None
        polish = report["polish"]
        expansion = report["expansion"]
        assert polish is not None
        assert expansion is not None
        assert polish["parse_failures"] == 1
        assert expansion["mode"] == "expand"

    async def test_generated_script_publishes_the_report(self, processor: JobProcessor):
        features = [
            {
                "content": (
                    "Feature: Tour\n"
                    "  @clip-1\n"
                    "  Scenario: Open the dashboard\n"
                    "    Given I am on the login page\n"
                    "    Then I see the dashboard\n"
                )
            }
        ]
        script = await processor._generate_narration(
            features,
            {"enabled": True, "tone": "tutorial", "llm_polish": False},
            recording=True,
            word_budgets=[200],
        )

        assert script["word_budget_report"]["budgeted_words"] == 200
        assert script["word_budget_report"]["sections_short"][0]["part_id"] == "clip-1"


# ---------------------------------------------------------------------------
# 1d — the TEMPLATE writes against the budget it is given
# ---------------------------------------------------------------------------


class TestTemplateIsBudgetAware:
    def _steps(self) -> list[dict[str, str]]:
        return [
            {"keyword": "Given", "text": "I am on the login page"},
            {"keyword": "Then", "text": "I see the dashboard"},
        ]

    def test_budget_aware_output_is_longer(self):
        engine = NarrationTemplateEngine(tone="tutorial")
        blind = engine.humanize_scenario("Open the dashboard", self._steps())
        budgeted = engine.humanize_scenario("Open the dashboard", self._steps(), word_budget=40)

        assert len(budgeted.split()) > len(blind.split())
        assert budgeted.endswith(blind)
        assert "Open the dashboard" in budgeted

    def test_no_budget_is_byte_identical(self):
        engine = NarrationTemplateEngine(tone="tutorial")
        assert engine.humanize_scenario("X", self._steps()) == engine.humanize_scenario(
            "X", self._steps(), word_budget=None
        )

    def test_lead_in_never_exceeds_the_budget(self):
        engine = NarrationTemplateEngine(tone="tutorial")
        blind = engine.humanize_scenario("Open the dashboard", self._steps())
        tight = engine.humanize_scenario(
            "Open the dashboard", self._steps(), word_budget=len(blind.split()) + 1
        )
        assert tight == blind
