"""The fixed per-tone fallback template must be OBSERVABLE, not silent.

When no narration pattern matches a step, the engine emits a canned sentence
("Next, {step_text}."). These tests pin that the text is unchanged AND that the
fallback leaves a log record, a per-part marker and a job-level summary.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.schemas.narration import NarrationPart
from src.services.job_processor import JobProcessor, _log_narration_sources
from src.services.narration.template_engine import NarrationTemplateEngine

MODULE = "src.services.job_processor"
UNMATCHED = "the frobnicator hums quietly"


@pytest.fixture
def processor():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="mocked LLM response")
    with patch(f"{MODULE}.LLMClient", return_value=llm):
        yield JobProcessor()


def test_unmatched_step_uses_fallback_and_logs(caplog):
    engine = NarrationTemplateEngine(tone="tutorial")
    with caplog.at_level(logging.WARNING):
        out = engine.humanize_step("When", UNMATCHED)

    assert out == f"Next, {UNMATCHED}."  # behaviour unchanged
    assert engine.fallback_steps == 1
    recs = [r for r in caplog.records if r.getMessage().startswith("narration_template_fallback")]
    assert len(recs) == 1
    rec = recs[0]
    assert rec.levelno == logging.WARNING
    assert rec.__dict__["narration_source"] == "template_fallback"
    assert rec.__dict__["tone"] == "tutorial"
    assert rec.__dict__["step_keyword"] == "When"
    assert rec.__dict__["step_text_len"] == len(UNMATCHED)
    assert UNMATCHED not in rec.getMessage()  # length only, never the text


def test_matched_step_does_not_count_as_fallback(caplog):
    engine = NarrationTemplateEngine(tone="tutorial")
    with caplog.at_level(logging.WARNING):
        engine.humanize_step("When", "I navigate to settings")
    assert engine.fallback_steps == 0
    assert not [r for r in caplog.records if "narration_template_fallback" in r.getMessage()]


async def test_parts_carry_narration_source(processor: JobProcessor, caplog):
    feature = (
        "Feature: F\n\n"
        "  @clip-1\n  Scenario: Canned\n    When " + UNMATCHED + "\n\n"
        "  @clip-2\n  Scenario: Matched\n    When I navigate to settings\n"
    )
    with caplog.at_level(logging.WARNING):
        script = await processor._generate_narration(
            [{"content": feature}], {"tone": "tutorial", "llm_polish": False}
        )

    canned, matched = script["parts"]
    assert canned["text"] == f"Next, {UNMATCHED}."
    assert canned["narration_source"] == "template_fallback"
    assert canned["template_fallback_steps"] == 1
    assert matched["narration_source"] == "pattern"
    assert matched["template_fallback_steps"] == 0
    # The marker survives the published (closed) API model.
    assert NarrationPart.model_validate(canned).narration_source == "template_fallback"


def test_job_level_summary_logs_fallback_count(caplog):
    script = {
        "parts": [
            {"narration_source": "template_fallback", "template_fallback_steps": 2},
            {"narration_source": "pattern", "template_fallback_steps": 0},
            {"narration_source": "authored"},
        ]
    }
    with caplog.at_level(logging.INFO):
        _log_narration_sources("job-1", script)
    msgs = [r for r in caplog.records if r.getMessage().startswith("narration_source_summary")]
    assert len(msgs) == 1
    assert msgs[0].levelno == logging.WARNING
    assert msgs[0].getMessage() == (
        "narration_source_summary job_id=job-1 total_lines=3 template_fallback_lines=1 "
        "template_fallback_steps=2 llm_polished=False"
    )


def test_summary_is_info_when_polish_rewrote_the_fallback_lines(caplog):
    """Live 2026-10-18: 9/9 pre-polish fallback lines, polish APPLIED -> not a WARNING.

    ``narration_source`` is pre-polish provenance; once the LLM pass rewrote every
    line, no canned text ships, so the summary must not read as an LLM failure.
    """
    script = {
        "parts": [{"narration_source": "template_fallback", "template_fallback_steps": 6}] * 9,
        "word_budget_report": {"polish": {"mode": "polish", "attempts": 1, "applied": True}},
    }
    with caplog.at_level(logging.INFO):
        _log_narration_sources("job-2", script)
    (rec,) = [r for r in caplog.records if r.getMessage().startswith("narration_source_summary")]
    assert rec.levelno == logging.INFO
    assert rec.getMessage().endswith(
        "template_fallback_lines=9 template_fallback_steps=54 llm_polished=True"
    )


def test_summary_warns_when_polish_was_not_applied(caplog):
    script = {
        "parts": [{"narration_source": "template_fallback", "template_fallback_steps": 1}],
        "word_budget_report": {"polish": {"mode": "polish", "attempts": 2, "applied": False}},
    }
    with caplog.at_level(logging.INFO):
        _log_narration_sources("job-3", script)
    (rec,) = [r for r in caplog.records if r.getMessage().startswith("narration_source_summary")]
    assert rec.levelno == logging.WARNING
    assert "llm_polished=False" in rec.getMessage()


def test_narration_part_marker_defaults_to_none():
    part = NarrationPart(
        part_id="clip-1",
        scenario_name="S",
        text="t",
        original_gherkin="g",
        word_count=1,
        estimated_duration_seconds=0.4,
    )
    assert part.narration_source is None
    assert part.template_fallback_steps is None
