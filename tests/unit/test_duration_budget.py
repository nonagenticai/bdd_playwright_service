"""The ORDERED film duration must reach the two numbers this service authors.

MEASURED DEFECT (2026-09-10). A demo_video ordered at 30s delivered 47.662s
(159%). Not a rate problem — narration-to-picture fit was ~100.2%. Structural:
the order reached this service only as a scenario COUNT ceiling, and one beat's
footage (~23.8s) is comparable to the whole order, so no integer scenario count
lands within +-10% (1 beat = 25.4s / 85%, 2 beats = 47.4s / 159%).

This service authors BOTH the narration text and the dwell, so it is the layer
where the order can bound them. These tests pin that, and pin the thing that
protects every current caller: with no duration ordered, NOTHING changes.
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.duration_budget import (
    MIN_SCENARIO_RECORDING_SECONDS,
    TRIM_RESCUE_CEILING_RATIO,
    narration_word_budgets,
    plan_duration_budget,
    raise_to_clip_floor,
    trim_to_word_budget,
)
from src.services.generation.feature_generator import _parse_features
from src.services.generation.step_def_generator import (
    GUARANTEED_RECORDING_SECONDS_PER_SCENARIO,
    StepCoverageReport,
    ensure_dwell_budget,
)
from src.services.job_processor import JobProcessor

MODULE = "src.services.job_processor"

#: Two scenarios in one feature — the exact shape of the 47.7s run (`01a08afc`
#: recorded 2 clips of ~23.8s against a 30s order).
#: Each scenario is deliberately step-RICH: the live 30s/47.7s run fitted
#: narration to picture at ~100.2%, i.e. ~59 spoken words per 23.8s clip, so a
#: two-step scenario would produce narration already inside a 15s budget and
#: every trim assertion below would be vacuous (it was, on the first draft).
_TWO_SCENARIO_FEATURE = """```gherkin
Feature: Tour
  Scenario: Open the dashboard
    Given I am on the login page
    When I enter my email address
    And I enter my password
    And I submit the sign in form
    Then I see the dashboard
    And I see the recent activity panel
    And I see the navigation sidebar
    And I see the account menu

  Scenario: Create a project
    Given I am on the dashboard
    When I open the projects page
    And I choose to create a new project
    And I enter a project name
    And I choose a project template
    And I confirm the new project
    Then I see the project list
    And I see the new project at the top
```"""

_ORDERED_SECONDS = 30.0
_WPM = 150  # `NarrationSettings.words_per_minute` default (src/config.py).


@pytest.fixture
def processor():
    """A JobProcessor whose LLM clients are mocked (no outbound calls)."""
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="mocked LLM response")
    with patch(f"{MODULE}.LLMClient", return_value=llm):
        yield JobProcessor()


class TestNoOrderIsUnchanged:
    """THE BACKWARD-COMPATIBILITY GUARD. Every caller that never sends
    ``target_duration_seconds`` must get byte-identical behaviour."""

    def test_absent_order_plans_nothing(self):
        assert plan_duration_budget(None, 2, words_per_minute=_WPM) is None

    def test_non_positive_and_zero_scenario_cases_plan_nothing(self):
        assert plan_duration_budget(0, 2, words_per_minute=_WPM) is None
        assert plan_duration_budget(-5, 2, words_per_minute=_WPM) is None
        assert plan_duration_budget(30, 0, words_per_minute=_WPM) is None

    def test_processor_plans_nothing_without_the_field(self, processor: JobProcessor):
        features = _parse_features(_TWO_SCENARIO_FEATURE)
        assert processor._plan_budget("demo_video", features, {}) is None
        # Present-but-null (the orchestrator will send the key unconditionally).
        null_field = processor._plan_budget(
            "demo_video", features, {"target_duration_seconds": None}
        )
        assert null_field is None

    def test_non_recording_stream_plans_nothing(self, processor: JobProcessor):
        """Only demo_video ships the dwell ledger; budgeting narration against
        footage that does not exist would be a fabricated number."""
        features = _parse_features(_TWO_SCENARIO_FEATURE)
        assert (
            processor._plan_budget(
                "qa_suite", features, {"target_duration_seconds": _ORDERED_SECONDS}
            )
            is None
        )

    async def test_narration_without_a_budget_is_identical(self, processor: JobProcessor):
        """The whole no-order path: same text, same word counts, and the default
        per-scenario recording duration on the wire."""
        features = _parse_features(_TWO_SCENARIO_FEATURE)
        cfg = {"enabled": True, "tone": "tutorial", "llm_polish": False}

        baseline = await processor._generate_narration(features, cfg, recording=True)
        explicit_none = await processor._generate_narration(
            features, cfg, recording=True, budget=None
        )

        assert baseline == explicit_none

        # ABSOLUTE anchor, not just self-consistency: each part must still be the
        # template engine's own untouched output. Comparing the two calls above
        # to each other would stay green if narration were trimmed on BOTH paths,
        # which is precisely the regression this guard exists to catch.
        from src.services.input.gherkin_parser import GherkinParser
        from src.services.narration.template_engine import NarrationTemplateEngine

        engine = NarrationTemplateEngine(tone="tutorial")
        scenarios = GherkinParser().parse_feature_text(features[0]["content"]).scenarios
        for part, scenario in zip(baseline["parts"], scenarios, strict=True):
            assert part["text"] == engine.humanize_scenario(
                scenario.name, [{"keyword": s.keyword, "text": s.text} for s in scenario.steps]
            )

        for part in baseline["parts"]:
            assert part["expected_recording_seconds"] == GUARANTEED_RECORDING_SECONDS_PER_SCENARIO
            # No trimming happened: the template line is whatever it is, and it is
            # LONGER than a 15s budget would allow (see the budgeted test below),
            # which is exactly the defect.
            assert part["word_count"] == len(part["text"].split())

    def test_no_budget_injects_no_dwell_module(self):
        """No ordered duration ⇒ the package is what it was yesterday: no
        ``steps_dwell_budget.ts``, no import, no call."""
        step_defs: list[dict[str, object]] = [{"filename": "demo.steps.ts", "content": "x"}]
        assert ensure_dwell_budget(step_defs, 0) == 0
        assert len(step_defs) == 1

    def test_coverage_payload_reports_no_budget(self):
        assert StepCoverageReport().as_dict()["duration_budget"] is None


class TestThirtySecondsOverTwoScenarios:
    """The measured case: 30s ordered, 2 scenarios generated ⇒ 15s each — in
    BOTH the dwell and the narration."""

    def test_per_scenario_seconds_and_word_budget(self):
        plan = plan_duration_budget(_ORDERED_SECONDS, 2, words_per_minute=_WPM)
        assert plan is not None
        assert plan.per_scenario_seconds == 15.0
        assert plan.dwell_budget_ms == 15_000
        # 15s at 150 wpm = 37.5 words; the budget floors to whole words.
        assert plan.narration_word_budget == 37
        assert plan.order_met is True
        assert plan.projected_total_seconds == 30.0

    def test_the_dwell_module_carries_the_budget(self):
        """The ledger in `lib/recording_helpers.ts` is what makes a clip 23.8s;
        this module is the only thing that narrows it."""
        plan = plan_duration_budget(_ORDERED_SECONDS, 2, words_per_minute=_WPM)
        assert plan is not None
        step_defs: list[dict[str, object]] = []
        assert ensure_dwell_budget(step_defs, plan.dwell_budget_ms) == 1
        content = str(step_defs[0]["content"])
        assert step_defs[0]["filename"] == "steps_dwell_budget.ts"
        assert "setScenarioDwellBudgetMs(15000);" in content
        assert "from '../lib/recording_helpers'" in content
        # Idempotent, like every other injection pass.
        assert ensure_dwell_budget(step_defs, plan.dwell_budget_ms) == 0
        assert len(step_defs) == 1

    def test_the_ledger_actually_spends_from_the_settable_budget(self):
        """VACUITY GUARD. Emitting `setScenarioDwellBudgetMs(15000)` changes
        nothing unless the ledger's remaining-budget calculation READS it — if
        `dwellRemainingMs` still subtracted from the fixed `HERO_DWELL_MS`, every
        clip would keep running to 22s and the module above would be decoration.
        """
        from src.services.generation.step_def_generator import _RECORDING_HELPERS_TS

        src = _RECORDING_HELPERS_TS.read_text(encoding="utf-8")
        assert "export function setScenarioDwellBudgetMs" in src
        remaining = src.split("export function dwellRemainingMs")[1].split("}")[0]
        assert "scenarioDwellBudgetMs" in remaining, (
            "dwellRemainingMs does not read the settable budget, so narrowing it "
            "is inert and every clip still runs to HERO_DWELL_MS."
        )

    async def test_narration_is_budgeted_and_the_duration_is_reported(
        self, processor: JobProcessor
    ):
        features = _parse_features(_TWO_SCENARIO_FEATURE)
        plan = plan_duration_budget(_ORDERED_SECONDS, 2, words_per_minute=_WPM)
        assert plan is not None

        script = await processor._generate_narration(
            features,
            {"enabled": True, "tone": "tutorial", "llm_polish": False},
            recording=True,
            budget=plan,
        )

        # NON-VACUITY: the same features WITHOUT a budget must exceed it, else
        # "word_count <= budget" is satisfied by narration that was already short
        # and this test would stay green with the trim deleted.
        unbudgeted = await processor._generate_narration(
            features,
            {"enabled": True, "tone": "tutorial", "llm_polish": False},
            recording=True,
        )
        assert any(p["word_count"] > plan.narration_word_budget for p in unbudgeted["parts"]), (
            "fixture narration is already within budget — the trim assertion below is vacuous"
        )

        assert len(script["parts"]) == 2
        for part in script["parts"]:
            assert part["word_count"] <= plan.narration_word_budget
            assert part["word_count"] == len(part["text"].split())
            # The published clip duration IS the budget, not the 22s default —
            # the orchestrator reads this field to lay out its timeline.
            assert part["expected_recording_seconds"] == 15.0
            assert part["expected_recording_seconds"] != GUARANTEED_RECORDING_SECONDS_PER_SCENARIO

    async def test_the_polish_pass_cannot_spend_past_the_budget(self, processor: JobProcessor):
        """The LLM rewrites the line after the template trim; re-applying the
        budget to its output is what keeps voiceover and dwell the same length."""
        features = _parse_features(_TWO_SCENARIO_FEATURE)
        plan = plan_duration_budget(_ORDERED_SECONDS, 2, words_per_minute=_WPM)
        assert plan is not None
        long_line = " ".join(["word"] * 120) + "."

        async def _polish(parts, tone, **_kwargs):
            return [{**p, "text": long_line} for p in parts]

        with patch.object(processor._humanizer, "polish", _polish):
            script = await processor._generate_narration(
                features,
                {"enabled": True, "tone": "tutorial", "llm_polish": True},
                recording=True,
                budget=plan,
            )

        for part in script["parts"]:
            assert part["word_count"] <= plan.narration_word_budget


class TestTheFloorClampsAndIsRecorded:
    """A 2-second clip that shows nothing is worse than a slightly long one."""

    def test_an_impossible_order_clamps_to_the_floor(self):
        # 10s over 5 scenarios = 2.0s each — the degenerate case.
        plan = plan_duration_budget(10.0, 5, words_per_minute=_WPM)
        assert plan is not None
        assert plan.per_scenario_seconds == MIN_SCENARIO_RECORDING_SECONDS
        assert plan.floor_applied is True
        assert plan.order_met is False
        # And it says what will actually be delivered, not what was asked for.
        assert plan.projected_total_seconds == 40.0
        assert plan.narration_word_budget >= 1

    def test_an_over_long_order_clamps_to_the_hero_dwell_ceiling(self):
        """The budget BOUNDS; it never stretches a clip past what the per-test
        Playwright timeout allows."""
        plan = plan_duration_budget(300.0, 2, words_per_minute=_WPM)
        assert plan is not None
        assert plan.per_scenario_seconds == GUARANTEED_RECORDING_SECONDS_PER_SCENARIO
        assert plan.ceiling_applied is True
        assert plan.order_met is False

    def test_a_missed_order_is_recorded_on_step_coverage_not_raised(self):
        """Recording only — no currently-working package may start failing."""
        plan = plan_duration_budget(10.0, 5, words_per_minute=_WPM)
        assert plan is not None
        report = StepCoverageReport(steps_total=4, steps_stubbed=0)
        report.duration_budget = plan.as_dict()

        payload = report.as_dict()
        budget = payload["duration_budget"]
        assert budget is not None
        assert budget["order_met"] is False
        assert budget["floor_applied"] is True
        assert budget["ordered_seconds"] == 10.0
        assert budget["scenario_count"] == 5
        assert budget["per_scenario_seconds"] == MIN_SCENARIO_RECORDING_SECONDS
        assert budget["floor_seconds"] == MIN_SCENARIO_RECORDING_SECONDS
        # The stub-visibility half of the same channel is untouched.
        assert payload["degraded"] is False

    def test_the_floor_is_long_enough_to_carry_one_narrated_sentence(self):
        """VACUITY GUARD on the floor itself: a floor that cannot fit a sentence
        is not a floor, it is a shorter degenerate clip."""
        words = MIN_SCENARIO_RECORDING_SECONDS * _WPM / 60
        assert words >= 15, (
            f"The {MIN_SCENARIO_RECORDING_SECONDS}s floor buys only {words:.0f} words "
            f"at {_WPM} wpm — too short for one complete narrated sentence."
        )


class TestTrimming:
    def test_text_within_budget_is_returned_unchanged(self):
        text = "Open the dashboard.  Then create a project."
        assert trim_to_word_budget(text, 50) == text

    def test_whole_sentences_are_dropped_first(self):
        text = "One two three. Four five six. Seven eight nine."
        assert trim_to_word_budget(text, 6) == "One two three. Four five six."

    def test_a_single_over_long_sentence_is_hard_cut(self):
        text = "one two three four five six"
        assert trim_to_word_budget(text, 3) == "one two three"


class TestTheHeroShareCapUnpinsADominatingBeat:
    """5b — a 44s order delivered +54.8%, and not because the script was long.

    ONE clip's own footage was ~35s of the 44s order. In the orchestrator a beat
    whose footage is that close to the whole film becomes
    ``min == ideal == max, elasticity 0, pinned`` — a shape narration can never
    shrink. The 22s ceiling cannot prevent it, because 22s IS most of a 44s order.
    """

    def test_an_over_long_hero_is_clamped_and_so_is_its_narration(self):
        plan = plan_duration_budget(44.0, 1, words_per_minute=_WPM)
        assert plan is not None
        # The fair share (44s) blew the 22s ceiling, so the order was already
        # unmeetable from footage — which is the only state the cap acts in.
        assert plan.ceiling_applied is True
        assert plan.per_scenario_seconds == GUARANTEED_RECORDING_SECONDS_PER_SCENARIO
        # 0.4 * 44 = 17.6s: the beat keeps 40% of the order and gains elasticity.
        assert plan.hero_cap_applied is True
        assert plan.dwell_seconds == pytest.approx(17.6)
        assert plan.dwell_budget_ms == 17_600
        # ⛔ THE WORD BUDGET COMES FROM THE CAPPED NUMBER, and it must.
        #
        # Sizing it from the UNCAPPED 22s share — which is what this shipped as —
        # writes ~22s of speech to be spoken over 17.6s of picture. The picture then
        # stops while the audio keeps going, which IS the truncation defect, not a
        # fix for it. There is no "keep the narration long so the film still fills
        # the order" here: the cap only fires when ``ceiling_applied``, i.e. when
        # ``order_met`` is ALREADY False and the miss is already recorded. The only
        # choice on the table is whether the audio and the picture agree.
        assert plan.narration_word_budget == int(17.6 * _WPM / 60)
        assert plan.narration_word_budget < int(
            GUARANTEED_RECORDING_SECONDS_PER_SCENARIO * _WPM / 60
        )
        assert plan.as_dict()["hero_cap_applied"] is True

    def test_the_spoken_line_never_outlasts_the_picture_it_rides_on(self):
        """The PROPERTY, stated independently of any one order.

        ``narration_word_budget`` seconds (at the rate it was derived from) must
        never exceed ``dwell_seconds`` — the number the dwell ledger is narrowed to
        and the number published as the clip floor. Swept across the cap's whole
        active band (20-55s) plus both edges, where the mechanism changes.
        """
        for ordered in (12.0, 20.0, 23.0, 30.0, 44.0, 54.0, 55.0, 60.0, 300.0):
            for scenarios in (1, 2, 3):
                plan = plan_duration_budget(ordered, scenarios, words_per_minute=_WPM)
                assert plan is not None
                spoken = plan.narration_word_budget * 60 / _WPM
                assert spoken <= plan.dwell_seconds + 1e-9, (
                    f"ordered={ordered}s over {scenarios} scenario(s): the line needs "
                    f"{spoken:.2f}s of speech over {plan.dwell_seconds:.2f}s of "
                    "picture — the picture ends first"
                )

    def test_a_proportionate_split_is_left_exactly_alone(self):
        """NON-VACUITY of the gate: the cap must not touch a meetable order.

        30s over 2 scenarios is 15s each and 15s is under the ceiling, so the
        footage DOES fill the order. A cap that bit here would hand back a film
        that ends early — the documented failure mode of every duration fix in
        this estate.
        """
        plan = plan_duration_budget(30.0, 2, words_per_minute=_WPM)
        assert plan is not None
        assert plan.ceiling_applied is False
        assert plan.hero_cap_applied is False
        assert plan.dwell_seconds == plan.per_scenario_seconds == 15.0
        assert plan.dwell_budget_ms == 15_000

    def test_the_cap_is_a_no_op_on_long_orders(self):
        """``22 / 0.4 = 55``: at or above a 55s order the 22s dwell is already
        under 40%, so long-form work cannot be touched at all."""
        plan = plan_duration_budget(60.0, 1, words_per_minute=_WPM)
        assert plan is not None
        assert plan.ceiling_applied is True
        assert plan.hero_cap_applied is False
        assert plan.dwell_seconds == GUARANTEED_RECORDING_SECONDS_PER_SCENARIO

    def test_the_cap_can_never_undercut_the_floor(self):
        """``0.4 * 20 = 8.0`` is exactly MIN_SCENARIO_RECORDING_SECONDS, so below a
        20s order the floor wins and no degenerate clip is producible."""
        plan = plan_duration_budget(23.0, 1, words_per_minute=_WPM)
        assert plan is not None
        assert plan.dwell_seconds >= MIN_SCENARIO_RECORDING_SECONDS

    async def test_the_published_clip_floor_follows_the_cap(self, processor: JobProcessor):
        """The orchestrator lays its timeline out off ``expected_recording_seconds``.
        Publishing the UNCAPPED share there would advertise a clip floor the
        package does not record."""
        features = _parse_features(_TWO_SCENARIO_FEATURE)
        plan = plan_duration_budget(44.0, 1, words_per_minute=_WPM)
        assert plan is not None
        assert plan.hero_cap_applied is True

        script = await processor._generate_narration(
            features,
            {"enabled": True, "tone": "tutorial", "llm_polish": False},
            recording=True,
            budget=plan,
        )

        for part in script["parts"]:
            assert part["expected_recording_seconds"] == pytest.approx(17.6)
            # NON-VACUITY: the capped value is genuinely different from both the
            # uncapped share and the unbudgeted 22s default.
            assert part["expected_recording_seconds"] != plan.per_scenario_seconds
            assert part["expected_recording_seconds"] != GUARANTEED_RECORDING_SECONDS_PER_SCENARIO


class TestTheOrchestratorsWordBudgetContract:
    """5c — `narration_words_target` / `narration_words_per_scenario`.

    BOTH are optional: this service receives payloads without them from any
    orchestrator that has not deployed yet, and those must behave as they do today.
    """

    def test_absent_keys_yield_no_budget(self):
        assert narration_word_budgets(None, None, 2) is None
        assert narration_word_budgets(None, [], 2) is None
        # Junk is refused rather than half-honoured.
        assert narration_word_budgets("120", None, 2) is None
        assert narration_word_budgets(0, None, 2) is None
        assert narration_word_budgets(True, None, 2) is None

    def test_a_total_is_split_evenly_and_loses_no_words(self):
        assert narration_word_budgets(100, None, 4) == [25, 25, 25, 25]
        # The remainder is spread over the leading beats — the parts sum EXACTLY
        # to the ordered total instead of losing 2 words to floor division.
        split = narration_word_budgets(101, None, 3)
        assert split is not None
        assert split == [34, 34, 33]
        assert sum(split) == 101

    def test_a_per_scenario_split_wins_over_the_total(self):
        assert narration_word_budgets(999, [10, 20, 30], 3) == [10, 20, 30]

    def test_a_mismatched_split_is_discarded_not_zipped_short(self):
        """The orchestrator planned against a beat count the LLM did not emit, so
        the per-beat split describes beats that do not exist. Zipping it short
        would silently starve the trailing scenarios."""
        assert narration_word_budgets(120, [10, 20], 3) == [40, 40, 40]
        assert narration_word_budgets(None, [10, 20], 3) is None

    async def test_the_supplied_budget_beats_the_locally_derived_one(self, processor: JobProcessor):
        features = _parse_features(_TWO_SCENARIO_FEATURE)
        plan = plan_duration_budget(30.0, 2, words_per_minute=_WPM)
        assert plan is not None
        assert plan.narration_word_budget == 37

        script = await processor._generate_narration(
            features,
            {"enabled": True, "tone": "tutorial", "llm_polish": False},
            recording=True,
            budget=plan,
            word_budgets=[12, 9],
        )

        parts = script["parts"]
        assert len(parts) == 2
        # NON-VACUITY: without the supplied budget these same parts sit at the
        # locally derived 37, so a dropped override could not pass this. The bound is
        # the trim's RESCUE ceiling, not the bare budget: one sentence that crosses the
        # budget survives when dropping it would under-fill the line.
        assert parts[0]["word_count"] <= math.floor(12 * TRIM_RESCUE_CEILING_RATIO)
        assert parts[1]["word_count"] <= math.floor(9 * TRIM_RESCUE_CEILING_RATIO)
        assert parts[0]["word_count"] > parts[1]["word_count"]

    async def test_the_route_config_reaches_the_budget(self, processor: JobProcessor):
        """The keys must survive the trip through the job config verbatim — the
        contract says DO NOT RENAME."""
        features = _parse_features(_TWO_SCENARIO_FEATURE)
        contents = [str(f.get("content", "")) for f in features]
        assert processor._narration_word_budgets(contents, None, None) is None
        assert processor._narration_word_budgets(contents, 50, None) == [25, 25]
        assert processor._narration_word_budgets(contents, None, [8, 4]) == [8, 4]


# --- live job 01a156a6-8dd9 (2026-10-20): five identical 9-word lines vs 15-word budgets ---

_BACKGROUND_FEATURE = """Feature: React TodoMVC Walkthrough

  Background:
    Given I open the React TodoMVC demo at "https://demo.playwright.dev/todomvc"

  @clip-1
  Scenario: Add a new todo item
    Given the page displays the heading "todos"
    When I type "Buy groceries" into the input field
    And I press the Enter key
    Then I see the text "Buy groceries" displayed on the page

  @clip-2
  Scenario: Mark a todo as completed
    When I click the checkbox next to the todo item "Walk the dog"
    Then the todo item "Walk the dog" is marked as completed
"""


class TestShowcaseNarrationFillsItsBudget:
    def test_trim_keeps_the_crossing_sentence_when_dropping_it_underfills(self) -> None:
        """20 words / 15 budget used to come back as the 7-word first sentence."""
        text = (
            'First, the page displays the heading "todos". Next, type "Buy milk" into '
            "the input field and press Enter to add."
        )
        assert len(text.split()) == 20
        assert trim_to_word_budget(text, 15) == text

    def test_trim_still_drops_a_sentence_far_past_the_rescue_ceiling(self) -> None:
        first = "One two three four five six seven."
        second = " ".join(["word"] * 20) + "."
        assert trim_to_word_budget(f"{first} {second}", 15) == first

    def test_trim_does_not_rescue_when_the_kept_text_already_fills_the_budget(self) -> None:
        first = " ".join(["word"] * 13) + "."
        assert trim_to_word_budget(f"{first} Three more words.", 15) == first

    async def test_budgeted_narration_does_not_open_every_beat_with_the_background(
        self, processor: JobProcessor
    ) -> None:
        features = _parse_features(_BACKGROUND_FEATURE)
        script = await processor._generate_narration(
            features,
            {"enabled": True, "tone": "tutorial", "llm_polish": False},
            recording=True,
            word_budgets=[15, 15],
        )
        texts = [p["text"] for p in script["parts"]]
        assert len(texts) == 2
        assert all("demo.playwright.dev" not in t for t in texts), texts
        assert texts[0] != texts[1]
        # The steps still RUN: the reconstructed gherkin keeps the background.
        assert "TodoMVC demo" in script["parts"][0]["original_gherkin"]

    async def test_unbudgeted_narration_still_narrates_the_background(
        self, processor: JobProcessor
    ) -> None:
        features = _parse_features(_BACKGROUND_FEATURE)
        script = await processor._generate_narration(
            features, {"enabled": True, "tone": "tutorial", "llm_polish": False}
        )
        assert any("TodoMVC demo" in p["text"] for p in script["parts"])

    def test_word_budget_is_raised_to_the_clip_floor(self) -> None:
        """30s / 5 scenarios = 6s shares clamped to 8s clips: 15 words -> 20."""
        plan = plan_duration_budget(30.0, 5, words_per_minute=_WPM)
        assert plan is not None and plan.floor_applied
        raised = raise_to_clip_floor([15] * 5, plan, 75, 30.0)
        assert raised is not None
        assert raised == [round(plan.dwell_seconds * 75 / 30.0)] * 5
        assert raised[0] > 15

    def test_word_budget_is_unchanged_when_the_order_share_fits(self) -> None:
        plan = plan_duration_budget(120.0, 4, words_per_minute=_WPM)
        assert plan is not None and not plan.floor_applied
        assert raise_to_clip_floor([75] * 4, plan, 300, 120.0) == [75] * 4
        assert raise_to_clip_floor(None, plan, 300, 120.0) is None
        floored = plan_duration_budget(30.0, 5, words_per_minute=_WPM)
        assert raise_to_clip_floor([15] * 5, floored, None, 30.0) == [15] * 5
