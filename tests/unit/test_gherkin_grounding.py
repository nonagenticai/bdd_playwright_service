"""Demo Gherkin TEXT must describe UI the page shows — not invented counters/features.

#166 grounded the step CODE's selectors; the Gherkin text is narrated verbatim, so an
invented "completed count" on TodoMVC became a false claim in the voiceover.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.services.generation.feature_generator import (
    UNGROUNDED_CONCEPTS_KEY,
    FeatureGenerator,
    _build_page_summary,
)
from src.services.generation.gherkin_grounding import (
    build_page_corpus,
    find_ungrounded_concepts,
)
from src.services.generation.llm_client import LLMClient
from src.services.generation.step_def_generator import StepCoverageReport

# A fresh TodoMVC: no todos yet, so no footer, no "items left", no filters rendered.
_TODOMVC: dict[str, Any] = {
    "base_url": "https://demo.playwright.dev/todomvc",
    "platform": "web",
    "pages": [
        {
            "url": "https://demo.playwright.dev/todomvc/#/",
            "title": "React • TodoMVC",
            "page_type": "form",
            "has_forms": False,
            "headings": ["todos"],
            "interactive_elements": [
                {
                    "tag": "input",
                    "label": "What needs to be done?",
                    "selector": "getByPlaceholder('What needs to be done?')",
                },
            ],
            "text_content": "todos\n\n  Double-click to edit a todo   Created by the React team",
            "dom_tokens": {"ids": ["root"], "classes": ["todoapp", "new-todo"], "testids": []},
        }
    ],
}


def _feature(body: str) -> list[dict[str, Any]]:
    return [{"filename": "demo.feature", "content": f"Feature: Todos\n{body}"}]


def _concepts(body: str, inspection: dict[str, Any] = _TODOMVC) -> list[str]:
    found = find_ungrounded_concepts(_feature(body), build_page_corpus(inspection))
    return [f["concept"] for f in found]


# Real TodoMVC UI that only renders AFTER a todo exists — absent from the snapshot above.
_REAL_HIDDEN_UI_SUITE = (
    "  Scenario: Walkthrough\n"
    '    When I add a todo "Buy milk"\n'
    "    Then the todo appears in the list\n"
    "    And the item is marked as completed\n"
    '    And I see "1 item left"\n'
    "    And the Active filter shows only open todos\n"
    "    And the Completed filter shows the done todo\n"
    '    And the "Clear completed" button is visible\n'
    "    And the item counter shows 2 items left\n"
    "    And the todo count updates\n"
    "    And the new todo input is empty\n"
    '    And I see the "What needs to be done?" placeholder\n'
    "    And the Clear completed button removes it\n"
)


class TestFindUngroundedConcepts:
    def test_invented_completed_count_is_flagged(self) -> None:
        body = (
            "  Scenario: Complete a todo\n"
            '    Given I add a todo "Walk the dog"\n'
            "    When I check the todo\n"
            "    Then the completed count shows 1\n"
        )
        assert _concepts(body) == ["completed count"]

    def test_real_ui_hidden_until_an_action_is_never_flagged(self) -> None:
        # First cut flagged 4 of these 11 — all real TodoMVC UI. The re-ask would have
        # told the model to delete real features from the main demo target.
        assert _concepts(_REAL_HIDDEN_UI_SUITE) == []

    def test_known_miss_modifier_used_elsewhere_in_the_suite(self) -> None:
        # TRADEOFF, pinned: the suite saying "completed" anywhere grounds the modifier, so
        # this invented counter is NOT caught. Precision over recall.
        body = (
            "  Scenario: Complete a todo\n"
            "    When I mark it as completed\n"
            "    Then the completed count shows 1\n"
        )
        assert _concepts(body) == []

    def test_modifier_grounded_by_another_feature_in_the_suite(self) -> None:
        feats = [
            {
                "filename": "a.feature",
                "content": "Feature: A\n  Scenario: S\n    Then the streak badge glows\n",
            },
            {
                "filename": "b.feature",
                "content": "Feature: B\n  Scenario: T\n    When I keep a streak\n",
            },
        ]
        assert find_ungrounded_concepts(feats, build_page_corpus(_TODOMVC)) == []

    def test_grounded_quoted_heading_text_is_never_flagged(self) -> None:
        # "todos" is the corpus's own heading/text_content — quoted judging must not
        # flag text the page genuinely shows just because it is quoted.
        body = '  Scenario: Open\n    Given I open the app\n    Then I see the heading "todos"\n'
        assert _concepts(body) == []

    def test_invented_quoted_aside_is_now_flagged(self) -> None:
        # #167 shipped this as a known miss (quoted strings judged nowhere); raising
        # recall means an invented quoted panel/aside with no corpus or suite backing
        # must be caught.
        body = (
            "  Scenario: Open\n    Given I open the app\n    Then I see the todos list\n"
            '    And I see "Productivity insights"\n'
        )
        assert _concepts(body) == ["Productivity insights"]

    def test_invented_label_is_now_flagged(self) -> None:
        # #167's second known miss: a `<modifier> label` claim was never judged.
        body = (
            "  Scenario: Open\n    Given I open the app\n    Then the priority label reads High\n"
        )
        assert _concepts(body) == ["priority label"]

    def test_counter_whose_word_is_a_dom_class_token_is_grounded(self) -> None:
        body = "  Scenario: Add\n    When I add a todo\n    Then the pending badge shows 1\n"
        assert _concepts(body) == ["pending badge"]
        page = {**_TODOMVC["pages"][0]}
        page["dom_tokens"] = {"ids": [], "classes": ["pending-badge"], "testids": []}
        assert _concepts(body, {"pages": [page]}) == []

    def test_counter_with_a_word_on_the_page_is_grounded(self) -> None:
        body = "  Scenario: Add\n    When I add a todo\n    Then the todo count shows 1\n"
        assert _concepts(body) == []

    def test_given_and_when_steps_are_never_judged(self) -> None:
        body = "  Scenario: S\n    Given the premium badge is shown\n    When I open the app\n"
        assert _concepts(body) == []

    def test_and_after_then_is_judged(self) -> None:
        body = "  Scenario: S\n    Given I open the app\n    Then I see the list\n    And the streak badge glows\n"
        assert _concepts(body) == ["streak badge"]

    def test_no_page_text_judges_nothing(self) -> None:
        bare = {"pages": [{**_TODOMVC["pages"][0], "text_content": ""}]}
        body = "  Scenario: S\n    Given I open the app\n    Then the overdue counter shows 1\n"
        assert _concepts(body, bare) == []


class TestPageSummaryVisibleText:
    def test_visible_text_is_included_collapsed(self) -> None:
        summary = _build_page_summary(_TODOMVC)
        assert (
            "Visible text: todos Double-click to edit a todo Created by the React team" in summary
        )

    def test_visible_text_is_bounded_per_page_and_in_total(self) -> None:
        page = {**_TODOMVC["pages"][0], "text_content": "word " * 2000}
        summary = _build_page_summary({"pages": [page] * 10})
        lines = [ln for ln in summary.splitlines() if "Visible text:" in ln]
        assert all(len(ln) <= 1500 + len("  Visible text: ") for ln in lines)
        assert sum(len(ln) - len("  Visible text: ") for ln in lines) <= 6000


_INVENTED = (
    "```gherkin\nFeature: Todos\n  Scenario: Complete\n"
    '    When I check "Walk the dog"\n    Then the completed count shows 1\n```'
)
_GROUNDED = (
    "```gherkin\nFeature: Todos\n  Scenario: Complete\n"
    '    When I add a todo "Walk the dog"\n    Then I see "Walk the dog" in the list\n```'
)


@pytest.fixture
def llm() -> AsyncMock:
    client = AsyncMock(spec=LLMClient)
    client.generate = AsyncMock()
    return client


class TestDemoGenerationGroundingReask:
    async def test_prompt_forbids_invented_ui(self, llm: AsyncMock) -> None:
        llm.generate.return_value = _GROUNDED
        await FeatureGenerator(llm).generate_demo_features(_TODOMVC, {})
        prompt = llm.generate.await_args_list[0].args[0]
        assert "DESCRIBE ONLY WHAT THE PAGE SHOWS" in prompt
        assert "Visible text:" in prompt

    async def test_ungrounded_then_triggers_one_reask_and_uses_the_fix(
        self, llm: AsyncMock
    ) -> None:
        llm.generate.side_effect = [_INVENTED, _GROUNDED]
        features = await FeatureGenerator(llm).generate_demo_features(_TODOMVC, {})
        assert llm.generate.await_count == 2
        reask = llm.generate.await_args_list[1].args[0]
        assert "completed count" in reask
        # The correction must not push the model to delete real after-action UI.
        assert "Do NOT remove real controls" in reask
        assert "completed count" not in features[0]["content"]
        assert features[0][UNGROUNDED_CONCEPTS_KEY] == []

    async def test_still_ungrounded_after_reask_is_kept_and_recorded(self, llm: AsyncMock) -> None:
        llm.generate.side_effect = [_INVENTED, _INVENTED]
        features = await FeatureGenerator(llm).generate_demo_features(_TODOMVC, {})
        # Exactly one re-ask — never a loop — and the step is KEPT, not dropped.
        assert llm.generate.await_count == 2
        assert "completed count" in features[0]["content"]
        recorded = features[0][UNGROUNDED_CONCEPTS_KEY]
        assert [r["concept"] for r in recorded] == ["completed count"]
        coverage = StepCoverageReport(ungrounded_gherkin_concepts=recorded).as_dict()
        assert coverage["ungrounded_gherkin_concepts_flagged"] == 1
        assert coverage["ungrounded_gherkin_concepts"][0]["step"] == "the completed count shows 1"

    async def test_broken_grounding_reask_keeps_the_valid_original(self, llm: AsyncMock) -> None:
        llm.generate.side_effect = [_INVENTED, "no gherkin here"]
        features = await FeatureGenerator(llm).generate_demo_features(_TODOMVC, {})
        assert llm.generate.await_count == 2
        assert "completed count" in features[0]["content"]


def _fenced(*scenarios: str) -> str:
    return "```gherkin\nFeature: Todos\n" + "".join(scenarios) + "```"


_ADD = '  Scenario: Add\n    When I add a todo "Walk the dog"\n    Then I see "Walk the dog"\n'
_COMPLETE_INVENTED = (
    "  Scenario: Complete\n    When I check it\n    Then the completed count shows 1\n"
)
_COMPLETE_FIXED = "  Scenario: Complete\n    When I check it\n    Then it is struck through\n"
_COMPLETE_OTHER_INVENTED = (
    "  Scenario: Complete\n    When I check it\n    Then the overdue counter shows 0\n"
)


class TestGroundingReaskAcceptance:
    """A grounding-only re-ask may reword Then steps; it may not cost footage or regress."""

    async def _run(self, llm: AsyncMock, retry: str) -> list[dict[str, Any]]:
        llm.generate.side_effect = [_fenced(_ADD, _COMPLETE_INVENTED), retry]
        features = await FeatureGenerator(llm).generate_demo_features(_TODOMVC, {})
        assert llm.generate.await_count == 2
        return features

    async def test_reask_that_drops_a_scenario_keeps_the_original(self, llm: AsyncMock) -> None:
        features = await self._run(llm, _fenced(_COMPLETE_FIXED))
        assert "Scenario: Add" in features[0]["content"]
        assert "completed count" in features[0]["content"]
        assert [r["concept"] for r in features[0][UNGROUNDED_CONCEPTS_KEY]] == ["completed count"]

    async def test_reask_introducing_a_new_ungrounded_counter_keeps_the_original(
        self, llm: AsyncMock
    ) -> None:
        features = await self._run(llm, _fenced(_ADD, _COMPLETE_OTHER_INVENTED))
        assert "completed count" in features[0]["content"]
        assert "overdue counter" not in features[0]["content"]

    async def test_same_count_fewer_findings_is_accepted(self, llm: AsyncMock) -> None:
        features = await self._run(llm, _fenced(_ADD, _COMPLETE_FIXED))
        assert "struck through" in features[0]["content"]
        assert "Scenario: Add" in features[0]["content"]
        assert features[0][UNGROUNDED_CONCEPTS_KEY] == []
