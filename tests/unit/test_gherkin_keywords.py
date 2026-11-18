"""Gherkin continuation-keyword resolution + the generated-code compile gate.

Regression cover for a LIVE dead render: bddgen exited 1 with
``ReferenceError: And is not defined`` because a generated step-def file called
``And(...)``, which ``createBdd()`` never exposes (corpus run 30793758726,
2026-08-03). ``And``/``But``/``*`` are Gherkin CONTINUATION keywords — they
inherit the kind of the preceding ``Given``/``When``/``Then``.
"""

from __future__ import annotations

import re

import pytest

from src.services.generation.step_def_generator import (
    _apply_compile_guarantees,
    _extract_steps_from_features,
    _parse_step_defs,
    _resolve_continuation_calls,
    _undefined_module_bindings,
)
from src.services.gherkin_keywords import (
    CONTINUATION_KEYWORDS,
    DEFAULT_BINDING,
    PRIMARY_KEYWORDS,
    is_continuation,
    iter_steps,
    resolve_binding,
)


class TestResolveBinding:
    """A binding must ALWAYS be one of Given/When/Then — never a continuation."""

    @pytest.mark.parametrize("keyword", PRIMARY_KEYWORDS)
    def test_primary_keyword_binds_to_itself(self, keyword: str) -> None:
        assert resolve_binding(keyword) == keyword
        assert resolve_binding(keyword, "When") == keyword

    @pytest.mark.parametrize("keyword", CONTINUATION_KEYWORDS)
    @pytest.mark.parametrize("preceding", PRIMARY_KEYWORDS)
    def test_continuation_inherits_preceding_primary(self, keyword: str, preceding: str) -> None:
        assert resolve_binding(keyword, preceding) == preceding

    @pytest.mark.parametrize("keyword", CONTINUATION_KEYWORDS)
    def test_leading_continuation_falls_back_and_never_returns_itself(self, keyword: str) -> None:
        """A scenario opening with `And` has nothing to inherit — it must not emit `And`."""
        resolved = resolve_binding(keyword, None)
        assert resolved == DEFAULT_BINDING
        assert resolved in PRIMARY_KEYWORDS

    def test_unknown_keyword_can_never_produce_a_non_binding(self) -> None:
        assert resolve_binding("Wibble", None) in PRIMARY_KEYWORDS
        assert resolve_binding("Wibble", "Then") in PRIMARY_KEYWORDS

    def test_bogus_default_is_ignored(self) -> None:
        """A caller passing a non-binding default must not poison the output."""
        assert resolve_binding("And", None, default="And") in PRIMARY_KEYWORDS

    @pytest.mark.parametrize("keyword", CONTINUATION_KEYWORDS)
    def test_is_continuation(self, keyword: str) -> None:
        assert is_continuation(keyword)

    @pytest.mark.parametrize("keyword", PRIMARY_KEYWORDS)
    def test_primary_is_not_continuation(self, keyword: str) -> None:
        assert not is_continuation(keyword)


class TestIterSteps:
    def test_given_and_and_chain(self) -> None:
        feature = (
            "Feature: F\n"
            "  Scenario: S\n"
            "    Given I am logged in\n"
            "    And I have items\n"
            "    And I opened the cart\n"
        )
        assert [s.binding for s in iter_steps(feature)] == ["Given", "Given", "Given"]
        assert [s.keyword for s in iter_steps(feature)] == ["Given", "And", "And"]

    def test_when_and_then_but(self) -> None:
        feature = (
            "  Scenario: S\n"
            "    When I click checkout\n"
            "    And I confirm\n"
            "    Then I see a receipt\n"
            "    But I see no discount\n"
        )
        assert [s.binding for s in iter_steps(feature)] == ["When", "When", "Then", "Then"]

    def test_bullet_form_is_a_continuation(self) -> None:
        feature = "  Scenario: S\n    Given a cart\n    * it has one item\n"
        assert [s.binding for s in iter_steps(feature)] == ["Given", "Given"]

    def test_inheritance_resets_at_a_scenario_boundary(self) -> None:
        """An `And` opening a scenario must NOT inherit from the scenario above it."""
        feature = (
            "  Scenario: first\n"
            "    Then I see the list\n"
            "  Scenario: second\n"
            "    And I am still here\n"
        )
        assert [s.binding for s in iter_steps(feature)] == ["Then", DEFAULT_BINDING]

    def test_no_binding_is_ever_a_continuation_keyword(self) -> None:
        feature = (
            "  Scenario: S\n"
            "    And leading\n"
            "    But another\n"
            "    * bullet\n"
            "    When act\n"
            "    And more\n"
        )
        assert all(s.binding in PRIMARY_KEYWORDS for s in iter_steps(feature))


class TestExtractStepsResolvesContinuations:
    def test_extracted_keyword_is_always_a_binding(self) -> None:
        features = [
            {
                "content": (
                    "  Scenario: S\n"
                    "    Given I am logged in\n"
                    "    And I have items\n"
                    "    When I check out\n"
                    "    But payment fails\n"
                )
            }
        ]
        steps = _extract_steps_from_features(features)
        assert [s["keyword"] for s in steps] == ["Given", "Given", "When", "When"]
        assert [s["gherkin_keyword"] for s in steps] == ["Given", "And", "When", "But"]


class TestResolveContinuationCalls:
    """The deterministic repair over already-emitted TypeScript."""

    def test_rewrites_and_but_to_preceding_kind(self) -> None:
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, When, Then } = createBdd();\n"
            "Given('a', async ({ page }) => {});\n"
            "And('b', async ({ page }) => {});\n"
            "When('c', async ({ page }) => {});\n"
            "But('d', async ({ page }) => {});\n"
        )
        new, rewrites = _resolve_continuation_calls(content)
        assert rewrites == 2
        assert "\nAnd(" not in new
        assert "\nBut(" not in new
        assert new.count("Given(") == 2
        assert new.count("When(") == 2

    def test_leading_continuation_becomes_the_default_binding(self) -> None:
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, When, Then } = createBdd();\n"
            "And('first', async ({ page }) => {});\n"
        )
        new, rewrites = _resolve_continuation_calls(content)
        assert rewrites == 1
        assert new.startswith("import")
        assert f"{DEFAULT_BINDING}('first'" in new

    def test_literal_and_inside_a_string_is_left_alone(self) -> None:
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            "Given('salt And pepper', async ({ page }) => {});\n"
        )
        new, rewrites = _resolve_continuation_calls(content)
        assert rewrites == 0
        assert new == content


class TestCompileGate:
    """A test asserting the output 'contains Given(' proves nothing — the live
    failure was a call to a symbol that was never in scope."""

    def test_gate_rejects_an_undefined_binding(self) -> None:
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            "Given('a', async ({ page }) => {});\n"
            "Wibble('b', async ({ page }) => {});\n"
        )
        assert _undefined_module_bindings([{"filename": "s.ts", "content": content}]) == [
            "s.ts: Wibble"
        ]

    def test_gate_accepts_a_well_formed_file(self) -> None:
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, When, Then } = createBdd();\n"
            "Given('a', async ({ page }) => {});\n"
            "When('b', async ({ page }) => {});\n"
        )
        assert _undefined_module_bindings([{"filename": "s.ts", "content": content}]) == []

    def test_missing_createbdd_destructure_is_repaired_not_rejected(self) -> None:
        """The LLM omits the boilerplate often enough that rejecting would be brittle;
        the repo already auto-repairs the analogous missing `expect` import.

        ⚠️ This asserts the PROPERTY (the called keyword ends up bound), not the literal
        `const { Given, When, Then } = createBdd();`. It used to assert that exact triple,
        which pinned an over-broad injection: emitting all three unconditionally is what
        REDECLARED `Given`/`Then` on a file that already destructured a subset, and Babel
        rejects the file with `VarRedeclaration` before a single assertion runs. Only the
        missing names are injected now. Nothing is lost by binding narrowly —
        `_apply_compile_guarantees` re-runs `_ensure_bdd_bindings_in_scope` (see its body)
        AFTER the keyword rebinds, so a `When`/`Then` introduced later by a repair still
        gets bound; `test_end_to_end_no_continuation_call_survives_the_guarantees` below
        covers exactly that path via `And(`.
        """
        raw = (
            "```typescript\n"
            "// filename: steps.ts\n"
            "Given('I am on the app', async ({ page }) => {});\n"
            "```"
        )
        step_defs = _parse_step_defs(raw)
        features = [{"content": "  Scenario: S\n    Given I am on the app\n"}]
        out = _apply_compile_guarantees(step_defs, features, recording=False)
        content = out[0]["content"]
        assert "import { createBdd } from 'playwright-bdd';" in content
        assert re.search(r"const \{[^}]*\bGiven\b[^}]*\} = createBdd\(\);", content), content
        assert _undefined_module_bindings(out) == []

    def test_end_to_end_no_continuation_call_survives_the_guarantees(self) -> None:
        """The exact live failure: an `And(` call must never reach the runner."""
        raw = (
            "```typescript\n"
            "// filename: todo.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, When, Then } = createBdd();\n"
            "Given('I am on the app', async ({ page }) => {});\n"
            "And('I see the list', async ({ page }) => {});\n"
            "```"
        )
        step_defs = _parse_step_defs(raw)
        features = [
            {"content": "  Scenario: S\n    Given I am on the app\n    And I see the list\n"}
        ]
        out = _apply_compile_guarantees(step_defs, features, recording=False)
        for sd in out:
            assert "\nAnd(" not in sd["content"], "a literal And( call reached the runner"
            assert "\nBut(" not in sd["content"]
        assert _undefined_module_bindings(out) == []
