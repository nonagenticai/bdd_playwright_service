"""Tests for step definition generator with mocked LLM."""

from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest

from src.services.generation.llm_client import LLMClient
from src.services.generation.step_def_generator import (
    _CUKE_PARAM_REGEX,
    _CUKE_PLACEHOLDER_RE,
    StepDefGenerationError,
    StepDefGenerator,
    _apply_compile_guarantees,
    _cuke_escape,
    _cuke_escape_slashes,
    _cuke_pattern_to_regex,
    _dedupe_step_defs,
    _duplicate_module_scope_bindings,
    _duplicate_step_patterns,
    _ensure_absolute_navigation,
    _ensure_bdd_bindings_in_scope,
    _ensure_expect_import,
    _ensure_grounded_role_names,
    _ensure_navigation_preconditions,
    _ensure_no_undefined_value_assertions,
    _ensure_resilient_import,
    _ensure_scenario_navigation,
    _ensure_scoped_resilient_targets,
    _ensure_step_coverage,
    _ensure_strict_safe_locators,
    _ensure_survivable_text_assertions,
    _ensure_valid_cuke_expressions,
    _extract_steps_from_features,
    _first_module_scope_violation,
    _inventory_role_names,
    _is_pure_literal,
    _iter_step_def_patterns,
    _normalize_param_types,
    _parse_step_defs,
    _reject_duplicate_module_scope_bindings,
    _sub_survivable_text_assertions,
    _subsumed_literal_patterns,
    _ts_decode_pattern,
    _uncovered_steps,
    _undefined_module_bindings,
)
from src.services.input.html_structure import LABEL_LENGTH_CAP

# A step file that (correctly) uses `page` only inside the callback.
_VALID_STEPS = (
    "```typescript\n"
    "// filename: todo.steps.ts\n"
    "import { createBdd } from 'playwright-bdd';\n"
    "const { Given, When, Then } = createBdd();\n"
    "const NEW_TODO = '.new-todo';\n"
    "Given('I am on the app', async ({ page }) => {\n"
    "  await page.goto('/');\n"
    "});\n"
    "When('I add a todo', async ({ page }) => {\n"
    "  await page.locator(NEW_TODO).fill('Buy milk');\n"
    "});\n"
    "```"
)

# A curried locator FACTORY at module scope: `page` is the INNER arrow's
# parameter, not a free module reference — valid TS, no ReferenceError (the
# function body isn't evaluated at module load). The guard's `const X = ...page...`
# regex FALSE-POSITIVED on this and hard-failed the run (observed live, todomvc
# 2026-07-24: `const TODO_ITEM = (text: string) => page => page.getByRole(...)`).
_MODULE_SCOPE_CURRIED_OK = (
    "```typescript\n"
    "// filename: todo.steps.ts\n"
    "import { createBdd } from 'playwright-bdd';\n"
    "const { Given, When, Then } = createBdd();\n"
    "const TODO_ITEM = (text: string) => page => page.getByRole('listitem').filter({ hasText: text });\n"
    "const ACTIVE_LINK = (page) => page.getByRole('link', { name: 'Active' });\n"
    "Given('I am on the app', async ({ page }) => {\n"
    "  await page.goto('/');\n"
    "  await TODO_ITEM('x')(page).click();\n"
    "});\n"
    "```"
)


# The observed live failure: a fixture used at MODULE scope (column 0) — throws
# `ReferenceError: page is not defined` when bddgen compiles.
_MODULE_SCOPE_STEPS = (
    "```typescript\n"
    "// filename: todo.steps.ts\n"
    "import { createBdd } from 'playwright-bdd';\n"
    "const { Given, When, Then } = createBdd();\n"
    "const FILTER_ACTIVE = page.getByRole('link', { name: 'Active' });\n"
    "Given('I am on the app', async ({ page }) => {\n"
    "  await page.goto('/');\n"
    "});\n"
    "```"
)


# The live kill (2026-08-15): the model answered with an English PLAN instead of
# code. Written verbatim as `steps.ts` this made Babel exit 1 mid-render.
_PROSE_ANSWER = (
    "# Step Definitions Plan\n\n"
    "I would structure the step definitions as follows.\n\n"
    "## Login\n"
    "First the user signs in, then the dashboard is asserted.\n"
)


# The observed live failure (todomvc, 2026-06-17): the same step pattern defined
# twice — bddgen fails with "Multiple definitions matched scenario step".
_DUPLICATE_STEP_STEPS = (
    "```typescript\n"
    "// filename: todo.steps.ts\n"
    "import { createBdd } from 'playwright-bdd';\n"
    "const { Given, When, Then } = createBdd();\n"
    "When('I add the following tasks:', async ({ page }) => {\n"
    "  await page.locator('.new-todo').fill('a');\n"
    "});\n"
    "When('I add the following tasks:', async ({ page }) => {\n"
    "  await page.locator('.new-todo').fill('b');\n"
    "});\n"
    "```"
)


# A duplicate whose SECOND (to-be-dropped) block contains a nested `});`
# (page.evaluate callback). A naive "next });" scan would truncate the block at
# the nested close and orphan the rest as broken TS.
_DUP_NESTED_IN_SECOND = (
    "```typescript\n"
    "// filename: t.steps.ts\n"
    "import { createBdd } from 'playwright-bdd';\n"
    "const { Given, When, Then } = createBdd();\n"
    "When('I compute', async ({ page }) => {\n"
    "  await page.goto('/');\n"
    "});\n"
    "When('I compute', async ({ page }) => {\n"
    "  const v = await page.evaluate(() => {\n"
    "    return document.title;\n"
    "  });\n"
    "  expect(v).toBeTruthy();\n"
    "});\n"
    "Then('I see results', async ({ page }) => {\n"
    "  await expect(page.locator('.results')).toBeVisible();\n"
    "});\n"
    "```"
)

# The live failure (todomvc, 2026-07-24 demo_video e2e): the LLM emitted BOTH a
# general parameterized def AND a hardcoded literal specialization of the SAME
# step. Their raw pattern strings differ, so exact dedup keeps both, and bddgen
# aborts: "Multiple definitions matched scenario step" — resolving
# `Given I have a task "Temporary task" in my list` to two defs. The literal is
# subsumed by the general def and must be dropped (keeping the general).
_GENERAL_PLUS_LITERAL_STEPS = (
    "```typescript\n"
    "// filename: todo.steps.ts\n"
    "import { createBdd } from 'playwright-bdd';\n"
    "const { Given, When, Then } = createBdd();\n"
    "Given('I have a task {string} in my list', async ({ page }, name) => {\n"
    "  await page.locator('.new-todo').fill(name);\n"
    "});\n"
    "Given('I have a task \"Temporary task\" in my list', async ({ page }) => {\n"
    "  await page.locator('.new-todo').fill('Temporary task');\n"
    "});\n"
    "```"
)


# A step-shaped pattern that only appears inside a COMMENT — must not count as a
# real definition (would otherwise trigger spurious dedup / block deletion).
_DUP_VIA_COMMENT = (
    "```typescript\n"
    "// filename: t.steps.ts\n"
    "import { createBdd } from 'playwright-bdd';\n"
    "const { Given, When, Then } = createBdd();\n"
    "When('I add a todo', async ({ page }) => {\n"
    "  // When('I add a todo') — implemented below\n"
    "  await page.locator('.new-todo').fill('x');\n"
    "});\n"
    "```"
)


@pytest.fixture
def mock_llm_client():
    """Create a mock LLMClient."""
    client = AsyncMock(spec=LLMClient)
    client.generate = AsyncMock()
    return client


@pytest.fixture
def step_gen(mock_llm_client):
    """Create a StepDefGenerator with a mock LLM client."""
    return StepDefGenerator(mock_llm_client)


class _Env:
    """`process.env` stand-in: a miss reads as undefined, exactly as in JS."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def __getattr__(self, name: str) -> str | None:
        return self._values.get(name)


class TestExtractSteps:
    """Tests for _extract_steps_from_features helper."""

    def test_extract_steps_basic(self):
        """Given/When/Then steps extracted from feature content."""
        features = [
            {
                "content": (
                    "Feature: Login\n"
                    "  Scenario: Login\n"
                    "    Given I am on the login page\n"
                    "    When I enter my username\n"
                    "    Then I should see the dashboard\n"
                ),
            }
        ]
        steps = _extract_steps_from_features(features)
        assert len(steps) == 3
        # `keyword` is the playwright-bdd BINDING; `gherkin_keyword` is the raw form.
        # For a primary keyword the two coincide.
        assert steps[0] == {
            "keyword": "Given",
            "gherkin_keyword": "Given",
            "text": "I am on the login page",
        }
        assert steps[1] == {
            "keyword": "When",
            "gherkin_keyword": "When",
            "text": "I enter my username",
        }
        assert steps[2] == {
            "keyword": "Then",
            "gherkin_keyword": "Then",
            "text": "I should see the dashboard",
        }

    def test_extract_steps_deduplication(self):
        """Duplicate steps filtered out."""
        features = [
            {
                "content": (
                    "  Scenario: A\n    Given I am on the login page\n    When I click submit\n"
                ),
            },
            {
                "content": (
                    "  Scenario: B\n    Given I am on the login page\n    When I click cancel\n"
                ),
            },
        ]
        steps = _extract_steps_from_features(features)
        texts = [s["text"] for s in steps]
        assert texts.count("I am on the login page") == 1
        assert len(steps) == 3

    def test_extract_steps_and_but_keywords(self):
        """And/But steps included."""
        features = [
            {
                "content": (
                    "  Scenario: Multi-step\n"
                    "    Given I am logged in\n"
                    "    And I have items in my cart\n"
                    "    When I click checkout\n"
                    "    But I have no payment method\n"
                    "    Then I should see an error\n"
                ),
            }
        ]
        steps = _extract_steps_from_features(features)
        assert len(steps) == 5
        # The RAW Gherkin keyword is preserved for diagnostics...
        assert [s["gherkin_keyword"] for s in steps] == ["Given", "And", "When", "But", "Then"]
        # ...but `keyword` is ALWAYS a real playwright-bdd binding: a continuation
        # inherits the kind of the preceding primary keyword. Emitting the raw `And`
        # here is what wrote a literal `And(...)` into a step-def file and killed a
        # live render with `ReferenceError: And is not defined`.
        assert [s["keyword"] for s in steps] == ["Given", "Given", "When", "When", "Then"]
        assert "And" not in [s["keyword"] for s in steps]
        assert "But" not in [s["keyword"] for s in steps]

    def test_extract_steps_empty_features(self):
        """No crashes on empty input."""
        assert _extract_steps_from_features([]) == []
        assert _extract_steps_from_features([{"content": ""}]) == []
        assert _extract_steps_from_features([{}]) == []


class TestParseStepDefs:
    """Tests for _parse_step_defs helper."""

    def test_parse_step_defs_fenced_blocks(self):
        """Extracts typescript code blocks."""
        raw = (
            "Here are the step defs:\n\n"
            "```typescript\n"
            "// filename: login_steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            "Given('I am on the login page', async ({ page }) => {\n"
            "  await page.goto('/login');\n"
            "});\n"
            "```\n\n"
            "```typescript\n"
            "// filename: nav_steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { When } = createBdd();\n"
            "When('I click {string}', async ({ page }, text) => {\n"
            "  await page.getByRole('button', { name: text }).click();\n"
            "});\n"
            "```\n"
        )
        defs = _parse_step_defs(raw)
        assert len(defs) == 2
        assert defs[0]["filename"] == "login_steps.ts"
        assert defs[1]["filename"] == "nav_steps.ts"
        assert "createBdd" in defs[0]["content"]
        assert defs[0]["step_count"] >= 1

    def test_parse_step_defs_filename_from_comment(self):
        """// filename: comment parsed."""
        raw = "```typescript\n// file: auth.ts\nGiven('I log in', async () => {});\n```"
        defs = _parse_step_defs(raw)
        assert len(defs) == 1
        assert defs[0]["filename"] == "auth.ts"

    def test_parse_step_defs_fallback(self):
        """No code blocks treats whole text as single file."""
        raw = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            "Given('step', async () => {});\n"
        )
        defs = _parse_step_defs(raw)
        assert len(defs) == 1
        assert defs[0]["filename"] == "steps.ts"
        assert "createBdd" in defs[0]["content"]

    def test_parse_step_defs_empty(self):
        """Returns empty list for empty or whitespace-only input."""
        assert _parse_step_defs("") == []
        assert _parse_step_defs("   \n  \n  ") == []

    # ── fence-tag tolerance + the prose regression (live kill 2026-08-15) ─────
    # The fence regex used to accept ONLY ```typescript / ```ts / bare ```, so any
    # other tag fell through to `blocks = [raw]` and the whole markdown answer was
    # written verbatim as steps.ts → Babel parsed English → bddgen exit 1.

    def test_parse_step_defs_tsx_fence(self):
        """A ```tsx fence is stripped, not swallowed into the raw fallback."""
        raw = (
            "```tsx\n"
            "// filename: tsx_steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            "Given('I open the app', async ({ page }) => {\n"
            "  await page.goto('/');\n"
            "});\n"
            "```\n"
        )
        defs = _parse_step_defs(raw)
        assert len(defs) == 1
        assert defs[0]["filename"] == "tsx_steps.ts"
        assert "```" not in defs[0]["content"]
        assert defs[0]["content"].startswith("// filename: tsx_steps.ts")

    def test_parse_step_defs_javascript_fence(self):
        """A ```javascript fence is stripped."""
        raw = (
            "```javascript\n"
            "// filename: js_steps.ts\n"
            "const { When } = createBdd();\n"
            "When('I click save', async ({ page }) => {});\n"
            "```\n"
        )
        defs = _parse_step_defs(raw)
        assert len(defs) == 1
        assert defs[0]["filename"] == "js_steps.ts"
        assert "```" not in defs[0]["content"]
        assert "javascript" not in defs[0]["content"]

    def test_parse_step_defs_capitalised_typescript_fence(self):
        """A capitalised ```TypeScript fence is stripped (tag match is case-insensitive)."""
        raw = (
            "```TypeScript\n"
            "// filename: caps_steps.ts\n"
            "const { Then } = createBdd();\n"
            "Then('I see the result', async ({ page }) => {});\n"
            "```\n"
        )
        defs = _parse_step_defs(raw)
        assert len(defs) == 1
        assert defs[0]["filename"] == "caps_steps.ts"
        assert "```" not in defs[0]["content"]
        assert "TypeScript\n" not in defs[0]["content"]

    def test_parse_step_defs_unfenced_pure_code(self):
        """An unfenced answer that IS code still parses as a single file."""
        raw = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, When } = createBdd();\n"
            "Given('I am signed in', async ({ page }) => {});\n"
            "When('I open settings', async ({ page }) => {});\n"
        )
        defs = _parse_step_defs(raw)
        assert len(defs) == 1
        assert defs[0]["filename"] == "steps.ts"
        assert defs[0]["step_count"] == 2

    def test_parse_step_defs_prose_answer_raises(self):
        """REGRESSION (prod kill 2026-08-15): a markdown/prose answer must RAISE.

        Previously `blocks = [raw]` wrote this entire response out as `steps.ts`;
        Babel then parsed English as TypeScript and bddgen exited 1 mid-render.
        """
        raw = (
            "# Step Definitions Plan\n\n"
            "I would structure the step definitions as follows.\n\n"
            "## Login\n"
            "First the user signs in, then the dashboard is asserted.\n"
        )
        with pytest.raises(StepDefGenerationError) as excinfo:
            _parse_step_defs(raw)
        # The message must carry the offending head so the failure is diagnosable.
        assert "Step Definitions Plan" in str(excinfo.value)

    def test_parse_step_defs_typescript_fence_unchanged(self):
        """A well-formed ```typescript answer is byte-identical to prior behaviour."""
        raw = (
            "Here you go:\n\n"
            "```typescript\n"
            "// filename: login_steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            "Given('I am on the login page', async ({ page }) => {\n"
            "  await page.goto('/login');\n"
            "});\n"
            "```\n"
        )
        defs = _parse_step_defs(raw)
        assert defs == [
            {
                "filename": "login_steps.ts",
                "content": (
                    "// filename: login_steps.ts\n"
                    "import { createBdd } from 'playwright-bdd';\n"
                    "const { Given } = createBdd();\n"
                    "Given('I am on the login page', async ({ page }) => {\n"
                    "  await page.goto('/login');\n"
                    "});"
                ),
                "step_count": 1,
            }
        ]

    # ── false-rejection regressions (adversarial review of the first cut) ─────
    # The first cut rejected PER BLOCK on "no Given/When/Then" and on any markdown
    # heading. Both fire on legal answers, and a rejection fails the whole paid job.

    def test_parse_step_defs_support_hooks_file_is_not_rejected(self):
        """D1: a steps file + a Before/After-only `hooks.ts` → BOTH files, no raise."""
        raw = (
            "```typescript\n"
            "// filename: login.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            "Given('I am on the login page', async ({ page }) => {\n"
            "  await page.goto('/login');\n"
            "});\n"
            "```\n\n"
            "```typescript\n"
            "// filename: hooks.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Before, After } = createBdd();\n"
            "Before(async ({ page }) => {\n"
            "  await page.setViewportSize({ width: 1280, height: 720 });\n"
            "});\n"
            "After(async ({ page }) => {\n"
            "  await page.close();\n"
            "});\n"
            "```\n"
        )
        defs = _parse_step_defs(raw)
        assert [d["filename"] for d in defs] == ["login.steps.ts", "hooks.ts"]

    def test_parse_step_defs_fixtures_file_is_not_rejected(self):
        """D1: a `test.extend` fixtures block alongside a steps block → both returned."""
        raw = (
            "```typescript\n"
            "// filename: fixtures.ts\n"
            "import { test } from 'playwright-bdd';\n"
            "export const adminTest = test.extend({\n"
            "  adminPage: async ({ page }, use) => { await use(page); },\n"
            "});\n"
            "```\n\n"
            "```typescript\n"
            "// filename: search.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { When } = createBdd();\n"
            "When('I search for {string}', async ({ page }, q) => {});\n"
            "```\n"
        )
        defs = _parse_step_defs(raw)
        assert [d["filename"] for d in defs] == ["fixtures.ts", "search.steps.ts"]

    def test_parse_step_defs_destructured_alias_binding(self):
        """D2: `const { Given: given } = createBdd(); given(...)` is valid and accepted."""
        raw = (
            "```typescript\n"
            "// filename: alias.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given: given, When: when } = createBdd();\n"
            "given('I am signed in', async ({ page }) => {});\n"
            "when('I open settings', async ({ page }) => {});\n"
            "```\n"
        )
        defs = _parse_step_defs(raw)
        assert len(defs) == 1
        assert defs[0]["filename"] == "alias.steps.ts"

    def test_parse_step_defs_markdown_heading_inside_template_literal(self):
        """D3: a `# Overview` line inside a TEMPLATE LITERAL is valid TypeScript."""
        raw = (
            "```typescript\n"
            "// filename: docs.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { When } = createBdd();\n"
            "const DOC = `\n"
            "# Overview\n"
            "\n"
            "Some body copy for the editor.\n"
            "`;\n"
            "When('I paste the document', async ({ page }) => {\n"
            "  await page.getByRole('textbox').fill(DOC);\n"
            "});\n"
            "```\n"
        )
        defs = _parse_step_defs(raw)
        assert len(defs) == 1
        assert "# Overview" in defs[0]["content"]

    def test_parse_step_defs_markdown_heading_inside_banner_comment(self):
        """D3: a `# heading` line inside a leading `/* ... */` banner is valid code."""
        raw = (
            "```typescript\n"
            "/*\n"
            "# Login steps\n"
            "Generated for the QA suite.\n"
            "*/\n"
            "// filename: banner.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Then } = createBdd();\n"
            "Then('I see the dashboard', async ({ page }) => {});\n"
            "```\n"
        )
        defs = _parse_step_defs(raw)
        assert len(defs) == 1
        assert defs[0]["filename"] == "banner.steps.ts"

    def test_parse_step_defs_pure_markdown_plan_still_raises(self):
        """PRODUCTION REGRESSION: a fenced markdown plan with no code signal RAISES."""
        raw = (
            "```markdown\n"
            "# Step Definitions Plan\n"
            "\n"
            "I would structure the step definitions as follows.\n"
            "\n"
            "## Login\n"
            "First the user signs in, then the dashboard is asserted.\n"
            "```\n"
        )
        with pytest.raises(StepDefGenerationError, match="not TypeScript step definitions"):
            _parse_step_defs(raw)

    def test_parse_step_defs_no_block_registers_a_step_raises(self):
        """SET-LEVEL rule: code everywhere but no step binding anywhere → RAISE."""
        raw = (
            "```typescript\n"
            "// filename: hooks.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Before } = createBdd();\n"
            "Before(async ({ page }) => {});\n"
            "```\n\n"
            "```typescript\n"
            "// filename: fixtures.ts\n"
            "import { test as base } from 'playwright-bdd';\n"
            "export const test = base.extend({});\n"
            "```\n"
        )
        with pytest.raises(StepDefGenerationError, match="no block registers"):
            _parse_step_defs(raw)


class TestProseAnswerSelfHeal:
    """A prose answer is TRANSIENT: `generate` must re-ask ONCE, not kill the run.

    Rejecting prose at the first parse fixed a corruption bug but would trade it for
    an availability regression — one bad completion failing a whole paid demo_video
    run. The retry is bounded at exactly one extra LLM call and the second attempt is
    gated just as strictly (no fallback to writing prose out).
    """

    @pytest.mark.asyncio
    async def test_prose_then_valid_regenerates_once_and_succeeds(
        self, step_gen, mock_llm_client, sample_inspection_result
    ):
        """Prose first, TypeScript second → the call SUCCEEDS with the 2nd response."""
        mock_llm_client.generate.side_effect = [_PROSE_ANSWER, _VALID_STEPS]
        features = [{"content": "  Scenario: A\n    Given I am on the app\n"}]

        result = await step_gen.generate(features, sample_inspection_result)

        # BOUND: exactly two LLM calls — one bad, one corrective retry, no more.
        assert mock_llm_client.generate.await_count == 2
        # The step defs come from the SECOND response, not the prose.
        assert result and result[0]["filename"] == "todo.steps.ts"
        joined = "\n".join(sd["content"] for sd in result)
        assert "Given('I am on the app'" in joined
        assert "Step Definitions Plan" not in joined
        # The retry carried the format correction on top of the original prompt.
        retry_prompt = mock_llm_client.generate.await_args_list[1].args[0]
        assert "CORRECTION" in retry_prompt
        assert "```typescript" in retry_prompt
        assert mock_llm_client.generate.await_args_list[0].args[0] in retry_prompt

    @pytest.mark.asyncio
    async def test_prose_twice_raises_without_further_retry(
        self, step_gen, mock_llm_client, sample_inspection_result
    ):
        """Prose on BOTH attempts → raise. No infinite retry, no prose written out."""
        mock_llm_client.generate.side_effect = [_PROSE_ANSWER, _PROSE_ANSWER]
        features = [{"content": "  Scenario: A\n    Given I am on the app\n"}]

        with pytest.raises(StepDefGenerationError, match="not TypeScript step definitions"):
            await step_gen.generate(features, sample_inspection_result)

        # BOUND: the second parse raises, so the LLM is called exactly twice.
        assert mock_llm_client.generate.await_count == 2


class TestStepDefGenerator:
    """Tests for StepDefGenerator.generate with mocked LLM."""

    async def test_generate_calls_llm(self, step_gen, mock_llm_client, sample_inspection_result):
        """Mock LLM, verify prompt contains steps and pages."""
        mock_llm_client.generate.return_value = (
            "```typescript\n"
            "// filename: steps.ts\n"
            "Given('I am on the login page', async ({ page }) => {});\n"
            "```"
        )

        features = [
            {
                "content": (
                    "  Scenario: Login\n"
                    "    Given I am on the login page\n"
                    "    When I enter my username\n"
                ),
            }
        ]

        await step_gen.generate(features, sample_inspection_result, platform="web")

        mock_llm_client.generate.assert_awaited_once()
        call_args = mock_llm_client.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")

        # Prompt should contain the step texts
        assert "I am on the login page" in prompt
        assert "I enter my username" in prompt

        # Prompt should contain page info from inspection
        assert "Login" in prompt
        assert "#username" in prompt or "username" in prompt

    async def test_generate_prompt_empty_label_falls_back_to_text(self, step_gen, mock_llm_client):
        """An element with an empty accessible name (label == "") must reach the
        LLM as its visible text — `.get("label", text)` returns the present but
        empty "", so a blank label was landing in the pages_summary."""
        mock_llm_client.generate.return_value = (
            "```typescript\n// filename: steps.ts\nGiven('x', async ({ page }) => {});\n```"
        )
        inspection = {
            "base_url": "https://example.com",
            "pages": [
                {
                    "title": "Home",
                    "url": "https://example.com",
                    "interactive_elements": [
                        {"tag": "button", "label": "", "text": "Save Changes", "selector": "#save"},
                    ],
                }
            ],
        }
        features = [{"content": "  Scenario: Home\n    Given I am on the page\n"}]

        await step_gen.generate(features, inspection, platform="web")

        call_args = mock_llm_client.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "Save Changes" in prompt
        # The blank-label line must not be emitted (was "button:  -> #save").
        assert "button:  ->" not in prompt

    async def test_generate_prompt_carries_content_visibility_gate(
        self, step_gen, mock_llm_client, sample_inspection_result
    ):
        """O2: the LLM demo path must instruct a networkidle + toBeVisible gate
        before capture, so a generated suite never dwells on a half-loaded UI."""
        mock_llm_client.generate.return_value = (
            "```typescript\n// filename: steps.ts\nGiven('x', async ({ page }) => {});\n```"
        )
        features = [{"content": "  Scenario: Home\n    Given I am on the page\n"}]

        await step_gen.generate(features, sample_inspection_result, platform="web")

        call_args = mock_llm_client.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "networkidle" in prompt
        assert "toBeVisible" in prompt
        # And it must say body/domcontentloaded alone is insufficient.
        assert "not sufficient" in prompt.lower()

    async def test_generate_returns_parsed_defs(
        self, step_gen, mock_llm_client, sample_inspection_result
    ):
        """Verify return structure matches expected format."""
        mock_llm_client.generate.return_value = (
            "```typescript\n"
            "// filename: login_steps.ts\n"
            "Given('I am logged in', async ({ page }) => {\n"
            "  await page.goto('/login');\n"
            "});\n"
            "When('I click submit', async ({ page }) => {\n"
            "  await page.click('button');\n"
            "});\n"
            "```"
        )

        features = [
            {
                "content": (
                    "  Scenario: Login\n    Given I am logged in\n    When I click submit\n"
                ),
            }
        ]

        result = await step_gen.generate(features, sample_inspection_result)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["filename"] == "login_steps.ts"
        assert "content" in result[0]
        assert result[0]["step_count"] >= 2


class TestModuleScopeGuard:
    """The generated step defs must not use `page`/`expect` at module scope."""

    def test_detects_module_scope_fixture(self):
        defs = _parse_step_defs(_MODULE_SCOPE_STEPS)
        v = _first_module_scope_violation(defs)
        assert v is not None
        assert "FILTER_ACTIVE" in v

    def test_passes_valid_callback_scoped_defs(self):
        defs = _parse_step_defs(_VALID_STEPS)
        assert _first_module_scope_violation(defs) is None

    def test_curried_page_factory_not_flagged(self):
        # `const X = (text) => page => page.getByRole(...)` and
        # `const Y = (page) => page.getByRole(...)` bind `page` as an arrow
        # PARAMETER — valid TS, no module-load ReferenceError. Must NOT be flagged
        # (this false positive hard-failed the live todomvc run 2026-07-24).
        defs = _parse_step_defs(_MODULE_SCOPE_CURRIED_OK)
        assert _first_module_scope_violation(defs) is None

    def test_direct_module_scope_page_still_flagged(self):
        # The real violation (`const F = page.getByRole(...)`, page dereferenced at
        # module load) must STILL be caught after the curried-factory relaxation.
        defs = _parse_step_defs(_MODULE_SCOPE_STEPS)
        v = _first_module_scope_violation(defs)
        assert v is not None
        assert "FILTER_ACTIVE" in v

    @pytest.mark.asyncio
    async def test_regenerates_once_then_succeeds(
        self, step_gen, mock_llm_client, sample_inspection_result
    ):
        """A module-scope violation triggers ONE corrective regeneration."""
        mock_llm_client.generate.side_effect = [_MODULE_SCOPE_STEPS, _VALID_STEPS]
        features = [{"content": "  Scenario: A\n    Given I am on the app\n"}]
        result = await step_gen.generate(features, sample_inspection_result)
        assert mock_llm_client.generate.await_count == 2
        assert result and result[0]["filename"] == "todo.steps.ts"
        # The retry carried the corrective hint.
        assert "CORRECTION" in mock_llm_client.generate.await_args_list[1].args[0]

    @pytest.mark.asyncio
    async def test_raises_when_still_invalid_after_retry(
        self, step_gen, mock_llm_client, sample_inspection_result
    ):
        """If regeneration is still module-scoped, fail with a precise error."""
        mock_llm_client.generate.side_effect = [_MODULE_SCOPE_STEPS, _MODULE_SCOPE_STEPS]
        features = [{"content": "  Scenario: A\n    Given I am on the app\n"}]
        with pytest.raises(StepDefGenerationError, match="outside a step callback"):
            await step_gen.generate(features, sample_inspection_result)
        assert mock_llm_client.generate.await_count == 2


class TestEnsureBddBindingsInScope:
    """`_ensure_bdd_bindings_in_scope` must inject only MISSING names.

    Regression for the reported bug: a file that already destructures a SUBSET
    (`const { Given, Then } = createBdd();`) but also calls `When(...)` used to
    get a second, FULL `const { Given, When, Then } = createBdd();` injected —
    redeclaring `Given` and `Then` and killing `bddgen` with an unattributed
    Babel `VarRedeclaration` before a single assertion ran.
    """

    def test_subset_binding_gets_only_the_missing_name(self):
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, Then } = createBdd();\n"
            "Given('a', async ({ page }) => {});\n"
            "When('b', async ({ page }) => {});\n"
            "Then('c', async ({ page }) => {});\n"
        )
        defs: list[dict[str, Any]] = [{"filename": "s.ts", "content": content}]
        fixed = _ensure_bdd_bindings_in_scope(defs)
        assert fixed == 1
        new = defs[0]["content"]
        # No redeclaration: exactly one `Given`/`Then` binding site each.
        assert not _duplicate_module_scope_bindings(new)
        # `When` is now bound, and the pre-existing subset destructure survives.
        assert "const { Given, Then } = createBdd();" in new
        assert "const { When } = createBdd();" in new

    def test_no_bindings_at_all_gets_the_full_destructure(self):
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "Given('a', async ({ page }) => {});\n"
            "When('b', async ({ page }) => {});\n"
            "Then('c', async ({ page }) => {});\n"
        )
        defs: list[dict[str, Any]] = [{"filename": "s.ts", "content": content}]
        fixed = _ensure_bdd_bindings_in_scope(defs)
        assert fixed == 1
        assert "const { Given, When, Then } = createBdd();" in defs[0]["content"]
        assert not _duplicate_module_scope_bindings(defs[0]["content"])

    def test_fully_bound_file_is_untouched(self):
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, When, Then } = createBdd();\n"
            "Given('a', async ({ page }) => {});\n"
            "When('b', async ({ page }) => {});\n"
            "Then('c', async ({ page }) => {});\n"
        )
        defs: list[dict[str, Any]] = [{"filename": "s.ts", "content": content}]
        fixed = _ensure_bdd_bindings_in_scope(defs)
        assert fixed == 0
        assert defs[0]["content"] == content


class TestRejectDuplicateModuleScopeBindings:
    """The new compile gate must catch a redeclared module-scope binding."""

    def test_raises_on_llm_emitted_duplicate_const(self):
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, Then } = createBdd();\n"
            "const { Given, When, Then } = createBdd();\n"
            "Given('a', async ({ page }) => {});\n"
            "When('b', async ({ page }) => {});\n"
            "Then('c', async ({ page }) => {});\n"
        )
        defs: list[dict[str, Any]] = [{"filename": "dup.steps.ts", "content": content}]
        with pytest.raises(StepDefGenerationError, match=r"dup\.steps\.ts") as exc_info:
            _reject_duplicate_module_scope_bindings(defs)
        # Names the duplicated identifier — the diagnosis Babel refuses to give.
        assert "Given" in str(exc_info.value)
        assert "Then" in str(exc_info.value)

    def test_no_duplicates_does_not_raise(self):
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, Then } = createBdd();\n"
            "const { When } = createBdd();\n"
            "Given('a', async ({ page }) => {});\n"
            "When('b', async ({ page }) => {});\n"
            "Then('c', async ({ page }) => {});\n"
        )
        defs: list[dict[str, Any]] = [{"filename": "ok.steps.ts", "content": content}]
        _reject_duplicate_module_scope_bindings(defs)  # must not raise

    def test_repeated_indented_local_across_step_bodies_is_not_flagged(self):
        """FALSE-POSITIVE regression: an ordinary `const row` declared inside TWO
        separate step callback bodies is valid, ordinary JavaScript — each is a
        local scoped to its own callback, not a module-scope redeclaration. The
        duplicate detector must be column-0-anchored, or it flags nearly every
        real generated file (this is the coordinator's exact reported case).
        """
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, When } = createBdd();\n"
            "Given('a todo exists', async ({ page }) => {\n"
            "  const row = page.getByRole('listitem').first();\n"
            "});\n"
            "When('I complete it', async ({ page }) => {\n"
            "  const row = page.getByRole('listitem').first();\n"
            "});\n"
        )
        defs: list[dict[str, Any]] = [{"filename": "row.steps.ts", "content": content}]
        assert _duplicate_module_scope_bindings(content) == []
        _reject_duplicate_module_scope_bindings(defs)  # must not raise

    def test_indented_local_shadowing_a_module_scope_name_is_not_flagged(self):
        """An indented local that SHADOWS a module-scope name (e.g. a step body
        declaring its own `const Given = ...` helper, however unlikely) is still
        lexically valid JS inside its own block — only a column-0 redeclaration
        of the SAME name is the real Babel `VarRedeclaration` failure mode.
        """
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, When } = createBdd();\n"
            "Given('a', async ({ page }) => {\n"
            "  const When = 'not the binding, just a shadowed local';\n"
            "  void When;\n"
            "});\n"
            "When('b', async ({ page }) => {});\n"
        )
        defs: list[dict[str, Any]] = [{"filename": "shadow.steps.ts", "content": content}]
        assert _duplicate_module_scope_bindings(content) == []
        _reject_duplicate_module_scope_bindings(defs)  # must not raise

    def test_repeated_decl_inside_template_literal_is_not_flagged(self):
        """FALSE-POSITIVE regression: a column-0-looking `const row = …` line
        that actually sits INSIDE a multi-line template literal (a banner, an
        embedded SQL/GraphQL string, a JSDoc usage example rendered as text)
        is not a JS declaration at all — it is string content. The anchored
        regexes have no string/comment awareness on their own, so without
        masking via `_code_mask` this valid, compiling file false-positives
        the compile gate and kills the whole generation.
        """
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const banner = `\n"
            "const row = 1;\n"
            "const row = 2;\n"
            "`;\n"
            "const { Given } = createBdd();\n"
            "Given('a', async ({ page }) => {\n"
            "  void banner;\n"
            "});\n"
        )
        defs: list[dict[str, Any]] = [{"filename": "template.steps.ts", "content": content}]
        assert _duplicate_module_scope_bindings(content) == []
        _reject_duplicate_module_scope_bindings(defs)  # must not raise

    def test_repeated_decl_inside_block_comment_is_not_flagged(self):
        """Same false-positive class as the template-literal case, for a
        `/* … */` block comment (e.g. a JSDoc usage example)."""
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "/*\n"
            "const helper = 1;\n"
            "const helper = 2;\n"
            "*/\n"
            "const { Given } = createBdd();\n"
            "Given('a', async ({ page }) => {});\n"
        )
        defs: list[dict[str, Any]] = [{"filename": "comment.steps.ts", "content": content}]
        assert _duplicate_module_scope_bindings(content) == []
        _reject_duplicate_module_scope_bindings(defs)  # must not raise


class TestDuplicateStepGuard:
    """The generated step defs must not define the same pattern twice — bddgen
    fails with 'Multiple definitions matched scenario step' otherwise."""

    def test_detects_duplicate_pattern(self):
        defs = _parse_step_defs(_DUPLICATE_STEP_STEPS)
        dups = _duplicate_step_patterns(defs)
        assert dups == ["I add the following tasks:"]

    def test_detects_duplicate_across_files(self):
        two_files = _DUPLICATE_STEP_STEPS.replace(
            "When('I add the following tasks:', async ({ page }) => {\n"
            "  await page.locator('.new-todo').fill('b');\n"
            "});\n",
            "```\n```typescript\n// filename: extra.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, When, Then } = createBdd();\n"
            "Then('I add the following tasks:', async ({ page }) => {\n"
            "  await page.goto('/');\n"
            "});\n",
        )
        defs = _parse_step_defs(two_files)
        assert _duplicate_step_patterns(defs) == ["I add the following tasks:"]

    def test_passes_distinct_patterns(self):
        defs = _parse_step_defs(_VALID_STEPS)
        assert _duplicate_step_patterns(defs) == []

    @pytest.mark.asyncio
    async def test_regenerates_once_then_succeeds(
        self, step_gen, mock_llm_client, sample_inspection_result
    ):
        """A duplicate pattern triggers ONE corrective regeneration."""
        mock_llm_client.generate.side_effect = [_DUPLICATE_STEP_STEPS, _VALID_STEPS]
        features = [{"content": "  Scenario: A\n    When I add the following tasks:\n"}]
        result = await step_gen.generate(features, sample_inspection_result)
        assert mock_llm_client.generate.await_count == 2
        assert result and result[0]["filename"] == "todo.steps.ts"
        correction = mock_llm_client.generate.await_args_list[1].args[0]
        assert "Multiple definitions matched" in correction

    @pytest.mark.asyncio
    async def test_dedupes_when_still_duplicated_after_retry(
        self, step_gen, mock_llm_client, sample_inspection_result
    ):
        """If the LLM still emits a duplicate after the corrective regeneration,
        the extra definition is dropped deterministically — the run does NOT fail."""
        mock_llm_client.generate.side_effect = [_DUPLICATE_STEP_STEPS, _DUPLICATE_STEP_STEPS]
        features = [{"content": "  Scenario: A\n    When I add the following tasks:\n"}]
        result = await step_gen.generate(features, sample_inspection_result)
        assert mock_llm_client.generate.await_count == 2
        # The pattern now appears exactly once across the deduped output.
        assert _duplicate_step_patterns(result) == []
        joined = "\n".join(sd["content"] for sd in result)
        assert joined.count("When('I add the following tasks:'") == 1

    def test_dedupe_keeps_first_block_and_drops_rest(self):
        """_dedupe_step_defs removes the 2nd+ definition of a pattern, keeping the
        first body and leaving distinct patterns intact."""
        defs = _parse_step_defs(_DUPLICATE_STEP_STEPS)
        deduped = _dedupe_step_defs(defs)
        content = deduped[0]["content"]
        assert content.count("When('I add the following tasks:'") == 1
        # The FIRST body (fill('a')) is the one kept.
        assert "fill('a')" in content
        assert "fill('b')" not in content
        assert _duplicate_step_patterns(deduped) == []

    def test_dedupe_across_files(self):
        two_files = _DUPLICATE_STEP_STEPS.replace(
            "When('I add the following tasks:', async ({ page }) => {\n"
            "  await page.locator('.new-todo').fill('b');\n"
            "});\n",
            "```\n```typescript\n// filename: extra.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, When, Then } = createBdd();\n"
            "Then('I add the following tasks:', async ({ page }) => {\n"
            "  await page.goto('/');\n"
            "});\n",
        )
        deduped = _dedupe_step_defs(_parse_step_defs(two_files))
        assert _duplicate_step_patterns(deduped) == []

    def test_dedupe_leaves_distinct_patterns_untouched(self):
        defs = _parse_step_defs(_VALID_STEPS)
        deduped = _dedupe_step_defs(defs)
        assert "I am on the app" in deduped[0]["content"]
        assert "I add a todo" in deduped[0]["content"]

    def test_dedupe_nested_close_does_not_truncate(self):
        """Dropping a duplicate whose body has a nested `});` must remove the
        WHOLE block, not stop at the nested close and orphan broken TS."""
        deduped = _dedupe_step_defs(_parse_step_defs(_DUP_NESTED_IN_SECOND))
        content = deduped[0]["content"]
        assert _duplicate_step_patterns(deduped) == []
        assert content.count("When('I compute'") == 1
        # The dropped 2nd block's body must be entirely gone — no orphaned tail.
        assert "toBeTruthy" not in content
        assert "document.title" not in content
        # The first block's body and the unrelated Then block survive intact.
        assert "page.goto('/')" in content
        assert "I see results" in content
        assert "toBeVisible" in content

    def test_comment_pattern_not_counted_as_duplicate(self):
        """A `Given(`/`When(`-shaped pattern inside a comment is not a definition."""
        defs = _parse_step_defs(_DUP_VIA_COMMENT)
        assert _duplicate_step_patterns(defs) == []
        # And dedupe leaves the single real block untouched (the comment, which
        # repeats the pattern text, must not trigger a removal).
        deduped = _dedupe_step_defs(defs)
        content = deduped[0]["content"]
        assert "await page.locator('.new-todo').fill('x')" in content
        assert content.count("async ({ page }) => {") == 1  # exactly one real def block


class TestEnsureScenarioNavigation:
    """Every @clip-<n> scenario records in a FRESH browser context (about:blank =
    black). A clip is non-blank only if a step runs page.goto — but the LLM emits
    later scenarios that assume the app is already loaded, and dwelling stubs /
    soft-failing selectors never navigate → black clip → OpenCV sharpness 0.0 →
    demo_video_record blank-gate hard-fail (observed live todomvc 2026-07-24,
    clips 3-6). A Before hook navigating to the app baseURL fixes it."""

    def test_injects_before_nav_hook(self):
        defs = _parse_step_defs(_VALID_STEPS)
        n = _ensure_scenario_navigation(defs)
        assert n == 1
        hook = defs[-1]
        assert hook["filename"] == "steps_hooks.ts"
        # playwright-bdd v8 scenario hook, its own createBdd(), navigates to app.
        assert "import { createBdd } from 'playwright-bdd';" in hook["content"]
        assert "const { Before } = createBdd();" in hook["content"]
        # Navigates to the FULL target via the baseURL fixture, NOT goto('/')
        # (which strips a sub-path off baseURL and records a blank clip).
        assert "Before(async ({ page, baseURL }) =>" in hook["content"]
        assert "await page.goto(baseURL, {" in hook["content"]
        assert "waitUntil: 'domcontentloaded'," in hook["content"]
        assert "await page.goto('/')" not in hook["content"]
        # Cold first nav must retry and wait for a REAL painted DOM, not networkidle
        # (which an SPA/redirect reaches while still blank — this is what blanked clip-1).
        assert "waitForFunction" in hook["content"]
        # The fire-and-forget `waitForLoadState('networkidle')` CALL is gone (networkidle
        # is not a paint signal); the word may still appear in an explanatory comment.
        assert "waitForLoadState('networkidle')" not in hook["content"]

    def test_hook_budget_is_bounded_by_the_per_test_timeout(self):
        # A `Before` hook runs INSIDE the test, so the per-test Playwright --timeout
        # (exported by frontend_test_runner as PW_TEST_TIMEOUT_MS) bounds it. The old
        # fixed `3 attempts x (implicit 15s goto + 10s wait)` = 75s could not fit in a
        # 20s test budget: attempts 2-3 were DEAD CODE and Playwright killed the test
        # inside the hook -> zero steps, page still about:blank -> guaranteed BLACK clip.
        defs = _parse_step_defs(_VALID_STEPS)
        _ensure_scenario_navigation(defs)
        content = defs[-1]["content"]
        assert "Number(process.env.PW_TEST_TIMEOUT_MS) || 20000" in content
        # Deadline loop, not a fixed attempt count.
        assert "const deadline = Date.now() + NAV_HOOK_BUDGET_MS;" in content
        assert "while (Date.now() < deadline)" in content
        assert "attempt < 3" not in content
        # BOTH legs carry an explicit timeout computed from the time remaining, so a
        # single attempt can never overrun the budget via the implicit navigationTimeout.
        assert "timeout: Math.max(1000, Math.floor(remaining * 0.6))," in content
        assert "{ timeout: Math.max(500, deadline - Date.now()) }," in content

    def test_paint_predicate_accepts_a_textless_app(self):
        # A canvas-only / image-only / icon-only app paints perfectly with zero
        # innerText; requiring text there burned the whole budget on a HEALTHY page.
        defs = _parse_step_defs(_VALID_STEPS)
        _ensure_scenario_navigation(defs)
        content = defs[-1]["content"]
        assert "b.innerText.trim().length > 0" in content
        assert "b.querySelectorAll('canvas, img, svg, video')" in content
        assert "backgroundImage" in content

    def test_media_counts_as_paint_only_when_actually_rendered(self):
        """A bare `querySelector('svg')` match is NOT paint.

        An app-shell `index.html` routinely ships an inline <svg> icon sprite, a
        `display:none` logo, or a 1x1 tracking pixel — all of which match on the
        FIRST poll while the page is still white. Waving those through would
        re-open the very blank-clip hole this hook exists to close, so every media
        candidate must clear a visibility + loaded check.
        """
        defs = _parse_step_defs(_VALID_STEPS)
        _ensure_scenario_navigation(defs)
        content = defs[-1]["content"]
        assert "const isPainted = (el) => {" in content
        # a rendered box, not a 1x1 pixel
        assert "r.width < MIN_PAINT_PX || r.height < MIN_PAINT_PX" in content
        # not hidden by any of the three usual mechanisms
        assert "s.display === 'none' || s.visibility === 'hidden'" in content
        assert "Number(s.opacity) === 0" in content
        # an <img> that has not decoded yet paints nothing
        assert "el.complete && el.naturalWidth > 0" in content
        # the background-image branch is gated by the same check
        assert "bg !== 'none' && isPainted(els[i])" in content

    def test_hook_budget_never_exceeds_the_test_timeout(self):
        """The 5000ms floor must not outrank the ceiling.

        FTR accepts `timeout_per_test_ms` down to 1000, and the orchestrator
        forwards a brief override with the same lower bound. An unclamped
        `Math.max(5000, ...)` would hand the hook a budget >= the whole test
        timeout for any value below ~8300 — Playwright then kills the test
        mid-hook, zero steps run and the clip records black, which is exactly the
        defect this rewrite fixes.
        """
        defs = _parse_step_defs(_VALID_STEPS)
        _ensure_scenario_navigation(defs)
        content = defs[-1]["content"]
        assert "Math.min(PW_TEST_TIMEOUT_MS - 2000" in content

    def test_exhausted_budget_soft_fail_marker_and_throws(self):
        # The old `catch {}` swallowed and returned normally, so a nav that never
        # painted produced a silently BLACK clip later blamed on the compiler.
        # The `throw` is what actually fails the scenario attributably. The
        # ##SOFT_FAIL## marker is diagnostic only: it matches the format
        # recording_helpers.ts emits, but FTR's `_has_soft_fail` reads the JSON-report
        # ANNOTATION, not stdout — so the marker aids log forensics, it does not
        # drive demotion.
        defs = _parse_step_defs(_VALID_STEPS)
        _ensure_scenario_navigation(defs)
        content = defs[-1]["content"]
        assert "console.warn(`##SOFT_FAIL## ${detail}`);" in content
        assert "throw new Error(" in content
        # The error is captured (bare `catch {` cannot report why it failed).
        assert "} catch (err) {" in content
        assert "lastError = String(err);" in content
        # ...but an absent baseURL is still a clean no-op, not a failure.
        assert "if (!baseURL) return;" in content

    def test_idempotent(self):
        defs = _parse_step_defs(_VALID_STEPS)
        assert _ensure_scenario_navigation(defs) == 1
        # Re-running finds the hook file already present → adds nothing.
        assert _ensure_scenario_navigation(defs) == 0
        assert sum(1 for sd in defs if sd["filename"] == "steps_hooks.ts") == 1

    def test_hook_is_inert_to_step_def_passes(self):
        # The Before hook is NOT a Given/When/Then, so dedup / duplicate / coverage
        # machinery must ignore it (no spurious duplicate or double-match).
        defs = _parse_step_defs(_VALID_STEPS)
        _ensure_scenario_navigation(defs)
        assert _duplicate_step_patterns(defs) == []
        assert _subsumed_literal_patterns(defs) == set()

    def test_recording_pipeline_injects_nav(self):
        from src.services.generation.step_def_generator import _apply_compile_guarantees

        defs = _parse_step_defs(_VALID_STEPS)
        out = _apply_compile_guarantees(defs, [], recording=True)
        hooks = [sd for sd in out if sd["filename"] == "steps_hooks.ts"]
        assert len(hooks) == 1
        assert "await page.goto(baseURL, {" in hooks[0]["content"]
        assert "PW_TEST_TIMEOUT_MS" in hooks[0]["content"]

    def test_non_recording_stream_gets_no_nav_hook(self):
        # QA/accessibility/documentation streams are honest test suites — forcing a
        # goto('/') before every scenario could mask a real nav failure or break a
        # test that navigates elsewhere. Recording (demo_video) stream only.
        from src.services.generation.step_def_generator import _apply_compile_guarantees

        defs = _parse_step_defs(_VALID_STEPS)
        out = _apply_compile_guarantees(defs, [], recording=False)
        assert not any(sd["filename"] == "steps_hooks.ts" for sd in out)


class TestPaintLeadAnnotation:
    """Every acceptance film opens on a blank/loading frame because the recorded
    video starts at CONTEXT CREATION (about:blank), before this hook's nav +
    paint-probe resolves — the lead-edge twin of the failed-scenario TAIL trimmed
    by #166/#1042. Measured live: kb3/kb4/kb5 acceptance runs all scored the
    opening shot 2.0/5 for a "blank/white frame" lasting up to ~2.3s. The hook
    records exactly how long that took as a `paint-lead` annotation — the same
    channel (`test.info().annotations.push`) `resilientClick`/`resilientFill`
    already use for `soft-fail` — so frontend_test_runner_service can trim the
    clip's front by a measured amount instead of guessing from pixels."""

    def test_hook_imports_test_for_annotations(self):
        defs = _parse_step_defs(_VALID_STEPS)
        _ensure_scenario_navigation(defs)
        content = defs[-1]["content"]
        assert "import { test } from '@playwright/test';" in content

    def test_hook_records_a_paint_lead_annotation_on_success(self):
        defs = _parse_step_defs(_VALID_STEPS)
        _ensure_scenario_navigation(defs)
        content = defs[-1]["content"]
        assert "const hookStartAt = Date.now();" in content
        assert "type: 'paint-lead'," in content
        assert "description: String(Date.now() - hookStartAt)," in content
        # Pushed right after the paint probe resolves (before the retry-loop's
        # success `return;`), not on the failure/throw path.
        push_at = content.index("type: 'paint-lead',")
        wait_for_function_at = content.index("await page.waitForFunction(")
        throw_at = content.index("throw new Error(")
        assert wait_for_function_at < push_at < throw_at

    def test_annotation_push_never_throws_out_of_the_hook(self):
        # An annotation-bookkeeping failure (e.g. this file unit-tested outside a
        # running Playwright test) must not turn a successful nav into a hard
        # scenario failure — the clip would still publish, just untrimmed.
        defs = _parse_step_defs(_VALID_STEPS)
        _ensure_scenario_navigation(defs)
        content = defs[-1]["content"]
        push_at = content.index("test.info().annotations.push({")
        preceding = content[:push_at]
        assert preceding.rstrip().endswith("try {")

    def test_hook_is_still_inert_to_step_def_passes(self):
        defs = _parse_step_defs(_VALID_STEPS)
        _ensure_scenario_navigation(defs)
        assert _duplicate_step_patterns(defs) == []
        assert _subsumed_literal_patterns(defs) == set()


class TestEnsureAbsoluteNavigation:
    """The nav `Before` hook opens the FULL `baseURL`, then an LLM-emitted
    `await page.goto('/')` inside a step navigates AWAY from it to the bare ORIGIN.
    Under a sub-pathed `use.baseURL` (e.g. https://host/todomvc) that origin is a
    404 / marketing shell / blank page, so the clip records black anyway — which is
    precisely why the hook alone (PR #92/#96) did not stop the blank clips. Rewrite
    the model's own navigation to be baseURL-preserving."""

    @staticmethod
    def _def(content: str) -> dict[str, Any]:
        return {"filename": "nav.steps.ts", "content": content, "step_count": 1}

    _RELATIVE = (
        "// filename: nav.steps.ts\n"
        "import { createBdd } from 'playwright-bdd';\n"
        "const { Given, When } = createBdd();\n"
        "Given('I am on the app', async ({ page }) => {\n"
        "  await page.goto('/');\n"
        "});\n"
        "When('I open settings', async ({ page }) => {\n"
        "  await page.goto('/settings', { waitUntil: 'domcontentloaded' });\n"
        "});\n"
    )

    @staticmethod
    def _assert_structurally_sane(content: str) -> None:
        """Cheap stand-in for `tsc --noEmit`, run on every rewritten output.

        An earlier version of this pass spliced `baseURL` into each callback's
        destructured fixture argument and emitted NON-COMPILING TypeScript for six
        ordinary shapes (verified with real tsc): a prettier trailing comma
        (`{ page, }` -> `{ page,, baseURL }`, TS1180), a multi-line destructure (same),
        `...rest` (TS2462), a `baseURL: alias` rename (TS2304), a `{ page }: { page: Page }`
        type annotation (TS2339), and a nested `async ({ request })` handler (TS2304).
        That is STRICTLY WORSE than the bug being fixed: a file that does not compile
        fails the whole suite, returns ZERO clips, and hard-fails the paid render —
        versus one wrong-page clip. The current pass never touches a destructure at
        all, and these assertions keep it that way.
        """
        assert ",," not in content, content
        assert "...rest, baseURL" not in content, content
        # The rewrite must not introduce a bare `baseURL` reference: the helper
        # sources the base URL from process.env, so no fixture is required.
        assert "absoluteNavUrl(baseURL" not in content, content
        for ch_open, ch_close in (("{", "}"), ("(", ")")):
            assert content.count(ch_open) == content.count(ch_close), content

    def test_relative_goto_rewritten_to_env_absolute(self):
        defs = [self._def(self._RELATIVE)]
        assert _ensure_absolute_navigation(defs) == 1
        c = defs[0]["content"]
        # The origin-stripping form is gone…
        assert "page.goto('/')" not in c
        assert "page.goto('/settings'" not in c
        # …replaced by the sub-path-preserving helper, which is emitted once and takes
        # ONLY the route (no fixture), so no destructure is ever modified.
        assert "await page.goto(absoluteNavUrl('/'));" in c
        assert "await page.goto(absoluteNavUrl('/settings'), {" in c
        assert c.count("function absoluteNavUrl(") == 1
        # `new URL(path, base)` would re-introduce the very bug (a leading-slash route
        # resolves against the ORIGIN), so the helper CONCATENATES: trim a trailing
        # `/` off the base, a leading `/` off the route, join with `/`.
        assert "const base = (env?.BASE_URL ?? '').replace(/\\/+$/, '');" in c
        assert "const rel = route.replace(/^\\/+/, '');" in c
        assert "return rel ? `${base}/${rel}` : base;" in c
        # The fixture destructures are untouched — that is the whole safety property.
        assert c.count("async ({ page })") == 2
        # No CODE line references a `baseURL` fixture (the helper's comments mention it).
        code = [ln for ln in c.splitlines() if not ln.lstrip().startswith("//")]
        assert not [ln for ln in code if "baseURL" in ln], code
        self._assert_structurally_sane(c)

    @pytest.mark.parametrize(
        "fixture_arg",
        [
            "async ({ page, })",  # prettier trailingComma:"all"
            "async ({\n  page,\n})",  # multi-line destructure
            "async ({ page, ...rest })",  # rest element
            "async ({ page, baseURL: b })",  # renamed fixture
            "async ({ page }: { page: Page })",  # type annotation
        ],
    )
    def test_exotic_destructures_are_never_mangled(self, fixture_arg: str):
        """Each of these emitted non-compiling TS in the destructure-splice design."""
        src = (
            "// filename: nav.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            f"Given('I am on the app', {fixture_arg} => {{\n"
            "  await page.goto('/');\n"
            "});\n"
        )
        defs = [self._def(src)]
        assert _ensure_absolute_navigation(defs) == 1
        c = defs[0]["content"]
        # The goto IS fixed, and the destructure is returned byte-identical.
        assert "await page.goto(absoluteNavUrl('/'));" in c
        assert fixture_arg in c
        self._assert_structurally_sane(c)

    def test_nested_handler_does_not_perturb_the_outer_step(self):
        """A nested `async ({ request })` handler used to become a rewrite anchor, so
        `baseURL` landed in the WRONG scope (TS2304). Nothing is spliced now."""
        src = (
            "// filename: nav.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            "Given('I am on the app', async ({ page }) => {\n"
            "  await page.route('**/x', async ({ request }) => { await request; });\n"
            "  await page.goto('/');\n"
            "});\n"
        )
        defs = [self._def(src)]
        assert _ensure_absolute_navigation(defs) == 1
        c = defs[0]["content"]
        assert "async ({ request })" in c
        assert "await page.goto(absoluteNavUrl('/'));" in c
        self._assert_structurally_sane(c)

    def test_tab_indented_steps_do_not_break_scoping(self):
        """An indented step file never matched the old column-0 `});` close, so the
        final segment ran to EOF and module scope got rewritten (TS2304)."""
        src = (
            "// filename: nav.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            "\tGiven('a', async ({ page }) => {\n"
            "\t\tawait page.goto('/');\n"
            "\t});\n"
        )
        defs = [self._def(src)]
        assert _ensure_absolute_navigation(defs) == 1
        self._assert_structurally_sane(defs[0]["content"])

    def test_commented_out_goto_is_left_alone(self):
        src = (
            "// filename: nav.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            "Given('a', async ({ page }) => {\n"
            "  // await page.goto('/old');\n"
            "  await page.goto('/');\n"
            "});\n"
        )
        defs = [self._def(src)]
        assert _ensure_absolute_navigation(defs) == 1
        c = defs[0]["content"]
        assert "// await page.goto('/old');" in c
        assert "await page.goto(absoluteNavUrl('/'));" in c

    def test_absolute_and_about_blank_left_byte_identical(self):
        src = (
            "// filename: nav.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            "Given('I am on the site', async ({ page }) => {\n"
            "  await page.goto('https://example.com/x');\n"
            "  await page.goto('about:blank');\n"
            "});\n"
        )
        defs = [self._def(src)]
        assert _ensure_absolute_navigation(defs) == 0
        assert defs[0]["content"] == src

    def test_already_baseurl_template_left_alone(self):
        src = (
            "// filename: nav.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            "Given('I am on the app', async ({ page, baseURL }) => {\n"
            "  await page.goto(`${baseURL}/settings`);\n"
            "});\n"
        )
        defs = [self._def(src)]
        assert _ensure_absolute_navigation(defs) == 0
        assert defs[0]["content"] == src

    def test_idempotent(self):
        once = [self._def(self._RELATIVE)]
        assert _ensure_absolute_navigation(once) == 1
        twice = [self._def(self._RELATIVE)]
        _ensure_absolute_navigation(twice)
        # A second run finds no string-literal argument and no missing helper.
        assert _ensure_absolute_navigation(twice) == 0
        assert twice[0]["content"] == once[0]["content"]

    def test_module_scope_helper_with_variable_route_left_intact(self):
        # `baseURL` is a per-step FIXTURE — it does not exist inside a shared
        # module-scope helper, and the argument is an identifier we cannot classify
        # as relative. Rewriting either would emit TS that does not compile, so the
        # call is LEFT (and logged) rather than broken.
        src = (
            "// filename: nav.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { When } = createBdd();\n"
            "async function open(page, route) {\n"
            "  await page.goto(route, { waitUntil: 'domcontentloaded' });\n"
            "}\n"
            "When('I open {string}', async ({ page }, route) => {\n"
            "  await open(page, route);\n"
            "});\n"
        )
        defs = [self._def(src)]
        assert _ensure_absolute_navigation(defs) == 0
        assert defs[0]["content"] == src

    def test_module_scope_helper_literal_is_also_fixed(self):
        """A module-scope helper's own relative goto is a real blank-clip source too.
        The destructure-splice design had to SKIP it (no `baseURL` fixture out there);
        sourcing the base URL from process.env means it is valid everywhere, so the
        fix now reaches it — the payoff of dropping the fixture."""
        src = (
            "// filename: nav.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { When } = createBdd();\n"
            "When('I start', async ({ page }) => {\n"
            "  await page.goto('/');\n"
            "});\n"
            "\n"
            "async function reset(page) {\n"
            "  await page.goto('/reset');\n"
            "}\n"
        )
        defs = [self._def(src)]
        assert _ensure_absolute_navigation(defs) == 1
        c = defs[0]["content"]
        assert "await page.goto(absoluteNavUrl('/'));" in c
        assert "await page.goto(absoluteNavUrl('/reset'));" in c
        self._assert_structurally_sane(c)

    def test_recording_pipeline_rewrites_relative_goto(self):
        from src.services.generation.step_def_generator import _apply_compile_guarantees

        defs = _parse_step_defs(_VALID_STEPS)
        out = _apply_compile_guarantees(defs, [], recording=True)
        steps = next(sd for sd in out if sd["filename"] == "todo.steps.ts")
        assert "page.goto('/')" not in steps["content"]
        assert "absoluteNavUrl('/')" in steps["content"]
        self._assert_structurally_sane(steps["content"])

    def test_non_recording_stream_is_not_rewritten(self):
        # QA/accessibility/documentation suites are honest tests, not recordings:
        # they may legitimately assert on relative navigation, and no
        # `lib/recording_helpers` package ships with them. Recording stream only.
        from src.services.generation.step_def_generator import _apply_compile_guarantees

        defs = _parse_step_defs(_VALID_STEPS)
        out = _apply_compile_guarantees(defs, [], recording=False)
        joined = "".join(sd["content"] for sd in out)
        assert "await page.goto('/');" in joined
        assert "absoluteNavUrl" not in joined


class TestSubsumedLiteralGuard:
    """A pure-literal step def that another (parameterized) def already matches is
    a bddgen 'Multiple definitions matched' hard failure — exact dedup misses it
    because the raw pattern strings differ. Drop the subsumed literal, keep the
    general def (coverage never shrinks: the literal's one concrete step is still
    matched by the surviving general def)."""

    def test_is_pure_literal(self):
        assert _is_pure_literal('I have a task "Temporary task" in my list') is True
        assert _is_pure_literal("I have a task {string} in my list") is False  # param
        assert _is_pure_literal("I press button(s)") is False  # optional group
        assert _is_pure_literal("I have a cat/dog") is False  # alternation
        assert _is_pure_literal(r"I use a \{ literal") is False  # escape

    def test_detects_literal_subsumed_by_string_param(self):
        defs = _parse_step_defs(_GENERAL_PLUS_LITERAL_STEPS)
        subsumed = _subsumed_literal_patterns(defs)
        assert subsumed == {'I have a task "Temporary task" in my list'}

    def test_general_def_is_never_flagged(self):
        # The parameterized def must never be dropped — it is the one we keep.
        defs = _parse_step_defs(_GENERAL_PLUS_LITERAL_STEPS)
        subsumed = _subsumed_literal_patterns(defs)
        assert "I have a task {string} in my list" not in subsumed

    def test_distinct_literals_not_flagged(self):
        # Two unrelated literal defs do not subsume each other.
        defs = _parse_step_defs(_VALID_STEPS)
        assert _subsumed_literal_patterns(defs) == set()

    def test_overlapping_params_not_flagged(self):
        # Neither def is a pure literal → out of this pass's precise scope (bddgen
        # remains the backstop). Documents the intentional boundary.
        two_params = (
            "```typescript\n// filename: t.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, When, Then } = createBdd();\n"
            "Given('I pick {string} now', async ({ page }, v) => { await page.goto('/'); });\n"
            "Given('I pick {word} now', async ({ page }, v) => { await page.goto('/'); });\n"
            "```"
        )
        assert _subsumed_literal_patterns(_parse_step_defs(two_params)) == set()

    def test_generate_drops_subsumed_literal_end_to_end(self):
        """The full compile-guarantee pass removes the literal block, keeps the
        general block intact, and yields a suite bddgen accepts (no ambiguity)."""
        from src.services.generation.step_def_generator import _apply_compile_guarantees

        defs = _parse_step_defs(_GENERAL_PLUS_LITERAL_STEPS)
        out = _apply_compile_guarantees(defs, [], recording=True)
        content = "\n".join(sd["content"] for sd in out)
        # The general (parameterized) def survives with its body.
        assert "I have a task {string} in my list" in content
        assert "fill(name)" in content
        # The subsumed literal def and its body are gone.
        assert 'I have a task "Temporary task" in my list' not in content
        assert "fill('Temporary task')" not in content
        # No residual ambiguity for bddgen.
        assert _subsumed_literal_patterns(out) == set()

    def test_literal_first_order_independent(self):
        """Whichever order the LLM emits them, the LITERAL is dropped and the
        GENERAL survives — never the reverse."""
        literal_first = (
            "```typescript\n// filename: todo.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, When, Then } = createBdd();\n"
            "Given('I have a task \"Temporary task\" in my list', async ({ page }) => {\n"
            "  await page.locator('.new-todo').fill('Temporary task');\n"
            "});\n"
            "Given('I have a task {string} in my list', async ({ page }, name) => {\n"
            "  await page.locator('.new-todo').fill(name);\n"
            "});\n"
            "```"
        )
        from src.services.generation.step_def_generator import _apply_compile_guarantees

        out = _apply_compile_guarantees(_parse_step_defs(literal_first), [], recording=True)
        content = "\n".join(sd["content"] for sd in out)
        assert "I have a task {string} in my list" in content
        assert 'I have a task "Temporary task" in my list' not in content
        assert "fill(name)" in content


# The live failure (todomvc, 2026-07): the LLM destructures only `{ page }` and
# calls `expect(...)` with NO import → `ReferenceError: expect is not defined` at
# run time (bddgen compiles fine) → every scenario fails → BLANK demo footage.
_EXPECT_NO_IMPORT = (
    "// filename: todo.steps.ts\n"
    "import { createBdd } from 'playwright-bdd';\n"
    "const { Given, When, Then } = createBdd();\n"
    "Then('I see the todo', async ({ page }, text) => {\n"
    "  await expect(page.getByLabel(text)).toBeVisible();\n"
    "});\n"
)

_FIXTURE_RE = re.compile(r"async\s*\(\s*\{[^}]*\bexpect\b")


class TestEnsureExpectImport:
    """`expect` must be a module import, never a fixture — the deterministic fix."""

    def test_injects_import_when_expect_called_without_import(self):
        defs = [{"filename": "todo.steps.ts", "content": _EXPECT_NO_IMPORT, "step_count": 1}]
        fixed = _ensure_expect_import(defs)
        assert fixed == 1
        content = defs[0]["content"]
        assert "import { expect } from '@playwright/test';" in content
        # placed right after the playwright-bdd import, before createBdd() use
        assert content.index("@playwright/test") < content.index("createBdd()")
        # exactly one import (no duplication)
        assert content.count("from '@playwright/test'") == 1

    def test_strips_expect_from_fixture_destructure(self):
        src = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Then } = createBdd();\n"
            "Then('x', async ({ page, expect }, t) => {\n"
            "  await expect(page.getByLabel(t)).toBeVisible();\n"
            "});\n"
        )
        defs = [{"filename": "s.ts", "content": src, "step_count": 1}]
        assert _ensure_expect_import(defs) == 1
        content = defs[0]["content"]
        # expect no longer destructured from ANY fixture arg
        assert _FIXTURE_RE.search(content) is None
        assert "import { expect } from '@playwright/test';" in content
        # page fixture survives
        assert "page" in content

    def test_strips_expect_when_first_in_destructure(self):
        src = (
            "import { createBdd } from 'playwright-bdd';\n"
            "Then('x', async ({ expect, page }) => {\n"
            "  await expect(page).toBeVisible();\n"
            "});\n"
        )
        defs = [{"filename": "s.ts", "content": src, "step_count": 1}]
        _ensure_expect_import(defs)
        content = defs[0]["content"]
        assert _FIXTURE_RE.search(content) is None
        assert "page" in content
        assert "import { expect } from '@playwright/test';" in content

    def test_idempotent_when_already_imported(self):
        src = (
            "import { createBdd } from 'playwright-bdd';\n"
            "import { expect } from '@playwright/test';\n"
            "const { Then } = createBdd();\n"
            "Then('x', async ({ page }) => {\n"
            "  await expect(page).toBeVisible();\n"
            "});\n"
        )
        defs = [{"filename": "s.ts", "content": src, "step_count": 1}]
        assert _ensure_expect_import(defs) == 0
        assert defs[0]["content"] == src
        assert defs[0]["content"].count("from '@playwright/test'") == 1

    def test_noop_when_expect_not_used(self):
        defs = [{"filename": "s.ts", "content": _VALID_STEPS, "step_count": 1}]
        before = defs[0]["content"]
        assert _ensure_expect_import(defs) == 0
        assert defs[0]["content"] == before

    def test_injects_at_top_without_pwbdd_anchor(self):
        # No playwright-bdd import line to anchor on — insert after the filename
        # comment.
        src = "// filename: s.ts\nThen('x', async ({ page }) => { await expect(page).toBeVisible(); });\n"
        defs = [{"filename": "s.ts", "content": src, "step_count": 1}]
        assert _ensure_expect_import(defs) == 1
        lines = defs[0]["content"].splitlines()
        assert lines[0].startswith("// filename:")
        assert lines[1] == "import { expect } from '@playwright/test';"

    async def test_generate_injects_import_end_to_end(self, step_gen, mock_llm_client):
        """The full generate() pipeline emits a compilable, expect-imported suite."""
        mock_llm_client.generate.return_value = "```typescript\n" + _EXPECT_NO_IMPORT + "```"
        features = [{"content": "Feature: Todo\n  Scenario: See\n    Then I see the todo\n"}]
        defs = await step_gen.generate(features, {"base_url": "https://x"})
        assert len(defs) == 1
        assert "import { expect } from '@playwright/test';" in defs[0]["content"]
        # regenerate must NOT have been triggered (no module-scope/dup issue)
        assert mock_llm_client.generate.await_count == 1


def _def(content: str, name: str = "steps.ts") -> dict[str, Any]:
    return {"filename": name, "content": content, "step_count": 1}


def _covers(kind: str, source: str, text: str) -> bool | None:
    rx = _cuke_pattern_to_regex(kind, source)
    return None if rx is None else bool(rx.fullmatch(text))


class TestCukePatternMatcher:
    """The coverage matcher must be a SUPERSET of cucumber-expressions 18.0.1:
    match ≥ everything cucumber does (so a covered step is never stubbed →
    double-match), yet tight enough that a generic def does not swallow an
    unrelated step (which would leave it unstubbed → undefined-step → blank clip).
    """

    def test_string_param_matches_quoted_token_only(self):
        assert _covers("str", "I click {string}", 'I click "Save"') is True
        assert _covers("str", "I click {string}", "I click 'Save'") is True
        # MUST NOT swallow trailing words (the real gherkin↔pattern-mismatch bug)
        assert _covers("str", "I click {string}", 'I click the box next to "X" again') is False

    def test_string_param_handles_escaped_quotes(self):
        assert _covers("str", "I type {string}", r'I type "a\"b"') is True

    def test_numeric_params_match_numbers_not_words(self):
        assert _covers("str", "I see {int} items", "I see 5 items") is True
        assert _covers("str", "I see {int} items", "I see -5 items") is True
        assert _covers("str", "I see {int} items", "I see many items") is False
        assert _covers("str", "I see {double} items", "I see 5.5 items") is True
        assert _covers("str", "I see {float} items", "I see .5 items") is True
        # cucumber's less-common numeric types must not be treated as literal text
        assert _covers("str", "I see {long} items", "I see 9 items") is True
        assert _covers("str", "I see {bigdecimal} items", "I see 9 items") is True

    def test_word_param(self):
        assert _covers("str", "I pick {word} now", "I pick red now") is True
        assert _covers("str", "I pick {word} now", "I pick a b now") is False

    def test_unknown_param_over_matches(self):
        # A custom/unknown param type falls back to permissive `.+?` (over-match,
        # the safe direction) so it never under-matches → never double-stubs.
        assert _covers("str", "I do {frobnicate} it", "I do anything it") is True

    def test_anonymous_param_matches_empty(self):
        # cucumber's anonymous `{}` is `.*` (zero-or-more) — the matcher MUST
        # match the empty capture too, else it under-matches → double-stub.
        assert _covers("str", "prefix {} suffix", "prefix  suffix") is True
        assert _covers("str", "prefix {} suffix", "prefix X suffix") is True

    def test_optional_group(self):
        assert _covers("str", "I press button(s)", "I press button") is True
        assert _covers("str", "I press button(s)", "I press buttons") is True

    def test_alternation_and_regex_literal_bail_to_covers_all(self):
        # We do not mirror UNESCAPED `/` alternation or JS regex literals, and we
        # refuse to guess at a TS escape outside the provable set — bail to
        # covers-all (None) rather than risk under-matching. (ESCAPED slashes ARE
        # modelled now; see TestCukePatternModelsEscapes.)
        assert _cuke_pattern_to_regex("str", "I have a cat/dog") is None
        assert _cuke_pattern_to_regex("re", "^anything$") is None
        # `\{` decodes to a bare `{` → unterminated parameter, unmodellable.
        assert _cuke_pattern_to_regex("str", r"I use a \{ literal") is None
        # a TS escape we will not decode (`\n`) makes the whole pattern unmodellable
        assert _cuke_pattern_to_regex("str", r"line\nbreak") is None

    def test_plain_literal_exact_match(self):
        assert _covers("str", "I am on the app", "I am on the app") is True
        assert _covers("str", "I am on the app", "I am on the app now") is False


class TestCukeEscape:
    """A stub's cucumber-expression literal must escape cucumber specials so a
    `/` (alternation), `(` (optional) or `{` (param) in the step text matches
    literally — and never breaks the generated TypeScript."""

    def test_escapes_cucumber_specials(self):
        assert _cuke_escape('I open "/dashboard" (x)') == r'I open "\/dashboard" \(x\)'
        assert _cuke_escape("a {b} c") == r"a \{b\} c"
        assert _cuke_escape(r"back\slash") == "back\\\\slash"

    def test_quotes_are_literal(self):
        # Double/single quotes are literal in a cucumber expression (not escaped).
        assert _cuke_escape('say "hi"') == 'say "hi"'


class TestEnsureValidCukeExpressions:
    """An LLM-authored pattern must COMPILE as a cucumber expression. `/` is the
    alternation operator, so the `//` of a URL mirrored out of the feature
    sentence is an EMPTY alternative — cucumber-expressions throws, bddgen exits
    1 and the paid demo_video render dies at `demo_video_record` (live
    2026-08-15). Escape `/` only: `{...}` params and `(...)` optional groups are
    correct uses the model makes constantly and MUST survive untouched.
    """

    def test_url_slashes_are_escaped_so_bddgen_compiles(self):
        defs = [_def("When('I navigate to \"https://example.com\"', async ({page}) => {});")]
        assert _ensure_valid_cuke_expressions(defs) == 1
        # TS source carries `\\/`, which decodes to the `\/` cucumber needs.
        assert r'I navigate to "https:\\/\\/example.com"' in defs[0]["content"]
        # no bare (unescaped) slash survives in the pattern
        patterns = _iter_step_def_patterns(defs[0]["content"])
        assert [k for k, _ in patterns] == ["str"]
        assert re.search(r"(?<!\\)/", patterns[0][1]) is None

    def test_parameters_and_optional_groups_are_untouched(self):
        src = "When('I click {string} {int} time(s)', async ({page}) => {});"
        defs = [_def(src)]
        assert _ensure_valid_cuke_expressions(defs) == 0
        assert defs[0]["content"] == src

    def test_already_escaped_slash_is_not_double_escaped(self):
        # `\\/` in TS source already decodes to the cucumber escape `\/`.
        defs = [_def("When('I open \"https:\\\\/\\\\/x.dev\"', async ({page}) => {});")]
        assert _ensure_valid_cuke_expressions(defs) == 0
        # and the pass is idempotent on its own output
        run1 = [_def("When('I go to \"https://x.dev\"', async ({page}) => {});")]
        _ensure_valid_cuke_expressions(run1)
        once = run1[0]["content"]
        assert _ensure_valid_cuke_expressions(run1) == 0
        assert run1[0]["content"] == once

    def test_optional_group_with_a_slash_keeps_its_group(self):
        defs = [_def("When('I press button(s) at /home', async ({page}) => {});")]
        assert _ensure_valid_cuke_expressions(defs) == 1
        assert r"I press button(s) at \\/home" in defs[0]["content"]

    def test_regex_literal_patterns_are_not_rewritten(self):
        # `/` is the RegExp delimiter there and has no cucumber meaning.
        src = "When(/^I navigate to .+$/, async ({page}) => {});"
        defs = [_def(src)]
        assert _ensure_valid_cuke_expressions(defs) == 0
        assert defs[0]["content"] == src

    def test_escaping_runs_before_coverage_and_keeps_matching_sound(self):
        """Guards the ordering claim: escaping must not make a covered step look
        uncovered (a stub for an already-defined step is a bddgen 'Multiple
        definitions matched' hard failure — strictly worse than the bug fixed).

        `_cuke_pattern_to_regex` now MODELS the escape it writes: `\\\\/` in TS
        source decodes to the cucumber escape `\\/`, which matches a literal `/`
        in the step text. So the escaped def still covers the original feature
        step and nothing is stubbed. (It used to reach the same 0 for the wrong
        reason — the bail disabled stubbing for the ENTIRE job; see
        TestCukePatternModelsEscapes.)
        """
        feats = [{"content": 'Feature: t\n  Scenario: s\n    When I navigate to "https://x.dev"\n'}]
        defs = [_def("When('I navigate to \"https://x.dev\"', async ({page}) => {});")]
        _ensure_valid_cuke_expressions(defs)
        assert _uncovered_steps(feats, defs) == []
        assert _ensure_step_coverage(feats, defs) == 0

    def test_full_guarantee_pipeline_escapes_llm_pattern(self):
        defs = [
            _def(
                "import { createBdd } from 'playwright-bdd';\n"
                "const { When } = createBdd();\n"
                "When('I navigate to \"https://example.com\"', async ({ page }) => {\n"
                "  await page.goto('https://example.com');\n"
                "});\n"
            )
        ]
        out = _apply_compile_guarantees(defs, [], recording=False)
        assert r"""When('I navigate to "https:\\/\\/example.com"',""" in out[0]["content"]
        # the step BODY (not a pattern) keeps its real, unescaped URL
        assert "await page.goto('https://example.com');" in out[0]["content"]


class TestCukeEscapeSlashes:
    """Source-level parity rule: a TS-source `\\` run of k backslashes decodes to
    k//2 value-level backslashes; the slash is escaped for cucumber iff that
    count is ODD."""

    def test_bare_slash_gets_a_source_level_double_backslash(self):
        assert _cuke_escape_slashes("a/b") == "a\\\\/b"
        assert _cuke_escape_slashes("https://x") == "https:\\\\/\\\\/x"

    def test_source_double_backslash_is_left_alone(self):
        assert _cuke_escape_slashes("a\\\\/b") == "a\\\\/b"

    def test_redundant_ts_escape_is_upgraded_not_kept(self):
        # `\/` in TS source decodes to a BARE `/` — cucumber never sees an escape,
        # so it must be upgraded, not mistaken for already-escaped.
        assert _cuke_escape_slashes("a\\/b") == "a\\\\\\/b"

    def test_is_idempotent(self):
        for src in ("a/b", "https://x", "a\\/b", "a\\\\/b", "no slashes"):
            once = _cuke_escape_slashes(src)
            assert _cuke_escape_slashes(once) == once

    def test_leaves_params_and_groups_alone(self):
        assert _cuke_escape_slashes("I click {string} {int} time(s)") == (
            "I click {string} {int} time(s)"
        )


class TestTsDecodePattern:
    """A pattern's raw text is still TypeScript-escaped. Decoding it is what lets
    the coverage matcher see what the CUCUMBER parser sees — but only decodes we
    can prove are faithful, since a wrong one under-matches (→ duplicate stub →
    bddgen 'Multiple definitions matched')."""

    def test_no_backslash_is_returned_unchanged(self):
        assert _ts_decode_pattern("I click {string}") == "I click {string}"

    def test_double_backslash_decodes_to_the_cucumber_escape(self):
        # `\\/` in TS source is the ONE spelling that reaches cucumber as `\/`.
        assert _ts_decode_pattern(r"https:\\/\\/x") == r"https:\/\/x"

    def test_redundant_ts_escape_decodes_to_a_bare_slash(self):
        assert _ts_decode_pattern(r"a\/b") == "a/b"

    def test_escaped_quotes_decode(self):
        assert _ts_decode_pattern(r"I type \"hi\"") == 'I type "hi"'

    def test_undecodable_escapes_bail(self):
        assert _ts_decode_pattern(r"line\nbreak") is None
        assert _ts_decode_pattern(r"tab\there") is None
        assert _ts_decode_pattern("dangling\\") is None


class TestCukePatternModelsEscapes:
    """THE GAP THIS CLOSES. `_ensure_valid_cuke_expressions` escapes `/` so bddgen
    stops dying on `https://`. But `_cuke_pattern_to_regex` used to bail on ANY
    `\\`, and `_uncovered_steps` returns `[]` (stub NOTHING for the WHOLE job) as
    soon as one def is unmodellable. So a URL-bearing pattern disabled coverage
    stubbing before the escape fix (via `/`) and still disabled it after (via
    `\\`): every genuinely uncovered step lost its autostub → 'undefined step' at
    record time → BLANK clip. Trading a loud crash for a blank clip is a bad deal.
    """

    _ESCAPED = r'I navigate to "https:\\/\\/x.dev"'  # TS source, post-escape
    _STEP_TEXT = 'I navigate to "https://x.dev"'

    def test_escaped_slash_pattern_is_modellable(self):
        assert _cuke_pattern_to_regex("str", self._ESCAPED) is not None

    def test_generated_regex_matches_the_original_unescaped_step_text(self):
        # The round trip: TS `\\/` → cucumber `\/` → a LITERAL `/` in step text.
        assert _covers("str", self._ESCAPED, self._STEP_TEXT) is True
        # and it stays tight — it must not swallow an unrelated step.
        assert _covers("str", self._ESCAPED, 'I navigate to "https://other.dev"') is False

    def test_escaped_backslash_matches_a_literal_backslash(self):
        # `_cuke_escape` emits 4 source backslashes for one literal `\`.
        assert _covers("str", r"back\\\\slash", r"back\slash") is True

    def test_params_and_optional_groups_still_work_beside_an_escape(self):
        src = r"I open \\/{word} {int} time(s)"
        assert _covers("str", src, "I open /home 2 times") is True
        assert _covers("str", src, "I open /home 2 time") is True
        assert _covers("str", src, "I open /home many times") is False

    def test_escaped_step_is_reported_covered_not_stubbed(self):
        # A stub for an already-defined step is 'Multiple definitions matched' —
        # a DIFFERENT bddgen crash. Coverage must see through the escape.
        feats = [{"content": f"Feature: t\n  Scenario: s\n    When {self._STEP_TEXT}\n"}]
        defs = [_def(f"When('{self._STEP_TEXT}', async ({{page}}) => {{}});")]
        assert _ensure_valid_cuke_expressions(defs) == 1
        assert _uncovered_steps(feats, defs) == []
        assert _ensure_step_coverage(feats, defs) == 0

    def test_stubbing_still_works_for_the_rest_of_the_job(self):
        """The regression this closes: ONE escaped-slash def used to disable
        step-coverage stubbing for EVERY step in the job."""
        feats = [
            {
                "content": (
                    "Feature: t\n  Scenario: s\n"
                    f"    When {self._STEP_TEXT}\n"
                    "    And I frobnicate the widget\n"
                )
            }
        ]
        defs = [_def(f"When('{self._STEP_TEXT}', async ({{page}}) => {{}});")]
        _ensure_valid_cuke_expressions(defs)
        assert _uncovered_steps(feats, defs) == ["I frobnicate the widget"]
        assert _ensure_step_coverage(feats, defs) == 1
        stub = defs[-1]["content"]
        assert '"I frobnicate the widget"' in stub
        # ...and the DEFINED step is not re-stubbed (no duplicate definition).
        assert "I navigate to" not in stub

    def test_unescaped_alternation_is_still_the_bail(self):
        # The bail is narrowed, not deleted: real alternation stays unmodellable.
        assert _cuke_pattern_to_regex("str", "I have a cat/dog") is None


class TestMultiLineStepDefPatterns:
    """`_STEP_DEF_RE` needs the quote on the binding-call line, so a pattern on
    the FOLLOWING line was invisible to every pattern pass — its `/` was never
    escaped and bddgen still died with 'Alternative may not be empty'."""

    _MULTILINE = (
        "import { createBdd } from 'playwright-bdd';\n"
        "const { When } = createBdd();\n"
        "When(\n"
        "  'I navigate to \"https://x.dev\"',\n"
        "  async ({ page }) => {},\n"
        ");\n"
    )

    def test_pattern_on_the_next_line_is_seen(self):
        assert _iter_step_def_patterns(self._MULTILINE) == [
            ("str", 'I navigate to "https://x.dev"')
        ]

    def test_multiline_pattern_gets_its_slashes_escaped(self):
        defs = [_def(self._MULTILINE)]
        assert _ensure_valid_cuke_expressions(defs) == 1
        assert r"""  'I navigate to "https:\\/\\/x.dev"',""" in defs[0]["content"]
        # the rest of the file is byte-identical
        assert "  async ({ page }) => {}," in defs[0]["content"]
        assert "const { When } = createBdd();" in defs[0]["content"]

    def test_multiline_def_counts_as_coverage(self):
        # Seeing the pattern for escaping but NOT for coverage would stub a step
        # that is already defined → 'Multiple definitions matched'.
        feats = [{"content": 'Feature: t\n  Scenario: s\n    When I navigate to "https://x.dev"\n'}]
        defs = [_def(self._MULTILINE)]
        _ensure_valid_cuke_expressions(defs)
        assert _ensure_step_coverage(feats, defs) == 0

    def test_only_a_bare_opening_line_is_treated_as_multiline(self):
        # Conservative by design: anything else on the opener line is left alone
        # rather than risk a rewrite that corrupts generated TypeScript.
        src = "When( // pattern below\n  'I go to \"https://x.dev\"',\n  async () => {},\n);\n"
        defs = [_def(src)]
        assert _ensure_valid_cuke_expressions(defs) == 0
        assert defs[0]["content"] == src
        assert _iter_step_def_patterns(src) == []

    def test_commented_out_multiline_pattern_is_ignored(self):
        src = "When(\n  // 'I go to \"https://x.dev\"',\n);\n"
        defs = [_def(src)]
        assert _ensure_valid_cuke_expressions(defs) == 0
        assert defs[0]["content"] == src


class TestEnsureStepCoverage:
    """Every Gherkin step must have a matching def before packaging — a missing
    one throws 'undefined step' at record time → blank clip → hard-fail."""

    _FEATURE = (
        "Feature: t\n  Scenario: s\n"
        '    When I click "Save"\n'
        '    And I click the checkbox next to "X" again\n'
    )

    def test_appends_stub_for_uncovered_step(self):
        feats = [{"content": self._FEATURE}]
        defs = [_def("When('I click {string}', async ({page}) => {});")]
        added = _ensure_step_coverage(feats, defs)
        assert added == 1
        stub = defs[-1]
        assert stub["filename"] == "steps_autostub.ts"
        assert "createBdd" in stub["content"]
        # cuke-escaped, json-encoded literal — a valid TS string, no /regex/ break
        assert r'When("I click the checkbox next to \"X\" again"' in stub["content"]

    def test_covered_step_is_not_stubbed(self):
        # The `I click {string}` def covers `I click "Save"` → no double-stub
        # (which would be a bddgen 'Multiple definitions matched' hard failure).
        feats = [{"content": 'Feature: t\n  Scenario: s\n    When I click "Save"\n'}]
        defs = [_def("When('I click {string}', async ({page}) => {});")]
        assert _ensure_step_coverage(feats, defs) == 0

    def test_stub_body_is_non_fatal(self):
        feats = [{"content": "Feature: t\n  Scenario: s\n    When I frobnicate\n"}]
        defs = [_def("When('other', async ({page}) => {});")]
        _ensure_step_coverage(feats, defs)
        body = defs[-1]["content"].split("createBdd()", 1)[1]
        assert "waitForLoadState('networkidle'" in body
        assert "throw" not in body  # must dwell, never throw

    def test_stub_dwell_is_budgeted_not_a_fixed_1200ms(self):
        # A stub takes no action, so its hold time IS its whole contribution to
        # the clip. 1200ms is 18x short of the HERO_DWELL_MS=22000 the compositor
        # expects at clip_anchor='end' (near-empty clip), and N x 1200ms had no
        # ceiling at all — enough stubs outlast the per-test timeout, the test is
        # killed mid-step and the clip records BLACK.
        feats = [{"content": "Feature: t\n  Scenario: s\n    When I frobnicate\n"}]
        defs = [_def("When('other', async ({page}) => {});")]
        _ensure_step_coverage(feats, defs)
        content = defs[-1]["content"]
        assert "waitForTimeout(1200)" not in content
        assert "await stubDwell(page);" in content
        # Both budgets derive from the runner-exported per-test timeout.
        assert "Number(process.env.PW_TEST_TIMEOUT_MS) || 20000" in content
        assert "const STUB_DWELL_MS" in content
        assert "const STUB_DWELL_BUDGET_MS" in content
        # The aggregate ceiling must stay strictly under the test timeout, or
        # this fix reintroduces the black-clip bug it exists to prevent.
        assert "PW_TEST_TIMEOUT_MS - 2000" in content

    def test_raising_the_per_test_timeout_does_not_lengthen_the_dwell(self):
        # THE REGRESSION THIS GUARDS. How long we FILM a static page is a pacing
        # decision; the per-test timeout is a liveness decision. While the dwell
        # was a pure function of the timeout, raising the timeout for an unrelated
        # reason silently changed pacing: w2v_orchestrator#852 took it 45s -> 180s
        # (to stop killing scenarios whose steps were resolving correctly), which
        # took the per-stub dwell 13.5s -> 22.0s and the per-scenario budget
        # 20.2s -> 81.0s. At ~25fps that is 4x more DUPLICATE FRAMES per clip —
        # the exact defect the capture chain is being fixed for, re-created as a
        # side effect of a timeout change.
        #
        # The property is ARITHMETIC, so this evaluates the generated expressions
        # rather than asserting on their text: a string check passes just as
        # happily against a formula that still scales with the timeout. The
        # expressions are evaluated, not reimplemented, so the test cannot drift
        # into agreeing with itself.
        feats = [{"content": "Feature: t\n  Scenario: s\n    When I frobnicate\n"}]
        defs = [_def("When('other', async ({page}) => {});")]
        _ensure_step_coverage(feats, defs)
        content = defs[-1]["content"]

        decls = dict(
            re.findall(
                r"const (STUB_DWELL_MS|STUB_DWELL_BUDGET_MS|PW_TEST_TIMEOUT_MS) = (.+?);",
                content,
                re.S,
            )
        )
        assert set(decls) == {
            "PW_TEST_TIMEOUT_MS",
            "STUB_DWELL_MS",
            "STUB_DWELL_BUDGET_MS",
        }, f"expected all three declarations, got {sorted(decls)}"

        def measure(timeout_ms: int) -> tuple[int, int]:
            """Evaluate the generated JS expressions for one timeout value."""
            env = {"PW_TEST_TIMEOUT_MS": str(timeout_ms)}
            scope: dict[str, Any] = {
                "Math": SimpleNamespace(min=min, max=max, floor=math.floor),
                "Number": lambda v: float(v) if v else 0,
                "process": SimpleNamespace(env=SimpleNamespace(**{})),
            }
            scope["process"].env = _Env(env)
            for name in ("PW_TEST_TIMEOUT_MS", "STUB_DWELL_MS", "STUB_DWELL_BUDGET_MS"):
                # `||` is the only JS-only operator these expressions use.
                scope[name] = eval(  # noqa: S307 - generated source under test
                    decls[name].replace("||", "or"), {"__builtins__": {}}, scope
                )
            return int(scope["STUB_DWELL_MS"]), int(scope["STUB_DWELL_BUDGET_MS"])

        at_45s = measure(45_000)
        at_180s = measure(180_000)
        assert at_45s == at_180s, (
            f"quadrupling the per-test timeout changed pacing: {at_45s} -> {at_180s}. "
            "The dwell must be its own knob, clamped by the timeout, not derived from it."
        )
        # ...and it is still clamped DOWN when the test is genuinely too short to
        # hold it, or the dwell outlasts its test and the clip records BLACK.
        short_dwell, short_budget = measure(20_000)
        assert short_dwell < at_45s[0]
        assert short_budget < at_45s[1]
        assert short_budget <= 20_000 - 2_000

    def test_stub_dwell_charges_the_shared_ordered_ledger(self):
        # ⛔ THE DEFECT THIS GUARDS: two dwell budgets, one clip.
        # The hero ledger in lib/recording_helpers is narrowed by an ordered
        # duration (setScenarioDwellBudgetMs); STUB_DWELL_BUDGET_MS here is
        # derived from PW_TEST_TIMEOUT_MS and was NEVER narrowed. So a scenario
        # could hold its ordered share there AND ~20s more here — which is how a
        # 30s order came back with 43.56s of footage, pinned the beats to that
        # footage, and clamped the narration to the vo_rate floor to fit.
        # When a duration WAS ordered, the stub must charge the SHARED ledger.
        feats = [{"content": "Feature: t\n  Scenario: s\n    When I frobnicate\n"}]
        defs = [_def("When('other', async ({page}) => {});")]
        _ensure_step_coverage(feats, defs)
        content = defs[-1]["content"]
        assert "spendDwell" in content
        assert "if (isDwellBudgetOrdered()) {" in content
        assert "await spendDwell(page, ms);" in content
        # The local timeout clamp must SURVIVE — it is what keeps a dwell inside
        # the test it runs in, and dropping it reintroduces the black-clip bug.
        assert "const remaining = STUB_DWELL_BUDGET_MS - used;" in content
        # With no order the previous, independent behaviour must remain reachable.
        assert "await page.waitForTimeout(ms);" in content

    def test_stub_file_only_imports_bindings_recording_helpers_exports(self):
        # NON-VACUOUS CROSS-FILE CHECK. The generated stub now imports from
        # ../lib/recording_helpers. Emitting a call to a binding that module does
        # not export is the `And is not defined` class (#104) — it breaks the
        # render at runtime, not at generation time, so nothing else here catches
        # it. Resolve every named import against the scaffold's real exports.
        feats = [{"content": "Feature: t\n  Scenario: s\n    When I frobnicate\n"}]
        defs = [_def("When('other', async ({page}) => {});")]
        _ensure_step_coverage(feats, defs)
        content = defs[-1]["content"]

        imported: set[str] = set()
        for spec in re.findall(
            r"import\s*\{([^}]*)\}\s*from\s*'\.\./lib/recording_helpers'", content
        ):
            for name in spec.split(","):
                name = name.strip()
                if name:
                    imported.add(name.split(" as ")[0].strip())
        assert imported, "the stub no longer imports from recording_helpers"

        helpers = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "templates"
            / "scaffold"
            / "lib"
            / "recording_helpers.ts"
        )
        source = helpers.read_text(encoding="utf-8")
        exported = set(
            re.findall(r"export\s+(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", source)
        ) | set(re.findall(r"export\s+const\s+([A-Za-z_$][\w$]*)", source))
        missing = sorted(imported - exported)
        assert not missing, f"stub imports bindings recording_helpers does not export: {missing}"

    def test_stub_dwell_spend_is_tracked_per_scenario(self):
        # One fresh context (= one `page` object) per @clip-<n> scenario, so a
        # WeakMap keyed by page is per-scenario accounting with no hook-ordering
        # assumption and nothing to reset between scenarios.
        feats = [{"content": "Feature: t\n  Scenario: s\n    When I frobnicate\n"}]
        defs = [_def("When('other', async ({page}) => {});")]
        _ensure_step_coverage(feats, defs)
        content = defs[-1]["content"]
        assert "new WeakMap<Page, number>()" in content
        assert "if (remaining <= 0) return;" in content
        # Exhausted budget returns immediately rather than overrunning the test.
        assert "Math.min(STUB_DWELL_MS, remaining)" in content

    def test_stub_file_passes_the_generated_ts_guards(self):
        # The budget helper adds module-scope code to a file that previously had
        # none. It must not trip the guards that exist because generated TS has
        # broken live renders: a module-scope fixture deref, or a call to a
        # binding the module never brings into scope (the `And is not defined`
        # class, #104).
        feats = [{"content": self._FEATURE}]
        defs = [_def("When('I click {string}', async ({page}) => {});")]
        _ensure_step_coverage(feats, defs)
        stub_only = [defs[-1]]
        assert _first_module_scope_violation(stub_only) is None
        assert _undefined_module_bindings(stub_only) == []

    def test_dwell_helper_survives_dedup(self):
        feats = [{"content": self._FEATURE}]
        defs = [_def("When('I click {string}', async ({page}) => {});")]
        _ensure_step_coverage(feats, defs)
        _dedupe_step_defs(defs)
        # The stub bodies call it; dropping the helper would be a ReferenceError
        # at record time — a whole scenario lost, not one step.
        assert "async function stubDwell(page: Page)" in defs[-1]["content"]

    def test_idempotent(self):
        feats = [{"content": self._FEATURE}]
        defs = [_def("When('I click {string}', async ({page}) => {});")]
        assert _ensure_step_coverage(feats, defs) == 1
        # Re-running finds the new stub covers the step → adds nothing.
        assert _ensure_step_coverage(feats, defs) == 0

    def test_stub_survives_dedup_and_duplicate_scan(self):
        feats = [{"content": self._FEATURE}]
        defs = [_def("When('I click {string}', async ({page}) => {});")]
        _ensure_step_coverage(feats, defs)
        assert _duplicate_step_patterns(defs) == []
        _dedupe_step_defs(defs)
        # The stub's When pattern survives the dedup scan (block not dropped).
        assert r'When("I click the checkbox next to \"X\" again"' in defs[-1]["content"]

    def test_unparseable_def_disables_stubbing_safely(self):
        # A def we cannot bound to a guaranteed superset (regex literal) → stub
        # nothing rather than risk a double-match hard failure.
        feats = [{"content": self._FEATURE}]
        defs = [_def("When(/^anything$/, async ({page}) => {});")]
        assert _uncovered_steps(feats, defs) == []
        assert _ensure_step_coverage(feats, defs) == 0

    def test_step_with_slash_produces_valid_stub(self):
        import json

        text = 'I open "/dashboard"'
        feats = [{"content": f"Feature: t\n  Scenario: s\n    When {text}\n"}]
        defs = [_def("When('other', async ({page}) => {});")]
        _ensure_step_coverage(feats, defs)
        content = defs[-1]["content"]
        # The stub is a cucumber-expression STRING literal, not a `/regex/` literal
        # (a `/` in the text would break a regex literal). It round-trips as valid
        # JSON/TS and the `/` is cuke-escaped.
        assert "When(/" not in content
        assert f"When({json.dumps(_cuke_escape(text))}, async" in content


class TestEnsureResilientImport:
    """Bare page.click/fill are rewritten to the resilient helpers (which soft-fail
    instead of aborting the clip) with a guaranteed import — recording stream only."""

    def test_rewrites_bare_click_and_fill(self):
        src = (
            "import { createBdd } from 'playwright-bdd';\n"
            "const { When } = createBdd();\n"
            "When('x', async ({ page }) => {\n"
            "  await page.click('#a');\n"
            "  await page.fill('#b', 'v');\n"
            "});\n"
        )
        defs = [_def(src)]
        assert _ensure_resilient_import(defs) == 1
        c = defs[0]["content"]
        assert "await resilientClick(page, '#a')" in c
        assert "await resilientFill(page, '#b', 'v')" in c
        assert "page.click(" not in c and "page.fill(" not in c
        assert "resilientClick, resilientFill" in c and "recording_helpers" in c

    def test_noop_without_bare_calls(self):
        defs = [_def("When('x', async ({ page }) => { await page.goto('/'); });")]
        before = defs[0]["content"]
        assert _ensure_resilient_import(defs) == 0
        assert defs[0]["content"] == before

    def test_import_injected_once(self):
        src = (
            "import { createBdd } from 'playwright-bdd';\n"
            "When('x', async ({ page }) => { await page.click('#a'); });\n"
            "When('y', async ({ page }) => { await page.click('#b'); });\n"
        )
        defs = [_def(src)]
        _ensure_resilient_import(defs)
        assert defs[0]["content"].count("recording_helpers") == 1

    def test_rewrites_variable_arg_forms(self):
        # The common generated form parameterises the selector as a variable.
        src = (
            "import { createBdd } from 'playwright-bdd';\n"
            "When('I click {string}', async ({ page }, selector) => {\n"
            "  await page.click(selector);\n"
            "});\n"
            "When('I fill {string}', async ({ page }, sel, value) => {\n"
            "  await page.fill(sel, value);\n"
            "});\n"
        )
        defs = [_def(src)]
        assert _ensure_resilient_import(defs) == 1
        c = defs[0]["content"]
        assert "await resilientClick(page, selector)" in c
        assert "await resilientFill(page, sel, value)" in c
        assert "page.click(" not in c and "page.fill(" not in c

    def test_leaves_locator_chains_alone(self):
        # Only bare single-selector forms are rewritten; a locator chain is not.
        src = "When('x', async ({ page }) => { await page.locator('#a').first().click(); });\n"
        defs = [_def(src)]
        assert _ensure_resilient_import(defs) == 0
        assert "page.locator('#a').first().click()" in defs[0]["content"]

    def test_leaves_complex_expression_args_alone(self):
        # A call-expression / arithmetic arg must NOT be rewritten (would corrupt).
        src = (
            "When('x', async ({ page }) => { await page.fill('#a', build(x) + y); });\n"
            "When('y', async ({ page }) => { await page.click(row.locator('.x')); });\n"
        )
        defs = [_def(src)]
        assert _ensure_resilient_import(defs) == 0
        assert "page.fill('#a', build(x) + y)" in defs[0]["content"]
        assert "page.click(row.locator('.x'))" in defs[0]["content"]

    async def test_generate_recording_emits_resilient_and_stub(self, step_gen, mock_llm_client):
        """End-to-end: the recording stream rewrites bare clicks to resilient
        helpers and stubs an uncovered step — a compilable, non-blank suite."""
        mock_llm_client.generate.return_value = (
            "```typescript\n"
            "// filename: steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { When } = createBdd();\n"
            "When('I click {string}', async ({ page }, sel) => { await page.click(sel); });\n"
            "```"
        )
        features = [
            {
                "content": (
                    "Feature: T\n  Scenario: S\n"
                    '    When I click "Save"\n'
                    "    And I do something bespoke\n"
                )
            }
        ]
        defs = await step_gen.generate(features, {"base_url": "https://x"}, recording=True)
        joined = "\n".join(d["content"] for d in defs)
        # uncovered `I do something bespoke` got a non-fatal stub
        assert any(d["filename"] == "steps_autostub.ts" for d in defs)
        assert "I do something bespoke" in joined

    async def test_generate_nonrecording_does_not_stub_uncovered(self, step_gen, mock_llm_client):
        """QA/accessibility/documentation streams (recording=False) must NOT
        stub an uncovered step: those run `missingSteps:'fail-on-gen'`, where an
        uncovered step is an HONEST generation failure the operator must see.
        Silently stubbing it to a dwelling no-op would be a false green."""
        mock_llm_client.generate.return_value = (
            "```typescript\n"
            "// filename: steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { When } = createBdd();\n"
            "When('I click {string}', async ({ page }, sel) => { await page.click(sel); });\n"
            "```"
        )
        features = [
            {
                "content": (
                    "Feature: T\n  Scenario: S\n"
                    '    When I click "Save"\n'
                    "    And I do something bespoke\n"
                )
            }
        ]
        defs = await step_gen.generate(features, {"base_url": "https://x"}, recording=False)
        # No fallback stub file — the uncovered step is left to fail honestly.
        assert not any(d["filename"] == "steps_autostub.ts" for d in defs)


class TestBlockCloseTolerateComment:
    """`_STEP_DEF_CLOSE_RE` must recognize a block close carrying a trailing `// comment`,
    else the scan merges it with the next block and a dedupe on the merged span silently
    deletes the unrelated, valid following step definition."""

    def test_comment_closed_duplicate_does_not_delete_next_block(self) -> None:
        content = (
            "```typescript\n"
            "// filename: t.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, When, Then } = createBdd();\n"
            "When('I add a task', async ({ page }) => {\n"
            "  await page.locator('.new-todo').fill('1');\n"
            "});\n"
            "When('I add a task', async ({ page }) => {\n"  # duplicate of the first
            "  await page.locator('.new-todo').fill('2');\n"
            "});  // add-task step\n"  # closes WITH a trailing comment
            "Then('I see results', async ({ page }) => {\n"  # distinct, WANTED, must survive
            "  await expect(page.locator('.results')).toBeVisible();\n"
            "});\n"
            "```"
        )
        deduped = _dedupe_step_defs(_parse_step_defs(content))
        joined = "\n".join(sd["content"] for sd in deduped)
        # The duplicate is dropped, the pattern appears once...
        assert joined.count("When('I add a task'") == 1
        # ...and the unrelated Then block was NOT swallowed by the comment-closed span.
        assert "Then('I see results'" in joined
        assert "toBeVisible" in joined
        assert _duplicate_step_patterns(deduped) == []


class TestNormalizeParamTypes:
    """`{number}` (and friends) are NOT cucumber-expressions 18.0.1 built-ins, so an
    unrewritten one aborts the whole compile with `Undefined parameter type` → 0 clips.
    `_normalize_param_types` rewrites them to the built-in `{float}` (a numeric superset)."""

    def _def(self, content: str) -> dict[str, Any]:
        return {"filename": "s.steps.ts", "content": content, "step_count": 1}

    def test_rewrites_number_to_float(self) -> None:
        d = self._def('When("I add {number} items", async ({ page }) => {});')
        assert _normalize_param_types([d]) == 1
        assert "{float}" in d["content"]
        assert "{number}" not in d["content"]

    @pytest.mark.parametrize("invented", ["number", "num", "numeric", "integer", "decimal"])
    def test_rewrites_every_invented_numeric_type(self, invented: str) -> None:
        d = self._def(f'Then("count is {{{invented}}}", async ({{ page }}) => {{}});')
        assert _normalize_param_types([d]) == 1
        assert "{float}" in d["content"]
        assert f"{{{invented}}}" not in d["content"]

    def test_leaves_builtin_types_untouched(self) -> None:
        # int/float/word/string are real built-ins — must NOT be rewritten.
        original = 'When("I type {string} {int} {float} {word}", async ({ page }) => {});'
        d = self._def(original)
        assert _normalize_param_types([d]) == 0
        assert d["content"] == original

    def test_multiple_files_counted(self) -> None:
        defs = [
            self._def('When("a {number}", async () => {});'),
            self._def('When("b {int}", async () => {});'),  # untouched
            self._def('When("c {decimal}", async () => {});'),
        ]
        assert _normalize_param_types(defs) == 2

    def test_wired_into_compile_guarantees(self) -> None:
        from src.services.generation.step_def_generator import _apply_compile_guarantees

        defs = [self._def('When("I add {number} todos", async ({ page }) => {});')]
        out = _apply_compile_guarantees(defs, [], recording=True)
        joined = "".join(sd["content"] for sd in out)
        assert "{number}" not in joined
        assert "{float}" in joined


class TestStrictSafeLocators:
    """Playwright locators are STRICT: a `getBy*` that resolves to >1 element THROWS
    (`strict mode violation: … resolved to 2 elements`), which aborts the scenario
    mid-recording — the clip records black and the orchestrator's blank-clip gate
    hard-fails the paid render. That is how the 2026-07-31 `Gestalt Calibration
    Corpus` sweep (run 30610585336) lost 3/3 renders on a bare
    `getByRole('link', { name: 'TodoMVC' })`. The prompt rule is advisory; this pass
    is the compile-time guarantee."""

    @staticmethod
    def _def(content: str) -> dict[str, Any]:
        return {"filename": "todo.steps.ts", "content": content, "step_count": 1}

    def test_bare_get_by_role_gets_first(self) -> None:
        # The exact live failure: two <a> match the accessible name 'TodoMVC'.
        d = self._def(
            "Then('I see the app', async ({ page }) => {\n"
            "  await expect(page.getByRole('link', { name: 'TodoMVC' })).toBeVisible();\n"
            "});\n"
        )
        assert _ensure_strict_safe_locators([d]) == 1
        assert (
            "await expect(page.getByRole('link', { name: 'TodoMVC' }).first()).toBeVisible();"
            in d["content"]
        )

    def test_regex_name_with_nested_parens_is_handled(self) -> None:
        # A regex `name:` argument carries its own parens/braces/slashes — the pass
        # must balance the call, not stop at the first `)`.
        d = self._def("await expect(page.getByRole('heading', { name: /a(b)c/ })).toBeVisible();")
        assert _ensure_strict_safe_locators([d]) == 1
        assert "{ name: /a(b)c/ }).first()).toBeVisible();" in d["content"]

    def test_all_four_factories_covered(self) -> None:
        d = self._def(
            "await page.getByText('Todos').click();\n"
            "await page.getByLabel('Email').fill('a@b.c');\n"
            "await page.getByPlaceholder('What needs doing?').press('Enter');\n"
        )
        assert _ensure_strict_safe_locators([d]) == 1
        c = d["content"]
        assert "page.getByText('Todos').first().click();" in c
        assert "page.getByLabel('Email').first().fill('a@b.c');" in c
        assert "page.getByPlaceholder('What needs doing?').first().press('Enter');" in c

    def test_already_first_is_untouched(self) -> None:
        src = "await expect(page.getByRole('link', { name: 'TodoMVC' }).first()).toBeVisible();"
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 0
        assert d["content"] == src

    def test_nth_is_untouched(self) -> None:
        # `.nth(2)` is a deliberate index — appending `.first()` would silently
        # re-target the assertion at a different element.
        src = "await expect(page.getByRole('listitem').nth(2)).toHaveText('third');"
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 0
        assert d["content"] == src

    def test_last_is_untouched(self) -> None:
        src = "await page.getByRole('row').last().click();"
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 0
        assert d["content"] == src

    def test_filter_chain_gets_first_at_the_chain_end(self) -> None:
        # `.filter()` returns a Locator and may STILL match several rows, so the
        # narrowing belongs AFTER it — appending mid-chain would change what the
        # filter applies to.
        d = self._def("await page.getByRole('row').filter({ hasText: 'Buy milk' }).click();")
        assert _ensure_strict_safe_locators([d]) == 1
        assert (
            "page.getByRole('row').filter({ hasText: 'Buy milk' }).first().click();" in d["content"]
        )

    def test_first_is_inserted_before_an_action_never_after(self) -> None:
        # `.click()` returns Promise<void>; `.click().first()` does not compile, and a
        # non-compiling file fails the WHOLE suite (zero clips) — strictly worse than
        # the strict-mode violation being fixed.
        d = self._def("await page.getByRole('button', { name: 'Save' }).click();")
        assert _ensure_strict_safe_locators([d]) == 1
        assert d["content"] == ("await page.getByRole('button', { name: 'Save' }).first().click();")
        assert ".click().first()" not in d["content"]

    def test_scoping_chain_narrows_at_the_end(self) -> None:
        d = self._def("await expect(page.getByRole('navigation').getByRole('link')).toBeVisible();")
        assert _ensure_strict_safe_locators([d]) == 1
        assert (
            "page.getByRole('navigation').getByRole('link').first()).toBeVisible();" in d["content"]
        )

    def test_multi_element_assertions_are_left_alone(self) -> None:
        # `.first()` would turn toHaveCount(3) into a guaranteed failure — i.e. cause
        # exactly the aborted scenario / black clip this pass exists to prevent.
        src = (
            "await expect(page.getByRole('listitem')).toHaveCount(3);\n"
            "const n = await page.getByText('todo').count();\n"
            "const all = await page.getByRole('row').all();\n"
        )
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 0
        assert d["content"] == src

    def test_string_literal_and_comment_occurrences_are_untouched(self) -> None:
        src = (
            "// Prefer page.getByRole('link', { name: 'Home' }) over a CSS selector.\n"
            "/* also page.getByText('Blocked') in a block comment */\n"
            "const DOC = \"use page.getByLabel('Email') here\";\n"
            "const TPL = `page.getByPlaceholder('x')`;\n"
        )
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 0
        assert d["content"] == src

    def test_code_after_a_comment_on_the_same_line_is_still_rewritten(self) -> None:
        d = self._def("/* note */ await expect(page.getByRole('heading')).toBeVisible(); // gate\n")
        assert _ensure_strict_safe_locators([d]) == 1
        assert "page.getByRole('heading').first()).toBeVisible();" in d["content"]

    def test_multiline_chain_is_left_alone_not_mangled(self) -> None:
        """A chain split across lines is SKIPPED, deliberately.

        Finding the true end of a multi-line chain needs real parsing; a regex that
        guesses will eventually insert `.first()` mid-chain and produce a step file
        that does not compile. A non-compiling package fails the whole demo run,
        which is strictly worse than the strict-mode violation this pass exists to
        prevent — so the conservative case wins. The prompt rule still steers the
        LLM toward single-line, already-disambiguated locators.
        """
        src = "await page\n  .getByRole('row')\n  .filter({ hasText: 'Buy milk' })\n  .click();\n"
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 0
        assert d["content"] == src

    def test_unbalanced_parens_are_skipped_not_mangled(self) -> None:
        # Truncated LLM output: there is no provable end of the expression, so the
        # pass refuses to insert anything.
        src = "await expect(page.getByRole('link', { name: 'x' }"
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 0
        assert d["content"] == src

    def test_is_idempotent(self) -> None:
        src = "await expect(page.getByRole('link', { name: 'TodoMVC' })).toBeVisible();"
        once = [self._def(src)]
        assert _ensure_strict_safe_locators(once) == 1
        twice = [self._def(once[0]["content"])]
        assert _ensure_strict_safe_locators(twice) == 0
        assert twice[0]["content"] == once[0]["content"]

    def test_paren_balance_preserved_across_a_realistic_file(self) -> None:
        # Cheap stand-in for `tsc --noEmit`: a rewrite must never unbalance delimiters.
        src = (
            "// filename: todo.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given, Then } = createBdd();\n"
            "Given('I am on the app', async ({ page }) => {\n"
            "  await expect(page.getByRole('link', { name: 'TodoMVC' })).toBeVisible();\n"
            "  await page.getByPlaceholder('What needs to be done?').fill('milk');\n"
            "});\n"
        )
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 1
        out = d["content"]
        for ch_open, ch_close in (("{", "}"), ("(", ")")):
            assert out.count(ch_open) == out.count(ch_close), out
        assert out.count(".first()") == 2

    def test_prettier_wrapped_to_have_count_is_left_alone(self) -> None:
        """The count guard is STATEMENT-scoped, not line-scoped.

        Prettier wraps a long chain, so `.toHaveCount(2)` routinely lands on the next
        line. A line-scoped guard never saw it, appended `.first()`, and turned a
        passing assertion into a guaranteed failure — scenario aborted, clip black,
        i.e. exactly the outcome this pass exists to prevent.
        """
        src = "await expect(page.getByRole('listitem'))\n  .toHaveCount(2);\n"
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 0
        assert d["content"] == src

    def test_multiline_count_call_across_three_lines_is_left_alone(self) -> None:
        src = (
            "await expect(\n"
            "  page.getByRole('row').filter({ hasText: 'milk' })\n"
            ").toHaveCount(3);\n"
        )
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 0
        assert d["content"] == src

    def test_bound_locator_used_with_to_have_count_is_left_alone(self) -> None:
        """The binding statement's own tail cannot see through the name."""
        src = "const rows = page.getByRole('row');\nawait expect(rows).toHaveCount(3);\n"
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 0
        assert d["content"] == src

    def test_bound_locator_counted_later_in_the_file_is_left_alone(self) -> None:
        src = (
            "const items = page.getByRole('listitem');\n"
            "await items.first().click();\n"
            "const n = await items.count();\n"
        )
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 0
        assert d["content"] == src

    def test_bound_locator_indexed_with_nth_or_last_is_left_alone(self) -> None:
        # `.nth(1)` on the BOUND name indexes into the match set; narrowing the binding
        # to `.first()` re-targets it at a different element (or at nothing).
        src = "const rows = page.getByRole('row');\nawait rows.nth(1).click();\n"
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 0
        assert d["content"] == src

    def test_bound_locator_passed_to_an_opaque_helper_is_left_alone(self) -> None:
        # A helper may consume it as a multi-element set; this pass cannot read inside.
        src = "const rows = page.getByRole('row');\nawait assertRows(rows);\n"
        d = self._def(src)
        assert _ensure_strict_safe_locators([d]) == 0
        assert d["content"] == src

    def test_bound_locator_with_only_safe_uses_is_still_narrowed(self) -> None:
        # Refusing every binding would be safe but useless — a provably
        # count-insensitive binding is the common case and still gets narrowed.
        d = self._def(
            "const save = page.getByRole('button', { name: 'Save' });\n"
            "await expect(save).toBeVisible();\n"
            "await save.click();\n"
        )
        assert _ensure_strict_safe_locators([d]) == 1
        assert "const save = page.getByRole('button', { name: 'Save' }).first();" in d["content"]

    def test_statement_scan_stops_at_the_semicolon(self) -> None:
        # A count assertion in the NEXT statement must not suppress this rewrite.
        d = self._def(
            "await page.getByRole('button', { name: 'Add' }).click();\n"
            "await expect(page.getByRole('listitem')).toHaveCount(2);\n"
        )
        assert _ensure_strict_safe_locators([d]) == 1
        c = d["content"]
        assert "{ name: 'Add' }).first().click();" in c
        assert "await expect(page.getByRole('listitem')).toHaveCount(2);" in c

    def test_regex_paren_does_not_close_the_call_early(self) -> None:
        # Without regex-literal masking the `)` inside `/\\)/` closed the call, so
        # `.first()` landed INSIDE the regex — it compiled, but matched nothing.
        d = self._def("await page.getByText(/\\)/).click();\n")
        assert _ensure_strict_safe_locators([d]) == 1
        assert d["content"] == "await page.getByText(/\\)/).first().click();\n"
        assert ".first()/)" not in d["content"]

    def test_apostrophe_in_a_regex_does_not_desync_the_mask(self) -> None:
        # `/won't/` used to open a phantom string literal, masking the REST of the file
        # as non-code so every later locator was silently skipped with no log.
        d = self._def(
            "await expect(page.getByText(/won't/)).toBeVisible();\n"
            "await page.getByRole('button', { name: 'Save' }).click();\n"
        )
        assert _ensure_strict_safe_locators([d]) == 1
        c = d["content"]
        assert "page.getByText(/won't/).first()).toBeVisible();" in c
        assert "{ name: 'Save' }).first().click();" in c

    def test_division_is_not_mistaken_for_a_regex(self) -> None:
        d = self._def("const half = viewport.width / 2;\nawait page.getByText('Todos').click();\n")
        assert _ensure_strict_safe_locators([d]) == 1
        assert "page.getByText('Todos').first().click();" in d["content"]

    def test_unterminated_string_logs_a_blind_pass_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A mask that ends mid-string means everything after it went unchecked. Say so
        # — a silent skip is indistinguishable from "nothing to do".
        d = self._def(
            "await expect(page.getByRole('heading')).toBeVisible();\nconst broken = 'oops;\n"
        )
        with caplog.at_level("WARNING"):
            assert _ensure_strict_safe_locators([d]) == 1
        assert "unterminated string or comment" in caplog.text

    def test_clean_file_logs_no_blind_pass_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        d = self._def("await expect(page.getByRole('heading')).toBeVisible();\n")
        with caplog.at_level("WARNING"):
            assert _ensure_strict_safe_locators([d]) == 1
        assert "unterminated string or comment" not in caplog.text

    def test_wired_into_recording_compile_guarantees(self) -> None:
        from src.services.generation.step_def_generator import _apply_compile_guarantees

        defs = [
            self._def(
                "// filename: todo.steps.ts\n"
                "import { createBdd } from 'playwright-bdd';\n"
                "const { Then } = createBdd();\n"
                "Then('I see it', async ({ page }) => {\n"
                "  await expect(page.getByRole('link', { name: 'TodoMVC' })).toBeVisible();\n"
                "});\n"
            )
        ]
        out = _apply_compile_guarantees(defs, [], recording=True)
        joined = "".join(sd["content"] for sd in out)
        assert "{ name: 'TodoMVC' }).first()).toBeVisible();" in joined

    def test_non_recording_stream_is_not_narrowed(self) -> None:
        # In a QA/accessibility suite a strict-mode violation is an HONEST failure the
        # operator must see; silently taking the first match would hide a real
        # ambiguity defect behind a false green (same rule as the step-coverage stubs).
        from src.services.generation.step_def_generator import _apply_compile_guarantees

        src = (
            "// filename: todo.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Then } = createBdd();\n"
            "Then('I see it', async ({ page }) => {\n"
            "  await expect(page.getByRole('link', { name: 'TodoMVC' })).toBeVisible();\n"
            "});\n"
        )
        out = _apply_compile_guarantees([self._def(src)], [], recording=False)
        joined = "".join(sd["content"] for sd in out)
        assert ".first()" not in joined


class TestSurvivableTextAssertions:
    """`toHaveText` is an EXACT match, so a generated
    `expect(page.locator('.todo-count').first()).toHaveText("2 items")` can never pass
    against TodoMVC's counter, which always renders "2 items left". The scenario aborted
    mid-recording and the demo render died as a poison job. For the RECORDING stream the
    text an app happens to append is not the thing under test, so this pass rewrites
    brittle exact-text assertions into containment (`toContainText`)."""

    @staticmethod
    def _def(content: str) -> dict[str, Any]:
        return {"filename": "todo.steps.ts", "content": content, "step_count": 1}

    def test_todomvc_counter_exact_text_becomes_containment(self) -> None:
        # The exact live failure: the counter reads "2 items left", never "2 items".
        d = self._def(
            "Then('I see two todos', async ({ page }) => {\n"
            "  await expect(page.locator('.todo-count').first()).toHaveText(\"2 items\");\n"
            "});\n"
        )
        assert _ensure_survivable_text_assertions([d]) == 1
        assert (
            "await expect(page.locator('.todo-count').first()).toContainText(\"2 items\");"
            in d["content"]
        )
        assert "toHaveText" not in d["content"]

    def test_single_quoted_argument_is_rewritten(self) -> None:
        d = self._def("await expect(page.locator('.todo-count').first()).toHaveText('2 items');\n")
        assert _ensure_survivable_text_assertions([d]) == 1
        assert (
            d["content"]
            == "await expect(page.locator('.todo-count').first()).toContainText('2 items');\n"
        )

    def test_template_literal_argument_is_rewritten(self) -> None:
        d = self._def("await expect(page.locator('.todo-count').first()).toHaveText(`2 items`);\n")
        assert _ensure_survivable_text_assertions([d]) == 1
        assert (
            d["content"]
            == "await expect(page.locator('.todo-count').first()).toContainText(`2 items`);\n"
        )

    def test_regex_argument_is_untouched(self) -> None:
        # A regex is already a partial match — the author chose the tolerance.
        src = "await expect(page.locator('.todo-count').first()).toHaveText(/2 items/);\n"
        d = self._def(src)
        assert _ensure_survivable_text_assertions([d]) == 0
        assert d["content"] == src

    def test_array_argument_is_untouched(self) -> None:
        # The array form asserts a whole match SET element-by-element; `toContainText`
        # with an array means something different, so rewriting changes the assertion.
        src = "await expect(page.getByRole('listitem')).toHaveText(['a', 'b']);\n"
        d = self._def(src)
        assert _ensure_survivable_text_assertions([d]) == 0
        assert d["content"] == src

    def test_negated_assertion_is_untouched(self) -> None:
        # `not.toHaveText('2 items')` PASSES on "2 items left" (the text is not exactly
        # that); `not.toContainText('2 items')` FAILS on it (the text does contain it).
        # Rewriting a negation therefore manufactures the very failure this pass
        # exists to prevent.
        src = "await expect(page.locator('.todo-count').first()).not.toHaveText('2 items');\n"
        d = self._def(src)
        assert _ensure_survivable_text_assertions([d]) == 0
        assert d["content"] == src

    def test_comment_and_string_literal_occurrences_are_untouched(self) -> None:
        src = (
            "// Prefer toContainText over toHaveText('2 items') for live counters.\n"
            "const DOC = \"use toHaveText('2 items') only for static labels\";\n"
        )
        d = self._def(src)
        assert _ensure_survivable_text_assertions([d]) == 0
        assert d["content"] == src
        assert _sub_survivable_text_assertions(src) == (src, 0, True)

    def test_file_without_the_matcher_is_untouched(self) -> None:
        src = "await expect(page.getByRole('heading')).toBeVisible();\n"
        d = self._def(src)
        assert _ensure_survivable_text_assertions([d]) == 0
        assert d["content"] == src

    # A pass that is never CALLED is indistinguishable from a pass that does not
    # exist: unwiring `_ensure_survivable_text_assertions` from
    # `_apply_recording_rewrites` left every other test in this file green. These two
    # assert the WIRING and the recording-only gating, not the substitution.
    _COUNTER_STEPS = (
        "```typescript\n"
        "// filename: todo.steps.ts\n"
        "import { createBdd } from 'playwright-bdd';\n"
        "import { expect } from '@playwright/test';\n"
        "const { Given, When, Then } = createBdd();\n"
        "Given('I am on the app', async ({ page }) => {\n"
        "  await page.goto('/');\n"
        "});\n"
        "Then('I see two todos', async ({ page }) => {\n"
        "  await expect(page.locator('.todo-count').first()).toHaveText('2 items');\n"
        "});\n"
        "```"
    )

    @staticmethod
    def _steps_content(out: list[dict[str, Any]]) -> str:
        return next(sd["content"] for sd in out if sd["filename"] == "todo.steps.ts")

    def test_recording_pipeline_relaxes_the_exact_assertion(self) -> None:
        from src.services.generation.step_def_generator import _apply_compile_guarantees

        defs = _parse_step_defs(self._COUNTER_STEPS)
        content = self._steps_content(_apply_compile_guarantees(defs, [], recording=True))
        assert "toContainText('2 items')" in content
        assert "toHaveText" not in content

    def test_non_recording_stream_keeps_exact_equality(self) -> None:
        # QA/accessibility/documentation streams are honest test suites: an exact-text
        # mismatch there is a real content defect the operator must see, and relaxing
        # it would convert that into a false green.
        from src.services.generation.step_def_generator import _apply_compile_guarantees

        defs = _parse_step_defs(self._COUNTER_STEPS)
        content = self._steps_content(_apply_compile_guarantees(defs, [], recording=False))
        assert "toHaveText('2 items')" in content
        assert "toContainText" not in content


class TestStrictModePromptRule:
    """The deterministic pass is the guarantee, but the prompt must still ASK for the
    right locator — a repaired `.first()` on a substring-matching name is a worse clip
    than an `{ exact: true }` the model wrote itself."""

    async def test_prompt_carries_strict_mode_rule(
        self, step_gen, mock_llm_client, sample_inspection_result
    ) -> None:
        mock_llm_client.generate.return_value = (
            "```typescript\n// filename: steps.ts\nGiven('x', async ({ page }) => {});\n```"
        )
        features = [{"content": "  Scenario: Home\n    Given I am on the page\n"}]

        await step_gen.generate(features, sample_inspection_result, platform="web")

        call_args = mock_llm_client.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "STRICT MODE" in prompt
        assert "more than one element" in prompt
        assert "`.first()`" in prompt
        assert "exact: true" in prompt
        # The content-visibility example must itself be strict-safe — it was the
        # template the model copied into the live failure.
        assert "page.getByRole('heading', { name: /.../ }).first()" in prompt
        assert (
            "await expect(page.getByRole('heading', { name: /.../ })).toBeVisible()" not in prompt
        )


class TestEnsureNavigationPreconditions:
    """MEASURED ON PROD: two demo_video runs recorded ZERO interaction — every clip a
    static landing page, `focus_track`/`cursor_track` null, 112/112 demo masters
    static (median duplicate_frame_ratio 0.905). Root cause was ONE step:

        Given('I am on the TodoMVC landing page', async ({ page }) => {
          await expect(page.getByRole('link', { name: 'TodoMVC', exact: true })
            .first()).toBeVisible();
        });

    TodoMVC has no such link (its heading is `todos`), so step 1 THREW and the whole
    scenario aborted before a single When ever ran. A precondition can only honestly
    assert that NAVIGATION SUCCEEDED; the decorative check must be non-fatal."""

    @staticmethod
    def _def(content: str) -> dict[str, Any]:
        return {"filename": "todo.steps.ts", "content": content, "step_count": 1}

    _LIVE_LANDING_GIVEN = (
        "import { createBdd } from 'playwright-bdd';\n"
        "const { Given } = createBdd();\n"
        "Given('I am on the TodoMVC landing page', async ({ page }) => {\n"
        "  await page.goto(absoluteNavUrl('/'));\n"
        "  await expect(page.getByRole('link', { name: 'TodoMVC', exact: true })"
        ".first()).toBeVisible();\n"
        "});\n"
    )

    def test_live_landing_given_is_softened_and_asserts_navigation(self) -> None:
        d = self._def(self._LIVE_LANDING_GIVEN)
        assert _ensure_navigation_preconditions([d]) == 1
        out = d["content"]
        # The invented element check can no longer abort the scenario …
        assert "softExpectVisible(page, page.getByRole('link'" in out
        assert "toBeVisible" not in out
        # … and the step still asserts the one thing it can honestly assert.
        assert "await assertNavigated(page);" in out
        assert (
            "import { assertNavigated, softExpectVisible } from '../lib/recording_helpers';" in out
        )

    def test_import_is_injected_once(self) -> None:
        d = self._def(self._LIVE_LANDING_GIVEN)
        _ensure_navigation_preconditions([d])
        assert d["content"].count("assertNavigated, softExpectVisible") == 1

    def test_is_idempotent(self) -> None:
        d = self._def(self._LIVE_LANDING_GIVEN)
        assert _ensure_navigation_preconditions([d]) == 1
        once = d["content"]
        assert _ensure_navigation_preconditions([d]) == 0
        assert d["content"] == once

    def test_then_step_assertion_is_left_hard(self) -> None:
        """The scenario's REAL claim must still fail loudly — softening a `Then`
        would turn a genuine product bug into a green run."""
        src = (
            "Then('I see the todo', async ({ page }) => {\n"
            "  await expect(page.getByText('Buy milk').first()).toBeVisible();\n"
            "});\n"
        )
        d = self._def(src)
        assert _ensure_navigation_preconditions([d]) == 0
        assert d["content"] == src

    def test_non_navigation_given_is_left_alone(self) -> None:
        src = (
            "Given('I have one completed todo', async ({ page }) => {\n"
            "  await expect(page.getByText('Buy milk').first()).toBeVisible();\n"
            "});\n"
        )
        d = self._def(src)
        assert _ensure_navigation_preconditions([d]) == 0
        assert d["content"] == src

    def test_negative_assertion_is_left_hard(self) -> None:
        """`.not.toBeVisible()` is a real claim about absence — softening it would
        hide a defect, not a brittle selector."""
        src = (
            "Given('I am on the app', async ({ page }) => {\n"
            "  await page.goto(absoluteNavUrl('/'));\n"
            "  await expect(page.getByText('Error').first()).not.toBeVisible();\n"
            "});\n"
        )
        d = self._def(src)
        assert _ensure_navigation_preconditions([d]) == 0
        assert d["content"] == src

    def test_to_have_url_is_left_hard(self) -> None:
        """A URL assertion IS a navigation assertion — never weaken it."""
        src = (
            "Given('I visit the pricing page', async ({ page }) => {\n"
            "  await expect(page).toHaveURL(/pricing/);\n"
            "});\n"
        )
        d = self._def(src)
        assert _ensure_navigation_preconditions([d]) == 0
        assert d["content"] == src

    def test_given_without_page_fixture_is_skipped(self) -> None:
        """Injecting `assertNavigated(page)` where `page` is not destructured would
        throw `ReferenceError: page is not defined`."""
        src = "Given('I am on the app', async () => {\n  await expect(banner).toBeVisible();\n});\n"
        d = self._def(src)
        assert _ensure_navigation_preconditions([d]) == 0
        assert d["content"] == src

    def test_navigating_given_qualifies_even_with_an_odd_sentence(self) -> None:
        src = (
            "Given('the dashboard is ready', async ({ page }) => {\n"
            "  await page.goto(absoluteNavUrl('/dash'));\n"
            "  await expect(page.getByRole('banner').first()).toBeVisible();\n"
            "});\n"
        )
        d = self._def(src)
        assert _ensure_navigation_preconditions([d]) == 1
        assert "softExpectVisible(page, page.getByRole('banner').first())" in d["content"]

    def test_string_and_comment_occurrences_are_untouched(self) -> None:
        src = (
            "Given('I am on the app', async ({ page }) => {\n"
            "  await page.goto(absoluteNavUrl('/'));\n"
            "  // await expect(page.getByRole('link')).toBeVisible();\n"
            "  const note = 'await expect(x).toBeVisible()';\n"
            "  await expect(page.getByRole('main').first()).toBeVisible();\n"
            "});\n"
        )
        d = self._def(src)
        assert _ensure_navigation_preconditions([d]) == 1
        out = d["content"]
        assert "// await expect(page.getByRole('link')).toBeVisible();" in out
        assert "const note = 'await expect(x).toBeVisible()';" in out
        assert "softExpectVisible(page, page.getByRole('main').first())" in out

    def test_paren_balance_is_preserved(self) -> None:
        d = self._def(self._LIVE_LANDING_GIVEN)
        _ensure_navigation_preconditions([d])
        out = d["content"]
        assert out.count("(") == out.count(")")
        assert out.count("{") == out.count("}")

    def test_wired_into_recording_compile_guarantees(self) -> None:
        defs = [self._def("// filename: todo.steps.ts\n" + self._LIVE_LANDING_GIVEN)]
        out = _apply_compile_guarantees(defs, [], recording=True)
        joined = "".join(sd["content"] for sd in out)
        assert "softExpectVisible(page," in joined
        assert "await assertNavigated(page);" in joined

    def test_non_recording_stream_is_not_softened(self) -> None:
        """In a QA/accessibility suite a failing landing assertion is an HONEST
        failure the operator must see (same rule as the strict-mode pass)."""
        defs = [self._def("// filename: todo.steps.ts\n" + self._LIVE_LANDING_GIVEN)]
        out = _apply_compile_guarantees(defs, [], recording=False)
        joined = "".join(sd["content"] for sd in out)
        assert "softExpectVisible" not in joined
        assert "assertNavigated" not in joined


class TestGroundedRoleNames:
    """Measured on orchestrator E2E run 32287791747 against demo.playwright.dev/todomvc:

        soft-failed selector(s): {"action":"expect-visible",
          "selector":"getByRole('heading', { name: /TodoMVC/ }).first()",
          "error":"TimeoutError: locator.waitFor: Timeout 2500ms exceeded"}

    BOTH scenarios were demoted over it. The page's only heading is `todos`; the model
    took the accessible name from the SITE name instead of the inventory, which had
    `todos` in it all along. #119's prompt rule did not stop the invention and #119's
    `softExpectVisible` only made it non-fatal — so drop it deterministically, the
    same policy as #121: an assertion that cannot pass is worse than no assertion."""

    # What the inventory actually carries: `headings` is a list of heading TEXT (role
    # is implicitly `heading`), and each interactive element carries its accessible
    # name in `label`/`text`/`name` with its ROLE recoverable only from the
    # `getByRole("<role>", …)` selector `html_structure._selector_for` emitted.
    _TODOMVC_INVENTORY: ClassVar[dict[str, Any]] = {
        "pages": [
            {
                "url": "https://demo.playwright.dev/todomvc",
                "headings": ["todos"],
                "interactive_elements": [
                    {
                        "tag": "a",
                        "label": "All",
                        "text": "All",
                        "name": "",
                        "selector": 'getByRole("link", { name: "All", exact: true })',
                    }
                ],
            }
        ]
    }

    @staticmethod
    def _def(content: str) -> dict[str, Any]:
        return {"filename": "todo.steps.ts", "content": content, "step_count": 1}

    @staticmethod
    def _given(assertion: str) -> str:
        return (
            "Given('I am on the TodoMVC landing page', async ({ page }) => {\n"
            "  await page.goto(absoluteNavUrl('/'));\n"
            f"  {assertion}\n"
            "  await assertNavigated(page);\n"
            "});\n"
        )

    def test_inventory_carries_role_and_accessible_name(self) -> None:
        """Ground truth for the whole pass: headings -> role `heading`, and an
        element's role comes out of its `getByRole(...)` selector."""
        roles = _inventory_role_names(self._TODOMVC_INVENTORY)
        assert roles == {"heading": {"todos"}, "link": {"all"}}

    def test_the_live_invented_heading_name_is_dropped(self) -> None:
        src = self._given(
            "await softExpectVisible(page, page.getByRole('heading', { name: /TodoMVC/ }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._TODOMVC_INVENTORY) == 1
        out = d["content"]
        assert "TodoMVC" not in out.replace("I am on the TodoMVC landing page", "")
        assert "getByRole" not in out
        # The step keeps the one thing it can honestly assert.
        assert "await assertNavigated(page);" in out
        assert "await page.goto(absoluteNavUrl('/'));" in out

    def test_a_name_present_in_the_inventory_survives_byte_identical(self) -> None:
        src = self._given(
            "await softExpectVisible(page, page.getByRole('heading', { name: 'todos' }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._TODOMVC_INVENTORY) == 0
        assert d["content"] == src

    def test_a_regex_matching_an_inventory_name_survives(self) -> None:
        src = self._given(
            "await softExpectVisible(page, page.getByRole('heading', { name: /todo/i }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._TODOMVC_INVENTORY) == 0
        assert d["content"] == src

    def test_a_regex_that_evaluates_and_matches_nothing_is_dropped(self) -> None:
        """The drop still stands where the pass really DID look: an evaluable pattern,
        checked against a bucket that is exhaustive."""
        src = self._given(
            "await softExpectVisible(page, page.getByRole('heading', { name: /TodoMVC/ }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._TODOMVC_INVENTORY) == 1
        assert "getByRole" not in d["content"]

    def test_empty_or_absent_inventory_changes_nothing(self) -> None:
        """A missing snapshot is not evidence of absence — letting it delete
        assertions would make a failed KAG scrape silently strip the suite."""
        src = self._given(
            "await softExpectVisible(page, page.getByRole('heading', { name: /TodoMVC/ }).first());"
        )
        for inventory in (None, {}, {"pages": []}, {"pages": [{"headings": []}]}):
            d = self._def(src)
            assert _ensure_grounded_role_names([d], inventory) == 0
            assert d["content"] == src

    def test_a_role_the_inventory_never_captures_is_left_alone(self) -> None:
        """`extract_structure` surfaces headings + interactive tags only, so it says
        nothing about `img`; zero names for a role means NOT AUTHORITATIVE, not absent."""
        src = self._given(
            "await softExpectVisible(page, page.getByRole('img', { name: 'Logo' }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._TODOMVC_INVENTORY) == 0
        assert d["content"] == src

    def test_non_navigation_given_is_untouched(self) -> None:
        src = (
            "Given('I have one completed todo', async ({ page }) => {\n"
            "  await expect(page.getByRole('heading', { name: 'TodoMVC' })).toBeVisible();\n"
            "});\n"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._TODOMVC_INVENTORY) == 0
        assert d["content"] == src

    def test_then_step_is_untouched(self) -> None:
        """A `Then` names post-interaction elements the PRE-interaction snapshot
        cannot know about — judging those would delete correct assertions."""
        src = (
            "Then('I see the header', async ({ page }) => {\n"
            "  await expect(page.getByRole('heading', { name: 'TodoMVC' })).toBeVisible();\n"
            "});\n"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._TODOMVC_INVENTORY) == 0
        assert d["content"] == src

    def test_an_action_statement_is_never_deleted(self) -> None:
        """Losing a click would blank the footage the recording stream exists for."""
        src = self._given("await page.getByRole('link', { name: 'Nonexistent' }).first().click();")
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._TODOMVC_INVENTORY) == 0
        assert d["content"] == src

    def test_a_non_literal_name_is_not_judged(self) -> None:
        src = self._given(
            "await softExpectVisible(page, page.getByRole('heading', { name: title }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._TODOMVC_INVENTORY) == 0
        assert d["content"] == src

    def test_string_and_comment_occurrences_are_untouched(self) -> None:
        src = self._given(
            "// await expect(page.getByRole('heading', { name: 'TodoMVC' })).toBeVisible();"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._TODOMVC_INVENTORY) == 0
        assert d["content"] == src

    def test_is_idempotent(self) -> None:
        src = self._given(
            "await softExpectVisible(page, page.getByRole('heading', { name: /TodoMVC/ }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._TODOMVC_INVENTORY) == 1
        once = d["content"]
        assert _ensure_grounded_role_names([d], self._TODOMVC_INVENTORY) == 0
        assert d["content"] == once

    def test_paren_and_brace_balance_is_preserved(self) -> None:
        src = self._given(
            "await softExpectVisible(page, page.getByRole('heading', { name: /TodoMVC/ }).first());"
        )
        d = self._def(src)
        _ensure_grounded_role_names([d], self._TODOMVC_INVENTORY)
        out = d["content"]
        assert out.count("(") == out.count(")")
        assert out.count("{") == out.count("}")

    def test_wired_into_recording_compile_guarantees(self) -> None:
        defs = [
            self._def(
                "// filename: todo.steps.ts\n"
                "import { createBdd } from 'playwright-bdd';\n"
                "const { Given } = createBdd();\n"
                "Given('I am on the TodoMVC landing page', async ({ page }) => {\n"
                "  await page.goto(absoluteNavUrl('/'));\n"
                "  await expect(page.getByRole('heading', { name: /TodoMVC/ })"
                ".first()).toBeVisible();\n"
                "});\n"
            )
        ]
        out = _apply_compile_guarantees(
            defs, [], recording=True, inspection=self._TODOMVC_INVENTORY
        )
        joined = "".join(sd["content"] for sd in out)
        assert "getByRole" not in joined
        assert "await assertNavigated(page);" in joined

    def test_not_wired_into_non_recording_streams(self) -> None:
        """In a QA suite a wrong landing assertion is an HONEST failure to surface."""
        src = "// filename: todo.steps.ts\n" + self._given(
            "await expect(page.getByRole('heading', { name: 'TodoMVC' })).toBeVisible();"
        )
        defs = [self._def(src)]
        out = _apply_compile_guarantees(
            defs, [], recording=False, inspection=self._TODOMVC_INVENTORY
        )
        joined = "".join(sd["content"] for sd in out)
        assert "page.getByRole('heading', { name: 'TodoMVC' })" in joined
        assert "toBeVisible" in joined


class TestGroundedRoleNamesIsNotOverEager:
    """Five ways the grounding pass deleted assertions it had NO evidence against —
    all five shipped live in #125. Absence from the inventory is only evidence when
    the inventory is exhaustive AND the parser is faithful; where it is neither, the
    pass must KEEP the assertion. And it must never leave a step asserting nothing."""

    _INVENTORY: ClassVar[dict[str, Any]] = {
        "pages": [
            {
                "url": "https://demo.playwright.dev/todomvc",
                "headings": ["todos"],
                "interactive_elements": [
                    {
                        "tag": "a",
                        "label": "All",
                        "text": "All",
                        "name": "",
                        "selector": 'getByRole("link", { name: "All", exact: true })',
                    }
                ],
            }
        ]
    }

    @staticmethod
    def _def(content: str) -> dict[str, Any]:
        return {"filename": "todo.steps.ts", "content": content, "step_count": 1}

    @staticmethod
    def _file(*body: str) -> str:
        return (
            "// filename: todo.steps.ts\n"
            "import { createBdd } from 'playwright-bdd';\n"
            "const { Given } = createBdd();\n"
            "Given('I am on the TodoMVC landing page', async ({ page }) => {\n"
            "  await page.goto(absoluteNavUrl('/'));\n"
            + "".join(f"  {line}\n" for line in body)
            + "});\n"
        )

    # ── B1 ──────────────────────────────────────────────────────────────────
    def test_a_nav_given_never_ends_up_asserting_nothing(self) -> None:
        """`_ensure_navigation_preconditions` only appends `assertNavigated` to steps
        whose `.toBeVisible()` it SOFTENED. A nav `Given` whose sole assertion is
        `toHaveText` therefore has none — and this pass will happily delete that one
        assertion, leaving `Given` = bare `page.goto(...)`: a step that asserts
        nothing and can never fail."""
        src = self._file(
            "await expect(page.getByRole('heading', { name: 'TodoMVC' })).toHaveText('TodoMVC');"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._INVENTORY) == 1
        out = d["content"]
        assert "getByRole" not in out
        assert "await assertNavigated(page);" in out
        assert "recording_helpers" in out
        assert out.count("(") == out.count(")")
        assert out.count("{") == out.count("}")

    def test_the_vacuous_guard_also_covers_to_be_attached_and_to_have_count(self) -> None:
        for matcher in (".toBeAttached();", ".toHaveCount(1);"):
            src = self._file(
                f"await expect(page.getByRole('heading', {{ name: 'TodoMVC' }})){matcher}"
            )
            d = self._def(src)
            assert _ensure_grounded_role_names([d], self._INVENTORY) == 1
            assert "await assertNavigated(page);" in d["content"]

    def test_a_step_that_already_asserts_navigation_is_not_double_backfilled(self) -> None:
        src = self._file(
            "await expect(page.getByRole('heading', { name: 'TodoMVC' })).toHaveText('x');",
            "await assertNavigated(page);",
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._INVENTORY) == 1
        assert d["content"].count("assertNavigated(page)") == 1

    # ── B2 ──────────────────────────────────────────────────────────────────
    def test_a_truncated_element_list_is_not_evidence_of_absence(self) -> None:
        """`html_structure` caps interactive elements at 40 and reports the overflow.
        Element #41's `getByRole('link', {name})` would PASS on the page; deleting it
        because it is not in the first 40 is deleting a correct assertion."""
        inventory = {
            "pages": [
                {
                    **self._INVENTORY["pages"][0],
                    "elements_truncated": 7,
                }
            ]
        }
        src = self._file(
            "await softExpectVisible(page, page.getByRole('link', { name: 'Contact' }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], inventory) == 0
        assert d["content"] == src

    def test_a_truncated_heading_list_is_not_evidence_of_absence(self) -> None:
        inventory = {"pages": [{**self._INVENTORY["pages"][0], "headings_truncated": 3}]}
        src = self._file(
            "await softExpectVisible(page, page.getByRole('heading', { name: 'TodoMVC' }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], inventory) == 0
        assert d["content"] == src

    def test_truncation_poisons_only_the_affected_bucket(self) -> None:
        """A capped ELEMENT list says nothing about headings — the heading bucket
        stays authoritative, so the live TodoMVC miss is still dropped."""
        inventory = {"pages": [{**self._INVENTORY["pages"][0], "elements_truncated": 7}]}
        assert _inventory_role_names(inventory) == {"heading": {"todos"}}

    # ── B3 ──────────────────────────────────────────────────────────────────
    def test_a_name_truncated_at_the_label_cap_is_not_evidence_of_absence(self) -> None:
        """`finalize()` truncates accessible names at `LABEL_LENGTH_CAP`, so the
        recorded name is a PREFIX. Playwright's substring test (`wanted in name`)
        then answers ABSENT for a name the page really has."""
        long_label = "Read the full quarterly report " + "x" * LABEL_LENGTH_CAP
        inventory = {
            "pages": [
                {
                    "headings": ["todos"],
                    "interactive_elements": [
                        {
                            "tag": "a",
                            "label": long_label[:LABEL_LENGTH_CAP],
                            "text": "",
                            "name": "",
                            "selector": 'getByRole("link", { name: "r", exact: true })',
                        }
                    ],
                }
            ]
        }
        src = self._file(
            "await softExpectVisible(page, page.getByRole('link', { name: 'Contact' }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], inventory) == 0
        assert d["content"] == src

    # ── B4 ──────────────────────────────────────────────────────────────────
    def test_a_curly_apostrophe_is_the_same_name_as_an_ascii_one(self) -> None:
        """A CMS emits `What\u2019s new`; the spec writes `What's new`. That is an
        encoding difference, not a different element."""
        inventory = {
            "pages": [{"headings": ["What\u2019s new \u2014 Q3"], "interactive_elements": []}]
        }
        src = self._file(
            "await softExpectVisible(page, page.getByRole('heading', "
            '{ name: "What\'s new - Q3" }).first());'
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], inventory) == 0
        assert d["content"] == src

    def test_the_typographic_fold_does_not_strip_accents(self) -> None:
        """Conservative on purpose: `Resume` and `R\u00e9sum\u00e9` are DIFFERENT names."""
        inventory = {"pages": [{"headings": ["R\u00e9sum\u00e9"], "interactive_elements": []}]}
        src = self._file(
            "await softExpectVisible(page, page.getByRole('heading', { name: 'Resume' }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], inventory) == 1

    # ── B5 ──────────────────────────────────────────────────────────────────
    def test_a_refused_statement_does_not_mask_later_ungrounded_asserts(self) -> None:
        """`_drop_assert_statement_at` refuses a `const loc = page.getByRole(...)`
        binding (correctly — it is not an assertion). RETURNING on that refusal made
        one such line suppress every later ungrounded assertion in the file, silently
        and with no log line. Skip it and keep going."""
        src = self._file(
            "const loc = page.getByRole('heading', { name: 'TodoMVC' });",
            "await softExpectVisible(page, page.getByRole('link', "
            "{ name: 'Nonexistent' }).first());",
            "await assertNavigated(page);",
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._INVENTORY) == 1
        out = d["content"]
        # The refused binding survives untouched...
        assert "const loc = page.getByRole('heading', { name: 'TodoMVC' });" in out
        # ...and the ungrounded assertion after it is gone.
        assert "Nonexistent" not in out
        assert "await assertNavigated(page);" in out

    def test_grounding_still_terminates_when_every_candidate_is_refused(self) -> None:
        src = self._file(
            "const a = page.getByRole('heading', { name: 'TodoMVC' });",
            "const b = page.getByRole('link', { name: 'Nonexistent' });",
            "await assertNavigated(page);",
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._INVENTORY) == 0
        assert d["content"] == src


class TestGroundedRoleNamesNeverDropsOnUnknown:
    """#125 treated "this pass could not evaluate the name" as "the page does not have
    it". Those are different statements. A pattern we cannot parse says nothing about
    the page — only about this pass — so it must be UNKNOWN and KEEP, exactly like a
    capped inventory or a lossy parser. Keeping costs one soft-failed check; dropping
    deletes a correct assertion silently."""

    _INVENTORY: ClassVar[dict[str, Any]] = {
        "pages": [{"headings": ["todos"], "interactive_elements": []}]
    }

    @staticmethod
    def _def(content: str) -> dict[str, Any]:
        return {"filename": "todo.steps.ts", "content": content, "step_count": 1}

    @staticmethod
    def _given(assertion: str) -> str:
        return (
            "Given('I am on the TodoMVC landing page', async ({ page }) => {\n"
            "  await page.goto(absoluteNavUrl('/'));\n"
            f"  {assertion}\n"
            "  await assertNavigated(page);\n"
            "});\n"
        )

    @pytest.mark.parametrize(
        ("pattern", "why"),
        [
            ("/(?<=x)todos/", "JS lookbehind — `_UNSAFE_JS_REGEX_RE` refuses it"),
            ("/\\p{L}+/u", "unicode property escape — Python `re` cannot compile it"),
            ("/todos{2,1}/", "`re.compile` raises: min repeat > max repeat"),
            ("/\\1todos/", "backreference — different meaning in JS and Python"),
        ],
    )
    def test_an_unevaluable_regex_is_unknown_and_kept(self, pattern: str, why: str) -> None:
        src = self._given(
            f"await softExpectVisible(page, page.getByRole('heading', "
            f"{{ name: {pattern} }}).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._INVENTORY) == 0, why
        assert d["content"] == src

    def test_an_over_long_regex_is_unknown_and_kept(self) -> None:
        src = self._given(
            "await softExpectVisible(page, page.getByRole('heading', "
            "{ name: /" + "a" * 250 + "/ }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._INVENTORY) == 0
        assert d["content"] == src

    def test_an_evaluable_regex_matching_nothing_is_still_dropped(self) -> None:
        """The correction must not disarm the pass: where it genuinely looked and the
        bucket is exhaustive, ABSENT is a real answer."""
        src = self._given(
            "await softExpectVisible(page, page.getByRole('heading', { name: /TodoMVC/ }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._INVENTORY) == 1
        assert "getByRole" not in d["content"]

    def test_a_concatenated_name_literal_is_unknown_and_kept(self) -> None:
        """Swept out of `_name_option_verdict`: `name: 'X' + suffix` reads the literal
        as if it were the WHOLE name, but it is only a prefix — the substring test can
        then answer ABSENT for a name the page really has."""
        src = self._given(
            "await softExpectVisible(page, page.getByRole('heading', "
            "{ name: 'Nope' + suffix }).first());"
        )
        d = self._def(src)
        assert _ensure_grounded_role_names([d], self._INVENTORY) == 0
        assert d["content"] == src


class TestNavigationPreconditionPromptRule:
    """The deterministic pass is the guarantee, but the prompt must still ASK for a
    navigation assertion and forbid invented selectors — a softened wrong locator is
    a worse clip than a right locator the model wrote itself."""

    async def test_recording_prompt_carries_navigation_precondition_rule(
        self, step_gen, mock_llm_client, sample_inspection_result
    ) -> None:
        mock_llm_client.generate.return_value = (
            "```typescript\n// filename: steps.ts\nGiven('x', async ({ page }) => {});\n```"
        )
        features = [{"content": "  Scenario: Home\n    Given I am on the page\n"}]

        await step_gen.generate(features, sample_inspection_result, recording=True)

        call_args = mock_llm_client.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "NAVIGATION PRECONDITIONS assert NAVIGATION, not decoration" in prompt
        assert "await assertNavigated(page);" in prompt
        assert "softExpectVisible(" in prompt
        assert "NEVER INVENT A SELECTOR" in prompt

    async def test_nonrecording_prompt_does_not_reference_recording_helpers(
        self, step_gen, mock_llm_client, sample_inspection_result
    ) -> None:
        """`lib/recording_helpers` ships ONLY with the demo_video package — telling a
        QA suite to import `assertNavigated` would emit an unresolvable import."""
        mock_llm_client.generate.return_value = (
            "```typescript\n// filename: steps.ts\nGiven('x', async ({ page }) => {});\n```"
        )
        features = [{"content": "  Scenario: Home\n    Given I am on the page\n"}]

        await step_gen.generate(features, sample_inspection_result, recording=False)

        call_args = mock_llm_client.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "assertNavigated" not in prompt
        assert "softExpectVisible" not in prompt
        # The selector-invention rule is stream-agnostic and MUST still be there.
        assert "NEVER INVENT A SELECTOR" in prompt

    async def test_prompt_no_longer_seeds_a_real_app_name(
        self, step_gen, mock_llm_client, sample_inspection_result
    ) -> None:
        """The STRICT MODE rule used to spell out
        `getByRole('link', { name: 'TodoMVC', exact: true })` — the exact invented
        locator the live landing Given copied. The rule stays; the app name goes."""
        mock_llm_client.generate.return_value = (
            "```typescript\n// filename: steps.ts\nGiven('x', async ({ page }) => {});\n```"
        )
        features = [{"content": "  Scenario: Home\n    Given I am on the page\n"}]

        await step_gen.generate(features, sample_inspection_result)

        call_args = mock_llm_client.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "TodoMVC" not in prompt
        assert "exact: true" in prompt
        assert "never copy a literal name" in prompt


class TestSurvivableTextPromptRule:
    """The deterministic rewrite is the guarantee, but the prompt must still ASK for a
    containment assertion — a model that writes `toContainText` itself picks a better
    substring than a mechanical rewrite of whatever exact string it invented."""

    async def test_prompt_carries_survivable_text_rule(
        self, step_gen, mock_llm_client, sample_inspection_result
    ) -> None:
        mock_llm_client.generate.return_value = (
            "```typescript\n// filename: steps.ts\nGiven('x', async ({ page }) => {});\n```"
        )
        features = [{"content": "  Scenario: Home\n    Given I am on the page\n"}]

        await step_gen.generate(features, sample_inspection_result, platform="web")

        call_args = mock_llm_client.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "toContainText" in prompt
        assert "toHaveText" in prompt


class TestUndefinedValueAssertionGuard:
    """A value matcher whose expected argument is provably `undefined` raises a
    Playwright MATCHER error, not an assertion failure — the step dies, the scenario
    aborts, and the recording is left on a static page. Measured live on orchestrator
    E2E run 32150382725 against TodoMVC:

        expect(locator('ul.todo-list')).toContainText(expected)
        Matcher error: expected value must be a string or regular expression
        Expected has value: undefined
    """

    def _sd(self, content: str) -> list[dict[str, object]]:
        return [{"filename": "steps/x.steps.ts", "content": content}]

    def test_unbound_positional_param_assertion_is_dropped(self) -> None:
        # Pattern declares NO placeholder, so `expected` can never be filled.
        sd = self._sd(
            "const { Then } = createBdd();\n"
            "Then('the list shows the task', async ({ page }, expected) => {\n"
            "  await expect(page.locator('ul.todo-list')).toContainText(expected);\n"
            "  await resilientClick(page, '.toggle', 'toggle');\n"
            "});\n"
        )
        assert _ensure_no_undefined_value_assertions(sd) == 1
        body = str(sd[0]["content"])
        assert "toContainText" not in body
        # The rest of the step survives — we drop one line, not the scenario.
        assert "resilientClick" in body

    def test_a_bound_placeholder_param_is_left_alone(self) -> None:
        sd = self._sd(
            "Then('the list shows {string}', async ({ page }, expected) => {\n"
            "  await expect(page.locator('ul.todo-list')).toContainText(expected);\n"
            "});\n"
        )
        assert _ensure_no_undefined_value_assertions(sd) == 0
        assert "toContainText(expected)" in str(sd[0]["content"])

    def test_a_string_literal_argument_is_left_alone(self) -> None:
        sd = self._sd(
            "Then('the list shows the task', async ({ page }) => {\n"
            "  await expect(page.locator('ul.todo-list')).toContainText('Walk the dog');\n"
            "});\n"
        )
        assert _ensure_no_undefined_value_assertions(sd) == 0
        assert "Walk the dog" in str(sd[0]["content"])

    def test_a_regex_pattern_step_is_never_flagged(self) -> None:
        # Capture groups are not counted, so a regex pattern must never be claimed
        # unbound — the guard may only fire on a PROVABLE case.
        sd = self._sd(
            "Then(/the list shows (.+)/, async ({ page }, expected) => {\n"
            "  await expect(page.locator('ul.todo-list')).toContainText(expected);\n"
            "});\n"
        )
        assert _ensure_no_undefined_value_assertions(sd) == 0

    def test_surplus_param_beyond_the_placeholder_count_is_dropped(self) -> None:
        # One placeholder fills `first`; `second` is always undefined.
        sd = self._sd(
            "Then('the list shows {string}', async ({ page }, first, second) => {\n"
            "  await expect(page.locator('h1')).toContainText(first);\n"
            "  await expect(page.locator('ul')).toHaveText(second);\n"
            "});\n"
        )
        assert _ensure_no_undefined_value_assertions(sd) == 1
        body = str(sd[0]["content"])
        assert "toContainText(first)" in body
        assert "toHaveText(second)" not in body

    def test_sibling_matchers_are_covered(self) -> None:
        for matcher in ("toHaveText", "toHaveValue", "toHaveAttribute"):
            sd = self._sd(
                "Then('a step', async ({ page }, ghost) => {\n"
                f"  await expect(page.locator('x')).{matcher}(ghost);\n"
                "});\n"
            )
            assert _ensure_no_undefined_value_assertions(sd) == 1, matcher
            assert matcher not in str(sd[0]["content"]), matcher

    def test_is_idempotent(self) -> None:
        sd = self._sd(
            "Then('a step', async ({ page }, ghost) => {\n"
            "  await expect(page.locator('x')).toContainText(ghost);\n"
            "});\n"
        )
        assert _ensure_no_undefined_value_assertions(sd) == 1
        assert _ensure_no_undefined_value_assertions(sd) == 0

    def test_an_occurrence_inside_a_comment_is_not_treated_as_code(self) -> None:
        sd = self._sd(
            "Then('a step', async ({ page }, ghost) => {\n"
            "  // await expect(page.locator('x')).toContainText(ghost);\n"
            "  await resilientClick(page, 'x', 'x');\n"
            "});\n"
        )
        assert _ensure_no_undefined_value_assertions(sd) == 0


class TestResilientRewriteImportsItsOwnBinding:
    """Bughunt 2026-08-21 #2: a rewrite must guarantee ITS OWN binding is imported.

    The old `_RESILIENT_IMPORT_RE` matched an import naming EITHER helper, so a file
    that imports only `resilientClick` and gets its `page.fill` rewritten to
    `resilientFill` sailed past the injector → `ReferenceError: resilientFill is
    not defined` at run time → the scenario aborts → black clip.
    """

    def test_fill_rewrite_adds_missing_fill_binding(self):
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "import { resilientClick } from '../lib/recording_helpers';\n"
            "const { When } = createBdd();\n"
            "When('a', async ({ page }) => {\n"
            "  await page.fill('#email', 'a@b.c');\n"
            "});\n"
        )
        step_defs: list[dict[str, Any]] = [{"filename": "s.ts", "content": content}]
        assert _ensure_resilient_import(step_defs) == 1
        new = step_defs[0]["content"]
        assert "resilientFill(page, '#email', 'a@b.c')" in new
        assert re.search(r"import\s*\{[^}]*\bresilientFill\b[^}]*\}", new), new
        # and must not RE-import the binding the file already had (TS redeclaration)
        assert len(re.findall(r"\bresilientClick\b(?=[^(]*\}\s*from)", new)) == 1, new

    def test_click_rewrite_adds_missing_click_binding(self):
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "import { resilientFill } from '../lib/recording_helpers';\n"
            "const { When } = createBdd();\n"
            "When('a', async ({ page }) => {\n"
            "  await page.click('#go');\n"
            "});\n"
        )
        step_defs: list[dict[str, Any]] = [{"filename": "s.ts", "content": content}]
        assert _ensure_resilient_import(step_defs) == 1
        new = step_defs[0]["content"]
        assert re.search(r"import\s*\{[^}]*\bresilientClick\b[^}]*\}", new), new

    def test_existing_correct_import_is_not_duplicated(self):
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "import { resilientClick, resilientFill } from '../lib/recording_helpers';\n"
            "const { When } = createBdd();\n"
            "When('a', async ({ page }) => {\n"
            "  await page.fill('#email', 'x');\n"
            "});\n"
        )
        step_defs: list[dict[str, Any]] = [{"filename": "s.ts", "content": content}]
        _ensure_resilient_import(step_defs)
        assert step_defs[0]["content"].count("recording_helpers") == 1


class TestCukePlaceholderVocabularyAgrees:
    """Bughunt 2026-08-21 #4: `_CUKE_PLACEHOLDER_RE` must honour every param type
    `_CUKE_PARAM_REGEX` knows, or a `{double}`-bound argument reads as unbound and
    the assertion is SILENTLY DELETED — a green `Then` that asserts nothing."""

    def test_placeholder_vocabulary_is_derived_from_the_param_regex_table(self):
        for name in _CUKE_PARAM_REGEX:
            assert _CUKE_PLACEHOLDER_RE.fullmatch("{" + name + "}"), name

    def test_double_placeholder_binds_its_param(self):
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "import { expect } from '@playwright/test';\n"
            "const { Then } = createBdd();\n"
            "Then('total is {double}', async ({ page }, v) => {\n"
            "  await expect(page.locator('#total')).toContainText(v);\n"
            "});\n"
        )
        step_defs: list[dict[str, Any]] = [{"filename": "s.ts", "content": content}]
        assert _ensure_no_undefined_value_assertions(step_defs) == 0
        assert "toContainText(v)" in step_defs[0]["content"]

    def test_anonymous_placeholder_binds_its_param(self):
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "import { expect } from '@playwright/test';\n"
            "const { Then } = createBdd();\n"
            "Then('total is {}', async ({ page }, v) => {\n"
            "  await expect(page.locator('#total')).toContainText(v);\n"
            "});\n"
        )
        step_defs: list[dict[str, Any]] = [{"filename": "s.ts", "content": content}]
        assert _ensure_no_undefined_value_assertions(step_defs) == 0
        assert "toContainText(v)" in step_defs[0]["content"]

    def test_long_and_byte_placeholders_bind_their_params(self):
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "import { expect } from '@playwright/test';\n"
            "const { Then } = createBdd();\n"
            "Then('a {long} b {byte}', async ({ page }, a, b) => {\n"
            "  await expect(page.locator('#x')).toContainText(a);\n"
            "  await expect(page.locator('#y')).toContainText(b);\n"
            "});\n"
        )
        step_defs: list[dict[str, Any]] = [{"filename": "s.ts", "content": content}]
        assert _ensure_no_undefined_value_assertions(step_defs) == 0

    def test_genuinely_surplus_param_is_still_dropped(self):
        """The guard must keep working: no placeholder at all ⇒ still unbound."""
        content = (
            "import { createBdd } from 'playwright-bdd';\n"
            "import { expect } from '@playwright/test';\n"
            "const { Then } = createBdd();\n"
            "Then('total is shown', async ({ page }, v) => {\n"
            "  await expect(page.locator('#total')).toContainText(v);\n"
            "});\n"
        )
        step_defs: list[dict[str, Any]] = [{"filename": "s.ts", "content": content}]
        assert _ensure_no_undefined_value_assertions(step_defs) == 1
        assert "toContainText(v)" not in step_defs[0]["content"]


class TestAutostubWarningStatesTheProportion:
    """An auto-stub PASSES while performing no action.

    Its whole body is `waitForLoadState('networkidle')` + `stubDwell` — it waits
    and returns green, and nothing downstream can distinguish a stubbed scenario
    from a real one. One stub in twenty is noise; most of the suite stubbed means
    the "demo" recording is a static page, which is precisely the shape behind
    duplicate-frame films.

    The count alone did not convey that. The warning now carries the SHARE.

    ⚠️ These assert the PRODUCTION call site, not a restatement of it. An earlier
    draft of this test rebuilt the log line itself and would have passed against
    code that never emitted it — the exact vacuous shape this suite exists to
    catch elsewhere.
    """

    @staticmethod
    def _emission_block() -> str:
        """The source of the `if recording:` block that emits the warning."""
        import inspect

        from src.services.generation import step_def_generator as mod

        src = inspect.getsource(mod)
        anchor = src.index("PASS WITHOUT ACTING")
        start = src.rindex("if recording:", 0, anchor)
        return src[start : anchor + 400]

    def test_the_warning_says_what_a_stub_actually_does(self) -> None:
        assert "PASS WITHOUT ACTING" in self._emission_block(), (
            "the operator must be told a stub PASSES, not merely that one exists"
        )

    def test_the_share_is_computed_from_the_real_step_list(self) -> None:
        block = self._emission_block()
        assert "_extract_steps_from_features(features)" in block, (
            "the denominator must come from the real step list, not a guess"
        )
        assert "stubbed / total_steps" in block, "the warning must carry a PROPORTION"

    def test_the_denominator_is_taken_before_stubbing(self) -> None:
        """Order matters: `_ensure_step_coverage` mutates `step_defs` in place.

        Counting after it would still be correct today (it appends defs, not
        steps), but it reads as if it might not be — and a future edit that made
        the count depend on step_defs would silently produce a share of 100%.
        """
        block = self._emission_block()
        assert block.index("total_steps = ") < block.index("_ensure_step_coverage("), (
            "compute the denominator before mutating"
        )

    def test_a_real_uncovered_suite_still_produces_stubs(self) -> None:
        """The behaviour the warning describes must actually happen."""
        from src.services.generation import step_def_generator as mod

        features = [
            {
                "content": (
                    "Feature: f\n  Scenario: s\n"
                    "    Given a covered step\n"
                    "    When an uncovered step\n"
                    "    Then another uncovered step\n"
                )
            }
        ]
        step_defs = [
            {
                "filename": "steps.ts",
                "content": (
                    "Given('a covered step', async ({ page }) => { await page.click('x'); });"
                ),
            }
        ]

        stubbed = mod._ensure_step_coverage(features, step_defs)  # type: ignore[arg-type]

        assert stubbed > 0
        assert any(sd.get("filename") == "steps_autostub.ts" for sd in step_defs)


class TestScopedResilientTargets:
    """`resilientClick`'s `primary` used to be a STRING selector only, while the
    prompt forbids a bare `page.click`. A step whose target is scoped ("the checkbox
    inside the 'Walk the dog' row") therefore had no way to say so: measured on five
    consecutive todomvc packages (bdd jobs 01a057ef, 01a05992, 01a059a2-87c6,
    01a059a2-eb89, 01a059d4-474a) the model built the scoped locator, then clicked a
    page-wide `input[type="checkbox"]` — TodoMVC's toggle-all box — and left the
    scoped locator bound and unused. `resilientClick` soft-fails silently, so the
    When step passed and run 01a059a2's NEXT step failed with an unchecked box."""

    @staticmethod
    def _def(content: str) -> dict[str, Any]:
        return {"filename": "todo.steps.ts", "content": content, "step_count": 1}

    # The verbatim shape from job 01a059d4-474a / 01a05992 / 01a059a2-87c6.
    _ABANDONED = (
        "When('I click the checkbox next to {string}', async ({ page }, text: string) => {\n"
        "  const todoItem = page.getByRole('listitem').filter({ hasText: text }).first();\n"
        "  const checkbox = todoItem.getByRole('checkbox').first();\n"
        "  await resilientClick(page, 'input[type=\"checkbox\"]', text);\n"
        "});\n"
    )
    # The verbatim shape from job 01a059a2-eb89: the Locator is handed to `opts`.
    _EATEN_BY_OPTS = (
        "When('I click the checkbox next to {string}', async ({ page }, text: string) => {\n"
        "  const listItem = page.getByRole('listitem').filter({ hasText: text }).first();\n"
        "  const checkbox = listItem.getByRole('checkbox').first();\n"
        "  await resilientClick(page, 'input[type=\"checkbox\"]', text, checkbox);\n"
        "});\n"
    )

    def test_abandoned_scoped_locator_becomes_the_click_target(self) -> None:
        d = self._def(self._ABANDONED)
        assert _ensure_scoped_resilient_targets([d]) == 1
        assert "await resilientClick(page, checkbox, text);" in d["content"]
        # The unscoped page-wide selector — TodoMVC's toggle-all — must be gone.
        assert 'input[type="checkbox"]' not in d["content"]

    def test_the_click_target_is_a_locator_that_actually_toggles(self) -> None:
        """Teeth: not merely "changed", but "the argument the helper clicks is the
        checkbox INSIDE the named row". Follows the binding chain rather than
        string-matching the call."""
        d = self._def(self._ABANDONED)
        _ensure_scoped_resilient_targets([d])
        call = re.search(r"resilientClick\(page,\s*([A-Za-z_$][\w$]*)\s*,", d["content"])
        assert call is not None, "primary is not a bound locator identifier"
        target = call.group(1)
        binding = re.search(rf"const {target} = ([^;]+);", d["content"])
        assert binding is not None
        chain = binding.group(1)
        assert "getByRole('checkbox')" in chain, chain
        assert chain.startswith("todoItem."), f"target is not scoped to the row: {chain}"

    def test_locator_eaten_by_the_opts_slot_is_hoisted_to_primary(self) -> None:
        d = self._def(self._EATEN_BY_OPTS)
        assert _ensure_scoped_resilient_targets([d]) == 1
        assert "await resilientClick(page, checkbox, text);" in d["content"]
        # The dead trailing argument must be removed, not left duplicating the target
        # (a Locator in the `opts` slot is silently dropped by the helper).
        assert ", text, checkbox)" not in d["content"]
        assert d["content"].count("resilientClick") == 1

    def test_resilient_fill_uses_its_own_slot_layout(self) -> None:
        """`resilientFill(page, primary, value, label, opts)` — the value is slot 2, so
        the label must never be mistaken for a stray Locator."""
        d = self._def(
            "When('I fill the row {string}', async ({ page }, text: string) => {\n"
            "  const row = page.getByRole('row').filter({ hasText: text }).first();\n"
            "  const field = row.getByRole('textbox').first();\n"
            "  await resilientFill(page, 'input', text, 'Name', field);\n"
            "});\n"
        )
        assert _ensure_scoped_resilient_targets([d]) == 1
        assert "await resilientFill(page, field, text, 'Name');" in d["content"]

    def test_two_abandoned_bindings_are_declined(self) -> None:
        """Guessing which node to click is worse than the unscoped selector."""
        d = self._def(
            "When('I click', async ({ page }) => {\n"
            "  const a = page.getByRole('checkbox').first();\n"
            "  const b = page.getByRole('button').first();\n"
            "  await resilientClick(page, 'input', 'x');\n"
            "});\n"
        )
        before = d["content"]
        assert _ensure_scoped_resilient_targets([d]) == 0
        assert d["content"] == before

    def test_a_used_binding_is_not_adopted(self) -> None:
        """A locator the step already asserts on is NOT abandoned — it is the step's
        own subject, and re-pointing the click at it would invent behaviour."""
        d = self._def(
            "When('I click', async ({ page }) => {\n"
            "  const row = page.getByRole('row').first();\n"
            "  await expect(row).toBeVisible();\n"
            "  await resilientClick(page, 'button.go', 'Go');\n"
            "});\n"
        )
        assert _ensure_scoped_resilient_targets([d]) == 0

    def test_a_binding_does_not_leak_across_step_callbacks(self) -> None:
        """A `const` is scoped to ONE step body; adopting another step's binding would
        emit a ReferenceError and abort the scenario mid-recording."""
        d = self._def(
            "Then('I see it', async ({ page }) => {\n"
            "  const orphan = page.getByRole('checkbox').first();\n"
            "});\n"
            "When('I click', async ({ page }) => {\n"
            "  await resilientClick(page, 'input', 'x');\n"
            "});\n"
        )
        assert _ensure_scoped_resilient_targets([d]) == 0
        assert "resilientClick(page, 'input', 'x')" in d["content"]

    def test_is_idempotent(self) -> None:
        d = self._def(self._ABANDONED)
        assert _ensure_scoped_resilient_targets([d]) == 1
        once = d["content"]
        assert _ensure_scoped_resilient_targets([d]) == 0
        assert d["content"] == once

    def test_wired_into_recording_compile_guarantees(self) -> None:
        d = self._def(self._ABANDONED)
        out = _apply_compile_guarantees([d], [], recording=True)
        assert "resilientClick(page, checkbox, text)" in out[0]["content"]

    def test_non_recording_stream_is_untouched(self) -> None:
        d = self._def(self._ABANDONED)
        out = _apply_compile_guarantees([d], [], recording=False)
        assert "resilientClick(page, 'input[type=\"checkbox\"]', text)" in out[0]["content"]


class TestScopedResilientTargetsBindingShapes:
    """The binding the pass adopts is found by parsing the statement, not by a single
    regex: a real package binds through intermediate chain links
    (`items.nth(i).getByRole('button', { name: 'Destroy' })`, job 01a057ef)."""

    @staticmethod
    def _def(content: str) -> dict[str, Any]:
        return {"filename": "todo.steps.ts", "content": content, "step_count": 1}

    def test_binding_through_intermediate_chain_links_is_adopted(self) -> None:
        d = self._def(
            "Given('the list is empty', async ({ page }) => {\n"
            "  const items = page.locator('.todo-list').locator('li');\n"
            "  const count = await items.count();\n"
            "  for (let i = 0; i < count; i++) {\n"
            "    const destroyBtn = items.nth(i).getByRole('button', { name: 'Destroy' });\n"
            "    await resilientClick(page, 'li button.destroy', 'Destroy');\n"
            "  }\n"
            "});\n"
        )
        assert _ensure_scoped_resilient_targets([d]) == 1
        assert "await resilientClick(page, destroyBtn, 'Destroy');" in d["content"]

    def test_an_awaited_binding_is_never_a_click_target(self) -> None:
        """`const count = await items.count()` is a NUMBER. Clicking it is a TypeError
        that aborts the scenario and blanks the clip — strictly worse than the
        unscoped selector this pass exists to replace."""
        d = self._def(
            "When('I click', async ({ page }) => {\n"
            "  const total = await page.locator('li').count();\n"
            "  await resilientClick(page, 'li button.destroy', 'Destroy');\n"
            "});\n"
        )
        assert _ensure_scoped_resilient_targets([d]) == 0
