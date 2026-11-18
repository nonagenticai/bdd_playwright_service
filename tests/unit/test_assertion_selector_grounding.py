"""Recording-stream grounding of assertion SELECTORS against the scraped DOM.

Replays acceptance run ``01a0b007-9c2c`` (bdd job ``01a0b007-bb54``, FTR 5 scenarios,
2 passed / 3 failed) against https://demo.playwright.dev/todomvc. The step file below
is the generated ``steps/todo_steps.ts`` read back from that job's artifact store
(comment walls trimmed, code verbatim), and the snapshot is the job's persisted
``inspection_data`` (input + 3 links) plus the ``dom_tokens`` inventory
``extract_structure`` now derives from the SAME rendered html KAG returns.

The measured defect: ``page.locator('#todo-count')`` — the app renders
``<span data-testid="todo-count" class="todo-count">``, only once a todo exists —
timed out for 5 s in clip-1 and clip-2 and froze the tail of shipped footage.

What the pass removes is "unverifiable against the pre-interaction snapshot", NOT
"guessed": a CORRECT post-interaction `getByTestId('todo-count')` is dropped too
(asserted below), which is acceptable only because a recording's product is footage.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pytest

from src.services.generation.step_def_generator import (
    StepCoverageReport,
    _apply_compile_guarantees,
    _ensure_grounded_assertion_selectors,
    _selector_claims,
)
from src.services.input import html_structure
from src.services.input.html_structure import extract_structure
from src.services.input.inspection_adapter import InspectionAdapter

# The rendered html KAG `/scrape/bulk/sync` returned for todomvc (same fixture as
# tests/unit/test_html_structure.py — captured live 2026-07-23).
TODOMVC_HTML = (
    '<!DOCTYPE html><html lang="en" data-framework="typescript"><head>'
    '<meta charset="utf-8"><title>React • TodoMVC</title>'
    '<link rel="stylesheet" href="base.css"></head><body>'
    "<div>This is just a demo of TodoMVC for testing, not the "
    '<a href="https://todomvc.com/">real TodoMVC app.</a></div>'
    '<section class="todoapp"><div><header class="header"><h1>todos</h1>'
    '<input class="new-todo" placeholder="What needs to be done?"></header></div></section>'
    '<footer class="info"><p>Double-click to edit a todo</p>'
    '<p>Created by <a href="http://github.com/remojansen/">Remo H. Jansen</a></p>'
    '<p>Part of <a href="http://todomvc.com">TodoMVC</a></p></footer>'
    '<script type="text/javascript" src="js/bundle.js"></script></body></html>'
)

# `bdd_jobs.inspection_data` of job 01a0b007-bb54 (id 651), verbatim, WITHOUT tokens.
_PERSISTED_PAGE: dict[str, Any] = {
    "url": "https://demo.playwright.dev/todomvc",
    "title": "React • TodoMVC",
    "page_id": "",
    "headings": ["todos"],
    "has_forms": True,
    "page_type": "form",
    "text_content": "React • TodoMVC\nThis is just a demo of TodoMVC for testing, not the\n"
    "real TodoMVC app.",
    "elements_truncated": 0,
    "headings_truncated": 0,
    "interactive_elements": [
        {
            "tag": "a",
            "href": "https://todomvc.com/",
            "name": "",
            "text": "real TodoMVC app.",
            "type": "",
            "label": "real TodoMVC app.",
            "selector": 'getByRole("link", { name: "real TodoMVC app.", exact: true })',
        },
        {
            "tag": "input",
            "href": "",
            "name": "",
            "text": "",
            "type": "",
            "label": "What needs to be done?",
            "selector": 'getByPlaceholder("What needs to be done?")',
        },
        {
            "tag": "a",
            "href": "http://github.com/remojansen/",
            "name": "",
            "text": "Remo H. Jansen",
            "type": "",
            "label": "Remo H. Jansen",
            "selector": 'getByRole("link", { name: "Remo H. Jansen", exact: true })',
        },
        {
            "tag": "a",
            "href": "http://todomvc.com",
            "name": "",
            "text": "TodoMVC",
            "type": "",
            "label": "TodoMVC",
            "selector": 'getByRole("link", { name: "TodoMVC", exact: true })',
        },
    ],
}


def _inspection(page_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    page = {**_PERSISTED_PAGE, "dom_tokens": extract_structure(TODOMVC_HTML)["dom_tokens"]}
    page.update(page_overrides or {})
    return {
        "pages": [page],
        "base_url": "https://demo.playwright.dev/todomvc",
        "platform": "web",
        "navigation_graph": [],
    }


# Generated `steps/todo_steps.ts` of job 01a0b007-bb54 — code verbatim, the model's
# multi-line deliberation comments trimmed.
REAL_STEPS = """// filename: todo_steps.ts
import { createBdd } from 'playwright-bdd';
import { expect } from '@playwright/test';
import {
    typeVisibly,
    resilientClick,
    resilientFill,
    assertNavigated,
    softExpectVisible,
    dwellWithScroll
} from '../lib/recording_helpers';

const { Given, When, Then } = createBdd();

// Selectors
const TODO_INPUT = 'input[placeholder="What needs to be done?"]';
const TODO_LIST = 'ul.todo-list';
const TODO_ITEM = 'li.todo-item';
const TODO_TITLE = 'span.title';
const TODO_CHECKBOX = 'input.toggle';
const TODO_COUNT = 'span#todo-count';
const COMPLETED_COUNT = 'span#completed-count';
const ACTIVE_COUNT = 'span#active-count';

Given('I am on the React TodoMVC application at {string}', async ({ page }, url: string) => {
    await page.goto(url);
    await assertNavigated(page);
    // Wait for the main content to be visible
    await softExpectVisible(page, page.getByPlaceholder('What needs to be done?').first());
});

When('I enter {string} into the todo input field', async ({ page }, text: string) => {
    const input = page.getByPlaceholder('What needs to be done?').first();
    await resilientFill(page, input, text, 'What needs to be done?');
});

When('I press the Enter key', async ({ page }) => {
    await page.keyboard.press('Enter');
});

Then('I should see {string} listed in the todo list', async ({ page }, text: string) => {
    const item = page.getByRole('listitem').filter({ hasText: text }).first();
    await expect(item).toBeVisible();
});

Then('the todo count should display {string}', async ({ page }, count: string) => {
    // The count is usually in a span with id "todo-count" or similar.
    const countElement = page.locator('#todo-count').first();
    await expect(countElement).toContainText(count);
});

Then('the completed count should display {string}', async ({ page }, count: string) => {
    // Since I don't have the full DOM, I will use a text-based locator
    const footer = page.locator('footer').first();
    const completedText = footer.getByText(new RegExp(`\\\\b${count}\\\\b`)).first();
    await expect(footer.getByText(new RegExp(`\\\\b${count}\\\\b`)).first()).toBeVisible();
});

Then('the active count should display {string}', async ({ page }, count: string) => {
    const footer = page.locator('footer').first();
    const activeText = footer.getByText(new RegExp(`\\\\b${count}\\\\b`)).first();
    await expect(activeText).toBeVisible();
});

When('I click the checkbox next to {string}', async ({ page }, text: string) => {
    const item = page.getByRole('listitem').filter({ hasText: text }).first();
    const checkbox = item.getByRole('checkbox').first();
    await resilientClick(page, checkbox, text);
});

Then('{string} should be marked as completed', async ({ page }, text: string) => {
    const item = page.getByRole('listitem').filter({ hasText: text }).first();
    const checkbox = item.getByRole('checkbox').first();
    await expect(checkbox).toBeChecked();
});

Then('the todo input field should be empty', async ({ page }) => {
    const input = page.getByPlaceholder('What needs to be done?').first();
    await expect(input).toHaveValue('');
});
"""

# The other measured/declared selector shapes, plus a GROUNDED css selector that must
# survive (`.new-todo` and `footer.info` ARE in the pre-interaction DOM).
SHAPES_STEPS = """import { createBdd } from 'playwright-bdd';
import { expect } from '@playwright/test';
const { Given, When, Then } = createBdd();
const TODO_COUNT = 'span#todo-count';

Then('the todo count via const is {string}', async ({ page }, count: string) => {
    await page.locator('.new-todo').fill(count);
    await expect(page.locator(TODO_COUNT)).toContainText(count);
});

Then('the completed count is {string}', async ({ page }, count: string) => {
    await expect(page.locator("span#completed-count")).toHaveText(count);
});

Then('the chained count is {string}', async ({ page }, count: string) => {
    const footer = page.locator('footer');
    await expect(footer.locator('.todo-count')).toContainText(count);
});

Then('the testid count is {string}', async ({ page }, count: string) => {
    await expect(page.getByTestId('todo-count')).toContainText(count);
});

Then('the new todo input is ready', async ({ page }) => {
    await expect(page.locator('input.new-todo')).toBeVisible();
    await expect(page.locator('footer.info')).toBeVisible();
});

Then('{string} is struck through', async ({ page }, text: string) => {
    const row = page.getByRole('listitem').filter({ hasText: text }).first();
    await expect(row).toHaveCSS('text-decoration', /line-through/);
});
"""


def _defs(content: str, filename: str = "todo_steps.ts") -> list[dict[str, Any]]:
    return [{"filename": filename, "content": content, "step_count": content.count("async (")}]


def _step_body(content: str, pattern: str) -> str:
    start = content.index(pattern)
    end = content.find("\n});", start)
    return content[start:end]


class TestRealTodomvcPackage:
    """The generated file of the measured run, against the measured snapshot."""

    def test_the_measured_todo_count_assertion_is_removed(self) -> None:
        defs = _defs(REAL_STEPS)
        coverage = StepCoverageReport()
        changed = _ensure_grounded_assertion_selectors(defs, _inspection(), coverage)
        out = defs[0]["content"]
        assert changed == 1
        body = _step_body(out, "the todo count should display")
        assert "expect(countElement)" not in body
        assert [(d["selector"], d["claim"]) for d in coverage.ungrounded_assertions] == [
            ("#todo-count", "#todo-count")
        ]
        assert coverage.ungrounded_assertions[0]["file"] == "todo_steps.ts"
        assert coverage.ungrounded_assertions[0]["step"] == "the todo count should display {string}"
        assert coverage.as_dict()["ungrounded_assertions_dropped"] == 1

    def test_every_action_and_every_role_or_text_assertion_is_kept(self) -> None:
        defs = _defs(REAL_STEPS)
        _ensure_grounded_assertion_selectors(defs, _inspection())
        out = defs[0]["content"]
        for kept in (
            "await page.goto(url);",
            "await assertNavigated(page);",
            "await softExpectVisible(page, page.getByPlaceholder('What needs to be done?')",
            "await resilientFill(page, input, text, 'What needs to be done?');",
            "await page.keyboard.press('Enter');",
            "await expect(item).toBeVisible();",
            "await resilientClick(page, checkbox, text);",
            "await expect(checkbox).toBeChecked();",
            "await expect(input).toHaveValue('');",
            # clip-3 class: `footer` is tag-only and the text is a step parameter —
            # no id/class claim, so this pass is not authoritative (residual).
            "await expect(activeText).toBeVisible();",
            "await expect(footer.getByText(",
        ):
            assert kept in out, kept
        assert out.count("expect(") == REAL_STEPS.count("expect(") - 1

    def test_it_is_idempotent(self) -> None:
        defs = _defs(REAL_STEPS)
        _ensure_grounded_assertion_selectors(defs, _inspection())
        once = defs[0]["content"]
        assert _ensure_grounded_assertion_selectors(defs, _inspection()) == 0
        assert defs[0]["content"] == once


class TestSelectorShapes:
    def _run(self) -> tuple[str, StepCoverageReport]:
        defs = _defs(SHAPES_STEPS)
        coverage = StepCoverageReport()
        _ensure_grounded_assertion_selectors(defs, _inspection(), coverage)
        return defs[0]["content"], coverage

    def test_unverifiable_ids_classes_testids_and_css_matchers_are_removed(self) -> None:
        out, coverage = self._run()
        assert "expect(page.locator(TODO_COUNT))" not in out
        assert 'expect(page.locator("span#completed-count"))' not in out
        assert "expect(footer.locator('.todo-count'))" not in out
        # CORRECT on the live app (`<span data-testid="todo-count">`), but rendered only
        # after a todo exists — unverifiable against the snapshot, so dropped by design.
        assert "expect(page.getByTestId('todo-count'))" not in out
        assert ".toHaveCSS(" not in out
        assert sorted(d["claim"] for d in coverage.ungrounded_assertions) == sorted(
            [
                "#todo-count",
                "#completed-count",
                ".todo-count",
                "[data-testid='todo-count']",
                "toHaveCSS",
            ]
        )

    def test_grounded_css_selectors_and_actions_are_kept(self) -> None:
        out, _ = self._run()
        assert "await expect(page.locator('input.new-todo')).toBeVisible();" in out
        assert "await expect(page.locator('footer.info')).toBeVisible();" in out
        assert "    await page.locator('.new-todo').fill(count);\n    // assertion dropped" in out
        assert (
            "const footer = page.locator('footer');\n"
            "    // assertion dropped: unverifiable against pre-interaction snapshot "
            "(.todo-count)\n});"
        ) in out

    def test_an_emptied_step_explains_itself(self) -> None:
        out, _ = self._run()
        assert (
            "Then('the testid count is {string}', async ({ page }, count: string) => {\n"
            "    // assertion dropped: unverifiable against pre-interaction snapshot "
            "([data-testid='todo-count'])\n});"
        ) in out
        assert "const footer = page.locator('footer');" in out
        assert "const row = page.getByRole('listitem')" in out


class TestNeverDropsOnUnknown:
    def test_no_dom_tokens_changes_nothing(self) -> None:
        # A legacy row / an unscanned page: absence was never looked for.
        inspection = {"pages": [dict(_PERSISTED_PAGE)]}
        defs = _defs(REAL_STEPS)
        assert _ensure_grounded_assertion_selectors(defs, inspection) == 0
        assert defs[0]["content"] == REAL_STEPS

    def test_one_unscanned_page_disables_the_union(self) -> None:
        inspection = _inspection()
        inspection["pages"].append({**_PERSISTED_PAGE, "dom_tokens": None})
        defs = _defs(REAL_STEPS)
        assert _ensure_grounded_assertion_selectors(defs, inspection) == 0

    def test_a_truncated_bucket_is_not_evidence(self) -> None:
        tokens = dict(extract_structure(TODOMVC_HTML)["dom_tokens"] or {})
        tokens["ids_truncated"] = 3
        defs = _defs(REAL_STEPS)
        assert _ensure_grounded_assertion_selectors(defs, _inspection({"dom_tokens": tokens})) == 0

    def test_a_token_free_dom_is_treated_as_an_unhydrated_shell(self) -> None:
        shell = extract_structure("<html><body><div></div></body></html>")["dom_tokens"]
        defs = _defs(REAL_STEPS)
        assert _ensure_grounded_assertion_selectors(defs, _inspection({"dom_tokens": shell})) == 0

    def test_an_assertion_bundled_with_an_action_is_refused(self) -> None:
        content = """const { Then } = createBdd();
Then('it saves', async ({ page }) => {
    await Promise.all([page.getByRole('button').click(), expect(page.locator('#nope')).toBeVisible()]);
});
"""
        defs = _defs(content)
        assert _ensure_grounded_assertion_selectors(defs, _inspection()) == 0
        assert defs[0]["content"] == content

    def test_unreadable_selectors_make_no_claim(self) -> None:
        for selector in (
            ".a:not(.nope)",  # the token must be ABSENT for this to match
            ".nope, .new-todo",  # a selector list: either suffices
            "text=Buy groceries",
            "//span[@id='nope']",
            "md\\\\:nope",
            "input[class='NOPE' i]",
            "footer",
            'input[placeholder="What needs to be done?"]',
        ):
            claims = _selector_claims(selector)
            assert all(token != "nope" for _, token in claims), (selector, claims)
            assert all(token != "NOPE" for _, token in claims), (selector, claims)

    def test_claims_are_read_from_ids_classes_attributes_and_chains(self) -> None:
        assert _selector_claims("span#todo-count") == [("id", "todo-count")]
        assert _selector_claims("ul.todo-list > li.todo-item") == [
            ("class", "todo-list"),
            ("class", "todo-item"),
        ]
        assert _selector_claims('[data-testid="todo-count"]') == [("testid", "todo-count")]
        assert _selector_claims('span[id="x"]') == [("id", "x")]
        assert _selector_claims(".a >> .b") == [("class", "a"), ("class", "b")]
        assert _selector_claims("li:has-text('1.5 items')") == []


class TestStreamScopeAndWiring:
    """Called on the RECORDING generation path, and ONLY there."""

    def _features(self) -> list[dict[str, Any]]:
        return [
            {"content": 'Feature: t\n  Scenario: s\n    Then the todo count should display "1"\n'}
        ]

    def test_recording_compile_guarantees_drop_and_record(self) -> None:
        coverage = StepCoverageReport()
        out = _apply_compile_guarantees(
            _defs(REAL_STEPS),
            self._features(),
            recording=True,
            inspection=_inspection(),
            coverage=coverage,
        )
        steps = next(sd for sd in out if sd["filename"] == "todo_steps.ts")
        assert "expect(countElement)" not in steps["content"]
        assert coverage.as_dict()["ungrounded_assertions"][0]["claim"] == "#todo-count"

    def test_non_recording_stream_keeps_the_assertion(self) -> None:
        out = _apply_compile_guarantees(
            _defs(REAL_STEPS), self._features(), recording=False, inspection=_inspection()
        )
        steps = next(sd for sd in out if sd["filename"] == "todo_steps.ts")
        assert "await expect(countElement).toContainText(count);" in steps["content"]


class TestParseFailureIsNotAnInventory:
    """A parse that raised mid-way leaves a PARTIAL token set; it must read as "not
    scanned" (``None``), never as an inventory whose gaps would drop assertions."""

    def test_parse_failure_yields_no_tokens_and_the_pass_is_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(self: object) -> list[object]:
            raise RuntimeError("finalize failed")

        monkeypatch.setattr(html_structure._StructureParser, "finalize", _boom)
        structure = extract_structure(TODOMVC_HTML)
        assert structure["dom_tokens"] is None
        defs = _defs(REAL_STEPS)
        inspection = _inspection({"dom_tokens": structure["dom_tokens"]})
        assert _ensure_grounded_assertion_selectors(defs, inspection) == 0
        assert defs[0]["content"] == REAL_STEPS


class TestDomTokenInventory:
    def test_todomvc_tokens_are_the_rendered_classes(self) -> None:
        tokens = extract_structure(TODOMVC_HTML)["dom_tokens"]
        assert tokens is not None
        assert tokens["classes"] == ["header", "info", "new-todo", "todoapp"]
        assert tokens["ids"] == []
        assert tokens["testids"] == []

    def test_unscanned_html_is_none_not_empty(self) -> None:
        assert extract_structure("")["dom_tokens"] is None

    def test_template_children_are_not_rendered_tokens(self) -> None:
        html = '<div id="a" data-testid="t"><template><span class="ghost"></span></template></div>'
        tokens = extract_structure(html)["dom_tokens"]
        assert tokens is not None
        assert tokens["ids"] == ["a"]
        assert tokens["testids"] == ["t"]
        assert "ghost" not in tokens["classes"]

    def test_the_adapter_carries_tokens_through_asdict(self) -> None:
        # The web_url path is `asdict(adapter.from_web_crawl(...))`, which drops any key
        # PageInfo does not declare.
        page = {**_PERSISTED_PAGE, "dom_tokens": extract_structure(TODOMVC_HTML)["dom_tokens"]}
        result = asdict(InspectionAdapter().from_web_crawl({"pages": [page]}))
        assert result["pages"][0]["dom_tokens"]["classes"] == [
            "header",
            "info",
            "new-todo",
            "todoapp",
        ]
