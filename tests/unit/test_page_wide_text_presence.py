"""Recording-stream guard: a generic text-presence ``Then`` asserts PAGE-WIDE.

Replays showcase run ``01a15d7c-ca2d-7786-8603-017f1bd1f465`` clip-5 against
https://demo.playwright.dev/todomvc: the step ``Then I see the text "This is just a demo
of TodoMVC for testing, not the real TodoMVC app." displayed on the page`` was
implemented as ``getByRole('listitem').filter({ hasText }).first()``. On that page the
disclaimer is a top-level ``<div>``, inside no list item, so the assertion timed out
after 5 s (``expect(locator).toContainText(expected)``) and froze the clip's tail.
"""

from __future__ import annotations

import inspect
from typing import Any

from src.services.generation import step_def_generator
from src.services.generation.step_def_generator import (
    _apply_compile_guarantees,
    _ensure_page_wide_text_presence,
)

DISCLAIMER = "This is just a demo of TodoMVC for testing, not the real TodoMVC app."

# The clip-5 shape: one step def shared by a todo-item check and a page-wide text check.
CLIP5_THEN = (
    "Then('I see the text {string} displayed on the page', async ({ page }, text) => {\n"
    "  await expect(page.getByRole('listitem').filter({ hasText: text }).first())\n"
    "    .toContainText(text);\n"
    "});\n"
)

TODO_ITEM_THEN = (
    "Then('the todo {string} is in the list', async ({ page }, text) => {\n"
    "  await expect(page.getByRole('listitem').filter({ hasText: text }).first())"
    ".toBeVisible();\n"
    "});\n"
)


def _run(content: str) -> tuple[str, int]:
    defs: list[dict[str, Any]] = [{"filename": "steps/todo_steps.ts", "content": content}]
    changed = _ensure_page_wide_text_presence(defs)
    return defs[0]["content"], changed


def test_clip5_generic_text_step_is_rewritten_page_wide() -> None:
    out, changed = _run(CLIP5_THEN)
    assert changed == 1
    assert "await expect(page.getByText(text).first())\n    .toContainText(text);" in out
    assert "listitem" not in out


def test_todo_item_step_keeps_listitem_scope() -> None:
    out, changed = _run(TODO_ITEM_THEN)
    assert changed == 0
    assert out == TODO_ITEM_THEN


def test_shared_file_rewrites_only_the_generic_step() -> None:
    out, changed = _run(TODO_ITEM_THEN + CLIP5_THEN)
    assert changed == 1
    assert out.count("getByRole('listitem').filter({ hasText: text }).first()") == 1
    assert out.count("page.getByText(text).first()") == 1
    assert out.startswith(TODO_ITEM_THEN)


def test_literal_text_with_comma_is_carried_verbatim() -> None:
    content = (
        "Then('the disclaimer is displayed', async ({ page }) => {\n"
        "  const note = page.getByRole('listitem').filter({ hasText: "
        f"'{DISCLAIMER}' }}).first();\n"
        "  await expect(note).toBeVisible();\n"
        "});\n"
    )
    out, changed = _run(content)
    assert changed == 1
    assert f"const note = page.getByText('{DISCLAIMER}').first();" in out


def test_row_and_cell_scopes_share_the_rewrite() -> None:
    content = (
        "Then('I should see {string}', async ({ page }, v) => {\n"
        '  await expect(page.getByRole("row").filter({ hasText: v }).nth(0)).toBeVisible();\n'
        "  await expect(page.getByRole('cell').filter({ hasText: v }).last()).toBeVisible();\n"
        "});\n"
    )
    out, changed = _run(content)
    assert changed == 1
    assert "page.getByText(v).nth(0)" in out
    assert "page.getByText(v).last()" in out


def test_child_query_into_the_scoped_item_is_kept() -> None:
    direct = (
        "Then('{string} is shown as completed', async ({ page }, t) => {\n"
        "  await expect(page.getByRole('listitem').filter({ hasText: t }).first()"
        ".getByRole('checkbox')).toBeChecked();\n"
        "});\n"
    )
    bound = (
        "Then('{string} is shown as completed', async ({ page }, t) => {\n"
        "  const item = page.getByRole('listitem').filter({ hasText: t }).first();\n"
        "  await expect(item.getByRole('checkbox')).toBeChecked();\n"
        "});\n"
    )
    for content in (direct, bound):
        out, changed = _run(content)
        assert changed == 0
        assert out == content


def test_unnarrowed_or_other_filter_options_are_kept() -> None:
    content = (
        "Then('I see {string}', async ({ page }, t) => {\n"
        "  await expect(page.getByRole('listitem').filter({ hasText: t })).toHaveCount(1);\n"
        "  await expect(page.getByRole('listitem').filter({ hasText: t, has: page.getByRole("
        "'checkbox') }).first()).toBeVisible();\n"
        "  await expect(page.getByRole('listitem', { name: t }).filter({ hasText: t })"
        ".first()).toBeVisible();\n"
        "});\n"
    )
    out, changed = _run(content)
    assert changed == 0
    assert out == content


def test_when_steps_and_non_literal_patterns_are_untouched() -> None:
    content = (
        "When('I see the text {string}', async ({ page }, t) => {\n"
        "  await expect(page.getByRole('listitem').filter({ hasText: t }).first())"
        ".toBeVisible();\n"
        "});\n"
        "Then(PATTERN, async ({ page }, t) => {\n"
        "  await expect(page.getByRole('listitem').filter({ hasText: t }).first())"
        ".toBeVisible();\n"
        "});\n"
    )
    out, changed = _run(content)
    assert changed == 0
    assert out == content


def test_recording_compile_guarantees_apply_the_pass_but_qa_does_not() -> None:
    features = [
        {
            "filename": "todo.feature",
            "content": (
                "Feature: Todo\n  Scenario: Disclaimer\n"
                f'    Then I see the text "{DISCLAIMER}" displayed on the page\n'
            ),
        }
    ]
    header = "import { expect } from '@playwright/test';\n"
    for recording, expect_rewrite in ((True, True), (False, False)):
        defs: list[dict[str, Any]] = [
            {"filename": "steps/todo_steps.ts", "content": header + CLIP5_THEN}
        ]
        out = _apply_compile_guarantees(defs, features, recording=recording)
        content = "\n".join(d["content"] for d in out)
        assert ("page.getByText(text).first()" in content) is expect_rewrite
        assert ("getByRole('listitem')" in content) is not expect_rewrite


def test_prompt_asks_for_page_wide_text_presence() -> None:
    source = inspect.getsource(step_def_generator)
    assert "TEXT PRESENCE: a step that only says some text is shown" in source
    assert "`await expect(page.getByText(value).first()).toBeVisible();`" in source


# demo_video run 01a163a7 clip-4 (TodoMVC): the pattern names no scoping noun, but the
# matcher asserts CONTAINER state. `completed` lives on the <li>; the text sits in a
# classless <label>, so a page-wide `getByText` received class "" and timed out.
COMPLETED_STYLE_THEN = (
    "Then('the text {string} should be displayed with a completed style', "
    "async ({ page }, name) => {\n"
    "  await expect(page.getByRole('listitem').filter({ hasText: name }).first())"
    ".toHaveClass(/completed/);\n"
    "});\n"
)


def test_container_state_matcher_keeps_role_scope() -> None:
    out, changed = _run(COMPLETED_STYLE_THEN)
    assert changed == 0
    assert out == COMPLETED_STYLE_THEN


def test_to_be_checked_keeps_role_scope() -> None:
    content = (
        "Then('{string} should be ticked', async ({ page }, t) => {\n"
        "  await expect(page.getByRole('listitem').filter({ hasText: t }).first())\n"
        "    .toBeChecked();\n"
        "});\n"
    )
    out, changed = _run(content)
    assert changed == 0
    assert out == content


def test_non_presence_matchers_and_uses_keep_role_scope() -> None:
    loc = "page.getByRole('listitem').filter({ hasText: t }).first()"
    bodies = [
        f"  await expect({loc}).toHaveAttribute('data-done', 'true');\n",
        f"  await expect({loc}).toHaveCSS('opacity', '0.5');\n",
        f"  await expect({loc}).toBeEnabled();\n",
        f"  await expect({loc}).toHaveCount(1);\n",
        f"  await expect({loc}).not.toBeVisible();\n",
        f"  await expect({loc}, 'msg').toBeVisible();\n",
        f"  await {loc}.click();\n",
        f"  await helper({loc});\n",
        f"  const row = {loc};\n  await expect(row).toBeVisible();\n"
        "  await expect(row).toHaveClass(/completed/);\n",
        f"  const row = {loc};\n  await helper(row);\n",
        f"  const row = {loc};\n",
    ]
    for body in bodies:
        content = f"Then('I see {{string}}', async ({{ page }}, t) => {{\n{body}}});\n"
        out, changed = _run(content)
        assert changed == 0, body
        assert out == content, body


def test_presence_uses_are_still_rewritten() -> None:
    loc = "page.getByRole('listitem').filter({ hasText: t }).first()"
    bodies = [
        f"  await expect({loc}).toBeVisible();\n",
        f"  await expect({loc}).toHaveText(t);\n",
        f"  await expect.soft({loc}).toContainText(t);\n",
        f"  await {loc}.waitFor();\n",
        f"  const row = {loc};\n  await row.waitFor();\n  await expect(row).toBeVisible();\n",
    ]
    for body in bodies:
        content = f"Then('I see {{string}}', async ({{ page }}, t) => {{\n{body}}});\n"
        out, changed = _run(content)
        assert changed == 1, body
        assert "page.getByText(t).first()" in out, body
        assert "listitem" not in out, body
