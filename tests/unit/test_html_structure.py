"""Unit tests for html_structure.extract_structure.

The fixture ``TODOMVC_HTML`` is the REAL rendered ``html_content`` returned by
KAG's ``/scrape/bulk/sync`` for https://demo.playwright.dev/todomvc (captured
live 2026-07-23) — so these tests prove the parser turns the actual scrape
output into the structured inspection the demo generators consume.
"""

from __future__ import annotations

from typing import Any

from src.services.input.html_structure import extract_structure

# Exact html_content KAG returned for todomvc (the rendered shell: heading +
# the new-todo input + footer links).
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


def test_todomvc_extracts_new_todo_input_with_placeholder_selector() -> None:
    out = extract_structure(TODOMVC_HTML)
    inputs = [e for e in out["interactive_elements"] if e["tag"] == "input"]
    assert len(inputs) == 1
    el = inputs[0]
    assert el["label"] == "What needs to be done?"
    # The generated steps select this with getByPlaceholder — the parser must
    # emit exactly that idiom so the LLM can copy it.
    assert el["selector"] == 'getByPlaceholder("What needs to be done?")'


def test_todomvc_extracts_heading() -> None:
    out = extract_structure(TODOMVC_HTML)
    assert out["headings"] == ["todos"]


def test_todomvc_extracts_links_with_accessible_names() -> None:
    out = extract_structure(TODOMVC_HTML)
    links = [e for e in out["interactive_elements"] if e["tag"] == "a"]
    labels = {e["label"] for e in links}
    assert "real TodoMVC app." in labels
    assert "Remo H. Jansen" in labels
    # A named link becomes a getByRole('link', {name}) selector.
    remo = next(e for e in links if e["label"] == "Remo H. Jansen")
    assert remo["selector"] == 'getByRole("link", { name: "Remo H. Jansen", exact: true })'
    assert remo["href"] == "http://github.com/remojansen/"


def test_todomvc_has_forms_and_page_type() -> None:
    out = extract_structure(TODOMVC_HTML)
    # An <input> present -> the page is form-interactive even without <form>.
    assert out["has_forms"] is True
    assert out["page_type"] == "form"


def test_script_text_is_not_a_heading_or_element() -> None:
    html = "<html><body><script>var h='<h1>fake</h1>';</script><h1>Real</h1></body></html>"
    out = extract_structure(html)
    assert out["headings"] == ["Real"]


def test_label_for_association_gives_input_accessible_name() -> None:
    html = (
        '<form><label for="email">Email address</label>'
        '<input id="email" type="email" name="email"></form>'
    )
    out = extract_structure(html)
    el = next(e for e in out["interactive_elements"] if e["tag"] == "input")
    assert el["label"] == "Email address"
    assert el["selector"] == 'getByRole("textbox", { name: "Email address", exact: true })'
    assert out["has_forms"] is True


def test_aria_label_takes_priority_over_text() -> None:
    html = '<button aria-label="Close dialog">X</button>'
    out = extract_structure(html)
    el = out["interactive_elements"][0]
    assert el["label"] == "Close dialog"
    assert el["selector"] == 'getByRole("button", { name: "Close dialog", exact: true })'


def test_hidden_inputs_are_ignored() -> None:
    html = '<input type="hidden" name="csrf" value="abc"><input placeholder="Name">'
    out = extract_structure(html)
    tags = [(e["tag"], e["type"]) for e in out["interactive_elements"]]
    assert ("input", "hidden") not in tags
    assert len(out["interactive_elements"]) == 1


def test_id_fallback_selector_uses_attribute_selector() -> None:
    html = '<button id="submit-btn"></button>'
    out = extract_structure(html)
    # Attribute selector (not bare #id) so ids with ":"/space/"/" stay valid CSS.
    assert out["interactive_elements"][0]["selector"] == 'locator("[id=\\"submit-btn\\"]")'


# --- adversarial hardening (see review 2026-07-23) ---


def test_injectable_role_attr_is_neutralized() -> None:
    # A hostile role attr must never reach getByRole verbatim.
    html = '<a href="#" role=\'button")//\'>Go</a>'
    sel = extract_structure(html)["interactive_elements"][0]["selector"]
    assert '")//' not in sel
    assert not sel.startswith("getByRole")  # invalid role → safe locator fallback


def test_placeholder_newline_is_collapsed_not_literal() -> None:
    html = '<input placeholder="Line1\nLine2">'
    sel = extract_structure(html)["interactive_elements"][0]["selector"]
    assert "\n" not in sel
    assert sel == 'getByPlaceholder("Line1 Line2")'


def test_quote_in_label_is_escaped() -> None:
    html = "<button aria-label='Say \"hi\"'>x</button>"
    sel = extract_structure(html)["interactive_elements"][0]["selector"]
    assert sel == 'getByRole("button", { name: "Say \\"hi\\"", exact: true })'


def test_id_with_colon_or_space_uses_valid_attribute_selector() -> None:
    for raw_id, expect in [
        ("form:email", 'locator("[id=\\"form:email\\"]")'),
        ("user name", 'locator("[id=\\"user name\\"]")'),
    ]:
        html = f'<input id="{raw_id}">'
        assert extract_structure(html)["interactive_elements"][0]["selector"] == expect


def test_class_with_slash_uses_attribute_word_selector() -> None:
    # Tailwind-style class with "/" is invalid in a bare .class selector.
    html = '<a class="w-1/2">x</a>'  # no href → no role → class fallback
    sel = extract_structure(html)["interactive_elements"][0]["selector"]
    assert sel == 'locator("a[class~=\\"w-1/2\\"]")'


def test_wrapping_label_gives_input_accessible_name() -> None:
    html = "<label>Email <input type='text'></label>"
    el = next(e for e in extract_structure(html)["interactive_elements"] if e["tag"] == "input")
    assert el["label"] == "Email"


def test_heading_text_survives_nested_inline_tags() -> None:
    html = '<h1>Welcome <a href="/x">home</a> today</h1>'
    out = extract_structure(html)
    assert out["headings"] == ["Welcome home today"]
    # The nested link is still surfaced as its own element.
    assert any(e["tag"] == "a" and e["label"] == "home" for e in out["interactive_elements"])


def test_distinct_links_with_same_shape_are_not_over_deduped() -> None:
    # Two icon-only links to different hrefs must both survive (href in dedup key).
    html = '<a href="/a"><svg></svg></a><a href="/b"><svg></svg></a>'
    out = extract_structure(html)
    assert len([e for e in out["interactive_elements"] if e["tag"] == "a"]) == 2


def test_empty_and_non_string_html_yield_empty_structure() -> None:
    # Deliberately exercise runtime robustness to non-str inputs (typed Any).
    bad_inputs: list[Any] = ["", None, 123]
    for bad in bad_inputs:
        out = extract_structure(bad)
        assert out["headings"] == []
        assert out["interactive_elements"] == []
        assert out["has_forms"] is False
        assert out["page_type"] == "content"


def test_malformed_html_does_not_crash() -> None:
    html = "<h1>unclosed <a href=<<<>>> <input placeholder='x' <button>"
    out = extract_structure(html)  # must not raise
    assert isinstance(out["interactive_elements"], list)


def test_element_cap_and_truncation_count() -> None:
    html = "<body>" + "".join(f'<a href="/{i}">L{i}</a>' for i in range(50)) + "</body>"
    out = extract_structure(html)
    assert len(out["interactive_elements"]) == 40
    assert out["elements_truncated"] == 10


def test_duplicate_selectors_are_deduped() -> None:
    html = '<a href="/x">Go</a><a href="/x">Go</a>'
    out = extract_structure(html)
    # Same tag+selector collapses to one entry.
    assert len(out["interactive_elements"]) == 1


def _js_decode(body: str) -> str:
    """Decode a JS double-quoted string body: ``\\\\`` -> ``\\``, ``\\"`` -> ``"``."""
    out: list[str] = []
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body) and body[i + 1] in ('"', "\\"):
            out.append(body[i + 1])
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def test_id_with_embedded_quote_emits_valid_escaped_css() -> None:
    # An id containing a literal " must be escaped at BOTH the CSS and JS layers,
    # or the emitted locator("[id=\"...\"]") is invalid CSS that throws at runtime.
    # An unlabelled input reaches the id-attribute branch (role+name needs a label).
    out = extract_structure("<input id='a\"b'>")
    sel = out["interactive_elements"][0]["selector"]
    assert sel.startswith('locator("') and sel.endswith('")')
    css = _js_decode(sel[len('locator("') : -len('")')])
    # The inner quote survives as a CSS-escaped quote (\"), not a bare " that
    # would terminate the attribute value early.
    assert css == '[id="a\\"b"]'


def test_anonymous_interactive_element_falls_back_to_get_by_text() -> None:
    # A link with visible text but no href/id/class/name/role must NOT become the
    # ambiguous locator("a") (which matches every anchor → strict-mode violation).
    out = extract_structure("<a>Bare Link</a>")
    sel = out["interactive_elements"][0]["selector"]
    assert sel == 'getByText("Bare Link")'


def test_headings_capped_at_max() -> None:
    html = "".join(f"<h2>H{i}</h2>" for i in range(25))
    out = extract_structure(html)
    assert len(out["headings"]) == 20


# ── heading capture gaps (parser lossiness became silent assertion deletions) ──
# `step_def_generator._ensure_grounded_role_names` DROPS a `getByRole('heading',
# {name})` assertion when the name is absent from `headings`. Every heading this
# parser failed to record was therefore a correct assertion deleted from a live
# suite — so the parser must record the accessible name, not just `handle_data`.


def test_div_with_role_heading_is_recorded() -> None:
    out = extract_structure('<div role="heading" aria-level="1">Dashboard</div>')
    assert "Dashboard" in out["headings"]


def test_aria_label_on_a_heading_is_recorded_alongside_its_text() -> None:
    # The ACCESSIBLE name is "Foo"; the rendered text is "Bar". Recording only the
    # text drops a correct assertion on the a11y name — and recording only the
    # a11y name would drop a correct assertion on the text. Keep BOTH.
    out = extract_structure('<h1 aria-label="Foo">Bar</h1>')
    assert "Foo" in out["headings"]
    assert "Bar" in out["headings"]


def test_image_only_heading_contributes_its_alt_text() -> None:
    out = extract_structure('<h1><img src="l.png" alt="Acme Logo"></h1>')
    assert "Acme Logo" in out["headings"]


def test_headings_truncation_is_reported() -> None:
    out = extract_structure("".join(f"<h2>H{i}</h2>" for i in range(25)))
    assert out["headings_truncated"] == 5
    assert extract_structure("<h1>only</h1>")["headings_truncated"] == 0
