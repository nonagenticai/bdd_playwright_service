"""Derive structured page inspection from raw scraped HTML.

webscraper's ``POST /api/scrape/bulk`` endpoint returns the rendered ``html_content`` but
NOT the structured fields (headings / forms / interactive elements with
selectors) that the demo-video generators consume. bdd's ``_page_from_scrape``
calls :func:`extract_structure` to parse that HTML into those fields locally,
using only the stdlib :mod:`html.parser` — no third-party dependency, so the
pip-audit / bandit surface is unchanged.

The output shape matches the ``interactive_elements`` / ``headings`` /
``has_forms`` / ``page_type`` keys read by
``inspection_adapter.InspectionAdapter._parse_pages`` and rendered into the LLM
prompt by ``feature_generator._build_page_summary`` (``- {tag}: {label}
[{selector}]``). Selectors are Playwright-idiomatic expressions the step-def
generator can copy verbatim (``getByPlaceholder(...)`` / ``getByRole(role,
{name})`` / ``locator(...)``), matching how existing generated steps select.

Robustness note: every value embedded into a selector expression is
whitespace-collapsed and escaped, and roles are restricted to a safe token set,
so hostile attribute values cannot emit malformed or injectable JS/CSS. The
parser degrades on malformed HTML and NEVER raises — a crash would fail a whole
demo_video job.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class InteractiveElementDict(TypedDict):
    """Adapter-shaped interactive element (all values are strings)."""

    tag: str
    type: str
    name: str
    label: str
    selector: str
    text: str
    href: str


class DomTokensDict(TypedDict):
    """Every ``id`` / ``class`` token / ``data-testid`` value in the rendered DOM.

    The inventory a CSS selector is GROUNDED against (``step_def_generator``'s
    assertion-grounding pass). ``interactive_elements`` cannot serve: it keeps one
    selector per control and drops the id/class of every element that got a
    role/placeholder selector, and it never sees non-interactive elements at all.
    ``*_truncated`` counts tokens dropped at ``_MAX_DOM_TOKENS``; a truncated bucket
    is NOT evidence of absence and the consumer must treat it as unknown.
    """

    ids: list[str]
    classes: list[str]
    testids: list[str]
    ids_truncated: int
    classes_truncated: int
    testids_truncated: int


class PageStructure(TypedDict):
    """Structured inspection derived from a page's rendered HTML."""

    headings: list[str]
    has_forms: bool
    interactive_elements: list[InteractiveElementDict]
    page_type: str
    elements_truncated: int
    headings_truncated: int
    #: ``None`` when the DOM was NOT fully scanned (empty html, parser failure) —
    #: "not looked" must never read as "looked and found nothing".
    dom_tokens: DomTokensDict | None


# Tags we surface as interactive elements the walkthrough can act on.
_INTERACTIVE_TAGS = frozenset({"a", "button", "input", "select", "textarea"})
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
# Skip data inside these — never part of the visible/accessible page.
_SKIP_TEXT_TAGS = frozenset({"script", "style", "template", "noscript"})
# Element cap: a walkthrough opener needs the salient controls, not every link
# on a link-heavy page. Callers log when truncation happens.
_MAX_ELEMENTS = 40
# Heading cap: enough for a page-structure hint; truncation past this is logged
# (never silent), matching how the element cap is surfaced.
_MAX_HEADINGS = 20
# Per-bucket cap on recorded DOM tokens: bounds the JSONB row on a class-heavy page
# (utility-CSS apps carry thousands). Past the cap the bucket is reported truncated.
_MAX_DOM_TOKENS = 2000
# Accessible names are capped at this many characters in ``finalize()``. EXPORTED
# because a consumer that judges "is this name in the inventory?" must know a
# recorded name may be a PREFIX of the real one — a substring test against a
# truncated name yields a false ABSENT. See ``step_def_generator``'s grounding pass.
LABEL_LENGTH_CAP = 120
# ARIA roles we are willing to emit into getByRole(...). A role must be a plain
# lowercase word to be usable; anything else (attribute injection, CSS-y junk)
# falls back to a locator() selector.
_ROLE_RE = re.compile(r"^[a-z]+$")


def _clean(value: str) -> str:
    """Collapse all whitespace (incl. newlines/tabs) to single spaces + trim."""
    return " ".join(value.split())


def _escape_js(value: str) -> str:
    """Escape a (already whitespace-collapsed) value for a double-quoted JS arg."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _escape_css_attr(value: str) -> str:
    """Escape a value that sits inside a CSS double-quoted attribute selector which is
    itself inside a double-quoted JS string — e.g. ``locator("[id=\\"<value>\\"]")``.

    TWO layers: a literal ``"`` (or ``\\``) in the value must be escaped for the CSS
    string AND then for the JS string, or the emitted selector is invalid CSS that
    throws at runtime (an ``id``/``class``/``name`` containing a quote is rare but real).
    Applying the JS escape twice yields exactly the CSS-then-JS encoding.
    """
    return _escape_js(_escape_js(value))


def _role_for(tag: str, attrs: dict[str, str]) -> str:
    """Best-effort ARIA role for a tag, restricted to a safe lowercase token."""
    explicit = attrs.get("role", "").strip().lower()
    if explicit:
        return explicit if _ROLE_RE.match(explicit) else ""
    if tag == "a":
        return "link" if attrs.get("href") else ""
    if tag == "button":
        return "button"
    if tag == "select":
        return "combobox"
    if tag == "textarea":
        return "textbox"
    if tag == "input":
        itype = attrs.get("type", "text").strip().lower()
        if itype in {"submit", "button", "reset", "image"}:
            return "button"
        if itype == "hidden":
            return ""
        # text / email / tel / url / password / and unknowns → textbox
        return {
            "checkbox": "checkbox",
            "radio": "radio",
            "search": "searchbox",
            "number": "spinbutton",
            "range": "slider",
        }.get(itype, "textbox")
    return ""


def _selector_for(tag: str, attrs: dict[str, str], label: str, role: str) -> str:
    """Build a resilient, accessibility-first Playwright locator expression.

    Order mirrors what robust generated steps already use: placeholder for
    inputs, role+accessible-name for named controls, then id / class / name via
    CSS *attribute* selectors (which tolerate ``:`` ``/`` spaces in the value,
    unlike a bare ``#id`` / ``.class``). Every embedded value is cleaned +
    escaped so the emitted expression is always valid, non-injectable JS.
    """
    placeholder = _clean(attrs.get("placeholder", ""))
    if tag in {"input", "textarea"} and placeholder:
        return f'getByPlaceholder("{_escape_js(placeholder)}")'
    if role and _ROLE_RE.match(role) and label:
        # exact: true — getByRole name matching is substring + case-insensitive by
        # default, so a short accessible name ("TodoMVC") collides with a longer one
        # ("real TodoMVC app.") and trips Playwright strict mode -> blank clip -> failed
        # render. Exact, full-string matching resolves to the intended element.
        return f'getByRole("{role}", {{ name: "{_escape_js(label)}", exact: true }})'
    elem_id = _clean(attrs.get("id", ""))
    if elem_id:
        return f'locator("[id=\\"{_escape_css_attr(elem_id)}\\"]")'
    classes = attrs.get("class", "").split()
    if classes:
        # ~= matches one whitespace-separated class token, tolerating "/" etc.
        return f'locator("{tag}[class~=\\"{_escape_css_attr(classes[0])}\\"]")'
    name = _clean(attrs.get("name", ""))
    if name:
        return f'locator("{tag}[name=\\"{_escape_css_attr(name)}\\"]")'
    # A visible-text fallback beats a bare ``locator("<tag>")``, which matches EVERY
    # such tag on the page (a Playwright strict-mode violation / wrong-element action).
    if label:
        return f'getByText("{_escape_js(label)}")'
    return f'locator("{tag}")'


class _StructureParser(HTMLParser):
    """Streaming extractor for headings, forms, and interactive elements.

    Text capture uses INDEPENDENT active buffers (heading / element-text /
    label) so nested inline tags don't stomp each other — e.g. an ``<a>`` inside
    an ``<h1>`` contributes to both the heading text and its own element text.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.headings: list[str] = []
        self.has_forms = False
        self._elements: list[dict[str, Any]] = []
        self.labels_for: dict[str, str] = {}
        self._skip_depth = 0
        # Stack of (tag, index-into-self.headings) so a `role="heading"` element on an
        # arbitrary tag can open a heading and be closed by its own end tag.
        self._heading_stack: list[tuple[str, int]] = []
        self._element_text_idx: int | None = None
        # Wrapping-label state: text inside a <label> + the elements nested in it.
        self._label_depth = 0
        self._label_buf = ""
        self._label_for = ""
        self._label_elem_idxs: list[int] = []
        # DOM token inventory (see `DomTokensDict`). Sets: order is irrelevant and a
        # repeated class must not spend the cap.
        self.dom_ids: set[str] = set()
        self.dom_classes: set[str] = set()
        self.dom_testids: set[str] = set()

    def _record_dom_tokens(self, attr_map: dict[str, str]) -> None:
        """Fold one element's ``id`` / ``class`` tokens / ``data-testid`` into the
        DOM token inventory. Values are kept VERBATIM (no case folding): CSS id,
        class and attribute matching are case-sensitive in standards mode."""
        elem_id = attr_map.get("id", "").strip()
        if elem_id:
            self.dom_ids.add(elem_id)
        self.dom_classes.update(attr_map.get("class", "").split())
        testid = attr_map.get("data-testid", "").strip()
        if testid:
            self.dom_testids.add(testid)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TEXT_TAGS:
            self._skip_depth += 1
            return
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if not self._skip_depth:
            # `<template>` children are parsed as tags but never rendered.
            self._record_dom_tokens(attr_map)
        if tag == "form":
            self.has_forms = True
        if tag in _HEADING_TAGS or attr_map.get("role", "").strip().lower() == "heading":
            self._open_heading(tag, attr_map)
            if tag in _HEADING_TAGS:
                return
        if tag == "img":
            # `<h1><img alt="Logo"></h1>`: the heading's accessible name comes from the
            # image's alt text, which `handle_data` can never see (img has no data).
            self._append_heading_text(_clean(attr_map.get("alt", "")))
        if tag == "label":
            self._label_depth += 1
            if self._label_depth == 1:
                self._label_buf = ""
                self._label_for = attr_map.get("for", "").strip()
                self._label_elem_idxs = []
            return
        if tag in _INTERACTIVE_TAGS:
            self._add_interactive(tag, attr_map)

    def _add_interactive(self, tag: str, attr_map: dict[str, str]) -> None:
        """Record one interactive element and arm text/label capture for it."""
        itype = attr_map.get("type", "").strip().lower()
        if tag == "input" and itype == "hidden":
            return
        if tag in {"input", "textarea", "select", "button"}:
            self.has_forms = True
        attr_label = _clean(
            attr_map.get("aria-label", "")
            or attr_map.get("placeholder", "")
            or attr_map.get("value", "")
            or attr_map.get("title", "")
            or attr_map.get("alt", "")
        )
        idx = len(self._elements)
        self._elements.append(
            {
                "tag": tag,
                "type": itype,
                "name": _clean(attr_map.get("name", "")),
                "href": attr_map.get("href", "").strip(),
                "id": attr_map.get("id", "").strip(),
                "role": _role_for(tag, attr_map),
                "attr_label": attr_label,
                "text": "",
                "attrs": attr_map,
            }
        )
        if self._label_depth:
            self._label_elem_idxs.append(idx)
        # Capture inner text for elements whose accessible name is their
        # content (links, buttons).
        if tag in {"a", "button"}:
            self._element_text_idx = idx

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TEXT_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._heading_stack and self._heading_stack[-1][0] == tag:
            self._heading_stack.pop()
        if tag in _HEADING_TAGS:
            return
        if tag in {"a", "button"}:
            self._element_text_idx = None
        elif tag == "label" and self._label_depth:
            self._label_depth -= 1
            if self._label_depth == 0:
                text = _clean(self._label_buf)
                if self._label_for:
                    self.labels_for[self._label_for] = text
                # Wrapping label (no `for`, or `for` plus nested inputs): assign
                # its text to the elements nested inside it.
                for i in self._label_elem_idxs:
                    self._elements[i].setdefault("wrap_label", text)
                self._label_elem_idxs = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        self._append_heading_text(text)
        if self._element_text_idx is not None:
            el = self._elements[self._element_text_idx]
            el["text"] = (el["text"] + " " + text).strip()
        if self._label_depth:
            self._label_buf = (self._label_buf + " " + text).strip()

    def _open_heading(self, tag: str, attr_map: dict[str, str]) -> None:
        """Push a heading capture for ``tag``.

        `aria-label` IS the accessible name and it overrides the inner text, so record
        BOTH as heading names: dropping the text would make a correct assertion on the
        visible text look ungrounded, and dropping the aria-label is the miss that
        motivated this (the a11y name is not the rendered text).
        """
        aria = _clean(attr_map.get("aria-label", ""))
        if aria:
            self.headings.append(aria)
        self.headings.append("")
        self._heading_stack.append((tag, len(self.headings) - 1))

    def _append_heading_text(self, text: str) -> None:
        """Fold ``text`` into the innermost open heading, if any."""
        if not text or not self._heading_stack:
            return
        idx = self._heading_stack[-1][1]
        self.headings[idx] = (self.headings[idx] + " " + text).strip()

    def finalize(self) -> list[InteractiveElementDict]:
        """Resolve accessible names + selectors into adapter-shaped element dicts."""
        out: list[InteractiveElementDict] = []
        seen: set[tuple[str, str, str, str]] = set()
        for el in self._elements:
            # Accessible name: aria/placeholder/value/title/alt, then visible
            # text (links/buttons), then a <label for=id> / wrapping <label>.
            label = (
                el["attr_label"]
                or _clean(el["text"])
                or self.labels_for.get(el["id"], "")
                or el.get("wrap_label", "")
            )
            label = label[:LABEL_LENGTH_CAP]
            selector = _selector_for(el["tag"], el["attrs"], label, el["role"])
            text = _clean(el["text"])[:LABEL_LENGTH_CAP]
            key = (el["tag"], selector, label, el["href"])
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "tag": el["tag"],
                    "type": el["type"],
                    "name": el["name"],
                    "label": label,
                    "selector": selector,
                    "text": text,
                    "href": el["href"],
                }
            )
        return out


def _infer_page_type(has_forms: bool, elements: list[InteractiveElementDict]) -> str:
    """Coarse page classification used only as an LLM hint."""
    inputs = sum(1 for e in elements if e["tag"] in {"input", "textarea", "select"})
    links = sum(1 for e in elements if e["tag"] == "a")
    if inputs >= 2 or (has_forms and inputs >= 1):
        return "form"
    if links >= 8 and inputs == 0:
        return "listing"
    return "content"


def extract_structure(html: str) -> PageStructure:
    """Parse rendered HTML into structured inspection fields.

    Returns a dict with ``headings`` (list[str]), ``has_forms`` (bool),
    ``interactive_elements`` (list of ``{tag,type,name,label,selector,text,
    href}``), ``page_type`` (str), ``elements_truncated`` (int — how many
    interactive elements were dropped by the ``_MAX_ELEMENTS`` cap) and
    ``headings_truncated`` (the same count for the ``_MAX_HEADINGS`` cap). Both
    counts MUST be carried into the inspection page dict: a consumer that reads
    the capped list as an exhaustive inventory would treat "not in the first 40"
    as "not on the page". Empty / non-string HTML yields empty structure (no regression
    versus the prior thin inspection).
    """
    if not html or not isinstance(html, str):
        return {
            "headings": [],
            "has_forms": False,
            "interactive_elements": [],
            "page_type": "content",
            "elements_truncated": 0,
            "headings_truncated": 0,
            "dom_tokens": None,
        }
    parser = _StructureParser()
    elements: list[InteractiveElementDict] = []
    scanned = False
    try:
        parser.feed(html)
        parser.close()
        # finalize() lives INSIDE the guard: name/selector resolution must also
        # degrade to a partial structure, never crash the calling job.
        elements = parser.finalize()
        scanned = True
    except Exception:
        logger.debug("html structure parse failed; returning partial structure", exc_info=True)
    truncated = max(0, len(elements) - _MAX_ELEMENTS)
    elements = elements[:_MAX_ELEMENTS]
    all_headings = [h for h in (parser.headings or []) if h]
    headings_truncated = max(0, len(all_headings) - _MAX_HEADINGS)
    if headings_truncated:
        logger.debug("html structure: %d headings, capping to %d", len(all_headings), _MAX_HEADINGS)
    headings = all_headings[:_MAX_HEADINGS]
    return {
        "headings": headings,
        "has_forms": parser.has_forms,
        "interactive_elements": elements,
        "page_type": _infer_page_type(parser.has_forms, elements),
        "elements_truncated": truncated,
        "headings_truncated": headings_truncated,
        # A partial parse is not an inventory: absence in it proves nothing.
        "dom_tokens": _dom_tokens(parser) if scanned else None,
    }


def _capped(tokens: set[str]) -> tuple[list[str], int]:
    """``tokens`` sorted (deterministic row) and capped, plus how many were dropped."""
    ordered = sorted(tokens)
    return ordered[:_MAX_DOM_TOKENS], max(0, len(ordered) - _MAX_DOM_TOKENS)


def _dom_tokens(parser: _StructureParser) -> DomTokensDict:
    """The parser's DOM token inventory, capped per bucket."""
    ids, ids_truncated = _capped(parser.dom_ids)
    classes, classes_truncated = _capped(parser.dom_classes)
    testids, testids_truncated = _capped(parser.dom_testids)
    return {
        "ids": ids,
        "classes": classes,
        "testids": testids,
        "ids_truncated": ids_truncated,
        "classes_truncated": classes_truncated,
        "testids_truncated": testids_truncated,
    }
