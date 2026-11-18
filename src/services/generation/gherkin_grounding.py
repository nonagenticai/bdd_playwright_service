"""Flag counter/badge concepts a demo ``Then`` step names that nothing else backs.

The demo Gherkin is narrated verbatim, so a scenario asserting on a concept the app does
not have (a "completed count" on TodoMVC) puts an invented feature into the voiceover —
and #166's selector grounding only catches the step CODE, never the Gherkin TEXT.

The snapshot is PRE-INTERACTION and the scrape holds only the rendered DOM: a SPA's
bundle (TodoMVC ships ``js/bundle.js``, no inline script) is never fetched, so UI that
appears only after an action — "Clear completed", the "Completed" filter, "items left" —
is simply absent from every carrier bdd has. Absence is therefore NOT evidence of
invention for most claim shapes, and a re-ask telling the model real UI is missing would
push it to delete real features. So the check is deliberately NARROW:

* ``<modifier> count / counter / badge / label / panel`` phrases, AND quoted strings on a
  ``Then`` step, are judged. A first cut judged label/button/tab/filter nouns and quoted
  strings against the page corpus ALONE and every flag it raised on a TodoMVC sample was
  real hidden UI (4/4 false positives) — that cut is what was dropped, not the concept.
  The difference here is the SECOND grounding path below, which a bare corpus check
  didn't have;
* a claim's words are grounded by the page corpus (visible text, title, headings, element
  names, role names, id/class/testid words) OR by ANY step text anywhere in the suite
  outside the judged phrase itself. A real counter or a real hidden control is normally
  described elsewhere too ("2 items left" ~ "the item is marked", "Clear completed" named
  in both a Given/When and a Then); an invented one usually is not. This is also why
  ``button``/``tab``/``filter`` stay OUT of the judged noun list — TodoMVC's real filter
  names ("Active", "Completed") are common generic/repeated words far more often than an
  invented label value is, so the false-positive risk for those nouns was not re-measured
  and they are left alone;
* a finding only feeds the EXISTING one-shot re-ask, worded to keep UI that appears after
  an action, and is RECORDED if it survives; nothing is ever dropped here.

Tradeoff (measured in the commit report): zero false positives on the plausible-lines
sample, at the cost of recall on ONE known shape — an invented counter whose modifier
word the suite uses anywhere else ("When I mark it as completed … Then the completed
count shows 1") is NOT flagged, because a same-scenario mention is indistinguishable
from a genuine cross-reference under this rule.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.database.models import UngroundedGherkinConcept
from src.services.generation.step_def_generator import (
    _Feature,
    _inventory_dom_tokens,
    _inventory_role_names,
)
from src.services.gherkin_keywords import SCENARIO_OPEN_RE, STEP_LINE_RE

_QUOTED_RE = re.compile(r"\"[^\"]*\"|\u201c[^\u201d]*\u201d")
_UI_NOUNS = ("count", "counter", "badge", "label", "panel")
_UI_CONCEPT_RE = re.compile(
    r"\b(?P<modifier>[A-Za-z][A-Za-z-]*)\s+(?P<noun>" + "|".join(_UI_NOUNS) + r")s?\b",
    re.IGNORECASE,
)
# Modifiers that name no page concept of their own ("the button", "a new tab").
_GENERIC_MODIFIERS = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "its",
        "their",
        "my",
        "your",
        "our",
        "each",
        "every",
        "any",
        "some",
        "one",
        "first",
        "second",
        "third",
        "last",
        "next",
        "previous",
        "new",
        "same",
        "other",
        "main",
        "primary",
        "secondary",
        "active",
        "selected",
        "current",
        "visible",
        "open",
        "browser",
        "and",
        "or",
        "of",
        "on",
        "in",
        "to",
        "with",
        "see",
        "shows",
        "show",
        "is",
        "has",
        "should",
    }
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_STEM_SUFFIX_RE = re.compile(r"(?:ing|ed|es|s|e)$")
# Whole-suite sample bound — one pathological step must not flood the correction prompt.
_MAX_FINDINGS = 20


def _stem(word: str) -> str:
    stemmed = _STEM_SUFFIX_RE.sub("", word)
    return stemmed if len(stemmed) >= 3 else word


def _words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD_RE.findall(_CAMEL_RE.sub(" ", text).casefold())}


@dataclass(frozen=True)
class PageCorpus:
    """Everything the snapshot proves is on the page, as a stemmed word set."""

    words: set[str]


def _page_parts(page: Mapping[str, Any]) -> list[str] | None:
    """One page's visible text, title, headings and element names; ``None`` without text."""
    text = page.get("text_content")
    if not isinstance(text, str) or not text.strip():
        return None
    parts = [text, str(page.get("title") or "")]
    headings = page.get("headings")
    if isinstance(headings, list):
        parts.extend(h for h in headings if isinstance(h, str))
    elements = page.get("interactive_elements")
    for el in elements if isinstance(elements, list) else []:
        if isinstance(el, dict):
            parts.extend(str(el.get(key) or "") for key in ("label", "text", "name", "placeholder"))
    return parts


def build_page_corpus(inspection: Mapping[str, Any] | None) -> PageCorpus | None:
    """The page corpus, or ``None`` — judge nothing — when any page carries no visible text.

    Without text for EVERY page, absence from the corpus is not evidence of absence (a
    ``gherkin_upload``/passthrough inspection has only structure).
    """
    if not isinstance(inspection, dict):
        return None
    pages = inspection.get("pages")
    if not isinstance(pages, list) or not pages:
        return None
    parts: list[str] = []
    for page in pages:
        page_parts = _page_parts(page) if isinstance(page, dict) else None
        if page_parts is None:
            return None
        parts.extend(page_parts)
    # #166's inventories: role names (normalised) and id/class/testid tokens, whose words
    # ground a concept the static DOM carries but does not render (a hidden `.todo-count`).
    for names in _inventory_role_names(inspection).values():
        parts.extend(names)
    token_words: list[str] = []
    for tokens in _inventory_dom_tokens(inspection).values():
        token_words.extend(t.replace("-", " ").replace("_", " ") for t in tokens)
    return PageCorpus(_words(" ".join(parts + token_words)))


def _scenarios(content: str) -> Iterable[tuple[str, list[tuple[str, str]], list[str]]]:
    """``(scenario title, [(binding, step text)], background step texts)`` per scenario."""
    background: list[str] = []
    title: str | None = None
    in_background = False
    steps: list[tuple[str, str]] = []
    last_primary = "Given"
    for raw in content.splitlines():
        line = raw.strip()
        if SCENARIO_OPEN_RE.match(line):
            if title is not None:
                yield title, steps, background
            in_background = line.startswith("Background")
            title = None if in_background or line.startswith(("Feature", "Rule")) else line
            steps = []
            last_primary = "Given"
            continue
        match = STEP_LINE_RE.match(line)
        if not match:
            continue
        keyword, text = match.group("keyword"), match.group("text").strip()
        if keyword in {"Given", "When", "Then"}:
            last_primary = keyword
        if in_background:
            background.append(text)
        elif title is not None:
            steps.append((last_primary, text))
    if title is not None:
        yield title, steps, background


#: A quoted phrase's stopword-ish/one- or two-letter words say nothing on their own
#: ("the", "to", numerals) — judging them would flag real quoted data for no reason.
#: Reuses `_GENERIC_MODIFIERS` rather than a second list: the same words that name no
#: page concept as a `<modifier> count` also carry no claim inside quotes.
def _quoted_content_words(raw: str) -> set[str]:
    return {w for w in _words(raw) if w not in _GENERIC_MODIFIERS and len(w) >= 3}


def _quoted_findings(text: str, grounded: set[str], suite_words: dict[str, int]) -> list[str]:
    """Quoted phrases in ``text`` whose content words nothing else backs.

    Same dual grounding as ``_step_findings``: a quoted phrase is safe when the page
    corpus has it OR when the suite mentions it (as words) somewhere other than this
    one quoted occurrence — which is exactly how real hidden-until-action text survives
    ("Clear completed" is also named in a Given/When step elsewhere in the walkthrough).
    An invented aside like "Productivity insights" appears nowhere else and is flagged.
    """
    findings: list[str] = []
    for m in _QUOTED_RE.finditer(text):
        raw = m.group(0).strip('"“”')
        words = _quoted_content_words(raw)
        if not words:
            continue
        if words <= grounded or all(suite_words.get(w, 0) > 1 for w in words):
            continue
        findings.append(raw)
    return findings


def _step_findings(text: str, grounded: set[str], suite_words: dict[str, int]) -> list[str]:
    """``<modifier> <noun>`` phrases plus ungrounded quoted text in ``text``.

    ``suite_words`` counts every stemmed word across the suite's step text; the judged
    phrase's own occurrence is subtracted (a word count of 1 means the claim only ever
    grounds itself), so a claim cannot ground itself.
    """
    findings: list[str] = []
    for m in _UI_CONCEPT_RE.finditer(_QUOTED_RE.sub(" ", text)):
        modifier = m.group("modifier").casefold()
        if modifier in _GENERIC_MODIFIERS:
            continue
        words = _words(modifier)
        if words <= grounded or all(suite_words.get(w, 0) > 1 for w in words):
            continue
        findings.append(f"{m.group('modifier')} {m.group('noun')}")
    findings.extend(_quoted_findings(text, grounded, suite_words))
    return findings


def _suite_word_counts(features: list[_Feature]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for feat in features:
        for _title, steps, background in _scenarios(str(feat.get("content", ""))):
            for text in background + [t for _binding, t in steps]:
                for word in _WORD_RE.findall(_CAMEL_RE.sub(" ", text).casefold()):
                    stem = _stem(word)
                    counts[stem] = counts.get(stem, 0) + 1
    return counts


def find_ungrounded_concepts(
    features: list[_Feature], corpus: PageCorpus | None
) -> list[UngroundedGherkinConcept]:
    """Every judged ``Then`` counter/badge claim neither the page nor the suite backs."""
    if corpus is None:
        return []
    suite_words = _suite_word_counts(features)
    out: list[UngroundedGherkinConcept] = []
    for feat in features:
        fname = str(feat.get("filename", "?"))
        for title, steps, _background in _scenarios(str(feat.get("content", ""))):
            for binding, text in steps:
                if binding != "Then":
                    continue
                for concept in _step_findings(text, corpus.words, suite_words):
                    finding: UngroundedGherkinConcept = {
                        "file": fname,
                        "scenario": title,
                        "step": text,
                        "concept": concept,
                    }
                    out.append(finding)
                if len(out) >= _MAX_FINDINGS:
                    return out[:_MAX_FINDINGS]
    return out


GROUNDING_CORRECTION = (
    "\n\nCORRECTION — these Then steps name a counter or badge that neither the page "
    "snapshot nor any other step mentions: {problems}. The scenarios are narrated to "
    "viewers, so an invented counter becomes a false claim in the video. If the app really "
    "shows it after an action in that scenario, keep it and make that action explicit in "
    "a When step; if the app does not have it, assert on something the app does show "
    "instead. Do NOT remove real controls, filters or text that only appear after an "
    "action. Keep every other rule.\n"
)


def grounding_correction(findings: list[UngroundedGherkinConcept]) -> str:
    """The re-ask suffix for ``findings``; empty when there is nothing to correct."""
    if not findings:
        return ""
    problems = "; ".join(f"{f['file']}: '{f['step']}' ({f['concept']})" for f in findings[:6])
    return GROUNDING_CORRECTION.format(problems=problems)
