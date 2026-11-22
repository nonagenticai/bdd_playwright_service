"""Deterministic, snapshot-grounded QA features + their validator.

Why this module exists
----------------------
The QA stream used to hand the LLM two jobs at once — invent the scenarios AND write
the TypeScript steps. It did both badly: features asserted flows the app has no UI
for (24h retention, email confirmation, expiry), and the step file shipped
``createBd()``, an ``await`` in a non-async arrow, and quoted ``"{string}"``
placeholders that can never match. Nothing type-checked it (the service has no Node
runtime), and nothing checked a claim against the snapshot.

So the QA stream now has a grounded mode with two halves:

* :func:`build_grounded_features` emits page-contract and captured-link navigation
  scenarios built ONLY from the inspection structure. Deterministic — same snapshot,
  same suite. This is the guaranteed floor.
* :func:`validate_features` is the gate every feature (deterministic or the opt-in
  LLM enrichment) must pass: a step outside the closed vocabulary, a heading/link/
  button/title/text the captured page does not contain, an absolute URL, or an
  assertion made before any page was opened gets the SCENARIO rejected. Rejects are
  reported, never silently dropped, and a suite with nothing left raises.

Honest scope: page contracts and navigation between captured pages. Business-state
flows cannot be grounded in a static snapshot and stay hand-authored.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from src.services.generation.qa_vocabulary import (
    QaStepSpec,
    match_step,
    vocabulary_prompt,
)
from src.services.gherkin_keywords import SCENARIO_OPENER_RE, iter_steps

#: How many landmark controls a page-contract scenario asserts. Three is enough to
#: prove the page rendered its furniture without turning every scenario into a link
#: inventory.
_MAX_LANDMARKS = 3
#: One navigation scenario per captured page: the first link that stays on the
#: captured site and lands on another CAPTURED path.
_MAX_NAV_PER_PAGE = 1
#: Values longer than this are not useful assertion labels and make the feature
#: unreadable; the structure extractor caps labels at 120 already.
_MAX_ARG_LEN = 160
_MAX_REJECTED_IN_REPORT = 20

#: A generated feature record — same shape as ``feature_generator._Feature``.
Feature = dict[str, Any]
#: The grounding report that rides on the job row and in ``validation.json``.
Report = dict[str, Any]

_OPEN_STEP = "the user opens the page {string}"
_CLICK_STEP = "the user clicks the link {string}"


@dataclass(frozen=True)
class GroundedPage:
    """One captured page, reduced to what a grounded assertion may use."""

    url: str
    path: str
    title: str
    headings: tuple[str, ...]
    elements: tuple[Mapping[str, Any], ...]
    text: str
    elements_truncated: int
    headings_truncated: int


@dataclass
class _Scenario:
    name: str
    tags: list[str]
    steps: list[str]


class GroundedSuiteError(RuntimeError):
    """No scenario survived grounding — the package would contain no executable test."""

    def __init__(self, message: str, report: Report | None = None) -> None:
        super().__init__(message)
        #: The full validation report, so a caller (or a test) can still read exactly
        #: which scenarios were rejected and why.
        self.report: Report = report or {}


def _norm(value: object) -> str:
    return " ".join(str(value or "").split())


def _norm_key(value: object) -> str:
    return _norm(value).casefold()


def _page_path(url: str) -> str | None:
    """Site-relative path of a captured URL; ``None`` for a non-web URL."""
    if not url:
        return None
    split = urlsplit(url)
    if split.scheme and split.scheme not in {"http", "https"}:
        return None
    return split.path or "/"


def _element_role(element: Mapping[str, Any]) -> str:
    role = str(element.get("role") or "")
    if role:
        return role
    tag = str(element.get("tag") or "")
    if tag == "a":
        return "link"
    if tag == "button":
        return "button"
    if tag in {"input", "select", "textarea"}:
        return ""  # inputs without an explicit role are not link/button landmarks
    return ""


def _build_pages(inspection: Mapping[str, Any]) -> dict[str, GroundedPage]:
    pages: dict[str, GroundedPage] = {}
    for raw in inspection.get("pages") or []:
        if not isinstance(raw, Mapping):
            continue
        url = str(raw.get("url") or "")
        path = _page_path(url)
        if path is None or path in pages:
            continue
        headings = tuple(h for h in (_norm(x) for x in raw.get("headings") or []) if h)
        elements = tuple(
            el for el in raw.get("interactive_elements") or [] if isinstance(el, Mapping)
        )
        pages[path] = GroundedPage(
            url=url,
            path=path,
            title=_norm(raw.get("title")),
            headings=headings,
            elements=elements,
            text=_norm(raw.get("text_content") or raw.get("text")),
            elements_truncated=_as_count(raw.get("elements_truncated")),
            headings_truncated=_as_count(raw.get("headings_truncated")),
        )
    return pages


def _as_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _slug(path: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", path.strip("/").lower()).strip("_")
    return cleaned or "root"


def _unique_filename(path: str, seen: set[str]) -> str:
    base = f"page_contract_{_slug(path)}"
    filename = f"{base}.feature"
    suffix = 2
    while filename in seen:
        filename = f"{base}_{suffix}.feature"
        suffix += 1
    seen.add(filename)
    return filename


def _arg_ok(value: str) -> str | None:
    """A step argument must be short, quote-free and site-relative."""
    if not value:
        return "empty step argument"
    if len(value) > _MAX_ARG_LEN:
        return f"step argument is longer than {_MAX_ARG_LEN} characters"
    if '"' in value or "'" in value:
        return "step argument contains a quote"
    if "://" in value or value.startswith("//"):
        return f"absolute URL in a step: {value!r}"
    return None


def _landmarks(page: GroundedPage) -> list[tuple[str, str]]:
    """Up to ``_MAX_LANDMARKS`` (role, label) pairs from the captured inventory."""
    picked: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for element in page.elements:
        role = _element_role(element)
        if role not in {"link", "button"}:
            continue
        label = _norm(element.get("label"))
        if not label or len(label) > _MAX_ARG_LEN:
            continue
        key = (role, label)
        if key in seen:
            continue
        seen.add(key)
        picked.append(key)
        if len(picked) >= _MAX_LANDMARKS:
            break
    return picked


def _href_path(source: GroundedPage, href: str) -> str | None:
    """The captured path an ``href`` lands on, or ``None``.

    Same-path and query-only links are skipped: ``?lang=en`` re-renders the page in
    another language (its heading is not the captured one), and ``?day=…`` changes
    state without proof the asserted content survives. v1 only follows links to
    another CAPTURED path.
    """
    href = href.strip()
    if not href or href.startswith("#") or href.startswith("?"):
        return None
    split = urlsplit(href)
    if split.scheme and split.scheme not in {"http", "https"}:
        return None
    if split.netloc:
        source_host = urlsplit(source.url).netloc
        if split.netloc != source_host:
            return None
    path = split.path or "/"
    if path == source.path:
        return None
    return path


def _navigation_scenario(page: GroundedPage, pages: Mapping[str, GroundedPage]) -> _Scenario | None:
    for element in page.elements:
        if _element_role(element) != "link":
            continue
        label = _norm(element.get("label"))
        if not label or len(label) > _MAX_ARG_LEN:
            continue
        target_path = _href_path(page, str(element.get("href") or ""))
        if target_path is None or target_path not in pages:
            continue
        target = pages[target_path]
        steps = [
            f'Given the user opens the page "{page.path}"',
            f'When the user clicks the link "{label}"',
            f'Then the page URL path is "{target.path}"',
        ]
        if target.headings:
            steps.append(f'Then the page shows the heading "{target.headings[0]}"')
        return _Scenario(
            name=f'the "{label}" link on {page.path} opens {target.path}',
            tags=["navigation", "grounded"],
            steps=steps,
        )
    return None


def _contract_scenario(page: GroundedPage) -> _Scenario:
    """The page-contract scenario: the page still opens and still shows what it showed."""
    steps = [
        f'Given the user opens the page "{page.path}"',
        f'Then the page URL path is "{page.path}"',
    ]
    if page.headings:
        steps.append(f'Then the page shows the heading "{page.headings[0]}"')
    if page.title:
        steps.append(f'Then the page title contains "{page.title}"')
    for role, label in _landmarks(page):
        noun = "link" if role == "link" else "button"
        steps.append(f'Then the page contains the {noun} "{label}"')
    return _Scenario(
        name=f"the {page.path} page still loads and shows its captured content",
        tags=["smoke", "grounded"],
        steps=steps,
    )


def _render_feature(page: GroundedPage, scenarios: list[_Scenario]) -> str:
    lines = [
        f"Feature: Page contract {page.path}",
        "  Generated from the captured page snapshot. It asserts only content that",
        "  snapshot proves exists; it does not invent business flows.",
        "",
    ]
    for scenario in scenarios:
        if scenario.tags:
            lines.append("  " + " ".join(f"@{tag}" for tag in scenario.tags))
        lines.append(f"  Scenario: {scenario.name}")
        for step in scenario.steps:
            lines.append(f"    {step}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_grounded_features(
    inspection: Mapping[str, Any], config: Mapping[str, Any]
) -> list[Feature]:
    """Deterministic page-contract + navigation features from the inspection.

    One feature per captured page; one contract scenario per page and at most one
    navigation scenario. Raises when the inspection carries no usable page — a QA
    package with zero features would complete green and run nothing.
    """
    pages = _build_pages(inspection)
    if not pages:
        raise GroundedSuiteError(
            "grounded QA needs at least one captured page with a web URL; the inspection "
            "carried none"
        )
    max_scenarios = config.get("max_scenarios_per_feature", 10)
    if not isinstance(max_scenarios, int) or isinstance(max_scenarios, bool):
        max_scenarios = 10
    max_scenarios = max(1, min(max_scenarios, 50))

    features: list[Feature] = []
    seen_filenames: set[str] = set()
    for page in pages.values():
        scenarios = [_contract_scenario(page)]
        if max_scenarios > len(scenarios):
            nav = _navigation_scenario(page, pages)
            if nav is not None:
                scenarios.append(nav)
        scenarios = scenarios[:max_scenarios]
        if not scenarios:
            continue
        features.append(
            {
                "filename": _unique_filename(page.path, seen_filenames),
                "content": _render_feature(page, scenarios),
                "scenario_count": len(scenarios),
                "tags": sorted({tag for scenario in scenarios for tag in scenario.tags}),
            }
        )
    if not features:
        raise GroundedSuiteError("grounded QA produced no feature files from the captured pages")
    return features


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class _Block:
    name: str
    lines: list[str]


def _split_feature(content: str) -> tuple[list[str], list[_Block]]:
    """Split a feature into its header (Feature/description/tags) and scenario blocks.

    Each block carries the tag lines immediately above its ``Scenario:`` — dropping a
    rejected scenario must not leave its ``@smoke`` tag attached to the next one.
    """
    lines = content.splitlines()
    header: list[str] = []
    blocks: list[_Block] = []
    pending_tags: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("@"):
            pending_tags.append(lines[i])
            i += 1
            continue
        opener = SCENARIO_OPENER_RE.match(stripped)
        if opener:
            name = opener.group("name") or "(unnamed scenario)"
            block_lines = [*pending_tags, lines[i]]
            pending_tags = []
            i += 1
            while i < len(lines):
                inner = lines[i].strip()
                if inner.startswith("@") or SCENARIO_OPENER_RE.match(inner):
                    break
                block_lines.append(lines[i])
                i += 1
            blocks.append(_Block(name=name, lines=block_lines))
            continue
        if pending_tags:
            header.extend(pending_tags)
            pending_tags = []
        header.append(lines[i])
        i += 1
    header.extend(pending_tags)
    return header, blocks


def _find_element(page: GroundedPage, role: str, label: str) -> Mapping[str, Any] | None:
    wanted = _norm(label)
    for element in page.elements:
        if _element_role(element) != role:
            continue
        if _norm(element.get("label")) == wanted:
            return element
    return None


def _check_page_path(value: str, pages: Mapping[str, GroundedPage]) -> str | None:
    path = value if value.startswith("/") else f"/{value}"
    if path not in pages:
        return f"page path {value!r} was not captured in the snapshot"
    return None


def _check_heading(value: str, page: GroundedPage) -> str | None:
    if _norm(value) in page.headings:
        return None
    if page.headings_truncated:
        return None  # capped inventory: absence proves nothing
    return f"heading {value!r} is not on the captured {page.path} page"


def _check_title(value: str, page: GroundedPage) -> str | None:
    if page.title and _norm(value) in page.title:
        return None
    return f"title fragment {value!r} is not the captured title of {page.path}"


def _check_element_claim(kind: str, value: str, page: GroundedPage) -> str | None:
    if _find_element(page, kind, value) is not None:
        return None
    if page.elements_truncated:
        return None  # capped inventory: absence proves nothing
    return f"{kind} {value!r} is not on the captured {page.path} page"


def _check_link(value: str, page: GroundedPage) -> str | None:
    return _check_element_claim("link", value, page)


def _check_button(value: str, page: GroundedPage) -> str | None:
    return _check_element_claim("button", value, page)


def _check_text(value: str, page: GroundedPage) -> str | None:
    if page.text and _norm(value) in page.text:
        return None
    return f"text {value!r} is not in the captured text of {page.path}"


_CLAIM_CHECKS = {
    "heading": _check_heading,
    "title": _check_title,
    "link": _check_link,
    "button": _check_button,
    "text": _check_text,
}


def _check_argument(
    spec: QaStepSpec,
    kind: str,
    value: str,
    current: GroundedPage | None,
    pages: Mapping[str, GroundedPage],
) -> str | None:
    """Return a rejection reason for one grounded argument, or ``None``."""
    arg_error = _arg_ok(value)
    if arg_error:
        return arg_error
    if kind == "page_path":
        return _check_page_path(value, pages)
    if current is None:
        return f"step {spec.keyword} {spec.text!r} asserts on a page before any page was opened"
    check = _CLAIM_CHECKS.get(kind)
    if check is None:
        return f"unknown grounding kind {kind!r}"
    return check(value, current)


def _link_target(
    page: GroundedPage, label: str, pages: Mapping[str, GroundedPage]
) -> GroundedPage | None:
    element = _find_element(page, "link", label)
    if element is None:
        return None
    target_path = _href_path(page, str(element.get("href") or ""))
    return pages.get(target_path) if target_path else None


def _validate_block(block: _Block, pages: Mapping[str, GroundedPage]) -> tuple[str | None, int]:
    """Validate one scenario block. Returns ``(reason | None, step_count)``."""
    steps = list(iter_steps("\n".join(block.lines)))
    if not steps:
        return "scenario has no steps", 0
    current: GroundedPage | None = None
    for step in steps:
        matched = match_step(step.binding, step.text)
        if matched is None:
            return (
                f"step outside the closed QA vocabulary: {step.keyword} {step.text}",
                len(steps),
            )
        spec, values = matched
        for kind, value in zip(spec.args, values, strict=True):
            reason = _check_argument(spec, kind, value, current, pages)
            if reason:
                return f"{reason} (step: {step.keyword} {step.text})", len(steps)
        if spec.keyword == "Given" and spec.text == _OPEN_STEP:
            path = values[0] if values[0].startswith("/") else f"/{values[0]}"
            current = pages[path]
        elif spec.text == _CLICK_STEP:
            current = _link_target(current, values[0], pages) if current else None
    return None, len(steps)


def validate_features(
    features: list[Feature], inspection: Mapping[str, Any]
) -> tuple[list[Feature], Report]:
    """Drop scenarios that are not grounded in the snapshot; report every reject.

    Returns ``(features, report)``. A feature with no surviving scenario is removed.
    Raises :class:`GroundedSuiteError` when nothing survives — shipping an empty
    suite as "completed" is the silent-zero failure this service exists to prevent.
    """
    pages = _build_pages(inspection)
    kept_features: list[Feature] = []
    rejected: list[Report] = []
    checked = 0

    for feature in features:
        content = str(feature.get("content") or "")
        header, blocks = _split_feature(content)
        if any(re.match(r"^\s*Background\s*:", line) for line in header):
            rejected.append(
                {
                    "feature": feature.get("filename", ""),
                    "scenario": "(whole feature)",
                    "reason": "Background is not supported in grounded QA mode",
                }
            )
            continue
        kept_blocks: list[_Block] = []
        for block in blocks:
            checked += 1
            reason, _ = _validate_block(block, pages)
            if reason is None:
                kept_blocks.append(block)
            else:
                rejected.append(
                    {
                        "feature": feature.get("filename", ""),
                        "scenario": block.name,
                        "reason": reason,
                    }
                )
        if not kept_blocks:
            continue
        body: list[str] = [*header]
        for block in kept_blocks:
            if body and body[-1].strip():
                body.append("")
            body.extend(block.lines)
        new_content = "\n".join(body).rstrip() + "\n"
        kept_features.append(
            {
                **feature,
                "content": new_content,
                "scenario_count": len(kept_blocks),
            }
        )

    report: Report = {
        "checked_scenarios": checked,
        "kept_scenarios": sum(int(f.get("scenario_count") or 0) for f in kept_features),
        "rejected_count": len(rejected),
        "rejected": rejected[:_MAX_REJECTED_IN_REPORT],
        "rejected_truncated": max(0, len(rejected) - _MAX_REJECTED_IN_REPORT),
        "vocabulary": vocabulary_prompt().splitlines(),
    }
    if not kept_features:
        reasons = "; ".join(f"{r['scenario']}: {r['reason']}" for r in rejected[:5]) or "none"
        raise GroundedSuiteError(
            "every generated scenario failed grounding; nothing executable remains "
            f"({len(rejected)} rejected: {reasons})",
            report=report,
        )
    return kept_features, report
