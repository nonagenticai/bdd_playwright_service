"""Generate Gherkin .feature files from application inspection data."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from src.services.generation.gherkin_grounding import (
    PageCorpus,
    build_page_corpus,
    find_ungrounded_concepts,
    grounding_correction,
)
from src.services.generation.llm_client import LLMClient
from src.services.gherkin_keywords import (
    EXAMPLES_BLOCK_RE,
    OUTLINE_KEYWORDS,
    SCENARIO_OPENER_RE,
    normalize_opener_keyword,
)

logger = logging.getLogger(__name__)

# A parsed Gherkin feature record (``{"content": <feature text>, …}``) — same shape as
# ``step_def_generator._Feature``.
_Feature = dict[str, Any]
# A single scraped/inspected page (``{"title", "url", "page_type", "text_content", …}``).
_Page = dict[str, Any]
# The full multi-page inspection payload (``{"pages": [...], "navigation_graph": [...]}``).
_Inspection = dict[str, Any]
# Generation config as handed down from the job (``{"focus_areas", "max_scenarios", …}``).
_Config = dict[str, Any]
# One mobile-app screen record, same shape as ``_Page``.
_Screen = dict[str, Any]

# Visible page text handed to the model so it describes what the page SHOWS instead of
# inventing counters/features. Bounded per page and in total: an inspection can carry up
# to 20 screens, and 20 x 1500 chars (~7.5k tokens) would crowd out the rest of the prompt.
_PAGE_TEXT_CHARS = 1500
_TOTAL_PAGE_TEXT_CHARS = 6000


def _visible_text(page: _Page, remaining: int) -> str:
    """``page``'s scraped text, whitespace-collapsed and truncated to the smaller bound."""
    text = page.get("text_content")
    if not isinstance(text, str):
        return ""
    collapsed = " ".join(text.split())
    limit = min(_PAGE_TEXT_CHARS, remaining)
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)].rstrip() + "…"


def _build_page_summary(inspection: _Inspection) -> str:
    """Build a human-readable summary of inspected pages and elements."""
    lines: list[str] = []
    pages = inspection.get("pages", [])
    base_url = inspection.get("base_url", "")
    lines.append(f"Application URL: {base_url}")
    lines.append(f"Platform: {inspection.get('platform', 'web')}")
    lines.append(f"Total pages discovered: {len(pages)}")
    lines.append("")

    text_budget = _TOTAL_PAGE_TEXT_CHARS
    for page in pages:
        lines.append(f"Page: {page.get('title', 'Untitled')} ({page.get('url', '')})")
        lines.append(f"  Type: {page.get('page_type', 'unknown')}")
        lines.append(f"  Has forms: {page.get('has_forms', False)}")

        elements = page.get("interactive_elements", [])
        if elements:
            lines.append(f"  Interactive elements ({len(elements)}):")
            for elem in elements:
                tag = elem.get("tag", "")
                label = elem.get("label") or elem.get("text") or elem.get("name") or ""
                selector = elem.get("selector", "")
                lines.append(f"    - {tag}: {label} [{selector}]")

        headings = page.get("headings", [])
        if headings:
            lines.append(f"  Headings: {', '.join(headings)}")
        visible = _visible_text(page, text_budget) if text_budget > 0 else ""
        if visible:
            text_budget -= len(visible)
            lines.append(f"  Visible text: {visible}")
        lines.append("")

    nav = inspection.get("navigation_graph", [])
    if nav:
        lines.append("Navigation:")
        for edge in nav:
            lines.append(
                f"  {edge.get('from_page')} -> {edge.get('to_page')} "
                f"via {edge.get('action')} ({edge.get('element_label', '')})"
            )

    return "\n".join(lines)


# The scenario-opener pattern is IMPORTED, not restated here. It used to be a local regex
# and the narration parser had its own, narrower one (``Scenario:``/``Scenario Outline:``
# only) — so an ``Example:`` scenario got a ``@clip-<n>`` tag and a recorded video but NO
# narration part, and every later clip's voiceover shifted onto the wrong screen. The two
# ends of the join spine now match with the SAME compiled object.
# ``_enforce_scenario_cap`` warns instead of silently doing nothing when it finds no opener.
_SCENARIO_OPENER_RE = SCENARIO_OPENER_RE

_DOCSTRING_DELIMS = ('"""', "'''")

# C0 controls + DEL: whitespace-like bytes ``str.split()`` does not collapse.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")

_FOCUS_AREAS_HEADER = (
    "\nFOCUS AREAS — the caller named these areas of the application explicitly; "
    "the generated scenarios MUST cover every one of them:\n"
)


def _focus_areas_instruction(config: _Config) -> str:
    """Render the caller's ``focus_areas`` as a labelled prompt block.

    ``focus_areas`` is accepted on the demo-video, QA-suite and accessibility request
    schemas and forwarded by ``w2v_orchestrator``; this is what makes it steer generation
    instead of terminating unread in the job config.

    Returns ``""`` when the caller supplied none (missing, not a list, or empty/blank
    entries) so a request WITHOUT focus areas produces a byte-identical prompt to the
    pre-focus-areas one — no stray header, no trailing blank line.

    Items are sanitised, not stringified. Only genuine ``str`` items survive (``str(item)``
    turned ``None``/``{"a": 1}``/``7`` into the literal bullets ``- None`` / ``- {'a': 1}`` /
    ``- 7``), and every run of whitespace/control characters inside an item collapses to a
    single space so ONE item can only ever produce ONE bullet line — a newline-bearing item
    otherwise injected raw lines straight into the prompt.
    """
    areas = _focus_areas(config)
    if not areas:
        return ""
    return _FOCUS_AREAS_HEADER + "".join(f"- {area}\n" for area in areas)


def _docstring_state(stripped: str, current: str | None) -> str | None:
    """Advance the Gherkin doc-string state machine by one (already-stripped) line.

    ``current`` is the delimiter that opened the doc string we are inside, or ``None``.

    A Gherkin doc string may legally carry a CONTENT TYPE on its opening line
    (``\"\"\"markdown``, ``\"\"\"json``) — the old ``stripped in ('\"\"\"', \"'''\")`` toggle
    never fired on those, so everything after such a line was treated as ordinary Gherkin
    and a quoted ``Scenario:`` inside it counted as a real scenario opener. The CLOSING
    delimiter, by contrast, is always bare and must match the opener, so while we are
    inside a doc string only an exactly-equal bare line terminates it (a body line that
    merely starts with ``\"\"\"`` is literal content). ``'''`` behaves symmetrically.
    """
    if current is not None:
        return None if stripped == current else current
    for delim in _DOCSTRING_DELIMS:
        if stripped.startswith(delim):
            # An inline ``\"\"\"text\"\"\"`` opens and closes on one line — never enter.
            if len(stripped) >= 2 * len(delim) and stripped.endswith(delim):
                return None
            return delim
    return None


def _scenario_start_lines(lines: Sequence[str]) -> list[int]:
    """Line indexes of every real scenario opener (``Scenario``/``Outline``/``Example``).

    Doc-string-aware via :func:`_docstring_state` — the same helper :func:`_stamp_clip_tags`
    uses, so the two can never disagree about what a scenario is: a ``Scenario:`` quoted
    inside a ``\"\"\"`` block is literal content, not an opener.
    """
    starts: list[int] = []
    delim: str | None = None
    for index, line in enumerate(lines):
        previous, delim = delim, _docstring_state(line.strip(), delim)
        if previous is not None or delim is not None:
            continue  # a doc-string boundary line or its literal body
        if _SCENARIO_OPENER_RE.match(line):
            starts.append(index)
    return starts


def _trim_to_cap(content: str, cap: int) -> str | None:
    """Trim ``content`` to its first ``cap`` scenarios, or ``None`` if already under it.

    The cut lands ABOVE the tag/comment block of the first dropped scenario so no orphan
    ``@clip-<n>``/``@smoke`` tag line survives without the scenario it annotated —
    an orphan tag is exactly the kind of stray line bddgen rejects. A ``#`` comment sitting
    between the tags and the ``Scenario:`` used to stop the walk-back and strand the tag
    above it. Everything the kept scenarios own (steps, data tables, ``Examples:``,
    doc strings) is preserved verbatim, so the result is still valid Gherkin.
    """
    lines = content.split("\n")
    starts = _scenario_start_lines(lines)
    if len(starts) <= cap:
        return None
    cut = starts[cap]
    while cut > 0 and _is_droppable_preamble(lines[cut - 1]):
        cut -= 1
    return "\n".join(lines[:cut]).rstrip()


def _is_droppable_preamble(line: str) -> bool:
    """A line that belongs to the FOLLOWING scenario's preamble: blank, tag, or comment."""
    stripped = line.strip()
    return not stripped or stripped.startswith(("@", "#"))


def _coerce_cap(value: object) -> int:
    """Coerce ``max_scenarios_per_feature`` to an int, or ``0`` (= no cap) if impossible.

    The value comes straight out of a caller-supplied job config, so it is whatever JSON
    carried: ``"3"``, ``None``, ``3.0`` or nonsense. Before the cap was enforced it was only
    interpolated into a prompt, so a non-int generated fine; a bare ``value < 1`` comparison
    would now raise ``TypeError`` on all four streams. Degrade to "no cap", never raise.
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


def _enforce_scenario_cap(features: list[_Feature], max_scenarios: object) -> list[_Feature]:
    """Enforce ``max_scenarios_per_feature`` as a real cap, not prompt prose.

    The limit is interpolated into every generation prompt, but an LLM routinely
    overshoots it; nothing downstream ever compared the parsed ``scenario_count``
    against the cap, so the cap neither capped nor floored. Trim in place and keep
    ``scenario_count``/``tags`` metadata consistent with the trimmed content.

    This runs AFTER ``_revalidate_features`` and deliberately stays there: revalidation
    REPLACES the feature list wholesale with ``_parse_features(retry_output)``, so a trim
    applied before it would simply be discarded on the retry path. The cost is that trimmed
    content is never re-checked by ``_feature_issues`` — so the trim re-checks ITSELF here
    and abandons any cut that would introduce a NEW syntax problem. An over-cap feature is a
    cosmetic overrun; unparseable Gherkin aborts the entire bddgen render, so a trim must
    never be the thing that emits it.
    """
    cap = _coerce_cap(max_scenarios)
    if cap < 1:
        return features
    for feat in features:
        content = str(feat.get("content", ""))
        fname = str(feat.get("filename", "?"))
        starts = _scenario_start_lines(content.split("\n"))
        if not starts:
            if content.strip():
                logger.warning(
                    "Feature file %s: no Scenario:/Scenario Outline:/Example: opener found, so "
                    "the max_scenarios_per_feature cap of %d could NOT be enforced (localized "
                    "or unsupported Gherkin keyword?)",
                    fname,
                    cap,
                )
            continue
        trimmed = _trim_to_cap(content, cap)
        if trimmed is None:
            continue
        _, before_problems = _feature_issues([{"filename": fname, "content": content}])
        _, after_problems = _feature_issues([{"filename": fname, "content": trimmed}])
        if len(after_problems) > len(before_problems):
            logger.error(
                "Feature file %s: trimming to the max_scenarios_per_feature cap of %d would "
                "emit invalid Gherkin (%s); keeping the over-cap content instead",
                fname,
                cap,
                "; ".join(after_problems[:3]),
            )
            continue
        after = len(_scenario_start_lines(trimmed.split("\n")))
        logger.warning(
            "Feature file %s has %d scenarios, over the max_scenarios_per_feature cap of "
            "%d; trimmed to the first %d",
            fname,
            len(starts),
            cap,
            after,
        )
        feat["content"] = trimmed
        feat["scenario_count"] = max(after, 1)
        feat["tags"] = re.findall(r"@(\S+)", trimmed)
    return features


def _focus_areas(config: _Config) -> list[str]:
    """The caller's ``focus_areas``, sanitised — the single source both the prompt
    block and the scenario floor are derived from, so they can never disagree."""
    raw: object = config.get("focus_areas")
    if not isinstance(raw, list):
        return []
    areas: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        if text := " ".join(_CONTROL_CHARS_RE.sub(" ", item).split()):
            areas.append(text)
    return areas


def _total_scenarios(features: list[_Feature]) -> int:
    """Scenarios across ALL feature files.

    Counted globally, not per file, because the demo stream's clip spine is global:
    ``_stamp_clip_tags`` numbers ``@clip-<n>`` in document order across every feature,
    and the downstream shot-count assertion counts DISTINCT clip ids for the whole run.
    """
    return sum(len(_scenario_start_lines(str(f.get("content", "")).split("\n"))) for f in features)


def _scenario_floor(config: _Config) -> int:
    """Minimum scenarios the caller's own inputs already imply, or 0 if none do.

    ``max_scenarios_per_feature`` is a CAP and has never implied a minimum — the count
    was otherwise purely the model's choice. Two caller inputs justify a floor, and only
    these two are enforced here:

    1. ``focus_areas`` — the prompt states the scenarios "MUST cover every one of them",
       so fewer scenarios than focus areas means the model ignored an explicit
       instruction. A detectable, correctable defect rather than a taste difference.
    2. ``min_scenarios`` — an explicit minimum the caller names outright. Added because
       the cap being the ONLY count signal is a measured supply-side defect: the
       orchestrator asks for a scenario count that is only a ceiling, and the model
       routinely lands far under it (asked 15, generated 5 — a 27.26s film against a
       60s target), so the few recorded scenarios get fanned onto many narration beats
       and the delivered video cycles the same visuals. A caller that knows how many
       distinct visuals its beat plan needs can now say so, instead of hoping the model
       volunteers them.

    The two combine as a ``max`` — each is independently a statement about the minimum,
    so satisfying the larger satisfies both.

    Clamped to the cap exactly as before, so the floor can never fight the trim: 5 focus
    areas (or ``min_scenarios=5``) capped at 2 still yields 2. An absent, zero or
    unparseable cap means "no cap" and leaves the floor untouched. A non-positive or
    unparseable ``min_scenarios`` contributes nothing, so existing callers are unchanged.
    """
    # _coerce_cap is a generic "JSON int or 0" degrade, not cap-specific: min_scenarios
    # arrives from the same caller-supplied job config and carries the same "3"/None/3.0
    # hazard, and must likewise degrade to "no floor" rather than raise mid-render.
    floor = max(len(_focus_areas(config)), _coerce_cap(config.get("min_scenarios", 0)))
    if floor < 1:
        return 0
    cap = _coerce_cap(config.get("max_scenarios_per_feature", 0))
    return min(floor, cap) if cap >= 1 else floor


_SCENARIO_FLOOR_CORRECTION = (
    "\n\nCORRECTION — the previous response returned only {actual} scenario(s) in total, "
    # Deliberately does NOT attribute the floor to the FOCUS AREAS block: since
    # _scenario_floor also honours a caller-supplied `min_scenarios`, a floor can exceed
    # the number of focus areas, or exist with no FOCUS AREAS block in the prompt at all.
    # Naming that block as the source would then state a falsehood to the model.
    "but this request requires at least {floor} in total. "
    "Regenerate the feature file(s) with AT LEAST {floor} scenarios, covering every "
    "focus area named above if any are listed, keeping every other requirement "
    "above unchanged.\n"
)


def _gherkin_quote(value: str) -> str:
    """Wrap ``value`` as a double-quoted Cucumber ``{string}`` argument.

    Selectors routinely contain double quotes (attribute selectors like
    ``input[name="title"]``), which would prematurely close the argument, so
    escape backslashes and double quotes the way the built-in ``{string}``
    parameter type expects.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


_PLACEHOLDER_RE = re.compile(r"<([^<>]*)>")
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|(.*)\|[ \t]*$")


def _table_cells(line: str) -> list[str] | None:
    """The trimmed cells of a Gherkin table row, or ``None`` if it is not one."""
    match = _TABLE_ROW_RE.match(line)
    if match is None:
        return None
    return [cell.strip() for cell in match.group(1).split("|")]


def _substitute(text: str, row: dict[str, str]) -> str:
    """Replace every ``<column>`` placeholder in ``text`` with that row's value."""
    return _PLACEHOLDER_RE.sub(lambda m: row.get(m.group(1).strip(), m.group(0)), text)


def _split_outline_body(body: list[str]) -> tuple[list[str], list[str]] | None:
    """Split an outline's body into ``(step template, Examples tail)``.

    The tail starts at the tag/comment/blank preamble of the FIRST ``Examples:`` line that
    is real Gherkin — a doc string quoting ``Examples:`` is literal content, so the same
    :func:`_docstring_state` machine the stamper uses drives the scan. Returns ``None``
    when the body has no ``Examples:`` block at all (there is nothing to expand: such an
    outline generates no tests, so it cannot mis-join a clip either).
    """
    delim: str | None = None
    for index, line in enumerate(body):
        previous, delim = delim, _docstring_state(line.strip(), delim)
        if previous is not None or delim is not None:
            continue
        if EXAMPLES_BLOCK_RE.match(line):
            start = index
            while start > 0 and _is_droppable_preamble(body[start - 1]):
                start -= 1
            return body[:start], body[start:]
    return None


def _examples_rows(tail: list[str]) -> list[dict[str, str]] | None:
    """Every ``Examples:`` data row in ``tail`` as a column->value map, in document order.

    bddgen generates one test — one VIDEO — per data row across ALL of an outline's
    ``Examples:`` blocks, so all of them are collected, not just the first.

    Returns ``None`` the moment the tail is anything but a clean run of ``Examples:``
    blocks and their tables: an escaped ``\\|`` cell (which naive splitting would corrupt),
    a doc string, a ragged row, a table with no data rows, or any other line. Refusing is
    the point — a half-understood outline must be left ALONE and reported, never rewritten
    into Gherkin that bddgen then rejects.
    """
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    seen_examples = False
    for line in tail:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            # A comment carries nothing downstream, so dropping it is cosmetic.
            continue
        if stripped.startswith("@"):
            # A TAG is not cosmetic. Gherkin allows tags on an ``Examples:`` block,
            # and the whole tail is DISCARDED on a successful expansion -- so
            # skipping one here deletes it silently. If it was ``@skip``/``@wip``,
            # rows the author excluded would start recording; if it was a selection
            # tag, they would join the wrong stream. That is wrong OUTPUT, which is
            # worse than the mis-join this expansion exists to fix. Refuse and
            # report, exactly as for any other tail line we do not fully understand.
            return None
        if EXAMPLES_BLOCK_RE.match(line):
            seen_examples = True
            header = None
            continue
        cells = _table_cells(line)
        if cells is None or not seen_examples or "\\|" in line:
            return None
        if header is None:
            header = cells
            continue
        if len(cells) != len(header):
            return None
        rows.append(dict(zip(header, cells, strict=True)))
    return rows or None


def _expand_one_outline(preamble: list[str], opener: str, body: list[str]) -> list[str] | None:
    """Rewrite one ``Scenario Outline:`` as one concrete ``Scenario:`` per Examples row.

    Returns ``None`` — meaning "leave this scenario byte-for-byte alone" — for anything
    that is not an outline, and for any outline this function does not fully understand.

    The preamble tags are REPLICATED above every generated scenario: the demo stream
    selects clips by ``@demo``/``@walkthrough``, so an expansion that left rows 2..N
    untagged would silently drop them from the recording run.
    """
    match = SCENARIO_OPENER_RE.match(opener)
    if match is None or normalize_opener_keyword(match.group("keyword")) not in OUTLINE_KEYWORDS:
        return None
    split = _split_outline_body(body)
    if split is None:
        return None
    template, tail = split
    rows = _examples_rows(tail)
    if not rows:
        return None
    # Every placeholder must resolve in EVERY row, or the rewrite would ship a literal
    # ``<username>`` into a step. A step that legitimately contains angle brackets (an
    # HTML tag) reads as an unresolvable placeholder here and correctly blocks expansion.
    placeholders = {
        m.group(1).strip()
        for line in [match.group("name"), *template]
        for m in _PLACEHOLDER_RE.finditer(line)
    }
    if any(placeholders - row.keys() for row in rows):
        return None

    indent = match.group("indent")
    kept_preamble = [line for line in preamble if line.strip()]
    while template and not template[-1].strip():
        template.pop()

    # Keep the blank line that separated this outline from the scenario above it, so the
    # expansion reads like hand-written Gherkin rather than running into the previous steps.
    out: list[str] = [""] if preamble and not preamble[0].strip() else []
    used_names: set[str] = set()
    for index, row in enumerate(rows, start=1):
        name = _substitute(match.group("name"), row).strip()
        # Two rows whose name carries no placeholder would produce two identically-named
        # scenarios — legal Gherkin, but the narration for both clips would be word-for-word
        # identical and unreviewable. Disambiguate with the row ordinal.
        if name in used_names:
            name = f"{name} [{index}]" if name else f"[{index}]"
        used_names.add(name)
        if index > 1:
            out.append("")
        out.extend(kept_preamble)
        out.append(f"{indent}Scenario: {name}".rstrip())
        out.extend(_substitute(line, row) for line in template)
    return out


def _expand_scenario_outlines(block: str) -> str:
    """Expand every well-formed ``Scenario Outline:`` into one scenario per Examples row.

    THE JOIN-SPINE FIX. ``@clip-<n>`` is stamped once per scenario OPENER, and the
    narration parser produces one part per opener — but bddgen generates one test, and
    therefore one recorded VIDEO, per ``Examples:`` ROW. An unexpanded three-row outline
    is three videos sharing one clip id and one narration line: the join is ambiguous, and
    anything resolving it positionally shifts every later clip's voiceover onto the wrong
    screen. Expanding BEFORE :func:`_stamp_clip_tags` restores 1 scenario = 1 tag = 1
    narration part = 1 video, and makes ``max_scenarios_per_feature`` count real clips
    instead of templates.

    Conservative by construction: any outline :func:`_expand_one_outline` does not fully
    understand is left untouched, and :func:`_outline_join_problems` then reports it as a
    defect rather than letting it ship silently mis-joined.
    """
    lines = block.split("\n")
    starts = _scenario_start_lines(lines)
    if not starts:
        return block

    # Each scenario owns its preamble (the contiguous tag/comment/blank run above it), so
    # the tags travel with the scenario when it is rewritten into N copies.
    bounds: list[tuple[int, int]] = []
    floor = 0
    for start in starts:
        pre = start
        while pre > floor and _is_droppable_preamble(lines[pre - 1]):
            pre -= 1
        bounds.append((pre, start))
        floor = start + 1

    out: list[str] = list(lines[: bounds[0][0]])
    for index, (pre, start) in enumerate(bounds):
        end = bounds[index + 1][0] if index + 1 < len(bounds) else len(lines)
        expanded = _expand_one_outline(lines[pre:start], lines[start], lines[start + 1 : end])
        out.extend(expanded if expanded is not None else lines[pre:end])
    return "\n".join(out)


def _outline_join_problems(content: str, fname: str) -> list[str]:
    """Report every ``Scenario Outline:`` that still owns an ``Examples:`` table.

    :func:`_expand_scenario_outlines` has already rewritten every outline it understood,
    so a survivor here is one it refused — and a refused outline is N videos sharing ONE
    ``@clip-<n>`` id. That is the silent wrong-voiceover-on-wrong-screen failure, so it is
    raised as a regeneration-worthy defect: loud beats mis-joined.
    """
    problems: list[str] = []
    lines = content.split("\n")
    starts = _scenario_start_lines(lines)
    for index, start in enumerate(starts):
        match = SCENARIO_OPENER_RE.match(lines[start])
        if match is None:
            continue
        if normalize_opener_keyword(match.group("keyword")) not in OUTLINE_KEYWORDS:
            continue
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        if _split_outline_body(lines[start + 1 : end]) is None:
            continue
        problems.append(
            f"{fname}: {match.group('keyword')}: {match.group('name')[:40]!r} keeps an "
            f"Examples: table — bddgen records one video per row, but a scenario carries "
            f"ONE @clip-<n> tag and one narration line, so the clips mis-join. Write each "
            f"case as its own Scenario: instead."
        )
    return problems


def _stamp_clip_tags(block: str, start_index: int) -> tuple[str, int]:
    """Stamp a deterministic ``@clip-<n>`` tag above each scenario in ``block``.

    The tag is the rename-proof join spine across the whole demo-video pipeline:
    narration parts, recorded Playwright clips, and compile shots all key off the
    same ``clip-<n>`` id. ``n`` is a 1-based counter assigned in document order,
    threaded across feature files via ``start_index`` so it is globally unique
    and stable for one generation request.

    A scenario that already carries an ``@clip-<n>`` tag is left untouched (so
    re-stamping is idempotent). Returns ``(stamped_block, next_index)``.
    """
    counter = start_index
    out_lines: list[str] = []
    delim: str | None = None
    for line in block.split("\n"):
        # A ``"""`` doc-string body is literal content — a ``Scenario:`` QUOTED inside one
        # (plausible on the documentation stream) must NOT be stamped, and must not inflate
        # the global counter the whole narration/clip join-spine keys off. Shares
        # :func:`_docstring_state` with :func:`_scenario_start_lines` so the two cannot
        # disagree; that shared helper is what makes ``"""markdown``-style content-typed
        # openers count as openers here too.
        previous, delim = delim, _docstring_state(line.strip(), delim)
        if previous is not None or delim is not None:
            out_lines.append(line)
            continue
        match = _SCENARIO_OPENER_RE.match(line)
        if match:
            # Idempotent re-stamp: scan the CONTIGUOUS run of tag/blank lines directly above
            # this scenario for an existing ``@clip-`` — not just the single preceding line,
            # which double-stamps when another tag (``@smoke``) sits between the clip tag and
            # the scenario.
            already = False
            for prev in reversed(out_lines):
                prev_s = prev.strip()
                if not prev_s:
                    continue
                if prev_s.startswith("@"):
                    already = already or "@clip-" in prev_s
                    continue
                break  # a non-tag, non-blank line ends the scenario's tag block
            if not already:
                counter += 1
                out_lines.append(f"{match.group('indent')}@clip-{counter}")
        out_lines.append(line)
    return "\n".join(out_lines), counter


# The prompt asks for a ```gherkin fence, but the model routinely answers with a
# different tag (```feature / ```cucumber) or a capitalised ```Gherkin. The original
# pattern accepted ONLY a lowercase `gherkin`/bare tag, so any other tag matched
# NOTHING. With ONE such block `findall` returned [] and the whole raw markdown —
# fence markers and prose included — became a single "feature" that bddgen rejects
# with a Parse error; with a MIXED pair the pattern instead matched the GAP BETWEEN
# the blocks, so the second feature was silently DROPPED and the job reported
# `features_generated` short with no exception. Strip ANY language tag,
# case-insensitively — the same fix already shipped as
# `step_def_generator._CODE_FENCE_RE`; this is the back-port.
_CODE_FENCE_RE = re.compile(r"```[A-Za-z0-9_+#-]*[ \t]*\r?\n(.*?)```", re.DOTALL)


def _parse_features(raw: str) -> list[_Feature]:
    """Parse LLM output into individual feature file dicts.

    Extracts code blocks containing Gherkin features and returns metadata
    for each one. Each scenario is stamped with a deterministic ``@clip-<n>``
    tag (the demo-video join spine) in document order across all features.
    """
    features: list[_Feature] = []

    # Try to extract fenced code blocks first
    blocks = _CODE_FENCE_RE.findall(raw)
    if not blocks:
        # Fallback: treat the whole response as a single feature
        blocks = [raw]

    clip_counter = 0
    seen_filenames: set[str] = set()
    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue

        # Expand Scenario Outlines FIRST: bddgen records one video per Examples row, so an
        # unexpanded outline would put N videos behind ONE @clip-<n> tag and one narration
        # line. Stamping only ever sees concrete, 1-video scenarios.
        block = _expand_scenario_outlines(block)

        # Stamp @clip-<n> tags (global document order) before anything reads them.
        block, clip_counter = _stamp_clip_tags(block, clip_counter)

        # Extract feature name
        name_match = re.search(r"Feature:\s*(.+)", block)
        feature_name = name_match.group(1).strip() if name_match else f"feature_{i}"

        # Build filename from feature name. Two feature titles can sanitize to the SAME slug
        # (e.g. "Settings" vs "Settings!!" → "settings"), which would silently OVERWRITE one
        # `.feature` file in package_builder / storage — so de-duplicate with a numeric suffix.
        slug = re.sub(r"[^a-z0-9]+", "_", feature_name.lower()).strip("_") or f"feature_{i}"
        filename = f"{slug}.feature"
        dedup = 2
        while filename in seen_filenames:
            filename = f"{slug}_{dedup}.feature"
            dedup += 1
        seen_filenames.add(filename)

        # Count scenarios (doc-string-aware, and via the same opener set the cap uses)
        scenario_count = len(_scenario_start_lines(block.split("\n")))

        # Extract tags
        tags = re.findall(r"@(\S+)", block)

        features.append(
            {
                "filename": filename,
                "content": block,
                "scenario_count": max(scenario_count, 1),
                "tags": tags,
            }
        )

    return features


# Gherkin structural keywords for a line-scan validity check (no parser dep). A step line
# must open with one of Given/When/Then/And/But/* — the #1 LLM mistake (bare prose on a
# step line) is exactly what makes cucumber's bddgen abort the whole render with a
# "Parse error", the same peer failure that broke the demo/relief renders this session.
_FEATURE_RE = re.compile(r"^Feature\s*:")
_EXAMPLES_RE = re.compile(r"^Examples\s*:")
# ``Scenario``/``Scenario Outline``/``Scenario Template``/``Example:`` open a scenario body;
# ``Background:`` behaves like one for the step-position scan. ``Example[ \t]*:`` cannot match
# the plural ``Examples:`` data block, which stays a legal in-scenario line.
_SCENARIO_KW_RE = re.compile(r"^(?:Scenario\b|Background\b|Example[ \t]*:)")
_RULE_RE = re.compile(r"^Rule\b")
_STEP_RE = re.compile(r"^(Given|When|Then|And|But|\*)\b")

_GHERKIN_SYNTAX_CORRECTION = (
    "\n\nCORRECTION — your previous Gherkin has a SYNTAX error that cucumber's bddgen "
    "rejects with a 'Parse error', which aborts the ENTIRE demo render: {problems}. Rules: "
    "every line inside a Scenario/Scenario Outline/Background must be EITHER a step opening "
    "with Given/When/Then/And/But (or '*'), OR a data-table row ('| ... |'), OR inside a "
    'triple-quoted """ docstring. NEVER put bare prose on a step line — prefix it with a step '
    'keyword. Close every """ docstring. Every feature needs a \'Feature:\' line. Do NOT use '
    "'Scenario Outline:' with an 'Examples:' table — each row would be recorded as a "
    "separate video clip but they would all share one narration line, so write every case "
    "as its own plain 'Scenario:'. Regenerate valid Gherkin only.\n"
)


def _is_step_table_or_tag(stripped: str) -> bool:
    """A legal non-structural line inside a scenario: a step, an Examples header, a data-
    table row, or a tag. Anything else in step position is a missing-keyword parse error."""
    return bool(_STEP_RE.match(stripped) or _EXAMPLES_RE.match(stripped)) or stripped.startswith(
        ("@", "|")
    )


def _is_skippable_in_scenario(stripped: str) -> bool:
    """A non-missing-keyword line inside a scenario: a step/table/tag/Examples header, OR a
    docstring boundary/inline docstring (a line CONTAINING ``\"\"\"``/``'''`` — e.g. an inline
    ``\"\"\"text\"\"\"`` the plain toggle misses — is docstring content, never a step error)."""
    return _is_step_table_or_tag(stripped) or '"""' in stripped or "'''" in stripped


def _next_meaningful_line(lines: list[str], start: int) -> str | None:
    """The next line after ``start`` that Gherkin treats as structural.

    Blank lines, comments and further tag lines are transparent — a tag block may stack
    several tags before the thing it annotates. Returns ``None`` at end of file.
    """
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "@")):
            continue
        return stripped
    return None


def _misplaced_tag_problem(lines: list[str], index: int, stripped: str, fname: str) -> str | None:
    """Flag a tag line that annotates nothing — a bddgen-fatal shape the old scan missed.

    A tag is legal ONLY directly above a Feature/Rule/Scenario/Background/Example opener
    (blank lines, comments and other tags may sit between). Anywhere else — most often
    between a ``Scenario:`` line and its first step — Gherkin switches to expecting
    ``#TagLine/#RuleLine/#Comment/#Empty`` and then aborts on the step that follows.

    Observed live 2026-08-04 (calibration render 019fcb30, sweep 30878780752): the model
    emitted ``@demo`` on its own line between ``Scenario:`` and its first ``When``, and
    bddgen died with three ``Parse error … got 'When I click on the "Active" filter
    option'`` lines, failing the whole paid render. The scan could not see it: tags are
    "skippable in a scenario" (they legally separate one scenario from the next, where
    ``in_scenario`` is still True from the previous one), so position — not the line
    itself — is the only thing that distinguishes legal from fatal. Hence the lookahead.

    Also catches a trailing orphan tag at EOF, which annotates nothing at all.
    """
    following = _next_meaningful_line(lines, index + 1)
    if following is None:
        return f"{fname}: tag line {stripped[:40]!r} annotates nothing (end of file)"
    if (
        _FEATURE_RE.match(following)
        or _RULE_RE.match(following)
        or _SCENARIO_KW_RE.match(following)
        # Tagging an `Examples:` block is legal Gherkin — do not call it misplaced.
        or _EXAMPLES_RE.match(following)
    ):
        return None
    return (
        f"{fname}: tag line {stripped[:40]!r} is not directly above a "
        f"Scenario/Background/Rule — it aborts bddgen at the following line "
        f"{following[:40]!r}"
    )


def _scan_one_feature(content: str, fname: str) -> list[str]:
    """Missing-keyword problems for one feature's Gherkin (docstring-aware).

    Tracks Gherkin triple-quote docstring state so literal docstring prose is never mistaken
    for a step. Flags TWO bddgen-aborting shapes: (1) a non-step line INSIDE a
    Scenario/Background (missing step keyword), and (2) a STEP-KEYWORD line OUTSIDE any
    scenario — a ``Given/When/Then/And/But/*`` before the first ``Scenario:``/``Background:``
    (or directly under ``Feature:``/``Rule:``). The parser is at feature/rule level there and
    aborts with ``expected #TagLine/#RuleLine/#Comment/#Empty, got 'Given …'``. Non-step free
    text before the first scenario is still legal (a Feature description) — only an actual step
    keyword is flagged.
    """
    problems: list[str] = []
    delim: str | None = None
    in_scenario = False
    lines = content.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        previous, delim = delim, _docstring_state(stripped, delim)
        if previous is not None or delim is not None:
            continue  # a doc-string boundary line or its literal body
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("@"):
            # Checked by POSITION, not by content: the same tag line is legal above a
            # scenario and fatal inside one, so it cannot be judged in isolation.
            if problem := _misplaced_tag_problem(lines, index, stripped, fname):
                problems.append(problem)
            continue
        if _FEATURE_RE.match(stripped) or _RULE_RE.match(stripped):
            in_scenario = False
        elif _SCENARIO_KW_RE.match(stripped):
            in_scenario = True
        elif in_scenario and not _is_skippable_in_scenario(stripped):
            problems.append(
                f"{fname}: line inside a scenario is missing a "
                f"Given/When/Then/And/But keyword: {stripped[:60]!r}"
            )
        elif not in_scenario and _STEP_RE.match(stripped):
            problems.append(
                f"{fname}: step keyword outside any Scenario/Background — a step before the "
                f"first scenario aborts bddgen: {stripped[:60]!r}"
            )
    return problems


def _feature_issues(features: list[_Feature]) -> tuple[str, list[str]]:
    """Detect Gherkin syntax defects a line-scan can catch BEFORE bddgen aborts the render.

    Returns ``(correction_prompt, problems)``; an empty correction string means the
    features are structurally sound. Flags the high-yield, deterministically-detectable
    mistakes: a step-position line with no step keyword (the usual 'Parse error' cause), an
    unterminated ``\"\"\"`` docstring, and a feature missing its ``Feature:`` line.
    """
    problems: list[str] = []
    for feat in features:
        content = str(feat.get("content", ""))
        fname = str(feat.get("filename", "?"))
        problems.extend(_scan_one_feature(content, fname))
        problems.extend(_outline_join_problems(content, fname))
        if not re.search(r"(?m)^\s*Feature\s*:", content):
            problems.append(f"{fname}: no 'Feature:' line")
        # Run the real doc-string state machine instead of counting BARE fences: an opener
        # carrying a content type (``"""markdown``) is not a bare fence, so the old parity
        # count read a perfectly balanced doc string as unterminated and burned a
        # regeneration on it.
        delim: str | None = None
        for ln in content.splitlines():
            delim = _docstring_state(ln.strip(), delim)
        if delim is not None:
            problems.append(f'{fname}: unterminated """ docstring')

    correction = (
        _GHERKIN_SYNTAX_CORRECTION.format(problems="; ".join(problems[:6])) if problems else ""
    )
    return correction, problems


def _grounding_reask_rejection(
    original: list[_Feature],
    retried: list[_Feature],
    original_findings: int,
    corpus: PageCorpus | None,
) -> str | None:
    """Why a GROUNDING-ONLY re-ask must not replace the valid original, or ``None``.

    A grounding correction asks for reworded Then steps, never fewer shots. A retry that
    comes back valid but shorter would otherwise be accepted, and ``_ensure_scenario_floor``
    only tops up BELOW the floor — so the film silently loses footage. Scenario NAMES are
    not compared: rewording a scenario is a legitimate response to the correction.
    """
    if not retried:
        return "returned no features"
    if _feature_issues(retried)[1]:
        return "returned invalid Gherkin"
    if _total_scenarios(retried) < _total_scenarios(original):
        return (
            f"returned fewer scenarios ({_total_scenarios(retried)} < {_total_scenarios(original)})"
        )
    retried_counts = {str(f.get("filename")): _total_scenarios([f]) for f in retried}
    for feat in original:
        fname = str(feat.get("filename"))
        if retried_counts.get(fname, 0) < _total_scenarios([feat]):
            return f"dropped scenarios from {fname}"
    retried_findings = len(find_ungrounded_concepts(retried, corpus))
    if retried_findings >= original_findings:
        return f"did not reduce ungrounded concepts ({retried_findings} >= {original_findings})"
    return None


#: Per-feature key carrying the grounding findings that survived the re-ask. The job
#: processor lifts them into ``step_coverage.ungrounded_gherkin_concepts``.
UNGROUNDED_CONCEPTS_KEY = "ungrounded_gherkin_concepts"


def _record_ungrounded_concepts(features: list[_Feature], corpus: PageCorpus | None) -> None:
    """Stamp each feature with the ungrounded ``Then`` concepts still present (warning only)."""
    # Judged suite-wide (a modifier used anywhere in the suite grounds it), then filed
    # per feature.
    findings = find_ungrounded_concepts(features, corpus)
    for feat in features:
        fname = str(feat.get("filename", "?"))
        found = [f for f in findings if f["file"] == fname]
        feat[UNGROUNDED_CONCEPTS_KEY] = found
        if found:
            logger.warning(
                "%s: %d Then step(s) still name UI the page does not show after the "
                "grounding re-ask; kept and recorded: %s",
                feat.get("filename", "?"),
                len(found),
                "; ".join(f"{u['step']!r} ({u['concept']})" for u in found[:3]),
            )


class FeatureGenerator:
    """Generates Gherkin .feature files from inspection data using an LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def _revalidate_features(
        self,
        features: list[_Feature],
        prompt: str,
        system_prompt: str,
        corpus: PageCorpus | None = None,
    ) -> list[_Feature]:
        """Reject syntactically-invalid Gherkin and regenerate ONCE with a correction.

        Mirrors ``step_def_generator``'s validate-then-regenerate guard: if the generated
        features carry a bddgen-breaking syntax defect (:func:`_feature_issues`), re-ask the
        model once with an explicit correction, then re-parse. Best-effort — if the retry is
        still invalid it is returned as-is (bddgen remains the hard backstop), never raising,
        so a validation false-positive can never fail a generation that would have worked.

        With a page ``corpus`` (demo stream) the SAME single re-ask also carries a grounding
        correction for ``Then`` steps naming UI the page does not show
        (:mod:`gherkin_grounding`). Grounding never adds a second retry, and a
        grounding-only re-ask whose answer is syntactically broken or empty is discarded in
        favour of the valid original — an ungrounded phrase is a warning, a parse error
        aborts the render.
        """
        corrections, problems = _feature_issues(features)
        ungrounded = find_ungrounded_concepts(features, corpus)
        grounding = grounding_correction(ungrounded)
        if not corrections and not grounding:
            return features
        if corrections:
            logger.warning(
                "Generated Gherkin has a syntax defect (%s); regenerating once",
                "; ".join(problems),
            )
        if grounding:
            logger.warning(
                "Generated Gherkin names %d UI concept(s) the page does not show (%s); "
                "regenerating once",
                len(ungrounded),
                "; ".join(f"{u['step']!r} ({u['concept']})" for u in ungrounded[:3]),
            )
        raw = await self._llm.generate(
            prompt + corrections + grounding, system_prompt=system_prompt
        )
        retried = _parse_features(raw)
        reason = (
            None
            if corrections
            else _grounding_reask_rejection(features, retried, len(ungrounded), corpus)
        )
        if reason:
            logger.warning("Grounding re-ask %s; keeping the valid original", reason)
            return features
        # A surviving Scenario Outline is the one residual defect that is SILENT downstream:
        # bddgen renders N videos against one @clip-<n> id and one narration line, and the
        # viewer just hears the wrong voiceover. Everything else here aborts loudly in
        # bddgen, so say this one out loud instead of returning it unremarked.
        if surviving := [
            problem
            for feat in retried
            for problem in _outline_join_problems(
                str(feat.get("content", "")), str(feat.get("filename", "?"))
            )
        ]:
            logger.error(
                "Regeneration still returned %d Scenario Outline(s) that cannot be expanded; "
                "their clips WILL mis-join to the narration: %s",
                len(surviving),
                "; ".join(surviving[:3]),
            )
        return retried

    async def _ensure_scenario_floor(
        self,
        features: list[_Feature],
        prompt: str,
        system_prompt: str,
        floor: int,
    ) -> list[_Feature]:
        """Regenerate ONCE when the model returned fewer scenarios than the caller implied.

        ``max_scenarios_per_feature`` is enforced as a cap and nothing has ever enforced a
        minimum, so the scenario count was purely the model's choice. Measured live on
        2026-08-03: three calibration renders of the IDENTICAL scenario against the same
        cluster produced 3, 3 and 2 clips — the third failed the downstream
        ``>= 3 distinct clip ids`` assertion on nothing but model variance.

        Best-effort by design, mirroring :meth:`_revalidate_features`: one retry, and a
        still-short result is returned with a WARNING rather than raised. A hard failure
        here would abort a paid render over a shortfall that costs a shot, not the film.
        The retry output is re-validated for syntax, so the top-up can never trade a
        thin-but-valid feature set for a fuller broken one.
        """
        actual = _total_scenarios(features)
        if floor < 1 or actual >= floor:
            return features
        logger.warning(
            "Generated %d scenario(s) against an implied floor of %d "
            "(from focus_areas / min_scenarios, clamped by the cap); "
            "regenerating once",
            actual,
            floor,
        )
        correction = _SCENARIO_FLOOR_CORRECTION.format(actual=actual, floor=floor)
        raw = await self._llm.generate(prompt + correction, system_prompt=system_prompt)
        retried = await self._revalidate_features(
            _parse_features(raw), prompt + correction, system_prompt
        )
        retried_total = _total_scenarios(retried)
        if retried_total < actual:
            logger.warning(
                "Scenario-floor retry returned FEWER scenarios (%d < %d); keeping the original",
                retried_total,
                actual,
            )
            return features
        if retried_total < floor:
            # Not fatal, but it is the honest explanation for a downstream shot-count
            # failure — say it here rather than leaving the render to fail unexplained.
            logger.warning(
                "Scenario-floor retry still returned %d scenario(s), under the floor of %d",
                retried_total,
                floor,
            )
        return retried

    async def generate_qa_features(
        self, inspection: _Inspection, config: _Config
    ) -> list[_Feature]:
        """Generate QA test features for the inspected application.

        Returns:
            List of dicts with keys: filename, content, scenario_count, tags.
        """
        page_summary = _build_page_summary(inspection)
        max_scenarios = config.get("max_scenarios_per_feature", 10)
        focus_instruction = _focus_areas_instruction(config)
        auth_config = config.get("auth_config")

        auth_instruction = ""
        if auth_config:
            auth_instruction = (
                "\nIMPORTANT — Authentication:\n"
                "The application requires login. Authentication is handled automatically "
                "via Playwright's global setup (storageState) — every test starts in an "
                "authenticated session. Generate tests for the authenticated pages and "
                "features, NOT for the login page itself (unless specifically testing "
                "login/logout flows). Assume the user is already logged in when tests run.\n"
            )

        prompt = (
            "Generate comprehensive Gherkin BDD feature files for QA testing "
            "the following web application.\n\n"
            f"{page_summary}\n\n"
            f"{auth_instruction}"
            f"{focus_instruction}"
            "Requirements:\n"
            f"- Maximum {max_scenarios} scenarios per feature file\n"
            "- Cover happy paths, edge cases, and error handling\n"
            "- Use descriptive scenario names\n"
            "- Include @smoke, @regression, or @critical tags as appropriate\n"
            "- Use Given/When/Then steps with concrete selectors from the inspection\n"
            "- Wrap each feature in a ```gherkin code block\n"
        )

        system_prompt = (
            "You are a senior QA engineer who writes precise Gherkin BDD feature files. "
            "Output only valid Gherkin syntax in fenced code blocks."
        )

        raw = await self._llm.generate(prompt, system_prompt=system_prompt)
        features = _parse_features(raw)
        features = await self._revalidate_features(features, prompt, system_prompt)
        features = _enforce_scenario_cap(features, max_scenarios)
        logger.info("Generated %d QA feature files", len(features))
        return features

    async def generate_accessibility_features(
        self, inspection: _Inspection, config: _Config
    ) -> list[_Feature]:
        """Generate accessibility test features.

        Returns:
            List of dicts with keys: filename, content, scenario_count, tags.
        """
        page_summary = _build_page_summary(inspection)
        max_scenarios = config.get("max_scenarios_per_feature", 10)
        focus_instruction = _focus_areas_instruction(config)

        prompt = (
            "Generate Gherkin BDD feature files for accessibility testing (WCAG 2.1) "
            "of the following web application.\n\n"
            f"{page_summary}\n\n"
            f"{focus_instruction}"
            "Requirements:\n"
            f"- Maximum {max_scenarios} scenarios per feature file\n"
            "- Test keyboard navigation, screen reader compatibility, color contrast\n"
            "- Test ARIA labels, focus management, and form accessibility\n"
            "- Use @a11y and @wcag tags\n"
            "- Wrap each feature in a ```gherkin code block\n"
        )

        system_prompt = (
            "You are an accessibility testing specialist who writes Gherkin BDD features "
            "focused on WCAG 2.1 compliance. Output only valid Gherkin syntax."
        )

        raw = await self._llm.generate(prompt, system_prompt=system_prompt)
        features = _parse_features(raw)
        features = await self._revalidate_features(features, prompt, system_prompt)
        features = _enforce_scenario_cap(features, max_scenarios)
        logger.info("Generated %d accessibility feature files", len(features))
        return features

    async def generate_demo_features(
        self, inspection: _Inspection, config: _Config
    ) -> list[_Feature]:
        """Generate demo/walkthrough features.

        ⚠️ The prompt's SELF-CONTAINED requirement is load-bearing — do not trim it as
        boilerplate. Each scenario is recorded as its own clip in its own fresh browser
        context (``src/templates/scaffold/lib/recording_helpers.ts`` documents the same
        constraint for wizards: "A stateful wizard recorded as separate clips loses its
        in-progress localStorage between clips"), so a scenario that asserts on data an
        earlier scenario created can only ever sit in a locator timeout.

        Without it, the LLM reads "show the key features in a logical order" as licence
        to write a narrative — add a todo in scenario 1, complete it in scenario 2 —
        and scenarios 2..N fail deterministically. Measured 2026-08-29 on three
        `demo_video_calibration` runs (01a04d34 / 01a04d44 / 01a04d57): the SAME two
        scenarios failed in all three on
        ``getByRole('listitem').filter({ hasText: 'Walk the dog' })``, a todo only the
        first scenario ever added. Those failures are also why captured footage varied
        40.56s / 110.76s / 19.28s for an identical suite — a failed assertion holds the
        recording open for its full 5s timeout, so clip length was an artifact of how
        long the failures took.

        Returns:
            List of dicts with keys: filename, content, scenario_count, tags.
        """
        page_summary = _build_page_summary(inspection)
        max_scenarios = config.get("max_scenarios_per_feature", 10)
        focus_instruction = _focus_areas_instruction(config)
        auth_config = config.get("auth_config")

        auth_instruction = ""
        if auth_config:
            auth_instruction = (
                "\nIMPORTANT — Authentication:\n"
                "The application requires login. Authentication is handled automatically "
                "via Playwright's global setup (storageState) — every scenario starts in "
                "an authenticated session. Demonstrate the authenticated interior screens, "
                "NOT the login page itself. Assume the user is already logged in.\n"
            )

        prompt = (
            "Generate Gherkin BDD feature files that demonstrate a guided walkthrough "
            "of the following web application. These scenarios will be narrated for "
            "a video demo.\n\n"
            f"{page_summary}\n\n"
            f"{auth_instruction}"
            f"{focus_instruction}"
            "Requirements:\n"
            f"- Maximum {max_scenarios} scenarios per feature file\n"
            "- Show the application's key features in a logical order\n"
            "- EVERY SCENARIO MUST BE SELF-CONTAINED. Each scenario is recorded as "
            "its OWN video clip in a FRESH browser context, so nothing a previous "
            "scenario created still exists: no todo it added, no record it saved, no "
            "form it filled. A scenario that asserts on data an earlier scenario "
            "created will sit in a locator timeout and fail. If a scenario needs "
            "data, it must create that data itself, in its own Given/When steps, "
            "before asserting on it.\n"
            "- Include clear, descriptive steps suitable for narration\n"
            "- DESCRIBE ONLY WHAT THE PAGE SHOWS. Steps (and especially Then steps) may "
            "name only controls, labels, text and counts that appear in the page "
            "information above (headings, interactive elements, visible text), or that "
            "directly result from an action described in the same scenario. Never invent "
            "a counter, badge, tab, filter or feature the page does not have — these "
            "steps are narrated, so an invented feature becomes a false claim in the "
            "video.\n"
            "- Use @demo and @walkthrough tags\n"
            "- Wrap each feature in a ```gherkin code block\n"
        )

        system_prompt = (
            "You are a product demo specialist who writes Gherkin BDD features "
            "for video walkthroughs. Focus on clarity and flow."
        )

        raw = await self._llm.generate(prompt, system_prompt=system_prompt)
        features = _parse_features(raw)
        corpus = build_page_corpus(inspection)
        features = await self._revalidate_features(features, prompt, system_prompt, corpus)
        # Floor BEFORE cap: the cap trims down, so topping up afterwards would hand the
        # trim a list it never inspected. Demo-only — this is the stream with a clip
        # spine and a downstream shot-count assertion; QA/accessibility have neither.
        features = await self._ensure_scenario_floor(
            features, prompt, system_prompt, _scenario_floor(config)
        )
        features = _enforce_scenario_cap(features, max_scenarios)
        _record_ungrounded_concepts(features, corpus)
        logger.info(
            "Generated %d demo feature files (%d scenarios)",
            len(features),
            _total_scenarios(features),
        )
        return features

    def build_showcase_features(self, screens: list[_Screen], config: _Config) -> list[_Feature]:
        """Deterministically emit a showcase-tour feature from a knowledge pack.

        Unlike ``generate_demo_features`` (which asks an LLM to INVENT a
        walkthrough and never reads ``screens[]``), this consumes the pack's
        authored ``screens[]`` verbatim — one ``@clip-<n>`` scenario per screen,
        in document order — and maps each screen's ``route`` + ``capture``
        directives onto the fixed showcase step vocabulary implemented in
        ``templates/scaffold/steps/showcase.steps.ts``:

          route            -> When I open "<route>" and wait for "<wait_for>"
          capture.click    -> When I click "<selector>"
          capture.scroll_to-> When I scroll to "<selector>"
          capture.hold_ms  -> Then I dwell for <ms> ms  (the settled hero tail)

        No LLM is involved in the structure or the steps — the LLM only polishes
        narration downstream. Returns the standard feature-dict shape
        (``filename``, ``content``, ``scenario_count``, ``tags``).
        """
        app_name = config.get("knowledge_app_name") or "Application"
        feature_title = f"{app_name} — showcase tour"
        lines: list[str] = [f"Feature: {feature_title}", ""]
        tags: list[str] = []
        count = 0
        for index, screen in enumerate(screens, start=1):
            capture = screen.get("capture") or {}
            route = screen.get("route") or "/"
            label = screen.get("label") or screen.get("id") or f"Screen {index}"
            # A screen with no content selector waits only for `body` (the app
            # shell), which resolves before real data renders — risking a clip of
            # a half-loaded UI. Warn loudly; the pack author should set a real
            # `capture.wait_for` selector (showcase.steps.ts still falls back to
            # `body` so this degrades, it doesn't break).
            wait_for = capture.get("wait_for")
            if not wait_for:
                logger.warning(
                    "showcase screen %r has no capture.wait_for; the recorder will "
                    "wait only for `body` (the app shell) and may capture a "
                    "half-loaded UI — set a real content selector.",
                    label,
                )
                wait_for = "body"
            hold_ms = int(capture.get("hold_ms") or 4000)
            clip = f"clip-{index}"
            tags.append(clip)
            lines.append(f"  @{clip}")
            lines.append(f"  Scenario: {label}")
            lines.append(
                f"    When I open {_gherkin_quote(route)} and wait for {_gherkin_quote(wait_for)}"
            )
            if capture.get("click"):
                lines.append(f"    And I click {_gherkin_quote(capture['click'])}")
            if capture.get("scroll_to"):
                lines.append(f"    And I scroll to {_gherkin_quote(capture['scroll_to'])}")
            lines.append(f"    Then I dwell for {hold_ms} ms")
            lines.append("")
            count += 1

        content = "\n".join(lines).rstrip() + "\n"
        filename = re.sub(r"[^a-z0-9]+", "_", feature_title.lower()).strip("_") + ".feature"
        logger.info("Emitted deterministic showcase feature: %d screens", count)
        return [
            {
                "filename": filename,
                "content": content,
                "scenario_count": max(count, 1),
                "tags": tags,
            }
        ]

    async def generate_documentation_features(
        self, inspection: _Inspection, config: _Config
    ) -> list[_Feature]:
        """Generate documentation features that describe application behavior.

        Returns:
            List of dicts with keys: filename, content, scenario_count, tags.
        """
        page_summary = _build_page_summary(inspection)
        max_scenarios = config.get("max_scenarios_per_feature", 10)

        prompt = (
            "Generate Gherkin BDD feature files that serve as living documentation "
            "for the following web application.\n\n"
            f"{page_summary}\n\n"
            "Requirements:\n"
            f"- Maximum {max_scenarios} scenarios per feature file\n"
            "- Describe expected behavior for each feature area\n"
            "- Use clear business-language steps\n"
            "- Use @docs and @specification tags\n"
            "- Wrap each feature in a ```gherkin code block\n"
        )

        system_prompt = (
            "You are a technical writer who uses Gherkin BDD to create living documentation. "
            "Write clear, specification-style feature files."
        )

        raw = await self._llm.generate(prompt, system_prompt=system_prompt)
        features = _parse_features(raw)
        features = await self._revalidate_features(features, prompt, system_prompt)
        features = _enforce_scenario_cap(features, max_scenarios)
        logger.info("Generated %d documentation feature files", len(features))
        return features
