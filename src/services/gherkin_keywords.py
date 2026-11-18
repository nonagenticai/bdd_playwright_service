"""Single source of truth for Gherkin step keywords and their BINDING kind.

Gherkin has three PRIMARY step keywords — ``Given``/``When``/``Then`` — and three
CONTINUATION forms — ``And``, ``But`` and the ``*`` bullet. A continuation is not a
kind of its own: it INHERITS the kind of the preceding primary keyword in the same
scenario. playwright-bdd only ever exposes ``Given``/``When``/``Then`` bindings from
``createBdd()``, so emitting a literal ``And(...)`` call into a step-definition file
produces ``ReferenceError: And is not defined``, bddgen exits 1 and the whole
(paid) render dies — observed live 2026-08-03, corpus run 30793758726.

Every place that has to turn a Gherkin keyword into a binding — the step-def prompt,
the deterministic step-def repair pass, and the narration engine — resolves it
through :func:`resolve_binding` here, so the "two functions agreeing on a wrong
answer" failure mode cannot recur.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Final, NamedTuple

#: The step keywords that carry a kind of their own.
PRIMARY_KEYWORDS: Final[tuple[str, ...]] = ("Given", "When", "Then")
#: The step keywords that inherit the kind of the preceding primary keyword.
CONTINUATION_KEYWORDS: Final[tuple[str, ...]] = ("And", "But", "*")
#: Every legal step-opening keyword.
STEP_KEYWORDS: Final[tuple[str, ...]] = PRIMARY_KEYWORDS + CONTINUATION_KEYWORDS

#: A step line: keyword + text. ``*`` needs no trailing space rule of its own — the
#: ``\s+`` covers every form.
STEP_LINE_RE: Final = re.compile(r"^(?P<keyword>Given|When|Then|And|But|\*)\s+(?P<text>.+)$")

#: A line that OPENS a new scenario body, at which point continuation inheritance
#: resets (an ``And`` cannot inherit across a scenario boundary).
SCENARIO_OPEN_RE: Final = re.compile(
    r"^(?:Scenario\b|Background\b|Example[ \t]*:|Rule\b|Feature\s*:)"
)

#: Every Gherkin keyword that OPENS a single scenario, spelled as it appears in a
#: feature file. ``Example`` is a legal alias for ``Scenario`` and ``Scenario Template``
#: for ``Scenario Outline`` (Gherkin 6), and an LLM emits them routinely.
SCENARIO_OPENER_KEYWORDS: Final[tuple[str, ...]] = (
    "Scenario",
    "Scenario Outline",
    "Scenario Template",
    "Example",
)

#: THE scenario-opener pattern — one compiled object, imported by BOTH ends of the
#: ``@clip-<n>`` join spine.
#:
#: The stamper (``generation.feature_generator._stamp_clip_tags``) decides which lines get
#: a ``@clip-<n>`` tag; the narration parser (``input.gherkin_parser``) decides which lines
#: get a narration part. They MUST accept exactly the same keyword set: a scenario the
#: stamper tags but the parser skips is a clip with no voiceover, and every later clip's
#: narration shifts onto the wrong screen. They were two independently-written regexes and
#: they DID diverge — the stamper took ``Example:``/``Scenario Template:``, the parser did
#: not. Import this; never re-spell it.
#:
#: The trailing ``[ \t]*:`` is what keeps the plural ``Examples:`` (the Scenario Outline
#: DATA block, which must NEVER open a scenario) out: after ``Example`` it sees ``s``, not
#: ``:``. ``name`` is optional because an unnamed ``Scenario:`` is legal Gherkin — the
#: stamper always tagged one, so requiring a name here would recreate the very divergence
#: this constant exists to prevent. Full i18n (``Szenario:``…) is out of scope.
SCENARIO_OPENER_RE: Final = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<keyword>Scenario(?:[ \t]+(?:Outline|Template))?|Example)"
    r"[ \t]*:[ \t]*(?P<name>.*?)[ \t]*$"
)

#: The ``Scenario Outline``/``Scenario Template`` keywords — the openers whose body is a
#: TEMPLATE over an ``Examples:`` table rather than one concrete scenario. One tag above
#: one of these cannot identify the N videos bddgen generates from N table rows, which is
#: why ``feature_generator`` expands them before stamping.
OUTLINE_KEYWORDS: Final[tuple[str, ...]] = ("Scenario Outline", "Scenario Template")

#: The ``Examples:`` (plural) data-block header of a Scenario Outline. Gherkin lets the
#: block carry a name (``Examples: valid logins``), so the tail is captured, not forbidden.
EXAMPLES_BLOCK_RE: Final = re.compile(r"^[ \t]*Examples[ \t]*:[ \t]*(?P<name>.*?)[ \t]*$")


def normalize_opener_keyword(keyword: str) -> str:
    """Collapse the internal whitespace of a matched opener keyword.

    ``SCENARIO_OPENER_RE`` accepts ``Scenario   Outline`` (any run of spaces/tabs); this
    maps what it matched back onto a member of :data:`SCENARIO_OPENER_KEYWORDS` so callers
    can compare against that tuple instead of re-parsing the spelling.
    """
    return " ".join(keyword.split())


#: What a leading continuation with nothing to inherit from binds as. ``Given`` is the
#: only safe choice for a step definition: it is a real playwright-bdd binding and a
#: scenario's first step is a precondition by convention.
DEFAULT_BINDING: Final[str] = "Given"


class ResolvedStep(NamedTuple):
    """One Gherkin step with its continuation keyword already resolved."""

    keyword: str
    """The keyword as WRITTEN in the feature file (may be ``And``/``But``/``*``)."""

    binding: str
    """The playwright-bdd binding to use — always one of ``PRIMARY_KEYWORDS``."""

    text: str
    """The step text with the keyword stripped."""


def is_continuation(keyword: str) -> bool:
    """True when ``keyword`` inherits its kind from the preceding primary keyword."""
    return keyword in CONTINUATION_KEYWORDS


def resolve_binding(
    keyword: str,
    last_primary: str | None = None,
    *,
    default: str = DEFAULT_BINDING,
) -> str:
    """Return the Given/When/Then binding for ``keyword``.

    A primary keyword binds to itself. ``And``/``But``/``*`` bind to
    ``last_primary``; when there is none (the continuation opens the scenario) they
    fall back to ``default`` — never to the continuation keyword itself, which is not
    a binding at all. An unrecognised keyword is treated as a continuation, so no
    input can ever produce a non-binding return value.
    """
    if keyword in PRIMARY_KEYWORDS:
        return keyword
    if last_primary in PRIMARY_KEYWORDS:
        # `last_primary in PRIMARY_KEYWORDS` narrows out None for the type checker.
        return str(last_primary)
    return default if default in PRIMARY_KEYWORDS else DEFAULT_BINDING


def iter_steps(feature_text: str, *, default: str = DEFAULT_BINDING) -> Iterator[ResolvedStep]:
    """Yield every step of a feature file with its continuation keyword resolved.

    Inheritance resets at each ``Scenario:``/``Background:``/``Rule:`` boundary, so a
    scenario opening with ``And`` gets ``default`` rather than the trailing keyword of
    the scenario above it.
    """
    last_primary: str | None = None
    for raw_line in feature_text.splitlines():
        line = raw_line.strip()
        if SCENARIO_OPEN_RE.match(line):
            last_primary = None
            continue
        match = STEP_LINE_RE.match(line)
        if not match:
            continue
        keyword = match.group("keyword")
        binding = resolve_binding(keyword, last_primary, default=default)
        if keyword in PRIMARY_KEYWORDS:
            last_primary = keyword
        yield ResolvedStep(keyword=keyword, binding=binding, text=match.group("text").strip())
