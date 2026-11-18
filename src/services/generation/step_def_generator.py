"""Generate TypeScript step definitions for playwright-bdd."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.database.models import (
    DurationBudget,
    StepCoverage,
    UngroundedAssertion,
    UngroundedGherkinConcept,
)
from src.services.generation.llm_client import LLMClient
from src.services.gherkin_keywords import (
    CONTINUATION_KEYWORDS,
    DEFAULT_BINDING,
    PRIMARY_KEYWORDS,
    iter_steps,
    resolve_binding,
)
from src.services.input.html_structure import LABEL_LENGTH_CAP

logger = logging.getLogger(__name__)

# A generated step-definition file record: ``{"filename", "content", "step_count"}``.
_StepDef = dict[str, Any]
# A parsed Gherkin feature record (``{"content": <feature text>, …}``).
_Feature = dict[str, Any]


class StepDefGenerationError(RuntimeError):
    """Raised when generated step definitions are structurally invalid."""


# ── stub visibility: a stubbed scenario must be distinguishable from a real one ─
# An auto-stub's whole body is `waitForLoadState('networkidle')` + `stubDwell`: it
# PASSES and performs no action, so a heavily-stubbed scenario films a page that
# never changed for the whole dwell. Until this report existed the ONLY trace of
# that was a log WARNING, i.e. nothing downstream could tell the two apart.
#
# THRESHOLD: `degraded` is raised at a MAJORITY of the suite's steps stubbed.
# Rationale: most real scenarios carry SOME stubs (an LLM misses a step or two)
# and those recordings are fine, so a low threshold would mark every working run.
# At >= 50% the median step of the "demo" does nothing, which is a static
# screenshot with a timer rather than a demo. The flag is STRUCTURED DATA ONLY —
# it deliberately does NOT fail the job: generation succeeded, and turning a
# currently-shipping package red is a product decision for the caller
# (w2v_orchestrator's demo_video_record path), not for this service.
STUB_DEGRADED_RATIO = 0.5
# How many stubbed step texts ride along for diagnosis (bounded: a suite can have
# hundreds and this rides in a JSONB column and a package manifest).
STUB_SAMPLE_LIMIT = 10

# `waitForLoadState('networkidle')` is a NAVIGATION-class wait, so with no
# explicit timeout it falls back to `navigationTimeout` (15_000ms in
# playwright.config.ts), not the shorter `actionTimeout`. The call is already
# `.catch()`-wrapped (best-effort settle), so a stall here pays the full 15s
# silently before continuing. That 15s lands directly in the recorded clip —
# the clip's measured length feeds the orchestrator's duration solver as a
# hard minimum — so an LLM-authored suite that draws one extra uncovered step
# can blow a beat's budget by exactly one of these stalls. Bound it short.
STUB_NETWORK_IDLE_TIMEOUT_MS = 4000


@dataclass
class StepCoverageReport:
    """How much of a generated suite PASSES WITHOUT ACTING.

    Filled in by ``_apply_compile_guarantees`` for the recording stream and
    carried out of the generator by the caller, so the stub share is readable as
    structured data instead of by grepping logs.
    """

    steps_total: int = 0
    steps_stubbed: int = 0
    stubbed_steps: list[str] = field(default_factory=list)
    #: The ordered-duration split, when the caller ordered one. Rides on this
    #: report rather than on a parallel channel so there is exactly ONE place a
    #: consumer looks to learn "was this package delivered as ordered?".
    #: ``None`` when no ``target_duration_seconds`` was sent.
    duration_budget: DurationBudget | None = None
    #: Assertions the recording-stream grounding pass removed because their selector
    #: names an id/class/testid the scraped DOM does not carry (or a `toHaveCSS`
    #: no snapshot can verify). Structured, so a consumer can tell "this clip's
    #: Then-step asserts nothing any more" without grepping a WARNING.
    ungrounded_assertions: list[UngroundedAssertion] = field(default_factory=list)
    #: Demo Gherkin ``Then`` steps naming UI the page snapshot does not show, left over
    #: after the feature generator's one grounding re-ask. Warning only — steps are kept.
    ungrounded_gherkin_concepts: list[UngroundedGherkinConcept] = field(default_factory=list)

    @property
    def stub_ratio(self) -> float:
        """Share of the suite's Gherkin steps that are non-acting stubs.

        A suite with no steps at all is 0.0, not 1.0: there is no footage claim to
        degrade, and reporting a full stub-out for an empty suite would fire the
        threshold on every non-recording stream.
        """
        if self.steps_total <= 0:
            return 0.0
        return self.steps_stubbed / self.steps_total

    @property
    def degraded(self) -> bool:
        """True when a majority of the suite's steps do nothing (see threshold)."""
        return self.stub_ratio >= STUB_DEGRADED_RATIO

    def as_dict(self) -> StepCoverage:
        """JSON-safe payload for the job row, the API result and the package."""
        return {
            "steps_total": self.steps_total,
            "steps_stubbed": self.steps_stubbed,
            "stub_ratio": round(self.stub_ratio, 4),
            "degraded": self.degraded,
            "degraded_threshold": STUB_DEGRADED_RATIO,
            "stubbed_steps": self.stubbed_steps[:STUB_SAMPLE_LIMIT],
            "duration_budget": self.duration_budget,
            "ungrounded_assertions_dropped": len(self.ungrounded_assertions),
            "ungrounded_assertions": self.ungrounded_assertions[:STUB_SAMPLE_LIMIT],
            "ungrounded_gherkin_concepts_flagged": len(self.ungrounded_gherkin_concepts),
            "ungrounded_gherkin_concepts": self.ungrounded_gherkin_concepts[:STUB_SAMPLE_LIMIT],
        }


# ── the per-scenario recording duration, ONE owner ────────────────────────────
# `templates/scaffold/lib/recording_helpers.ts` declares `HERO_DWELL_MS` and is the
# canonical copy of that library (hand-authored packages carry a byte-identical
# one). Restating the number here in Python would make it a magic constant living
# in two files — and, once published to the orchestrator, in two REPOS, which is
# exactly how `w2v_orchestrator`'s `_SECONDS_PER_SCENARIO = 4.0` came to sit 5.5x
# below what a recording actually yields. So PARSE it out of the TypeScript source
# instead: the .ts file stays the single owner and a drift is impossible rather
# than merely discouraged.
_RECORDING_HELPERS_TS = (
    Path(__file__).resolve().parent.parent.parent
    / "templates"
    / "scaffold"
    / "lib"
    / "recording_helpers.ts"
)
_HERO_DWELL_DECL_RE = re.compile(
    r"^export\s+const\s+HERO_DWELL_MS\s*=\s*([\d_]+)\s*;", re.MULTILINE
)


def _read_hero_dwell_ms() -> int:
    """`HERO_DWELL_MS` as declared by the canonical helper library.

    Raises rather than defaulting: a silent fallback here would publish a
    plausible-but-wrong per-scenario duration to the orchestrator, which is the
    fail-open-on-a-measurement-gap shape this whole change exists to remove.
    """
    src = _RECORDING_HELPERS_TS.read_text(encoding="utf-8")
    m = _HERO_DWELL_DECL_RE.search(src)
    if not m:
        raise StepDefGenerationError(
            f"Could not read `export const HERO_DWELL_MS` from {_RECORDING_HELPERS_TS}. "
            "That declaration is the single owner of the per-scenario recording "
            "duration; if it is renamed or reformatted, update _HERO_DWELL_DECL_RE — "
            "do NOT restate the number in Python."
        )
    return int(m.group(1).replace("_", ""))


#: Per-scenario hero-dwell budget in milliseconds, owned by `recording_helpers.ts`.
HERO_DWELL_MS: int = _read_hero_dwell_ms()

#: ⭐ THE EXPORTED PER-SCENARIO RECORDING DURATION.
#:
#: The floor, in seconds, that a demo_video scenario is GUARANTEED to record, now
#: that `_ensure_recording_pacing` force-injects the dwell instead of asking the
#: model for it. Published on the wire as
#: ``narration_script.parts[].expected_recording_seconds`` (built in
#: ``job_processor._generate_narration``) — that field is what `w2v_orchestrator`
#: must read in place of its hardcoded `_SECONDS_PER_SCENARIO`.
#:
#: It is a FLOOR, not a prediction: a scenario's own step actions run before the
#: dwell, so real clips land at or above it (measured 16.7-36.9s against a 22s
#: floor). Sizing a timeline off the floor under-fills at worst, never over-fills.
GUARANTEED_RECORDING_SECONDS_PER_SCENARIO: float = HERO_DWELL_MS / 1000.0


# playwright-bdd injects `page` (and other Playwright fixtures) per-step,
# destructured inside each Given/When/Then async callback. `expect` is NOT a
# fixture — it is a module import (`import { expect } from '@playwright/test'`,
# enforced by `_ensure_expect_import`). CALLING either at module/top level
# (e.g. `const FILTER = page.getByRole(...)` at file scope) throws
# `ReferenceError: page is not defined` when the suite runs — a real failure
# observed live. Step bodies are indented; module-scope statements sit at
# column 0, so an unindented line that USES `page`/`expect(` is the violation.
_MODULE_SCOPE_DECL_RE = re.compile(r"(?:export\s+)?(?:const|let|var)\s+\w+\s*=.*\bpage\b")
_MODULE_SCOPE_USE_RE = re.compile(r"(?:await\s+)?(?:page\b|expect\s*\()")
# A module-scope decl whose RHS is a FUNCTION expression (arrow or `function`).
# Inside a function body nothing runs at module load, and an arrow's parameter
# list can legitimately introduce `page` (a curried locator factory —
# `const TODO_ITEM = (text) => page => page.getByRole(...)`, or
# `const F = (page) => page.locator(...)`). So `page` appearing there is NOT a
# module-load `ReferenceError`; only a DIRECT deref (`const F = page.getByRole(...)`)
# is. Exclude the function-RHS form from the violation (a false positive on it
# hard-failed the live todomvc demo run, 2026-07-24).
_MODULE_SCOPE_FUNC_DECL_RE = re.compile(
    r"(?:export\s+)?(?:const|let|var)\s+\w+\s*(?::[^=]+?)?=\s*"
    r"(?:async\s+)?(?:function\b|[A-Za-z_$][\w$]*\s*=>|\([^)]*\)\s*(?::[^=]+?)?=>)"
)

_MODULE_SCOPE_CORRECTION = (
    "\n\nCORRECTION — your previous output CALLED `page` or `expect` at module "
    "(file) scope, which throws `ReferenceError: page is not defined`. `page` and "
    "every Playwright fixture exist ONLY inside the Given/When/Then async "
    "callbacks; `expect` must be imported at module scope "
    "(`import { expect } from '@playwright/test'`) and called inside callbacks. "
    "Declare selectors as plain STRING constants at module scope and call "
    "`page.locator(sel)` INSIDE each callback. Regenerate accordingly.\n"
)

# A step-definition call: Given/When/Then with either a quoted-string pattern or a
# /regex/ literal. playwright-bdd registers Given/When/Then in ONE shared registry,
# so two definitions sharing a pattern make bddgen fail with "Multiple definitions
# matched scenario step" — a real failure observed live (todomvc, 2026-06-17).
_STEP_DEF_RE = re.compile(
    r"\b(?:Given|When|Then)\s*\(\s*"
    r"(?:(?P<q>['\"`])(?P<str>(?:\\.|(?!(?P=q)).)*)(?P=q)"
    r"|/(?P<re>(?:\\.|[^/\n])+)/)"
)

_DUPLICATE_STEP_CORRECTION = (
    "\n\nCORRECTION — your previous output defined the same step pattern more than "
    "once ({dups}), so playwright-bdd's bddgen fails with 'Multiple definitions "
    "matched scenario step'. Given/When/Then share ONE registry, so each step "
    "pattern must have EXACTLY ONE definition across ALL files. Merge the duplicate "
    "definitions into a single step (parameterise with {{int}}/{{string}} if they "
    "differ only by a literal). Regenerate accordingly.\n"
)

# `expect` is a standalone `@playwright/test` export, NOT a playwright-bdd fixture:
# it must be imported at module scope (`import { expect } from '@playwright/test'`).
# A suite that calls `expect(...)` without importing it throws `ReferenceError:
# expect is not defined` at TEST RUN — bddgen still COMPILES, so this slips past
# the module-scope and duplicate guards and only surfaces when the runner executes
# the suite: every scenario fails, the app is never driven, and the demo_video
# pipeline records BLANK footage (observed live, todomvc, 2026-07). Fix it
# deterministically here rather than trusting the LLM to import it.
_EXPECT_CALL_RE = re.compile(r"\bexpect\s*\(")
_EXPECT_IMPORT_RE = re.compile(
    r"""import\s*\{[^}]*\bexpect\b[^}]*\}\s*from\s*['"]@playwright/test['"]"""
)
# `expect` destructured from a step's fixture arg — `async ({ page, expect }) =>`.
# It is not a real fixture (resolves to `undefined`) and would SHADOW the module
# import, so strip it and let every call bind to the imported `expect`.
_EXPECT_FIXTURE_RE = re.compile(r"(async\s*\(\s*\{[^}]*?),?\s*\bexpect\b\s*,?(\s*[^}]*\})")
_PW_BDD_IMPORT_RE = re.compile(
    r"^import\s.*\bfrom\s*['\"]playwright-bdd['\"];?[^\n]*$", re.MULTILINE
)
_EXPECT_IMPORT_LINE = "import { expect } from '@playwright/test';"


def _first_module_scope_violation(step_defs: list[_StepDef]) -> str | None:
    """Return a human-readable locator of the first module-scope fixture use, if any."""
    for sd in step_defs:
        for i, line in enumerate(sd.get("content", "").splitlines(), 1):
            # Blank or indented lines are inside a block (function body) — fine.
            if not line or line[0] in " \t":
                continue
            # A `const X = <function expr>` defers `page`/`expect` into a body or
            # binds `page` as an arrow parameter — not evaluated at module load,
            # so no ReferenceError. Only a DIRECT module-scope deref is a real
            # violation.
            if _MODULE_SCOPE_DECL_RE.match(line) and not _MODULE_SCOPE_FUNC_DECL_RE.match(line):
                return f"{sd.get('filename', '?')} line {i}: {line.strip()[:80]}"
            if _MODULE_SCOPE_USE_RE.match(line):
                return f"{sd.get('filename', '?')} line {i}: {line.strip()[:80]}"
    return None


def _step_def_pattern_on_line(line: str) -> str | None:
    """If ``line`` opens a Given/When/Then step definition, return its pattern
    text (string content or regex source); otherwise None.

    Anchored at the first non-space character and comment-aware: a `Given(` that
    appears inside a string literal mid-line, or in a `//`/`*` comment, must NOT
    register as a definition — those phantom patterns would corrupt duplicate
    detection and (worse) cause `_dedupe_step_defs` to delete a real adjacent
    block.
    """
    stripped = line.lstrip()
    if stripped.startswith(("//", "*", "/*")):
        return None
    m = _STEP_DEF_RE.match(stripped)
    if not m:
        return None
    return m.group("str") if m.group("str") is not None else m.group("re")


def _duplicate_step_patterns(step_defs: list[_StepDef]) -> list[str]:
    """Return step patterns defined more than once across all step-def files.

    Keyed on the raw pattern text (string content or regex source), ignoring the
    Given/When/Then keyword since playwright-bdd matches steps from one shared
    registry. A duplicate pattern is always a bddgen-breaking defect. (Semantic
    equivalence — e.g. a string pattern vs an identical regex — is intentionally
    NOT normalised here; bddgen remains the real backstop for that.)
    """
    seen: set[str] = set()
    dups: list[str] = []
    for sd in step_defs:
        for line in sd.get("content", "").splitlines():
            pattern = _step_def_pattern_on_line(line)
            if pattern is None:
                continue
            if pattern in seen:
                if pattern not in dups:
                    dups.append(pattern)
            else:
                seen.add(pattern)
    return dups


def _step_def_issues(step_defs: list[_StepDef]) -> tuple[str, list[str]]:
    """Detect known bddgen-breaking defects in generated step defs.

    Returns ``(combined_correction_prompt, problem_descriptions)``. An empty
    correction string means the step defs are structurally sound.
    """
    corrections = ""
    problems: list[str] = []

    violation = _first_module_scope_violation(step_defs)
    if violation:
        problems.append(f"fixture at module scope ({violation})")
        corrections += _MODULE_SCOPE_CORRECTION

    dups = _duplicate_step_patterns(step_defs)
    if dups:
        shown = ", ".join(repr(d) for d in dups[:5])
        problems.append(f"duplicate step pattern(s): {shown}")
        corrections += _DUPLICATE_STEP_CORRECTION.format(dups=shown)

    undefined = _undefined_module_bindings(step_defs)
    if undefined:
        shown = ", ".join(undefined[:5])
        problems.append(f"undefined step binding(s): {shown}")
        corrections += _UNDEFINED_BINDING_CORRECTION.format(names=shown)

    return corrections, problems


def _ensure_expect_import(step_defs: list[_StepDef]) -> int:
    """Guarantee every step file that calls ``expect(...)`` imports it.

    ``expect`` is a standalone ``@playwright/test`` export, not a playwright-bdd
    fixture, so a suite that calls it without importing throws ``ReferenceError:
    expect is not defined`` at run time while bddgen still compiles — the demo
    then records blank footage. Deterministically (a) strip any ``expect``
    destructured from a step's fixture arg (it is not a real fixture and would
    shadow the module import as ``undefined``) and (b) inject the module-level
    import when it is missing. Mutates ``step_defs`` in place; returns the number
    of files changed.
    """
    fixed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        if not _EXPECT_CALL_RE.search(content):
            continue
        # (a) drop `expect` from any fixture destructuring so it can't shadow the
        # module import. Repeat until stable (a single arg may list it once).
        new = _EXPECT_FIXTURE_RE.sub(r"\1\2", content)
        # (b) ensure the module-level import is present.
        if not _EXPECT_IMPORT_RE.search(new):
            anchor = _PW_BDD_IMPORT_RE.search(new)
            if anchor:
                new = new[: anchor.end()] + "\n" + _EXPECT_IMPORT_LINE + new[anchor.end() :]
            else:
                # No playwright-bdd import to anchor on: insert after an optional
                # leading `// filename:` comment, else at the very top.
                lines = new.splitlines(keepends=True)
                at = 1 if lines and lines[0].lstrip().startswith("//") else 0
                lines.insert(at, _EXPECT_IMPORT_LINE + "\n")
                new = "".join(lines)
        if new != content:
            sd["content"] = new
            fixed += 1
    return fixed


def _ensure_bdd_bindings_in_scope(step_defs: list[_StepDef]) -> int:
    """Guarantee every file calling Given/When/Then actually brings them into scope.

    `createBdd()` bindings are not globals: a file that calls `Given(...)` without
    `const { Given, When, Then } = createBdd();` throws `ReferenceError: Given is not
    defined` at module load and bddgen exits 1 — the same class of dead render as the
    `And(` bug, just with a valid binding name. The LLM omits the boilerplate often
    enough that failing the generation would be needlessly brittle, and this repo
    already REPAIRS the analogous missing `expect` import rather than rejecting.

    Injects the playwright-bdd import (when absent) and the destructure, anchored
    exactly like `_ensure_expect_import`. Only the names still missing from module
    scope are bound — a file that already destructures a SUBSET (e.g. `const {
    Given, Then } = createBdd();`) but also calls `When(...)` gets a second,
    narrower `const { When } = createBdd();`, not a full redeclaration of `Given`
    and `Then` (a second full destructure throws Babel's `VarRedeclaration` before
    a single assertion runs). Names are always emitted in `PRIMARY_KEYWORDS` order
    so the injected line is deterministic regardless of set iteration order.
    Mutates in place; returns files changed.
    """
    fixed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        called = {kw for kw in PRIMARY_KEYWORDS if re.search(rf"^{kw}\s*\(", content, re.MULTILINE)}
        if not called:
            continue
        scope_names = _module_scope_names(content)
        missing = [kw for kw in PRIMARY_KEYWORDS if kw in called and kw not in scope_names]
        if not missing:
            continue
        new = content
        if not _PW_BDD_IMPORT_RE.search(new):
            lines = new.splitlines(keepends=True)
            at = 1 if lines and lines[0].lstrip().startswith("//") else 0
            lines.insert(at, "import { createBdd } from 'playwright-bdd';\n")
            new = "".join(lines)
        anchor = _PW_BDD_IMPORT_RE.search(new)
        # Anchor is guaranteed by the insert above; keep the guard for type-narrowing.
        if anchor:
            destructure = f"\nconst {{ {', '.join(missing)} }} = createBdd();"
            new = new[: anchor.end()] + destructure + new[anchor.end() :]
        if new != content:
            sd["content"] = new
            fixed += 1
    if fixed:
        logger.warning("Injected missing createBdd() bindings into %d step file(s)", fixed)
    return fixed


# A step-def block closes on a `});` line (the generator emits each Given/When/Then
# as a top-level call closed at column 0). Allow leading whitespace defensively.
# The step-def block close: `});` at the opening call's indent. Tolerate a trailing
# line comment (`});  // navigate step`) — a plausible LLM habit; without this the scan
# misses the close, MERGES this block with the next, and a later dedupe/subsumed drop on
# the merged span silently deletes the unrelated, valid following step definition.
_STEP_DEF_CLOSE_RE = re.compile(r"^\s*\}\)\s*;?\s*(?://.*)?$")


def _filter_step_def_blocks(
    step_defs: list[_StepDef], drop: Callable[[str], bool]
) -> list[_StepDef]:
    """Remove whole step-definition blocks for which ``drop(pattern)`` is True,
    in place, recomputing ``step_count``. A block spans from its
    ``Given/When/Then(`` opening line to the next ``});`` at the opening call's
    indentation; a block with no clean close is left untouched rather than
    risking removing unrelated code. ``drop`` is invoked once per cleanly-closed
    block, in document order, so a stateful predicate (e.g. first-wins dedupe)
    sees blocks in order. Shared by the exact-duplicate dedupe and the
    subsumed-literal passes so both use one proven, indent-aware scanner.
    """
    for sd in step_defs:
        lines = sd.get("content", "").splitlines()
        out: list[str] = []
        i, n = 0, len(lines)
        while i < n:
            pattern = _step_def_pattern_on_line(lines[i])
            if pattern is not None:
                open_indent = len(lines[i]) - len(lines[i].lstrip())
                # The real close sits at the opening call's indentation (the
                # generator emits step calls at column 0). A `});` indented
                # DEEPER closes a nested callback inside the body — e.g.
                # `page.evaluate(() => { ... });` — not the step itself; skipping
                # those avoids truncating the block and orphaning broken TS.
                j = i + 1
                while j < n and not (
                    _STEP_DEF_CLOSE_RE.match(lines[j])
                    and (len(lines[j]) - len(lines[j].lstrip())) <= open_indent
                ):
                    j += 1
                if j < n:  # found the block close at the step's own indent
                    if drop(pattern):
                        i = j + 1
                        # also swallow a single trailing blank separator line
                        if i < n and not lines[i].strip():
                            i += 1
                        continue
                    out.extend(lines[i : j + 1])
                    i = j + 1
                    continue
            out.append(lines[i])
            i += 1
        sd["content"] = "\n".join(out)
        sd["step_count"] = max(
            sum(1 for ln in sd["content"].splitlines() if _step_def_pattern_on_line(ln)), 1
        )
    return step_defs


def _dedupe_step_defs(step_defs: list[_StepDef]) -> list[_StepDef]:
    """Drop duplicate step-definition blocks in place, keeping the first.

    The LLM is unreliable at avoiding duplicate step patterns even after a
    corrective regeneration, and playwright-bdd's bddgen hard-fails on them
    ("Multiple definitions matched scenario step"). Since Given/When/Then share
    one registry, keeping the first definition of each pattern is a safe,
    deterministic guarantee that the suite compiles.
    """
    seen: set[str] = set()

    def _is_repeat(pattern: str) -> bool:
        if pattern in seen:
            return True
        seen.add(pattern)
        return False

    return _filter_step_def_blocks(step_defs, _is_repeat)


def _is_pure_literal(source: str) -> bool:
    """True if a ``str``-kind cucumber pattern is a FIXED literal — no parameter
    ``{...}``, optional group ``(...)``, alternation ``/``, or escape ``\\``.

    Such a pattern matches exactly one concrete step text, equal to ``source``
    itself, which is what lets ``_subsumed_literal_patterns`` reason about it
    precisely (any of those specials would make the matched-text set larger than
    one string, so we treat the pattern as non-literal and out of scope).
    """
    return not any(c in source for c in "{}()/\\")


def _subsumed_literal_patterns(step_defs: list[_StepDef]) -> set[str]:
    """Raw pattern text of pure-literal step defs that ANOTHER def already covers.

    The LLM sometimes emits BOTH a general parameterized def
    (``I have a task {string} in my list``) AND a hardcoded literal
    specialization (``I have a task "Temporary task" in my list``). Their raw
    pattern strings differ, so exact dedup (`_dedupe_step_defs`) keeps both and
    bddgen aborts: "Multiple definitions matched scenario step" (observed live,
    todomvc demo_video e2e 2026-07-24).

    A pure literal matches exactly one concrete step; if any OTHER def's cucumber
    pattern — compiled to its guaranteed SUPERSET regex via
    ``_cuke_pattern_to_regex`` — also matches that one string, the literal is
    redundant. Dropping it and keeping the general def is behaviour-preserving
    for that step (the general still matches it) and removes the ambiguity.
    Only pure literals are ever returned (a parameterized def is never dropped),
    and only when a concrete covering def exists, so coverage never shrinks.
    Run AFTER exact dedup so identical literals are already collapsed.
    """
    patterns: list[tuple[str, str]] = []
    for sd in step_defs:
        patterns.extend(_iter_step_def_patterns(sd.get("content", "")))

    drop: set[str] = set()
    for kind, source in patterns:
        if kind != "str" or not _is_pure_literal(source):
            continue
        for other_kind, other_source in patterns:
            if other_kind == kind and other_source == source:
                continue  # itself, or an exact duplicate (handled by dedupe)
            if other_kind == "str" and _is_pure_literal(other_source):
                continue  # a different fixed literal only matches its own text
            rx = _cuke_pattern_to_regex(other_kind, other_source)
            if rx is not None and rx.fullmatch(source):
                drop.add(source)
                break
    return drop


def _extract_steps_from_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract unique steps from feature file contents, keyword already RESOLVED.

    ``keyword`` is always a real playwright-bdd binding (Given/When/Then): the
    ``And``/``But``/``*`` continuation forms are resolved to the kind of the preceding
    primary keyword by :func:`~src.services.gherkin_keywords.iter_steps`. The raw
    Gherkin keyword is kept under ``gherkin_keyword`` for diagnostics. Emitting the
    raw ``And`` here is what put a literal ``And(...)`` call into a generated step-def
    file and killed a live render with ``ReferenceError: And is not defined``.
    """
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()

    for feature in features:
        content = feature.get("content", "")
        for step in iter_steps(content):
            if step.text in seen:
                continue
            seen.add(step.text)
            steps.append(
                {
                    "keyword": step.binding,
                    "gherkin_keyword": step.keyword,
                    "text": step.text,
                }
            )
    return steps


# The prompt asks for a ```typescript fence, but the model routinely answers with
# a different tag (```tsx / ```js / ```javascript) or a capitalised ```TypeScript.
# The original pattern accepted ONLY `typescript`/`ts`/bare, so any other tag
# matched NOTHING, `findall` returned [] and the ENTIRE raw markdown answer —
# prose, headings, fence markers — was written verbatim as `steps.ts`. Babel then
# parsed English as TypeScript and bddgen exited 1 inside `babelBundleImpl.js`,
# killing the whole demo_video run (live kill 2026-08-15 04:52Z + that night's
# orchestrator run). So strip ANY language tag, case-insensitively.
_CODE_FENCE_RE = re.compile(r"```[A-Za-z0-9_+#-]*[ \t]*\r?\n(.*?)```", re.DOTALL)

# A playwright-bdd step binding call, including the `bdd.Given(` member form
# (`\b` matches straight after the dot). A file that contains none of these — and
# no alias of one — registers no step.
_STEP_BINDING_CALL_RE = re.compile(r"\b(?:Given|When|Then)\s*\(")

# `const { Given: given, When: when } = createBdd();` — a destructuring RENAME.
# Valid playwright-bdd that compiles and registers steps, but carries no literal
# `Given(` call, so the call regex alone false-rejects it (defect D2). Capture the
# local alias so we can look for ITS call site.
_STEP_BINDING_ALIAS_RE = re.compile(r"\b(?:Given|When|Then)\s*:\s*([A-Za-z_$][A-Za-z0-9_$]*)")


def _registers_step(text: str) -> bool:
    """Does ``text`` register at least one Gherkin step binding?

    Direct call (`Given(`, `bdd.Given(`) or a destructured alias that is itself
    called. This is deliberately NOT a per-file requirement — a legal answer may
    include a support file (hooks/fixtures) that registers nothing; see
    `_parse_step_defs`, which applies it to the whole SET.
    """
    if _STEP_BINDING_CALL_RE.search(text):
        return True
    return any(
        re.search(rf"\b{re.escape(alias)}\s*\(", text)
        for alias in _STEP_BINDING_ALIAS_RE.findall(text)
    )


# ── positive code signals ────────────────────────────────────────────────────
# The first cut of this gate was a NEGATIVE heuristic (reject on a markdown
# heading, reject a block with no Given/When/Then). It traded the corruption bug
# for an availability regression: a markdown heading occurs inside perfectly valid
# TypeScript (a template literal holding markdown, a `/* ... */` banner), and a
# legal multi-file answer whose support file registers no step hard-failed a paid
# render. So invert it: a block is PROSE only when it shows NO code signal at all.
# Each signal is anchored at line start where the keyword is also an ordinary
# English word, so narrative prose ("export the report", "I would import ...")
# cannot fake it.
_CODE_SIGNAL_RES: tuple[re.Pattern[str], ...] = (
    # ES module / CJS wiring — every real step-def or support file has one.
    re.compile(r"^[ \t]*import[ \t\n{*'\"]", re.MULTILINE),
    re.compile(r"^[ \t]*export\b", re.MULTILINE),
    re.compile(r"\brequire\s*\("),
    # The playwright-bdd entry point itself.
    re.compile(r"\bcreateBdd\s*\("),
    # Fixture files (`fixtures.ts`) extend the base test instead of binding steps.
    re.compile(r"\btest\s*\.\s*extend\s*\("),
    # Hook files (`hooks.ts`) register Before/After(All|Each) and nothing else.
    re.compile(r"\b(?:Before|After)(?:All|Each)?\s*\("),
    # Any declaration at line start: `const`/`let`/`var`/`function`/`class`.
    re.compile(r"^[ \t]*(?:async[ \t]+)?(?:const|let|var|function|class)\b", re.MULTILINE),
)


def _looks_like_typescript(text: str) -> bool:
    """Structural gate: is ``text`` code at all (as opposed to a prose answer)?

    Deterministic and cheap — the existing "COMPILE GATE" (`_undefined_module_bindings`)
    is name-resolution only and `_step_def_issues` is regex heuristics, so NOTHING
    upstream of this stopped a markdown answer from reaching disk.

    Two rules only:
      * a surviving ``` fence marker is a HARD reject — unambiguous evidence that raw
        markdown, not the fenced payload, is about to be written out;
      * otherwise the text must show at least one positive code signal. A markdown
        heading no longer rejects on its own; it only "counts" in the sense that a
        pure heading-and-paragraphs answer has no code signal to show.
    """
    if "```" in text:
        return False
    if _registers_step(text):
        return True
    return any(rx.search(text) for rx in _CODE_SIGNAL_RES)


def _reject_non_step_defs(content: str, reason: str) -> None:
    """Raise with the head of the offending content so the failure is diagnosable."""
    raise StepDefGenerationError(
        f"generated step definitions are not TypeScript step definitions ({reason}); "
        "a markdown/prose answer written as `steps.ts` makes bddgen exit 1 mid-render. "
        f"First 200 chars: {content[:200]!r}"
    )


# A prose answer is TRANSIENT misbehaviour a single re-ask fixes — unlike the
# module-scope/duplicate/undefined-binding defects, nothing about it is structural.
# Same shape as `_MODULE_SCOPE_CORRECTION` & co: appended to the ORIGINAL prompt for
# exactly one regeneration.
_PROSE_ANSWER_CORRECTION = (
    "\n\nCORRECTION — your previous output was PROSE (explanation, a plan, markdown "
    "headings), not TypeScript. It cannot be written as `steps.ts`: Babel parses the "
    "file as TypeScript and bddgen exits 1, killing the whole render. Respond with "
    "ONLY a single ```typescript fenced code block containing the step definitions. "
    "No prose, no headings, no commentary before or after the fence. Regenerate "
    "accordingly.\n"
)


def _parse_step_defs(raw: str) -> list[_StepDef]:
    """Parse LLM output into step definition file dicts.

    Raises :class:`StepDefGenerationError` if the model answered with prose/markdown
    rather than code — writing that out verbatim is what kills the render downstream.
    """
    step_defs: list[_StepDef] = []

    # An empty/whitespace-only answer yields no files (callers treat that as "nothing
    # generated"); it is not the prose failure mode, so it is not an error here.
    if not raw.strip():
        return []

    blocks = _CODE_FENCE_RE.findall(raw)
    if not blocks:
        # An unfenced pure-code answer is legitimate and must keep working. An
        # unfenced PROSE answer must not: it is the exact input that shipped English
        # to Babel. Gate before falling back rather than trusting the response.
        if not _looks_like_typescript(raw):
            _reject_non_step_defs(raw, "no code fence and the response is not TypeScript")
        blocks = [raw]

    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue

        # FINAL gate on the content that is about to be emitted as a `.ts` file —
        # the single chokepoint through which every LLM-authored step-def file passes
        # (both the first generation and the post-correction regeneration).
        # PER-BLOCK gate: "is this code at all". Deliberately NOT "does it bind a
        # step" — a legal answer wraps each FILE in its own fence, and a support
        # file (hooks.ts, fixtures.ts) binds nothing. Requiring a binding per block
        # killed a whole paid render over one legal support file.
        if not _looks_like_typescript(block):
            _reject_non_step_defs(block, "fence marker survived, or no TypeScript code signal")

        # Count step definitions (Given/When/Then calls)
        step_count = len(_STEP_BINDING_CALL_RE.findall(block))

        # Try to extract filename from comments. The value is LLM-controlled and
        # becomes a path segment under the job's ``steps/`` dir, so strip any
        # directory component (``../evil.ts`` → ``evil.ts``) to keep the write
        # confined to the storage root.
        default_name = f"steps_{i}.ts" if i > 0 else "steps.ts"
        filename_match = re.search(r"//\s*(?:file|filename):\s*(\S+)", block)
        filename = Path(filename_match.group(1)).name if filename_match else default_name
        if not filename:
            filename = default_name

        step_defs.append(
            {
                "filename": filename,
                "content": block,
                "step_count": max(step_count, 1),
            }
        )

    # SET-LEVEL gate: the answer as a whole must bind at least one step. One support
    # file among others is fine; an answer where NOTHING anywhere registers a step is
    # not step definitions, whatever it is, and there would be nothing for bddgen to
    # match — so it is the same failure as prose and must raise.
    if step_defs and not any(_registers_step(sd["content"]) for sd in step_defs):
        _reject_non_step_defs(step_defs[0]["content"], "no block registers a Given/When/Then step")

    return step_defs


# ── step-coverage guarantee (every Gherkin step must have a matching def) ─────
# playwright-bdd compiles a suite even when a Gherkin step has NO matching step
# def (`missingSteps:'fail-on-run'` for the video streams) and then THROWS
# "undefined step" when the runner reaches it — aborting the scenario so its clip
# records BLANK footage → the demo_video blank-clip gate hard-fails the paid
# render. Nothing else cross-checks coverage, so guarantee it deterministically:
# match every feature step against the generated def patterns and append a
# non-fatal stub for any still uncovered.


# A step-def binding call whose pattern is NOT on the same line — the prettier /
# LLM habit of
#     When(
#       'I navigate to "https://x"',
#       async ({ page }) => {…},
#     );
# `_STEP_DEF_RE` requires the quote on the opening line, so such a definition is
# INVISIBLE to every pattern pass: its `/` is never cuke-escaped and bddgen still
# dies with "Alternative may not be empty". Handled conservatively: the opening
# line must be EXACTLY `Given(`/`When(`/`Then(` (nothing else on it), and the
# pattern must be the first thing on the very next line. Anything more exotic is
# left alone rather than risking a rewrite that corrupts generated TypeScript.
_STEP_DEF_CONT_OPEN_RE = re.compile(r"^(?:Given|When|Then)\s*\(\s*$")
_STEP_DEF_CONT_PATTERN_RE = re.compile(
    r"^(?:(?P<q>['\"`])(?P<str>(?:\\.|(?!(?P=q)).)*)(?P=q)|/(?P<re>(?:\\.|[^/\n])+)/)"
)


def _iter_step_def_pattern_sites(content: str) -> list[tuple[str, str, int, int, int]]:
    """Locate every step-def pattern in ``content``.

    Yields ``(kind, source, line_index, start, end)`` where ``start``/``end`` are
    offsets into the LSTRIPPED text of ``lines[line_index]`` — enough for a caller
    to rewrite the pattern span in place. ``kind`` is ``"str"`` for a
    cucumber-expression string pattern or ``"re"`` for a ``/regex/`` literal.

    Covers both the same-line form and the two-line form above. Comment-aware via
    the same anchoring as ``_step_def_pattern_on_line``. NOTE: the block-deleting
    passes (`_filter_step_def_blocks`, `_duplicate_step_patterns`) deliberately do
    NOT use this — they need block boundaries, and mis-scanning there DELETES real
    code. Seeing more patterns here is safe in one direction only, and it is the
    right one: a multi-line def is now escaped and counted as COVERAGE, so the
    autostub cannot emit a duplicate definition for a step it already defines.
    """
    lines = content.splitlines()
    sites: list[tuple[str, str, int, int, int]] = []
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("//", "*", "/*")):
            continue
        m = _STEP_DEF_RE.match(stripped)
        if m is not None:
            kind = "str" if m.group("str") is not None else "re"
            start, end = m.span(kind)
            sites.append((kind, m.group(kind), idx, start, end))
            continue
        if not _STEP_DEF_CONT_OPEN_RE.match(stripped) or idx + 1 >= len(lines):
            continue
        nxt = lines[idx + 1].lstrip()
        if nxt.startswith(("//", "*", "/*")):
            continue
        m2 = _STEP_DEF_CONT_PATTERN_RE.match(nxt)
        if m2 is None:
            continue
        kind = "str" if m2.group("str") is not None else "re"
        start, end = m2.span(kind)
        sites.append((kind, m2.group(kind), idx + 1, start, end))
    return sites


def _iter_step_def_patterns(content: str) -> list[tuple[str, str]]:
    """Return ``(kind, source)`` for each step def opened in ``content``.

    ``kind`` is ``"str"`` for a cucumber-expression string pattern or ``"re"`` for
    a ``/regex/`` literal; ``source`` is the raw pattern text (still
    TypeScript-escaped). Thin projection of :func:`_iter_step_def_pattern_sites`.
    """
    return [(kind, source) for kind, source, _idx, _s, _e in _iter_step_def_pattern_sites(content)]


def _cuke_escape(text: str) -> str:
    """Escape a literal string for safe use as a cucumber EXPRESSION.

    Cucumber-expression specials are ``{ } ( ) / \\`` — backslash-escape them so
    the text matches literally (a `/` would otherwise mean alternation, `(` an
    optional group, `{` a parameter). All other characters — including regex
    metacharacters and quotes — are already literal in a cucumber expression.
    """
    return re.sub(r"([{}()/\\])", r"\\\1", text)


# ── LLM-authored patterns must be VALID cucumber expressions ─────────────────
# `/` is the ALTERNATION operator in a cucumber expression, so the `//` of a URL
# parses as an EMPTY alternative and cucumber-expressions throws at COMPILE time:
#
#   Error: This Cucumber Expression has a problem at column 23:
#   I navigate to "https://example.com"
#                         ^
#   Alternative may not be empty.
#
# bddgen exits 1 and the whole paid demo_video render dies at
# `demo_video_record` (observed live 2026-08-15, reproduced twice). The model
# mirrors the feature's step sentence verbatim as the pattern, and
# `feature_generator` injects `Application URL: {base_url}` into the Gherkin, so
# a concrete `https://…` routinely lands inside the sentence.
#
# `_cuke_escape` above is the LITERAL escaper for text we author ourselves; it is
# deliberately NOT reused here, because an LLM pattern is a cucumber EXPRESSION,
# not a literal: it legitimately contains `{string}`/`{int}` parameters and
# `(s)` optional groups, and escaping those would turn every parameter into dead
# literal text (the step would then never match → undefined step → blank clip).
# So escape ONLY the character that is invalid as written: `/`.
#
# `{` and `(` are deliberately left ALONE. Both are ambiguous — `{string}` /
# `(s)` are correct uses the model makes constantly, and nothing distinguishes an
# intended-literal `{` from an intended-parameter one without re-implementing the
# cucumber parser. `/` is not ambiguous: a step SENTENCE never wants alternation
# (the model writes prose, and the feature step it must match contains the
# literal slash), so escaping every `/` is always the behaviour-preserving
# repair. Narrow and provable beats broad.
#
# NOTE the two encoding layers. The pattern lives inside a TypeScript string
# literal, so a `\` reaching the cucumber parser needs TWO backslashes in the TS
# source (`'https:\\/\\/x'` → value `https:\/\/x`). A single `\/` in TS source is
# just a redundant escape for `/` and reaches cucumber UNESCAPED — which is why
# the parity rule below counts SOURCE backslashes and asks whether the DECODED
# value has an odd number of them before the slash.
_CUKE_SLASH_RE = re.compile(r"(?P<bs>\\*)/")


def _cuke_escape_slashes(source: str) -> str:
    """Backslash-escape every cucumber-alternation `/` in a TS-source pattern.

    ``source`` is the RAW text between the quotes of a step-def string literal
    (i.e. still TypeScript-escaped). Returns it with every `/` that is not
    already escaped *as seen by cucumber* prefixed with a source-level ``\\\\``.

    A run of ``k`` source backslashes before the slash decodes to
    ``k // 2`` value-level backslashes (an odd trailing one is consumed by the
    `\\/` escape, which yields a bare `/`). The slash is already escaped for
    cucumber iff that count is ODD, so the pass is IDEMPOTENT: re-running it
    never double-escapes. Parameters ``{...}`` and optional groups ``(...)`` are
    untouched — see the module comment above for why.
    """

    def _fix(m: re.Match[str]) -> str:
        backslashes = m.group("bs")
        if (len(backslashes) // 2) % 2 == 1:
            return m.group(0)  # already escaped for cucumber — leave it alone
        return "\\\\" + backslashes + "/"

    return _CUKE_SLASH_RE.sub(_fix, source)


def _ensure_valid_cuke_expressions(step_defs: list[_StepDef]) -> int:
    """Make every LLM-authored step-def pattern a compilable cucumber expression.

    Only STRING patterns are touched; a ``/regex/`` literal is a JS RegExp where
    `/` is the delimiter and has no cucumber meaning, so rewriting it would
    corrupt a valid definition. Comment-aware, and covers the two-line
    ``When(\\n  'pattern',`` form via :func:`_iter_step_def_pattern_sites`.
    Mutates in place; returns the number of files changed.
    """
    changed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        lines = content.splitlines()
        touched = False
        for kind, source, idx, start, end in _iter_step_def_pattern_sites(content):
            if kind != "str":
                continue
            fixed = _cuke_escape_slashes(source)
            if fixed == source:
                continue
            line = lines[idx]
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            lines[idx] = indent + stripped[:start] + fixed + stripped[end:]
            touched = True
        if touched:
            sd["content"] = "\n".join(lines) + ("\n" if content.endswith("\n") else "")
            changed += 1
    return changed


# Per-type parameter regexes, each a SUPERSET of cucumber-expressions 18.0.1's
# built-in type (so we never judge a covered step "uncovered" → no double-match)
# yet tight enough that a generic `{string}`/`{int}` def does NOT swallow an
# unrelated step (which would leave that step unstubbed → runtime "undefined
# step" → blank clip). Unknown/custom param types fall back to permissive `.+?`
# (over-match — the safe direction). `{string}` handles both quote styles +
# escapes; the numeric types share a broad number regex.
_NUM_RE = r"[-+]?(?:\d+[.,]?\d*|[.,]\d+)(?:[eE][-+]?\d+)?"
_CUKE_PARAM_REGEX = {
    "": r".*?",  # cucumber anonymous {} is `.*` (zero-or-more) — must not require ≥1
    "int": r"[-+]?\d+",
    "byte": r"[-+]?\d+",
    "short": r"[-+]?\d+",
    "long": r"[-+]?\d+",
    "biginteger": r"[-+]?\d+",
    "float": _NUM_RE,
    "double": _NUM_RE,
    "bigdecimal": _NUM_RE,
    "word": r"\S+",
    "string": r"""(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""",
}


# TypeScript string-literal escapes this module decodes with CERTAINTY.
#
# A pattern's `source` is the raw span between the quotes of a TS string literal,
# so a `\` in it may belong to EITHER encoding layer (see the `_CUKE_SLASH_RE`
# note above): `\\/` in TS source is the ONE spelling that reaches the cucumber
# parser as the escape `\/`. To model such a pattern we must decode the TS layer
# first, then parse what cucumber actually sees.
#
# Only these forms are unambiguous — each yields exactly the following character,
# with no codepoint arithmetic and no ambiguity about how many characters were
# consumed. `\n`, `\t`, `\u….`, `\x..` and octal forms are deliberately ABSENT:
# a step SENTENCE never contains them, and mis-decoding one would silently
# mis-model the pattern in the under-match direction (→ a stub for an
# already-covered step → 'Multiple definitions matched'). Anything else makes the
# whole pattern unmodellable and we bail to covers-all.
_TS_DECODABLE_ESCAPES = "\\/'\"`$"


def _ts_decode_pattern(source: str) -> str | None:
    """Decode a TS string-literal body into the text the cucumber parser sees.

    Returns ``None`` when the literal contains a TS escape outside
    ``_TS_DECODABLE_ESCAPES`` (or a dangling trailing backslash), i.e. whenever a
    faithful decode is not provable.
    """
    if "\\" not in source:
        return source
    out: list[str] = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n or source[i + 1] not in _TS_DECODABLE_ESCAPES:
            return None
        out.append(source[i + 1])
        i += 2
    return "".join(out)


def _cuke_param_regex(src: str, i: int) -> tuple[str, int] | None:
    """``{type}`` at ``src[i]`` → ``(regex, next_index)``; None if unterminated.

    An unknown/custom parameter type falls back to permissive ``.+?`` — over-match
    is the safe direction (see :func:`_cuke_pattern_to_regex`).
    """
    j = src.find("}", i)
    if j == -1:
        return None
    return _CUKE_PARAM_REGEX.get(src[i + 1 : j].strip(), r".+?"), j + 1


def _cuke_optional_group_regex(src: str, i: int) -> tuple[str, int] | None:
    """``(text)`` at ``src[i]`` → ``((?:text)?, next_index)``.

    None when the group is unbalanced, NESTED, or contains an escape — modelling
    those precisely is not provable, so bail to covers-all instead.
    """
    j = src.find(")", i)
    if j == -1:
        return None
    body = src[i + 1 : j]
    if "(" in body or "\\" in body:
        return None
    return f"(?:{re.escape(body)})?", j + 1


def _cuke_expression_parts(src: str) -> list[str] | None:
    """Tokenise a DECODED cucumber expression into regex fragments.

    ``src`` must already be TS-decoded (see :func:`_ts_decode_pattern`) — the
    escapes handled here are the CUCUMBER ones. Returns ``None`` for any
    construct that cannot be modelled as a guaranteed superset.
    """
    parts: list[str] = ["^"]
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch == "\\":  # cucumber escape → the NEXT character is a LITERAL
            # This is the whole point of the decode: `\/` matches a literal `/` in
            # the step text, so `I navigate to "https:\/\/x"` matches the feature's
            # `I navigate to "https://x"` and that step is seen as COVERED.
            if i + 1 >= n:
                return None  # dangling escape — not a valid cucumber expression
            parts.append(re.escape(src[i + 1]))
            i += 2
        elif ch == "/":
            # UNESCAPED `/` is cucumber ALTERNATION (`a/b`), whose whitespace-
            # delimited scoping we do not mirror. This is the ONLY remaining
            # textual bail, and after `_ensure_valid_cuke_expressions` it is
            # unreachable for LLM-authored string patterns (every `/` is escaped).
            return None
        elif ch in "{(":  # {param} / (optional) — see helpers above
            emitted = (_cuke_param_regex if ch == "{" else _cuke_optional_group_regex)(src, i)
            if emitted is None:
                return None
            parts.append(emitted[0])
            i = emitted[1]
        elif ch == ")":
            return None  # stray close
        else:
            parts.append(re.escape(ch))
            i += 1
    parts.append("$")
    return parts


def _cuke_pattern_to_regex(kind: str, source: str) -> re.Pattern[str] | None:
    """Compile a step-def pattern into a Python regex that is a SUPERSET of cucumber.

    Superset (over-match) bias: the result matches at least everything cucumber
    would, so a step this def covers is never judged "uncovered" (which would make
    the stub double-match → bddgen 'Multiple definitions matched', a hard failure
    strictly worse than the undefined-step bug). Returns ``None`` when a guaranteed
    superset cannot be built — the caller then treats this def as covering
    everything (stubs nothing for the job), which is safe but blunt, so the bail
    condition is kept as NARROW as provably possible.

    ``source`` is still TypeScript-escaped, so it is decoded to the cucumber-level
    value first and escapes are then modelled as literals. Bails only on: a
    ``/regex/`` literal, a TS escape outside ``_TS_DECODABLE_ESCAPES``, an
    UNESCAPED ``/`` (cucumber alternation), an unterminated ``{param}``/``(opt)``,
    a nested/escape-bearing optional group, or a stray ``)``.
    """
    if kind == "re":
        # JS↔Python regex parity is not guaranteed; never risk under-matching.
        return None
    # Decode the TypeScript layer first: what cucumber parses is the DECODED
    # value, and that is also what must be matched against the raw feature step
    # text. `_ensure_valid_cuke_expressions` deliberately writes `\\/` into the
    # source, so refusing to decode would (and did) leave every escaped pattern
    # unmodellable and disable step-coverage stubbing for the whole job.
    src = _ts_decode_pattern(source)
    if src is None:
        return None
    parts = _cuke_expression_parts(src)
    if parts is None:
        return None
    try:
        return re.compile("".join(parts))
    except re.error:
        return None


def _uncovered_steps(features: list[_Feature], step_defs: list[_StepDef]) -> list[str]:
    """Texts of the feature steps with no matching step-def pattern (over-match bias).

    If any def cannot be bounded to a guaranteed superset, returns ``[]`` (stub
    nothing) — safer than risking a double-match hard failure.
    """
    patterns: list[re.Pattern[str]] = []
    for sd in step_defs:
        for kind, source in _iter_step_def_patterns(sd.get("content", "")):
            rx = _cuke_pattern_to_regex(kind, source)
            if rx is None:
                # One def is unmodellable (a `/regex/` literal, a cuke expression
                # with UNESCAPED `/` alternation, or a TS escape we refuse to
                # decode) → it could match any step, so stubbing anything risks a
                # `Multiple definitions matched` hard failure. Escaped slashes are
                # NOT in this set any more: `_ensure_valid_cuke_expressions` runs
                # first and every `\\/` it writes is now modelled as a literal
                # `/`, so a URL-bearing def no longer silently disables stubbing
                # for the whole job (it did before, in exactly the jobs the escape
                # fix rescues — trading a loud bddgen crash for a blank clip).
                # Disable coverage stubbing for the whole
                # job (best-effort, never worse than the pre-existing undefined-
                # step behaviour) but LOG it so the silent no-op is diagnosable
                # when a live record run still blanks a clip.
                logger.warning(
                    "step-coverage stubbing disabled for this job: step-def "
                    "pattern %r (%s) cannot be bounded to a guaranteed superset",
                    source,
                    kind,
                )
                return []
            patterns.append(rx)
    return [
        step["text"]
        for step in _extract_steps_from_features(features)
        if not any(p.fullmatch(step["text"]) for p in patterns)
    ]


def _ensure_step_coverage(
    features: list[_Feature],
    step_defs: list[_StepDef],
    *,
    stubbed_out: list[str] | None = None,
) -> int:
    """Append a non-fatal stub def for every uncovered Gherkin step.

    Guarantees bddgen never hits an undefined step at record time. Each stub is a
    cucumber-expression STRING def (``json.dumps`` of the cuke-escaped literal —
    never a ``/regex/`` literal, whose delimiter a `/` in the step text would
    break) whose body settles the page and dwells WITHOUT throwing, so the clip
    records the current screen instead of blanking.

    The dwell is BUDGETED, not fixed. A stub takes no action, so its hold time is
    the whole of its contribution to the clip, and ``clip_anchor="end"`` means the
    tail must be a long settled hero (``HERO_DWELL_MS = 22000`` in
    ``lib/recording_helpers.ts``). The previous fixed ``1200`` was 18x short of
    that — a fully-stubbed scenario recorded a near-empty clip — while N stubs at
    1200ms each had no aggregate ceiling and could outlast the per-test timeout,
    which kills the test mid-step and records BLACK. The generated file therefore
    carries a per-stub dwell and a per-scenario total, tracking spend per ``page``
    — one page per ``@clip-<n>`` scenario, so no hook ordering is assumed.

    Both pacing values are their OWN knobs (``PW_STUB_DWELL_MS``,
    ``PW_STUB_DWELL_BUDGET_MS``) rather than pure functions of
    ``PW_TEST_TIMEOUT_MS``. They used to be derived from the timeout alone, which
    coupled how long we FILM a static page to a knob raised for an unrelated
    reason: when the per-test timeout went 45s -> 180s
    (``w2v_orchestrator#852``), the dwell jumped 13.5s -> 22.0s and the budget
    20.2s -> 81.0s, i.e. 4x more filmed dead time per clip at ~25 identical frames
    per second — re-creating the duplicate-frame defect the capture chain is being
    fixed for. ``PW_TEST_TIMEOUT_MS`` still applies as a SAFETY CLAMP, because a
    dwell that does not fit inside its test is killed mid-step and records BLACK.

    The superset matcher has
    already proven no existing def covers these steps, so the literal stub cannot
    double-match. Appends a fresh ``steps_autostub.ts`` (its own ``createBdd()``)
    so it never perturbs existing files or ``_dedupe_step_defs``. Mutates
    ``step_defs`` in place; returns the number of stubs added. ``stubbed_out``, when
    given, receives the stubbed step TEXTS — a stub passes without acting, so the
    caller needs the texts to report WHICH steps are non-acting, not just how many.
    """
    uncovered = _uncovered_steps(features, step_defs)
    if not uncovered:
        return 0
    if stubbed_out is not None:
        stubbed_out.extend(uncovered)
    lines = [
        "// filename: steps_autostub.ts",
        "// Auto-generated fallback steps: each Gherkin step below had no matching",
        "// definition; a non-fatal stub keeps the recording running (non-blank)",
        "// instead of throwing 'undefined step' and blanking the clip.",
        "import type { Page } from '@playwright/test';",
        "import { createBdd } from 'playwright-bdd';",
        "import { isDwellBudgetOrdered, spendDwell } from '../lib/recording_helpers';",
        "",
        "const { When } = createBdd();",
        "",
        "// DWELL BUDGET (do not replace with a bare waitForTimeout again).",
        "// A stub performs no action, so the ONLY thing it contributes to the clip is",
        "// hold time on the current screen. The old fixed 1200ms was wrong in both",
        "// directions:",
        "//   * too SHORT to be usable footage — the compositor anchors each clip at",
        "//     clip_anchor='end', so the tail must be a long settled hero (the shared",
        "//     helper lib/recording_helpers.ts holds HERO_DWELL_MS = 22000 for exactly",
        "//     this reason). A stubbed scenario held 1200ms and produced a near-empty",
        "//     clip that read as a blank/degenerate render downstream.",
        "//   * UNBOUNDED in aggregate — N stubs cost N x 1200ms with no ceiling, so a",
        "//     heavily-stubbed scenario could outlast the per-test timeout, be killed",
        "//     mid-step and record BLACK. That is the same class of bug the nav hook",
        "//     in steps_hooks.ts was rewritten to fix.",
        "// So: dwell long, but spend from a per-scenario budget derived from the same",
        "// per-test timeout the runner (frontend_test_runner_service) exports. The",
        "// budget deliberately leaves room for the nav hook and for the scenario's",
        "// real steps; when it is exhausted a stub returns immediately instead of",
        "// pushing the test over its timeout.",
        "const PW_TEST_TIMEOUT_MS = Number(process.env.PW_TEST_TIMEOUT_MS) || 20000;",
        "// HOW LONG WE FILM A STATIC PAGE IS A PACING DECISION, NOT A TIMEOUT DECISION.",
        "// These were derived from PW_TEST_TIMEOUT_MS alone, which silently coupled",
        "// them to a knob raised for an unrelated reason: when the per-test timeout",
        "// went 45s -> 180s (w2v_orchestrator#852, to stop killing scenarios whose",
        "// steps were resolving correctly), the per-stub dwell jumped 13.5s -> 22.0s",
        "// and the per-scenario dwell budget 20.2s -> 81.0s — 4x more FILMED DEAD",
        "// TIME per clip, at ~25 identical frames per second of it. That is the",
        "// duplicate-frame defect the capture chain is being fixed for, re-created as",
        "// a side effect of a timeout change.",
        "// So the pacing values are their own knobs with their own defaults, and the",
        "// timeout only acts as a SAFETY CLAMP: a dwell must still fit inside the test",
        "// it runs in, or it is killed mid-step and records BLACK.",
        "const STUB_DWELL_MS = Math.min(",
        "  22000,",
        "  Number(process.env.PW_STUB_DWELL_MS) || 13500,",
        "  Math.max(1200, Math.floor(PW_TEST_TIMEOUT_MS * 0.3)),",
        ");",
        "const STUB_DWELL_BUDGET_MS = Math.max(",
        "  1200,",
        "  Math.min(",
        "    Number(process.env.PW_STUB_DWELL_BUDGET_MS) || 20250,",
        "    Math.floor(PW_TEST_TIMEOUT_MS * 0.45),",
        "    PW_TEST_TIMEOUT_MS - 2000,",
        "  ),",
        ");",
        "",
        "// Each @clip-<n> scenario records in its OWN fresh context, so its `page` is a",
        "// distinct object — a WeakMap keyed by page is per-scenario accounting with no",
        "// hook ordering assumption and nothing to reset.",
        "const spent = new WeakMap<Page, number>();",
        "",
        "// ⛔ TWO BUDGETS, ONE CLIP — the defect this shares away.",
        "// STUB_DWELL_BUDGET_MS is derived from PW_TEST_TIMEOUT_MS alone, so it is a",
        "// SAFETY clamp (a dwell must fit inside the test it runs in). It is NOT a",
        "// pacing budget, and it was never narrowed by an ordered duration — while the",
        "// hero ledger in lib/recording_helpers WAS. So one scenario could hold its",
        "// ordered share there AND up to ~20s more here: a 30s order came back with",
        "// 43.56s of footage, the beats got pinned to that footage, and the narration",
        "// was clamped to the vo_rate floor to fit. Both clamps now apply: the local",
        "// one keeps us inside the test timeout, and spendDwell() charges the SHARED",
        "// per-scenario ledger so hero + stub together cannot exceed what was ordered.",
        "// With no order, isDwellBudgetOrdered() is false and this is byte-identical",
        "// to the previous behaviour.",
        "async function stubDwell(page: Page): Promise<void> {",
        "  const used = spent.get(page) ?? 0;",
        "  const remaining = STUB_DWELL_BUDGET_MS - used;",
        "  if (remaining <= 0) return;",
        "  const ms = Math.min(STUB_DWELL_MS, remaining);",
        "  spent.set(page, used + ms);",
        "  if (isDwellBudgetOrdered()) {",
        "    await spendDwell(page, ms);",
        "    return;",
        "  }",
        "  await page.waitForTimeout(ms);",
        "}",
        "",
    ]
    for text in uncovered:
        pattern = json.dumps(_cuke_escape(text))
        lines.append(f"When({pattern}, async ({{ page }}) => {{")
        lines.append(
            "  await page.waitForLoadState('networkidle', "
            f"{{ timeout: {STUB_NETWORK_IDLE_TIMEOUT_MS} }}).catch(() => {{}});"
        )
        lines.append("  await stubDwell(page);")
        lines.append("});")
        lines.append("")
    step_defs.append(
        {
            "filename": "steps_autostub.ts",
            "content": "\n".join(lines),
            "step_count": len(uncovered),
        }
    )
    return len(uncovered)


# ── per-scenario navigation guarantee (a fresh context must not stay blank) ────
# playwright-bdd records ONE video per @clip-<n> scenario in a FRESH BrowserContext
# that starts on about:blank (a BLACK frame). A clip is non-blank only if one of the
# scenario's executed steps runs `page.goto`. But the LLM routinely emits later
# scenarios that ASSUME the app is already loaded (no goto), and both the dwelling
# coverage stub and a soft-failing resilient selector navigate NOWHERE — so such a
# scenario records an all-black clip → OpenCV Laplacian sharpness 0.0 → the
# orchestrator demo_video_record blank-gate HARD-FAILS the run (observed live,
# todomvc 2026-07-24: clips 3-6 blank while 1-2 rendered). Guarantee it here rather
# than trusting the LLM to open every scenario with a navigation step.
_NAV_HOOK_FILENAME = "steps_hooks.ts"


def _ensure_scenario_navigation(step_defs: list[_StepDef]) -> int:
    """Inject a playwright-bdd ``Before`` hook that navigates every scenario's
    fresh context to the app before its steps run, so no clip is
    blank-from-never-navigating. Appends a fresh ``steps_hooks.ts`` (its own
    ``createBdd()``, loaded by the ``steps/**/*.ts`` glob) so it never perturbs
    existing files or the dedupe/coverage passes — a ``Before`` hook is not a
    Given/When/Then, so every step-pattern pass ignores it. Navigates to the
    ``baseURL`` fixture (the FULL app URL from ``use.baseURL``) — NOT
    ``page.goto('/')``, which strips any sub-path off baseURL and records a blank
    clip. RETRIES the cold navigation and waits for a REAL painted DOM (not
    ``networkidle``, which an SPA/redirect reaches while still white) so clip-1 —
    the first nav against a cold app — does not record black.

    The retry loop is DEADLINE-bounded against the per-test Playwright timeout
    (``PW_TEST_TIMEOUT_MS``, exported by ``frontend_test_runner``): a
    playwright-bdd ``Before`` hook runs INSIDE the test, so a hook budget larger
    than that timeout is unreachable dead code and Playwright kills the test
    mid-hook — recording a fully black clip with zero executed steps. The hook
    therefore takes ~60% of the test budget (floor 5s), derives an EXPLICIT
    per-leg timeout for both ``goto`` and ``waitForFunction`` from the remaining
    time, and on exhaustion emits a ``##SOFT_FAIL##`` marker and THROWS, so the
    failure is attributable instead of a silent black clip. Idempotent +
    RECORDING stream only. Returns 1 if injected, else 0.
    """
    if any(sd.get("filename") == _NAV_HOOK_FILENAME for sd in step_defs):
        return 0
    content = "\n".join(
        [
            f"// filename: {_NAV_HOOK_FILENAME}",
            "// Auto-generated: every @clip-<n> scenario records in a FRESH browser",
            "// context that starts on about:blank (black). Navigate to the app before",
            "// each scenario so no clip is blank-from-never-navigating (a scenario",
            "// that never runs page.goto would score OpenCV sharpness 0.0 and the",
            "// demo_video_record blank-gate would hard-fail the run).",
            "import { createBdd } from 'playwright-bdd';",
            "import { test } from '@playwright/test';",
            "",
            "const { Before } = createBdd();",
            "",
            "// BUDGET COUPLING (do not raise blindly): a playwright-bdd `Before` hook runs",
            "// INSIDE the test, so the per-test Playwright `--timeout` bounds it. The runner",
            "// (frontend_test_runner_service) exports that same value as PW_TEST_TIMEOUT_MS.",
            "// A hook budget >= the test timeout is unreachable: Playwright kills the test",
            "// mid-hook, ZERO steps run, the page never leaves about:blank and the clip is",
            "// recorded FULLY BLACK (OpenCV sharpness 0.0 -> demo_video_record blank-gate",
            "// hard-fails the paid render). So take a fraction of the test budget and leave",
            "// the rest for the scenario's own steps.",
            "// The 5000ms floor must NEVER outrank the ceiling: FTR accepts a",
            "// timeout_per_test_ms as low as 1000, and a floor above the test timeout",
            "// recreates the very bug this hook was rewritten to fix. Clamp so the hook",
            "// always leaves the scenario at least 2s of its own.",
            "const PW_TEST_TIMEOUT_MS = Number(process.env.PW_TEST_TIMEOUT_MS) || 20000;",
            "const NAV_HOOK_BUDGET_MS = Math.max(",
            "  1000,",
            "  Math.min(PW_TEST_TIMEOUT_MS - 2000, Math.max(5000, Math.floor(PW_TEST_TIMEOUT_MS * 0.6))),",
            ");",
            "",
            "Before(async ({ page, baseURL }) => {",
            "  // Recorded video starts at CONTEXT CREATION (about:blank), which happens",
            "  // just before this hook runs — so `hookStartAt` is the best available proxy",
            "  // for the clip's t=0. Every ms this hook spends before the paint probe",
            "  // resolves is dead footage at the LEAD of the clip, the mirror image of the",
            "  // failed-scenario TAIL trimmed by frontend_test_runner_service/w2v_orchestrator",
            "  // (#166/#1042). That fix derives its cut from structured per-test evidence",
            "  // (the wait-timeout in the error), never from re-decoding pixels; do the same",
            "  // here via a `paint-lead` annotation (the same channel `resilientClick`/",
            "  // `resilientFill` already use for `soft-fail`) instead of asking the runner to",
            "  // analyze frames.",
            "  const hookStartAt = Date.now();",
            "  // Navigate to the FULL target URL. baseURL (from playwright.config",
            "  // `use.baseURL`) can carry a sub-path, e.g. https://host/todomvc;",
            "  // goto(baseURL) preserves it (goto('/') would strip it -> wrong/blank page).",
            "  // A fire-and-forget goto is NOT enough: a cold first navigation (SPA bundle",
            "  // download / server cold-start / auth redirect) can blow the nav timeout OR",
            "  // reach networkidle while the DOM is still blank/white -> the clip records",
            "  // black. So retry until the DEADLINE below, waiting for a REAL painted DOM.",
            "  // Every leg gets an EXPLICIT timeout derived from the time left, so no single",
            "  // attempt can overrun the budget (an implicit navigationTimeout would).",
            "  if (!baseURL) return;",
            "  const deadline = Date.now() + NAV_HOOK_BUDGET_MS;",
            "  let lastError = 'no attempt completed';",
            "  while (Date.now() < deadline) {",
            "    const remaining = deadline - Date.now();",
            "    if (remaining < 1000) break;",
            "    try {",
            "      await page.goto(baseURL, {",
            "        waitUntil: 'domcontentloaded',",
            "        timeout: Math.max(1000, Math.floor(remaining * 0.6)),",
            "      });",
            "      await page.waitForFunction(",
            "        () => {",
            "          // Cheap synchronous paint probe. Text is NOT required: a canvas-only,",
            "          // image-only or icon-only app paints perfectly with zero innerText,",
            "          // and demanding text there burns the whole budget on a healthy page.",
            "          const b = document.body;",
            "          if (!b) return false;",
            "          const box = b.getBoundingClientRect();",
            "          if (box.height <= 0 || box.width <= 0) return false;",
            "          if (b.innerText.trim().length > 0) return true;",
            "          // Media counts ONLY when it is actually RENDERED. A bare",
            "          // querySelector match is not paint: an inline <svg> icon sprite, a",
            "          // display:none logo, a 1x1 tracking pixel or a not-yet-decoded <img>",
            "          // all match on the first poll while the page is still white — which",
            "          // would wave through exactly the blank clip this hook exists to stop.",
            "          const MIN_PAINT_PX = 16;",
            "          const isPainted = (el) => {",
            "            const r = el.getBoundingClientRect();",
            "            if (r.width < MIN_PAINT_PX || r.height < MIN_PAINT_PX) return false;",
            "            const s = window.getComputedStyle(el);",
            "            if (s.display === 'none' || s.visibility === 'hidden') return false;",
            "            if (Number(s.opacity) === 0) return false;",
            "            if (el.tagName === 'IMG' && !(el.complete && el.naturalWidth > 0)) return false;",
            "            return true;",
            "          };",
            "          const media = b.querySelectorAll('canvas, img, svg, video');",
            "          for (let i = 0; i < media.length; i++) {",
            "            if (isPainted(media[i])) return true;",
            "          }",
            "          const els = b.querySelectorAll('*');",
            "          const limit = Math.min(els.length, 200);",
            "          for (let i = 0; i < limit; i++) {",
            "            const bg = window.getComputedStyle(els[i]).backgroundImage;",
            "            if (bg && bg !== 'none' && isPainted(els[i])) return true;",
            "          }",
            "          return false;",
            "        },",
            "        { timeout: Math.max(500, deadline - Date.now()) },",
            "      );",
            "      // Record how long this clip sat on an unpainted page before its first",
            "      // real frame, so the runner can trim exactly that much off the front of",
            "      // the recording — never more, never a guess from decoded pixels. Never",
            "      // fatal to the scenario: an annotation-bookkeeping failure must not turn",
            "      // a successful nav into a hard failure.",
            "      try {",
            "        test.info().annotations.push({",
            "          type: 'paint-lead',",
            "          description: String(Date.now() - hookStartAt),",
            "        });",
            "      } catch {",
            "        // not inside a running test (e.g. imported by a unit test) — the clip",
            "        // simply publishes untrimmed, byte-identical to today.",
            "      }",
            "      return;",
            "    } catch (err) {",
            "      // cold first paint / transient nav failure — retry while time remains",
            "      lastError = String(err);",
            "    }",
            "  }",
            "  // Budget exhausted. Do NOT fall through: a silent return records a black clip",
            "  // that the blank-gate later blames on the compiler. Emit the marker the runner",
            "  // parses, then fail the scenario attributably.",
            "  const detail = JSON.stringify({",
            "    action: 'navigate',",
            "    selector: baseURL,",
            "    label: 'nav-hook',",
            "    error: lastError,",
            "  });",
            "  console.warn(`##SOFT_FAIL## ${detail}`);",
            "  throw new Error(`nav hook: no paint within ${NAV_HOOK_BUDGET_MS}ms - ${detail}`);",
            "});",
            "",
        ]
    )
    step_defs.append({"filename": _NAV_HOOK_FILENAME, "content": content, "step_count": 0})
    return 1


# ── per-scenario pacing guarantee (clip length must not depend on a prompt) ───
# THE ASYMMETRY THAT WAS THE BUG. `lib/recording_helpers` ships both the pacing
# helpers (`heroDwell`) and the survivability helpers (`softExpectVisible`,
# `resilientClick`, `resilientFill`, `assertNavigated`). The survivability half is
# FORCE-INJECTED — `_ensure_navigation_preconditions` and `_ensure_resilient_import`
# rewrite call sites and guarantee the import deterministically. The pacing half was
# only ASKED FOR, in the generator prompt (`helpers_instruction`). A prompt rule is
# advisory, so the model complied on some suites and not others and demo duration
# came out BIMODAL: 55 runs recorded 16.7-36.9s per clip, 24 runs recorded
# 1.24-3.94s. Both report `status=completed` with pass_rate 100% — nothing in the
# suite can tell them apart — and the 18x shortfall only surfaces at the far end of
# the pipeline as `timeline GROSSLY UNDERFULL — 4.06s of the 30.00s target (13.5%)`,
# 12 production failures deep.
#
# Fix it the same way the survivability half is fixed: inject it. An `After` hook is
# the right seam because the dwell must be the clip's TAIL — the compositor shows
# each clip via clip_anchor="end" — and an After hook is by construction the last
# thing in the scenario, whatever steps the model wrote.
#
# ⚠️ The dwell length itself is NOT reduced to control total duration. HERO_DWELL_MS
# holds the screen WHILE THE NARRATION PLAYS and must stay >= the longest answer
# line for render's `-t` to trim cleanly; shortening it truncates the voiceover.
# Total duration is reconciled upstream by SCENARIO COUNT, off the value this module
# now publishes (`GUARANTEED_RECORDING_SECONDS_PER_SCENARIO`).
#
# IDEMPOTENCY, the obvious regression: a scenario whose steps ALREADY call
# `heroDwell` must not dwell twice and record ~44s. That is not solved by grepping
# the step files — a step definition is shared across scenarios, so "this file calls
# heroDwell" says nothing about whether THIS scenario did. It is solved at runtime by
# the per-page dwell LEDGER in `recording_helpers.ts`: every discretionary hold spends
# from one per-scenario budget of HERO_DWELL_MS, so the hook's call tops the scenario
# up to the budget and is a no-op when the steps already spent it. Injection is also
# idempotent at the pass level, by filename, exactly like the nav hook.
_PACING_HOOK_FILENAME = "steps_pacing.ts"


def _ensure_recording_pacing(step_defs: list[_StepDef]) -> int:
    """Inject a playwright-bdd ``After`` hook that holds every scenario's settled
    tail for the per-scenario dwell budget, so clip length stops depending on
    whether the LLM happened to honour a prompt rule.

    Appends a fresh ``steps_pacing.ts`` with its own ``createBdd()`` (loaded by the
    ``steps/**/*.ts`` glob) so it never perturbs existing files or the
    dedupe/coverage passes — an ``After`` hook is not a Given/When/Then. Idempotent
    (returns 0 when already present) and RECORDING stream only: in a QA suite a
    22-second pause per scenario is pure cost with no footage to show for it.
    Returns 1 if injected, else 0.
    """
    if any(sd.get("filename") == _PACING_HOOK_FILENAME for sd in step_defs):
        return 0
    content = "\n".join(
        [
            f"// filename: {_PACING_HOOK_FILENAME}",
            "// Auto-generated: hold every @clip-<n> scenario's settled tail for the",
            "// per-scenario dwell budget. The compositor shows each clip via",
            '// clip_anchor="end", so the tail IS the footage; without this the clip',
            "// length depended on whether the model honoured an advisory prompt rule",
            "// and came out bimodal (16.7-36.9s vs 1.24-3.94s, both reporting",
            "// status=completed) -> `timeline GROSSLY UNDERFULL` downstream.",
            "import { createBdd } from 'playwright-bdd';",
            "import { dwellRemainingMs, heroDwell } from '../lib/recording_helpers';",
            "",
            "const { After } = createBdd();",
            "",
            "// BUDGET COUPLING (same rule as the nav Before hook): a playwright-bdd",
            "// scenario hook runs INSIDE the test, so the per-test Playwright --timeout",
            "// bounds it. The runner (frontend_test_runner_service) exports that value as",
            "// PW_TEST_TIMEOUT_MS; the orchestrator's demo_video record stage defaults it",
            "// to 45000, comfortably above the 22000 budget. But FTR accepts values down",
            "// to 1000, and a dwell longer than the test timeout would have Playwright",
            "// kill the test MID-HOOK -> the clip is truncated or lost entirely. So clamp.",
            "const PW_TEST_TIMEOUT_MS = Number(process.env.PW_TEST_TIMEOUT_MS) || 20000;",
            "const DWELL_CEILING_MS = Math.max(0, PW_TEST_TIMEOUT_MS - 2000);",
            "",
            "After(async ({ page }) => {",
            "  // What the scenario still owes, after whatever its own steps already held.",
            "  const owed = dwellRemainingMs(page);",
            "  const grant = Math.min(owed, DWELL_CEILING_MS);",
            "  if (grant > 0) await heroDwell(page, grant);",
            "  // A clamp is a MEASUREMENT GAP: the clip is now shorter than the duration",
            "  // this package published to the orchestrator, and a silent short clip is",
            "  // indistinguishable from a healthy one (that is the whole defect being",
            "  // fixed). Say so on the channel the runner already parses.",
            "  if (grant < owed) {",
            "    const detail = JSON.stringify({",
            "      action: 'hero-dwell',",
            "      selector: 'steps_pacing.ts',",
            "      label: 'pacing-hook',",
            "      error:",
            "        `dwell clamped to ${grant}ms of ${owed}ms owed by ` +",
            "        `PW_TEST_TIMEOUT_MS=${PW_TEST_TIMEOUT_MS} - raise timeout_per_test_ms`,",
            "    });",
            "    console.warn(`##SOFT_FAIL## ${detail}`);",
            "  }",
            "});",
            "",
        ]
    )
    step_defs.append({"filename": _PACING_HOOK_FILENAME, "content": content, "step_count": 0})
    return 1


# ── ordered-duration budget (make the dwell PROPORTIONATE to what was ordered) ─
# The pacing hook above made clip length DETERMINISTIC — every scenario records
# HERO_DWELL_MS. Deterministic is not the same as proportionate: a 30s order
# whose features yielded 2 scenarios still recorded 2 x ~23.8s = 47.7s (159%),
# and no integer scenario count lands within +-10% of that order because one
# beat's footage is comparable to the whole order. So the ORDER itself has to
# reach the ledger. This module is how: it narrows the per-scenario dwell budget
# that `heroDwell` / `dwellWithScroll` / the After hook all spend from.
#
# A SEPARATE FILE, not a line in the pacing hook, for two reasons: the
# knowledge-pack showcase path ships `showcase.steps.ts` and never runs
# `_ensure_recording_pacing` at all (yet dwells via `Then I dwell for <ms> ms`,
# which spends from the same ledger), and a package with no ordered duration must
# keep the pacing hook byte-identical to what it shipped yesterday.
#
# ⛔ NEVER emitted without an ordered duration. The absent case must be today's
# behaviour exactly — no file, no import, no call.
_DWELL_BUDGET_FILENAME = "steps_dwell_budget.ts"


def ensure_dwell_budget(step_defs: list[_StepDef], budget_ms: int) -> int:
    """Inject the module that narrows the package's per-scenario dwell budget.

    Idempotent by filename, like every other injection pass. Returns 1 if
    injected, else 0 (already present, or a non-positive budget — a zero-length
    dwell is never what an order means).
    """
    if budget_ms <= 0:
        return 0
    if any(sd.get("filename") == _DWELL_BUDGET_FILENAME for sd in step_defs):
        return 0
    content = "\n".join(
        [
            f"// filename: {_DWELL_BUDGET_FILENAME}",
            "// Auto-generated: the caller ordered a total film duration, and this is",
            "// its per-scenario share. It NARROWS the ledger in lib/recording_helpers",
            "// (clamped there to HERO_DWELL_MS, so it can only shorten a clip, never",
            "// stretch one past what the per-test Playwright timeout allows).",
            "//",
            "// Duplicate frames inside the dwell are INTENTIONAL — the compositor shows",
            '// each clip via clip_anchor="end", so the settled tail IS the footage.',
            "// This makes that tail proportionate to the order; it does not strip it.",
            "import { setScenarioDwellBudgetMs } from '../lib/recording_helpers';",
            "",
            f"setScenarioDwellBudgetMs({budget_ms});",
            "",
        ]
    )
    step_defs.append({"filename": _DWELL_BUDGET_FILENAME, "content": content, "step_count": 0})
    return 1


# ── resilient interaction rewrite (a missing selector must not abort the clip) ─
# A bare ``page.click('sel')`` / ``page.fill('sel', v)`` THROWS when the selector
# never resolves, aborting the scenario → blank clip. Rewrite the simple bare
# forms to the shipped ``resilientClick`` / ``resilientFill`` helpers (which
# self-heal via accessible locators and soft-fail without throwing) and guarantee
# their import. Only the single-selector forms are rewritten; arbitrary locator
# chains are left to the LLM (rewriting them is fragile). RECORDING streams only —
# the ``lib/recording_helpers`` module ships only with the demo_video package.
# PER-HELPER import probes. A single combined `resilient(?:Click|Fill)` probe matched an
# import naming EITHER helper, so a file that imported only `resilientClick` and had its
# `page.fill` rewritten to `resilientFill` looked "already imported" and got nothing
# injected → `ReferenceError: resilientFill is not defined` at run time → the scenario
# aborts → black clip. `_undefined_module_bindings` cannot see it because the module IS
# imported. A rewrite must therefore guarantee ITS OWN binding, name by name.
_RESILIENT_IMPORT_RES = {
    name: re.compile(
        rf"import\s*\{{[^}}]*\b{name}\b[^}}]*\}}\s*from\s*['\"][^'\"]*recording_helpers['\"]"
    )
    for name in ("resilientClick", "resilientFill")
}
_RESILIENT_HELPERS = ("resilientClick", "resilientFill")


def _resilient_import_line(names: list[str]) -> str:
    """One `recording_helpers` import naming exactly ``names``.

    Only the MISSING bindings are named: re-importing a name the file already
    imports is a TS redeclaration error, which is how the naive "always import
    both" line would break a file that already imported one of them."""
    return f"import {{ {', '.join(names)} }} from '../lib/recording_helpers';"


# A SIMPLE argument: a quoted string literal or a (dotted) identifier. Both are
# always string selectors/values in a valid `page.click`/`page.fill` call, so
# rewriting them to the resilient helpers is safe. Anything else (a call
# expression, arithmetic, a `${}` template, a locator chain) does NOT match and is
# left untouched rather than risk corrupting it.
_SIMPLE_ARG = r"(?:['\"`][^'\"`]*['\"`]|[A-Za-z_$][\w$.]*)"
_BARE_CLICK_RE = re.compile(rf"\bawait\s+page\.click\(\s*({_SIMPLE_ARG})\s*\)")
_BARE_FILL_RE = re.compile(rf"\bawait\s+page\.fill\(\s*({_SIMPLE_ARG})\s*,\s*({_SIMPLE_ARG})\s*\)")


# A probe that never matches: for callers that have ALREADY decided the line is
# missing and only want `_inject_helper_import`'s placement logic.
_NEVER_RE = re.compile(r"(?!)")


def _inject_helper_import(content: str, present_re: re.Pattern[str], line: str) -> str:
    """Guarantee ``line`` (a `lib/recording_helpers` import) is present in ``content``.

    Inserted just after the `playwright-bdd` import when there is one, else at the
    top (below a leading `// filename:` comment, which downstream passes read).
    Returns ``content`` unchanged when ``present_re`` already matches.
    """
    if present_re.search(content):
        return content
    anchor = _PW_BDD_IMPORT_RE.search(content)
    if anchor:
        return content[: anchor.end()] + "\n" + line + content[anchor.end() :]
    lines = content.splitlines(keepends=True)
    at = 1 if lines and lines[0].lstrip().startswith("//") else 0
    lines.insert(at, line + "\n")
    return "".join(lines)


def _ensure_resilient_import(step_defs: list[_StepDef]) -> int:
    """Rewrite bare ``page.click``/``page.fill`` to the resilient helpers and
    guarantee the ``recording_helpers`` import. Mutates in place; returns the
    number of files changed. Call only for the recording (demo_video) stream.
    """
    changed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        new = _BARE_CLICK_RE.sub(r"await resilientClick(page, \1)", content)
        new = _BARE_FILL_RE.sub(r"await resilientFill(page, \1, \2)", new)
        # Guarantee EACH called helper's own binding — see `_RESILIENT_IMPORT_RES`.
        missing = [
            name
            for name in _RESILIENT_HELPERS
            if re.search(rf"\b{name}\s*\(", new) and not _RESILIENT_IMPORT_RES[name].search(new)
        ]
        if missing:
            new = _inject_helper_import(new, _NEVER_RE, _resilient_import_line(missing))
        if new != content:
            sd["content"] = new
            changed += 1
    return changed


# ── absolute in-step navigation (a relative goto strips baseURL's sub-path) ───
# The `Before` nav hook above navigates every fresh context to the FULL `baseURL`,
# but an LLM-emitted step body routinely then calls `await page.goto('/')` — which
# navigates AWAY from that URL to the bare ORIGIN. When `use.baseURL` carries a
# sub-path (e.g. https://host/todomvc) the origin is a 404 / marketing shell /
# blank page, so the clip records black even though the hook did its job. That is
# exactly why the hook alone (PR #92/#96) did NOT stop the blank clips: nothing
# sanitised the model's OWN navigation. Rewrite the relative forms
# deterministically here rather than trusting the prompt.
#
# `new URL(route, baseURL)` is the WRONG fix: a leading-slash route resolves
# against the origin, so it discards baseURL's path just like `goto('/')` does.
# The only sub-path-safe join is textual concatenation, which is what the emitted
# `absoluteNavUrl` helper does (and what the hook relies on by passing `baseURL`
# verbatim).
_ABS_NAV_HELPER = "absoluteNavUrl"
_ABS_NAV_HELPER_DECL_RE = re.compile(rf"\bfunction\s+{_ABS_NAV_HELPER}\s*\(")
_ABS_NAV_HELPER_SRC = "\n".join(
    [
        "// Auto-generated (blank-clip guard): resolve a RELATIVE in-step navigation",
        "// against the app's FULL base URL without dropping its own sub-path.",
        "// `new URL('/x', 'https://host/app')` must NOT be used here: a leading-slash",
        "// route resolves against the ORIGIN and silently discards `/app`, so the",
        "// recording captures the wrong (usually blank) page. Concatenate instead, and",
        "// treat the degenerate route '/' or '' as the base URL itself.",
        "//",
        "// Reads BASE_URL from the environment — the SAME source playwright.config.ts",
        "// already uses for `use.baseURL` (frontend_test_runner exports it as the full",
        "// target URL). Deliberately NOT the `baseURL` fixture: taking it from the",
        "// environment means this helper needs nothing added to any step's destructured",
        "// fixture argument, so the rewrite can never mangle an exotic destructure into",
        "// TypeScript that fails to compile — which would fail the WHOLE suite and",
        "// return zero clips, strictly worse than one wrong-page clip.",
        "// Read via `globalThis` rather than a bare `process`: the generated package has",
        "// NO @types/node (nothing in the @playwright/test tree provides it), so a bare",
        "// `process` only survives because Playwright transpiles without typechecking.",
        "// This form needs no ambient types and compiles under `strict`.",
        "// With BASE_URL unset the route is returned unchanged, i.e. exactly the",
        "// pre-existing Playwright behaviour.",
        f"function {_ABS_NAV_HELPER}(route: string): string {{",
        "  const env = (globalThis as { process?: { env?: Record<string, string | undefined> } })",
        "    .process?.env;",
        "  const base = (env?.BASE_URL ?? '').replace(/\\/+$/, '');",
        "  if (!base) return route;",
        "  const rel = route.replace(/^\\/+/, '');",
        "  return rel ? `${base}/${rel}` : base;",
        "}",
    ]
)
# A `page.goto(` whose first argument is a STRING literal (single/double quote or
# template). Anything else — an identifier, a call expression — is not safely
# classifiable as relative-vs-absolute, so it is skipped, not rewritten.
_GOTO_LITERAL_RE = re.compile(
    r"(?P<call>\bpage\.goto\(\s*)(?P<q>['\"`])(?P<url>(?:\\.|(?!(?P=q)).)*)(?P=q)"
)
# A `page.goto(` whose first argument is NOT a string literal — used for the
# skipped-count diagnostic only.
_GOTO_NONLITERAL_RE = re.compile(r"\bpage\.goto\(\s*(?!['\"`])(?P<arg>[^,)\n]*)")
# An absolute navigation target: a scheme (`https:`, `about:`, `data:`) or a
# protocol-relative `//host`. Those already carry their own origin+path and must
# be left byte-identical.
_ABSOLUTE_URL_RE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//)")
# A line whose `page.goto(` sits inside a `//` line comment — never rewrite there.
_LINE_COMMENT_RE = re.compile(r"//")


def _is_absolute_nav_target(url: str) -> bool:
    """True if a ``page.goto`` string literal already resolves on its own.

    Covers `http(s)://`, `about:blank`, `data:`/`file:`, protocol-relative
    `//host`, and any template literal that already interpolates ``baseURL``.
    """
    return bool(_ABSOLUTE_URL_RE.match(url)) or "baseURL" in url


def _count_relative_gotos(text: str) -> int:
    """Count ``page.goto`` calls still navigating to a baseURL-unsafe target.

    Both the literal-relative form and any non-literal argument (an identifier or
    call expression this pass refuses to touch) count — they are the residual
    blank-clip risk worth logging.
    """
    n = sum(
        1 for m in _GOTO_LITERAL_RE.finditer(text) if not _is_absolute_nav_target(m.group("url"))
    )
    n += sum(
        1
        for m in _GOTO_NONLITERAL_RE.finditer(text)
        if "baseURL" not in m.group("arg") and _ABS_NAV_HELPER not in m.group("arg")
    )
    return n


def _in_line_comment(text: str, pos: int) -> bool:
    """True if ``pos`` sits after a ``//`` on its own line."""
    line_start = text.rfind("\n", 0, pos) + 1
    comment = _LINE_COMMENT_RE.search(text, line_start, pos)
    return comment is not None


def _sub_relative_gotos(text: str) -> tuple[str, int]:
    """Rewrite relative ``page.goto('<literal>')`` to the sub-path-safe helper.

    Returns ``(new_text, rewrites)``. Absolute targets and commented-out calls are
    returned verbatim (the substitution reproduces the original match text), so a
    file with only absolute navigation stays byte-identical.

    The helper takes ONLY the route: it sources the base URL from
    ``process.env.BASE_URL``, so no step's destructured fixture argument is touched
    and the rewrite cannot produce non-compiling TypeScript.
    """
    rewrites = 0

    def _repl(m: re.Match[str]) -> str:
        nonlocal rewrites
        url = m.group("url")
        if _is_absolute_nav_target(url) or _in_line_comment(text, m.start()):
            return m.group(0)
        rewrites += 1
        q = m.group("q")
        return f"{m.group('call')}{_ABS_NAV_HELPER}({q}{url}{q})"

    return _GOTO_LITERAL_RE.sub(_repl, text), rewrites


def _insert_abs_nav_helper(content: str) -> str:
    """Add the module-scope ``absoluteNavUrl`` helper if absent (idempotent).

    Anchored exactly like ``_ensure_expect_import`` / ``_ensure_resilient_import``:
    after the ``playwright-bdd`` import, else after a leading ``// filename:``
    comment, else at the top. A ``function`` declaration hoists, so position never
    affects the call sites.
    """
    if _ABS_NAV_HELPER_DECL_RE.search(content):
        return content
    anchor = _PW_BDD_IMPORT_RE.search(content)
    if anchor:
        return content[: anchor.end()] + "\n" + _ABS_NAV_HELPER_SRC + content[anchor.end() :]
    lines = content.splitlines(keepends=True)
    at = 1 if lines and lines[0].lstrip().startswith("//") else 0
    lines.insert(at, _ABS_NAV_HELPER_SRC + "\n")
    return "".join(lines)


def _rewrite_relative_gotos(content: str) -> str:
    """Rewrite every relative ``page.goto`` string literal in one file.

    No scoping analysis is needed and none is done. Because the helper reads
    ``process.env.BASE_URL`` rather than the ``baseURL`` fixture, a rewritten call is
    valid EVERYWHERE — inside a step callback, inside a nested handler, and in a
    module-scope shared helper alike. That is the whole point of sourcing it from the
    environment: an earlier version spliced ``baseURL`` into each callback's
    destructured fixture argument and produced non-compiling TypeScript for six
    ordinary shapes (a prettier trailing comma, a multi-line destructure,
    ``...rest``, a ``baseURL: alias`` rename, a type annotation, and a nested
    ``async ({ request })`` handler) — which fails the whole suite and returns ZERO
    clips, strictly worse than the wrong-page clip it was fixing.

    Non-literal arguments and commented-out calls are left byte-identical; the caller
    logs the residue via ``_count_relative_gotos``.
    """
    return _sub_relative_gotos(content)[0]


def _ensure_absolute_navigation(step_defs: list[_StepDef]) -> int:
    """Make every in-step navigation preserve baseURL's sub-path.

    An LLM-emitted `await page.goto('/')` navigates AWAY from the full app URL the
    nav `Before` hook just opened, to the bare origin — a 404/blank page under a
    sub-pathed `use.baseURL` — so the clip records black and the
    `demo_video_record` blank-gate hard-fails the paid render. Rewrites the
    relative literal forms to `absoluteNavUrl('<route>')` and injects that helper,
    which sources the base URL from `process.env.BASE_URL` — so NO step's
    destructured fixture argument is modified and the rewrite cannot emit
    non-compiling TypeScript. Idempotent (the rewritten argument is no longer a bare
    string literal, and the helper injection is guarded). Logs its own outcome —
    both the rewrite and any unrewritable residue — so the next blank-clip incident
    is greppable in pod logs. Mutates in place; returns the number of files changed.
    Call only for the recording (demo_video) stream.
    """
    changed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        if "page.goto(" not in content:
            continue
        new = _rewrite_relative_gotos(content)
        if new != content:
            sd["content"] = _insert_abs_nav_helper(new)
            changed += 1
        residual = _count_relative_gotos(sd.get("content", ""))
        if residual:
            logger.warning(
                "Left %d baseURL-unsafe page.goto call(s) in %s unrewritten (no "
                "`baseURL` fixture in scope, or a non-literal argument) — a "
                "sub-pathed baseURL can still record a blank clip there",
                residual,
                sd.get("filename", "?"),
            )
    if changed:
        logger.warning(
            "Rewrote relative in-step page.goto to baseURL-absolute navigation in "
            "%d step file(s) (blank-clip guard)",
            changed,
        )
    return changed


# ── Playwright strict-mode disambiguation (>1 match THROWS, blanking a clip) ─
# Playwright locators are STRICT: `expect(page.getByRole('link', { name: 'TodoMVC' }))
# .toBeVisible()` throws `strict mode violation: … resolved to 2 elements` the moment
# the page has a second matching node, which aborts the scenario mid-recording — the
# clip records black and the orchestrator's blank-clip gate hard-fails the paid render.
# That is exactly how the 2026-07-31 `Gestalt Calibration Corpus` sweep lost 3/3
# renders (run 30610585336).
#
# The prompt now carries a `.first()` rule, but a prompt rule is ADVISORY — the model
# drifts back to the bare form. This pass is the compile-time guarantee: it appends
# `.first()` to every generated `page.getBy*` locator that is not already narrowed.
#
# Deliberately NOT a regex over the whole call. `getByRole('link', { name: /a(b)c/ })`
# nests parens, braces, quotes and regex literals, and a mangled step definition does
# not compile — which fails the WHOLE suite and returns ZERO clips, strictly worse than
# one strict-mode violation. So the pass scans balanced parens over a string/comment
# mask and only ever INSERTS a `.first()` at a position it has proven to be the end of
# a locator-returning expression. Correctness over coverage.
#
# RECORDING streams only. In the QA/accessibility/documentation streams a strict-mode
# violation is an HONEST test failure the operator must see; silently narrowing to the
# first match there would convert a real ambiguity defect into a false green (same
# reasoning as `_ensure_step_coverage`).
_STRICT_LOCATOR_CALL_RE = re.compile(
    r"\bpage\.(?:getByRole|getByText|getByLabel|getByPlaceholder)\s*\("
)
# Chained call in a locator expression: `.filter({…})`, `.first()`, `.click()`, …
_CHAIN_SEGMENT_RE = re.compile(r"\s*\.\s*([A-Za-z_$][\w$]*)\s*\(")
# Already disambiguated — anywhere in the chain means hands off entirely.
_DISAMBIGUATING_METHODS = frozenset({"first", "last", "nth"})
# Methods that return another Locator, so `.first()` may be appended AFTER them.
# Anything not listed (click, fill, press, textContent, waitFor, …) terminates the
# locator chain, and `.first()` is inserted BEFORE it — appending after an action
# would be `.click().first()`, i.e. `.first()` on a `Promise<void>`, which does not
# compile.
_LOCATOR_NARROWING_METHODS = frozenset(
    {
        "filter",
        "locator",
        "and",
        "or",
        "getByRole",
        "getByText",
        "getByLabel",
        "getByPlaceholder",
        "getByTestId",
        "getByTitle",
        "getByAltText",
    }
)
# Consumers that are only meaningful on a MULTI-element locator. Narrowing to
# `.first()` would silently change `toHaveCount(3)` into `toHaveCount(1)` and turn a
# passing assertion into a scenario-aborting failure — the very outcome this pass
# exists to prevent. Leave those locators exactly as the model wrote them.
# `.nth(n)` / `.last()` are here too: they index INTO the match set, so narrowing to
# the first match re-targets them at a different element (or at nothing).
_MULTI_ELEMENT_CONSUMER_RE = re.compile(
    r"\.\s*(?:toHaveCount|allTextContents|allInnerTexts|all|count|nth|last)\s*\("
)
# `const rows = page.getByRole('row')` — a locator bound to a name, whose
# count-sensitivity is decided by the LATER uses of that name, not by this statement.
_LOCATOR_BINDING_RE = re.compile(
    r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::\s*[^=;]+?)?=\s*(?:await\s+)?$"
)
# A `/` starts a regex literal only in expression position. After an identifier, a
# number, `)` or `]` it is division. Getting this wrong in the SAFE direction (calling
# a division a regex) is caught by the newline bail-out in `_mask_regex`.
_REGEX_START_PREV = frozenset("(,=:[!&|?{};+-*%~^<>")
_REGEX_START_KEYWORDS = frozenset(
    {"return", "typeof", "instanceof", "case", "in", "of", "do", "else", "await", "new", "yield"}
)
_TRAILING_IDENT_RE = re.compile(r"([A-Za-z_$][\w$]*)$")


def _mask_line_comment(text: str, mask: list[bool], i: int) -> int:
    """Mask a `//` comment through end-of-line; return the index after it."""
    n = len(text)
    while i < n and text[i] != "\n":
        mask[i] = False
        i += 1
    return i


def _mask_block_comment(text: str, mask: list[bool], i: int) -> tuple[int, bool]:
    """Mask a `/* … */` comment; return ``(index after `*/`, closed)``."""
    n = len(text)
    mask[i] = mask[i + 1] = False
    i += 2
    while i < n:
        mask[i] = False
        if text[i] == "*" and i + 1 < n and text[i + 1] == "/":
            mask[i + 1] = False
            return i + 2, True
        i += 1
    return i, False


def _mask_string(text: str, mask: list[bool], i: int, quote: str) -> tuple[int, bool]:
    """Mask a string literal (escapes honoured); return ``(index after it, closed)``."""
    n = len(text)
    mask[i] = False
    i += 1
    while i < n:
        mask[i] = False
        if text[i] == "\\":
            if i + 1 < n:
                mask[i + 1] = False
            i += 2
            continue
        if text[i] == quote:
            return i + 1, True
        i += 1
    return i, False


def _regex_can_start(text: str, i: int) -> bool:
    """Is the `/` at ``i`` in expression position (regex literal) rather than division?"""
    j = i - 1
    while j >= 0 and text[j] in " \t\r\n":
        j -= 1
    if j < 0:
        return True
    if text[j] in _REGEX_START_PREV:
        return True
    ident = _TRAILING_IDENT_RE.search(text[: j + 1])
    return ident is not None and ident.group(1) in _REGEX_START_KEYWORDS


def _mask_regex(text: str, mask: list[bool], i: int) -> int | None:
    """Mask a `/…/flags` literal; return the index after it, or ``None``.

    ``None`` means "this `/` is not a regex literal after all" — the scan hit a
    newline (a regex literal cannot span one) or end-of-text before the closing `/`.
    The caller then treats the `/` as ordinary code, so a misjudged division never
    blanks out the rest of the file. `[...]` character classes and backslash escapes
    are honoured, so `/[/]/` and `/\\//` terminate in the right place.
    """
    n = len(text)
    j = i + 1
    in_class = False
    while j < n:
        ch = text[j]
        if ch == "\n":
            return None
        if ch == "\\":
            j += 2
            continue
        if in_class:
            if ch == "]":
                in_class = False
        elif ch == "[":
            in_class = True
        elif ch == "/":
            end = j + 1
            while end < n and text[end].isalpha():
                end += 1
            for k in range(i, end):
                mask[k] = False
            return end
        j += 1
    return None


def _code_mask(text: str) -> tuple[list[bool], bool]:
    """Per-character mask (``True`` = CODE), plus a "the scan ended cleanly" flag.

    String literals (single/double/backtick, backslash escapes honoured), `//` line
    comments, `/* */` block comments and regex literals are masked ``False`` so a
    locator mentioned in prose, in a quoted selector, or a `)` inside a regex is
    never treated as code. A template literal is masked whole — its `${…}`
    interpolations are treated as non-code, which is the conservative direction (a
    locator built inside an interpolation is skipped, not mangled).

    The second element is ``False`` when the text ends inside an unterminated string
    or block comment. Everything after that point is masked non-code, so the caller
    would silently rewrite nothing; it logs instead of reporting a blind pass as
    "nothing to do".
    """
    mask = [True] * len(text)
    n = len(text)
    i = 0
    healthy = True
    while i < n:
        ch = text[i]
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i = _mask_line_comment(text, mask, i)
        elif ch == "/" and i + 1 < n and text[i + 1] == "*":
            i, closed = _mask_block_comment(text, mask, i)
            healthy = healthy and closed
        elif ch == "/" and _regex_can_start(text, i):
            i = _mask_regex(text, mask, i) or i + 1
        elif ch in "'\"`":
            i, closed = _mask_string(text, mask, i, ch)
            healthy = healthy and closed
        else:
            i += 1
    return mask, healthy


def _closing_paren(text: str, mask: list[bool], open_idx: int) -> int | None:
    """Index just past the `)` matching the `(` at ``open_idx``, or ``None``.

    Parens inside strings/comments (per ``mask``) do not count toward the depth.
    """
    depth = 0
    for i in range(open_idx, len(text)):
        if not mask[i]:
            continue
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _chain_segments(text: str, mask: list[bool], pos: int) -> list[tuple[str, int]]:
    """Parse the `.method(...)` chain starting at ``pos``.

    Returns ``[(method_name, index_just_past_its_closing_paren), …]``, stopping at the
    first thing that is not a balanced `.name(...)` call (a `)`, `;`, a property
    access, an unterminated call). Whitespace/newlines between links are allowed, so a
    prettier-wrapped chain parses the same as a one-liner.
    """
    segments: list[tuple[str, int]] = []
    while True:
        m = _CHAIN_SEGMENT_RE.match(text, pos)
        if m is None or not mask[m.start(1)]:
            return segments
        end = _closing_paren(text, mask, m.end() - 1)
        if end is None:
            return segments
        segments.append((m.group(1), end))
        pos = end


def _statement_tail(text: str, mask: list[bool], pos: int) -> str:
    """The rest of the STATEMENT at ``pos`` — through newlines, to its terminating `;`.

    Scoping this to a line would miss the normal prettier-wrapped shape
    ``await expect(page.getByRole('listitem'))\\n  .toHaveCount(2);`` and let the
    count-sensitivity guard append `.first()` to the very assertion it exists to
    protect. The scan tracks `([{` depth over the code mask and stops at the first
    `;` at depth ≤ 0, or at a `}` that closes the enclosing block (a statement with
    no semicolon, e.g. the last line of an arrow-function body).
    """
    depth = 0
    i = pos
    n = len(text)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth <= 0 and ch == "}":
                break
            depth -= 1
        elif ch == ";" and depth <= 0:
            i += 1
            break
        i += 1
    return text[pos:i]


def _escapes_into_opaque_call(text: str, mask: list[bool], start: int) -> bool:
    """Is the identifier at ``start`` passed as an argument to a call we cannot see into?

    ``expect(rows)`` is transparent — the matcher that decides count-sensitivity is
    right there in the same statement. Anything else (`assertRows(rows)`, a second
    argument) could consume the locator as a multi-element set inside a helper, so the
    locator is treated as count-sensitive and left alone.
    """
    j = start - 1
    while j >= 0 and text[j] in " \t\r\n":
        j -= 1
    if j < 0 or not mask[j] or text[j] not in "(,":
        return False
    if text[j] == ",":
        return True
    callee = _TRAILING_IDENT_RE.search(text[:j].rstrip())
    return callee is None or callee.group(1) != "expect"


def _binding_is_count_safe(text: str, mask: list[bool], name: str, after: int) -> bool:
    """Are ALL later uses of the bound locator ``name`` count-INsensitive?

    ``const rows = page.getByRole('row'); await expect(rows).toHaveCount(3);`` must not
    be narrowed: the tail of the binding statement cannot see through the name. Every
    subsequent use is checked instead, and anything not provably safe — a
    count-sensitive consumer, or the name escaping into a call this pass cannot read —
    vetoes the rewrite. Re-declaration/shadowing is not modelled; extra matches can
    only ever suppress a rewrite, which is the safe direction.
    """
    uses = re.compile(rf"\b{re.escape(name)}\b")
    for m in uses.finditer(text, after):
        if not mask[m.start()]:
            continue
        if _MULTI_ELEMENT_CONSUMER_RE.search(_statement_tail(text, mask, m.end())):
            return False
        if _escapes_into_opaque_call(text, mask, m.start()):
            return False
    return True


def _locator_insert_point(text: str, mask: list[bool], match: re.Match[str]) -> int | None:
    """Where `.first()` belongs for the ``page.getBy*`` call at ``match``, or ``None``.

    ``None`` = leave this locator byte-identical (already narrowed, unbalanced parens,
    consumed as a multi-element set, or bound to a name whose later uses are
    count-sensitive).
    """
    end = _closing_paren(text, mask, match.end() - 1)
    if end is None:
        return None
    segments = _chain_segments(text, mask, end)
    if any(name in _DISAMBIGUATING_METHODS for name, _ in segments):
        return None
    insert_at = end
    for name, seg_end in segments:
        if name not in _LOCATOR_NARROWING_METHODS:
            break
        insert_at = seg_end
    if _MULTI_ELEMENT_CONSUMER_RE.search(_statement_tail(text, mask, insert_at)):
        return None
    binding = _LOCATOR_BINDING_RE.search(text[: match.start()])
    if binding is not None and not _binding_is_count_safe(text, mask, binding.group(1), insert_at):
        return None
    return insert_at


def _sub_strict_safe_locators(text: str) -> tuple[str, int, bool]:
    """Append `.first()` to every ambiguous ``page.getBy*`` locator expression.

    Returns ``(new_text, rewrites, mask_healthy)``. A locator is left byte-identical
    when it is already narrowed (`.first()`/`.last()`/`.nth(n)` anywhere in its chain),
    when it sits inside a string literal, comment or regex, when its parens are
    unbalanced, when its STATEMENT (not merely its line) consumes it as a multi-element
    locator (`toHaveCount`, `count()`, `all()`, `nth()`, `last()`), or when it is bound
    to a name that is later used that way. Idempotent: a rewritten chain now contains
    `first`, so a second pass skips it.

    Insertion point is the END of the locator-returning prefix of the chain, so
    `page.getByRole('row').filter({…}).click()` becomes
    `page.getByRole('row').filter({…}).first().click()` — never `.click().first()`.
    """
    mask, healthy = _code_mask(text)
    out: list[str] = []
    cursor = 0
    rewrites = 0
    for m in _STRICT_LOCATOR_CALL_RE.finditer(text):
        if m.start() < cursor or not mask[m.start()]:
            continue
        insert_at = _locator_insert_point(text, mask, m)
        if insert_at is None:
            continue
        out.append(text[cursor:insert_at])
        out.append(".first()")
        cursor = insert_at
        rewrites += 1
    out.append(text[cursor:])
    return "".join(out), rewrites, healthy


def _ensure_strict_safe_locators(step_defs: list[_StepDef]) -> int:
    """Make every generated ``page.getBy*`` locator survive Playwright strict mode.

    A locator that resolves to more than one element THROWS, aborting the scenario
    mid-recording so the clip records black and the blank-clip gate hard-fails the
    paid render (live: run 30610585336, 3/3 renders lost to
    `getByRole('link', { name: 'TodoMVC' }) resolved to 2 elements`). Mutates in
    place; returns the number of files changed. Call only for the recording
    (demo_video) stream — elsewhere a strict violation is an honest test failure.
    """
    changed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        if "page.getBy" not in content:
            continue
        new, rewrites, healthy = _sub_strict_safe_locators(content)
        if not healthy:
            logger.warning(
                "Strict-locator scan of %s ended inside an unterminated string or comment; "
                "locators after that point were NOT checked",
                sd.get("filename", "<unnamed>"),
            )
        if rewrites:
            sd["content"] = new
            changed += 1
    return changed


# ── navigation preconditions must assert NAVIGATION, not decoration ──────────
# THE most expensive brittleness measured in production: the landing `Given` of a
# generated scenario asserted a NAMED element instead of navigation —
#     Given('I am on the TodoMVC landing page', async ({ page }) => {
#       await expect(page.getByRole('link', { name: 'TodoMVC', exact: true }).first())
#         .toBeVisible();
#     });
# TodoMVC has no such link (its heading is `todos`), so step 1 THREW, the whole
# scenario aborted, every later When/Then NEVER RAN, and the clip recorded a
# static landing page with ZERO interaction. Live: two demo_video runs recorded
# no interaction at all; 112/112 demo masters static, median
# duplicate_frame_ratio 0.905.
#
# The root cause is structural, not a bad completion: the prompt's page inventory
# is built from the PRE-interaction inspection snapshot, so any element that only
# exists AFTER a user acts (a todo row, a filter chip, a toast) is necessarily
# invented by the model. A precondition step can therefore only honestly assert
# that NAVIGATION SUCCEEDED. Rewrite its hard `expect(...).toBeVisible()` into the
# non-fatal `softExpectVisible` (still records a soft-fail the runner demotes the
# scenario for — see `recording_helpers.softFail`) and append a real
# `assertNavigated(page)`, which THROWS on `about:blank`/no body. Net effect: a
# wrong decorative selector costs ONE annotation instead of the whole scenario,
# and a genuine navigation failure still hard-fails.
#
# RECORDING streams only. In a QA/accessibility suite a failing `expect` is an
# HONEST test failure the operator must see (same rule as `_ensure_step_coverage`
# and `_ensure_strict_safe_locators`).
_GIVEN_CALL_RE = re.compile(r"\bGiven\s*\(")
# A precondition sentence about being/getting somewhere: "I am on …", "I'm on …",
# "I open …", "I visit …", "I go to …", "I navigate to …", "I land on …".
_NAV_PRECONDITION_RE = re.compile(
    r"^\s*i\s*(?:'m\s+|am\s+)?(?:on|open|visit|view|go\s+to|navigate\s+to|land\s+on)\b",
    re.IGNORECASE,
)
# A body that navigates is a navigation precondition whatever its sentence says.
_NAV_CALL_IN_BODY_RE = re.compile(r"\b(?:page\.goto|absoluteNavUrl)\s*\(")
# The callback must destructure the `page` fixture, or the injected helper calls
# would reference an undefined name.
_PAGE_FIXTURE_RE = re.compile(r"async\s*\(\s*\{[^})]*\bpage\b")
# NOTE: no `\A` — `pattern.match(text, pos)` already anchors at `pos`, whereas
# `\A` would assert offset 0 of the whole file and never match.
_TO_BE_VISIBLE_TAIL_RE = re.compile(r"\s*\.\s*toBeVisible\s*(\()")
_AWAIT_PREFIX_RE = re.compile(r"await\s+\Z")
_ASSERT_NAVIGATED_RE = re.compile(r"\bassertNavigated\s*\(")
_SOFT_NAV_IMPORT_RE = re.compile(
    r"import\s*\{[^}]*\bsoftExpectVisible\b[^}]*\}\s*from\s*['\"][^'\"]*recording_helpers['\"]"
)
_SOFT_NAV_IMPORT_LINE = (
    "import { assertNavigated, softExpectVisible } from '../lib/recording_helpers';"
)


def _string_literal_at(text: str, pos: int) -> tuple[str, int] | None:
    """Read the quoted string literal starting at the first non-space char at/after
    ``pos``. Returns ``(value, end_offset)``; ``None`` when there is no literal
    (e.g. a `/regex/` pattern), which callers treat as "not classifiable".
    """
    i = pos
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] not in "'\"`":
        return None
    quote = text[i]
    out: list[str] = []
    i += 1
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            out.append(text[i + 1])
            i += 2
            continue
        if ch == quote:
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    return None


def _nav_given_bodies(text: str, mask: list[bool]) -> list[tuple[int, int]]:
    """Spans ``(body_start, close_paren)`` of every ``Given(...)`` call in ``text``
    that is a NAVIGATION PRECONDITION — either its sentence says so or its body
    actually navigates. ``body_start`` is just past the opening paren and
    ``close_paren`` indexes the matching ``)`` itself (``_closing_paren`` returns
    the offset just PAST it).
    """
    spans: list[tuple[int, int]] = []
    for m in _GIVEN_CALL_RE.finditer(text):
        if not mask[m.start()]:
            continue
        open_idx = m.end() - 1
        past = _closing_paren(text, mask, open_idx)
        if past is None:
            continue
        close = past - 1
        body = text[m.end() : close]
        literal = _string_literal_at(text, m.end())
        sentence = literal[0] if literal is not None else ""
        if not _NAV_PRECONDITION_RE.search(sentence) and not _NAV_CALL_IN_BODY_RE.search(body):
            continue
        if not _PAGE_FIXTURE_RE.search(body):
            continue
        spans.append((m.end(), close))
    return spans


def _soften_visibility_asserts(
    text: str, mask: list[bool], start: int, end: int
) -> list[tuple[int, int, str]]:
    """Replacements turning every ``await expect(X).toBeVisible(...)`` inside
    ``[start, end)`` into ``await softExpectVisible(page, X)``.

    Deliberately narrow: only the POSITIVE `.toBeVisible` assertion is softened.
    `.not.toBeVisible()`, `toHaveURL`, `toHaveText`, `toHaveCount` and every other
    matcher are left byte-identical — softening a negative or a content assertion
    would hide a real defect rather than a brittle selector.
    """
    repls: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\bexpect\s*\(", text[start:end]):
        abs_start = start + m.start()
        if not mask[abs_start]:
            continue
        if _AWAIT_PREFIX_RE.search(text[max(0, abs_start - 12) : abs_start]) is None:
            continue
        arg_open = start + m.end() - 1
        past_arg = _closing_paren(text, mask, arg_open)
        if past_arg is None or past_arg > end:
            continue
        tail = _TO_BE_VISIBLE_TAIL_RE.match(text, past_arg)
        if tail is None:
            continue
        past_matcher = _closing_paren(text, mask, tail.start(1))
        if past_matcher is None or past_matcher > end:
            continue
        arg = text[arg_open + 1 : past_arg - 1].strip()
        if not arg or arg == "page":
            continue
        repls.append((abs_start, past_matcher, f"softExpectVisible(page, {arg})"))
    return repls


def _append_navigation_assert(text: str, close: int) -> tuple[int, str] | None:
    """Insertion ``(offset, snippet)`` placing ``await assertNavigated(page);`` at the
    END of the callback body closing at ``close`` (after the goto and after the
    softened checks), or ``None`` when the body's shape is not the plain
    ``(pattern, async (…) => { … })`` form.
    """
    i = close - 1
    while i > 0 and (text[i].isspace() or text[i] == ","):
        i -= 1
    if i <= 0 or text[i] != "}":
        return None
    line_start = text.rfind("\n", 0, i) + 1
    indent = text[line_start:i]
    inner = (indent if indent.strip() == "" else "  ") + "  "
    return i, f"{inner}await assertNavigated(page);\n{indent if indent.strip() == '' else ''}"


def _nav_precondition_edits(text: str, mask: list[bool]) -> list[tuple[int, int, str]]:
    """``(start, end, replacement)`` edits softening the decorative visibility
    asserts of every navigation ``Given`` in ``text`` and appending an
    ``assertNavigated(page)`` to each one that lost a hard assertion.

    Empty when no navigation ``Given`` hard-asserts an element — a step that never
    made a brittle claim keeps its body byte-identical.
    """
    edits: list[tuple[int, int, str]] = []
    for body_start, close in _nav_given_bodies(text, mask):
        softened = _soften_visibility_asserts(text, mask, body_start, close)
        if not softened:
            continue
        edits.extend(softened)
        if _ASSERT_NAVIGATED_RE.search(text[body_start:close]) is not None:
            continue
        appended = _append_navigation_assert(text, close)
        if appended is not None:
            at, snippet = appended
            edits.append((at, at, snippet))
    return edits


def _ensure_navigation_preconditions(step_defs: list[_StepDef]) -> int:
    """Make every navigation ``Given`` assert NAVIGATION, not a named element.

    Mutates in place; returns the number of files changed. Call only for the
    recording (demo_video) stream. Idempotent: a rewritten body contains
    `softExpectVisible`/`assertNavigated`, and neither matches the patterns above.
    """
    changed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        if "Given" not in content:
            continue
        mask, healthy = _code_mask(content)
        if not healthy:
            logger.warning(
                "Navigation-precondition scan of %s ended inside an unterminated string "
                "or comment; later Given bodies were NOT checked",
                sd.get("filename", "<unnamed>"),
            )
        edits = _nav_precondition_edits(content, mask)
        if not edits:
            continue
        new = content
        for at, upto, snippet in sorted(edits, key=lambda e: e[0], reverse=True):
            new = new[:at] + snippet + new[upto:]
        sd["content"] = _inject_helper_import(new, _SOFT_NAV_IMPORT_RE, _SOFT_NAV_IMPORT_LINE)
        changed += 1
    return changed


# ── a role NAME must come from the PAGE, not from the app's name ─────────────
# Measured on orchestrator E2E run 32287791747 against demo.playwright.dev/todomvc:
#     soft-failed selector(s): {"action":"expect-visible",
#       "selector":"getByRole('heading', { name: /TodoMVC/ }).first()",
#       "error":"TimeoutError: locator.waitFor: Timeout 2500ms exceeded"}
# BOTH scenarios were demoted over that one line. The page's only heading is
# `todos`; the model inferred the accessible name from the SITE name it had been
# told ("TodoMVC") instead of reading it off the inventory — where `todos` was
# sitting the whole time. #119's "NEVER INVENT A SELECTOR" prompt rule did not
# stop it (prompt rules are advisory), and #119's `softExpectVisible` only made
# the miss non-fatal; the assertion is still pure cost. So ground it
# deterministically, the same shape as #121's `_ensure_no_undefined_value_assertions`:
# an assertion that CANNOT pass is worse than no assertion, so DROP it. Never
# substitute a guessed name — inventing `todos` here would be the same class of
# error the model just made.
#
# Scope is deliberately one step kind: the NAVIGATION `Given` (the exact spans
# `_ensure_navigation_preconditions` already targets, via the shared
# `_nav_given_bodies`). That step runs against the landing page, which IS what the
# PRE-INTERACTION snapshot describes, so the inventory is authoritative there and
# ONLY there. A `When`/`Then` legitimately names elements that appear only after
# interaction (a todo row, a filter chip, a toast); judging those against a
# pre-interaction snapshot would delete correct assertions.
#
# Three more no-touch rules, each closing a way this pass could do harm:
#   * inventory EMPTY/absent for the run  -> change NOTHING. A missing snapshot is
#     not evidence of absence, and letting it delete assertions would make a KAG
#     scrape failure silently strip the suite.
#   * role the inventory never captures   -> change NOTHING. `extract_structure`
#     only surfaces headings + `_INTERACTIVE_TAGS`, so it says nothing about
#     `img`/`banner`/`alert`; a role with zero inventory names is a role this pass
#     is not authoritative about, not an absent name.
#   * a name this pass cannot READ or EVALUATE (a variable, a `${…}` interpolation,
#     a regex using a JS-only construct / too long / that `re.compile` rejects)
#     -> UNKNOWN, change NOTHING. Failing to parse a name says nothing about the
#     page, only about this pass. #125 dropped on an unevaluable regex; that was the
#     same mistake as trusting a capped inventory, and is fixed.
_GET_BY_ROLE_CALL_RE = re.compile(r"\bgetByRole\s*\(")
_NAME_OPTION_RE = re.compile(r"\bname\s*:\s*")
_EXACT_TRUE_RE = re.compile(r"\bexact\s*:\s*true\b")
# `name: 'Save ' + label` — the literal this pass can read is only a PREFIX of the
# real name, so a substring test against it can answer ABSENT for a name the page
# has. Same class as the unevaluable regex: not determinable ⇒ UNKNOWN ⇒ keep.
_CONCAT_TAIL_RE = re.compile(r"\s*\+")
# The role token inside an inventory element's `getByRole("role", { name: … })`
# selector — `html_structure._selector_for` is the only thing that emits these, and
# it restricts the role to `_ROLE_RE` (a plain lowercase word).
_INVENTORY_ROLE_RE = re.compile(r"getByRole\(\s*[\"']([a-z]+)[\"']")
# JS regex constructs Python's `re` does not accept or does not mean the same thing
# by. A name pattern using one is NOT evaluated — it is treated as absent.
_UNSAFE_JS_REGEX_RE = re.compile(r"\\[pPku]|\(\?<[=!]|\\\d")
_MAX_EVALUABLE_PATTERN = 200
# A statement this pass is willing to delete. Restricting the drop to assertions
# means a stray `getByRole` inside a CLICK/FILL action is never removed — losing an
# action would blank the very footage the recording stream exists to produce.
_DROPPABLE_ASSERT_RE = re.compile(r"\b(?:expect|softExpectVisible)\s*\(")
# ANY claim a nav `Given` can still be making after this pass has run. Wider than
# `_DROPPABLE_ASSERT_RE` because `assertNavigated(page)` is exactly the assertion
# this pass BACKFILLS and must therefore also recognise as already present.
_ANY_ASSERT_RE = re.compile(r"\b(?:expect|softExpectVisible|assertNavigated)\s*\(")
# Sentinel role key: "interactive_elements was capped, so every role derived from
# it is unreliable". Not a real ARIA role, and `_ROLE_RE`/`_INVENTORY_ROLE_RE` only
# ever yield plain lowercase words, so it can never collide with one.
_ELEMENTS_CAPPED = "*elements-capped*"


# Typographic variants a CMS/word-processor emits where the generated spec (and a
# hand-written one) uses the ASCII original: a curly apostrophe in the page's
# "What's new" vs `'` in the assertion is a pure encoding difference, and letting
# it read as ABSENT deletes a correct assertion. Deliberately narrow: quotes,
# dashes and exotic spaces only — NO accent stripping and NO general unicode
# folding, either of which would make genuinely different names compare equal.
_TYPOGRAPHIC_MAP = {
    ord("\u2018"): "'",  # left single quote
    ord("\u2019"): "'",  # right single quote / apostrophe
    ord("\u201a"): "'",  # single low-9 quote
    ord("\u201b"): "'",  # single high-reversed-9 quote
    ord("\u2032"): "'",  # prime
    ord("\u201c"): '"',  # left double quote
    ord("\u201d"): '"',  # right double quote
    ord("\u201e"): '"',  # double low-9 quote
    ord("\u201f"): '"',  # double high-reversed-9 quote
    ord("\u2033"): '"',  # double prime
    ord("\u2010"): "-",  # hyphen
    ord("\u2011"): "-",  # non-breaking hyphen
    ord("\u2012"): "-",  # figure dash
    ord("\u2013"): "-",  # en dash
    ord("\u2014"): "-",  # em dash
    ord("\u2015"): "-",  # horizontal bar
    ord("\u2212"): "-",  # minus sign
    ord("\u00a0"): " ",  # no-break space
    ord("\u2007"): " ",  # figure space
    ord("\u202f"): " ",  # narrow no-break space
    ord("\u200b"): "",  # zero-width space
    ord("\ufeff"): "",  # zero-width no-break space / BOM
}


def _norm_name(value: str) -> str:
    """Typographic-fold + whitespace-collapse + casefold, matching Playwright's
    case-insensitive accessible-name matching and `html_structure._clean`'s
    normalization. See `_TYPOGRAPHIC_MAP` for why the fold is there and how far
    it deliberately does NOT go."""
    return " ".join(value.translate(_TYPOGRAPHIC_MAP).split()).casefold()


def _inventory_role_names(inspection: Mapping[str, Any] | None) -> dict[str, set[str]]:
    """Accessible names the PRE-INTERACTION snapshot proves exist, keyed by ARIA role.

    Reads exactly two carriers, which is everything the inventory has:
    ``page["headings"]`` (``list[str]`` of heading TEXT — the role is implicitly
    ``heading``) and ``page["interactive_elements"]``, whose ``label`` / ``text`` /
    ``name`` carry the accessible name and whose ROLE is recoverable only from the
    ``getByRole("<role>", …)`` selector `html_structure` emitted for it.

    The union across pages is deliberate: a navigation `Given` may land on any page
    the snapshot covers, and a union can only ever KEEP an assertion, never delete
    one a narrower reading would have kept. Empty dict = no usable inventory.
    """
    roles: dict[str, set[str]] = {}
    if not isinstance(inspection, dict):
        return roles
    element_roles: set[str] = set()
    # Role buckets this inventory is NOT authoritative about. `_ELEMENTS_CAPPED` is a
    # sentinel standing for "every role that came out of interactive_elements".
    untrusted: set[str] = set()
    pages = inspection.get("pages")
    for page in pages if isinstance(pages, list) else []:
        if isinstance(page, dict):
            _collect_page_role_names(page, roles, element_roles, untrusted)
    if _ELEMENTS_CAPPED in untrusted:
        untrusted |= element_roles
    for role in untrusted:
        roles.pop(role, None)
    return roles


def _positive_int(value: object) -> int:
    """``value`` as a non-negative int; 0 for anything that is not an int count.

    ``bool`` is excluded on purpose — a `True` here would silently read as 1.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _collect_page_role_names(
    page: Mapping[str, Any],
    roles: dict[str, set[str]],
    element_roles: set[str],
    untrusted: set[str],
) -> None:
    """Fold one inventory page's headings + interactive elements into ``roles``, and
    record in ``untrusted`` every role bucket this page proves is NOT exhaustive.

    Two ways a bucket stops being evidence of absence:

    * the list was CAPPED (`html_structure._MAX_ELEMENTS` / `_MAX_HEADINGS`, surfaced
      as `elements_truncated` / `headings_truncated`). Element #41 of a link-heavy
      landing page is on the page and simply not in the list; reading the cap as an
      exhaustive inventory deletes an assertion that would have passed.
    * a NAME was capped at `LABEL_LENGTH_CAP`. The recorded name is then a PREFIX of
      the real one, so Playwright's substring test (`wanted in name`) can answer
      ABSENT for a name the page really has.
    """
    if _positive_int(page.get("headings_truncated")):
        untrusted.add("heading")
    if _positive_int(page.get("elements_truncated")):
        untrusted.add(_ELEMENTS_CAPPED)
    headings = page.get("headings")
    for heading in headings if isinstance(headings, list) else []:
        if isinstance(heading, str) and heading.strip():
            roles.setdefault("heading", set()).add(_norm_name(heading))
    _collect_page_element_names(page, roles, element_roles, untrusted)


def _collect_page_element_names(
    page: Mapping[str, Any],
    roles: dict[str, set[str]],
    element_roles: set[str],
    untrusted: set[str],
) -> None:
    """Fold one page's ``interactive_elements`` into ``roles``; see the caller."""
    elements = page.get("interactive_elements")
    for el in elements if isinstance(elements, list) else []:
        if not isinstance(el, dict):
            continue
        selector = el.get("selector")
        found = _INVENTORY_ROLE_RE.search(selector) if isinstance(selector, str) else None
        if found is None:
            continue
        role = found.group(1)
        element_roles.add(role)
        bucket = roles.setdefault(role, set())
        for candidate in (el.get("label"), el.get("text"), el.get("name")):
            if isinstance(candidate, str) and candidate.strip():
                if len(candidate) >= LABEL_LENGTH_CAP:
                    untrusted.add(role)
                bucket.add(_norm_name(candidate))


def _js_regex_literal_at(text: str, pos: int) -> tuple[str, str] | None:
    """Read the ``/body/flags`` regex literal starting at ``pos``; ``None`` if absent."""
    if pos >= len(text) or text[pos] != "/":
        return None
    i = pos + 1
    in_class = False
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "\n":
            return None
        if ch == "[":
            in_class = True
        elif ch == "]":
            in_class = False
        elif ch == "/" and not in_class:
            end = i + 1
            while end < len(text) and text[end].isalpha():
                end += 1
            return text[pos + 1 : i], text[i + 1 : end]
        i += 1
    return None


def _regex_name_is_present(body: str, names: set[str]) -> bool | None:
    """Whether a JS regex name pattern matches any inventory name, case-insensitively.

    ``None`` when the pattern cannot be evaluated SAFELY (empty, a JS-only construct
    Python's `re` does not accept or does not mean the same thing by, an over-long
    body, anything `re.compile` rejects) — i.e. UNKNOWN, which KEEPS the assertion.

    #125 returned ``False`` here (= ABSENT = drop), on the reasoning that guessing an
    unreadable pattern matches is the invention this pass exists to remove. That was
    wrong, and wrong for the same reason a capped inventory or a lossy parser is:
    ABSENCE IS ONLY EVIDENCE WHEN WE ACTUALLY LOOKED. Failing to parse a pattern is
    not looking — it says nothing about the page, only about this function. Keeping
    an assertion we cannot judge costs at most one soft-failed check; dropping it
    deletes a correct assertion with no log and no way to notice.

    A pattern that DOES evaluate and genuinely matches nothing still returns
    ``False``: that is a real, computed answer against a bucket the caller has
    already established is exhaustive.
    """
    if not body or len(body) > _MAX_EVALUABLE_PATTERN or _UNSAFE_JS_REGEX_RE.search(body):
        return None
    try:
        compiled = re.compile(body, re.IGNORECASE)
    except re.error:
        return None
    return any(compiled.search(name) is not None for name in names)


def _role_name_verdict(
    text: str, mask: list[bool], call_start: int, roles: dict[str, set[str]]
) -> bool | None:
    """``False`` when this ``getByRole(...)`` names an element the inventory does not
    have, ``True`` when the inventory confirms it, ``None`` when the call is not
    judgeable (no role literal, no `name:` option, a role the inventory never
    captures, or a name that is not a readable literal).
    """
    open_idx = call_start + text[call_start:].index("(")
    role_literal = _string_literal_at(text, open_idx + 1)
    if role_literal is None:
        return None
    names = roles.get(role_literal[0].strip().lower())
    if not names:
        return None
    span = _role_options_span(text, mask, role_literal[1])
    if span is None:
        return None
    return _name_option_verdict(text, span, names)


def _role_options_span(text: str, mask: list[bool], after_role: int) -> tuple[int, int] | None:
    """``(open_brace, past_close_brace)`` of the options object literal following the
    role argument, or ``None`` when there is no plain ``, { … }`` second argument."""
    i = after_role
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != ",":
        return None
    i += 1
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    close = _closing_brace(text, mask, i)
    return None if close is None else (i, close)


def _name_option_verdict(text: str, span: tuple[int, int], names: set[str]) -> bool | None:
    """Evaluate the options object's ``name:`` value against the inventory ``names``."""
    start, close = span
    options = text[start:close]
    opt = _NAME_OPTION_RE.search(options)
    if opt is None:
        return None
    value_at = start + opt.end()
    literal = _string_literal_at(text, value_at)
    if literal is not None:
        if "${" in text[value_at : literal[1]]:
            return None
        if _CONCAT_TAIL_RE.match(text, literal[1]) is not None:
            return None
        wanted = _norm_name(literal[0])
        if not wanted:
            return None
        if _EXACT_TRUE_RE.search(options):
            return wanted in names
        # Playwright's default name matching is substring + case-insensitive.
        return any(wanted in name for name in names)
    pattern = _js_regex_literal_at(text, value_at)
    if pattern is None:
        return None
    return _regex_name_is_present(pattern[0], names)


def _closing_brace(text: str, mask: list[bool], open_idx: int) -> int | None:
    """Index just past the ``}`` matching the ``{`` at ``open_idx``, or ``None``.
    Braces inside strings/comments/regexes (per ``mask``) do not count."""
    depth = 0
    for i in range(open_idx, len(text)):
        if not mask[i]:
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _drop_assert_statement_at(text: str, mask: list[bool], idx: int) -> tuple[str, bool]:
    """Remove the whole assertion statement containing offset *idx*.

    Returns ``(new_text, dropped)``. The statement is bounded by the previous
    code-position ``;``/``{``/``}`` and the next code-position ``;``, and its
    indentation and trailing newline go with it. Refuses (``False``) when the
    statement does not read as an assertion — see ``_DROPPABLE_ASSERT_RE``.
    """
    start = 0
    for i in range(idx - 1, -1, -1):
        if mask[i] and text[i] in ";{}":
            start = i + 1
            break
    end = -1
    for i in range(idx, len(text)):
        if mask[i] and text[i] == ";":
            end = i + 1
            break
    if end == -1:
        return text, False
    if _DROPPABLE_ASSERT_RE.search(text[start:end]) is None:
        return text, False
    # Remove from the start of the statement's OWN line when only indentation precedes
    # it there; `start` itself sits just past the previous `;`/`{` (usually on the line
    # above), and cutting from it would glue the next line onto that one.
    stmt = start + len(text[start:end]) - len(text[start:end].lstrip())
    line_start = text.rfind("\n", 0, stmt) + 1
    if text[line_start:stmt].strip():
        line_start = start
    while end < len(text) and text[end] in " \t":
        end += 1
    if end < len(text) and text[end] == "\n":
        end += 1
    return text[:line_start] + text[end:], True


def _drop_ungrounded_role_names(text: str, roles: dict[str, set[str]]) -> tuple[str, int, int]:
    """Drop every nav-``Given`` assertion naming a role the inventory contradicts.

    Returns ``(new_text, dropped, backfilled)``. Re-derives the mask and the
    navigation spans after each drop so offsets are never stale. Idempotent: a
    dropped assertion is gone, and a surviving one is grounded.

    A ``getByRole`` whose statement `_drop_assert_statement_at` REFUSES (a
    `const loc = page.getByRole(...)` binding, an action) is skipped, not returned
    on — returning would let one such line mask every later ungrounded assertion in
    the file, which is silent under-enforcement with no log line anywhere.
    """
    dropped = 0
    refused: set[int] = set()
    while True:
        mask, _ = _code_mask(text)
        target = _next_ungrounded_role_call(text, mask, roles, refused)
        if target is None:
            break
        new_text, ok = _drop_assert_statement_at(text, mask, target)
        if not ok:
            refused.add(target)
            continue
        text = new_text
        dropped += 1
        # Every surviving offset just shifted; a previously refused one is re-tested
        # against the new text (and simply refused again) rather than mis-skipped.
        refused.clear()
    if not dropped:
        return text, 0, 0
    text, backfilled = _backfill_navigation_asserts(text)
    return text, dropped, backfilled


def _next_ungrounded_role_call(
    text: str, mask: list[bool], roles: dict[str, set[str]], refused: set[int]
) -> int | None:
    """Offset of the first nav-``Given`` ``getByRole(`` the inventory contradicts and
    that has not already been refused, or ``None``."""
    for body_start, close in _nav_given_bodies(text, mask):
        for m in _GET_BY_ROLE_CALL_RE.finditer(text, body_start, close):
            if not mask[m.start()] or m.start() in refused:
                continue
            if _role_name_verdict(text, mask, m.start(), roles) is False:
                return m.start()
    return None


def _backfill_navigation_asserts(text: str) -> tuple[str, int]:
    """Guarantee no navigation ``Given`` was left asserting NOTHING, appending
    ``await assertNavigated(page);`` to any that now has zero claims.

    A `Given` reduced to a bare `page.goto(...)` always passes and proves nothing —
    the vacuous-test failure mode. `_ensure_navigation_preconditions` only appends
    `assertNavigated` to steps whose `.toBeVisible()` it SOFTENED, so a nav `Given`
    whose only assertion was `toHaveText` / `toBeAttached` / `toHaveCount` never got
    one, and this pass is free to delete that single assertion. Adding the honest
    assertion is preferred over declining the drop: the dropped claim is one the
    inventory PROVES cannot pass, so keeping it would only re-fail the step, whereas
    `assertNavigated(page)` is true of every step that just navigated.
    """
    added = 0
    while True:
        mask, _ = _code_mask(text)
        vacuous = next(
            (
                close
                for body_start, close in _nav_given_bodies(text, mask)
                if _ANY_ASSERT_RE.search(text[body_start:close]) is None
            ),
            None,
        )
        if vacuous is None:
            return text, added
        appended = _append_navigation_assert(text, vacuous)
        if appended is None:
            # Body shape this pass cannot safely edit — stop rather than loop forever.
            return text, added
        at, snippet = appended
        text = text[:at] + snippet + text[at:]
        added += 1


def _ensure_grounded_role_names(
    step_defs: list[_StepDef], inspection: Mapping[str, Any] | None
) -> int:
    """Drop navigation-``Given`` assertions whose ``getByRole`` name is not in the
    scraped page inventory. Mutates in place; returns the number of files changed.

    Returns 0 without touching anything when the inventory is empty or absent — and
    likewise for any role bucket the inventory proves is not exhaustive (a capped
    element/heading list, a name truncated at ``LABEL_LENGTH_CAP``).

    A file whose last assertion was dropped gets ``assertNavigated(page)`` backfilled
    (and the helper import with it), so this pass can never leave a step asserting
    nothing.
    """
    roles = _inventory_role_names(inspection)
    if not roles:
        return 0
    changed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        if "getByRole" not in content or "Given" not in content:
            continue
        new, dropped, backfilled = _drop_ungrounded_role_names(content, roles)
        if not dropped:
            continue
        if backfilled:
            new = _inject_helper_import(new, _SOFT_NAV_IMPORT_RE, _SOFT_NAV_IMPORT_LINE)
            logger.warning(
                "Navigation Given in %s lost its last assertion to the grounding pass; "
                "backfilled assertNavigated(page) in %d step(s) (vacuous-step guard)",
                sd.get("filename", "<unnamed>"),
                backfilled,
            )
        sd["content"] = new
        changed += 1
    return changed


# ── survivable text assertions (exact equality blanks a clip) ───────────────
# `toHaveText` is EXACT equality on the element's full text. Against TodoMVC the
# model wrote `await expect(page.locator('.todo-count').first()).toHaveText("2 items")`
# — but that counter always reads "2 items left", so the assertion can NEVER pass. The
# scenario aborted mid-recording, the clip recorded black, and the whole demo render
# died as a poison job. The model was not wrong about the SCREEN; it was wrong about
# the surrounding copy, which is the failure mode every counter/label/status string
# invites.
#
# For a RECORDING package the trade is one-sided: surviving to produce footage is worth
# far more than assertion precision. A substring match still proves the right screen is
# up, and it does not turn stray trailing copy into a lost paid render. So every plain
# string-literal `toHaveText` becomes `toContainText`.
#
# RECORDING streams only — in the QA/accessibility/documentation streams an exact-text
# mismatch is an HONEST test failure the operator must see, and loosening it there would
# convert a real content defect into a false green (same reasoning as
# `_ensure_strict_safe_locators`).
#
# `arg` is the first non-space character after the paren, and it decides everything:
#   `'` `"` `` ` ``  → a plain string literal; this is the brittle case, rewrite it.
#   `/`              → a regex literal; the author already chose a fuzzy match on
#                      purpose, and `toContainText(/…/)` is not the same predicate.
#   `[`              → the multi-element array form, whose `toContainText` semantics
#                      differ (per-element substring vs. exact list); leave it alone.
#   anything else    → a variable/expression this pass cannot read. Skipping is safe.
_TO_HAVE_TEXT_CALL_RE = re.compile(r"\.\s*(toHaveText)\s*\(\s*(?P<arg>[\s\S])")
_SURVIVABLE_TEXT_ARG_QUOTES = frozenset("'\"`")
# A NEGATED assertion must be left byte-identical, and the reason is the exact inverse
# of the bug above: `not.toHaveText("2 items")` PASSES against "2 items left" (the texts
# differ), while `not.toContainText("2 items")` FAILS against it (the substring is
# there). Rewriting a negated assertion would CREATE the scenario-aborting failure this
# pass exists to prevent.
_NEGATED_ASSERTION_RE = re.compile(r"\.\s*not\s*$")


def _sub_survivable_text_assertions(text: str) -> tuple[str, int, bool]:
    """Rewrite exact-equality ``toHaveText`` into substring ``toContainText``.

    Returns ``(new_text, rewrites, mask_healthy)``. An assertion is left byte-identical
    when its argument is not a plain string literal (a regex literal or an array literal
    was a deliberate choice, and a variable cannot be read here), when it is negated
    (`expect(x).not.toHaveText(…)`, where the rewrite would INVERT into a failure), or
    when it sits inside a string literal, comment or regex. Idempotent: a rewritten call
    no longer contains `toHaveText`, so a second pass skips it.
    """
    mask, healthy = _code_mask(text)
    out: list[str] = []
    cursor = 0
    rewrites = 0
    for m in _TO_HAVE_TEXT_CALL_RE.finditer(text):
        if m.start() < cursor or not mask[m.start()]:
            continue
        if m.group("arg") not in _SURVIVABLE_TEXT_ARG_QUOTES:
            continue
        if _NEGATED_ASSERTION_RE.search(text[: m.start()]):
            continue
        out.append(text[cursor : m.start(1)])
        out.append("toContainText")
        cursor = m.end(1)
        rewrites += 1
    out.append(text[cursor:])
    return "".join(out), rewrites, healthy


# Matchers whose expected value MUST be a string/regex at call time. Playwright
# raises a *matcher* error ("expected value must be a string or regular expression")
# when the argument is `undefined`, which aborts the step, kills the scenario, and
# leaves the recording on a static page — measured live on orchestrator E2E run
# 32150382725 against TodoMVC:
#     expect(locator('ul.todo-list')).toContainText(expected)
#     Matcher error: ... Expected has value: undefined
_VALUE_MATCHERS = ("toContainText", "toHaveText", "toHaveValue", "toHaveAttribute")
_VALUE_MATCHER_CALL_RE = re.compile(
    r"\.\s*(?P<matcher>" + "|".join(_VALUE_MATCHERS) + r")\s*\(\s*(?P<arg>[A-Za-z_$][\w$]*)\s*[,)]"
)
# Every cucumber-expression placeholder that passes a positional argument to the
# callback. This must NOT be a hand-listed subset: it used to read
# `\{(?:string|int|word|float)\}`, which missed the anonymous `{}` plus `{double}`,
# `{long}` and `{byte}` — all four honoured by `_CUKE_PARAM_REGEX` above. A
# `Then('total is {double}', async ({page}, v) => …)` therefore counted ZERO
# placeholders, judged `v` permanently `undefined` and made
# `_ensure_no_undefined_value_assertions` SILENTLY DELETE the assertion, leaving a
# green `Then` that asserts nothing.
#
# So match the anonymous `{}` and ANY `{name}` rather than a list: a custom
# (registered) parameter type binds an argument too, and over-counting placeholders
# is the only safe direction here — it can make the unbound-param check more
# conservative, never make it delete more. `_CUKE_PARAM_REGEX`'s whole key set is
# covered by construction (every key is an identifier, and "" is the `{}` form);
# `test_placeholder_vocabulary_is_derived_from_the_param_regex_table` pins that so
# the two cannot drift apart again.
_CUKE_PLACEHOLDER_RE = re.compile(r"\{(?:[A-Za-z_][A-Za-z0-9_]*)?\}")
# `async ({ page }, a, b) => {` — the fixture object, then the positional params that
# cucumber fills from the placeholders, in order.
_STEP_CALLBACK_RE = re.compile(r",\s*async\s*\(\s*\{[^}]*\}\s*(?P<rest>[^)]*)\)")
_AWAIT_EXPECT_STMT_RE = re.compile(r"^[ \t]*(?:await\s+)?expect\s*\(", re.MULTILINE)


def _unbound_positional_params(pattern: str, header: str, *, is_regex: bool) -> set[str]:
    """Positional callback params that cucumber can NEVER fill, so they are always
    ``undefined`` at runtime.

    A step's positional arguments come from its cucumber-expression placeholders, in
    order. A callback that declares MORE positional params than the pattern has
    placeholders leaves the surplus permanently ``undefined`` — a regex pattern is
    treated as opaque (capture groups are not counted here) and therefore never
    reported, so this can only ever flag a provable case.
    """
    if is_regex:
        return set()
    m = _STEP_CALLBACK_RE.search(header)
    if not m:
        return set()
    names = [n.strip() for n in m.group("rest").split(",") if n.strip()]
    names = [n for n in names if re.fullmatch(r"[A-Za-z_$][\w$]*", n)]
    return set(names[len(_CUKE_PLACEHOLDER_RE.findall(pattern)) :])


def _drop_statement_at(text: str, idx: int) -> tuple[str, bool]:
    """Remove the whole ``await expect(...)…;`` statement containing offset *idx*."""
    start = None
    for m in _AWAIT_EXPECT_STMT_RE.finditer(text):
        if m.start() > idx:
            break
        start = m.start()
    if start is None:
        return text, False
    end = text.find(";", idx)
    if end == -1:
        return text, False
    end += 1
    while end < len(text) and text[end] == "\n":
        end += 1
    return text[:start] + text[end:], True


def _first_undefined_value_arg(text: str, mask: list[bool]) -> int | None:
    """Offset of the first value-matcher argument that is provably ``undefined``."""
    pattern: str | None = None
    header = ""
    pattern_is_regex = False
    for line_start, line in _iter_lines_with_offsets(text):
        found = _step_def_pattern_on_line(line)
        if found is not None:
            pattern = found
            header = text[line_start : line_start + 400]
            m_def = _STEP_DEF_RE.match(line.lstrip())
            pattern_is_regex = bool(m_def and m_def.group("re") is not None)
        if pattern is None:
            continue
        unbound = _unbound_positional_params(pattern, header, is_regex=pattern_is_regex)
        if not unbound:
            continue
        for m in _VALUE_MATCHER_CALL_RE.finditer(line):
            off = line_start + m.start("arg")
            if off < len(mask) and not mask[off]:
                continue
            if m.group("arg") in unbound:
                return off
    return None


def _sub_undefined_value_assertions(text: str) -> tuple[str, int]:
    """Drop value-matcher assertions whose expected argument is provably ``undefined``.

    Dropping is deliberate. Substituting ``?? ""`` would make the assertion vacuously
    PASS, which is the silent-green class this stream exists to avoid; an ABSENT
    assertion is strictly better than one that always raises a matcher error and
    destroys the recording. A literal (string/regex/array/number) argument, and any
    identifier the pattern actually binds, are left byte-identical.
    """
    dropped = 0
    while True:
        mask, _ = _code_mask(text)
        off = _first_undefined_value_arg(text, mask)
        if off is None:
            return text, dropped
        text, ok = _drop_statement_at(text, off)
        if not ok:
            return text, dropped
        dropped += 1


def _iter_lines_with_offsets(text: str) -> Iterator[tuple[int, str]]:
    off = 0
    for line in text.splitlines(keepends=True):
        yield off, line
        off += len(line)


def _ensure_no_undefined_value_assertions(step_defs: list[_StepDef]) -> int:
    """Strip assertions that can only ever raise a Playwright matcher error.

    Returns the number of files changed. Safe for every stream: an assertion whose
    expected value is provably ``undefined`` cannot pass anywhere, so removing it
    never hides a real product failure — it only stops one broken line from aborting
    the scenario that surrounds it.
    """
    changed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        if not content:
            continue
        new, dropped = _sub_undefined_value_assertions(content)
        if dropped:
            sd["content"] = new
            changed += 1
    return changed


def _ensure_survivable_text_assertions(step_defs: list[_StepDef]) -> int:
    """Keep a generated text assertion from aborting the recording over trailing copy.

    `toHaveText("2 items")` against TodoMVC's "2 items left" counter can never pass, so
    the scenario dies mid-recording, the clip records black, and the demo render is lost
    as a poison job. Relax plain string-literal comparisons to a substring match, which
    still proves the right screen is up. Mutates in place; returns the number of files
    changed. Call only for the recording (demo_video) stream — elsewhere an exact-text
    mismatch is an honest failure the operator must see.
    """
    changed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        if "toHaveText" not in content:
            continue
        new, rewrites, healthy = _sub_survivable_text_assertions(content)
        if not healthy:
            logger.warning(
                "Text-assertion scan of %s ended inside an unterminated string or comment; "
                "assertions after that point were NOT checked",
                sd.get("filename", "<unnamed>"),
            )
        if rewrites:
            sd["content"] = new
            changed += 1
    return changed


# ── cucumber parameter-type normalization ───────────────────────────────────
# The LLM routinely invents a `{number}` (also `{num}`/`{numeric}`/`{integer}`/
# `{decimal}`) cucumber parameter type. cucumber-expressions 18.0.1 has NO such
# built-in (the built-ins are exactly `_CUKE_PARAM_REGEX`'s keys), and no scaffold
# registers one via `defineParameterType`, so bddgen aborts the whole compile with
# `Undefined parameter type 'number'` → 0 clips render. Rewrite the invented numeric
# types to the built-in `{float}` (its regex is a superset that also matches integers),
# the same deterministic-self-heal shape as the dedupe / subsumed-literal passes.
_INVENTED_NUM_PARAM_RE = re.compile(r"\{(number|num|numeric|integer|decimal)\}")


def _normalize_param_types(step_defs: list[_StepDef]) -> int:
    """Rewrite non-built-in cucumber parameter types the LLM invents (e.g. ``{number}``)
    to the nearest built-in so bddgen never throws ``Undefined parameter type``. Mutates
    in place; returns the number of files changed."""
    changed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        new = _INVENTED_NUM_PARAM_RE.sub("{float}", content)
        if new != content:
            sd["content"] = new
            changed += 1
    return changed


# ── Gherkin continuation keywords are NOT playwright-bdd bindings ────────────
# `createBdd()` exposes exactly Given/When/Then (+ the Before/After hooks). `And`,
# `But` and the `*` bullet are Gherkin CONTINUATION keywords: they inherit the kind
# of the preceding Given/When/Then. The LLM, shown a step list that echoed the raw
# feature keywords, emitted a literal `And('...', async ({ page }) => {...});` — an
# undefined symbol. The step file throws `ReferenceError: And is not defined` at
# module load, bddgen exits 1 and the ENTIRE paid render dies (observed live
# 2026-08-03, Gestalt corpus run 30793758726). Repair it deterministically: bind
# each continuation call to the kind of the preceding definition in the same file.
_CONTINUATION_BINDING_NAMES = frozenset(k for k in CONTINUATION_KEYWORDS if k.isidentifier())
# A module-scope (column 0) statement-position call of a Capitalised identifier —
# the shape every playwright-bdd binding call has (`Given(`, `When(`, `Before(`).
_MODULE_SCOPE_CALL_RE = re.compile(r"^(?P<name>[A-Z][\w$]*)\s*\(", re.MULTILINE)
# Names a module brings into scope: destructured (`const { Given, When } = createBdd()`),
# plain declarations, functions, classes and every import form.
_DESTRUCTURE_RE = re.compile(r"(?:const|let|var)\s*\{(?P<names>[^}]*)\}\s*=")
_DECL_RE = re.compile(r"(?:^|\s)(?:const|let|var|function|class)\s+(?P<name>[A-Za-z_$][\w$]*)")
_IMPORT_CLAUSE_RE = re.compile(r"^import\s+(?P<clause>[^'\"]+?)\s+from\s*['\"]", re.MULTILINE)

# COLUMN-0-anchored variants of the two regexes above, for DUPLICATE detection
# only (never for name resolution). `_DESTRUCTURE_RE`/`_DECL_RE` deliberately
# match anywhere (`(?:^|\s)` / no anchor at all) because over-collecting names
# only makes resolution MORE permissive, which is harmless. Counting
# OCCURRENCES with that same unanchored pair is a different use entirely: an
# ordinary indented `const row = ...` inside TWO separate step callbacks (each
# a normal top-level `Given(`/`When(` call closed at column 0, so their bodies
# are indented) would register as a "duplicate" and false-positive the
# compile gate on nearly every real generated file. True module-scope
# declarations in generated step files sit at column 0; locals inside a step
# callback body are indented. Anchoring on that shape (optionally preceded by
# `export `) is what actually means "module scope" here.
_MODULE_SCOPE_DESTRUCTURE_ANCHORED_RE = re.compile(
    r"^(?:export\s+)?(?:const|let|var)\s*\{(?P<names>[^}]*)\}\s*=", re.MULTILINE
)
_MODULE_SCOPE_DECL_ANCHORED_RE = re.compile(
    r"^(?:export\s+)?(?:async\s+)?(?:const|let|var|function|class)\s+(?P<name>[A-Za-z_$][\w$]*)",
    re.MULTILINE,
)

_UNDEFINED_BINDING_CORRECTION = (
    "\n\nCORRECTION — your previous output CALLED a step binding that is not defined "
    "or imported anywhere in the file ({names}), which throws a `ReferenceError` at "
    "module load and makes bddgen exit 1. `createBdd()` returns ONLY `Given`, `When`, "
    "`Then` (plus the `Before`/`After` hooks). `And`, `But` and `*` are Gherkin "
    "CONTINUATION keywords, NOT bindings: a step written `And I click save` after a "
    "`When` step must be defined as `When('I click save', ...)`. Never write `And(` or "
    "`But(`. Regenerate accordingly.\n"
)


def _resolve_continuation_calls(content: str) -> tuple[str, int]:
    """Rewrite module-scope ``And(``/``But(`` calls to the preceding binding kind.

    Returns ``(new_content, rewrites)``. Inheritance follows the same rule as Gherkin
    itself (:func:`~src.services.gherkin_keywords.resolve_binding`): the kind of the
    nearest preceding Given/When/Then definition in the file, falling back to
    ``Given`` when the continuation comes first. Only CODE positions are touched — a
    literal ``And(`` inside a string or comment is left alone.
    """
    mask, _ = _code_mask(content)
    last_primary: str | None = None
    out: list[str] = []
    cursor = 0
    rewrites = 0
    for match in re.finditer(r"^(?P<name>[A-Za-z_$][\w$]*)(?=\s*\()", content, re.MULTILINE):
        if not mask[match.start()]:
            continue
        name = match.group("name")
        if name in PRIMARY_KEYWORDS:
            last_primary = name
            continue
        if name not in _CONTINUATION_BINDING_NAMES:
            continue
        binding = resolve_binding(name, last_primary, default=DEFAULT_BINDING)
        out.append(content[cursor : match.start()])
        out.append(binding)
        cursor = match.end()
        rewrites += 1
    if not rewrites:
        return content, 0
    out.append(content[cursor:])
    return "".join(out), rewrites


def _ensure_valid_step_bindings(step_defs: list[_StepDef]) -> int:
    """Bind every ``And``/``But`` step-def call to the preceding Given/When/Then kind.

    Mutates ``step_defs`` in place; returns the number of files changed.
    """
    changed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        new, rewrites = _resolve_continuation_calls(content)
        if rewrites:
            sd["content"] = new
            changed += 1
    return changed


def _module_scope_name_occurrences(content: str) -> list[str]:
    """Every module-scope-shaped binding OCCURRENCE (declared, destructured, imported).

    Duplicates are preserved (unlike `_module_scope_names`), but this parser is
    intentionally UNANCHORED (`_DESTRUCTURE_RE`/`_DECL_RE` match anywhere, not
    just at column 0) — used only for NAME RESOLUTION, where over-collecting a
    name from inside a callback only makes resolution more permissive, which is
    harmless. Do NOT use this for duplicate/redeclaration counting: an ordinary
    `const row = ...` local repeated across two separate step callback bodies
    would register as a false "duplicate". See `_module_scope_occurrences_anchored`
    for that.
    """
    names: list[str] = []
    for match in _DESTRUCTURE_RE.finditer(content):
        for part in match.group("names").split(","):
            # `{ Given: G }` / `{ Given as G }` — the usable name is the last token.
            tokens = re.findall(r"[A-Za-z_$][\w$]*", part)
            if tokens:
                names.append(tokens[-1])
    names.extend(m.group("name") for m in _DECL_RE.finditer(content))
    for match in _IMPORT_CLAUSE_RE.finditer(content):
        clause = match.group("clause")
        for part in re.split(r"[,{}]", clause):
            tokens = re.findall(r"[A-Za-z_$][\w$]*", part)
            tokens = [t for t in tokens if t not in {"as", "type"}]
            if tokens:
                names.append(tokens[-1])
    return names


def _module_scope_names(content: str) -> set[str]:
    """Every identifier the module brings into scope (declared, destructured, imported)."""
    return set(_module_scope_name_occurrences(content))


def _module_scope_occurrences_anchored(content: str) -> list[str]:
    """Every binding OCCURRENCE actually declared at COLUMN 0 (true module scope).

    Column-0-anchored counterpart to `_module_scope_name_occurrences`, for
    DUPLICATE detection only. Generated step files always emit each
    Given/When/Then as a top-level call closed at column 0, so a real
    module-scope declaration sits at column 0 and a step-body local sits
    indented — anchoring on that (optionally preceded by `export `) is what
    "module scope" means for this generator. Reuses `_IMPORT_CLAUSE_RE`
    unchanged: it is already `^`-anchored, i.e. already column-0-only.

    Masked via `_code_mask` before matching: a line beginning with `const`/
    `let`/`function`/… INSIDE a template literal or a `/* */` block comment
    (e.g. a JSDoc usage example, or a multi-line SQL/banner string) is text,
    not a declaration, and must never be counted as a module-scope
    occurrence — `_MODULE_SCOPE_DESTRUCTURE_ANCHORED_RE`/
    `_MODULE_SCOPE_DECL_ANCHORED_RE` are plain `^`-anchored regexes with no
    knowledge of string/comment context, so without this filter two such
    lines false-positive `_reject_duplicate_module_scope_bindings` on
    perfectly valid, compiling TypeScript (proven live: a template literal
    or block comment repeating `const row = …` twice raised
    `StepDefGenerationError` and killed the whole generation).
    """
    mask, _ = _code_mask(content)
    names: list[str] = []
    for match in _MODULE_SCOPE_DESTRUCTURE_ANCHORED_RE.finditer(content):
        if not mask[match.start()]:
            continue
        for part in match.group("names").split(","):
            tokens = re.findall(r"[A-Za-z_$][\w$]*", part)
            if tokens:
                names.append(tokens[-1])
    names.extend(
        m.group("name") for m in _MODULE_SCOPE_DECL_ANCHORED_RE.finditer(content) if mask[m.start()]
    )
    for match in _IMPORT_CLAUSE_RE.finditer(content):
        if not mask[match.start()]:
            continue
        clause = match.group("clause")
        for part in re.split(r"[,{}]", clause):
            tokens = re.findall(r"[A-Za-z_$][\w$]*", part)
            tokens = [t for t in tokens if t not in {"as", "type"}]
            if tokens:
                names.append(tokens[-1])
    return names


def _duplicate_module_scope_bindings(content: str) -> list[str]:
    """Module-scope (column-0) identifiers declared/destructured/imported more than once.

    Returns names in first-duplicate-seen order, each listed once. This is the
    check `_undefined_module_bindings` cannot make: a repaired subset destructure
    (`const { Given, Then } = createBdd();`) sitting beside a second, full one
    (`const { Given, When, Then } = createBdd();`) resolves every name fine under
    name-resolution alone — the redeclaration of `Given`/`Then` only shows up when
    occurrences, not just membership, are counted. Uses the COLUMN-0-anchored
    occurrence scan, not the permissive one `_module_scope_names` uses — an
    ordinary indented `const row = ...` local repeated across two different step
    callbacks is valid JS and must NOT be flagged.
    """
    seen: set[str] = set()
    dups: list[str] = []
    for name in _module_scope_occurrences_anchored(content):
        if name in seen:
            if name not in dups:
                dups.append(name)
        else:
            seen.add(name)
    return dups


def _undefined_module_bindings(step_defs: list[_StepDef]) -> list[str]:
    """Locate module-scope calls to a binding the file never defines or imports.

    This is the compile gate for generated TypeScript. A unit test asserting the
    emitted string "contains `Given(`" proves nothing — the live failure was an
    EXTRA call to a symbol that was never in scope. Here we recover the set of names
    the module actually brings into scope and flag any column-0, statement-position
    call of a Capitalised identifier outside it (`And(`, `But(`, or any other
    hallucinated binding). Restricted to module scope and Capitalised names — the
    shape of every playwright-bdd binding call — so a locally-scoped helper inside a
    callback can never be a false positive.

    Returns ``["<filename>: And", …]``, empty when every call resolves.
    """
    problems: list[str] = []
    for sd in step_defs:
        content = sd.get("content", "")
        mask, _ = _code_mask(content)
        defined = _module_scope_names(content)
        for match in _MODULE_SCOPE_CALL_RE.finditer(content):
            if not mask[match.start()]:
                continue
            name = match.group("name")
            if name in defined:
                continue
            entry = f"{sd.get('filename', '?')}: {name}"
            if entry not in problems:
                problems.append(entry)
    return problems


def _reject_duplicate_module_scope_bindings(step_defs: list[_StepDef]) -> None:
    """Compile gate: no generated file may redeclare a module-scope binding.

    `_undefined_module_bindings` is name-resolution only (a set), and
    `_step_def_issues` is regex heuristics — neither catches a redeclared
    `const`/`let`/`var`/`function`/`class`/`import`. Babel throws an unattributed
    `VarRedeclaration` for this before a single assertion runs, and the job
    workspace is `shutil.rmtree`'d in a `finally`, so this diagnosis (file +
    identifier) is otherwise unrecoverable after the fact. Raising here — rather
    than repairing — is correct: today these runs already die at `bddgen`, so
    failing earlier with a usable message is the same outcome, sooner.
    """
    for sd in step_defs:
        content = sd.get("content", "")
        dups = _duplicate_module_scope_bindings(content)
        if dups:
            raise StepDefGenerationError(
                f"{sd.get('filename', '?')}: duplicate module-scope binding(s) "
                f"({', '.join(dups)}); playwright-bdd/Babel throws `VarRedeclaration` "
                "at compile time for a redeclared const/let/var/function/class/import"
            )


# ── scoped resilient click/fill target (an unscoped selector clicks another node) ─
#
# `resilientClick(page, primary, label?, opts?)` typed `primary` as a plain STRING
# selector, while the recording prompt orders "NEVER call raw page.click — use
# `resilientClick(page, '<selector>', '<accessible label>')`". A step whose target
# is SCOPED ("the checkbox inside the 'Walk the dog' row") therefore had no way to
# say so. Measured live on FIVE consecutive todomvc packages (bdd jobs 01a057ef,
# 01a05992, 01a059a2-87c6, 01a059a2-eb89, 01a059d4-474a): the model built the
# right locator, deliberated in up to 40 lines of comments about whether the
# helper would accept it, and then fell back to the page-wide
# `'input[type="checkbox"]'` — leaving the scoped locator BOUND AND UNUSED, or
# passing it into the `opts` slot where it is silently dropped:
#
#     const todoItem = page.getByRole('listitem').filter({ hasText: text }).first();
#     const checkbox = todoItem.getByRole('checkbox').first();
#     await resilientClick(page, 'input[type="checkbox"]', text);
#
# On TodoMVC `input[type="checkbox"]` resolves first to the toggle-all box, not the
# row's toggle, so the row never toggles. `resilientClick` SOFT-FAILS without
# throwing, so the When step still "passes" and the defect only surfaces one step
# later as `expect(checkbox).toBeChecked()` receiving an unchecked box (run
# 01a059a2, 2026-08-31). The shipped helper now accepts a Locator for `primary`;
# this pass rewires the call to the scoped locator the model ALREADY built. A
# prompt rule alone would be advisory (#119) and would ship inert.
_RESILIENT_CALL_RE = re.compile(r"\bresilient(?P<kind>Click|Fill)\s*\(")

# `const <name> = <expr>` where <expr> is a synchronous locator chain. Matched in two
# steps rather than one regex: the chain can carry arbitrary intermediate links
# (`items.nth(i).getByRole('button', { name: 'Destroy' })` — job 01a057ef), and a
# single pattern spanning them either misses those or over-matches across statements.
_CONST_BIND_RE = re.compile(r"\bconst\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*(?=[A-Za-z_$])")
_LOCATOR_EXPR_RE = re.compile(r"\.\s*(?:getBy[A-Za-z]+|locator)\s*\(")
# `const count = await items.count()` is a NUMBER, not a locator; an awaited
# expression is never a Locator and must never become a click target.
_AWAITED_RE = re.compile(r"\Aawait\b")
_BARE_IDENT_RE = re.compile(r"[A-Za-z_$][\w$]*\Z")
_STEP_CALL_RE = re.compile(r"\b(?:Given|When|Then)\s*\(")

# (index of `primary`, index of the first slot PAST the documented arguments).
# `resilientClick(page, primary, label, opts)` / `resilientFill(page, primary,
# value, label, opts)`: a Locator handed to the trailing slot lands on `opts` and
# is silently eaten, which is exactly what job 01a059a2-eb89 emitted.
_RESILIENT_SLOTS = {"Click": (1, 3), "Fill": (1, 4)}


def _split_call_args(text: str, mask: list[bool], open_idx: int) -> list[tuple[int, int]] | None:
    """Whitespace-trimmed spans of the TOP-LEVEL arguments of the call at ``open_idx``.

    ``None`` when the call's parens are unbalanced. Commas inside nested
    ``([{``/strings/comments do not split, so an inline locator chain or an options
    object stays one argument.
    """
    past = _closing_paren(text, mask, open_idx)
    if past is None:
        return None
    close = past - 1
    raw: list[tuple[int, int]] = []
    depth = 0
    start = open_idx + 1
    for i in range(open_idx + 1, close):
        if not mask[i]:
            continue
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            raw.append((start, i))
            start = i + 1
    raw.append((start, close))
    return _trimmed_spans(text, raw)


def _trimmed_spans(text: str, raw: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """``raw`` spans with surrounding whitespace removed; empty spans dropped."""
    spans: list[tuple[int, int]] = []
    for s, e in raw:
        while s < e and text[s] in " \t\r\n":
            s += 1
        while e > s and text[e - 1] in " \t\r\n":
            e -= 1
        if s < e:
            spans.append((s, e))
    return spans


def _step_def_bodies(text: str, mask: list[bool]) -> list[tuple[int, int]]:
    """Spans ``(body_start, close_paren)`` of every ``Given``/``When``/``Then`` call.

    The generalisation of :func:`_nav_given_bodies` (which is navigation-Given only):
    a locator binding is scoped to ONE step callback, so any pass that pairs a
    binding with a use must not read across step boundaries.
    """
    spans: list[tuple[int, int]] = []
    for m in _STEP_CALL_RE.finditer(text):
        if not mask[m.start()]:
            continue
        past = _closing_paren(text, mask, m.end() - 1)
        if past is None:
            continue
        spans.append((m.end(), past - 1))
    return spans


def _code_uses(text: str, mask: list[bool], name: str, start: int, end: int) -> int:
    """How many times identifier ``name`` appears as CODE in ``[start, end)``."""
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    return sum(1 for m in pattern.finditer(text, start, end) if mask[m.start()])


def _locator_bindings(text: str, mask: list[bool], start: int, end: int) -> dict[str, int]:
    """``{name: declaration offset}`` for every synchronous locator ``const`` in the span."""
    bindings: dict[str, int] = {}
    for m in _CONST_BIND_RE.finditer(text, start, end):
        if not mask[m.start()]:
            continue
        tail = _statement_tail(text, mask, m.end())
        if _AWAITED_RE.search(tail) or not _LOCATOR_EXPR_RE.search(tail):
            continue
        bindings[m.group("name")] = m.start()
    return bindings


def _scoped_target_rewrites(
    text: str, mask: list[bool], start: int, end: int
) -> list[tuple[int, int, str]]:
    """Replacements rewiring the resilient calls in ``[start, end)`` to a scoped Locator.

    Two shapes, in priority order, both confined to ONE step callback:

    1. **hoist** — a Locator identifier handed to the trailing (`opts`) slot, where the
       helper silently drops it. It becomes the `primary` target and the dead
       trailing argument is removed.
    2. **adopt** — a string-literal `primary` alongside EXACTLY ONE locator binding in
       the same callback that is referenced nowhere else (declared, then abandoned).
       Ambiguity is declined: two unused bindings, or none, leaves the call
       byte-identical, because guessing which node to click is worse than the
       unscoped selector.
    """
    bindings = _locator_bindings(text, mask, start, end)
    if not bindings:
        return []
    dead = [name for name in bindings if _code_uses(text, mask, name, start, end) == 1]
    repls: list[tuple[int, int, str]] = []
    for m in _RESILIENT_CALL_RE.finditer(text, start, end):
        if not mask[m.start()]:
            continue
        args = _split_call_args(text, mask, m.end() - 1)
        if args is None:
            continue
        primary_slot, opts_slot = _RESILIENT_SLOTS[m.group("kind")]
        if len(args) <= primary_slot:
            continue
        p_start, p_end = args[primary_slot]
        hoisted = next(
            (
                idx
                for idx in range(opts_slot, len(args))
                if _BARE_IDENT_RE.fullmatch(text[args[idx][0] : args[idx][1]]) is not None
                and text[args[idx][0] : args[idx][1]] in bindings
            ),
            None,
        )
        if hoisted is not None:
            repls.append((p_start, p_end, text[args[hoisted][0] : args[hoisted][1]]))
            repls.append((args[hoisted - 1][1], args[hoisted][1], ""))
            continue
        if text[p_start] not in "'\"`" or len(dead) != 1:
            continue
        repls.append((p_start, p_end, dead[0]))
    return repls


def _ensure_scoped_resilient_targets(step_defs: list[_StepDef]) -> int:
    """Point every resilient click/fill at the SCOPED locator its step already built.

    Returns the number of files changed; mutates ``content`` in place. Recording
    stream only — ``lib/recording_helpers`` ships with the demo_video package.
    """
    changed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        mask, ok = _code_mask(content)
        if not ok:
            continue
        repls: list[tuple[int, int, str]] = []
        for body_start, body_end in _step_def_bodies(content, mask):
            repls.extend(_scoped_target_rewrites(content, mask, body_start, body_end))
        if not repls:
            continue
        new = content
        for r_start, r_end, replacement in sorted(repls, key=lambda r: r[0], reverse=True):
            new = new[:r_start] + replacement + new[r_end:]
        if new != content:
            sd["content"] = new
            changed += 1
    return changed


# ── an assertion's CSS selector must come from the PAGE, not from memory ─────
# Measured on acceptance run 01a0b007-9c2c (bdd job 01a0b007-bb54, FTR 5 scenarios,
# 2 passed / 3 failed) against demo.playwright.dev/todomvc:
#     const countElement = page.locator('#todo-count').first();
#     await expect(countElement).toContainText(count);          // 5s timeout
# The app renders `<span data-testid="todo-count" class="todo-count">` — no id, and
# only once a todo exists. The model wrote "Since I don't have the full DOM" and then
# guessed one. The "NEVER INVENT A SELECTOR" prompt rule is advisory, and
# `_ensure_grounded_role_names` only judges `getByRole` NAMES in the navigation Given.
# A failed Then does not blank a clip — it FREEZES its tail for the assertion timeout,
# and that frozen footage ships (critique 2.0/4.0 "near-zero motion", film ran long).
#
# RULE. For each assertion (`expect(…)`, `expect.soft(…)`, `softExpectVisible(…)`) the
# locator is resolved through the step's own `const`/`let` bindings (and module-scope
# string consts) to every `locator('<css>')` / `getByTestId('<id>')` it is built from.
# Each `#id`, `.class`, `[id=…]`, `[class~=…]`, `[data-testid=…]` and testid is a CLAIM
# about the DOM. An assertion is UNGROUNDED when any claim names a token absent from
# the scraped DOM's token inventory (`html_structure.DomTokensDict`) — and it is
# REMOVED; the action statements around it are untouched, so the footage still shows
# the interaction. A `toHaveCSS` matcher is removed too: a computed style is carried by
# no snapshot (measured: `line-through` asserted on the `li`, style is on the label).
#
# ⚠️ WHAT IS REMOVED IS "UNVERIFIABLE AGAINST THE PRE-INTERACTION SNAPSHOT", NOT
# "GUESSED". The snapshot is the page on first load, so an element that only renders
# after the user acts is absent from it whether the selector is invented or CORRECT.
# On TodoMVC `getByTestId('todo-count')` and `getByTestId('todo-item')` are both
# correct post-interaction assertions, and this pass drops them too. This pass cannot
# tell a guessed selector from a correct post-interaction one, and does not try.
#
# WHY that is acceptable HERE, when `_ensure_grounded_role_names` deliberately refuses
# to judge post-interaction elements: the product of a RECORDING stream is footage,
# not verification. An assertion this pass cannot verify buys the film nothing when
# it passes and costs it a frozen tail when it fails; losing a correct one costs the
# film nothing. Assertions written the way the prompt asks for post-interaction
# content — role / label / text supplied by the scenario (`getByRole('listitem')
# .filter({ hasText })`) — make no id/class/testid claim and are kept.
# Each drop leaves a one-line `// assertion dropped: …` comment in the step body.
#
# SCOPE: recording (demo_video) stream only. In QA/accessibility/documentation an
# unverifiable selector failing is an HONEST test failure the operator must see.
#
# No-touch rules (unknown ⇒ KEEP, the same stance as the role pass):
#   * any page lacking `dom_tokens` (not scanned / legacy row), or an inventory with
#     NO ids, classes or testids at all (an unhydrated SPA shell)   -> change nothing.
#   * a truncated bucket                                           -> its claims unjudged.
#   * a selector this pass cannot read: non-literal argument, template interpolation,
#     CSS escapes, a selector list (`,`), a non-CSS engine (`text=`, `xpath=`, `//`),
#     a case-insensitive attribute flag                            -> no claims.
#   * tag-only / other-attribute selectors (`footer`, `input[placeholder=…]`) and
#     every `:not(…)`/`:is(…)`/`:has(…)` argument                  -> no claims.
#   * a statement that is not a bare `[await] expect…;` / `softExpectVisible…;` (an
#     assertion inside `Promise.all([...click(), expect()])`, or one whose argument
#     holds a `;`)                                                 -> refused, kept.
_GROUNDABLE_ASSERT_CALL_RE = re.compile(
    r"(?<![\w$.])(?:expect(?:\s*\.\s*soft)?|softExpectVisible)\s*\("
)
_SELECTOR_CALL_RE = re.compile(r"(?<![\w$])(?P<kind>locator|getByTestId)\s*\(")
_CODE_IDENT_RE = re.compile(r"(?<![\w$.])[A-Za-z_$][\w$]*")
_ANY_BINDING_RE = re.compile(
    r"\b(?:const|let)\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?::[^=;]+?)?=(?![=>])\s*"
)
_STRING_CONST_RE = re.compile(
    r"\bconst\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?::\s*string\s*)?=\s*(?=['\"`])"
)
_TO_HAVE_CSS_RE = re.compile(r"\.\s*toHaveCSS\s*\(")
# No `\A`: `.match(text, pos)` already anchors at `pos`; `\A` would mean offset 0.
_WHOLE_ASSERT_STATEMENT_RE = re.compile(r"\s*(?:await\s+)?(?:expect|softExpectVisible)\b")
_SELECTOR_STRING_RE = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'")
_ATTR_CLAIM_RE = re.compile(
    r"\[\s*(?P<attr>[\w-]+)\s*(?P<op>~=|=)\s*"
    r"(?:\"(?P<dq>[^\"]*)\"|'(?P<sq>[^']*)'|(?P<bare>[^\]\s\"']+))\s*(?P<flag>[iIsS])?\s*\]"
)
_ENGINE_PREFIX_RE = re.compile(r"\A(?P<engine>[a-z][\w-]*)\s*=(?![=])")
_ID_CLASS_TOKEN_RE = re.compile(
    r"(?P<sigil>[#.])(?P<token>-?[_a-zA-Z\u00a0-\uffff][\w\u00a0-\uffff-]*)"
)
_MAX_BINDING_DEPTH = 8
# (claim bucket, `DomTokensDict` key)
_DOM_BUCKETS = (("id", "ids"), ("class", "classes"), ("testid", "testids"))
_Claim = tuple[str, str]  # (bucket, token)


def _inventory_dom_tokens(inspection: Mapping[str, Any] | None) -> dict[str, frozenset[str]]:
    """``{bucket: tokens}`` for every bucket the scraped DOM inventory is EXHAUSTIVE for.

    Union across pages (a step may run on any scanned page). Empty dict — judge
    nothing — when there are no pages, when ANY page was not scanned (the union would
    then not cover it), or when the whole inventory holds no token at all.
    """
    if not isinstance(inspection, dict):
        return {}
    pages = inspection.get("pages")
    if not isinstance(pages, list) or not pages:
        return {}
    merged: dict[str, set[str]] = {bucket: set() for bucket, _ in _DOM_BUCKETS}
    trusted = {bucket for bucket, _ in _DOM_BUCKETS}
    for page in pages:
        tokens = page.get("dom_tokens") if isinstance(page, dict) else None
        if not isinstance(tokens, dict):
            return {}
        for bucket, key in _DOM_BUCKETS:
            values = tokens.get(key)
            if not isinstance(values, list) or _positive_int(tokens.get(f"{key}_truncated")):
                trusted.discard(bucket)
                continue
            merged[bucket].update(v for v in values if isinstance(v, str))
    if not any(merged.values()):
        return {}
    return {bucket: frozenset(merged[bucket]) for bucket in trusted}


def _strip_paren_args(part: str) -> str:
    """``part`` with every balanced ``( … )`` argument removed (strings pre-neutralised).

    Drops `:not(.x)` (whose token must be ABSENT), `:is(.a, .b)` (either suffices) and
    `:has(.x)` alike — losing a claim can only KEEP an assertion.
    """
    out: list[str] = []
    depth = 0
    for ch in part:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _attribute_claims(part: str) -> list[_Claim]:
    """Claims made by ``[id=…]`` / ``[class~=…]`` / ``[class=…]`` / ``[data-testid=…]``."""
    claims: list[_Claim] = []
    for m in _ATTR_CLAIM_RE.finditer(part):
        if m.group("flag") in {"i", "I"}:
            continue
        value = next((v for v in (m.group("dq"), m.group("sq"), m.group("bare")) if v), "")
        attr, op = m.group("attr").lower(), m.group("op")
        if attr == "id" and op == "=" and value:
            claims.append(("id", value))
        elif attr == "class":
            claims.extend(("class", token) for token in value.split())
        elif attr == "data-testid" and op == "=" and value:
            claims.append(("testid", value))
    return claims


def _css_part_claims(part: str) -> list[_Claim]:
    """Claims of ONE `>>`-segment of a Playwright selector; ``[]`` when unreadable."""
    part = part.strip()
    engine = _ENGINE_PREFIX_RE.match(part)
    if engine is not None:
        name, value = engine.group("engine"), part[engine.end() :].strip().strip("\"'")
        if name == "css":
            part = part[engine.end() :]
        elif name == "id" and value:
            return [("id", value)]
        elif name == "data-testid" and value:
            return [("testid", value)]
        else:
            return []
    # XPath (`//a`, `..`, `(//a)[1]`), a quoted text selector, an internal engine, or CSS
    # escapes: not read.
    if not part or part.startswith(("//", "..", "(", '"', "'", "internal:")) or "\\" in part:
        return []
    claims = _attribute_claims(part)
    neutral = _SELECTOR_STRING_RE.sub('""', part)
    neutral = re.sub(r"\[[^\]]*\]", " ", neutral)
    neutral = _strip_paren_args(neutral)
    if "," in neutral:
        return []
    claims.extend(
        ("id" if m.group("sigil") == "#" else "class", m.group("token"))
        for m in _ID_CLASS_TOKEN_RE.finditer(neutral)
    )
    return claims


def _selector_claims(selector: str) -> list[_Claim]:
    """Every DOM token a Playwright selector string requires to exist.

    Segments chained with ``>>`` must ALL match, so their claims add up. A ``>>``
    inside a quoted value would split wrongly, so a selector carrying both yields none.
    """
    if ">>" in selector and ("'" in selector or '"' in selector):
        return []
    claims: list[_Claim] = []
    for part in selector.split(">>"):
        claims.extend(_css_part_claims(part))
    return claims


def _string_consts(text: str, mask: list[bool]) -> dict[str, str]:
    """``{NAME: value}`` for each ``const NAME = '<literal>';`` declared exactly once."""
    found: dict[str, str | None] = {}
    for m in _STRING_CONST_RE.finditer(text):
        if not mask[m.start()]:
            continue
        literal = _string_literal_at(text, m.end())
        name = m.group("name")
        if name in found or literal is None or "${" in text[m.end() : literal[1]]:
            found[name] = None
            continue
        tail = text[literal[1] : literal[1] + 2].lstrip(" \t")
        found[name] = literal[0] if tail[:1] in {";", "\n", "\r", ""} else None
    return {name: value for name, value in found.items() if value is not None}


def _expression_bindings(
    text: str, mask: list[bool], start: int, end: int
) -> dict[str, tuple[int, int]]:
    """``{name: (expr_start, expr_end)}`` for each ``const``/``let`` in ONE step body.

    First declaration wins; reassignment and shadowing are not modelled.
    """
    bindings: dict[str, tuple[int, int]] = {}
    for m in _ANY_BINDING_RE.finditer(text, start, end):
        if not mask[m.start()]:
            continue
        tail = _statement_tail(text, mask, m.end())
        bindings.setdefault(m.group("name"), (m.end(), min(end, m.end() + len(tail))))
    return bindings


def _literal_selector_arg(text: str, span: tuple[int, int], consts: dict[str, str]) -> str | None:
    """The selector string passed as ``span``, or ``None`` when it is not a readable
    literal (or a module-scope string const)."""
    s, e = span
    raw = text[s:e]
    if _BARE_IDENT_RE.fullmatch(raw):
        return consts.get(raw)
    literal = _string_literal_at(text, s)
    if literal is None or literal[1] != e or "${" in raw:
        return None
    return literal[0]


@dataclass
class _GroundingScope:
    """Everything one step body's claim collection reads, bundled to keep signatures short."""

    text: str
    mask: list[bool]
    bindings: dict[str, tuple[int, int]]
    consts: dict[str, str]


def _collect_selector_claims(
    scope: _GroundingScope, start: int, end: int, seen: set[str], depth: int = 0
) -> list[tuple[str, _Claim]]:
    """``[(selector, claim)]`` for every locator the expression in ``[start, end)`` is
    built from, following the step's own bindings (cycle- and depth-bounded)."""
    text, mask = scope.text, scope.mask
    out: list[tuple[str, _Claim]] = []
    for m in _SELECTOR_CALL_RE.finditer(text, start, end):
        if not mask[m.start()]:
            continue
        args = _split_call_args(text, mask, m.end() - 1)
        selector = _literal_selector_arg(text, args[0], scope.consts) if args else None
        if selector is None:
            continue
        if m.group("kind") == "getByTestId":
            out.append((selector, ("testid", selector)))
        else:
            out.extend((selector, claim) for claim in _selector_claims(selector))
    if depth >= _MAX_BINDING_DEPTH:
        return out
    for m in _CODE_IDENT_RE.finditer(text, start, end):
        name = m.group(0)
        if not mask[m.start()] or name not in scope.bindings or name in seen:
            continue
        seen.add(name)
        b_start, b_end = scope.bindings[name]
        out.extend(_collect_selector_claims(scope, b_start, b_end, seen, depth + 1))
    return out


def _ungrounded_claim(
    scope: _GroundingScope, call: re.Match[str], inventory: dict[str, frozenset[str]]
) -> tuple[str, str] | None:
    """``(selector, claim)`` that makes the assertion at ``call`` ungrounded, or ``None``."""
    text, mask = scope.text, scope.mask
    past = _closing_paren(text, mask, call.end() - 1)
    if past is None:
        return None
    if _TO_HAVE_CSS_RE.search(_statement_tail(text, mask, past)):
        return "", "toHaveCSS"
    for selector, (bucket, token) in _collect_selector_claims(scope, call.end(), past - 1, set()):
        known = inventory.get(bucket)
        if known is not None and token not in known:
            sigil = {"id": "#", "class": "."}.get(bucket)
            return selector, (f"{sigil}{token}" if sigil else f"[data-testid={token!r}]")
    return None


def _is_whole_assert_statement(text: str, mask: list[bool], at: int) -> bool:
    """Is the statement holding ``at`` exactly one bare assertion that
    `_drop_assert_statement_at` will remove WHOLE?

    Refuses a statement that also does something else (`Promise.all([click(),
    expect(…)])` — dropping it would delete an action) and one whose first `;` sits
    INSIDE the call (`expect(async () => { …; }).toPass()` — a partial drop does not
    compile).
    """
    start = 0
    for i in range(at - 1, -1, -1):
        if mask[i] and text[i] in ";{}":
            start = i + 1
            break
    if _WHOLE_ASSERT_STATEMENT_RE.match(text, start) is None or text[start:at].strip() not in {
        "",
        "await",
    }:
        return False
    first_semi = next((i for i in range(at, len(text)) if mask[i] and text[i] == ";"), None)
    tail = _statement_tail(text, mask, at)
    return first_semi is not None and at + len(tail) == first_semi + 1


def _next_ungrounded_assertion(
    text: str, mask: list[bool], inventory: dict[str, frozenset[str]], refused: set[int]
) -> tuple[int, UngroundedAssertion] | None:
    """Offset + record of the first droppable ungrounded assertion, or ``None``."""
    consts = _string_consts(text, mask)
    for body_start, body_end in _step_def_bodies(text, mask):
        scope = _GroundingScope(
            text, mask, _expression_bindings(text, mask, body_start, body_end), consts
        )
        for call in _GROUNDABLE_ASSERT_CALL_RE.finditer(text, body_start, body_end):
            at = call.start()
            if not mask[at] or at in refused:
                continue
            found = _ungrounded_claim(scope, call, inventory)
            if found is None:
                continue
            if not _is_whole_assert_statement(text, mask, at):
                refused.add(at)
                continue
            step = _string_literal_at(text, body_start)
            return at, {
                "file": "",
                "step": step[0] if step is not None else "",
                "selector": found[0],
                "claim": found[1],
            }
    return None


def _drop_ungrounded_assertions(
    text: str, inventory: dict[str, frozenset[str]]
) -> tuple[str, list[UngroundedAssertion]]:
    """Remove every ungrounded assertion statement; re-mask after each drop."""
    drops: list[UngroundedAssertion] = []
    refused: set[int] = set()
    while True:
        mask, _ = _code_mask(text)
        target = _next_ungrounded_assertion(text, mask, inventory, refused)
        if target is None:
            return text, drops
        at, record = target
        new_text, ok = _drop_assert_statement_at(text, mask, at)
        if not ok:
            refused.add(at)
            continue
        text = _leave_drop_comment(text, new_text, at, record["claim"])
        drops.append(record)
        refused.clear()


_DROP_COMMENT = "// assertion dropped: unverifiable against pre-interaction snapshot"


def _leave_drop_comment(old: str, new: str, at: int, claim: str) -> str:
    """``new`` with a one-line comment where the assertion at ``at`` was, so a step body
    left asserting nothing explains itself. Only when the statement had its own line
    (the drop then starts at that line); otherwise ``new`` unchanged."""
    line_start = old.rfind("\n", 0, at) + 1
    before = old[line_start:at]
    indent = before[: len(before) - len(before.lstrip())]
    if before.strip() not in {"", "await"} or new[:line_start] != old[:line_start]:
        return new
    safe_claim = " ".join(claim.split())
    return f"{new[:line_start]}{indent}{_DROP_COMMENT} ({safe_claim})\n{new[line_start:]}"


def _ensure_grounded_assertion_selectors(
    step_defs: list[_StepDef],
    inspection: Mapping[str, Any] | None,
    coverage: StepCoverageReport | None = None,
) -> int:
    """Drop every assertion UNVERIFIABLE against the pre-interaction snapshot: its locator
    names an id/class/testid the scraped DOM does not carry (whether invented or a
    correct post-interaction element), or it is a `toHaveCSS`. RECORDING stream only —
    see the block comment for why dropping correct assertions is acceptable there.

    Mutates in place; returns the number of files changed; appends one structured
    record per dropped assertion to ``coverage.ungrounded_assertions``. A navigation
    Given left asserting nothing is backfilled with ``assertNavigated(page)``, exactly
    as `_ensure_grounded_role_names` does.
    """
    inventory = _inventory_dom_tokens(inspection)
    if not inventory:
        return 0
    changed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        _, healthy = _code_mask(content)
        if not healthy:
            continue
        new, drops = _drop_ungrounded_assertions(content, inventory)
        if not drops:
            continue
        new, backfilled = _backfill_navigation_asserts(new)
        if backfilled:
            new = _inject_helper_import(new, _SOFT_NAV_IMPORT_RE, _SOFT_NAV_IMPORT_LINE)
        filename = str(sd.get("filename", "<unnamed>"))
        for record in drops:
            record["file"] = filename
            logger.warning(
                "Dropped assertion in %s step %r: %s (selector %r) is unverifiable against the "
                "pre-interaction snapshot (possibly a correct post-interaction element) — a "
                "miss freezes the clip's tail for the timeout "
                "(recording grounding guard)",
                filename,
                record["step"],
                record["claim"],
                record["selector"],
            )
        if coverage is not None:
            coverage.ungrounded_assertions.extend(drops)
        sd["content"] = new
        changed += 1
    return changed


# ── a generic text-presence Then asserts PAGE-WIDE, not inside a list item ────
# Measured on showcase run 01a15d7c (clip-5) against demo.playwright.dev/todomvc:
#     Then('I see the text {string} displayed on the page', async ({ page }, text) => {
#       await expect(page.getByRole('listitem').filter({ hasText: text }).first())
#         .toContainText(text);
#     });
# invoked with 'This is just a demo of TodoMVC for testing, not the real TodoMVC
# app.' — a top-level <div> on that page, inside no list item, so the locator could
# never match: a 5 s timeout froze the clip's tail. The recording prompt used to give
# `getByRole('listitem').filter({ hasText })` as THE form for scenario-supplied text,
# and the model applied it to a step that was shared by a todo-item assertion and a
# page-wide one.
#
# RULE. Inside a `Then` whose PATTERN names no list/item/row/todo/task/entry/cell, a
# `page.getByRole('listitem'|'row'|'cell').filter({ hasText: X })` narrowed with
# `.first()`/`.last()`/`.nth(n)` becomes `page.getByText(X)` with the same narrowing.
# Page-wide is a SUPERSET of the role-scoped match (any listitem containing X also
# contains the element `getByText(X)` resolves to), so the rewrite can only turn a
# never-matching assertion into a matching one — never the reverse.
#
# WHY no snapshot grounding: the inspection a step-def generator receives carries flat
# `text_content`, headings, interactive elements and id/class/testid tokens — NO
# element containment, so it cannot say whether X sits inside a list item. And X is
# usually a step PARAMETER, unknown at generation time. The pattern is the only
# evidence available, so the pattern decides.
#
# No-touch rules (unknown ⇒ KEEP):
#   * a non-literal step pattern; a pattern naming a scoping noun.
#   * an un-narrowed locator (may be a `toHaveCount`/`.count()` subject).
#   * a chain that continues into a child query (`.getByRole('checkbox')`, `.locator`,
#     `.filter`, `.and`, `.or`) — directly, or through a `const`/`let` binding the step
#     queries into later: the child lives in the row, not in the text node.
#   * any `filter` option other than `hasText` (`has`, `hasNot`, `hasNotText`).
#   * any use that is not TEXT PRESENCE. The rewrite only holds for "this text is
#     shown": `expect(LOC).toBeVisible|toContainText|toHaveText(...)` or `LOC.waitFor()`
#     (directly, or through a binding EVERY use of which is one of those). A matcher on
#     the CONTAINER's state (`toHaveClass(/completed/)`, `toBeChecked`, `toHaveCount`,
#     `.not.*`, …) asserts the <li>, not the classless <label> holding the text —
#     demo_video run 01a163a7 clip-4 received class "" and froze. An allow-list, so an
#     unknown matcher keeps the scope.
_PRESENCE_MATCHERS = frozenset({"toBeVisible", "toContainText", "toHaveText"})
_EXPECT_OPEN_BEFORE_RE = re.compile(r"(?<![\w$.])expect\s*(?:\.\s*soft\s*)?\(\s*\Z")
_MATCHER_AFTER_RE = re.compile(r"\s*\.\s*(?P<name>[A-Za-z_$][\w$]*)\s*\(")
_THEN_CALL_RE = re.compile(r"(?<![\w$.])Then\s*\(")
_PAGE_ROLE_SCOPE_RE = re.compile(r"(?<![\w$.])page(?=\s*\.\s*getByRole\s*\()")
_TEXT_SCOPE_ROLE_ARG_RE = re.compile(
    r"\A\s*\.\s*getByRole\s*\(\s*(?P<q>['\"])(?:listitem|row|cell)(?P=q)\s*\)\Z"
)
_FILTER_OPEN_RE = re.compile(r"\s*\.\s*filter\s*\(")
_HAS_TEXT_ENTRY_RE = re.compile(r"\AhasText\s*(?::\s*(?P<value>.+))?\Z", re.DOTALL)
_SCOPING_NOUN_RE = re.compile(
    r"(?i)\b(?:list\w*|items?|rows?|to-?dos?|tasks?|entr(?:y|ies)|cells?|tables?)\b"
)
_CHILD_QUERY_METHODS = frozenset({"locator", "filter", "and", "or"})
_BINDING_BEFORE_RE = re.compile(r"(?:const|let)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*\Z")


def _is_child_query(method: str) -> bool:
    return method.startswith("getBy") or method in _CHILD_QUERY_METHODS


def _then_pattern(text: str, mask: list[bool], open_idx: int) -> str | None:
    """The literal pattern (string or regex source, quotes included) of the ``Then``
    call at ``open_idx``; ``None`` when it is not a literal."""
    args = _split_call_args(text, mask, open_idx)
    if not args:
        return None
    s, e = args[0]
    first = text[s:e]
    return first if first[:1] in "'\"`/" else None


def _top_level_spans(text: str, mask: list[bool], s: int, e: int) -> list[tuple[int, int]]:
    """Trimmed comma-separated spans of ``[s, e)`` at bracket depth 0 (code only)."""
    raw: list[tuple[int, int]] = []
    depth, start = 0, s
    for i in range(s, e):
        if not mask[i]:
            continue
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            raw.append((start, i))
            start = i + 1
    raw.append((start, e))
    return _trimmed_spans(text, raw)


def _has_text_value(text: str, mask: list[bool], open_idx: int) -> tuple[int, int] | None:
    """Span of ``X`` in a ``.filter({ hasText: X })`` call whose ``(`` is at ``open_idx``
    (shorthand ``{ hasText }`` included); ``None`` for any other option shape."""
    args = _split_call_args(text, mask, open_idx)
    if args is None or len(args) != 1:
        return None
    s, e = args[0]
    if text[s] != "{" or text[e - 1] != "}" or not mask[s] or not mask[e - 1]:
        return None
    spans = _top_level_spans(text, mask, s + 1, e - 1)
    if len(spans) != 1:
        return None
    es, ee = spans[0]
    m = _HAS_TEXT_ENTRY_RE.match(text[es:ee])
    if m is None:
        return None
    if m.group("value") is None:  # shorthand `{ hasText }`
        return es, es + len("hasText")
    return es + m.start("value"), es + m.end("value")


def _role_scoped_text_locator(
    text: str, mask: list[bool], page_end: int
) -> tuple[int, tuple[int, int], int, list[str]] | None:
    """For the ``page`` ending at ``page_end``: ``(filter_end, hasText value span,
    narrowed_end, tail method names)`` when it heads ``.getByRole('listitem'|'row'|
    'cell').filter({ hasText }).first()|last()|nth()`` with no child query after it;
    else ``None``."""
    segments = _chain_segments(text, mask, page_end)
    if len(segments) < 3 or segments[0][0] != "getByRole" or segments[1][0] != "filter":
        return None
    if segments[2][0] not in _DISAMBIGUATING_METHODS:
        return None
    tail = [name for name, _ in segments[3:]]
    if any(_is_child_query(name) for name in tail):
        return None
    role_end = segments[0][1]
    if not _TEXT_SCOPE_ROLE_ARG_RE.match(text[page_end:role_end]):
        return None
    filter_open = _FILTER_OPEN_RE.match(text, role_end)
    if filter_open is None:
        return None
    value = _has_text_value(text, mask, filter_open.end() - 1)
    if value is None:
        return None
    return segments[1][1], value, segments[2][1], tail


def _expect_matcher(
    text: str, mask: list[bool], span_start: int, expr_start: int, expr_end: int
) -> str | None:
    """The matcher applied when ``[expr_start, expr_end)`` is the SOLE argument of an
    ``expect(…)``/``expect.soft(…)`` call (``.not`` and other property hops return
    ``None``); else ``None``."""
    before = _EXPECT_OPEN_BEFORE_RE.search(text, span_start, expr_start)
    if before is None or before.end() != expr_start or not mask[before.start()]:
        return None
    open_idx = text.rindex("(", before.start(), expr_start)
    close = _closing_paren(text, mask, open_idx)
    if close is None or text[expr_end : close - 1].strip():
        return None
    m = _MATCHER_AFTER_RE.match(text, close)
    if m is None or not mask[m.start("name")]:
        return None
    return m.group("name")


def _is_presence_use(
    text: str, mask: list[bool], span_start: int, expr_start: int, expr_end: int
) -> bool:
    """Whether the locator expression ``[expr_start, expr_end)`` (no chain after it) is
    used only to assert its TEXT is present."""
    if _chain_segments(text, mask, expr_end):
        return False
    return _expect_matcher(text, mask, span_start, expr_start, expr_end) in _PRESENCE_MATCHERS


def _binding_uses_are_presence(
    text: str, mask: list[bool], span: tuple[int, int], name: str, after: int
) -> bool:
    """Whether the step (``span``) uses binding ``name`` at least once after ``after``,
    and every use is a text-presence assertion or a bare ``.waitFor()``."""
    start, end = span
    ref = re.compile(rf"(?<![\w$.]){re.escape(name)}(?![\w$])")
    uses = [u for u in ref.finditer(text, after, end) if mask[u.start()]]
    if not uses:
        return False
    for u in uses:
        tail = [n for n, _ in _chain_segments(text, mask, u.end())]
        if tail == ["waitFor"]:
            continue
        if tail or not _is_presence_use(text, mask, start, u.start(), u.end()):
            return False
    return True


def _locator_is_presence_only(
    text: str,
    mask: list[bool],
    span: tuple[int, int],
    at: int,
    narrowed_end: int,
    tail: list[str],
) -> bool:
    """Whether the role-scoped locator starting at ``at`` is used ONLY for text
    presence — the only case where widening it to the page is sound."""
    if tail:
        return tail == ["waitFor"]
    bound = _BINDING_BEFORE_RE.search(text, span[0], at)
    if bound is not None and bound.end() == at:
        return _binding_uses_are_presence(text, mask, span, bound.group("name"), narrowed_end)
    return _is_presence_use(text, mask, span[0], at, narrowed_end)


def _page_wide_text_rewrites(
    text: str, mask: list[bool], start: int, end: int
) -> list[tuple[int, int, str]]:
    """Replacements turning role-scoped text locators in ``[start, end)`` page-wide."""
    repls: list[tuple[int, int, str]] = []
    for m in _PAGE_ROLE_SCOPE_RE.finditer(text, start, end):
        if not mask[m.start()]:
            continue
        found = _role_scoped_text_locator(text, mask, m.end())
        if found is None:
            continue
        filter_end, (vs, ve), narrowed_end, tail = found
        if not _locator_is_presence_only(text, mask, (start, end), m.start(), narrowed_end, tail):
            continue
        repls.append((m.start(), filter_end, f"page.getByText({text[vs:ve]})"))
    return repls


def _ensure_page_wide_text_presence(step_defs: list[_StepDef]) -> int:
    """Rewrite list-item-scoped text locators in generic text-presence ``Then`` steps to
    page-wide ``page.getByText``. Mutates in place; returns the number of files changed.
    RECORDING stream only, beside the other recording rewrites."""
    changed = 0
    for sd in step_defs:
        content = sd.get("content", "")
        if "getByRole" not in content or "Then" not in content:
            continue
        mask, ok = _code_mask(content)
        if not ok:
            continue
        repls: list[tuple[int, int, str]] = []
        for call in _THEN_CALL_RE.finditer(content):
            if not mask[call.start()]:
                continue
            open_idx = call.end() - 1
            pattern = _then_pattern(content, mask, open_idx)
            if pattern is None or _SCOPING_NOUN_RE.search(pattern):
                continue
            past = _closing_paren(content, mask, open_idx)
            if past is None:
                continue
            repls.extend(_page_wide_text_rewrites(content, mask, call.end(), past - 1))
        if not repls:
            continue
        new = content
        for r_start, r_end, replacement in sorted(repls, key=lambda r: r[0], reverse=True):
            new = new[:r_start] + replacement + new[r_end:]
        logger.warning(
            "Rewrote %d role-scoped text locator(s) to page-wide getByText in %s: the "
            "Then step's pattern names no list/item/row, so the text need not sit in one "
            "(page-wide text-presence guard)",
            len(repls),
            sd.get("filename", "<unnamed>"),
        )
        sd["content"] = new
        changed += 1
    return changed


def _apply_recording_rewrites(
    step_defs: list[_StepDef],
    inspection: Mapping[str, Any] | None,
    coverage: StepCoverageReport | None = None,
) -> None:
    """Recording-only deterministic rewrites, in order.

    Split out of ``_apply_compile_guarantees`` to keep that function under the
    complexity ceiling; the ordering here is load-bearing and unchanged.
    """
    healed = _ensure_resilient_import(step_defs)
    if healed:
        logger.warning("Rewrote bare click/fill to resilient helpers in %d file(s)", healed)

    # A resilient click whose `primary` is a page-wide selector clicks the FIRST
    # matching node in the document, not the one the step is about — on TodoMVC the
    # toggle-all box instead of the row's toggle — and soft-fails silently, so the
    # step "passes" and the NEXT step's `toBeChecked()` carries the failure. Rewire
    # it to the scoped Locator the step already bound. Runs right after the bare
    # rewrite so calls this pass produced are covered too.
    scoped = _ensure_scoped_resilient_targets(step_defs)
    if scoped:
        logger.warning(
            "Rewired resilient click/fill to the step's SCOPED locator in %d step "
            "file(s) (unscoped-selector guard)",
            scoped,
        )

    # An in-step `page.goto('/')` navigates AWAY from the full app URL to the
    # bare ORIGIN, which under a sub-pathed `use.baseURL` is a 404/blank page —
    # so the nav `Before` hook below CANNOT mask it (this is exactly why the
    # hook alone did not stop the blank clips). Rewrite relative navigation to
    # the sub-path-preserving `absoluteNavUrl('<route>')` helper, which reads the
    # base URL from the environment so NO fixture destructure is touched. It logs its
    # own outcome (rewrites AND the unrewritable residue).
    _ensure_absolute_navigation(step_defs)

    # A `page.getBy*` locator that matches TWO elements throws a Playwright
    # strict-mode violation, which aborts the scenario mid-recording → black
    # clip → blank-gate hard-fail on a paid render. Deterministically narrow the
    # still-ambiguous ones with `.first()` (the prompt rule alone is advisory).
    strict = _ensure_strict_safe_locators(step_defs)
    if strict:
        logger.warning(
            "Appended `.first()` to ambiguous getBy* locator(s) in %d step file(s) "
            "(Playwright strict-mode guard)",
            strict,
        )

    # A landing `Given` that hard-asserts a NAMED element throws when that element
    # was invented from the pre-interaction snapshot — step 1 aborts, every later
    # When/Then never runs, and the clip records a STATIC page (112/112 demo
    # masters, median duplicate_frame_ratio 0.905). Make the precondition assert
    # NAVIGATION and soft-fail the decorative check. Runs AFTER the strict pass so
    # the locator it wraps is already `.first()`-narrowed.
    nav = _ensure_navigation_preconditions(step_defs)
    if nav:
        logger.warning(
            "Softened decorative visibility assert(s) in the navigation Given of %d "
            "step file(s) and asserted navigation instead (static-clip guard)",
            nav,
        )

    # A landing `Given` may still NAME an element the page does not have — measured
    # live as `getByRole('heading', { name: /TodoMVC/ })` against a page whose only
    # heading is `todos`, which soft-failed and demoted BOTH scenarios. Ground the
    # name against the scraped inventory and drop what it contradicts. Runs AFTER
    # the nav pass so the step keeps the `assertNavigated(page)` that pass appended
    # — the drop removes the invented claim, not the step's honest one.
    grounded = _ensure_grounded_role_names(step_defs, inspection)
    if grounded:
        logger.warning(
            "Dropped navigation assertion(s) naming a getByRole element absent from the "
            "scraped page inventory in %d step file(s) (invented-accessible-name guard)",
            grounded,
        )

    # A Then may assert on an id/class/testid absent from the pre-interaction snapshot
    # (measured: an invented `#todo-count`; a correct post-interaction
    # `getByTestId('todo-count')` is equally absent). A miss freezes the clip's tail
    # for the whole timeout. Drop every assertion UNVERIFIABLE against the snapshot's
    # DOM token inventory — correct ones included, see the pass — actions stay. Runs after the role pass so both
    # backfills see the final assertion set; logs + records each drop itself.
    _ensure_grounded_assertion_selectors(step_defs, inspection, coverage)

    # A generic "I see the text X" Then scoped to `getByRole('listitem').filter(...)`
    # can never match text that sits outside a list (showcase 01a15d7c clip-5: a
    # top-level <div> disclaimer, 5 s frozen tail). Widen it to page-wide `getByText`.
    # Logs its own outcome.
    _ensure_page_wide_text_presence(step_defs)

    # `toHaveText` is EXACT equality, so `toHaveText("2 items")` against TodoMVC's
    # "2 items left" counter can never pass — the scenario aborts mid-recording,
    # the clip records black, and the render dies as a poison job. In a recording
    # package, surviving to produce footage beats assertion precision: relax plain
    # string comparisons to a substring match.
    survivable = _ensure_survivable_text_assertions(step_defs)
    if survivable:
        logger.warning(
            "Relaxed exact `toHaveText` to `toContainText` in %d step file(s) "
            "(recording-survivability guard)",
            survivable,
        )


def _apply_recording_hooks(step_defs: list[_StepDef]) -> None:
    """The two per-scenario hooks that bracket a recorded clip.

    Split out of ``_apply_compile_guarantees`` for the same reason
    ``_apply_recording_rewrites`` was: that function sits on ruff's C901 ceiling,
    so a third recording-only branch has to live here. Both hooks are idempotent
    and both are RECORDING stream only.
    """
    # BEFORE: guarantee every scenario's fresh browser context navigates to the app
    # before its steps, so a scenario that never runs `page.goto` (assumes prior
    # state / all dwelling-stubs / soft-failed selectors) records the app instead of
    # an all-black about:blank clip that the demo_video_record blank-gate would
    # reject (sharpness 0.0).
    if _ensure_scenario_navigation(step_defs):
        logger.warning("Injected per-scenario navigation Before hook (blank-clip guard)")

    # AFTER: guarantee every scenario ends on a settled hero held for the dwell
    # budget, so clip length stops depending on whether the LLM honoured an advisory
    # prompt rule. Without it demo duration is BIMODAL (16.7-36.9s vs 1.24-3.94s per
    # clip, both `status=completed` with pass_rate 100%) and the shortfall surfaces
    # only 12 production failures later as `timeline GROSSLY UNDERFULL`.
    if _ensure_recording_pacing(step_defs):
        logger.warning(
            "Injected per-scenario hero-dwell After hook — every scenario now records "
            "at least %.1fs (pacing-determinism guard)",
            GUARANTEED_RECORDING_SECONDS_PER_SCENARIO,
        )


def _repair_step_patterns(step_defs: list[_StepDef]) -> None:
    """Rebind Gherkin continuations, then make every pattern a valid cuke expression.

    Both passes must run BEFORE dedupe/subsumption/coverage and in this order:
    ``_STEP_DEF_RE`` only sees Given/When/Then, so an unrewritten ``And(`` block is
    invisible to every pattern-based pass — including the escaping below.
    """
    rebound = _ensure_valid_step_bindings(step_defs)
    if rebound:
        logger.warning(
            "Rebound Gherkin continuation call(s) (`And(`/`But(`) to the preceding "
            "Given/When/Then binding in %d step file(s)",
            rebound,
        )

    # An unescaped `/` — the `//` of a URL the model mirrored out of the feature
    # sentence — is an EMPTY ALTERNATIVE, and cucumber-expressions throws at compile
    # time, so bddgen exits 1 and the paid render dies. Escaping is purely additive on
    # the pattern text and never introduces a `/` or `\` into a pattern that had none,
    # so every downstream pass sees byte-identical input for the patterns it can model.
    escaped = _ensure_valid_cuke_expressions(step_defs)
    if escaped:
        logger.warning(
            "Escaped cucumber-alternation `/` in step-def pattern(s) in %d step file(s) "
            "(bddgen 'Alternative may not be empty' guard)",
            escaped,
        )


def _apply_compile_guarantees(
    step_defs: list[_StepDef],
    features: list[_Feature],
    *,
    recording: bool,
    inspection: Mapping[str, Any] | None = None,
    coverage: StepCoverageReport | None = None,
) -> list[_StepDef]:
    """Deterministic passes that make the generated suite compile AND drive the
    app (never blank a clip), regardless of LLM imperfections. Order matters:
    cuke-expression escaping → dedupe → expect import → resilient rewrite →
    step-coverage LAST (so it measures the final def set and its stub is never
    removed downstream).
    """
    # FIRST: normalize invented cucumber param types (e.g. `{number}` → `{float}`) so
    # every downstream pattern-based pass (dedupe / subsumed-literal) and bddgen itself
    # see only registrable built-in types — an unrewritten `{number}` aborts the whole
    # compile with `Undefined parameter type`.
    renamed = _normalize_param_types(step_defs)
    if renamed:
        logger.warning(
            "Normalized invented cucumber param type(s) to `{float}` in %d file(s)", renamed
        )

    # BEFORE every pattern-based pass: `And(`/`But(` are not playwright-bdd bindings
    # (`ReferenceError: And is not defined` → bddgen exit 1 → dead render). Rebind
    # them to the preceding Given/When/Then kind. Must run first — `_STEP_DEF_RE`
    # only sees Given/When/Then, so an unrewritten `And(` block would be invisible to
    # dedupe, subsumption and coverage.
    # Alongside the rebinding: a file may call a VALID binding it never brought into
    # scope (the LLM omitted `const { Given, When, Then } = createBdd();`). That is the
    # same dead-render class, so repair it here rather than letting the compile gate
    # reject an otherwise-good generation.
    _ensure_bdd_bindings_in_scope(step_defs)
    _repair_step_patterns(step_defs)

    # A value matcher whose expected argument is provably `undefined` raises a
    # Playwright MATCHER error, not an assertion failure — the step dies, the scenario
    # aborts, and a recording is left on a static page. Measured live on orchestrator
    # E2E 32150382725. Applies to EVERY stream: an assertion that cannot pass anywhere
    # hides no real product failure, so dropping it is always the honest repair.
    stripped = _ensure_no_undefined_value_assertions(step_defs)
    if stripped:
        logger.warning(
            "Dropped value assertion(s) with a provably-undefined expected argument "
            "in %d step file(s) (matcher-error guard)",
            stripped,
        )

    # Duplicates are deterministically resolvable — the LLM keeps emitting them
    # even after a correction, so drop the extra definitions (keep the first).
    if _duplicate_step_patterns(step_defs):
        before = sum(sd.get("step_count", 0) for sd in step_defs)
        step_defs = _dedupe_step_defs(step_defs)
        after = sum(sd.get("step_count", 0) for sd in step_defs)
        logger.warning("Dropped %d duplicate step definition(s) deterministically", before - after)

    # Exact dedup only catches identical raw patterns. The LLM also emits a
    # general parameterized def AND a hardcoded literal specialization of the
    # SAME step (different raw strings) — bddgen then aborts with 'Multiple
    # definitions matched'. Drop the subsumed literal (a covering general def
    # survives, so coverage never shrinks). Runs after dedup so identical
    # literals are already collapsed.
    subsumed = _subsumed_literal_patterns(step_defs)
    if subsumed:
        before = sum(sd.get("step_count", 0) for sd in step_defs)
        step_defs = _filter_step_def_blocks(step_defs, lambda pattern: pattern in subsumed)
        after = sum(sd.get("step_count", 0) for sd in step_defs)
        logger.warning(
            "Dropped %d step def(s) subsumed by a general pattern (bddgen ambiguity)",
            before - after,
        )

    # Guarantee `expect(...)` is in scope — calling it without an import compiles
    # but throws `ReferenceError: expect is not defined` and blanks the footage.
    fixed = _ensure_expect_import(step_defs)
    if fixed:
        logger.warning("Injected missing `expect` import into %d step file(s)", fixed)

    # Recording packages only: rewrite any bare `page.click`/`page.fill` to the
    # shipped resilient helpers so a stale selector soft-fails (keeps the clip
    # non-blank) instead of throwing and aborting the scenario. The
    # `lib/recording_helpers` module ships only with the demo_video package.
    if recording:
        _apply_recording_rewrites(step_defs, inspection, coverage)

    # Recording packages only: the two per-scenario hooks that bracket a clip.
    if recording:
        _apply_recording_hooks(step_defs)

    # LAST, recording packages only — append a non-fatal dwelling stub for any
    # still-uncovered Gherkin step so bddgen never hits an "undefined step" throw
    # that would blank a clip. NOT applied to the QA/accessibility/documentation
    # streams: those run `missingSteps:'fail-on-gen'`, where an uncovered step is
    # an HONEST generation failure the operator must see — silently stubbing it to
    # a dwelling no-op would convert a real QA gap into a false green.
    if recording:
        _apply_step_coverage(features, step_defs, coverage)

    return step_defs


def _apply_step_coverage(
    features: list[_Feature],
    step_defs: list[_StepDef],
    coverage: StepCoverageReport | None,
) -> None:
    """Stub every uncovered step, then MEASURE the result.

    Split out of ``_apply_compile_guarantees`` because that function sits on ruff's
    C901 ceiling. The measurement is the point: an auto-stub PASSES and performs no
    action, so without a structured record nothing downstream can distinguish a
    stubbed scenario from a real one — the WARNING below is not a signal a caller
    can act on.
    """
    total_steps = len(_extract_steps_from_features(features))
    stubbed_texts: list[str] = []
    stubbed = _ensure_step_coverage(features, step_defs, stubbed_out=stubbed_texts)
    if coverage is not None:
        coverage.steps_total = total_steps
        coverage.steps_stubbed = stubbed
        coverage.stubbed_steps = stubbed_texts[:STUB_SAMPLE_LIMIT]
    if not stubbed:
        return
    # ⚠️ The PROPORTION is the signal, not the count. One stub in twenty is noise;
    # most of the steps stubbed means the "demo" is a recording of a static page,
    # which is exactly the shape behind duplicate-frame films.
    share = (stubbed / total_steps) if total_steps else 1.0
    logger.warning(
        "Added %d fallback stub(s) for uncovered Gherkin step(s) — %.0f%% of "
        "%d step(s) in this suite now PASS WITHOUT ACTING. A high share means "
        "the recording will show a static page: the stub only waits.",
        stubbed,
        share * 100,
        total_steps,
    )


class StepDefGenerator:
    """Generates TypeScript step definitions for playwright-bdd."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def generate(
        self,
        features: list[dict[str, Any]],
        inspection: dict[str, Any],
        platform: str = "web",
        *,
        auth_config: dict[str, Any] | None = None,
        recording: bool = False,
        coverage: StepCoverageReport | None = None,
    ) -> list[_StepDef]:
        """Generate playwright-bdd step definitions from feature files.

        Args:
            features: List of feature dicts with filename and content.
            inspection: Inspection data with page details and selectors.
            platform: Target platform (web, mobile).
            auth_config: Optional auth config (login_url, type, etc.).
            recording: True for the demo_video stream — the package ships the
                shared ``lib/recording_helpers`` library, so the prompt invites
                the LLM to reuse those helpers for common recording idioms.
            coverage: Optional out-parameter. When given (recording stream), it is
                filled with how many of the suite's Gherkin steps ended up as
                non-acting auto-stubs, so the caller can publish the stub share as
                structured data instead of leaving it in a log line. Passed IN
                rather than returned so no existing caller's return type changes,
                and per-call rather than on ``self`` so concurrent jobs sharing one
                generator instance cannot read each other's measurement.

        Returns:
            List of dicts with keys: filename, content, step_count.
        """
        steps = _extract_steps_from_features(features)
        base_url = inspection.get("base_url", "")

        steps_summary = "\n".join(
            f"  {s['keyword']} {s['text']}"
            for s in steps[:100]  # Limit to avoid token overflow
        )

        pages_summary_lines: list[str] = []
        for page in inspection.get("pages", []):
            pages_summary_lines.append(f"Page: {page.get('title', '')} ({page.get('url', '')})")
            for elem in page.get("interactive_elements", []):
                tag = elem.get("tag", "")
                label = elem.get("label") or elem.get("text") or elem.get("name") or ""
                selector = elem.get("selector", "")
                pages_summary_lines.append(f"  {tag}: {label} -> {selector}")
        pages_summary = "\n".join(pages_summary_lines)

        auth_instruction = ""
        if auth_config:
            auth_instruction = (
                "\nIMPORTANT — Authentication:\n"
                "The application requires login. Authentication is handled automatically "
                "by a global setup project that saves Playwright storageState. All tests "
                "run in an authenticated browser context. Do NOT implement login steps in "
                "the step definitions — the user is already logged in when tests start.\n\n"
            )

        helpers_instruction = ""
        if recording:
            helpers_instruction = (
                "\nShared recording helpers (this is a video-recording package):\n"
                "A ready-made helper library is shipped alongside your steps at "
                "`../lib/recording_helpers`. PREFER importing from it over re-deriving "
                "these idioms — e.g. `import { typeVisibly, dwellWithScroll, heroDwell, "
                "dismissCoachMark, clearNameCollisions, recoverEditorIfCrashed } from "
                "'../lib/recording_helpers';`. Use `typeVisibly(locator, value)` for "
                "human-paced typing, `dwellWithScroll(page)` to ease over a settled hero "
                'with motion (clips are shown via clip_anchor="end"), and the '
                "wizard/editor helpers for multi-step flows. Only hand-roll a helper "
                "the library does not provide.\n"
                # The dwell is NOT asked for any more — `_ensure_recording_pacing`
                # injects an `After` hook that holds every scenario's tail. Asking was
                # the defect: the model complied on some suites and not others, and clip
                # length came out bimodal (16.7-36.9s vs 1.24-3.94s, both
                # `status=completed`). The line below is stated so a model that reaches
                # for a manual dwell knows it is redundant rather than missing; the
                # per-page ledger makes it harmless either way.
                "You do NOT need to add a final hold/dwell to a scenario: the package "
                "ships an `After` hook that holds every scenario's settled tail for the "
                "shared dwell budget. An extra `heroDwell(page)` is not an error (holds "
                "share one per-scenario budget, so it will not double the clip) but it "
                "is unnecessary — spend your steps on DRIVING the app instead.\n"
                "For EVERY click or fill, use the resilient helpers instead of a bare "
                "`page.click`/`page.fill`: `await resilientClick(page, '<selector>', "
                "'<accessible label>')` and `await resilientFill(page, '<selector>', "
                "value, '<accessible label>')`, imported from `../lib/recording_helpers`. "
                "Take `<accessible label>` from the inspection `label -> selector` "
                "metadata. They self-heal a stale selector (role/text/label fallback) "
                "and, on a total miss, record a soft-fail WITHOUT throwing — so one bad "
                "selector never aborts the whole scenario and blanks the clip. NEVER call "
                "raw `page.click`/`page.fill` in a recording package.\n"
                # The `<selector>` slot accepts a LOCATOR too. Five consecutive todomvc
                # packages read the string-only signature, concluded a scoped target was
                # inexpressible, and clicked a page-wide `input[type="checkbox"]` (i.e.
                # TodoMVC's toggle-all) while leaving the scoped locator bound and
                # unused. Advisory, per #119: `_ensure_scoped_resilient_targets` is the
                # deterministic repair; this only stops the model deliberating.
                "The FIRST target argument accepts a Playwright LOCATOR as well as a "
                "selector string, so a target that must be SCOPED to a row is written "
                "directly: `const row = page.getByRole('listitem').filter({ hasText: "
                "text }).first(); await resilientClick(page, row.getByRole('checkbox')"
                ".first(), text);`. NEVER widen a scoped target to a page-wide selector "
                'like `input[type="checkbox"]` — that clicks the FIRST such element in '
                "the document, not the one the step is about.\n"
                # WHY: measured on prod — the landing `Given` hard-asserted an INVENTED
                # named element, threw, and aborted step 1, so every later When/Then
                # never ran and the clip recorded a static page with ZERO interaction
                # (112/112 demo masters static, median duplicate_frame_ratio 0.905).
                # `_ensure_navigation_preconditions` is the deterministic backstop; this
                # is the belt, and it stops the model wasting step 1 on decoration.
                "NAVIGATION PRECONDITIONS assert NAVIGATION, not decoration. A `Given` "
                "that merely puts the browser somewhere ('I am on the X page', 'I open "
                "X', 'I visit X') must navigate and then assert only that navigation "
                "SUCCEEDED — `await assertNavigated(page);` (it waits for the document "
                "and throws only on a genuinely blank/unnavigated page). Do NOT open a "
                "scenario by asserting some named link/heading/banner is visible: if "
                "that element is named differently in the real app the step THROWS, the "
                "scenario ABORTS at step 1, and every later When/Then never runs — the "
                "clip then records a static page with no interaction at all, which is "
                "the single most expensive defect in this pipeline. Keep hard "
                "`expect(...)` assertions for the `Then` steps that carry the "
                "scenario's actual claim; for a merely decorative presence check "
                "inside a precondition use the non-fatal `await softExpectVisible("
                "page, <locator>);`. The CONTENT-VISIBILITY GATE below still applies "
                "to steps that OPEN A SCREEN mid-scenario.\n\n"
            )

        prompt = (
            "Generate TypeScript step definition files for playwright-bdd using "
            "the createBdd() pattern.\n\n"
            f"Base URL: {base_url}\n"
            f"Platform: {platform}\n\n"
            f"{auth_instruction}"
            f"{helpers_instruction}"
            "Steps to implement:\n"
            f"{steps_summary}\n\n"
            "Available page elements and selectors:\n"
            f"{pages_summary}\n\n"
            "Requirements:\n"
            "- Import { createBdd } from 'playwright-bdd'\n"
            "- Import { expect } from '@playwright/test' — `expect` is a standalone "
            "assertion helper, NOT a fixture. Import it at module scope and call it "
            "directly (`await expect(locator).toBeVisible()`). Do NOT destructure "
            "`expect` from the callback fixture arg — it is not a fixture and would "
            "be undefined.\n"
            "- Use const { Given, When, Then } = createBdd()\n"
            # WHY: the step list above used to echo the RAW Gherkin keyword, so the model
            # dutifully wrote `And('...', ...)`. `createBdd()` returns no such binding —
            # the file throws `ReferenceError: And is not defined` at module load, bddgen
            # exits 1 and the whole paid render dies (live, corpus run 30793758726,
            # 2026-08-03). The step list is now pre-resolved and this rule is the belt to
            # `_ensure_valid_step_bindings`' braces.
            "- ONLY `Given`, `When` and `Then` are step bindings. `And`, `But` and `*` "
            "are Gherkin CONTINUATION keywords that INHERIT the kind of the preceding "
            "Given/When/Then step — they are NOT functions. NEVER write `And(...)` or "
            "`But(...)`: define a step written `And I see the list` after a `When` step "
            "as `When('I see the list', async ({ page }) => { ... });`. The step list "
            "above is already given to you with each keyword resolved — use exactly the "
            "keyword shown.\n"
            "- `page` and ALL Playwright FIXTURES are available ONLY inside the "
            "Given/When/Then async callbacks, destructured from the fixture arg, "
            "e.g. `Given('I am on the page', async ({ page }) => { ... });`. NEVER "
            "reference `page` at module/top level (no `const x = page.locator(...)` "
            "at file scope) — that throws `ReferenceError: page is not defined`. "
            "Declare reusable selectors as plain STRING constants at module scope "
            "and call `page.locator(sel)` INSIDE each callback.\n"
            "- PREFER resilient, accessibility-first locators that survive DOM "
            "churn: `page.getByRole(...)`, `page.getByText(...)`, "
            "`page.getByLabel(...)`, `page.getByPlaceholder(...)`. Fall back to a "
            "CSS/XPath `page.locator(sel)` ONLY when the element has no accessible "
            "name or role to target. When the inspection data gives a raw CSS "
            "selector, translate it to a role/text/label locator where the element "
            "metadata (label/text/tag) makes that possible.\n"
            # WHY this rule exists: Playwright locators are STRICT. The moment a
            # `getBy*` locator resolves to MORE THAN ONE element, the action or
            # assertion throws `strict mode violation: … resolved to 2 elements`,
            # which aborts the scenario mid-recording — the clip records BLACK and
            # the orchestrator's blank-clip gate hard-fails the paid render. That is
            # how the 2026-07-31 corpus sweep lost 3/3 renders on a bare
            # `getByRole('link', { name: 'TodoMVC' })` matching both the nav link and
            # a body link. `_ensure_strict_safe_locators` enforces this
            # deterministically for recording packages; the rule is here so the model
            # emits the intended locator rather than relying on the repair pass.
            "- STRICT MODE (mandatory): Playwright THROWS if a locator resolves to "
            "more than one element, which aborts the scenario and records a BLACK "
            "clip. EVERY `getByRole`/`getByText`/`getByLabel`/`getByPlaceholder` "
            "locator you assert on or act upon MUST be unambiguous: end the chain "
            "with `.first()` (or narrow it first with `.filter({ hasText: '...' })`, "
            "a scoping `page.getByRole('navigation').getByRole('link', ...)`, or "
            "`{ exact: true }` on the name). Prefer `{ exact: true }` for a short "
            "generic name — a substring match like `{ name: 'Settings' }` also matches "
            "'Account Settings'. Write `await expect(page.getByRole('link', "
            "{ name: 'Settings', exact: true }).first()).toBeVisible();`, NEVER the "
            "bare `await expect(page.getByRole('link', { name: 'Settings' }))"
            ".toBeVisible();`. The ONLY exception is a deliberately multi-element "
            "assertion (`toHaveCount`, `.count()`, `.all()`) — leave those unnarrowed. "
            "The names above are FORMAT examples only — never copy a literal name "
            "out of this instruction into your code; take every name from the page "
            "inventory below.\n"
            # WHY this rule exists: `toHaveText` is EXACT equality on the element's
            # FULL text. `toHaveText("2 items")` against TodoMVC's counter — which
            # always reads "2 items left" — can never pass, so the scenario aborts
            # mid-recording, the clip records BLACK and the demo render dies as a
            # poison job. `_ensure_survivable_text_assertions` relaxes these
            # deterministically for recording packages; the rule is here so the model
            # picks the right matcher in the first place.
            "- TEXT ASSERTIONS: prefer `toContainText` over `toHaveText` whenever the "
            "text may carry surrounding copy — counters, labels, badges, status text "
            "(`toHaveText('2 items')` FAILS on '2 items left' and aborts the "
            "recording). Reserve `toHaveText` for text you know is complete and "
            "exact, such as a heading you read verbatim from the inspection data.\n"
            "- Use the provided selectors as the source of truth for WHICH element "
            "to act on, but express the locator resiliently per the rule above\n"
            "- CONTENT-VISIBILITY GATE (critical — this is a video-recording "
            "package; a clip of a rolling/half-loaded screen is a defect). In "
            "every step that navigates or opens a screen, AWAIT the real content "
            "before dwelling, scrolling, or capturing: first "
            "`await page.waitForLoadState('networkidle').catch(() => {});`, then "
            "assert the screen's key content container is visible — e.g. "
            "`await expect(page.getByRole('heading', { name: /.../ }).first())"
            ".toBeVisible();` (note the `.first()` — see the STRICT MODE rule above; "
            "the bare form throws and blanks the clip as soon as a second heading "
            "matches) "
            "(use the most specific real content locator the inspection data "
            "gives you), and ONLY THEN dwell/capture. Waiting for `body` or "
            "`domcontentloaded` alone is NOT sufficient — those resolve on the "
            "app shell before data has rendered.\n"
            # WHY: the page inventory is captured from a PRE-INTERACTION snapshot, so
            # rows/chips/toasts that only exist after the user acts are simply absent —
            # and the model fills the gap by inventing a CSS class chain that never
            # resolves (live: `li.todo-list:has-text(...)`, `.title` against an app
            # whose real DOM is `ul.todo-list > li` / `label`).
            "- NEVER INVENT A SELECTOR. Every selector you write must come from the "
            "page inventory above, or be a role/label/text query for content the "
            "Gherkin step itself names (e.g. the todo text the user just typed). The "
            "inventory is a snapshot taken BEFORE any interaction, so elements that "
            "only appear AFTER the user acts are NOT listed — for those, target them "
            "by the text the scenario itself supplies, NEVER by a guessed class chain "
            "like `li.todo-item .title`. When you are unsure of an element's markup, "
            "prefer `getByRole`/`getByLabel`/`getByText` over any CSS class selector.\n"
            # WHY: this rule used to give ONE form for all scenario-supplied text —
            # `getByRole('listitem').filter({ hasText })` — so the model scoped a generic
            # "I see the text X displayed on the page" step to a list item. On
            # demo.playwright.dev/todomvc the disclaimer is a top-level <div>, so the
            # locator could never match (showcase run 01a15d7c clip-5, 5 s timeout,
            # frozen tail). `_ensure_page_wide_text_presence` is the deterministic
            # backstop; this rule makes the model emit the right scope first.
            "- TEXT PRESENCE: a step that only says some text is shown — 'I see the "
            "text X', 'X is displayed', 'the page shows X', 'I should see X' — asserts "
            "PAGE-WIDE: `await expect(page.getByText(value).first()).toBeVisible();` "
            "(or `.toContainText(value)`). Do NOT assume the text sits in a list. Scope "
            "to a list item — `page.getByRole('listitem').filter({ hasText: value })"
            ".first()` — ONLY when the step itself names a list, item, row, todo, task "
            "or entry (e.g. 'the todo X is in the list').\n"
            "- NEVER navigate with a RELATIVE path: `await page.goto('/')` (or "
            "`'/login'`, `'/#/active'`) resolves against the bare ORIGIN and STRIPS "
            "any sub-path off baseURL, so the app never loads and the clip records "
            "BLANK. Destructure the `baseURL` fixture and navigate to the full URL, "
            "e.g. `Given('I am on the app', async ({ page, baseURL }) => { await "
            "page.goto(baseURL!); });`; for a sub-route, append to `baseURL` "
            "(`` `${baseURL}/settings` ``) — do NOT use `new URL(path, baseURL)`, "
            "which discards baseURL's own path.\n"
            # WHY: `/` is the ALTERNATION operator in a cucumber expression, so the
            # `//` of a URL mirrored verbatim out of the feature sentence parses as an
            # EMPTY alternative — cucumber-expressions throws 'Alternative may not be
            # empty', bddgen exits 1 and the whole paid render dies (live 2026-08-15).
            # `_ensure_valid_cuke_expressions` is the deterministic backstop; this is
            # the belt.
            "- NEVER embed a URL, path or any other literal containing `/` in the step "
            "PATTERN string — `/` means ALTERNATION in a cucumber expression, so "
            "`When('I navigate to \"https://example.com\"', ...)` fails to compile. "
            "Parameterise it as `{string}` and read the value from the callback arg, "
            "e.g. `When('I navigate to {string}', async ({ page }, url: string) => "
            "{ ... });` (playwright-bdd passes the fixtures arg FIRST, then the "
            "captured parameters in order).\n"
            "- Include proper async/await and assertion patterns\n"
            "- Wrap each file in a ```typescript code block\n"
            "- Add a // filename: <name>.ts comment at the top of each block\n"
        )

        system_prompt = (
            "You are a senior test automation engineer who writes TypeScript "
            "playwright-bdd step definitions. Output only valid TypeScript code."
        )

        raw = await self._llm.generate(prompt, system_prompt=system_prompt)
        try:
            step_defs = _parse_step_defs(raw)
        except StepDefGenerationError as exc:
            # A prose answer is a one-off bad completion, not a permanent defect, so
            # hard-failing the FIRST parse would trade a corruption bug for an
            # availability regression — a single bad completion killing a paid run.
            # Self-heal it exactly like the structural defects below: regenerate ONCE
            # with a corrective instruction appended to the same prompt. The second
            # parse is deliberately NOT wrapped — it raises, which bounds this at one
            # extra LLM call and keeps `_looks_like_typescript` gating attempt 2
            # just as strictly as attempt 1 (no fallback to writing prose out).
            logger.warning("Generated step defs are not step definitions (%s); regenerating", exc)
            raw = await self._llm.generate(
                prompt + _PROSE_ANSWER_CORRECTION, system_prompt=system_prompt
            )
            step_defs = _parse_step_defs(raw)

        # Self-heal the observed LLM failure modes that make bddgen fail downstream
        # in the test runner: (1) `page`/`expect` used at module scope →
        # `ReferenceError: page is not defined`; (2) the same step pattern defined
        # twice → "Multiple definitions matched scenario step". Regenerate ONCE with
        # explicit corrections (gives the model a chance to fix both cleanly), then:
        # BEFORE judging: repair the mechanically-fixable omission of
        # `const { Given, When, Then } = createBdd();`. Otherwise a missing line of
        # boilerplate reads as an "undefined binding" and burns a whole LLM
        # regeneration to re-derive something we can inject deterministically.
        _ensure_bdd_bindings_in_scope(step_defs)

        corrections, problems = _step_def_issues(step_defs)
        if corrections:
            logger.warning("Generated step defs invalid (%s); regenerating", "; ".join(problems))
            raw = await self._llm.generate(prompt + corrections, system_prompt=system_prompt)
            step_defs = _parse_step_defs(raw)

        # Module-scope fixtures can't be auto-fixed safely (the body references a
        # name that only exists per-step) — fail loudly if still present.
        violation = _first_module_scope_violation(step_defs)
        if violation:
            raise StepDefGenerationError(
                "generated step definitions reference `page`/`expect` outside a step "
                f"callback ({violation}) after one regeneration; playwright-bdd exposes "
                "them only as per-step fixtures"
            )

        # Deterministic compile/run guarantees (dedupe, expect import, resilient
        # rewrite, step-coverage stubs) — see `_apply_compile_guarantees`.
        step_defs = _apply_compile_guarantees(
            step_defs,
            features,
            recording=recording,
            inspection=inspection,
            coverage=coverage,
        )

        # COMPILE GATE: no generated file may redeclare a module-scope binding —
        # Babel throws an unattributed `VarRedeclaration` for this, which the
        # earlier `_ensure_bdd_bindings_in_scope` repair could itself introduce if
        # it ever regressed to injecting a full destructure alongside a partial one.
        _reject_duplicate_module_scope_bindings(step_defs)

        # COMPILE GATE (last): every module-scope binding call must resolve to a name
        # the file defines or imports. `_ensure_valid_step_bindings` has already
        # rebound the known `And(`/`But(` case, so a residue here is a symbol the LLM
        # invented — it would throw `ReferenceError` at module load and make bddgen
        # exit 1 mid-render. Fail the generation with the symbol named instead.
        undefined = _undefined_module_bindings(step_defs)
        if undefined:
            raise StepDefGenerationError(
                "generated step definitions call undefined module-scope binding(s) "
                f"({', '.join(undefined[:5])}); playwright-bdd's createBdd() exposes only "
                "Given/When/Then — And/But/* are Gherkin continuation keywords, not bindings"
            )

        logger.info("Generated %d step definition files", len(step_defs))
        return step_defs
