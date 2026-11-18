"""A helper the recording NEEDS must be INJECTED, never merely asked for.

``lib/recording_helpers.ts`` ships two kinds of helper. The survivability half —
``resilientClick``, ``resilientFill``, ``softExpectVisible``, ``assertNavigated``
— is FORCE-INJECTED by ``step_def_generator``: the passes rewrite call sites and
guarantee the import, deterministically. The pacing half — ``heroDwell`` — was
only ASKED FOR, in the generator prompt's ``helpers_instruction``.

That asymmetry was the bug. A prompt rule is advisory, so the model honoured it
on some suites and not others, and demo duration came out BIMODAL: 55 runs at
16.7-36.9s per clip, 24 runs at 1.24-3.94s. **Both report ``status=completed``
with pass_rate 100%** — nothing inside the suite can tell them apart — and the
18x shortfall surfaces only at the far end of the pipeline, as
``timeline GROSSLY UNDERFULL - 4.06s of the 30.00s target (13.5%)``, 12
production failures deep.

This module is the gate that keeps that class closed. It does three things a
single "is heroDwell injected?" assertion would not:

* **It ENUMERATES the library.** Every export of ``recording_helpers.ts`` must
  appear in exactly one of ``REQUIRED_INJECTED`` / ``OPTIONAL_BY_DESIGN`` below,
  with a written reason. Adding a new helper to the library and leaving it to the
  prompt therefore fails the build — which is precisely the position ``heroDwell``
  was in. A test that only knew about ``heroDwell`` would have caught this bug
  once; this one catches the next one.
* **It has a COUNT FLOOR.** "Every export is classified" is trivially satisfied by
  deleting the exports. ``MIN_EXPORTED_HELPERS`` is the assertion that catches
  that (reference: ``html_render_service`` ``test_remote_navigation_is_dns_pinned``).
* **It proves injection END TO END, from naive input.** The fixture below is what
  a model that ignored every prompt rule emits — bare ``page.click``/``page.fill``,
  a hard ``toBeVisible`` landing assertion, no dwell anywhere. Every REQUIRED
  helper must appear in the package that ``_apply_compile_guarantees`` returns.
  Asserting the pass exists would prove nothing about whether it runs.

Parsing is over the ``export`` declarations themselves, not a keyword grep: this
module's own prose names ``heroDwell`` a dozen times, and a regex that matched
prose would report the library as fully exported no matter what the file said.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src.services.generation.step_def_generator import (
    GUARANTEED_RECORDING_SECONDS_PER_SCENARIO,
    HERO_DWELL_MS,
    _apply_compile_guarantees,
    _ensure_recording_pacing,
    _parse_step_defs,
    _read_hero_dwell_ms,
)

_HELPERS_TS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "templates"
    / "scaffold"
    / "lib"
    / "recording_helpers.ts"
)

# `export function f(`, `export async function f(`, `export const C =`.
# Anchored at line start on the `export` keyword, so a mention inside a comment,
# a docstring or a string literal is invisible to it.
_EXPORT_RE = re.compile(
    r"^export\s+(?:async\s+)?(?:function|const)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)

#: Helpers whose ABSENCE changes the recording's observable outcome, so the
#: generator must inject them rather than ask for them. Each maps to what breaks
#: without it. Every one of these must be reachable from a naive package.
REQUIRED_INJECTED: dict[str, str] = {
    "heroDwell": (
        "THE PACING FIX. The clip's tail is what the compositor shows "
        '(clip_anchor="end"); without a dwell the scenario records 1.24-3.94s '
        "instead of 16.7-36.9s and the timeline comes out GROSSLY UNDERFULL. "
        "Injected by `_ensure_recording_pacing` as an `After` hook in "
        "steps_pacing.ts, which is by construction the scenario's last act."
    ),
    "dwellRemainingMs": (
        "The idempotency half of the pacing fix: the After hook tops the scenario "
        "up to the budget rather than dwelling unconditionally, so a scenario "
        "whose own steps already called heroDwell records ~22s, not ~44s. Without "
        "it the injection would itself be the regression."
    ),
    "resilientClick": (
        "A bare page.click on a stale selector THROWS, aborting the scenario "
        "mid-recording -> blank clip -> the demo_video blank-gate hard-fails a "
        "paid render. Injected by `_ensure_resilient_import`."
    ),
    "resilientFill": (
        "Same failure mode as resilientClick for page.fill. Injected by "
        "`_ensure_resilient_import` — which probes PER NAME, because a combined "
        "probe once matched an import naming only the other helper and shipped a "
        "ReferenceError."
    ),
    "softExpectVisible": (
        "A landing Given's decorative `expect(...).toBeVisible()` names an element "
        "chosen from the PRE-interaction snapshot, i.e. often invented. Hard-"
        "asserting it aborts step 1 and every later step never runs (112/112 demo "
        "masters static). Injected by `_ensure_navigation_preconditions`."
    ),
    "assertNavigated": (
        "The honest other half of the same rewrite: softening the decorative check "
        "must not also soften a REAL navigation failure, so the precondition still "
        "throws on about:blank. Injected by `_ensure_navigation_preconditions`."
    ),
}

#: Helpers that are genuinely the model's (or a hand-author's) call. Each maps to
#: why forcing it would be wrong — not merely why it was not forced. "Nobody got
#: round to it" is not a reason that belongs in this dict.
OPTIONAL_BY_DESIGN: dict[str, str] = {
    "HERO_DWELL_MS": (
        "A constant, not a call site — there is no such thing as injecting it. It "
        "is nonetheless load-bearing and single-owned: `_read_hero_dwell_ms` parses "
        "THIS declaration, and the tests below pin that binding."
    ),
    "NETWORK_IDLE_TIMEOUT_MS": (
        "A constant, not a call site — nothing to inject. Every "
        "`waitForLoadState('networkidle').catch(...)` best-effort settle in this "
        "file (and the sites that import it) reads it, bounding what would "
        "otherwise silently fall back to the 15s `navigationTimeout` and land in "
        "the recorded clip's measured length. See "
        "`test_networkidle_waits_carry_an_explicit_timeout.py`."
    ),
    "setScenarioDwellBudgetMs": (
        "CALLER-CONDITIONAL, not the model's call and not forceable. It narrows "
        "the ledger to an ORDERED film duration's per-scenario share, and it is "
        "force-injected — as `steps_dwell_budget.ts`, by `ensure_dwell_budget` — "
        "whenever, and only whenever, the caller sent `target_duration_seconds`. "
        "Injecting it unconditionally would emit "
        "`setScenarioDwellBudgetMs(HERO_DWELL_MS)`, a no-op that puts a duration "
        "decision into packages for callers who never ordered one. The model is "
        "never asked for it in any prompt, so the bimodal-duration class this "
        "file guards cannot reopen through it."
    ),
    "spendDwell": (
        "THE LEDGER ITSELF, not a call site the model should ever reach. It is the "
        "single charge point for every discretionary hold, and it is exported ONLY "
        "so the force-injected `steps_autostub.ts` can charge the SAME per-scenario "
        "ledger `heroDwell` charges. Before that, the stub kept a private WeakMap "
        "and its own ~20.25s ceiling derived from `PW_TEST_TIMEOUT_MS` — a second "
        "budget that no ordered duration narrowed, so one clip could spend both "
        "(a 30s order came back with 43.56s of footage). Forcing the model to call "
        "it would be WRONG: a model-authored spend is exactly the discretionary "
        "hold `heroDwell` already wraps, and asking for the raw ledger invites "
        "double-charging the thing this file exists to keep single-owned."
    ),
    "isDwellBudgetOrdered": (
        "A PREDICATE READ BY GENERATED CODE, never a model call. `steps_autostub.ts` "
        "uses it to decide whether to charge the shared ledger (a duration WAS "
        "ordered) or keep its own independent timeout-derived budget (none was) — "
        "which is what preserves the no-order path byte-for-byte. It performs no "
        "pacing action of its own, so there is nothing for an injection to force."
    ),
    "RESILIENT_ATTEMPT_MS": (
        "Per-attempt timeout consumed internally by the resilient helpers, which "
        "are themselves injected. Nothing calls it directly."
    ),
    "typeVisibly": (
        "PACING-FLAVOURED BUT DELIBERATELY NOT FORCED, and this is the entry most "
        "worth arguing with. Two reasons. (1) It is no longer duration-load-"
        "bearing: with the dwell budget enforced, per-scenario length is pinned by "
        "heroDwell whether or not typing is paced, so forcing it would buy "
        "aesthetics, not determinism. (2) Forcing it would REGRESS survivability. "
        "typeVisibly takes a Locator, has no accessible-label fallback, and its "
        "`target.click()` THROWS on a missing element — so rewriting resilientFill "
        "call sites into it would reopen the abort-mid-scenario blank-clip class "
        "that resilientFill exists to close. Cheaper typing is not worth a black "
        "clip on a paid render."
    ),
    "dwellWithScroll": (
        "The motion-flavoured ALTERNATIVE to a frozen hero, chosen per scenario "
        "when narration is long. Not required — heroDwell already guarantees the "
        "floor — but it is not free either: its holds spend from the SAME "
        "per-scenario ledger, so a scenario using both records 22s, not 32s. "
        "Optional in whether it is called, not in how it is accounted."
    ),
    "saveDraft": "App-specific wizard state chaining across per-clip contexts; meaningless without a storageKey the generator does not know.",
    "resumeFromDraft": "The other half of draft chaining; same app-specific inputs.",
    "writeJsonFile": "Sidecar id hand-off between clips; the file path and payload are per-package.",
    "readJsonFile": "The read half of the sidecar hand-off; same.",
    "idFromUrl": "Needs an app-specific capture regex; nothing generic to inject.",
    "dismissCoachMark": "Needs the app's coach-mark text; a wrong guess would hide real UI.",
    "clearNameCollisions": "Needs the app's 'use suggested name' copy; recording-account specific.",
    "submitUntilAdvanced": "Needs the app's submit CTA and an advanced-when-hidden marker locator.",
    "advanceWizardUntil": "Needs per-app advance labels and a done predicate.",
    "tailViaBack": "A tail-shaping choice for flows whose submit navigates away from the money shot; wrong for flows that do not.",
    "recoverEditorIfCrashed": "WebGL/canvas-editor specific crash copy; irrelevant to most apps.",
    "fitCanvasByKeys": "Canvas zoom-to-fit key sequence; entirely per-app.",
}

#: VACUITY FLOOR. "Every export is classified" is satisfiable by emptying the
#: library. Measured 2026-09-07: 21 exports. The floor sits just below so ordinary
#: additions do not trip it, but a gutting does. Do not lower this to make a
#: deletion pass.
MIN_EXPORTED_HELPERS = 20

# What a model that ignored every prompt rule emits: bare click/fill, a hard
# decorative assertion in the landing Given, and no dwell anywhere.
_NAIVE_RECORDING_STEPS = (
    "```typescript\n"
    "// filename: demo.steps.ts\n"
    "import { createBdd } from 'playwright-bdd';\n"
    "import { expect } from '@playwright/test';\n"
    "const { Given, When, Then } = createBdd();\n"
    "Given('I am on the todo app', async ({ page }) => {\n"
    "  await page.goto('https://example.test/todomvc');\n"
    "  await expect(page.getByRole('heading', { name: 'todos' })).toBeVisible();\n"
    "});\n"
    "When('I add a todo', async ({ page }) => {\n"
    "  await page.fill('.new-todo', 'Buy milk');\n"
    "  await page.click('.add-button');\n"
    "});\n"
    "```"
)


def _exported_helpers() -> set[str]:
    return set(_EXPORT_RE.findall(_HELPERS_TS.read_text(encoding="utf-8")))


def _recording_package(source: str = _NAIVE_RECORDING_STEPS) -> str:
    """Every file `_apply_compile_guarantees` produces for the recording stream,
    concatenated — i.e. what actually ships in the package."""
    out = _apply_compile_guarantees(_parse_step_defs(source), [], recording=True)
    return "\n".join(sd.get("content", "") for sd in out)


class TestHelperLibraryIsFullyClassified:
    def test_the_library_still_has_helpers(self):
        """VACUITY FLOOR: classification is trivial over an empty library."""
        exported = _exported_helpers()
        assert len(exported) >= MIN_EXPORTED_HELPERS, (
            f"Only {len(exported)} exports parsed out of {_HELPERS_TS} "
            f"(floor {MIN_EXPORTED_HELPERS}, 21 at the 2026-09-07 measurement). "
            "Either the library was gutted, or `_EXPORT_RE` stopped matching its "
            "declaration style — in which case every other assertion in this file "
            "is passing over an EMPTY set and proving nothing. Fix the regex; do "
            "not lower the floor."
        )

    def test_every_export_is_classified_required_or_optional(self):
        """The gate that keeps the `heroDwell` class closed.

        A new helper the recording NEEDS, added to the library and left to the
        prompt, is invisible to every other test in this repo — it just quietly
        makes some suites shorter than others. Forcing a classification makes
        that a build failure at the moment the helper is written.
        """
        exported = _exported_helpers()
        classified = set(REQUIRED_INJECTED) | set(OPTIONAL_BY_DESIGN)

        unclassified = sorted(exported - classified)
        assert not unclassified, (
            f"{_HELPERS_TS.name} exports helper(s) {unclassified} that are in "
            "neither REQUIRED_INJECTED nor OPTIONAL_BY_DESIGN.\n\n"
            "Decide which: does the recording's observable outcome change when "
            "the LLM omits it? If yes it is REQUIRED and the generator must "
            "INJECT it — `heroDwell` sat in exactly this position, asked for in "
            "the prompt only, and produced bimodal 1.2s/22s clips that both "
            "reported status=completed. If no, add it to OPTIONAL_BY_DESIGN with "
            "a reason why forcing it would be WRONG (not merely why nobody has)."
        )

        phantom = sorted(classified - exported)
        assert not phantom, (
            f"Classified helper(s) {phantom} are no longer exported by "
            f"{_HELPERS_TS.name}. A stale entry makes this gate look like it "
            "covers more than it does — and a REQUIRED entry naming a deleted "
            "helper would keep passing while the guarantee is gone. Remove them."
        )

    def test_required_and_optional_are_disjoint_and_reasoned(self):
        overlap = sorted(set(REQUIRED_INJECTED) & set(OPTIONAL_BY_DESIGN))
        assert not overlap, f"Helper(s) {overlap} are classified both ways."
        for name, reason in {**REQUIRED_INJECTED, **OPTIONAL_BY_DESIGN}.items():
            assert len(reason.strip()) >= 40, (
                f"{name} carries a reason too short to be a real decision "
                f"({reason!r}). A one-word waiver is indistinguishable from an "
                "accident — state what breaks, or what forcing it would cost."
            )


class TestRequiredHelpersAreForceInjected:
    """Every REQUIRED helper must reach a package built from naive model output.

    This is the assertion that actually fails when a force-injection is removed —
    proving the pass RUNS, not merely that it exists.
    """

    @pytest.mark.parametrize("helper", sorted(REQUIRED_INJECTED))
    def test_helper_reaches_the_recording_package(self, helper: str):
        package = _recording_package()
        assert helper in package, (
            f"`{helper}` is classified REQUIRED_INJECTED but does NOT appear in "
            "the package `_apply_compile_guarantees(..., recording=True)` builds "
            "from step definitions that never mention it.\n\n"
            f"Why it is required: {REQUIRED_INJECTED[helper]}\n\n"
            "Either wire the injecting pass into `_apply_compile_guarantees` / "
            "`_apply_recording_rewrites`, or — if it genuinely is the model's "
            "call — move it to OPTIONAL_BY_DESIGN with a reason why forcing it "
            "would be wrong. Do NOT satisfy this by relaxing the assertion: a "
            "prompt rule asking for the helper is exactly the arrangement that "
            "produced the bimodal-duration defect."
        )

    def test_naive_input_really_is_naive(self):
        """Control: the fixture must not already contain what we claim is injected.

        Without this, every assertion above could be passing on the fixture's own
        text and would stay green with the whole pacing pass deleted.
        """
        for helper in REQUIRED_INJECTED:
            assert helper not in _NAIVE_RECORDING_STEPS, (
                f"`{helper}` already appears in the naive fixture, so "
                "`test_helper_reaches_the_recording_package` cannot distinguish "
                "injection from the fixture. Remove it from the fixture."
            )

    def test_non_recording_stream_gets_no_pacing_hook(self):
        """A 22-second pause per scenario in a QA suite is pure cost — there is no
        footage to show for it."""
        out = _apply_compile_guarantees(
            _parse_step_defs(_NAIVE_RECORDING_STEPS), [], recording=False
        )
        assert not any(sd["filename"] == "steps_pacing.ts" for sd in out)


class TestPacingHookShape:
    def test_injects_an_after_hook_that_dwells(self):
        defs = _parse_step_defs(_NAIVE_RECORDING_STEPS)
        assert _ensure_recording_pacing(defs) == 1
        hook = defs[-1]
        assert hook["filename"] == "steps_pacing.ts"
        content = hook["content"]
        # Its own createBdd(), like the nav hook — so no existing file is perturbed.
        assert "import { createBdd } from 'playwright-bdd';" in content
        assert "const { After } = createBdd();" in content
        # AFTER, not Before: the dwell must be the clip's TAIL, because the
        # compositor shows each clip via clip_anchor="end".
        assert "After(async ({ page }) => {" in content
        assert "Before(" not in content
        assert "await heroDwell(page, grant);" in content

    def test_hook_tops_up_rather_than_dwelling_unconditionally(self):
        """The idempotency mechanism, read off the emitted hook.

        A hook that just called `heroDwell(page)` would double the dwell of any
        scenario whose steps already called it — ~44s clips, i.e. a new variance
        in place of the old one.
        """
        defs = _parse_step_defs(_NAIVE_RECORDING_STEPS)
        _ensure_recording_pacing(defs)
        content = defs[-1]["content"]
        assert "const owed = dwellRemainingMs(page);" in content
        assert "const grant = Math.min(owed, DWELL_CEILING_MS);" in content
        assert "if (grant > 0)" in content

    def test_a_clamped_dwell_is_loud(self):
        """A clip shorter than the duration this package PUBLISHED is a measurement
        gap, and a silent short clip is indistinguishable from a healthy one —
        which is the entire defect. It must announce itself."""
        defs = _parse_step_defs(_NAIVE_RECORDING_STEPS)
        _ensure_recording_pacing(defs)
        content = defs[-1]["content"]
        assert "if (grant < owed) {" in content
        assert "console.warn(`##SOFT_FAIL## ${detail}`);" in content

    def test_bounded_by_the_per_test_timeout(self):
        """Same budget coupling as the nav Before hook: a scenario hook runs INSIDE
        the test, so a dwell longer than the Playwright --timeout has the test
        killed mid-hook."""
        defs = _parse_step_defs(_NAIVE_RECORDING_STEPS)
        _ensure_recording_pacing(defs)
        content = defs[-1]["content"]
        assert "Number(process.env.PW_TEST_TIMEOUT_MS) || 20000" in content
        assert "const DWELL_CEILING_MS = Math.max(0, PW_TEST_TIMEOUT_MS - 2000);" in content

    def test_idempotent_at_the_pass_level(self):
        defs = _parse_step_defs(_NAIVE_RECORDING_STEPS)
        assert _ensure_recording_pacing(defs) == 1
        assert _ensure_recording_pacing(defs) == 0
        assert sum(1 for sd in defs if sd["filename"] == "steps_pacing.ts") == 1

    def test_pipeline_injects_exactly_one_pacing_hook(self):
        out = _apply_compile_guarantees(
            _parse_step_defs(_NAIVE_RECORDING_STEPS), [], recording=True
        )
        assert sum(1 for sd in out if sd["filename"] == "steps_pacing.ts") == 1


class TestExportedDurationHasOneOwner:
    """The number must live in ONE place. `w2v_orchestrator` sizes its scenario
    count off it, so a second hardcoded copy is a cross-REPO drift waiting to
    happen — which is how its `_SECONDS_PER_SCENARIO = 4.0` came to sit 5.5x below
    what a recording actually yields."""

    def test_python_reads_the_value_from_the_typescript_source(self):
        raw = _HELPERS_TS.read_text(encoding="utf-8")
        declared = re.search(
            r"^export\s+const\s+HERO_DWELL_MS\s*=\s*([\d_]+)\s*;", raw, re.MULTILINE
        )
        assert declared, "recording_helpers.ts no longer declares HERO_DWELL_MS at line start."
        assert int(declared.group(1).replace("_", "")) == HERO_DWELL_MS, (
            "step_def_generator.HERO_DWELL_MS disagrees with the .ts declaration "
            "it is supposed to be parsed from."
        )
        # ...and the number is not RESTATED as a Python literal anywhere in the
        # generator. AST, not a substring scan: the module explains the 22000ms
        # budget in several comments and emits it inside generated TypeScript
        # comment strings, and a text scan would flag that prose — which is how a
        # gate ends up weakened to make its own false positive go away. An integer
        # `ast.Constant` is the thing that would actually cause drift; a comment
        # cannot, and a string is not an int node.
        py_src = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "services"
            / "generation"
            / "step_def_generator.py"
        ).read_text(encoding="utf-8")
        literals = [
            n.lineno
            for n in ast.walk(ast.parse(py_src))
            if isinstance(n, ast.Constant)
            and n.value is not True
            and n.value is not False
            and isinstance(n.value, int)
            and n.value == HERO_DWELL_MS
        ]
        assert not literals, (
            f"step_def_generator.py restates the dwell length {HERO_DWELL_MS} as a "
            f"Python integer literal at line(s) {literals}. It must be PARSED from "
            "recording_helpers.ts (`_read_hero_dwell_ms`) so the .ts file stays the "
            "single owner: this number is published to w2v_orchestrator, and a "
            "second copy is a cross-repo drift waiting to happen."
        )

    def test_a_missing_declaration_raises_rather_than_defaulting(self, tmp_path, monkeypatch):
        """Fail-open here would publish a plausible-but-wrong duration to the
        orchestrator — the fail-open-on-a-measurement-gap shape this change exists
        to remove."""
        import src.services.generation.step_def_generator as sdg

        broken = tmp_path / "recording_helpers.ts"
        broken.write_text("export const SOMETHING_ELSE = 1;\n", encoding="utf-8")
        monkeypatch.setattr(sdg, "_RECORDING_HELPERS_TS", broken)
        with pytest.raises(sdg.StepDefGenerationError, match="HERO_DWELL_MS"):
            _read_hero_dwell_ms()

    def test_exported_seconds_matches_the_dwell(self):
        assert GUARANTEED_RECORDING_SECONDS_PER_SCENARIO == HERO_DWELL_MS / 1000.0
        # A sanity band, not a restatement: a value this far off would mean the
        # parse picked up an unrelated number.
        assert 5.0 <= GUARANTEED_RECORDING_SECONDS_PER_SCENARIO <= 120.0
