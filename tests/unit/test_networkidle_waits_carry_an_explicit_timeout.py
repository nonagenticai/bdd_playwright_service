"""Every ``waitForLoadState('networkidle')`` settle call must carry an explicit
short timeout.

``networkidle`` is a NAVIGATION-class wait, so with no explicit ``timeout`` it
falls back to Playwright's `navigationTimeout` (15_000ms in
`playwright.config.ts`), not the shorter `actionTimeout` — and every call site
is already `.catch(() => {})`-wrapped (best-effort settle), so a stall pays the
full 15s silently before execution continues. That 15s lands directly inside
whatever it is called from — for the auto-stub emitted by
`_ensure_step_coverage`, that is the RECORDED CLIP itself, whose measured
length feeds the orchestrator's duration solver as a hard minimum. One extra
LLM-drawn uncovered step (varying run to run, since the Gherkin has no fixed
step count) pays this tax once and can blow a beat's budget by ~15s — the
confirmed mechanism behind a `showcase` clip recording 8.295s -> 24.76s.

⚠️ A test that only greps for the STRING `timeout:` somewhere in the file
proves nothing — it cannot see that the key is ABSENT from a specific call
site (a prior capture test in this repo did exactly that and printed PASS
against a deliberately broken config). So every assertion below anchors on the
`waitForLoadState('networkidle'` call itself and requires the very next thing
to be a `timeout:` option, not just a `timeout:` anywhere nearby.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.services.generation.step_def_generator import (
    STUB_NETWORK_IDLE_TIMEOUT_MS,
    StepCoverageReport,
    _apply_compile_guarantees,
)

_SCAFFOLD_LIB = Path(__file__).resolve().parents[2] / "src" / "templates" / "scaffold" / "lib"
_RECORDING_HELPERS_TS = _SCAFFOLD_LIB / "recording_helpers.ts"
_SHOWCASE_STEPS_TS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "templates"
    / "scaffold"
    / "steps"
    / "showcase.steps.ts"
)

# Matches `waitForLoadState('networkidle'` followed, before the closing paren,
# by an explicit numeric `timeout:` option — i.e. NOT the bare 1-arg call.
_NETWORKIDLE_WITH_TIMEOUT = re.compile(
    r"waitForLoadState\(\s*'networkidle'\s*,\s*\{\s*timeout:\s*[A-Za-z0-9_.]+\s*\}\s*\)"
)
# Any call to networkidle at all (with or without a timeout) — used to prove the
# positive-match regex above isn't simply failing to find its target.
_ANY_NETWORKIDLE_CALL = re.compile(r"waitForLoadState\(\s*'networkidle'")


def _feature(*steps: str) -> list[dict[str, object]]:
    body = "".join(f"    When {s}\n" for s in steps)
    return [{"filename": "f.feature", "content": f"Feature: F\n  Scenario: S\n{body}"}]


class TestAutoStubCarriesAnExplicitTimeout:
    def _autostub_content(self) -> str:
        step_defs = _apply_compile_guarantees(
            [],
            _feature("I do a thing nothing covers"),
            recording=True,
            coverage=StepCoverageReport(),
        )
        stub = next(d for d in step_defs if d["filename"] == "steps_autostub.ts")
        return stub["content"]

    def test_the_emitted_stub_bounds_networkidle_with_a_short_explicit_timeout(self) -> None:
        content = self._autostub_content()
        assert _ANY_NETWORKIDLE_CALL.search(content), "fixture drifted: no networkidle call at all"
        match = _NETWORKIDLE_WITH_TIMEOUT.search(content)
        assert match, (
            "the auto-stub's waitForLoadState('networkidle') has no explicit "
            f"timeout — it will silently fall back to the 15s navigationTimeout. "
            f"Content:\n{content}"
        )

    def test_the_bound_is_the_named_constant_and_is_short(self) -> None:
        content = self._autostub_content()
        assert f"timeout: {STUB_NETWORK_IDLE_TIMEOUT_MS}" in content
        # Short enough that it cannot itself reintroduce a multi-second-per-step
        # tax, but long enough to be a real settle attempt, not a no-op.
        assert 1000 <= STUB_NETWORK_IDLE_TIMEOUT_MS <= 5000

    def test_vacuity_guard_removing_the_timeout_arg_goes_red(self) -> None:
        """Prove the regex above can actually fail, not just always pass.

        Mutates the emitted content in-memory (not the source file) to the
        pre-fix shape — the bare, no-timeout call — and asserts the same
        positive-match assertion used above now fails against it.
        """
        content = self._autostub_content()
        assert _NETWORKIDLE_WITH_TIMEOUT.search(content), "sanity: fix must be present first"
        mutated = re.sub(
            r"waitForLoadState\(\s*'networkidle'\s*,\s*\{\s*timeout:\s*[A-Za-z0-9_.]+\s*\}\s*\)",
            "waitForLoadState('networkidle')",
            content,
        )
        assert mutated != content, "mutation did not change anything — test is vacuous"
        assert not _NETWORKIDLE_WITH_TIMEOUT.search(mutated), (
            "the positive-match assertion still passes after removing the "
            "timeout argument — this test cannot detect the regression it "
            "claims to guard against"
        )


class TestRecordingHelpersAndShowcaseStepsBoundEveryNetworkidleWait:
    def test_recording_helpers_ts_bounds_every_networkidle_call(self) -> None:
        # Only actual call SITES (an `await ...` statement) — the doc comment
        # above NETWORK_IDLE_TIMEOUT_MS quotes the pre-fix bare call in prose,
        # which must not count as an unbound site.
        lines = [
            line
            for line in _RECORDING_HELPERS_TS.read_text().splitlines()
            if _ANY_NETWORKIDLE_CALL.search(line) and "await" in line
        ]
        content = "\n".join(lines)
        all_calls = _ANY_NETWORKIDLE_CALL.findall(content)
        bound_calls = _NETWORKIDLE_WITH_TIMEOUT.findall(content)
        assert len(all_calls) >= 2, "fixture drifted: expected at least 2 networkidle call sites"
        assert len(bound_calls) == len(all_calls), (
            f"{len(all_calls) - len(bound_calls)} of {len(all_calls)} "
            "networkidle call(s) in recording_helpers.ts have no explicit timeout"
        )

    def test_showcase_steps_ts_bounds_its_networkidle_call(self) -> None:
        content = _SHOWCASE_STEPS_TS.read_text()
        assert _ANY_NETWORKIDLE_CALL.search(content), "fixture drifted: no networkidle call at all"
        all_calls = _ANY_NETWORKIDLE_CALL.findall(content)
        bound_calls = _NETWORKIDLE_WITH_TIMEOUT.findall(content)
        assert len(bound_calls) == len(all_calls)
