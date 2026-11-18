"""A STUBBED scenario must be distinguishable from a real one.

``_ensure_step_coverage`` appends an auto-stub — ``waitForLoadState('networkidle')``
+ ``stubDwell`` — for every uncovered Gherkin step. That stub PASSES and performs
no action, so a heavily stubbed scenario films a page that never changed for the
whole dwell. The generator's own comment said "nothing downstream can distinguish
a stubbed scenario from a real one": the share existed only in a log WARNING.

These tests pin the structured carrier (``StepCoverageReport``), its threshold,
and — the false-positive guard that protects every currently-working run — that a
fully covered suite is NOT marked degraded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.generation.step_def_generator import (
    STUB_DEGRADED_RATIO,
    STUB_SAMPLE_LIMIT,
    StepCoverageReport,
    StepDefGenerator,
    _apply_compile_guarantees,
)
from src.services.job_processor import JobProcessor
from src.services.package_builder import PackageBuilder
from src.services.storage_service import StorageService

_HEADER = (
    "import { createBdd } from 'playwright-bdd';\nconst { Given, When, Then } = createBdd();\n"
)


def _feature(*steps: str) -> list[dict[str, Any]]:
    body = "".join(f"    {s}\n" for s in steps)
    return [{"filename": "f.feature", "content": f"Feature: F\n  Scenario: S\n{body}"}]


def _defs(*patterns: str) -> list[dict[str, Any]]:
    calls = "".join(f"When('{p}', async ({{ page }}) => {{}});\n" for p in patterns)
    return [{"filename": "s.ts", "content": _HEADER + calls, "step_count": len(patterns)}]


def _run(steps: list[str], covered: list[str]) -> StepCoverageReport:
    report = StepCoverageReport()
    _apply_compile_guarantees(
        _defs(*covered),
        _feature(*(f"When {s}" for s in steps)),
        recording=True,
        coverage=report,
    )
    return report


class TestStubCountIsSurfaced:
    def test_partially_covered_suite_reports_count_and_ratio(self) -> None:
        report = _run(
            ["I open the list", "I add an item", "I rename it", "I archive it"],
            ["I open the list"],
        )
        assert report.steps_total == 4
        assert report.steps_stubbed == 3
        assert report.stub_ratio == pytest.approx(0.75)
        assert set(report.stubbed_steps) == {"I add an item", "I rename it", "I archive it"}

    def test_stubbed_step_sample_is_bounded(self) -> None:
        steps = [f"I do thing {i}" for i in range(STUB_SAMPLE_LIMIT + 5)]
        report = _run(steps, [])
        assert report.steps_stubbed == len(steps)
        assert len(report.stubbed_steps) == STUB_SAMPLE_LIMIT

    def test_report_is_json_safe_for_the_job_row_and_the_package(self) -> None:
        payload = json.loads(json.dumps(_run(["a", "b"], ["a"]).as_dict()))
        assert payload["steps_total"] == 2
        assert payload["steps_stubbed"] == 1
        assert payload["stub_ratio"] == pytest.approx(0.5)
        assert payload["degraded"] is True
        assert payload["degraded_threshold"] == STUB_DEGRADED_RATIO


class TestFalsePositiveGuard:
    """⛔ The guard that protects every currently-working run: a suite whose steps
    are all covered must report ZERO stubs and must NOT be marked degraded."""

    def test_fully_covered_suite_reports_zero_and_is_not_degraded(self) -> None:
        report = _run(["I open the list", "I add an item"], ["I open the list", "I add an item"])
        assert report.steps_stubbed == 0
        assert report.stubbed_steps == []
        assert report.stub_ratio == 0.0
        assert report.degraded is False

    def test_empty_suite_is_not_degraded(self) -> None:
        """No steps = no footage claim. Reporting a full stub-out here would fire
        the threshold on every stream that generates no Gherkin steps at all."""
        report = StepCoverageReport()
        assert report.stub_ratio == 0.0
        assert report.degraded is False


class TestDegradedThreshold:
    """The threshold fires EXACTLY at its boundary: `>= STUB_DEGRADED_RATIO`."""

    def test_just_below_the_threshold_is_not_degraded(self) -> None:
        # 2 of 5 stubbed = 0.4 — a normal suite with a couple of missed steps.
        report = _run(["a", "b", "c", "d", "e"], ["a", "b", "c"])
        assert report.steps_stubbed == 2
        assert report.stub_ratio == pytest.approx(0.4)
        assert report.degraded is False

    def test_exactly_at_the_threshold_is_degraded(self) -> None:
        # 2 of 4 stubbed = 0.5 — the boundary itself.
        report = _run(["a", "b", "c", "d"], ["a", "b"])
        assert report.stub_ratio == pytest.approx(STUB_DEGRADED_RATIO)
        assert report.degraded is True

    def test_above_the_threshold_is_degraded(self) -> None:
        report = _run(["a", "b", "c", "d"], ["a"])
        assert report.stub_ratio == pytest.approx(0.75)
        assert report.degraded is True

    def test_degradation_is_a_flag_not_a_failure(self) -> None:
        """A fully stubbed suite still RETURNS a package — marking it degraded must
        not raise, or every currently-shipping heavily-stubbed run turns red."""
        report = StepCoverageReport()
        out = _apply_compile_guarantees(
            _defs(),
            _feature("When a", "When b"),
            recording=True,
            coverage=report,
        )
        assert report.degraded is True
        assert any(sd["filename"] == "steps_autostub.ts" for sd in out)


class TestNonRecordingStreamsAreUnmeasured:
    def test_qa_stream_does_not_stub_and_reports_nothing(self) -> None:
        """QA/accessibility run `missingSteps:'fail-on-gen'` — no stub pass runs
        there, so the report must stay at zero rather than invent a share."""
        report = StepCoverageReport()
        _apply_compile_guarantees(
            _defs("a"),
            _feature("When a", "When b"),
            recording=False,
            coverage=report,
        )
        assert report.steps_total == 0
        assert report.steps_stubbed == 0
        assert report.degraded is False


# ── the CALLER-visible surfaces ───────────────────────────────────────────────
# Measuring the stub share is worthless if it stops at the generator. Two carriers
# leave this service: the job row (returned verbatim by GET /api/bdd/{id}/result,
# which is what w2v_orchestrator's demo_video_record path reads) and
# `step_coverage.json` inside the downloadable package.

MODULE = "src.services.job_processor"


def _demo_job() -> MagicMock:
    job = MagicMock()
    job.job_id = "stubvis-1"
    job.status = "pending"
    job.stream = "demo_video"
    job.tenant_id = "n8n-dev"
    job.input_type = "web_url"
    job.target_url = "https://example.com"
    job.app_path = None
    job.config = {}
    job.narration_config = {}
    job.inspection_data = None
    return job


async def _run_demo_job(stubbed: int, total: int) -> tuple[dict[str, Any], list[Any]]:
    """Drive `process_job` for the demo_video stream with a generator that reports
    `stubbed` of `total` steps auto-stubbed. Returns (mark_completed kwargs,
    store_text calls)."""

    async def _fake_generate(*_a: Any, coverage: StepCoverageReport | None = None, **_k: Any):
        if coverage is not None:
            coverage.steps_total = total
            coverage.steps_stubbed = stubbed
            coverage.stubbed_steps = [f"step {i}" for i in range(stubbed)]
        return [{"filename": "s.ts", "content": "// steps", "step_count": 1}]

    features = [{"filename": "f.feature", "content": "Feature: F\n  Scenario: S\n    Given a\n"}]
    inspection = {"pages": [], "navigation_graph": [], "platform": "web", "base_url": "https://x"}

    with (
        patch(f"{MODULE}.LLMClient", return_value=MagicMock(generate=AsyncMock(return_value=""))),
        patch(f"{MODULE}.JobManager.get_job", new_callable=AsyncMock, return_value=_demo_job()),
        patch(f"{MODULE}.JobManager.mark_processing", new_callable=AsyncMock),
        patch(f"{MODULE}.JobManager.mark_completed", new_callable=AsyncMock) as completed,
        patch(f"{MODULE}.JobManager.record_usage", new_callable=AsyncMock),
        patch.object(
            JobProcessor, "_resolve_input", new_callable=AsyncMock, return_value=inspection
        ),
        patch.object(
            JobProcessor, "_generate_features", new_callable=AsyncMock, return_value=features
        ),
        patch.object(StepDefGenerator, "generate", new=_fake_generate),
        patch.object(
            StorageService, "store_text", new_callable=AsyncMock, return_value="/tmp/a"
        ) as store_text,
        patch.object(
            PackageBuilder, "build_package", new_callable=AsyncMock, return_value="/tmp/p.zip"
        ),
    ):
        await JobProcessor().process_job(AsyncMock(), "stubvis-1", "n8n-dev")

    return completed.call_args[1], store_text.call_args_list


class TestCallerVisibleSurfaces:
    async def test_job_result_carries_the_stub_share(self) -> None:
        kwargs, _ = await _run_demo_job(stubbed=3, total=4)
        coverage = kwargs["step_coverage"]
        assert coverage["steps_total"] == 4
        assert coverage["steps_stubbed"] == 3
        assert coverage["stub_ratio"] == pytest.approx(0.75)
        assert coverage["degraded"] is True
        assert coverage["stubbed_steps"][0] == "step 0"

    async def test_a_clean_demo_job_is_not_marked_degraded(self) -> None:
        """False-positive guard at the CALLER boundary, not just in the dataclass."""
        kwargs, _ = await _run_demo_job(stubbed=0, total=6)
        assert kwargs["step_coverage"]["steps_stubbed"] == 0
        assert kwargs["step_coverage"]["degraded"] is False

    async def test_package_carries_a_step_coverage_manifest(self) -> None:
        _, calls = await _run_demo_job(stubbed=3, total=4)
        written = {c.args[2]: c.args[1] for c in calls if len(c.args) >= 3}
        assert "step_coverage.json" in written, sorted(written)
        payload = json.loads(written["step_coverage.json"])
        assert payload["steps_stubbed"] == 3
        assert payload["degraded"] is True


# ── assertNavigated: a 404 page used to PASS ──────────────────────────────────
# `assertNavigated` asserted only "url is http(s)" + "body is visible". An error
# page satisfies both, so a 404 landing recorded as a successful "demo". The
# hardening asserts the DELIVERED document status via PerformanceNavigationTiming
# (which folds a redirect chain into one entry and reports the FINAL status, so a
# legitimate 30x -> 200 redirect is untouched) and reports it through the existing
# soft-fail channel — the runner demotes a soft-failed scenario from passed —
# rather than throwing, which would abort the scenario and blank the clip.

_HELPERS_TS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "templates"
    / "scaffold"
    / "lib"
    / "recording_helpers.ts"
)


def _assert_navigated_body() -> str:
    src = _HELPERS_TS.read_text()
    start = src.index("export async function assertNavigated(")
    end = src.index("\n}\n", start)
    return src[start:end]


class TestAssertNavigatedRejectsAnErrorPage:
    def test_it_reads_the_delivered_document_status(self) -> None:
        body = _assert_navigated_body()
        assert "responseStatus" in body, "a 404 page still passes assertNavigated"
        # The BOUNDARY, not a substring: `status >= 40000` contains "status >= 400".
        assert re.search(r"\bstatus >= 400\)", body), body

    def test_a_bad_status_is_soft_failed_not_thrown(self) -> None:
        """Throwing here would abort the scenario and blank the rest of the clip —
        the exact failure `softExpectVisible` exists to avoid."""
        body = _assert_navigated_body()
        after = body[body.index("status >= 400") :]
        assert "softFail(" in after
        assert "throw" not in after, "a bad status must not abort the recording"

    def test_an_unmeasurable_navigation_changes_nothing(self) -> None:
        """A client-side SPA route (or a browser without `responseStatus`) yields no
        measurement; it must read as 0, never as a failure."""
        body = _assert_navigated_body()
        assert ": 0;" in body and ".catch(() => 0)" in body
