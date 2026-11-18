"""Tests for JobProcessor pipeline orchestration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from w2v_common.tenant import UnvouchedTenantError

from src.services.generation.feature_generator import FeatureGenerator
from src.services.generation.step_def_generator import StepDefGenerator
from src.services.input.gherkin_parser import GherkinParser
from src.services.input.inspection_adapter import InspectionAdapter
from src.services.job_processor import JobProcessor, _interior_urls
from src.services.package_builder import PackageBuilder
from src.services.storage_service import StorageService


def _unvouched_tenant_error() -> UnvouchedTenantError:
    """The REAL exception the outbound identity hook raises, not a stand-in.

    Constructing the genuine class is the point: the whole finding is about its
    MRO (UnvouchedTenantError -> TenantError -> ValueError), so a hand-rolled
    double would test nothing. This assertion pins that MRO — if w2v_common ever
    re-parents it, this test says so instead of silently going vacuous.
    """
    exc = UnvouchedTenantError(
        "outbound POST http://worker-management:6301/workers/x/message: refusing to "
        "delegate tenant t — provenance is 'unrecorded', which is not vouched."
    )
    assert isinstance(exc, ValueError)
    assert not isinstance(exc, OSError | RuntimeError)
    return exc


def _make_mock_job(
    job_id: str = "proc-job-1",
    status: str = "pending",
    stream: str = "qa_suite",
    **kwargs: object,
) -> MagicMock:
    """Create a mock BDDJob object for processor tests."""
    job = MagicMock()
    job.job_id = job_id
    job.status = status
    job.stream = stream
    job.tenant_id = kwargs.get("tenant_id", "n8n-dev")
    job.input_type = kwargs.get("input_type", "web_url")
    job.target_url = kwargs.get("target_url", "https://example.com")
    job.app_path = kwargs.get("app_path")
    job.config = kwargs.get("config", {})
    job.narration_config = kwargs.get("narration_config", {})
    job.inspection_data = None
    return job


MODULE = "src.services.job_processor"


@pytest.fixture
def mock_llm():
    """Mock LLMClient (wraps WMClient) to avoid real outbound calls."""
    mock = MagicMock()
    mock.generate = AsyncMock(return_value="mocked LLM response")
    return mock


@pytest.fixture
def _patch_llm(mock_llm: MagicMock):
    """Patch the LLMClient class so JobProcessor builds mocked instances."""
    with patch(f"{MODULE}.LLMClient", return_value=mock_llm):
        yield mock_llm


@pytest.fixture
def mock_features():
    """Sample feature output from the feature generator."""
    return [
        {
            "filename": "login.feature",
            "content": "Feature: Login\n  Scenario: Valid login\n    Given I am on the login page",
        },
        {
            "filename": "dashboard.feature",
            "content": "Feature: Dashboard\n  Scenario: View dashboard\n    Given I am logged in",
        },
    ]


@pytest.fixture
def mock_step_defs():
    """Sample step definition output."""
    return [
        {"filename": "login.steps.ts", "content": "// step defs for login"},
        {"filename": "dashboard.steps.ts", "content": "// step defs for dashboard"},
    ]


@pytest.fixture
def mock_inspection():
    """Sample inspection data returned from web crawl."""
    return {
        "pages": [{"page_id": "p0", "url": "https://example.com", "title": "Home"}],
        "navigation_graph": [],
        "platform": "web",
        "base_url": "https://example.com",
    }


class TestProcessJobHappyPath:
    @pytest.mark.usefixtures("_patch_llm")
    async def test_process_job_qa_suite_basic(
        self,
        mock_db_session: AsyncMock,
        mock_features: list[dict[str, object]],
        mock_step_defs: list[dict[str, object]],
        mock_inspection: dict[str, object],
    ):
        """Happy path: qa_suite stream processes end-to-end."""
        job = _make_mock_job(stream="qa_suite")

        with (
            patch(
                f"{MODULE}.JobManager.get_job",
                new_callable=AsyncMock,
                return_value=job,
            ),
            patch(f"{MODULE}.JobManager.mark_processing", new_callable=AsyncMock),
            patch(f"{MODULE}.JobManager.mark_completed", new_callable=AsyncMock) as mock_completed,
            patch(f"{MODULE}.JobManager.record_usage", new_callable=AsyncMock),
            patch.object(
                FeatureGenerator,
                "generate_qa_features",
                new_callable=AsyncMock,
                return_value=mock_features,
            ),
            patch.object(
                StepDefGenerator,
                "generate",
                new_callable=AsyncMock,
                return_value=mock_step_defs,
            ),
            patch.object(
                StorageService,
                "store_text",
                new_callable=AsyncMock,
                return_value="/tmp/test/artifact.txt",
            ),
            patch.object(
                PackageBuilder,
                "build_package",
                new_callable=AsyncMock,
                return_value="/tmp/test/package.zip",
            ),
            patch.object(
                JobProcessor,
                "_resolve_input",
                new_callable=AsyncMock,
                return_value=mock_inspection,
            ),
        ):
            processor = JobProcessor()

            await processor.process_job(mock_db_session, "proc-job-1", "n8n-dev")

            mock_completed.assert_called_once()
            call_kwargs = mock_completed.call_args[1]
            assert call_kwargs["features_generated"] == 2
            assert call_kwargs["step_defs_generated"] == 2

    @pytest.mark.usefixtures("_patch_llm")
    async def test_usage_record_db_error_is_best_effort(
        self,
        mock_db_session: AsyncMock,
        mock_features: list[dict[str, object]],
        mock_step_defs: list[dict[str, object]],
        mock_inspection: dict[str, object],
    ):
        """A DB commit failure while recording usage must NOT escape process_job.

        JobManager.record_usage / db.commit() can raise
        ``sqlalchemy.exc.OperationalError`` (MRO: OperationalError -> ... ->
        SQLAlchemyError -> Exception) which is NOT OSError/RuntimeError. On
        current code the best-effort handler only catches (OSError,
        RuntimeError), so this escapes process_job -- despite the job already
        being mark_completed -- and the BDDUsage row is silently dropped.
        The completed job's terminal state must be unaffected.
        """
        from sqlalchemy.exc import OperationalError

        job = _make_mock_job(stream="qa_suite")

        with (
            patch(
                f"{MODULE}.JobManager.get_job",
                new_callable=AsyncMock,
                return_value=job,
            ),
            patch(f"{MODULE}.JobManager.mark_processing", new_callable=AsyncMock),
            patch(f"{MODULE}.JobManager.mark_completed", new_callable=AsyncMock) as mock_completed,
            patch(
                f"{MODULE}.JobManager.record_usage",
                new_callable=AsyncMock,
                side_effect=OperationalError("INSERT INTO bdd_usage", {}, Exception("boom")),
            ),
            patch.object(
                FeatureGenerator,
                "generate_qa_features",
                new_callable=AsyncMock,
                return_value=mock_features,
            ),
            patch.object(
                StepDefGenerator,
                "generate",
                new_callable=AsyncMock,
                return_value=mock_step_defs,
            ),
            patch.object(
                StorageService,
                "store_text",
                new_callable=AsyncMock,
                return_value="/tmp/test/artifact.txt",
            ),
            patch.object(
                PackageBuilder,
                "build_package",
                new_callable=AsyncMock,
                return_value="/tmp/test/package.zip",
            ),
            patch.object(
                JobProcessor,
                "_resolve_input",
                new_callable=AsyncMock,
                return_value=mock_inspection,
            ),
        ):
            processor = JobProcessor()

            # Must NOT raise: the usage-record failure is best-effort.
            await processor.process_job(mock_db_session, "proc-job-1", "n8n-dev")

            # Terminal state of the completed job is unaffected.
            mock_completed.assert_called_once()

    @pytest.mark.usefixtures("_patch_llm")
    async def test_process_job_marks_processing(
        self,
        mock_db_session: AsyncMock,
        mock_inspection: dict[str, object],
    ):
        """Verify mark_processing is called before generation."""
        job = _make_mock_job()

        with (
            patch(
                f"{MODULE}.JobManager.get_job",
                new_callable=AsyncMock,
                return_value=job,
            ),
            patch(f"{MODULE}.JobManager.mark_processing", new_callable=AsyncMock) as mock_mark_proc,
            patch(f"{MODULE}.JobManager.mark_completed", new_callable=AsyncMock),
            patch(f"{MODULE}.JobManager.record_usage", new_callable=AsyncMock),
            patch.object(
                JobProcessor,
                "_resolve_input",
                new_callable=AsyncMock,
                return_value=mock_inspection,
            ),
            patch.object(
                FeatureGenerator,
                "generate_qa_features",
                new_callable=AsyncMock,
                # Non-empty: zero features is now a hard failure (a package with no
                # .feature file records no clip), so [] would fail the pipeline here.
                return_value=[{"filename": "a.feature", "content": "Feature: A", "tags": []}],
            ),
            patch.object(
                StepDefGenerator,
                "generate",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                StorageService,
                "store_text",
                new_callable=AsyncMock,
                return_value="/tmp/x",
            ),
            patch.object(
                PackageBuilder,
                "build_package",
                new_callable=AsyncMock,
                return_value="/tmp/pkg.zip",
            ),
        ):
            processor = JobProcessor()

            await processor.process_job(mock_db_session, "proc-job-1", "n8n-dev")

            mock_mark_proc.assert_called_once_with(mock_db_session, "proc-job-1")

    @pytest.mark.usefixtures("_patch_llm")
    async def test_process_job_marks_completed(
        self,
        mock_db_session: AsyncMock,
        mock_features: list[dict[str, object]],
        mock_step_defs: list[dict[str, object]],
        mock_inspection: dict[str, object],
    ):
        """Verify mark_completed is called with correct result counts."""
        job = _make_mock_job()

        with (
            patch(
                f"{MODULE}.JobManager.get_job",
                new_callable=AsyncMock,
                return_value=job,
            ),
            patch(f"{MODULE}.JobManager.mark_processing", new_callable=AsyncMock),
            patch(f"{MODULE}.JobManager.mark_completed", new_callable=AsyncMock) as mock_completed,
            patch(f"{MODULE}.JobManager.record_usage", new_callable=AsyncMock),
            patch.object(
                JobProcessor,
                "_resolve_input",
                new_callable=AsyncMock,
                return_value=mock_inspection,
            ),
            patch.object(
                FeatureGenerator,
                "generate_qa_features",
                new_callable=AsyncMock,
                return_value=mock_features,
            ),
            patch.object(
                StepDefGenerator,
                "generate",
                new_callable=AsyncMock,
                return_value=mock_step_defs,
            ),
            patch.object(
                StorageService,
                "store_text",
                new_callable=AsyncMock,
                return_value="/tmp/x",
            ),
            patch.object(
                PackageBuilder,
                "build_package",
                new_callable=AsyncMock,
                return_value="/tmp/pkg.zip",
            ),
        ):
            processor = JobProcessor()

            await processor.process_job(mock_db_session, "proc-job-1", "n8n-dev")

            mock_completed.assert_called_once()
            kw = mock_completed.call_args[1]
            assert kw["features_generated"] == 2
            assert kw["step_defs_generated"] == 2
            assert "package_zip" in kw["artifact_paths"]


class TestProcessJobErrors:
    @pytest.mark.usefixtures("_patch_llm")
    async def test_process_job_not_found(self, mock_db_session: AsyncMock):
        """Job not found raises RuntimeError."""
        with patch(
            f"{MODULE}.JobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            processor = JobProcessor()

            with pytest.raises(RuntimeError, match="not found"):
                await processor.process_job(mock_db_session, "missing-id", "n8n-dev")

    @pytest.mark.usefixtures("_patch_llm")
    async def test_process_job_unknown_stream(
        self,
        mock_db_session: AsyncMock,
        mock_inspection: dict[str, object],
    ):
        """Unknown stream raises ValueError."""
        job = _make_mock_job(stream="invalid_stream")

        with (
            patch(
                f"{MODULE}.JobManager.get_job",
                new_callable=AsyncMock,
                return_value=job,
            ),
            patch(f"{MODULE}.JobManager.mark_processing", new_callable=AsyncMock),
            patch.object(
                JobProcessor,
                "_resolve_input",
                new_callable=AsyncMock,
                return_value=mock_inspection,
            ),
        ):
            processor = JobProcessor()

            with pytest.raises(ValueError, match="Unknown stream"):
                await processor.process_job(mock_db_session, "proc-job-1", "n8n-dev")


class TestProcessJobNarration:
    @pytest.mark.usefixtures("_patch_llm")
    async def test_process_job_with_narration(
        self,
        mock_db_session: AsyncMock,
        mock_features: list[dict[str, object]],
        mock_step_defs: list[dict[str, object]],
        mock_inspection: dict[str, object],
    ):
        """Narration enabled triggers narration generation."""
        job = _make_mock_job(
            narration_config={"enabled": True, "tone": "tutorial", "llm_polish": False},
        )

        narration_result = {
            "parts": [{"scenario_name": "login", "text": "narrated text", "word_count": 5}],
            "total_words": 5,
            "estimated_total_duration_seconds": 2.0,
            "tone": "tutorial",
        }

        with (
            patch(
                f"{MODULE}.JobManager.get_job",
                new_callable=AsyncMock,
                return_value=job,
            ),
            patch(f"{MODULE}.JobManager.mark_processing", new_callable=AsyncMock),
            patch(f"{MODULE}.JobManager.mark_completed", new_callable=AsyncMock) as mock_completed,
            patch(f"{MODULE}.JobManager.record_usage", new_callable=AsyncMock),
            patch.object(
                JobProcessor,
                "_resolve_input",
                new_callable=AsyncMock,
                return_value=mock_inspection,
            ),
            patch.object(
                FeatureGenerator,
                "generate_qa_features",
                new_callable=AsyncMock,
                return_value=mock_features,
            ),
            patch.object(
                StepDefGenerator,
                "generate",
                new_callable=AsyncMock,
                return_value=mock_step_defs,
            ),
            patch.object(
                StorageService,
                "store_text",
                new_callable=AsyncMock,
                return_value="/tmp/x",
            ),
            patch.object(
                PackageBuilder,
                "build_package",
                new_callable=AsyncMock,
                return_value="/tmp/pkg.zip",
            ),
            patch.object(
                JobProcessor,
                "_generate_narration",
                new_callable=AsyncMock,
                return_value=narration_result,
            ) as mock_gen_narration,
        ):
            processor = JobProcessor()

            await processor.process_job(mock_db_session, "proc-job-1", "n8n-dev")

            mock_gen_narration.assert_called_once()
            kw = mock_completed.call_args[1]
            assert kw["narration_script"] == narration_result

    @pytest.mark.usefixtures("_patch_llm")
    async def test_process_job_without_narration(
        self,
        mock_db_session: AsyncMock,
        mock_features: list[dict[str, object]],
        mock_step_defs: list[dict[str, object]],
        mock_inspection: dict[str, object],
    ):
        """Narration disabled skips narration generation."""
        job = _make_mock_job(narration_config={"enabled": False})

        with (
            patch(
                f"{MODULE}.JobManager.get_job",
                new_callable=AsyncMock,
                return_value=job,
            ),
            patch(f"{MODULE}.JobManager.mark_processing", new_callable=AsyncMock),
            patch(f"{MODULE}.JobManager.mark_completed", new_callable=AsyncMock) as mock_completed,
            patch(f"{MODULE}.JobManager.record_usage", new_callable=AsyncMock),
            patch.object(
                JobProcessor,
                "_resolve_input",
                new_callable=AsyncMock,
                return_value=mock_inspection,
            ),
            patch.object(
                FeatureGenerator,
                "generate_qa_features",
                new_callable=AsyncMock,
                return_value=mock_features,
            ),
            patch.object(
                StepDefGenerator,
                "generate",
                new_callable=AsyncMock,
                return_value=mock_step_defs,
            ),
            patch.object(
                StorageService,
                "store_text",
                new_callable=AsyncMock,
                return_value="/tmp/x",
            ),
            patch.object(
                PackageBuilder,
                "build_package",
                new_callable=AsyncMock,
                return_value="/tmp/pkg.zip",
            ),
        ):
            processor = JobProcessor()

            await processor.process_job(mock_db_session, "proc-job-1", "n8n-dev")

            kw = mock_completed.call_args[1]
            assert kw["narration_script"] is None


class TestProcessJobArtifacts:
    @pytest.mark.usefixtures("_patch_llm")
    async def test_process_job_stores_artifacts(
        self,
        mock_db_session: AsyncMock,
        mock_features: list[dict[str, object]],
        mock_step_defs: list[dict[str, object]],
        mock_inspection: dict[str, object],
    ):
        """Verify storage is called for each feature, step def, and config file."""
        job = _make_mock_job()

        with (
            patch(
                f"{MODULE}.JobManager.get_job",
                new_callable=AsyncMock,
                return_value=job,
            ),
            patch(f"{MODULE}.JobManager.mark_processing", new_callable=AsyncMock),
            patch(f"{MODULE}.JobManager.mark_completed", new_callable=AsyncMock),
            patch(f"{MODULE}.JobManager.record_usage", new_callable=AsyncMock),
            patch.object(
                JobProcessor,
                "_resolve_input",
                new_callable=AsyncMock,
                return_value=mock_inspection,
            ),
            patch.object(
                FeatureGenerator,
                "generate_qa_features",
                new_callable=AsyncMock,
                return_value=mock_features,
            ),
            patch.object(
                StepDefGenerator,
                "generate",
                new_callable=AsyncMock,
                return_value=mock_step_defs,
            ),
            patch.object(
                StorageService,
                "store_text",
                new_callable=AsyncMock,
                return_value="/tmp/stored",
            ) as mock_store,
            patch.object(
                PackageBuilder,
                "build_package",
                new_callable=AsyncMock,
                return_value="/tmp/pkg.zip",
            ),
        ):
            processor = JobProcessor()

            await processor.process_job(mock_db_session, "proc-job-1", "n8n-dev")

            # 2 features + 2 step defs + playwright_config + package_json + tsconfig = 7
            assert mock_store.call_count == 7


class TestGenerateFeatures:
    @pytest.mark.usefixtures("_patch_llm")
    async def test_generate_features_all_streams(self, mock_inspection: dict[str, object]):
        """Each stream dispatches to the correct feature generator method."""
        processor = JobProcessor()

        stream_method_map = {
            "qa_suite": "generate_qa_features",
            "accessibility": "generate_accessibility_features",
            "demo_video": "generate_demo_features",
            "documentation": "generate_documentation_features",
        }

        for stream, method_name in stream_method_map.items():
            with patch.object(
                FeatureGenerator,
                method_name,
                new_callable=AsyncMock,
                return_value=[{"filename": "test.feature", "content": "F"}],
            ) as mock_method:
                result = await processor._generate_features(stream, mock_inspection, {})

                mock_method.assert_called_once_with(mock_inspection, {})
                assert len(result) == 1


class TestGenerateNarration:
    @pytest.mark.usefixtures("_patch_llm")
    async def test_narration_is_per_scenario_keyed_by_clip_id(self):
        """One narration part per scenario, keyed by its @clip-<n> join id."""
        from src.services.generation.feature_generator import _parse_features

        features = _parse_features(
            """```gherkin
@smoke
Feature: Login
  Scenario: Successful login
    Given I am on the login page
    When I click Sign In
    Then I see the dashboard

  Scenario: Failed login
    Given I am on the login page
    When I submit bad creds
    Then I see an error
```"""
        )
        processor = JobProcessor()

        result = await processor._generate_narration(
            features,
            {"enabled": True, "tone": "tutorial", "llm_polish": False},
        )

        parts = result["parts"]
        assert [p["part_id"] for p in parts] == ["clip-1", "clip-2"]
        assert parts[0]["scenario_name"] == "Successful login"
        assert parts[1]["scenario_name"] == "Failed login"
        for part in parts:
            assert part["text"].strip()
            assert part["estimated_duration_seconds"] > 0
            assert "Scenario:" in part["original_gherkin"]

    @pytest.mark.usefixtures("_patch_llm")
    async def test_recording_parts_publish_the_guaranteed_clip_duration(self):
        """The per-scenario recording FLOOR must reach the wire, and survive polish.

        ``estimated_duration_seconds`` is how long the NARRATION takes to speak;
        ``expected_recording_seconds`` is how long the CLIP is guaranteed to last.
        Conflating them is how a 4-second narration line came to imply a 4-second
        clip — and how ``w2v_orchestrator`` ended up sizing its scenario count off
        its own hardcoded ``_SECONDS_PER_SCENARIO = 4.0``, 5.5x below what a
        recording actually yields. This field is what replaces that constant, so
        the number lives in ONE repo.
        """
        from src.services.generation.feature_generator import _parse_features
        from src.services.generation.step_def_generator import (
            GUARANTEED_RECORDING_SECONDS_PER_SCENARIO,
        )

        features = _parse_features(
            """```gherkin
Feature: Login
  Scenario: Successful login
    Given I am on the login page
    Then I see the dashboard
```"""
        )
        processor = JobProcessor()

        # The polish pass rewrites `text` and must preserve every other key — a
        # real WM shape, not a passthrough, so this proves survival not absence.
        async def _polish(parts, tone, **_kwargs):
            from src.services.narration.humanizer import PolishReport

            return (
                [{**p, "text": "polished " + p["text"]} for p in parts],
                PolishReport(mode="polish", attempts=1, applied=True),
            )

        with patch.object(processor._humanizer, "polish_with_report", _polish):
            recorded = await processor._generate_narration(
                features,
                {"enabled": True, "tone": "tutorial", "llm_polish": True},
                recording=True,
            )

        part = recorded["parts"][0]
        assert part["text"].startswith("polished ")
        assert part["expected_recording_seconds"] == GUARANTEED_RECORDING_SECONDS_PER_SCENARIO
        # It is NOT the narration duration. If these ever coincide the assertion
        # above would pass for the wrong reason.
        assert part["expected_recording_seconds"] != part["estimated_duration_seconds"]

        # The other streams ship no lib/ and no pacing hook, so claiming a
        # recording duration for them would be a fabricated number.
        qa = await processor._generate_narration(
            features,
            {"enabled": True, "tone": "tutorial", "llm_polish": False},
        )
        assert "expected_recording_seconds" not in qa["parts"][0]

    @pytest.mark.usefixtures("_patch_llm")
    async def test_polish_failure_degrades_to_template_narration(self):
        """A WM/LLM failure during the best-effort polish pass must NOT fail the job —
        it degrades to the deterministic template narration. Regression: an unguarded
        polish call would abort the whole job and discard the generated features/steps."""
        import httpx

        from src.services.generation.feature_generator import _parse_features

        features = _parse_features(
            """```gherkin
Feature: Login
  Scenario: Successful login
    Given I am on the login page
    Then I see the dashboard
```"""
        )
        processor = JobProcessor()
        # Make the polish pass raise the way WM does on a 5xx/timeout.
        polish_raises = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=httpx.Request("POST", "http://wm"), response=httpx.Response(500)
            )
        )

        with patch.object(processor._humanizer, "polish", polish_raises):
            result = await processor._generate_narration(
                features,
                {"enabled": True, "tone": "tutorial", "llm_polish": True},
            )

        # No raise; the deterministic template parts survived.
        parts = result["parts"]
        assert parts and parts[0]["scenario_name"] == "Successful login"
        assert parts[0]["text"].strip()

    @pytest.mark.usefixtures("_patch_llm")
    @pytest.mark.parametrize(
        ("label", "exc"),
        [
            # The outbound identity hook (w2v_common.auth.identity_context) raises
            # this on a DELEGATED hop to a METERED destination with unvouched
            # provenance. worker_management IS metered, so every polish call passes
            # through that gate. MRO: UnvouchedTenantError -> TenantError ->
            # ValueError — no OSError, no RuntimeError, no httpx.HTTPError.
            ("unvouched-tenant", _unvouched_tenant_error()),
            # BaseServiceClient._request ends in `return response.json()`; WM
            # answering 2xx with a non-JSON body (an ingress/proxy error page)
            # raises json.JSONDecodeError -> ValueError.
            ("non-json-body", json.JSONDecodeError("Expecting value", "<html>", 0)),
        ],
    )
    async def test_polish_valueerror_family_degrades_instead_of_failing_the_job(
        self, label: str, exc: Exception
    ):
        """The two failures this call stack ACTUALLY produces are ValueErrors.

        The handler's contract is "a WM/LLM failure here must NOT fail the whole job
        and discard the generated features/step-defs/package". Both of these escaped
        it, so `_process_job`'s caller marked the job `failed` and threw away a
        complete package — for a pass documented as a cosmetic uplift.
        """
        from src.services.generation.feature_generator import _parse_features

        features = _parse_features(
            """```gherkin
Feature: Login
  Scenario: Successful login
    Given I am on the login page
    Then I see the dashboard
```"""
        )
        processor = JobProcessor()

        with patch.object(processor._humanizer, "polish", AsyncMock(side_effect=exc)):
            result = await processor._generate_narration(
                features,
                {"enabled": True, "tone": "tutorial", "llm_polish": True},
            )

        parts = result["parts"]
        assert parts and parts[0]["scenario_name"] == "Successful login", label
        assert parts[0]["text"].strip(), label
        assert result["total_words"] > 0, label


class TestInteriorUrls:
    def test_landing_first_then_resolved_paths(self):
        urls, dropped = _interior_urls("https://app.example.com/", ["/board", "/build"])
        assert urls == [
            "https://app.example.com/",
            "https://app.example.com/board",
            "https://app.example.com/build",
        ]
        assert dropped == 0

    def test_absolute_urls_pass_through_and_dedupe(self):
        urls, _ = _interior_urls(
            "https://app.example.com/",
            ["https://app.example.com/", "https://other.example.com/x", "/x"],
        )
        # Landing dedupes against the duplicate absolute; both interiors kept.
        assert urls == [
            "https://app.example.com/",
            "https://other.example.com/x",
            "https://app.example.com/x",
        ]

    def test_caps_and_reports_dropped(self):
        pages = [f"/p{i}" for i in range(30)]
        urls, dropped = _interior_urls("https://app.example.com/", pages)
        assert len(urls) == 20  # _MAX_INSPECT_SCREENS
        assert dropped == 11  # 1 landing + 30 pages = 31 → 11 over the cap


def test_page_from_scrape_derives_structure_from_html():
    """_page_from_scrape parses the scraped html into the structured inspection
    fields (headings/forms/selectors) the demo generators consume."""
    from types import SimpleNamespace

    from src.services.job_processor import _page_from_scrape

    resp = SimpleNamespace(
        url="https://x/",
        title="T",
        html='<h1>Todos</h1><input class="new-todo" placeholder="What needs to be done?">',
        text="body",
        links=[],
        images=[],
    )
    page = _page_from_scrape(resp)
    assert page["headings"] == ["Todos"]
    assert page["has_forms"] is True
    assert page["page_type"] in {"form", "content", "listing"}
    el = next(e for e in page["interactive_elements"] if e["tag"] == "input")
    assert el["selector"] == 'getByPlaceholder("What needs to be done?")'
    # Empty html → empty structure (no regression vs prior text-only inspection).
    empty = _page_from_scrape(
        SimpleNamespace(url="u", title="", html="", text="", links=[], images=[])
    )
    assert empty["interactive_elements"] == []
    assert empty["headings"] == []


class TestResolveInput:
    @pytest.mark.usefixtures("_patch_llm")
    async def test_resolve_input_web_url(self):
        """web_url input type uses WebscraperClient to scrape the target URL."""
        from w2v_common.clients.webscraper import ScrapeResponse

        processor = JobProcessor()

        scrape_response = ScrapeResponse(
            url="https://example.com",
            title="Example",
            html="<html></html>",
            text="Example text",
            links=[],
            images=[],
        )
        mock_ws = AsyncMock()
        mock_ws.scrape_bulk = AsyncMock(return_value=[scrape_response])
        mock_ws.close = AsyncMock()

        with (
            patch("w2v_common.clients.WebscraperClient", return_value=mock_ws),
            patch("w2v_common.clients._ssl.get_ssl_context", return_value=True),
            patch.object(
                InspectionAdapter,
                "from_web_crawl",
            ) as mock_from_web_crawl,
            patch(f"{MODULE}.asdict", return_value={"pages": [], "platform": "web"}),
        ):
            mock_result = MagicMock()
            mock_from_web_crawl.return_value = mock_result

            result = await processor._resolve_input(
                input_type="web_url",
                target_url="https://example.com",
                app_path="",
                config={},
            )

            # Single page, no auth → one scrape_bulk call over a
            # one-element batch (webscraper returns html inline;
            # auth forwarded even when None so a login-gated app is inspected authed).
            mock_ws.scrape_bulk.assert_called_once_with(
                ["https://example.com"], auth_config=None, delay=2.0
            )
            mock_ws.close.assert_awaited_once()
            assert result == {"pages": [], "platform": "web"}

            # The built page dict must carry the page body under the canonical
            # "text_content" key the adapter reads (was "text" → silently empty).
            crawl_data = mock_from_web_crawl.call_args.args[0]
            page = crawl_data["pages"][0]
            assert page["text_content"] == "Example text"
            assert "text" not in page

    @pytest.mark.usefixtures("_patch_llm")
    async def test_resolve_input_web_url_forwards_auth(self):
        """auth_config is forwarded to the scrape so a login-gated app is
        inspected as an authenticated session (was dropped → login-wall scrape)."""
        from w2v_common.clients.webscraper import ScrapeResponse

        processor = JobProcessor()
        mock_ws = AsyncMock()
        mock_ws.scrape_bulk = AsyncMock(
            return_value=[ScrapeResponse(url="https://app.example.com", title="App", html="<x/>")]
        )
        mock_ws.close = AsyncMock()
        auth = {"credential_id": "cred-123"}

        with (
            patch("w2v_common.clients.WebscraperClient", return_value=mock_ws),
            patch("w2v_common.clients._ssl.get_ssl_context", return_value=True),
            patch.object(InspectionAdapter, "from_web_crawl", return_value=MagicMock()),
            patch(f"{MODULE}.asdict", return_value={"pages": [], "platform": "web"}),
        ):
            await processor._resolve_input(
                input_type="web_url",
                target_url="https://app.example.com",
                app_path="",
                config={"auth_config": auth},
            )
            mock_ws.scrape_bulk.assert_called_once_with(
                ["https://app.example.com"], auth_config=auth, delay=2.0
            )

    @pytest.mark.usefixtures("_patch_llm")
    async def test_resolve_input_web_url_multiscreen_bulk(self):
        """target_pages → authed multi-screen scrape_bulk over resolved interior
        URLs (webscraper logs in once, reuses the session), so steps hit real screens."""
        from w2v_common.clients.webscraper import ScrapeResponse

        processor = JobProcessor()
        mock_ws = AsyncMock()
        mock_ws.scrape_bulk = AsyncMock(
            return_value=[
                ScrapeResponse(url="https://app.example.com/", title="Home", html="<h/>"),
                ScrapeResponse(url="https://app.example.com/board", title="Board", html="<b/>"),
                # A failed interior page is dropped, not fatal.
                ScrapeResponse(url="https://app.example.com/x", error="timeout"),
            ]
        )
        mock_ws.close = AsyncMock()
        auth = {"credential_id": "cred-123"}
        captured: dict[str, list] = {}

        def _capture(crawl_data):  # type: ignore[no-untyped-def]
            captured["pages"] = crawl_data["pages"]
            return MagicMock()

        with (
            patch("w2v_common.clients.WebscraperClient", return_value=mock_ws),
            patch("w2v_common.clients._ssl.get_ssl_context", return_value=True),
            patch.object(InspectionAdapter, "from_web_crawl", side_effect=_capture),
            patch(f"{MODULE}.asdict", return_value={"pages": [], "platform": "web"}),
        ):
            await processor._resolve_input(
                input_type="web_url",
                target_url="https://app.example.com/",
                app_path="",
                config={
                    "auth_config": auth,
                    "target_pages": ["/board", "https://app.example.com/x"],
                },
            )
            mock_ws.scrape_bulk.assert_called_once_with(
                [
                    "https://app.example.com/",
                    "https://app.example.com/board",
                    "https://app.example.com/x",
                ],
                auth_config=auth,
                delay=2.0,
            )
            mock_ws.scrape_url.assert_not_called()
            # The two OK screens flow into inspection; the errored one is dropped.
            assert [p["url"] for p in captured["pages"]] == [
                "https://app.example.com/",
                "https://app.example.com/board",
            ]

    @pytest.mark.usefixtures("_patch_llm")
    async def test_resolve_input_web_url_passes_post_load_delay(self):
        """The inspection scrape opts into a post-load delay so client-rendered
        SPAs hydrate before we snapshot the DOM (networkidle is NOT used — it
        times out / returns empty on live SPAs with analytics)."""
        from w2v_common.clients.webscraper import ScrapeResponse

        processor = JobProcessor()
        mock_ws = AsyncMock()
        mock_ws.scrape_bulk = AsyncMock(
            return_value=[ScrapeResponse(url="https://spa.example.com", title="SPA", html="<x/>")]
        )
        mock_ws.close = AsyncMock()

        with (
            patch("w2v_common.clients.WebscraperClient", return_value=mock_ws),
            patch("w2v_common.clients._ssl.get_ssl_context", return_value=True),
            patch.object(InspectionAdapter, "from_web_crawl", return_value=MagicMock()),
            patch(f"{MODULE}.asdict", return_value={"pages": [], "platform": "web"}),
        ):
            await processor._resolve_input(
                input_type="web_url",
                target_url="https://spa.example.com",
                app_path="",
                config={},
            )

        assert mock_ws.scrape_bulk.call_args.kwargs["delay"] == 2.0

    @pytest.mark.usefixtures("_patch_llm")
    async def test_resolve_input_gherkin_upload(self, tmp_path: object):
        """gherkin_upload input type reads feature files from storage."""
        processor = JobProcessor()

        feature_text = "Feature: Test\n  Scenario: Hello\n    Given I exist"

        mock_parsed = MagicMock()
        mock_parsed.model_dump.return_value = {
            "name": "Test",
            "scenarios": [{"name": "Hello"}],
        }

        with (
            patch.object(
                StorageService,
                "retrieve_text",
                new_callable=AsyncMock,
                return_value=feature_text,
            ) as mock_retrieve,
            patch.object(
                GherkinParser,
                "parse_feature_text",
                return_value=mock_parsed,
            ),
        ):
            result = await processor._resolve_input(
                input_type="gherkin_upload",
                target_url="",
                app_path="",
                config={"feature_files": ["uploads/tenant-1/test.feature"]},
                tenant_id="01950000-0000-7000-8000-000000000001",
            )

            mock_retrieve.assert_called_once_with(
                "uploads/tenant-1/test.feature", tenant_id="01950000-0000-7000-8000-000000000001"
            )
            assert result["platform"] == "web"
            assert len(result["parsed_features"]) == 1

    @pytest.mark.usefixtures("_patch_llm")
    async def test_resolve_input_unknown_type(self):
        """Unknown input type raises ValueError."""
        processor = JobProcessor()

        with pytest.raises(ValueError, match="Unknown input type"):
            await processor._resolve_input(
                input_type="telekinesis",
                target_url="",
                app_path="",
                config={},
            )


class TestLicenseTokenForwarding:
    """Verify the request-scoped license token is threaded through to the
    downstream peer-service clients, so they authenticate via
    ``X-Tenant-Id: Bearer <token>`` instead of relying on the trusted-network
    fallback. These tests assert the real outgoing HTTP header by capturing
    requests through ``httpx.MockTransport``."""

    @pytest.mark.usefixtures("_patch_llm")
    async def test_resolve_input_mobile_app_forwards_license_token(self):
        """``_resolve_input(input_type='mobile_app', ...)`` constructs
        AppInspectorClient with the request's license token, and the
        resulting outgoing request carries ``X-Tenant-Id: Bearer <token>``."""
        import httpx

        captured_headers: list[httpx.Headers] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.append(request.headers)
            # First call: submit_job → return job_id; subsequent: poll status
            if request.url.path.endswith("/api/inspect") and request.method == "POST":
                return httpx.Response(200, json={"job_id": "insp-1"})
            return httpx.Response(
                200,
                json={
                    "job_id": "insp-1",
                    "status": "completed",
                    "result": {"pages": [], "platform": "ios"},
                },
            )

        mock_transport = httpx.MockTransport(handler)
        original_async_client_init = httpx.AsyncClient.__init__

        def patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["transport"] = mock_transport
            original_async_client_init(self, *args, **kwargs)

        processor = JobProcessor()

        with (
            patch.object(httpx.AsyncClient, "__init__", patched_init),
            patch.object(
                InspectionAdapter,
                "from_mobile_inspection",
                return_value=MagicMock(),
            ),
            patch("src.services.job_processor.asdict", return_value={"platform": "ios"}),
        ):
            result = await processor._resolve_input(
                input_type="mobile_app",
                target_url="",
                app_path="/path/to/app",
                config={},
                tenant_id="n8n-dev",
                license_token="tok-mobile-xyz",  # pragma: allowlist secret
            )

        assert result == {"platform": "ios"}
        assert captured_headers, "expected at least one request to app_inspector"
        for hdrs in captured_headers:
            assert hdrs.get("x-tenant-id") == "Bearer tok-mobile-xyz", (  # pragma: allowlist secret
                f"expected X-Tenant-Id 'Bearer tok-mobile-xyz', got {hdrs.get('x-tenant-id')!r}"
            )


class TestZeroFeaturesIsNotSuccess:
    """Bughunt 2026-08-21 #1b: a generation that yields ZERO features must not
    reach `mark_completed(features_generated=0)`. An empty feature set means the
    package has no `.feature` file at all — bddgen runs nothing, no clip is
    recorded, and the job reports success for an empty deliverable. Every
    `_generate_features` branch is a generator that is *supposed* to produce
    features (the knowledge-pack branch is only taken when `screens` is
    non-empty), so there is no legitimate zero-feature success here.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("stream", "method"),
        [
            ("qa_suite", "generate_qa_features"),
            ("accessibility", "generate_accessibility_features"),
            ("demo_video", "generate_demo_features"),
            ("documentation", "generate_documentation_features"),
        ],
    )
    async def test_empty_feature_list_raises(self, stream, method):
        processor = JobProcessor()
        processor._feature_gen = MagicMock(spec=FeatureGenerator)
        getattr(processor._feature_gen, method).return_value = []
        with pytest.raises(RuntimeError, match="zero feature"):
            await processor._generate_features(stream, {"pages": []}, {})

    @pytest.mark.asyncio
    async def test_non_empty_feature_list_passes_through(self):
        processor = JobProcessor()
        processor._feature_gen = MagicMock(spec=FeatureGenerator)
        processor._feature_gen.generate_qa_features.return_value = [{"filename": "a.feature"}]
        out = await processor._generate_features("qa_suite", {"pages": []}, {})
        assert out == [{"filename": "a.feature"}]
