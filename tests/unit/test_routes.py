"""Tests for stream route endpoints."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.storage_service import StorageService

# Real registered tenant (licenses.tenants -> fr-test-user). The routes no longer
# substitute a default tenant for an unauthenticated caller, so the harness must
# forward a REAL one exactly like production traffic does.
TEST_TENANT_ID = "0195f48b-d000-7000-8000-000000000099"


def _make_mock_job(
    job_id: str = "test-job-id",
    status: str = "pending",
    stream: str = "qa_suite",
    **kwargs,
) -> MagicMock:
    """Create a mock BDDJob object."""
    job = MagicMock()
    job.job_id = job_id
    job.status = status
    job.stream = stream
    job.tenant_id = kwargs.get("tenant_id", "n8n-dev")
    job.input_type = kwargs.get("input_type", "web_url")
    job.target_url = kwargs.get("target_url")
    job.app_path = kwargs.get("app_path")
    job.features_generated = kwargs.get("features_generated", 0)
    job.step_defs_generated = kwargs.get("step_defs_generated", 0)
    job.artifact_paths = kwargs.get("artifact_paths", {})
    job.narration_script = kwargs.get("narration_script", {})
    job.downstream_jobs = kwargs.get("downstream_jobs", {})
    job.generation_time_seconds = kwargs.get("generation_time_seconds")
    job.error_message = kwargs.get("error_message")
    job.created_at = kwargs.get("created_at")
    job.completed_at = kwargs.get("completed_at")
    job.config = kwargs.get("config", {})
    job.narration_config = kwargs.get("narration_config", {})
    return job


@pytest.fixture
def mock_db_ctx():
    """Create a mock for get_db_session async context manager."""
    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.flush = AsyncMock()

    @asynccontextmanager
    async def _mock_get_db_session():
        yield mock_session

    return _mock_get_db_session, mock_session


@pytest.fixture
def client(mock_db_ctx):
    """Create a test client with mocked DB session -- no lifespan.

    Registers the asgi envelope handlers so error responses match the
    standardized ``{error: {code, message, status, ...}}`` shape that
    consumers (incl. ``tests/test_error_envelope.py``) assert against.
    """
    from w2v_common.asgi import register_error_envelope_handlers

    mock_get_db_session, _ = mock_db_ctx

    app = FastAPI()
    # Same canonical envelope handlers the service registers in src/asgi.py.
    register_error_envelope_handlers(app)

    # Stand in for AuthMiddleware: forward a REAL registered tenant on every
    # request. Without it these calls are unattributed and now 401 (by design).
    # Like the real middleware it binds BOTH request.state and the identity
    # ContextVars — dispatch_job captures the job's identity from the latter.
    # Provenance stays None (unvouched): these route tests must not let the
    # detached job spend or sign as the tenant against unreachable peers.
    @app.middleware("http")
    async def _set_tenant(request, call_next):
        from w2v_common.auth import clear_identity_context, set_identity_context

        request.state.tenant_id = TEST_TENANT_ID
        set_identity_context(TEST_TENANT_ID, None, None, tenant_source=None)
        try:
            return await call_next(request)
        finally:
            clear_identity_context()

    from src.api.routes import router as api_router

    app.include_router(api_router)

    with patch("src.api.routes.qa_suite.get_db_session", mock_get_db_session):
        with patch("src.api.routes.accessibility.get_db_session", mock_get_db_session):
            with patch("src.api.routes.documentation.get_db_session", mock_get_db_session):
                with patch("src.api.routes.demo_video.get_db_session", mock_get_db_session):
                    with patch("src.api.routes.packages.get_db_session", mock_get_db_session):
                        with patch("src.api.routes.jobs.get_db_session", mock_get_db_session):
                            with TestClient(app, raise_server_exceptions=True) as c:
                                yield c


class TestQASuiteRoute:
    def test_start_qa_suite_with_url(self, client, mock_db_ctx):
        """POST /api/bdd/qa-suite with a URL should create a job."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="qa-123", stream="qa_suite")

        with patch(
            "src.api.routes.qa_suite.JobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.post("/api/bdd/qa-suite", json={"url": "https://example.com"})

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "qa-123"
        assert data["status"] == "pending"
        assert data["poll_url"] == "/api/bdd/qa-123"

    def test_start_qa_suite_with_app_path(self, client, mock_db_ctx):
        """POST /api/bdd/qa-suite with app_path should create a job."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="qa-456", stream="qa_suite", input_type="mobile_app")

        with patch(
            "src.api.routes.qa_suite.JobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.post("/api/bdd/qa-suite", json={"app_path": "/path/to/app.apk"})

        assert response.status_code == 200
        assert response.json()["job_id"] == "qa-456"

    def test_start_qa_suite_missing_input(self, client):
        """POST /api/bdd/qa-suite with no input source should return 400."""
        response = client.post("/api/bdd/qa-suite", json={})
        assert response.status_code == 400
        assert "input source" in response.json()["error"]["message"].lower()

    def test_start_qa_suite_with_feature_files(self, client, mock_db_ctx):
        """POST /api/bdd/qa-suite with feature_files should create a job."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="qa-789", stream="qa_suite", input_type="gherkin_upload")

        with patch(
            "src.api.routes.qa_suite.JobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.post(
                "/api/bdd/qa-suite", json={"feature_files": ["/tmp/login.feature"]}
            )

        assert response.status_code == 200
        assert response.json()["job_id"] == "qa-789"

    def test_start_qa_suite_with_inspection_job_id(self, client, mock_db_ctx):
        """POST /api/bdd/qa-suite with inspection_job_id should create a job."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(
            job_id="qa-abc", stream="qa_suite", input_type="inspection_passthrough"
        )

        with patch(
            "src.api.routes.qa_suite.JobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.post("/api/bdd/qa-suite", json={"inspection_job_id": "inspect-999"})

        assert response.status_code == 200
        assert response.json()["job_id"] == "qa-abc"


class TestAccessibilityRoute:
    def test_start_accessibility_with_url(self, client, mock_db_ctx):
        """POST /api/bdd/accessibility with a URL should create a job."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="a11y-123", stream="accessibility")

        with patch(
            "src.api.routes.accessibility.JobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.post("/api/bdd/accessibility", json={"url": "https://example.com"})

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "a11y-123"
        assert data["poll_url"] == "/api/bdd/a11y-123"

    def test_start_accessibility_missing_input(self, client):
        """POST /api/bdd/accessibility with no input source should return 400."""
        response = client.post("/api/bdd/accessibility", json={})
        assert response.status_code == 400

    def test_start_accessibility_with_app_path(self, client, mock_db_ctx):
        """POST /api/bdd/accessibility with app_path should create a job."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="a11y-456", stream="accessibility")

        with patch(
            "src.api.routes.accessibility.JobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.post("/api/bdd/accessibility", json={"app_path": "/path/to/app"})

        assert response.status_code == 200
        assert response.json()["job_id"] == "a11y-456"


class TestDocumentationRoute:
    def test_start_documentation_with_url(self, client, mock_db_ctx):
        """POST /api/bdd/documentation with a URL should create a job."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="doc-123", stream="documentation")

        with patch(
            "src.api.routes.documentation.JobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.post("/api/bdd/documentation", json={"url": "https://example.com"})

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "doc-123"

    def test_start_documentation_missing_input(self, client):
        """POST /api/bdd/documentation with no input source should return 400."""
        response = client.post("/api/bdd/documentation", json={})
        assert response.status_code == 400

    def test_start_documentation_with_feature_files(self, client, mock_db_ctx):
        """POST /api/bdd/documentation with feature_files should create a job."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="doc-456", stream="documentation")

        with patch(
            "src.api.routes.documentation.JobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.post(
                "/api/bdd/documentation", json={"feature_files": ["/tmp/test.feature"]}
            )

        assert response.status_code == 200
        assert response.json()["job_id"] == "doc-456"


class TestDemoVideoRoute:
    def test_start_demo_video_with_url(self, client, mock_db_ctx):
        """POST /api/bdd/demo-video with a URL should create a job."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="vid-123", stream="demo_video")

        with patch(
            "src.api.routes.demo_video.JobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.post("/api/bdd/demo-video", json={"url": "https://example.com"})

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "vid-123"

    def test_start_demo_video_missing_input(self, client):
        """POST /api/bdd/demo-video with no input source should return 400."""
        response = client.post("/api/bdd/demo-video", json={})
        assert response.status_code == 400

    def test_start_demo_video_with_app_path(self, client, mock_db_ctx):
        """POST /api/bdd/demo-video with app_path should create a job."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="vid-456", stream="demo_video")

        with patch(
            "src.api.routes.demo_video.JobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.post("/api/bdd/demo-video", json={"app_path": "/path/to/app"})

        assert response.status_code == 200
        assert response.json()["job_id"] == "vid-456"

    def test_start_demo_video_forwards_auth_config(self, client, mock_db_ctx):
        """auth_config from the request body must reach JobManager.create_job in config."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="vid-789", stream="demo_video")
        auth_config = {"storage_state_path": "uploads/t/state.json"}

        with patch(
            "src.api.routes.demo_video.JobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ) as mock_create:
            response = client.post(
                "/api/bdd/demo-video",
                json={"url": "https://fr.nonagentic.ai", "auth_config": auth_config},
            )

        assert response.status_code == 200
        assert mock_create.await_args is not None
        assert mock_create.await_args.kwargs["config"]["auth_config"] == auth_config

    def test_start_demo_video_forwards_min_scenarios_to_the_scenario_floor(
        self, client, mock_db_ctx
    ):
        """min_scenarios must survive BOTH hops and bind the generator's floor.

        Deliberately asserts on the config dict the generator actually receives, and
        then feeds that exact dict to ``_scenario_floor``. A substring/contains check
        cannot see a key's ABSENCE, and `extra="ignore"` drops an undeclared field with
        no 422 — so a schema-only or route-only fix would pass a weaker test.
        """
        from src.services.generation.feature_generator import _scenario_floor

        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="vid-floor", stream="demo_video")

        with patch(
            "src.api.routes.demo_video.JobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ) as mock_create:
            response = client.post(
                "/api/bdd/demo-video",
                json={
                    "url": "https://example.com",
                    "min_scenarios": 6,
                    "max_scenarios_per_feature": 10,
                },
            )

        assert response.status_code == 200
        assert mock_create.await_args is not None
        config = mock_create.await_args.kwargs["config"]
        assert "min_scenarios" in config, (
            "min_scenarios was dropped before the generator: the schema ignores "
            "undeclared fields silently and the route copies config key by key"
        )
        assert config["min_scenarios"] == 6
        # The consumer must actually bind on this exact key, not merely carry it.
        assert _scenario_floor(config) == 6

    def test_start_demo_video_without_min_scenarios_implies_no_floor(self, client, mock_db_ctx):
        """Omitting it stays backward compatible: None, and a floor of 0."""
        from src.services.generation.feature_generator import _scenario_floor

        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="vid-nofloor", stream="demo_video")

        with patch(
            "src.api.routes.demo_video.JobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ) as mock_create:
            response = client.post("/api/bdd/demo-video", json={"url": "https://example.com"})

        assert response.status_code == 200
        assert mock_create.await_args is not None
        config = mock_create.await_args.kwargs["config"]
        assert config["min_scenarios"] is None
        assert _scenario_floor(config) == 0


class TestJobStatusRoute:
    def test_get_job_status(self, client, mock_db_ctx):
        """GET /api/bdd/{job_id} should return job status."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(
            job_id="status-123",
            status="completed",
            stream="qa_suite",
            features_generated=5,
            step_defs_generated=10,
        )

        with patch(
            "src.api.routes.jobs.JobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.get("/api/bdd/status-123")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "status-123"
        assert data["status"] == "completed"
        assert data["features_generated"] == 5

    def test_get_job_status_not_found(self, client, mock_db_ctx):
        """GET /api/bdd/{job_id} should return 404 for unknown jobs."""
        with patch(
            "src.api.routes.jobs.JobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.get("/api/bdd/nonexistent")

        assert response.status_code == 404

    def test_get_job_result(self, client, mock_db_ctx):
        """GET /api/bdd/{job_id}/result should return full result."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(
            job_id="result-123",
            status="completed",
            stream="qa_suite",
            features_generated=3,
            step_defs_generated=6,
        )

        with patch(
            "src.api.routes.jobs.JobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.get("/api/bdd/result-123/result")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "result-123"
        assert data["status"] == "completed"

    def test_get_job_result_still_processing(self, client, mock_db_ctx):
        """GET /api/bdd/{job_id}/result should return 409 if job is still processing."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="proc-123", status="processing", stream="qa_suite")

        with patch(
            "src.api.routes.jobs.JobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.get("/api/bdd/proc-123/result")

        assert response.status_code == 409
        assert response.headers.get("retry-after") == "30"


class TestPackagesRoute:
    def test_download_not_found(self, client, mock_db_ctx):
        """GET /api/bdd/{job_id}/download should return 404 for unknown jobs."""
        with patch(
            "src.api.routes.packages.JobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.get("/api/bdd/nonexistent/download")

        assert response.status_code == 404

    def test_download_not_completed(self, client, mock_db_ctx):
        """GET /api/bdd/{job_id}/download should return 409 for incomplete jobs."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(job_id="dl-123", status="processing", stream="qa_suite")

        with patch(
            "src.api.routes.packages.JobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.get("/api/bdd/dl-123/download")

        assert response.status_code == 409
        assert response.headers.get("retry-after") == "30"

    def test_download_no_package_file(self, client, mock_db_ctx):
        """GET /api/bdd/{job_id}/download should return 404 if package zip is missing."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(
            job_id="dl-456",
            status="completed",
            stream="qa_suite",
            artifact_paths={},
        )

        with patch(
            "src.api.routes.packages.JobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.get("/api/bdd/dl-456/download")

        assert response.status_code == 404

    def test_download_completed_with_file(self, client, mock_db_ctx, tmp_path):
        """GET /api/bdd/{job_id}/download should return file for completed jobs."""
        _, _mock_session = mock_db_ctx

        # Create a real temp zip file
        zip_path = tmp_path / "test.zip"
        import zipfile

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("features/login.feature", "Feature: Login")

        mock_job = _make_mock_job(
            job_id="dl-789",
            status="completed",
            stream="qa_suite",
            artifact_paths={"package_zip": str(zip_path)},
        )

        with patch(
            "src.api.routes.packages.JobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            response = client.get("/api/bdd/dl-789/download")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

    def test_download_streams_from_object_store(self, client, mock_db_ctx):
        """Multi-replica safety: the download resolves the package from the SHARED
        store by object KEY (not a pod-local file), so a replica that did not
        produce the job still serves it. Regression guard for the cross-pod 404."""
        _, _mock_session = mock_db_ctx
        mock_job = _make_mock_job(
            job_id="dl-obj",
            status="completed",
            stream="qa_suite",
            # A bare MinIO object key — there is no local file behind it.
            artifact_paths={"package_zip": "dl-obj/packages/bdd-qa_suite-dlobj.zip"},
        )

        class _FakeStream:
            status = 200
            headers: ClassVar[dict[str, str]] = {"content-length": "3", "etag": '"abc"'}

            async def iter_chunks(self, chunk_size: int = 65536):
                yield b"zip"

            async def aclose(self) -> None:
                return None

        with (
            patch(
                "src.api.routes.packages.JobManager.get_job",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch.object(
                StorageService,
                "open_stream",
                new_callable=AsyncMock,
                return_value=_FakeStream(),
            ) as mock_open,
        ):
            response = client.get("/api/bdd/dl-obj/download")

        assert response.status_code == 200
        assert response.content == b"zip"
        assert response.headers["content-type"] == "application/zip"
        assert "attachment" in response.headers["content-disposition"]
        # Resolved by the object key stored in artifact_paths, never a local path.
        mock_open.assert_awaited_once()
        assert mock_open.await_args is not None
        assert mock_open.await_args.args[0] == "dl-obj/packages/bdd-qa_suite-dlobj.zip"


class TestGherkinUploadRoute:
    def test_upload_feature_file(self, client, monkeypatch):
        """POST /api/bdd/gherkin/upload should accept .feature files."""
        monkeypatch.setenv("STORAGE_LOCAL_PATH", "/tmp/test_bdd_upload")
        from src.config import clear_settings_cache

        clear_settings_cache()

        content = b"Feature: Test\n  Scenario: Hello\n    Given I exist"
        files = [("files", ("test.feature", io.BytesIO(content), "text/plain"))]

        with patch(
            "src.api.routes.gherkin.StorageService.store_bytes",
            new_callable=AsyncMock,
            return_value="/tmp/test_bdd_upload/uploads/n8n-dev/test.feature",
        ):
            response = client.post("/api/bdd/gherkin/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["uploaded"] == 1
        assert len(data["paths"]) == 1

    def test_upload_non_feature_file(self, client, monkeypatch):
        """POST /api/bdd/gherkin/upload should reject non-.feature files."""
        monkeypatch.setenv("STORAGE_LOCAL_PATH", "/tmp/test_bdd_upload")
        from src.config import clear_settings_cache

        clear_settings_cache()

        content = b"not a feature file"
        files = [("files", ("test.txt", io.BytesIO(content), "text/plain"))]

        response = client.post("/api/bdd/gherkin/upload", files=files)
        assert response.status_code == 400
        assert ".feature" in response.json()["error"]["message"]

    def test_upload_no_files(self, client):
        """POST /api/bdd/gherkin/upload with no files should return 422."""
        response = client.post("/api/bdd/gherkin/upload")
        assert response.status_code == 422
