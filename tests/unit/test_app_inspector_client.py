"""Tests for AppInspectorClient HTTP client (BaseServiceClient-based)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from w2v_common.clients.base import BaseServiceClient

from src.services.input.app_inspector_client import AppInspectorClient


@pytest.fixture
def client():
    """Create an AppInspectorClient with test base URL."""
    return AppInspectorClient(base_url="http://inspector-test:6311")


class TestConstructor:
    def test_inherits_base_service_client(self, client):
        """AppInspectorClient is a BaseServiceClient."""
        assert isinstance(client, BaseServiceClient)

    def test_default_settings(self, client):
        """Verify service_name, timeout, and base_url are set correctly."""
        assert client.service_name == "app_inspector"
        assert client.timeout == 60.0
        assert client.base_url == "http://inspector-test:6311"

    def test_trailing_slash_stripped(self):
        """Trailing slash in base_url is stripped by BaseServiceClient."""
        c = AppInspectorClient(base_url="http://example.com:6311/")
        assert c.base_url == "http://example.com:6311"

    def test_license_token_stored(self):
        """license_token is passed through to BaseServiceClient."""
        c = AppInspectorClient(base_url="http://example.com:6311", license_token="tok-abc")
        assert c.license_token == "tok-abc"

    def test_license_token_default_none(self, client):
        """license_token defaults to None."""
        assert client.license_token is None


class TestStartInspection:
    async def test_calls_submit_job_with_full_payload(self, client):
        """start_inspection delegates to submit_job with correct payload."""
        mock_submit = AsyncMock(return_value="insp-123")
        with patch.object(client, "submit_job", mock_submit):
            result = await client.start_inspection("/app/path", platform="web", max_screens=10)

        assert result == "insp-123"
        mock_submit.assert_awaited_once_with(
            "/api/inspect",
            {"app_path": "/app/path", "platform": "web", "max_screens": 10},
        )

    async def test_omits_platform_when_none(self, client):
        """When platform is None, it is excluded from the payload."""
        mock_submit = AsyncMock(return_value="insp-456")
        with patch.object(client, "submit_job", mock_submit):
            result = await client.start_inspection("/app/path")

        assert result == "insp-456"
        mock_submit.assert_awaited_once_with(
            "/api/inspect",
            {"app_path": "/app/path", "max_screens": 20},
        )

    async def test_default_max_screens(self, client):
        """Default max_screens is 20."""
        mock_submit = AsyncMock(return_value="insp-789")
        with patch.object(client, "submit_job", mock_submit):
            await client.start_inspection("/my/app")

        payload = mock_submit.call_args[0][1]
        assert payload["max_screens"] == 20

    async def test_custom_max_screens(self, client):
        """Custom max_screens value is passed through."""
        mock_submit = AsyncMock(return_value="insp-999")
        with patch.object(client, "submit_job", mock_submit):
            await client.start_inspection("/my/app", max_screens=5)

        payload = mock_submit.call_args[0][1]
        assert payload["max_screens"] == 5


class TestGetInspection:
    async def test_delegates_to_request_get(self, client):
        """get_inspection calls _request('GET', ...) with correct endpoint."""
        expected = {"status": "completed", "pages": [{"url": "https://example.com"}]}
        mock_request = AsyncMock(return_value=expected)
        with patch.object(client, "_request", mock_request):
            result = await client.get_inspection("insp-123")

        assert result == expected
        mock_request.assert_awaited_once_with("GET", "/api/inspect/insp-123")

    async def test_returns_in_progress_status(self, client):
        """Returns partial data when job is still processing."""
        in_progress = {"status": "processing", "progress": 45}
        mock_request = AsyncMock(return_value=in_progress)
        with patch.object(client, "_request", mock_request):
            result = await client.get_inspection("insp-abc")

        assert result["status"] == "processing"
        assert result["progress"] == 45


class TestWaitForInspection:
    async def test_delegates_to_poll_job(self, client):
        """wait_for_inspection delegates to poll_job with correct args."""
        completed = {"status": "completed", "pages": []}
        mock_poll = AsyncMock(return_value=completed)
        with patch.object(client, "poll_job", mock_poll):
            result = await client.wait_for_inspection("insp-123", timeout=120.0)

        assert result == completed
        mock_poll.assert_awaited_once_with("/api/inspect", "insp-123", timeout=120.0)

    async def test_default_timeout(self, client):
        """Default timeout is 300 seconds."""
        mock_poll = AsyncMock(return_value={"status": "completed"})
        with patch.object(client, "poll_job", mock_poll):
            await client.wait_for_inspection("insp-456")

        mock_poll.assert_awaited_once_with("/api/inspect", "insp-456", timeout=300.0)

    async def test_propagates_timeout_error(self, client):
        """TimeoutError from poll_job propagates to caller."""
        mock_poll = AsyncMock(side_effect=TimeoutError("timed out"))
        with (
            patch.object(client, "poll_job", mock_poll),
            pytest.raises(TimeoutError, match="timed out"),
        ):
            await client.wait_for_inspection("insp-789")

    async def test_propagates_runtime_error_on_failure(self, client):
        """RuntimeError from poll_job (failed job) propagates to caller."""
        mock_poll = AsyncMock(
            side_effect=RuntimeError("app_inspector job insp-fail failed: Not found"),
        )
        with (
            patch.object(client, "poll_job", mock_poll),
            pytest.raises(RuntimeError, match=r"failed.*Not found"),
        ):
            await client.wait_for_inspection("insp-fail")


class TestFullWorkflow:
    """End-to-end workflow tests mocking _request (one level below domain methods)."""

    async def test_inspect_then_poll_to_completion(self, client):
        """Simulate full flow: submit -> poll processing -> poll completed."""
        submit_response = {"job_id": "wf-001"}
        processing_response = {"status": "processing", "progress": 50}
        completed_response = {"status": "completed", "pages": [{"url": "https://example.com"}]}

        call_count = 0

        async def mock_request(method: str, endpoint: str, **kwargs):
            nonlocal call_count
            call_count += 1
            if method == "POST":
                return submit_response
            # GET calls: first processing, then completed
            if call_count == 2:
                return processing_response
            return completed_response

        with patch.object(client, "_request", side_effect=mock_request):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                job_id = await client.start_inspection("/my/app", platform="android")
                assert job_id == "wf-001"

                result = await client.wait_for_inspection(job_id, timeout=60.0)
                assert result["status"] == "completed"
                assert len(result["pages"]) == 1

    async def test_inspect_job_fails(self, client):
        """Simulate a job that fails after submission."""
        submit_response = {"job_id": "wf-fail"}
        failed_response = {"status": "failed", "error": "App crashed during inspection"}

        async def mock_request(method: str, endpoint: str, **kwargs):
            if method == "POST":
                return submit_response
            return failed_response

        with patch.object(client, "_request", side_effect=mock_request):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                job_id = await client.start_inspection("/broken/app")
                assert job_id == "wf-fail"

                with pytest.raises(RuntimeError, match=r"failed.*App crashed"):
                    await client.wait_for_inspection(job_id)


class TestCleanup:
    async def test_close_delegates_to_base(self, client):
        """close() is available from BaseServiceClient."""
        mock_close = AsyncMock()
        with patch.object(client, "close", mock_close):
            await client.close()
        mock_close.assert_awaited_once()

    async def test_health_check_available(self, client):
        """health_check() is inherited from BaseServiceClient."""
        health_data = {"status": "healthy", "service": "app_inspector"}
        mock_health = AsyncMock(return_value=health_data)
        with patch.object(client, "health_check", mock_health):
            result = await client.health_check()
        assert result["status"] == "healthy"
        assert result["service"] == "app_inspector"
