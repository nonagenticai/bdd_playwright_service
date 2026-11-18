"""Tests for VideoProductionClient HTTP client (backed by BaseServiceClient)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.services.clients.video_production_client import VideoProductionClient


@pytest.fixture
def vp_client() -> VideoProductionClient:
    """Create a VideoProductionClient with explicit base URL."""
    return VideoProductionClient(base_url="http://vp-test:6308")


class TestStartRecording:
    async def test_start_recording_delegates_to_submit_job(self, vp_client: VideoProductionClient):
        """start_recording delegates to submit_job and returns the job ID string."""
        mock_submit = AsyncMock(return_value="rec-001")
        with patch.object(vp_client, "submit_job", mock_submit):
            payload = {"script": "test scenario", "format": "mp4"}
            result = await vp_client.start_recording(payload)

        assert result == "rec-001"
        assert isinstance(result, str)
        mock_submit.assert_awaited_once_with("/api/record/part", payload)

    async def test_start_recording_sends_complex_payload(self, vp_client: VideoProductionClient):
        """start_recording passes all payload fields through to submit_job."""
        mock_submit = AsyncMock(return_value="rec-002")
        with patch.object(vp_client, "submit_job", mock_submit):
            payload = {
                "script": "multi-step scenario",
                "format": "mp4",
                "resolution": "1920x1080",
                "audio_url": "/audio/narration.wav",
            }
            result = await vp_client.start_recording(payload)

        assert result == "rec-002"
        sent_payload = mock_submit.call_args[0][1]
        assert sent_payload["resolution"] == "1920x1080"
        assert sent_payload["audio_url"] == "/audio/narration.wav"


class TestGetRecording:
    async def test_get_recording_returns_full_status(self, vp_client: VideoProductionClient):
        """get_recording returns the full recording status dict."""
        expected = {"job_id": "rec-001", "status": "completed", "video_url": "/video/001.mp4"}
        mock_req = AsyncMock(return_value=expected)
        with patch.object(vp_client, "_request", mock_req):
            result = await vp_client.get_recording("rec-001")

        assert result["job_id"] == "rec-001"
        assert result["status"] == "completed"
        assert result["video_url"] == "/video/001.mp4"
        mock_req.assert_awaited_once_with("GET", "/api/record/jobs/rec-001")

    async def test_get_recording_returns_processing_status(self, vp_client: VideoProductionClient):
        """get_recording returns processing status when job is in progress."""
        expected = {"job_id": "rec-002", "status": "processing", "progress": 45}
        mock_req = AsyncMock(return_value=expected)
        with patch.object(vp_client, "_request", mock_req):
            result = await vp_client.get_recording("rec-002")

        assert result["status"] == "processing"
        assert result["progress"] == 45


class TestWaitForRecording:
    async def test_wait_for_recording_delegates_to_poll_job(self, vp_client: VideoProductionClient):
        """wait_for_recording delegates to poll_job with correct endpoint and timeout."""
        expected = {"status": "completed", "video_url": "/video/done.mp4"}
        mock_poll = AsyncMock(return_value=expected)
        with patch.object(vp_client, "poll_job", mock_poll):
            result = await vp_client.wait_for_recording("rec-002")

        assert result["status"] == "completed"
        assert result["video_url"] == "/video/done.mp4"
        mock_poll.assert_awaited_once_with(
            endpoint="/api/record/jobs",
            job_id="rec-002",
            timeout=600.0,
        )

    async def test_wait_for_recording_custom_timeout(self, vp_client: VideoProductionClient):
        """Custom timeout is forwarded to poll_job."""
        mock_poll = AsyncMock(return_value={"status": "completed"})
        with patch.object(vp_client, "poll_job", mock_poll):
            await vp_client.wait_for_recording("rec-003", timeout=120.0)

        mock_poll.assert_awaited_once_with(
            endpoint="/api/record/jobs",
            job_id="rec-003",
            timeout=120.0,
        )

    async def test_wait_for_recording_propagates_timeout_error(
        self, vp_client: VideoProductionClient
    ):
        """TimeoutError from poll_job propagates up."""
        mock_poll = AsyncMock(side_effect=TimeoutError("timed out"))
        with patch.object(vp_client, "poll_job", mock_poll):
            with pytest.raises(TimeoutError, match="timed out"):
                await vp_client.wait_for_recording("rec-slow")

    async def test_wait_for_recording_propagates_runtime_error(
        self, vp_client: VideoProductionClient
    ):
        """RuntimeError from poll_job (failed status) propagates up."""
        mock_poll = AsyncMock(side_effect=RuntimeError("job failed: render error"))
        with patch.object(vp_client, "poll_job", mock_poll):
            with pytest.raises(RuntimeError, match="failed"):
                await vp_client.wait_for_recording("rec-fail")


class TestClientConfiguration:
    def test_base_url_stripped(self):
        """base_url has trailing slashes stripped."""
        client = VideoProductionClient(base_url="http://vp:6308/")
        assert client.base_url == "http://vp:6308"

    def test_service_name_set(self):
        """service_name is set correctly."""
        client = VideoProductionClient(base_url="http://vp:6308")
        assert client.service_name == "video_production_service"

    def test_timeout_set_to_120(self):
        """Default timeout is 120 seconds for video production."""
        client = VideoProductionClient(base_url="http://vp:6308")
        assert client.timeout == 120.0


class TestHealthCheck:
    async def test_health_check_returns_healthy_status(self, vp_client: VideoProductionClient):
        """Healthy service returns expected status in response dict."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "healthy", "uptime": 3600}
        mock_resp.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.is_closed = False
        mock_http_client.get = AsyncMock(return_value=mock_resp)
        vp_client._client = mock_http_client

        result = await vp_client.health_check()
        assert result["status"] == "healthy"
        assert result["service"] == "video_production_service"

    async def test_health_check_returns_unhealthy_with_error_detail(
        self, vp_client: VideoProductionClient
    ):
        """Connection failure returns unhealthy status with descriptive error."""
        mock_http_client = AsyncMock()
        mock_http_client.is_closed = False
        mock_http_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        vp_client._client = mock_http_client

        result = await vp_client.health_check()
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert isinstance(result["error"], str)
        assert len(result["error"]) > 0
