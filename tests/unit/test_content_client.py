"""Tests for ContentClient HTTP client (backed by BaseServiceClient)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.services.clients.content_client import ContentClient


@pytest.fixture
def content_client() -> ContentClient:
    """Create a ContentClient with explicit base URL."""
    return ContentClient(base_url="http://content-test:6305")


class TestEnhanceNarration:
    async def test_enhance_narration_sends_correct_payload(self, content_client: ContentClient):
        """enhance_narration sends correct payload to the right endpoint and returns result."""
        expected = {"shot_list": {"enhanced": True}, "token_usage": {"total": 100}}
        mock_req = AsyncMock(return_value=expected)
        with patch.object(content_client, "_request", mock_req):
            result = await content_client.enhance_narration(
                shot_list={"shots": [{"action": "click"}]},
                storyline={"chapters": []},
                analysis={"pages": [{"url": "https://example.com"}]},
            )

        assert result == expected
        mock_req.assert_awaited_once_with(
            "POST",
            "/api/shotlist/enhance",
            json={
                "shot_list": {"shots": [{"action": "click"}]},
                "storyline": {"chapters": []},
                "analysis": {"pages": [{"url": "https://example.com"}]},
            },
        )

    async def test_enhance_narration_propagates_request_error(self, content_client: ContentClient):
        """Errors from _request propagate up as-is."""
        mock_req = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
        )
        with patch.object(content_client, "_request", mock_req):
            with pytest.raises(httpx.HTTPStatusError, match="500"):
                await content_client.enhance_narration(
                    shot_list={}, storyline={}, analysis={"pages": []}
                )


class TestClientConfiguration:
    def test_base_url_stripped_of_trailing_slash(self):
        """base_url has trailing slashes stripped."""
        client = ContentClient(base_url="http://content:6305/")
        assert client.base_url == "http://content:6305"

    def test_service_name_set_correctly(self):
        """service_name is set to content_service."""
        client = ContentClient(base_url="http://content:6305")
        assert client.service_name == "content_service"

    def test_timeout_set_to_60(self):
        """Default timeout is 60 seconds."""
        client = ContentClient(base_url="http://content:6305")
        assert client.timeout == 60.0

    def test_license_token_stored(self):
        """License token is passed to the base class."""
        client = ContentClient(base_url="http://content:6305", license_token="tok-123")
        assert client.license_token == "tok-123"


class TestHealthCheck:
    async def test_health_check_healthy(self, content_client: ContentClient):
        """Healthy service returns health data with service name."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "healthy"}
        mock_resp.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.is_closed = False
        mock_http_client.get = AsyncMock(return_value=mock_resp)
        content_client._client = mock_http_client

        result = await content_client.health_check()
        assert result["status"] == "healthy"
        assert result["service"] == "content_service"

    async def test_health_check_unhealthy(self, content_client: ContentClient):
        """Unhealthy service returns error info when connection fails."""
        mock_http_client = AsyncMock()
        mock_http_client.is_closed = False
        mock_http_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        content_client._client = mock_http_client

        result = await content_client.health_check()
        assert result["status"] == "unhealthy"
        assert "error" in result
        assert "connection refused" in result["error"]
