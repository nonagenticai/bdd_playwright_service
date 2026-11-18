"""Tests for web_url input resolution via WebscraperClient in JobProcessor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from w2v_common.clients.webscraper import ImageRef, ScrapeResponse


class TestResolveWebUrlInput:
    """Tests for JobProcessor._resolve_input with input_type='web_url'."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with the (sole) downstream license_token configured.

        Per-peer URLs are no longer read from settings -- the JobProcessor calls
        ``w2v_common.discovery.service_url()`` directly. Only ``license_token``
        is still pulled from ``settings.downstream``.
        """
        settings = MagicMock()
        settings.downstream.license_token = "test-token"
        settings.narration.words_per_minute = 150
        return settings

    @pytest.fixture
    def processor(self, mock_settings):
        """Create a JobProcessor with mocked dependencies."""
        with (
            patch("src.services.job_processor.get_settings", return_value=mock_settings),
            patch("src.services.job_processor.LLMClient"),
            patch("src.services.job_processor.FeatureGenerator"),
            patch("src.services.job_processor.StepDefGenerator"),
            patch("src.services.job_processor.StorageService"),
            patch("src.services.job_processor.PackageBuilder"),
            patch("src.services.job_processor.NarrationHumanizer"),
        ):
            from src.services.job_processor import JobProcessor

            return JobProcessor()

    async def test_resolve_web_url_success(self, processor):
        """Successful web scrape produces crawl data for the inspection adapter."""
        scrape_response = ScrapeResponse(
            url="https://example.com",
            title="Example",
            html="<html><body>Hello</body></html>",
            text="Hello",
            links=["https://example.com/about"],
            images=[ImageRef(src="https://example.com/logo.png", alt="logo")],
        )

        mock_ws = AsyncMock()
        mock_ws.scrape_bulk = AsyncMock(return_value=[scrape_response])
        mock_ws.close = AsyncMock()

        with (
            patch("w2v_common.clients.WebscraperClient", return_value=mock_ws),
            patch("w2v_common.clients._ssl.get_ssl_context", return_value=True),
        ):
            # The adapter's from_web_crawl will be called with the crawl data
            adapter_result = MagicMock()
            adapter_result_dict = {
                "pages": [
                    {
                        "url": "https://example.com",
                        "title": "Example",
                    }
                ],
                "platform": "web",
                "base_url": "https://example.com",
                "navigation_graph": [],
            }
            processor._adapter.from_web_crawl = MagicMock(return_value=adapter_result)

            # Mock asdict to return the dict
            with patch("src.services.job_processor.asdict", return_value=adapter_result_dict):
                result = await processor._resolve_input(
                    input_type="web_url",
                    target_url="https://example.com",
                    app_path="",
                    config={},
                )

        assert result["platform"] == "web"
        assert result["base_url"] == "https://example.com"

        # Verify WebscraperClient was used correctly (bulk, single-URL batch)
        mock_ws.scrape_bulk.assert_awaited_once_with(
            ["https://example.com"], auth_config=None, delay=2.0
        )
        mock_ws.close.assert_awaited_once()

        # Verify adapter received crawl data with pages
        adapter_call = processor._adapter.from_web_crawl.call_args[0][0]
        assert len(adapter_call["pages"]) == 1
        assert adapter_call["pages"][0]["url"] == "https://example.com"
        assert adapter_call["pages"][0]["title"] == "Example"
        assert adapter_call["pages"][0]["html"] == "<html><body>Hello</body></html>"
        # Canonical key the adapter reads (was "text" → silently empty text_content).
        assert adapter_call["pages"][0]["text_content"] == "Hello"
        assert "text" not in adapter_call["pages"][0]
        # ImageRef objects are flattened to src strings (JSON-serialisable page dict).
        assert adapter_call["pages"][0]["images"] == ["https://example.com/logo.png"]
        assert adapter_call["url"] == "https://example.com"

    async def test_resolve_web_url_scrape_error(self, processor):
        """Scrape error raises RuntimeError with descriptive message."""
        scrape_response = ScrapeResponse(
            url="https://example.com",
            error="Connection timeout",
        )

        mock_ws = AsyncMock()
        mock_ws.scrape_bulk = AsyncMock(return_value=[scrape_response])
        mock_ws.close = AsyncMock()

        with (
            patch("w2v_common.clients.WebscraperClient", return_value=mock_ws),
            patch("w2v_common.clients._ssl.get_ssl_context", return_value=True),
            pytest.raises(RuntimeError, match=r"Failed to crawl.*Connection timeout"),
        ):
            await processor._resolve_input(
                input_type="web_url",
                target_url="https://example.com",
                app_path="",
                config={},
            )

        # Client should always be closed, even on error
        mock_ws.close.assert_awaited_once()

    async def test_resolve_web_url_closes_client_on_exception(self, processor):
        """WebscraperClient is always closed even when the scrape raises."""
        mock_ws = AsyncMock()
        mock_ws.scrape_bulk = AsyncMock(side_effect=RuntimeError("network error"))
        mock_ws.close = AsyncMock()

        with (
            patch("w2v_common.clients.WebscraperClient", return_value=mock_ws),
            patch("w2v_common.clients._ssl.get_ssl_context", return_value=True),
            pytest.raises(RuntimeError, match="network error"),
        ):
            await processor._resolve_input(
                input_type="web_url",
                target_url="https://example.com",
                app_path="",
                config={},
            )

        mock_ws.close.assert_awaited_once()

    async def test_resolve_unknown_input_type_raises(self, processor):
        """Unknown input_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown input type"):
            await processor._resolve_input(
                input_type="invalid_type",
                target_url="",
                app_path="",
                config={},
            )

    async def test_resolve_web_url_uses_discovery_for_webscraper_url(self, processor, monkeypatch):
        """JobProcessor must build the WebscraperClient with the discovery URL.

        We override W2V_WEBSCRAPER_URL so we can assert exactly what URL
        ``service_url('webscraper')`` produced and was forwarded to the client.
        The license_token must still be sourced from settings.downstream.
        """
        monkeypatch.setenv("W2V_WEBSCRAPER_URL", "http://webscraper-test.local:9119")

        scrape_response = ScrapeResponse(
            url="https://example.com",
            title="Example",
            html="<html/>",
            text="",
            links=[],
            images=[],
        )

        mock_ws = AsyncMock()
        mock_ws.scrape_bulk = AsyncMock(return_value=[scrape_response])
        mock_ws.close = AsyncMock()

        constructed_with: dict[str, object] = {}

        def _factory(**kwargs):
            constructed_with.update(kwargs)
            return mock_ws

        with (
            patch("w2v_common.clients.WebscraperClient", side_effect=_factory),
            patch("w2v_common.clients._ssl.get_ssl_context", return_value="ssl-ctx"),
            patch(
                "src.services.job_processor.asdict",
                return_value={"platform": "web", "base_url": "https://example.com"},
            ),
        ):
            processor._adapter.from_web_crawl = MagicMock(return_value=MagicMock())
            await processor._resolve_input(
                input_type="web_url",
                target_url="https://example.com",
                app_path="",
                config={},
            )

        assert constructed_with["base_url"] == "http://webscraper-test.local:9119"
        assert constructed_with["license_token"] == "test-token"
        assert constructed_with["ssl_context"] == "ssl-ctx"

    async def test_resolve_web_url_client_gets_webscraper_lane_not_tenant_id(self, processor):
        """WebscraperClient takes no tenant_id: identity rides the webscraper lane.

        The job tenant reaches webscraper via the ``webscraper`` TenantLicenceLane
        (bound identity context) or the Bearer licence; a plain tenant_id header
        fallback (the WebscraperClient shape) must not be reintroduced.
        """
        scrape_response = ScrapeResponse(url="https://example.com", title="X", html="<html/>")
        mock_ws = AsyncMock()
        mock_ws.scrape_bulk = AsyncMock(return_value=[scrape_response])
        mock_ws.close = AsyncMock()

        constructed_with: dict[str, object] = {}

        def _factory(**kwargs):
            constructed_with.update(kwargs)
            return mock_ws

        with (
            patch("w2v_common.clients.WebscraperClient", side_effect=_factory),
            patch("w2v_common.clients._ssl.get_ssl_context", return_value=None),
            patch(
                "src.services.job_processor.asdict",
                return_value={"platform": "web", "base_url": "https://example.com"},
            ),
        ):
            processor._adapter.from_web_crawl = MagicMock(return_value=MagicMock())
            await processor._resolve_input(
                input_type="web_url",
                target_url="https://example.com",
                app_path="",
                config={},
                tenant_id="n8n-dev",
            )

        assert "tenant_id" not in constructed_with
        assert constructed_with["licence_lane"] is processor._webscraper_lane
        assert processor._webscraper_lane.hop == "webscraper"

    async def test_resolve_web_url_http_error_degrades_to_runtime_error(self, processor):
        """A non-2xx from webscraper RAISES (the old KAG client swallowed it into
        ``.error``); the resolver keeps the old ``Failed to crawl`` degrade."""
        request = httpx.Request("POST", "http://webscraper.test/api/scrape/bulk")
        response = httpx.Response(502, request=request)
        mock_ws = AsyncMock()
        mock_ws.scrape_bulk = AsyncMock(
            side_effect=httpx.HTTPStatusError("bad gateway", request=request, response=response)
        )
        mock_ws.close = AsyncMock()

        with (
            patch("w2v_common.clients.WebscraperClient", return_value=mock_ws),
            patch("w2v_common.clients._ssl.get_ssl_context", return_value=None),
            pytest.raises(RuntimeError, match=r"Failed to crawl https://example\.com"),
        ):
            await processor._resolve_input(
                input_type="web_url",
                target_url="https://example.com",
                app_path="",
                config={},
            )

        mock_ws.close.assert_awaited_once()

    async def test_resolve_mobile_app_uses_discovery_for_app_inspector_url(
        self, processor, monkeypatch
    ):
        """mobile_app input_type must build AppInspectorClient with discovery URL."""
        monkeypatch.setenv("W2V_APP_INSPECTOR_URL", "http://app-inspector-test.local:6311")

        captured: dict[str, object] = {}

        mock_client = AsyncMock()
        mock_client.start_inspection = AsyncMock(return_value="insp-123")
        mock_client.wait_for_inspection = AsyncMock(return_value={"screens": []})
        mock_client.close = AsyncMock()

        def _factory(base_url, **kwargs):
            captured["base_url"] = base_url
            captured.update(kwargs)
            return mock_client

        with (
            patch(
                "src.services.input.app_inspector_client.AppInspectorClient",
                side_effect=_factory,
            ),
            patch(
                "src.services.job_processor.asdict",
                return_value={"platform": "ios", "base_url": ""},
            ),
        ):
            processor._adapter.from_mobile_inspection = MagicMock(return_value=MagicMock())
            await processor._resolve_input(
                input_type="mobile_app",
                target_url="",
                app_path="/tmp/app.ipa",
                config={},
            )

        assert captured["base_url"] == "http://app-inspector-test.local:6311"
        mock_client.close.assert_awaited_once()
