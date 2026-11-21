"""web_url input resolution through the in-service page fetcher.

The pipeline no longer calls ``webscraper_service``: ``JobProcessor`` builds a
:class:`PageFetcher` and derives structure from the returned pages. These tests
pin that wiring (URL list, auth/state plumbing, close-on-error) without starting
a browser.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.input.page_fetcher import FetchedImage, FetchedPage


class _FakeFetcher:
    """Stand-in for ``PageFetcher``: records its call, returns canned pages."""

    def __init__(self, pages=None, error: Exception | None = None) -> None:
        self.pages = pages or []
        self.error = error
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.closed = False

    async def fetch_pages(self, urls, **kwargs):
        self.calls.append((list(urls), kwargs))
        if self.error is not None:
            raise self.error
        return self.pages

    async def close(self) -> None:
        self.closed = True


class TestResolveWebUrlInput:
    """Tests for JobProcessor._resolve_input with input_type='web_url'."""

    @pytest.fixture
    def mock_settings(self):
        settings = MagicMock()
        settings.downstream.license_token = "test-token"
        settings.narration.words_per_minute = 150
        return settings

    @pytest.fixture
    def processor(self, mock_settings):
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
        """Fetched pages reach the inspection adapter with canonical keys."""
        page = FetchedPage(
            url="https://example.com",
            title="Example",
            html="<html><body>Hello</body></html>",
            text="Hello",
            links=["https://example.com/about"],
            images=[FetchedImage(src="https://example.com/logo.png")],
        )
        fetcher = _FakeFetcher(pages=[page])

        def _factory() -> _FakeFetcher:
            return fetcher

        adapter_result = MagicMock()
        adapter_dict = {
            "pages": [{"url": "https://example.com", "title": "Example"}],
            "platform": "web",
            "base_url": "https://example.com",
            "navigation_graph": [],
        }
        with (
            patch("src.services.job_processor.PageFetcher", side_effect=_factory),
            patch("src.services.job_processor.asdict", return_value=adapter_dict),
        ):
            processor._adapter.from_web_crawl = MagicMock(return_value=adapter_result)
            result = await processor._resolve_input(
                input_type="web_url",
                target_url="https://example.com",
                app_path="",
                config={"target_pages": ["/pricing"]},
                tenant_id="0199a1b2-0000-7000-8000-00000000000a",
            )

        assert result["platform"] == "web"
        assert fetcher.closed, "fetcher must always be closed"
        urls, kwargs = fetcher.calls[0]
        assert urls == ["https://example.com", "https://example.com/pricing"]
        assert kwargs["javascript_enabled"] is True
        assert kwargs["storage_state"] is None

        adapter_call = processor._adapter.from_web_crawl.call_args[0][0]
        assert adapter_call["pages"][0]["url"] == "https://example.com"
        assert adapter_call["pages"][0]["text_content"] == "Hello"
        assert "text" not in adapter_call["pages"][0]
        assert adapter_call["pages"][0]["images"] == ["https://example.com/logo.png"]
        assert adapter_call["url"] == "https://example.com"

    async def test_resolve_web_url_fetch_error(self, processor):
        """A page error raises RuntimeError with the fetcher's reason."""
        fetcher = _FakeFetcher(pages=[FetchedPage(url="https://example.com", error="timeout")])
        with (
            patch("src.services.job_processor.PageFetcher", return_value=fetcher),
            pytest.raises(RuntimeError, match=r"Failed to fetch.*timeout"),
        ):
            await processor._resolve_input(
                input_type="web_url",
                target_url="https://example.com",
                app_path="",
                config={},
            )
        assert fetcher.closed

    async def test_resolve_web_url_closes_fetcher_on_boot_failure(self, processor):
        """The fetcher is closed even when starting it raises."""
        fetcher = _FakeFetcher(error=RuntimeError("chromium missing"))
        with (
            patch("src.services.job_processor.PageFetcher", return_value=fetcher),
            pytest.raises(RuntimeError, match="chromium missing"),
        ):
            await processor._resolve_input(
                input_type="web_url",
                target_url="https://example.com",
                app_path="",
                config={},
            )
        assert fetcher.closed

    async def test_resolve_web_url_reads_storage_state_from_tenant_uploads(self, processor):
        """auth_config.storage_state_path is resolved through tenant-confined storage."""
        state = {"cookies": [{"name": "session", "value": "x"}], "origins": []}
        processor._storage.retrieve_text = AsyncMock(return_value=json.dumps(state))
        fetcher = _FakeFetcher(pages=[FetchedPage(url="https://example.com", html="<html/>")])
        with (
            patch("src.services.job_processor.PageFetcher", return_value=fetcher),
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
                config={"auth_config": {"storage_state_path": "uploads/t/state.json"}},
                tenant_id="0199a1b2-0000-7000-8000-00000000000a",
            )

        processor._storage.retrieve_text.assert_awaited_once_with(
            "uploads/t/state.json", tenant_id="0199a1b2-0000-7000-8000-00000000000a"
        )
        _, kwargs = fetcher.calls[0]
        assert kwargs["storage_state"] == state

    async def test_resolve_web_url_storage_state_requires_tenant(self, processor):
        """Without a tenant there is no upload prefix to confine the state to."""
        with pytest.raises(RuntimeError, match="requires a tenant identity"):
            await processor._resolve_input(
                input_type="web_url",
                target_url="https://example.com",
                app_path="",
                config={"auth_config": {"storage_state_path": "uploads/t/state.json"}},
            )

    async def test_resolve_web_url_static_mode_flag(self, processor):
        """javascript_enabled=false is forwarded to the fetcher."""
        fetcher = _FakeFetcher(pages=[FetchedPage(url="https://example.com", html="<html/>")])
        with (
            patch("src.services.job_processor.PageFetcher", return_value=fetcher),
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
                config={"javascript_enabled": False},
            )
        _, kwargs = fetcher.calls[0]
        assert kwargs["javascript_enabled"] is False

    async def test_resolve_unknown_input_type_raises(self, processor):
        """Unknown input_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown input type"):
            await processor._resolve_input(
                input_type="invalid_type",
                target_url="",
                app_path="",
                config={},
            )

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
