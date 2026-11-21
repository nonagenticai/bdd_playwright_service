"""Unit tests for the in-service page fetcher (static mode + policy guards).

Rendered tests are limited to the guard: they must not start a browser, because
CI installs Chromium only when the suite is run there. The static path exercises
the real parser, redirect handling and every refusal rule.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from src.config import FetchSettings
from src.services.input.page_fetcher import (
    FetchedPage,
    PageFetcher,
    PageFetchError,
    validate_auth_config,
)

_REAL_ASYNC_CLIENT = httpx.AsyncClient
_PUBLIC_IP = "93.184.216.34"
_PUBLIC_URL = f"https://{_PUBLIC_IP}"


def _settings(**overrides: Any) -> FetchSettings:
    return FetchSettings(
        _env_file=None,
        timeout_seconds=2.0,
        post_load_delay_seconds=0.0,
        **overrides,
    )


def _patch_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    def _factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _factory)


async def test_static_fetch_extracts_structure(monkeypatch):
    html = (
        "<html><head><title>Prijs — App</title></head><body>"
        "<h1>Prijzen</h1><a href='/contact'>Contact</a>"
        "<img src='/logo.png'><script>ignore()</script>"
        "</body></html>"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text=html)

    _patch_transport(monkeypatch, handler)
    fetcher = PageFetcher(_settings())
    pages = await fetcher.fetch_pages([_PUBLIC_URL], javascript_enabled=False)

    assert len(pages) == 1
    page = pages[0]
    assert page.error is None
    assert page.title == "Prijs — App"
    assert "Prijzen" in page.text
    assert "ignore" not in page.text
    assert page.links == [f"{_PUBLIC_URL}/contact"]
    assert [image.src for image in page.images] == [f"{_PUBLIC_URL}/logo.png"]


async def test_private_address_is_refused_by_default(monkeypatch):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<html/>")

    _patch_transport(monkeypatch, handler)
    fetcher = PageFetcher(_settings())
    pages = await fetcher.fetch_pages(["http://127.0.0.1/admin"], javascript_enabled=False)

    assert pages[0].error is not None
    assert "not a public address" in pages[0].error
    assert calls == [], "the guard must run before any HTTP request"


async def test_link_local_metadata_address_is_refused(monkeypatch):
    _patch_transport(monkeypatch, lambda _request: httpx.Response(200))
    fetcher = PageFetcher(_settings())
    pages = await fetcher.fetch_pages(
        ["http://169.254.169.254/latest/meta-data"], javascript_enabled=False
    )

    assert pages[0].error is not None
    assert "not a public address" in pages[0].error


async def test_private_address_allowed_when_opted_in(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<html>x</html>")

    _patch_transport(monkeypatch, handler)
    fetcher = PageFetcher(_settings(allow_private=True))
    pages = await fetcher.fetch_pages(["http://10.1.2.3/internal"], javascript_enabled=False)

    assert pages[0].error is None


async def test_non_html_content_type_is_refused(monkeypatch):
    _patch_transport(
        monkeypatch,
        lambda _request: httpx.Response(
            200, headers={"content-type": "application/json"}, text='{"a": 1}'
        ),
    )
    fetcher = PageFetcher(_settings())
    pages = await fetcher.fetch_pages([_PUBLIC_URL], javascript_enabled=False)

    assert pages[0].error is not None
    assert "not an HTML page" in pages[0].error


async def test_redirect_chain_is_bounded(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/again"})

    _patch_transport(monkeypatch, handler)
    fetcher = PageFetcher(_settings())
    pages = await fetcher.fetch_pages([f"{_PUBLIC_URL}/loop"], javascript_enabled=False)

    assert pages[0].error is not None
    assert "too many redirects" in pages[0].error


async def test_redirect_to_private_address_is_refused(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})

    _patch_transport(monkeypatch, handler)
    fetcher = PageFetcher(_settings())
    pages = await fetcher.fetch_pages([_PUBLIC_URL], javascript_enabled=False)

    assert pages[0].error is not None
    assert "not a public address" in pages[0].error


async def test_html_is_capped_at_max_bytes(monkeypatch):
    html = "<html><body>" + ("x" * 500) + "</body></html>"

    _patch_transport(
        monkeypatch,
        lambda _request: httpx.Response(200, headers={"content-type": "text/html"}, text=html),
    )
    fetcher = PageFetcher(_settings(max_bytes=100))
    pages = await fetcher.fetch_pages([_PUBLIC_URL], javascript_enabled=False)

    assert pages[0].error is None
    assert len(pages[0].html) == 100


async def test_batch_is_capped_at_max_pages(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<html/>")

    _patch_transport(monkeypatch, handler)
    fetcher = PageFetcher(_settings(max_pages=2))
    pages = await fetcher.fetch_pages(
        [f"{_PUBLIC_URL}/{index}" for index in range(5)], javascript_enabled=False
    )

    assert len(pages) == 2


async def test_rendered_mode_checks_the_target_before_launching_a_browser():
    fetcher = PageFetcher(_settings())
    pages = await fetcher.fetch_pages(["http://127.0.0.1/"], javascript_enabled=True)

    assert isinstance(pages[0], FetchedPage)
    assert pages[0].error is not None
    assert "not a public address" in pages[0].error


def test_auth_config_rejects_credential_shaped_keys():
    with pytest.raises(PageFetchError, match="credential_id"):
        validate_auth_config({"credential_id": "abc"})
    with pytest.raises(PageFetchError, match="password"):
        validate_auth_config({"username": "u", "password": "p"})


def test_auth_config_accepts_references_only():
    validate_auth_config({"storage_state_path": "uploads/t/state.json"})
    validate_auth_config({"user_agent": "custom/1.0"})
    validate_auth_config(None)
