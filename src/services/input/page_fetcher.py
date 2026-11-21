"""Self-contained page fetcher for the ``web_url`` inspection input.

Replaces the retired ``webscraper_service`` hop. Two modes:

* **rendered** (default): a headless Chromium via Playwright loads the page so
  client-rendered SPAs hydrate before the DOM is snapshotted.
* **static**: a plain ``httpx`` GET, for callers that explicitly ask for no JS
  (``javascript_enabled=False``) and for tests that must not start a browser.

Both modes return :class:`FetchedPage` objects whose attribute names match the
subset of ``w2v_common.clients.webscraper.ScrapeResponse`` that
``job_processor._page_from_scrape`` reads (``url``/``title``/``html``/``text``/
``links``/``images[.src]``/``error``), so the downstream structure extraction is
unchanged.

Credentials: the job row persists the request ``auth_config`` verbatim, so this
fetcher refuses anything secret-shaped inline. Login state is supplied as a
Playwright storage-state object the job processor reads from the tenant's
upload prefix via ``storage_state_path``. Only ``storage_state_path`` and
``user_agent`` are accepted as ``auth_config`` keys.

SSRF: every navigation and redirect is resolved and checked; loopback,
link-local, multicast, reserved and unspecified addresses are always refused,
and RFC1918/ULA addresses are refused unless ``BDD_FETCH_ALLOW_PRIVATE=true``.
In rendered mode the same guard runs as a route handler, so a page cannot pull
private-address subresources either. DNS is resolved at check time; a residual
rebinding window remains and is accepted for an in-cluster internal tool.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import ipaddress
import logging
import re
import socket
from collections.abc import Callable, Mapping
from html.parser import HTMLParser
from typing import TYPE_CHECKING, cast
from urllib.parse import urljoin, urlparse

import httpx
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from src.config import FetchSettings, get_settings

if TYPE_CHECKING:
    from playwright.async_api import (
        Browser,
        Playwright,
        PlaywrightContextManager,
        Route,
        StorageState,
    )

logger = logging.getLogger(__name__)

_MAX_REDIRECTS = 5
_MAX_TEXT_CHARS = 200_000
_MAX_LINKS = 200
_MAX_IMAGES = 100
_FETCH_AUTH_KEYS = frozenset({"storage_state_path", "user_agent"})
_BLOCKED_TAGS = frozenset({"script", "style", "noscript", "template"})
_WHITESPACE = re.compile(r"\s+")


class PageFetchError(RuntimeError):
    """A page could not be fetched, or the request was refused by policy."""


@dataclasses.dataclass(slots=True)
class FetchedImage:
    """An image discovered on a page (only the ``src`` is kept)."""

    src: str


@dataclasses.dataclass(slots=True)
class FetchedPage:
    """One fetched page, shaped for ``job_processor._page_from_scrape``."""

    url: str
    title: str = ""
    html: str = ""
    text: str = ""
    links: list[str] = dataclasses.field(default_factory=list)
    images: list[FetchedImage] = dataclasses.field(default_factory=list)
    error: str | None = None


def validate_auth_config(auth_config: Mapping[str, object] | None) -> None:
    """Refuse credential-shaped keys that would be persisted on the job row."""
    if not auth_config:
        return
    unknown = sorted(set(auth_config) - _FETCH_AUTH_KEYS)
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise PageFetchError(
            f"auth_config key(s) {names} are not supported by the in-service page fetcher. "
            "Only 'storage_state_path' (a Playwright storage-state JSON in this tenant's "
            "uploads) and 'user_agent' are accepted; inline credentials would be stored on "
            "the job row in plaintext."
        )


class TargetGuard:
    """Resolve-then-check guard against private/loopback targets."""

    def __init__(self, *, allow_private: bool) -> None:
        self._allow_private = allow_private
        self._verdicts: dict[str, str | None] = {}

    async def check(self, url: str) -> None:
        """Raise :class:`PageFetchError` when ``url`` must not be fetched."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise PageFetchError(f"refused {url!r}: only http(s) URLs are fetched")
        host = parsed.hostname
        if not host:
            raise PageFetchError(f"refused {url!r}: no host")
        error = await self._verdict(host)
        if error is not None:
            raise PageFetchError(f"refused {url!r}: {error}")

    async def _verdict(self, host: str) -> str | None:
        if host in self._verdicts:
            return self._verdicts[host]
        try:
            infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
        except OSError as exc:
            raise PageFetchError(f"cannot resolve {host!r}: {exc}") from exc
        verdict: str | None = None
        for info in infos:
            address = ipaddress.ip_address(str(info[4][0]))
            if (
                address.is_loopback
                or address.is_link_local
                or address.is_multicast
                or address.is_reserved
                or address.is_unspecified
            ):
                verdict = f"{address} is not a public address"
                break
            if address.is_private and not self._allow_private:
                verdict = (
                    f"{address} is a private address "
                    "(set BDD_FETCH_ALLOW_PRIVATE=true to allow internal targets)"
                )
                break
        self._verdicts[host] = verdict
        return verdict


class _PageScan(HTMLParser):
    """Minimal static-HTML scan for title, text, links and images."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self.images: list[str] = []
        self._in_title = False
        self._blocked_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag in _BLOCKED_TAGS:
            self._blocked_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "a" and values.get("href"):
            self.links.append(str(values["href"]))
        elif tag == "img" and values.get("src"):
            self.images.append(str(values["src"]))

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCKED_TAGS and self._blocked_depth:
            self._blocked_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        elif not self._blocked_depth:
            self.text_parts.append(data)


class PageFetcher:
    """Fetch inspection pages in-process (Chromium, or static httpx)."""

    def __init__(self, settings: FetchSettings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings().fetch
        self._guard = TargetGuard(allow_private=self._settings.allow_private)
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._browser_lock = asyncio.Lock()

    async def fetch_pages(
        self,
        urls: list[str],
        *,
        javascript_enabled: bool = True,
        auth_config: Mapping[str, object] | None = None,
        storage_state: Mapping[str, object] | None = None,
    ) -> list[FetchedPage]:
        """Fetch each URL, returning one page per URL (errors carried on the page)."""
        validate_auth_config(auth_config)
        user_agent = str((auth_config or {}).get("user_agent") or self._settings.user_agent)
        pages: list[FetchedPage] = []
        for url in list(urls)[: self._settings.max_pages]:
            try:
                if javascript_enabled:
                    page = await self._fetch_rendered(
                        url, user_agent=user_agent, storage_state=storage_state
                    )
                else:
                    page = await self._fetch_static(url, user_agent=user_agent)
            except PageFetchError as exc:
                page = FetchedPage(url=url, error=str(exc))
            except (PlaywrightError, httpx.HTTPError, OSError, ValueError) as exc:
                page = FetchedPage(url=url, error=str(exc))
            pages.append(page)
        return pages

    async def close(self) -> None:
        """Close the shared browser and its Playwright driver."""
        async with self._browser_lock:
            browser, self._browser = self._browser, None
            playwright, self._playwright = self._playwright, None
        with contextlib.suppress(PlaywrightError):
            if browser is not None:
                await browser.close()
        with contextlib.suppress(PlaywrightError):
            if playwright is not None:
                await playwright.stop()

    # -- rendered mode --------------------------------------------------------

    async def _ensure_browser(
        self, start_playwright: Callable[[], PlaywrightContextManager]
    ) -> Browser:
        async with self._browser_lock:
            if self._browser is None or not self._browser.is_connected():
                if self._playwright is None:
                    self._playwright = await start_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    chromium_sandbox=False,
                    args=["--disable-dev-shm-usage"],
                )
            return self._browser

    async def _fetch_rendered(
        self,
        url: str,
        *,
        user_agent: str,
        storage_state: Mapping[str, object] | None,
    ) -> FetchedPage:
        await self._guard.check(url)
        browser = await self._ensure_browser(async_playwright)
        context = await browser.new_context(
            user_agent=user_agent,
            storage_state=cast("StorageState | None", storage_state or None),
            viewport={"width": 1280, "height": 720},
        )

        async def guard_route(route: Route) -> None:
            try:
                await self._guard.check(route.request.url)
            except PageFetchError:
                with contextlib.suppress(PlaywrightError):
                    await route.abort()
                return
            with contextlib.suppress(PlaywrightError):
                await route.continue_()

        await context.route("**/*", guard_route)
        try:
            page = await context.new_page()
            try:
                response = await page.goto(
                    url,
                    wait_until="load",
                    timeout=self._settings.timeout_seconds * 1000,
                )
                if response is not None and response.status >= 400:
                    raise PageFetchError(f"HTTP {response.status} for {url}")
                await page.wait_for_timeout(self._settings.post_load_delay_seconds * 1000)
                final_url = page.url
                await self._guard.check(final_url)
                html = await page.content()
                title = await page.title()
                try:
                    text = await page.inner_text("body")
                except PlaywrightError:
                    text = ""
                links = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
                images = await page.eval_on_selector_all("img[src]", "els => els.map(e => e.src)")
            except PlaywrightError as exc:
                raise PageFetchError(f"navigation failed for {url}: {exc}") from exc
        finally:
            with contextlib.suppress(PlaywrightError):
                await context.close()

        return FetchedPage(
            url=final_url,
            title=title,
            html=self._cap_html(html, final_url),
            text=self._cap_text(text),
            links=_clean_refs(links, _MAX_LINKS),
            images=[FetchedImage(src=src) for src in _clean_refs(images, _MAX_IMAGES)],
        )

    # -- static mode ----------------------------------------------------------

    async def _fetch_static(self, url: str, *, user_agent: str) -> FetchedPage:
        headers = {"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"}
        current = url
        html = ""
        async with httpx.AsyncClient(
            timeout=self._settings.timeout_seconds,
            follow_redirects=False,
            headers=headers,
        ) as client:
            for _ in range(_MAX_REDIRECTS + 1):
                await self._guard.check(current)
                response = await client.get(current)
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise PageFetchError(f"redirect without Location from {current}")
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    raise PageFetchError(f"HTTP {response.status_code} for {current}")
                content_type = response.headers.get("content-type", "")
                if "html" not in content_type.lower():
                    raise PageFetchError(
                        f"not an HTML page ({content_type or 'no content-type'}) at {current}"
                    )
                html = response.text
                break
            else:
                raise PageFetchError(f"too many redirects for {url}")

        scan = _PageScan()
        scan.feed(html)
        scan.close()
        return FetchedPage(
            url=current,
            title=_WHITESPACE.sub(" ", "".join(scan.title_parts)).strip(),
            html=self._cap_html(html, current),
            text=self._cap_text(" ".join(scan.text_parts)),
            links=_clean_refs([urljoin(current, href) for href in scan.links], _MAX_LINKS),
            images=[
                FetchedImage(src=src)
                for src in _clean_refs([urljoin(current, src) for src in scan.images], _MAX_IMAGES)
            ],
        )

    # -- helpers --------------------------------------------------------------

    def _cap_html(self, html: str, url: str) -> str:
        if len(html.encode()) <= self._settings.max_bytes:
            return html
        capped = html[: self._settings.max_bytes]
        logger.warning(
            "page %s exceeds %d bytes; truncated before structure extraction",
            url,
            self._settings.max_bytes,
        )
        return capped

    @staticmethod
    def _cap_text(text: str) -> str:
        return _WHITESPACE.sub(" ", text).strip()[:_MAX_TEXT_CHARS]


def _clean_refs(values: list[str], limit: int) -> list[str]:
    """Absolute http(s) refs, de-duplicated in order, capped at ``limit``."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.startswith(("http://", "https://")):
            continue
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
        if len(cleaned) >= limit:
            break
    return cleaned
