"""``html_snapshot`` input: caller-captured pages parsed like a live fetch.

The snapshot is the deterministic fallback for authenticated or VPN-only apps:
the caller captures ``{url, html, ...}`` entries with Playwright and uploads the
JSON under their tenant prefix. Structure comes from the same
``extract_structure`` the live path uses, so the generators see identical pages.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

TENANT = "0199a1b2-0000-7000-8000-00000000000a"
SNAPSHOT = "uploads/t/snapshot.json"


@pytest.fixture
def processor():
    settings = MagicMock()
    settings.downstream.license_token = "test-token"
    settings.narration.words_per_minute = 150
    with (
        patch("src.services.job_processor.get_settings", return_value=settings),
        patch("src.services.job_processor.LLMClient"),
        patch("src.services.job_processor.FeatureGenerator"),
        patch("src.services.job_processor.StepDefGenerator"),
        patch("src.services.job_processor.StorageService"),
        patch("src.services.job_processor.PackageBuilder"),
        patch("src.services.job_processor.NarrationHumanizer"),
    ):
        from src.services.job_processor import JobProcessor

        return JobProcessor()


def _snapshot(*entries: dict[str, object]) -> str:
    return json.dumps(list(entries))


async def _resolve(processor, path: str = SNAPSHOT, tenant_id: str | None = TENANT):
    return await processor._resolve_input(
        input_type="html_snapshot",
        target_url="",
        app_path="",
        config={"html_snapshot_path": path},
        tenant_id=tenant_id,
    )


async def test_snapshot_pages_are_structured_like_a_fetch(processor):
    """Headings/forms/elements are derived from the captured DOM, not invented."""
    html = (
        "<html><head><title>Boeking</title></head><body>"
        "<h1>Afspraak maken</h1>"
        "<form><label>Naam</label><input name='naam'/>"
        "<button type='submit'>Verstuur</button></form>"
        "</body></html>"
    )
    processor._storage.retrieve_text = AsyncMock(
        return_value=_snapshot(
            {"url": "https://app.test/boeking", "html": html, "title": "Boeking"}
        )
    )

    crawl = await processor._resolve_html_snapshot_input(
        config={"html_snapshot_path": SNAPSHOT}, tenant_id=TENANT
    )

    pages = crawl["pages"]
    assert len(pages) == 1
    page = pages[0]
    assert page["url"] == "https://app.test/boeking"
    assert page["title"] == "Boeking"
    assert page["has_forms"] is True
    assert any("Afspraak maken" in heading for heading in page["headings"])
    assert page["text_content"] == ""
    processor._storage.retrieve_text.assert_awaited_once_with(SNAPSHOT, tenant_id=TENANT)


async def test_snapshot_requires_a_tenant(processor):
    with pytest.raises(RuntimeError, match="requires a tenant identity"):
        await _resolve(processor, tenant_id=None)


async def test_snapshot_absent_reference_fails_loudly(processor):

    processor._storage.retrieve_text = AsyncMock(return_value=None)
    with pytest.raises(RuntimeError, match="absent, empty, or outside this tenant"):
        await _resolve(processor)


async def test_snapshot_invalid_json_fails_loudly(processor):

    processor._storage.retrieve_text = AsyncMock(return_value="{not json")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        await _resolve(processor)


async def test_snapshot_without_usable_pages_fails_loudly(processor):

    processor._storage.retrieve_text = AsyncMock(
        return_value=_snapshot({"url": "", "html": "<html/>"}, {"html": "<html/>"})
    )
    with pytest.raises(RuntimeError, match="no usable pages"):
        await _resolve(processor)


async def test_snapshot_caps_pages(processor):

    entries = [
        {"url": f"https://app.test/{index}", "html": "<html><body>x</body></html>"}
        for index in range(25)
    ]
    processor._storage.retrieve_text = AsyncMock(return_value=json.dumps(entries))

    crawl = await processor._resolve_html_snapshot_input(
        config={"html_snapshot_path": SNAPSHOT}, tenant_id=TENANT
    )

    assert len(crawl["pages"]) == 20
    assert crawl["base_url"] == "https://app.test/0"
