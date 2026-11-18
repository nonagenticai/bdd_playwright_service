"""Every job must close the MinIO client's aiohttp session.

Live 2026-10-18: each demo-video job ended in ``asyncio ERROR Unclosed client
session`` + ``Unclosed connector`` right after the package zip was streamed to
MinIO. Each ``JobProcessor`` builds its own ``StorageService``, whose
``miniopy_async.Minio`` lazily opens an ``aiohttp.ClientSession`` and never
closes it. These tests drive a REAL ``Minio`` client (no network: the session is
created by ``_ensure_session`` and never used) so they fail without the fix.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp_retry import RetryClient
from miniopy_async import Minio

from src.services.job_processor import JobProcessor
from src.services.storage_service import StorageService

MODULE = "src.services.job_processor"


def _real_minio_session(storage: StorageService) -> Any:
    """Attach a real Minio client with an OPEN aiohttp session; return the session."""
    client = Minio("127.0.0.1:1", access_key="k", secret_key="s", secure=False)
    client._ensure_session()
    storage._storage._minio_client = client
    # RetryClient wraps the aiohttp.ClientSession that aiohttp reports as unclosed.
    retry_client = client._session
    assert isinstance(retry_client, RetryClient)
    return retry_client._client


async def test_storage_aclose_closes_the_minio_aiohttp_session() -> None:
    storage = StorageService()
    session = _real_minio_session(storage)
    assert not session.closed

    await storage.aclose()

    assert session.closed
    # Idempotent, and a later use re-initialises instead of reusing a dead client.
    await storage.aclose()
    assert storage._initialized is False


async def test_storage_aclose_is_noop_without_a_minio_client() -> None:
    storage = StorageService()  # local backend in tests
    assert getattr(storage._storage, "_minio_client", None) is None

    await storage.aclose()

    assert storage._initialized is False


@pytest.fixture
def processor():
    with patch(f"{MODULE}.LLMClient", return_value=MagicMock(generate=AsyncMock())):
        yield JobProcessor()


async def test_process_job_closes_the_session_on_success(processor: JobProcessor) -> None:
    session = _real_minio_session(processor._storage)
    with patch.object(JobProcessor, "_run_job", new_callable=AsyncMock):
        await processor.process_job(AsyncMock(), "job-ok", "n8n-dev")
    assert session.closed


async def test_process_job_closes_the_session_on_failure(processor: JobProcessor) -> None:
    session = _real_minio_session(processor._storage)
    with (
        patch.object(
            JobProcessor, "_run_job", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await processor.process_job(AsyncMock(), "job-fail", "n8n-dev")
    assert session.closed


async def test_teardown_error_never_masks_the_job_outcome(processor: JobProcessor) -> None:
    with (
        patch.object(
            JobProcessor, "_run_job", new_callable=AsyncMock, side_effect=RuntimeError("job")
        ),
        patch.object(StorageService, "aclose", AsyncMock(side_effect=OSError("close"))),
        pytest.raises(RuntimeError, match="job"),
    ):
        await processor.process_job(AsyncMock(), "job-x", "n8n-dev")
