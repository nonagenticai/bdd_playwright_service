"""Unit tests for background job dispatch — cost-correlation threading."""

from __future__ import annotations

import contextlib
import uuid
from unittest.mock import AsyncMock, MagicMock

from src.services import background
from src.services.task_identity import TaskIdentity
from tests.conftest import TENANT_A


@contextlib.asynccontextmanager
async def _fake_db_session():
    yield MagicMock()


async def test_process_job_bg_reestablishes_correlation_context(monkeypatch):
    """The detached BDD job re-establishes cost-correlation (run_id + stage).

    Without this, the WM LLM calls the job makes (feature/step/narration
    generation) emit cost_events with correlation_id NULL and billing can't tie
    the spend back to the orchestrator run. dispatch_job snapshots correlation in
    request scope; _process_job_bg re-sets it via set_correlation_context().
    """
    from w2v_common.auth import (
        clear_identity_context,
        get_correlation_id_from_context,
        get_stage_from_context,
    )

    seen: dict[str, str] = {}

    async def fake_process_job(_db, _job_id, _tenant_id, license_token=None):
        seen["correlation_id"] = get_correlation_id_from_context()
        seen["stage"] = get_stage_from_context()

    processor = MagicMock()
    processor.process_job = AsyncMock(side_effect=fake_process_job)

    monkeypatch.setattr(background, "get_db_session", _fake_db_session)
    clear_identity_context()

    await background._process_job_bg(
        processor,
        "job-1",
        TENANT_A,
        license_token="tok",  # pragma: allowlist secret
        correlation_id="run-xyz",
        stage="bdd",
    )

    assert seen["correlation_id"] == "run-xyz"
    assert seen["stage"] == "bdd"


async def test_process_job_bg_reestablishes_tenant_provenance(monkeypatch):
    """The detached BDD job re-establishes tenant PROVENANCE, not just the tenant.

    Regression for the 2026-08-27 Gestalt calibration outage. `_process_job_bg`
    seeded `set_identity_context(tenant_id, None, None)` — correct for RLS, which
    only needs `app.current_tenant`, and complete for years. WS-4 then made
    provenance a second, independent requirement for any METERED outbound hop, and
    a detached asyncio task does not inherit the ContextVar `AuthMiddleware` set on
    the request. Provenance therefore arrived as `None`, which `build_outbound_identity`
    reports as `'unrecorded'` and refuses:

        refusing to delegate tenant <t> to metered destination <kag>:
        provenance is 'unrecorded', which is not vouched

    Every calibration render died ~16s in on POST kag/ingest/scrape/bulk/sync. The
    request itself was properly authenticated; only the background continuation was
    blind — and because the job returns 202 and fails later, the error surfaces on
    the CALLEE's URL and reads as a KAG fault. KAG was refusing correctly.
    (KAG was retired 2026-10-17; web inspection now calls webscraper directly,
    which is not metered. The job's LLM hops to worker_management still are.)

    Asserting the ContextVar (not a call-arg spy) is deliberate: the ContextVar is
    what the outbound hook actually reads, so this fails if the value is threaded
    but never bound.
    """
    from w2v_common.auth import clear_identity_context
    from w2v_common.auth.identity_context import get_tenant_source_from_context
    from w2v_common.tenant import TenantSource

    seen: dict[str, object] = {}

    async def fake_process_job(_db, _job_id, _tenant_id, license_token=None):
        seen["tenant_source"] = get_tenant_source_from_context()

    processor = MagicMock()
    processor.process_job = AsyncMock(side_effect=fake_process_job)

    monkeypatch.setattr(background, "get_db_session", _fake_db_session)
    clear_identity_context()

    await background._process_job_bg(
        processor,
        "job-1",
        TENANT_A,
        license_token="tok",  # pragma: allowlist secret
        identity=TaskIdentity(uuid.UUID(TENANT_A), TenantSource.LICENSE_TOKEN),
    )

    assert seen["tenant_source"] is TenantSource.LICENSE_TOKEN, (
        "provenance did not reach the job body; every metered outbound hop it makes "
        "will fail closed as 'unrecorded'"
    )


def test_dispatch_job_snapshots_provenance_from_request_scope(monkeypatch):
    """dispatch_job must snapshot provenance while still in request scope.

    Threading the parameter is only half the fix: if `dispatch_job` does not read
    `get_tenant_source_from_context()` at dispatch time, the detached task gets
    `None` and fails closed exactly as before. `dispatch_job` already snapshots
    correlation_id and stage for the same reason — provenance was simply missing
    from that list.
    """
    import asyncio

    from w2v_common.auth import clear_identity_context, set_identity_context
    from w2v_common.tenant import TenantSource

    captured: dict[str, object] = {}

    async def fake_bg(*_args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(background, "_process_job_bg", fake_bg)

    async def drive():
        clear_identity_context()
        set_identity_context(TENANT_A, None, None, tenant_source=TenantSource.LICENSE_TOKEN)
        background.dispatch_job(
            MagicMock(), "job-1", TENANT_A, license_token="tok"
        )  # pragma: allowlist secret
        # let the created task run
        await asyncio.sleep(0)
        clear_identity_context()

    asyncio.run(drive())

    identity = captured.get("identity")
    assert isinstance(identity, TaskIdentity)
    assert identity.tenant_source is TenantSource.LICENSE_TOKEN, (
        f"dispatch_job did not snapshot provenance from request scope: {captured!r}"
    )


async def test_process_job_bg_provenance_reaches_the_outbound_hook(monkeypatch):
    """The propagated provenance survives all the way to the OUTBOUND boundary.

    The two tests above stop at the ContextVar. This one drives the real
    ``identity_event_hook`` from inside the job body, against the real metered
    destination the job still calls (worker_management, every LLM step; the
    prod break was on KAG, now retired), on a service-credentialed request —
    i.e. the exact delegated-hop branch of ``_apply_outbound_tenant`` that raised
    ``UnvouchedTenantError``. The observable asserted is the header that
    ``X-Effective-Tenant-Id`` actually carries on the wire, not a mock call.
    """
    import httpx
    from w2v_common.auth import clear_identity_context
    from w2v_common.auth.identity_context import identity_event_hook
    from w2v_common.auth.outbound import EFFECTIVE_TENANT_HEADER, is_metered_destination
    from w2v_common.discovery import service_url
    from w2v_common.tenant import TenantSource

    wm_url = f"{service_url('worker_management').rstrip('/')}/workers/bdd/message"
    assert is_metered_destination(wm_url), "test target must be a metered peer"

    seen: dict[str, str] = {}

    async def fake_process_job(_db, _job_id, _tenant_id, license_token=None):
        request = httpx.Request(
            "POST",
            wm_url,
            # A baked-in service-account bearer is what makes this a DELEGATED
            # hop — the branch that fails closed on unvouched provenance.
            headers={"X-Tenant-Id": "Bearer sa-token"},  # pragma: allowlist secret
        )
        await identity_event_hook(request)
        seen["effective_tenant"] = request.headers.get(EFFECTIVE_TENANT_HEADER, "")

    processor = MagicMock()
    processor.process_job = AsyncMock(side_effect=fake_process_job)

    monkeypatch.setattr(background, "get_db_session", _fake_db_session)
    clear_identity_context()

    await background._process_job_bg(
        processor,
        "job-1",
        TENANT_A,
        license_token="tok",  # pragma: allowlist secret
        identity=TaskIdentity(uuid.UUID(TENANT_A), TenantSource.LICENSE_TOKEN),
    )

    assert seen["effective_tenant"] == str(TENANT_A), (
        "the delegated hop did not carry X-Effective-Tenant-Id; without vouched "
        "provenance the outbound hook raises UnvouchedTenantError instead"
    )


async def test_process_job_bg_cancellation_marks_failed_and_reraises(monkeypatch):
    """A CANCELLED job must be marked failed AND the cancellation must propagate.

    ``asyncio.CancelledError`` is a ``BaseException`` on Python 3.8+, so the
    ``except Exception as exc`` arm below it can never see it. Before the
    dedicated ``except asyncio.CancelledError`` clause, a cancelled task (pod
    shutdown, ``task.cancel()``) left ``bdd_jobs`` in ``running`` forever — the
    row stranded for ~65 minutes until the stuck-job reaper noticed.

    Both halves are asserted deliberately:
      * the row is marked failed (the bug being fixed), and
      * the ``CancelledError`` is RE-RAISED. Swallowing cancellation to "handle"
        it is the classic over-correction: it turns a graceful shutdown into a
        hang, because ``_shutdown_cleanup``-style awaiters never see the task
        finish as cancelled.
    """
    import asyncio

    import pytest
    from w2v_common.auth import clear_identity_context

    marked: list[tuple[str, str]] = []

    async def fake_mark_failed(_db, job_id, reason):
        marked.append((job_id, reason))

    async def cancelled_process_job(_db, _job_id, _tenant_id, license_token=None):
        raise asyncio.CancelledError

    processor = MagicMock()
    processor.process_job = AsyncMock(side_effect=cancelled_process_job)

    monkeypatch.setattr(background, "get_db_session", _fake_db_session)
    monkeypatch.setattr(background.JobManager, "mark_failed", fake_mark_failed)
    clear_identity_context()

    with pytest.raises(asyncio.CancelledError):
        await background._process_job_bg(
            processor,
            "job-cancelled",
            TENANT_A,
            license_token="tok",  # pragma: allowlist secret
        )

    assert marked, (
        "a cancelled job was not marked failed — the row strands in 'running' "
        "until the ~65-minute stuck-job reaper picks it up"
    )
    assert marked[0][0] == "job-cancelled"


async def test_webscraper_hop_is_not_metered_and_never_fails_closed(monkeypatch):
    """Inverse of the KAG-era guard: webscraper is NOT a metered destination.

    Web inspection used to go through KAG, a metered peer, so a provenance gap in
    the detached job failed the scrape closed. webscraper (which replaced the KAG
    hop) is not metered: even with UNRECORDED provenance the outbound hook lets a
    service-credentialed request through and stamps no delegated tenant.
    """
    import httpx
    from w2v_common.auth import clear_identity_context, set_identity_context
    from w2v_common.auth.identity_context import identity_event_hook
    from w2v_common.auth.outbound import EFFECTIVE_TENANT_HEADER, is_metered_destination
    from w2v_common.discovery import service_url

    webscraper_url = f"{service_url('webscraper').rstrip('/')}/api/scrape/bulk"
    assert not is_metered_destination(webscraper_url), "webscraper must not be metered"

    clear_identity_context()
    set_identity_context(TENANT_A, None, None)  # provenance unrecorded
    try:
        request = httpx.Request(
            "POST",
            webscraper_url,
            headers={"X-Tenant-Id": "Bearer sa-token"},  # pragma: allowlist secret
        )
        await identity_event_hook(request)  # must not raise UnvouchedTenantError
    finally:
        clear_identity_context()

    assert EFFECTIVE_TENANT_HEADER not in request.headers
    assert request.headers["X-Tenant-Id"] == "Bearer sa-token"  # pragma: allowlist secret
