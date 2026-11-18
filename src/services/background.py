"""Background task dispatch for BDD job processing."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from redis.exceptions import RedisError
from w2v_common.auth.identity_context import (
    get_correlation_id_from_context,
    get_stage_from_context,
    set_correlation_context,
)

from src.cache.redis_client import get_redis_client
from src.database.session import get_db_session
from src.middleware.rate_limiter import concurrent_jobs_key, slot_ttl_seconds
from src.services.job_manager import JobManager, JobStateConflictError
from src.services.task_identity import TaskIdentity, bind_task_identity, capture_task_identity

if TYPE_CHECKING:
    from src.services.job_processor import JobProcessor

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task[None]] = set()


async def _acquire_concurrency_slot(tenant_id: str, job_id: str) -> bool:
    """Register ``job_id`` in the per-tenant running-jobs sorted set.

    This is the WRITE side of ``RateLimitSettings.concurrent_jobs_per_tenant``.
    Until it existed, the middleware read a key nothing ever wrote, so the cap
    was advertised and structurally unenforceable and a tenant could keep any
    number of metered WM/LLM generation jobs in flight.

    The member's SCORE is its expiry DEADLINE (``now + slot_ttl_seconds()``),
    not the start time. That is what makes the slot leak-proof: the admission
    script prunes members whose deadline has passed, so a task that is cancelled,
    evicted, or SIGKILLed before its ``finally`` runs releases its slot at its own
    deadline instead of holding it forever — and because the deadline is written
    ONCE here and never re-scored (this service has no heartbeat touching the
    member), nothing can push that backstop out while the job is wedged.

    Returns ``True`` if the slot was registered. Returns ``False`` only when the
    ``ZADD`` itself failed — the caller must then NOT release, and the job
    proceeds unmetered by the cap. Failing open there is deliberate and matches
    the middleware: a Redis blip must not kill an accepted job.

    ⚠️ The return value gates the ``finally`` ``ZREM`` in ``_process_job_bg``, so
    it must answer "is there a member to remove", NOT "did every Redis call
    succeed". The ``ZADD`` and the housekeeping ``EXPIRE`` are two round trips:
    with both under one handler, a connection reset BETWEEN them — a Redis
    restart/failover, the exact blip this is guarded for — reported ``False``
    for a slot that HAD been added, the release was skipped, and the member sat
    in the zset until its own deadline score pruned it. The admission script
    counts that member, so the tenant is short one slot for the whole TTL and a
    few blips zero out the default cap: every ``POST /execute`` 429s with
    "Concurrent job limit exceeded" while nothing is running.
    """
    try:
        redis_client = get_redis_client()
        client = redis_client._ensure_connected()
        ttl = slot_ttl_seconds()
        await client.zadd(concurrent_jobs_key(tenant_id), {job_id: time.time() + ttl})
    except OSError, RuntimeError, RedisError:
        logger.warning("Failed to acquire concurrency slot for job %s", job_id)
        return False

    # The member EXISTS from here on — every exit below must return True so the
    # caller's ``finally`` releases it.
    try:
        # Bound an idle tenant's key so it self-collects once every member has
        # expired. Generous vs. the member deadline so the key never disappears
        # out from under a live member.
        await client.expire(concurrent_jobs_key(tenant_id), ttl + 60)
    except OSError, RuntimeError, RedisError:
        # Housekeeping only: every member still carries its own deadline score and
        # the admission script prunes by it, so a missing key TTL costs nothing
        # but an empty zset lingering in Redis.
        logger.warning("Acquired concurrency slot for job %s but could not set the key TTL", job_id)
    return True


async def _release_concurrency_slot(tenant_id: str, job_id: str) -> None:
    """Remove ``job_id`` from the per-tenant running-jobs sorted set.

    Idempotent: ``ZREM`` of an absent/already-pruned member is a no-op, so a late
    release after the slot self-expired cannot free someone else's slot. Never
    raises — this runs in a ``finally`` on the shutdown/cancellation path, where
    an exception would mask the original one.
    """
    try:
        redis_client = get_redis_client()
        client = redis_client._ensure_connected()
        await client.zrem(concurrent_jobs_key(tenant_id), job_id)
    except OSError, RuntimeError, RedisError:
        logger.warning("Failed to release concurrency slot for job %s", job_id)


# ⛔ `extract_license_token` USED TO LIVE HERE and was DELETED (2026-09-13). It was a
# copy of w2v_common's PRE-0.74.0 rule — token read ONLY from `X-Tenant-Id: Bearer
# <tok>` — and it silently drifted when 0.74.0 widened the canonical rule to also
# accept an opaque `Authorization: Bearer <tok>` (the only shape the `license-authz`
# ext-authz gate forwards). Behind that gate the copy returned None, `dispatch_job`
# got license_token=None, and every background peer call fell back to trusted-network
# auth. Nothing failed; no test covered it.
#
# The replacement is `src.api.routes._helpers.get_license_token`, which reads the token
# AuthMiddleware already extracted and introspected off `request.state.license_token`.
# DO NOT re-add a header-parsing copy here: the canonical rule has an order
# (`X-Tenant-Id` first, `Authorization` only when not JWT-shaped) that a copy getting
# backwards re-attributes delegated calls to the service account's tenant.


async def _process_job_bg(
    processor: JobProcessor,
    job_id: str,
    tenant_id: str,
    license_token: str | None = None,
    correlation_id: str = "",
    stage: str = "",
    identity: TaskIdentity | None = None,
) -> None:
    """Run a BDD generation job in the background.

    Uses a fresh DB session (not request-scoped) and catches all errors
    to mark the job as failed rather than crashing silently. ``license_token``
    (request-scoped) is forwarded to ``process_job`` so peer-service calls
    authenticate as the CALLER rather than falling back to this service's
    trusted-network / service-account credential. The token itself may have
    arrived on either accepted shape — the legacy ``X-Tenant-Id: Bearer <tok>``
    or, behind the ``license-authz`` gate, an opaque ``Authorization: Bearer
    <tok>``; the routes read whichever one ``AuthMiddleware`` resolved.
    """
    # Bind the identity the ORIGINATING request authenticated as — tenant, its
    # provenance, run, w2v project and human actor — explicitly, from the snapshot
    # `dispatch_job` took in request scope. Two consumers depend on it:
    #
    # * RLS: w2v_common's session bind and the after_begin listener apply
    #   app.current_tenant from it on every transaction of this multi-commit job.
    # * P7 in-cell auth: `identity_event_hook` mints a fresh X-W2V-Cell-Token per
    #   outbound send from exactly these values, and ONLY for a vouched provenance.
    #   Before this was explicit the job re-seeded the tenant alone, so the actor
    #   was cleared (cell tokens went out with actor=null) and the run/project
    #   survived only by context inheritance.
    #
    # ⚠️ Provenance is load-bearing, not bookkeeping: without it every metered hop
    # fails closed ("provenance is 'unrecorded'", 2026-08-27 Gestalt outage) and no
    # cell token is minted. It is captured TOGETHER with its tenant, so it can never
    # follow the context onto a different tenant. `identity=None` (direct callers,
    # tests) binds the tenant with NO provenance — RLS only, nothing signed or
    # delegated. The licence below is the dual-send fallback; nothing here reads it.
    bind_task_identity(identity if identity is not None else TaskIdentity.unvouched(tenant_id))
    # Re-establish cost-correlation (run_id + DAG stage) so the WM LLM calls this
    # job makes propagate X-Correlation-Id/X-Stage → WM stamps them on the
    # emitted cost_event, tying BDD-generation spend back to the orchestrator run.
    set_correlation_context(correlation_id, stage)
    # ⚠️ ACQUIRE IMMEDIATELY BEFORE THE `try:` WHOSE `finally` RELEASES, with NOTHING
    # in between. This fleet lost 303 jobs and every free slot to exactly the opposite
    # shape: a slot acquired, then a statement that raised before the `try:` was
    # entered, so the `finally` never ran and the slot leaked. Anything you are tempted
    # to add here belongs INSIDE the `try`.
    slot_acquired = await _acquire_concurrency_slot(tenant_id, job_id)
    try:
        async with get_db_session() as db:
            await processor.process_job(db, job_id, tenant_id, license_token=license_token)
    except JobStateConflictError:
        logger.info("BDD job %s lost a state-transition race — another worker owns it", job_id)
    except asyncio.CancelledError:
        # ⚠️ MUST precede `except Exception`. CancelledError is a BaseException on
        # Python 3.8+, so the broad handler below can NEVER see it — before this
        # clause existed, a cancelled task (shutdown, task.cancel(), a client
        # disconnect that reaps the dispatch) left the row in `running` forever
        # and it stranded until the ~65-minute stuck-job reaper noticed.
        logger.warning("BDD job %s cancelled — marking failed before re-raising", job_id)
        try:
            async with get_db_session() as db:
                await JobManager.mark_failed(db, job_id, "cancelled")
        except JobStateConflictError:
            logger.info("BDD job %s already in terminal state — skip mark_failed", job_id)
        except Exception:
            logger.exception("Failed to mark cancelled job %s as failed", job_id)
        # ⛔ NEVER swallow cancellation: the event loop and `_shutdown_cleanup`-style
        # awaiters rely on it propagating, and a suppressed CancelledError turns a
        # graceful shutdown into a hang.
        raise
    except Exception as exc:
        logger.exception("BDD job %s failed", job_id)
        try:
            async with get_db_session() as db:
                await JobManager.mark_failed(db, job_id, str(exc))
        except JobStateConflictError:
            logger.info("BDD job %s already in terminal state — skip mark_failed", job_id)
        except Exception:
            logger.exception("Failed to mark job %s as failed", job_id)
    finally:
        # EVERY exit path — success, JobStateConflictError, cancellation (which
        # re-raises above), any other exception, and a mark_failed that itself
        # blew up — releases the slot. Without this the cap becomes a one-way
        # ratchet that permanently locks a tenant out after N jobs.
        if slot_acquired:
            await _release_concurrency_slot(tenant_id, job_id)


def dispatch_job(
    processor: JobProcessor,
    job_id: str,
    tenant_id: str,
    license_token: str | None = None,
) -> None:
    """Fire-and-forget a job processing background task.

    The task is kept in a module-level set to prevent garbage collection.
    Call it in REQUEST scope: the job's identity (tenant + provenance, run,
    project, actor) is captured here from ``AuthMiddleware``'s binding and
    re-bound explicitly at task start, which is what the job's in-cell calls
    mint their cell tokens from. ``license_token`` is the dual-send licence
    fallback only; the cell binding does not depend on it (on the cell lane it
    is ``None``).

    Raises ``MissingTenantError`` when no authenticated tenant is bound, or the
    bound one is not ``tenant_id`` — before any task exists.
    """
    # Snapshot identity and cost-correlation now, in request scope, where
    # AuthMiddleware populated them. The task does inherit them (`create_task`
    # copies the context), but `_process_job_bg` re-seeds those ContextVars, so the
    # snapshot is what survives that re-seed — and a missing tenant fails loud HERE
    # instead of after a 202, from inside a job whose calls would all 401.
    identity = capture_task_identity(tenant_id)
    task = asyncio.create_task(
        _process_job_bg(
            processor,
            job_id,
            tenant_id,
            license_token=license_token,
            correlation_id=get_correlation_id_from_context(),
            stage=get_stage_from_context(),
            identity=identity,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    logger.debug("Dispatched background task for job %s", job_id)


# ⚠️ Tuned against the k8s default 30s terminationGracePeriod: settle + cancel must
# both fit inside it, or the kubelet SIGKILLs us mid-rescue and we are back to the
# stranded-`pending` behaviour this drain exists to prevent.
DRAIN_SETTLE_SECONDS = 5.0
DRAIN_CANCEL_SECONDS = 10.0


async def drain_background_tasks(
    *,
    settle_seconds: float | None = None,
    cancel_seconds: float | None = None,
) -> None:
    """Settle, then cancel, the in-flight job tasks — while the DB engine is STILL UP.

    ⛔ THIS IS ORDERING-CRITICAL. ``_process_job_bg`` already catches
    ``asyncio.CancelledError`` and calls ``JobManager.mark_failed`` (which matches
    ``status IN ('pending','processing')``), so the rescue that turns a stranded row
    terminal exists — it just could never run. The lifespan cancelled ONLY the cleanup
    task and then went straight to ``_shutdown_database()``, disposing the engine with
    every job task still pending. Those tasks were then cancelled by the event loop at
    interpreter teardown, and their rescue opened a ``get_db_session()`` against an
    engine that was already gone, so it raised and the row stayed ``pending`` — the
    one state nothing reclaims.

    Draining HERE, before the engine is disposed, is what makes that existing rescue
    reachable. It cannot help on SIGKILL/eviction, which is why the ``pending`` arm of
    ``prune_stuck_bdd_jobs`` is the other half of this fix and not an alternative to it.

    Never raises: a shutdown path that propagates turns a graceful stop into a crash.
    """
    # Resolved at CALL time, not bound as a default: a module-level default is
    # captured once at import and a test (or an operator patching the module) could
    # never change it.
    settle = DRAIN_SETTLE_SECONDS if settle_seconds is None else settle_seconds
    cancel_wait = DRAIN_CANCEL_SECONDS if cancel_seconds is None else cancel_seconds

    tasks = {task for task in _background_tasks if not task.done()}
    if not tasks:
        return

    logger.info("Draining %d in-flight BDD job task(s)", len(tasks))
    # A short settle first, so a task that is milliseconds from finishing is allowed to
    # commit its real result instead of being cancelled into a spurious 'failed'.
    _done, pending = await asyncio.wait(tasks, timeout=settle)
    if not pending:
        logger.info("All in-flight BDD job tasks finished before cancellation")
        return

    logger.warning("Cancelling %d BDD job task(s) that did not finish draining", len(pending))
    for task in pending:
        task.cancel()
    # Await the cancellations so each task's CancelledError handler gets to run
    # mark_failed against the live engine. `wait` is bounded on purpose: a task that
    # swallows cancellation must not wedge shutdown past the grace period.
    _done, still_pending = await asyncio.wait(pending, timeout=cancel_wait)
    if still_pending:
        logger.error(
            "%d BDD job task(s) did not respond to cancellation within %.1fs — "
            "their rows will be reclaimed by the pending/stuck job sweep",
            len(still_pending),
            cancel_wait,
        )
