"""Redis-based sliding window + concurrent-job rate limiter.

A single Lua script evaluates the sliding-window count, the concurrent-job
count, and (on success) writes the new window entry. Ported from
``frontend_test_runner_service``, which fixed both of these defects first;
only the key prefixes and the job-lifecycle hook points differ.

Two defects this module previously carried:

**D1 — the concurrent cap was structurally dead.** ``dispatch`` read the
integer key ``bdd_playwright:concurrent:{tenant_id}``, and NOTHING in this
service ever wrote it. ``client.get()` always returned ``None``, so
``if concurrent_count and ...`` could never fire: ``concurrent_jobs_per_tenant``
was advertised and unenforceable, and a tenant could hold unbounded metered
WM/LLM generation jobs in flight. The counter is now a per-tenant SORTED SET
of running ``job_id``s (``bdd_playwright:concurrent:{tenant_id}:jobs``), which
``services.background._process_job_bg`` ``ZADD``s on start and ``ZREM``s in a
``finally``.

**D2 — the sliding window was check-then-act.** ``ZREMRANGEBYSCORE`` → ``ZCARD``
→ ``await`` → ``ZADD`` let N concurrent POSTs at 29/30 all observe 29 and all
admit. The whole admission decision now happens inside one ``EVAL``, which
Redis runs atomically. The window member is also the ZSET score rendered with
enough precision plus a per-call unique suffix, so two requests landing on the
same clock tick can no longer collapse into a single member and undercount.

Slot expiry, and why it cannot be refreshed away
------------------------------------------------
Each running job's member is scored by an expiry DEADLINE
(``now + stuck_job_timeout_minutes * 60``), and the admission script prunes
members whose deadline has passed (``ZREMRANGEBYSCORE ... now``) BEFORE counting.
So a job whose task is cancelled, whose pod is evicted, or which is SIGKILLed
before its ``finally`` runs cannot hold a slot forever — its member self-expires
at its own deadline, without ever pruning a live job's slot.

The deadline is computed ONCE, at acquire, and is never re-scored: this service
has no heartbeat that touches the member. That is deliberate. A sibling's TTL
backstop failed to save it precisely because a heartbeat kept refreshing the
token, so the "backstop" could be pushed out indefinitely by the very job that
was stuck. Here the only way a member's deadline moves is for the job to end and
a NEW job to be registered.

``cleanup.stuck_job_timeout_minutes`` (default 60) is the grounding: it is this
service's own declaration of when a still-``processing`` job is considered dead
and is reaped. A slot must not outlive the row it represents, and must not
expire while the job is still legitimately running — the same number satisfies
both, and it moves with the operator's setting instead of a second magic number.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Final, cast

from fastapi import HTTPException, Request
from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.cache.redis_client import get_redis_client
from src.config import get_settings

logger = logging.getLogger(__name__)


# Lua script: atomic sliding-window + concurrent-cap admission control.
#
# KEYS[1] = window sorted-set key
# KEYS[2] = concurrent running-jobs sorted-set key (member=job_id, score=deadline)
# ARGV[1] = now (float seconds)
# ARGV[2] = window seconds (e.g. 60)
# ARGV[3] = rpm limit
# ARGV[4] = concurrent limit
# ARGV[5] = window TTL seconds (e.g. 120)
# ARGV[6] = unique window member for THIS request
#
# Returns:
#   {"ok",         new_count}          -> admit
#   {"rate",       current_count}      -> 429 sliding-window
#   {"concurrent", current_concurrent} -> 429 concurrent-cap
_ADMIT_SCRIPT: Final[str] = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local rpm_limit = tonumber(ARGV[3])
local concurrent_limit = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])
local member = ARGV[6]

redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, now - window)
local count = redis.call('ZCARD', KEYS[1])
if count >= rpm_limit then
    return {'rate', count}
end

-- Prune running-job members whose expiry deadline has passed (a job that was
-- cancelled/evicted/killed and never ZREM'd its slot), then count the
-- survivors. A live job's deadline is in the future, so it is never pruned.
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now)
local concurrent = redis.call('ZCARD', KEYS[2])
if concurrent >= concurrent_limit then
    return {'concurrent', concurrent}
end

redis.call('ZADD', KEYS[1], now, member)
redis.call('EXPIRE', KEYS[1], ttl)
return {'ok', count + 1}
"""


# Sliding-window length. ``requests_per_minute`` is per THIS many seconds.
_WINDOW_SECONDS: Final[int] = 60
# TTL on the window key — 2x the window, so an idle tenant's key self-collects.
_WINDOW_TTL_SECONDS: Final[int] = 120


def concurrent_jobs_key(tenant_id: str) -> str:
    """Redis key for the per-tenant running-jobs sorted set.

    Member = ``job_id``, score = the slot's expiry deadline.

    A NEW key suffix (``:jobs``) — distinct from the retired, never-written
    integer key ``bdd_playwright:concurrent:{tenant_id}`` — so a rolling deploy
    can never find a string where a zset is expected (``WRONGTYPE``). The old
    key was never written by anything, so nothing is stranded by abandoning it.
    """
    return f"bdd_playwright:concurrent:{tenant_id}:jobs"


def window_key(tenant_id: str) -> str:
    """Redis key for the per-tenant sliding-window sorted set."""
    return f"bdd_playwright:ratelimit:{tenant_id}:window"


def slot_ttl_seconds() -> int:
    """Lifetime of a concurrency slot, in seconds.

    Grounded in ``cleanup.stuck_job_timeout_minutes`` — the point at which this
    service itself declares a still-``processing`` job dead. Floored at 60s so a
    misconfigured 0 cannot make every slot expire instantly and re-open the
    unbounded-spend hole this cap exists to close.
    """
    return max(60, get_settings().cleanup.stuck_job_timeout_minutes * 60)


def window_member(now: float) -> str:
    """Unique member for this request's sliding-window entry.

    The retired implementation used ``str(now)`` alone: two requests served on
    the same clock tick produced the SAME ZSET member, the second ``ZADD``
    overwrote the first rather than adding to it, and the window undercounted.
    A random suffix makes every admitted request its own member; the SCORE stays
    ``now``, which is what ``ZREMRANGEBYSCORE`` ages out.
    """
    return f"{now!r}:{uuid.uuid7()}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-tenant sliding window + concurrent-job limiter using Redis."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Enforce per-tenant rate limits on BDD job creation requests."""
        settings = get_settings()

        # Only rate-limit POST requests to BDD endpoints
        if request.method != "POST" or not request.url.path.startswith("/api/bdd/"):
            return await call_next(request)

        tenant_id = getattr(request.state, "tenant_id", "anonymous")
        requests_per_minute = settings.rate_limit.requests_per_minute
        concurrent_limit = settings.rate_limit.concurrent_jobs_per_tenant

        now = time.time()
        try:
            redis_client = get_redis_client()

            verdict = await cast(
                Any,
                redis_client.eval_script(
                    _ADMIT_SCRIPT,
                    keys=[window_key(tenant_id), concurrent_jobs_key(tenant_id)],
                    args=[
                        str(now),
                        str(_WINDOW_SECONDS),
                        str(requests_per_minute),
                        str(concurrent_limit),
                        str(_WINDOW_TTL_SECONDS),
                        window_member(now),
                    ],
                ),
            )
        except OSError, RuntimeError, RedisError:
            # Fail OPEN on ANY redis error (connection/script/WRONGTYPE/OOM):
            # redis-py raises redis.exceptions.RedisError
            # (ConnectionError/TimeoutError, MRO ... -> RedisError -> Exception),
            # which is NOT an OSError/RuntimeError. Catch the RedisError base so a
            # Redis outage allows the request through instead of 500-ing every
            # POST /api/bdd/*. OSError/RuntimeError kept as belt-and-suspenders.
            logger.warning("Rate limiter failed (Redis unavailable), allowing request")
            return await call_next(request)

        # verdict = [tag, value]; redis-py decodes bytes when decode_responses=True,
        # but we accept both for safety.
        tag = verdict[0]
        if isinstance(tag, bytes):
            tag = tag.decode()

        if tag == "rate":
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {requests_per_minute} requests per minute",
            )
        if tag == "concurrent":
            raise HTTPException(
                status_code=429,
                detail=f"Concurrent job limit exceeded: {concurrent_limit} jobs",
            )

        return await call_next(request)
