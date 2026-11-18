"""Health check endpoints for bdd_playwright_service."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import redis.asyncio as redis_lib
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.applications import Starlette

from src.cache.redis_client import get_redis_client
from src.config import get_settings
from src.database.connection import get_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# The SQLAlchemy exception tree below is NOT optional and must never be
# "simplified" away. ``psycopg.OperationalError`` reaches us re-wrapped as
# ``sqlalchemy.exc.OperationalError``, and none of those inherit from the
# builtins whose names they shadow::
#
#     sqlalchemy.exc.OperationalError.__mro__
#         -> DatabaseError -> DBAPIError -> StatementError -> SQLAlchemyError
#            -> HasDescriptionCode -> Exception
#     issubclass(sqlalchemy.exc.OperationalError, OSError | RuntimeError) -> False
#
# So a builtins-only ``except (OSError, TimeoutError, RuntimeError)`` lets every
# REAL database fault escape these probes and FastAPI turns it into a 500. On
# /health/ready (and its /readyz alias) that breaks the Kubernetes readiness
# contract: a DB fault must be a 503 ("not ready"), never a 500 ("app broken").
#
# Probe failure = the dependency is down, not the process. The builtins stay in
# the tuple because a socket-level error, a cancelled ``asyncio.timeout`` (which
# raises the builtin ``TimeoutError`` since 3.11), or a driver misconfiguration
# can still surface as one.
_DB_PROBE_ERRORS = (SQLAlchemyError, TimeoutError, OSError, RuntimeError)

# Upper bound on the readiness DB round trip. Must stay well under the kubelet
# readinessProbe timeoutSeconds so a hung database FAILS the probe rather than
# stalling it.
_READY_DB_TIMEOUT = 5.0  # seconds

# Upper bound on the storage reachability probe. A hung object store must make
# the probe REPORT, not stall the whole /health response behind it.
_STORAGE_PROBE_TIMEOUT = 3.0  # seconds

_health_cache: dict[str, Any] = {}
_health_cache_ts: float = 0.0
_HEALTH_CACHE_TTL = 5.0
_health_lock = asyncio.Lock()

# Simple module-level counter for cache-stampede-avoided events
# (concurrent cold-cache callers that awaited a single in-flight fill).
# Exposed as a function for tests; no Prometheus dependency yet.
_health_cache_stampede_avoided = 0


def get_health_cache_stampede_avoided() -> int:
    """Return number of cache-stampede events avoided (single-flight)."""
    return _health_cache_stampede_avoided


def _cache_fresh(now: float) -> bool:
    return bool(_health_cache) and (now - _health_cache_ts) < _HEALTH_CACHE_TTL


async def _probe_storage(app: Starlette) -> str:
    """Probe the storage backend the service is ACTUALLY configured to use.

    ⚠️ This used to be an unconditional ``Path(storage.local_path).exists()`` +
    ``os.access(W_OK)`` with no backend branch. Prod runs
    ``STORAGE_BACKEND=minio`` (bucket ``w2v-bdd-playwright``), where
    ``local_path`` is nothing but the pod-local scratch dir the package builder
    assembles into. That directory exists and is writable on a pod that cannot
    reach MinIO at all, so the probe reported ``"storage": "healthy"`` while
    every job died at its first store write — and the health cache then served
    that wrong answer for the rest of its TTL.

    On ``minio`` we therefore stat the real bucket, bounded by
    ``_STORAGE_PROBE_TIMEOUT``. When there is nothing to stat with — the app has
    no initialized storage service (no lifespan, or startup did not get that
    far) — we return ``"unknown"``, NOT ``"healthy"``: an unmeasured dependency
    is unmeasured. ``"unknown"`` is not ``"healthy"``, so the aggregate status
    degrades and the gap is visible in the body instead of being papered over.
    """
    # ``get_settings`` stays INSIDE the guard: a config read that fails is a
    # storage answer of "unhealthy", not a 500 out of the whole probe.
    try:
        settings = get_settings()
        backend = settings.storage.backend

        if backend != "minio":
            # Local backend: the scratch root IS the store, so the original
            # filesystem probe is the correct one here.
            storage_path = Path(settings.storage.local_path)
            if not storage_path.exists():
                return "path not found"
            if not os.access(str(storage_path), os.W_OK):
                return "not writable"
            return "healthy"
    except (OSError, PermissionError) as e:
        return f"unhealthy: {e}"

    service = getattr(getattr(app, "state", None), "storage_service", None)
    if service is None:
        return "unknown: storage service not initialized"

    try:
        async with asyncio.timeout(_STORAGE_PROBE_TIMEOUT):
            reachable = await service.check_health()
    except (TimeoutError, OSError, RuntimeError) as e:
        return f"unhealthy: {e}"
    if not reachable:
        return f"unhealthy: bucket {settings.storage.minio_bucket} unreachable"
    return "healthy"


@router.get(
    "/health",
    response_model=None,
    operation_id="health_check",
    summary="Combined health probe (DB, Redis, storage)",
)
async def health_check(request: Request) -> dict[str, Any] | JSONResponse:
    """Combined health check for all components.

    Takes ``request`` only to reach ``app.state.storage_service`` — the storage
    client the lifespan already initialized and the job path actually writes
    through. Probing THAT object is what makes the storage line honest.
    """
    global _health_cache, _health_cache_ts, _health_cache_stampede_avoided
    now = time.monotonic()
    if _cache_fresh(now):
        return _health_cache

    # Single-flight: only one concurrent caller fills the cache; others await.
    async with _health_lock:
        # Double-check: a peer may have populated while we awaited the lock.
        now = time.monotonic()
        if _cache_fresh(now):
            _health_cache_stampede_avoided += 1
            return _health_cache

        try:
            db_status = "healthy"
            try:
                engine = await get_engine()
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
            # Same narrowness bug as readiness had: OperationalError/InterfaceError
            # alone missed every other SQLAlchemy fault (DisconnectionError,
            # DBAPIError, TimeoutError on pool checkout). SQLAlchemyError is the
            # common ancestor. OSError/RuntimeError deliberately stay OUT of this
            # inner tuple -- they are the aggregate handler's 503 path below.
            except (SQLAlchemyError, TimeoutError) as e:
                db_status = f"unhealthy: {e}"

            redis_status = "healthy"
            try:
                redis_client = get_redis_client()
                if redis_client:
                    await redis_client.ping()
                else:
                    redis_status = "not configured"
            except redis_lib.RedisError as e:
                redis_status = f"unhealthy: {e}"

            storage_status = await _probe_storage(request.app)

            overall = (
                "healthy"
                if all(s == "healthy" for s in [db_status, redis_status, storage_status])
                else "degraded"
            )

            result = {
                "status": overall,
                "components": {
                    "database": db_status,
                    "redis": redis_status,
                    "storage": storage_status,
                    "api": "healthy",
                },
            }
            _health_cache = result
            _health_cache_ts = now
            return result
        except (OSError, RuntimeError) as e:
            return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


@router.get(
    "/health/live",
    response_model=None,
    operation_id="health_live",
    summary="Kubernetes liveness probe",
)
async def health_live() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "ok"}


@router.get("/healthz", operation_id="health.live")
async def healthz() -> dict[str, str]:
    """Modern K8s liveness probe path (alias of /health/live)."""
    return {"status": "ok"}


@router.get(
    "/health/ready",
    response_model=None,
    operation_id="health_ready",
    summary="Kubernetes readiness probe (DB connectivity)",
)
async def health_ready() -> dict[str, Any] | JSONResponse:
    """Kubernetes readiness probe -- checks database connectivity.

    Readiness must be decided by a REAL round trip. SQLAlchemy async engines are
    lazy: obtaining the engine (or reading ``engine.pool.status()``) opens no
    connection and performs no I/O, so it cannot raise -- a pod with a dead
    database reported Ready and kept receiving traffic. We therefore execute
    ``SELECT 1`` on a real connection, bounded by a short timeout so a hung
    database fails readiness instead of stalling the probe. Pool status is kept
    as diagnostic detail only -- it never decides ``ready``.
    """
    try:
        async with asyncio.timeout(_READY_DB_TIMEOUT):
            engine = await get_engine()
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        pool = engine.pool
        pool_status = pool.status() if pool is not None else "unknown"
        return {
            "status": "healthy",
            "ready": True,
            "checks": {"database": {"status": "up", "pool": pool_status}},
        }
    except _DB_PROBE_ERRORS as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "ready": False,
                "checks": {"database": {"status": "down", "error": str(e)}},
            },
        )


@router.get("/readyz", operation_id="health.ready", response_model=None)
async def readyz() -> dict[str, Any] | JSONResponse:
    """Modern K8s readiness probe path (alias of /health/ready)."""
    return await health_ready()
