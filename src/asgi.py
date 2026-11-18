"""ASGI application for bdd_playwright_service."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from typing import Any, Protocol

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.asgi import register_error_envelope_handlers
from w2v_common.observability import database_endpoint, redis_endpoint

from src.cache.redis_client import close_redis, get_redis_client, init_redis
from src.config import get_settings
from src.database.connection import close_engine, get_engine
from src.database.session import get_db_session
from src.services.background import drain_background_tasks
from src.services.cleanup_service import CleanupService
from src.services.storage_service import StorageService

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


# ============================================================================
# Health Check Logging Filter
# ============================================================================


class HealthCheckLoggingFilter(logging.Filter):
    """Suppress noisy health-check endpoint log lines from uvicorn access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(
            path in message
            for path in ["/health", "/health/ready", "/health/live", "/readyz", "/healthz"]
        )


uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(HealthCheckLoggingFilter())


# ============================================================================
# Lifespan
# ============================================================================


async def _init_database(app: FastAPI) -> None:
    """Initialise the DB engine and create tables."""
    # Phase D RLS: install the after_begin GUC re-bind listener before any session
    # is opened, so the multi-commit background pipeline keeps app.current_tenant
    # bound across each commit under enforce.
    from src.database.rls_events import install_rls_tenant_rebind

    install_rls_tenant_rebind()
    try:
        db_engine = await get_engine()
        app.state.db_engine = db_engine
        logger.info("Database connected: %s", database_endpoint(settings.database.url))
    except Exception as e:
        logger.error("Failed to connect to database: %s", e, exc_info=True)
        raise

    # Schema is owned by alembic -- the gitops Deployment runs
    # ``alembic upgrade head`` in an initContainer before this app
    # starts (see gitops-production
    # clusters/strato/services/w2v/deployments.yaml). Do NOT call
    # ``Base.metadata.create_all`` here: it bypasses migration history
    # and can drift the live schema away from what alembic expects.
    logger.info("Database connected (schema managed by alembic)")


async def _init_redis(app: FastAPI) -> None:
    """Initialise the Redis client; failure is non-fatal."""
    try:
        redis_client = await init_redis()
        await redis_client.ping()
        app.state.redis_client = get_redis_client()
        logger.info("Redis connected: %s", redis_endpoint(settings.redis.url))
    except Exception as e:
        logger.error("Failed to connect to Redis: %s", e, exc_info=True)
        logger.warning("Continuing without Redis -- caching will be disabled")


CLEANUP_INTERVAL_SECONDS = 300


class JobSweeper(Protocol):
    """What the cleanup loop actually needs — two sweeps, nothing else.

    Typed as a Protocol rather than the concrete ``CleanupService`` so the loop
    states its real dependency and a test can supply a stand-in that records
    what the sweep DID. ``CleanupService`` satisfies it structurally; nothing
    about production behaviour changes.
    """

    async def cleanup_stuck_jobs(
        self, db: AsyncSession, *, stuck_timeout_minutes: int, pending_timeout_minutes: int
    ) -> int: ...

    async def cleanup_old_jobs(self, db: AsyncSession, *, retention_days: int) -> int: ...


async def _run_cleanup_loop(
    cleanup_service: JobSweeper,
    interval_seconds: float = CLEANUP_INTERVAL_SECONDS,
) -> None:
    """Sweep stuck/old jobs forever, sweeping FIRST and sleeping afterwards.

    ⚠️ The sleep is at the TAIL, not the head. With it at the head, a pod that
    restarted while jobs were mid-flight left them stuck for a full interval
    before the first sweep ran — exactly the window in which a rolling restart
    strands rows.

    ⚠️ The error path sleeps too. Moving the sleep to the tail without that
    turns a persistently failing sweep (DB down, for instance) into a tight
    busy-loop hammering the logger and the DB. A ``finally`` is the WRONG tool
    here: the ``CancelledError`` branch would then sleep on shutdown and delay
    termination.
    """
    while True:
        try:
            async with get_db_session() as db:
                stuck = await cleanup_service.cleanup_stuck_jobs(
                    db,
                    stuck_timeout_minutes=settings.cleanup.stuck_job_timeout_minutes,
                    pending_timeout_minutes=settings.cleanup.pending_job_timeout_minutes,
                )
                if stuck:
                    logger.info("Cleaned up %d stranded jobs", stuck)
                old = await cleanup_service.cleanup_old_jobs(
                    db,
                    retention_days=settings.cleanup.retention_days,
                )
                if old:
                    logger.info("Cleaned up %d old jobs", old)
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Cleanup task error: %s", e, exc_info=True)
            await asyncio.sleep(interval_seconds)


async def _init_storage_and_cleanup(app: FastAPI) -> asyncio.Task[None]:
    """Initialise storage service and start the background cleanup loop."""
    storage = StorageService()
    await storage.initialize()
    app.state.storage_service = storage

    cleanup_service = CleanupService()

    cleanup_task = asyncio.create_task(_run_cleanup_loop(cleanup_service))
    logger.info("Cleanup background task started")
    return cleanup_task


async def _shutdown_cleanup(cleanup_task: asyncio.Task[None]) -> None:
    """Cancel and await the background cleanup task."""
    cleanup_task.cancel()
    with suppress(asyncio.CancelledError):
        await cleanup_task
    logger.info("Cleanup background task stopped")


async def _shutdown_storage(app: FastAPI) -> None:
    """Close the app-level StorageService's MinIO HTTP session."""
    storage = getattr(app.state, "storage_service", None)
    if storage is None:
        return
    try:
        await storage.aclose()
    except Exception as e:
        logger.error("Error closing storage: %s", e, exc_info=True)


async def _shutdown_redis() -> None:
    """Close the Redis connection."""
    try:
        await close_redis()
    except Exception as e:
        logger.error("Error closing Redis: %s", e, exc_info=True)


async def _shutdown_database() -> None:
    """Dispose of the DB engine."""
    try:
        await close_engine()
    except Exception as e:
        logger.error("Error closing database: %s", e, exc_info=True)


async def _shutdown_wm_client() -> None:
    """Close the shared WMClient (worker_management gateway)."""
    from src.services.generation.llm_client import aclose_shared_wm_client

    try:
        await aclose_shared_wm_client()
    except Exception as e:
        logger.error("Error closing WMClient: %s", e, exc_info=True)


async def _shutdown(cleanup_task: asyncio.Task[None]) -> None:
    """Tear down in dependency order — DRAIN JOB TASKS BEFORE DISPOSING THE ENGINE.

    ⛔ The ordering here is the fix, not decoration. ``_process_job_bg`` catches
    ``CancelledError`` and calls ``mark_failed`` to turn a non-terminal row terminal,
    but that rescue opens a NEW DB session. With ``_shutdown_database()`` running
    first (and nothing cancelling ``_background_tasks`` at all), the rescue always met
    a disposed engine, raised, and left the row in ``pending`` — the state nothing
    reclaimed. ``drain_background_tasks()`` must therefore stay ABOVE
    ``_shutdown_database()``; moving it below silently restores the leak.
    """
    logger.info("Shutting down BDD Playwright Service")
    await _shutdown_cleanup(cleanup_task)
    await drain_background_tasks()
    await _shutdown_wm_client()
    await _shutdown_redis()
    await _shutdown_database()
    logger.info("BDD Playwright Service shutdown complete")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan -- init DB, Redis, Storage."""
    logger.info("=" * 70)
    logger.info("Starting BDD Playwright Service")
    logger.info("=" * 70)

    # WEBSCRAPER_AUTH_MODE (w2v_common TenantLicenceLane): a typo, or tenant_licence
    # without the BUNDLE_MINTER_* settings, fails the pod here, not the first job.
    from w2v_common.auth.tenant_licence_lane import TenantLicenceLane

    logger.info("webscraper hop auth mode: %s", TenantLicenceLane("webscraper").mode.value)

    await _init_database(app)
    await _init_redis(app)
    cleanup_task = await _init_storage_and_cleanup(app)

    logger.info("=" * 70)
    logger.info("BDD Playwright Service started successfully")
    logger.info("Environment: %s", settings.environment)
    logger.info("API: %s:%d", settings.api.host, settings.api.port)
    logger.info("=" * 70)

    yield

    await _shutdown(cleanup_task)
    # After the job drain inside `_shutdown`: a draining job may still upload.
    await _shutdown_storage(app)


# ============================================================================
# Public-endpoint policy (kept in sync with w2v_common.auth.middleware)
# ============================================================================
# Used only by custom_openapi() below to decide which paths advertise the
# Bearer security requirement in the schema. The actual auth enforcement is
# done by w2v_common.auth.middleware.AuthMiddleware.
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/health",
        "/health/live",
        "/health/ready",
        "/healthz",
        "/readyz",
        "/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
        "/metrics",
    }
)
_PUBLIC_PREFIXES: tuple[str, ...] = ("/docs", "/redoc", "/health/")


# ============================================================================
# Create App
# ============================================================================

app = FastAPI(
    title=settings.app_name,
    description="BDD Playwright script generation microservice with Gherkin-based narration",
    version=settings.app_version,
    lifespan=lifespan,
    redirect_slashes=False,
    swagger_ui_parameters={"persistAuthorization": True},
    # Fallback unique-id generator. Every route still sets `operation_id=`
    # explicitly per OPERATION_ID_PLAN.md; this exists so a future route
    # that forgets the kwarg degrades to `<tag>_<func>` instead of FastAPI's
    # default `<func>_<path>_<method>` (garbage in generated clients).
    generate_unique_id_function=lambda route: (
        f"{route.tags[0] if route.tags else 'default'}_{route.name}"
    ),
    openapi_tags=[
        {"name": "health", "description": "Liveness, readiness, dependency probes."},
        {
            "name": "bdd",
            "description": (
                "BDD Playwright generation jobs: QA suites, accessibility audits, "
                "documentation, demo videos, Gherkin upload, and artifact download."
            ),
        },
    ],
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=settings.api.cors_allow_credentials,
    allow_methods=settings.api.cors_allow_methods,
    allow_headers=settings.api.cors_allow_headers,
)


# Rate limiter middleware
from src.middleware.rate_limiter import (  # noqa: E402 -- must import after app initialization
    RateLimitMiddleware,
)

app.add_middleware(RateLimitMiddleware)
logger.info("Rate limiter middleware enabled")

# Auth middleware
from src.auth.middleware import AuthMiddleware  # noqa: E402 -- must import after app initialization

if settings.auth.enabled:
    app.add_middleware(AuthMiddleware)
    logger.info("Authentication middleware enabled")
else:
    logger.warning("Running WITHOUT authentication -- development mode only!")

# Request ID middleware -- registered last so it is the OUTERMOST layer and runs first
# inbound, binding the request_id ContextVar before auth/handlers execute.
from w2v_common.observability.request_context import RequestIdLogFilter  # noqa: E402

from src.middleware.request_id import RequestIdMiddleware  # noqa: E402

app.add_middleware(RequestIdMiddleware)
logging.getLogger().addFilter(RequestIdLogFilter())


# ============================================================================
# Root & Health Endpoints
# ============================================================================


@app.get(
    "/",
    operation_id="service_root",
    tags=["health"],
    summary="Service banner + endpoint index",
)
async def root() -> dict[str, Any]:
    """Return service metadata and available endpoints."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "BDD Playwright script generation with Gherkin-based narration",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "qa_suite": "/api/bdd/qa-suite",
            "accessibility": "/api/bdd/accessibility",
            "documentation": "/api/bdd/documentation",
            "demo_video": "/api/bdd/demo-video",
            "jobs": "/api/bdd/{job_id}",
            "download": "/api/bdd/{job_id}/download",
        },
        "status": "running",
        "environment": settings.environment,
    }


# Mount routers
from src.api.health import (  # noqa: E402 -- must import after app initialization
    router as health_router,
)
from src.api.routes import (  # noqa: E402 -- must import after app initialization
    router as api_router,
)

app.include_router(health_router)
app.include_router(api_router)


# ============================================================================
# OpenAPI customisation -- BearerAuth scheme + TenantIdHeader parameter
# ============================================================================
#
# Mirrors the kag / worker_management / nlp_service / analysis_service /
# video_compiler_service / video_analyzer_service pattern: every protected
# operation advertises ``security: [{BearerAuth: []}]`` and a reusable
# ``TenantIdHeader`` parameter so Swagger UI exposes the license-token
# transport correctly. The actual enforcement lives in
# ``w2v_common.auth.middleware.AuthMiddleware``.
def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )

    components = schema.setdefault("components", {})
    components.setdefault("securitySchemes", {})["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "License",
        "description": (
            "License token. Transport: `X-Tenant-Id: Bearer <token>` "
            "(not the Authorization header -- that is reserved for caller-side auth)."
        ),
    }
    components.setdefault("parameters", {})["TenantIdHeader"] = {
        "name": "X-Tenant-Id",
        "in": "header",
        "required": False,
        "description": "License token transport: `Bearer <token>`.",
        "schema": {"type": "string"},
    }

    for path, item in schema.get("paths", {}).items():
        clean = path.rstrip("/") or "/"
        is_public = clean in _PUBLIC_PATHS or any(clean.startswith(p) for p in _PUBLIC_PREFIXES)
        for operation in item.values():
            if not isinstance(operation, dict):
                continue
            if is_public:
                operation.pop("security", None)
            else:
                operation.setdefault("security", [{"BearerAuth": []}])
            params = operation.setdefault("parameters", [])
            if not any(
                p.get("name", "").lower() == "x-tenant-id" and p.get("in") == "header"
                for p in params
            ):
                params.append({"$ref": "#/components/parameters/TenantIdHeader"})

    app.openapi_schema = schema
    return app.openapi_schema


object.__setattr__(app, "openapi", custom_openapi)


# ============================================================================
# Error Handlers — canonical fleet error envelope (shape #2), from w2v_common
# ============================================================================
register_error_envelope_handlers(app, debug=settings.api.debug)


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Main entry point for running the ASGI application."""
    import uvicorn

    uvicorn.run(
        "src.asgi:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
        reload=settings.api.reload,
    )


if __name__ == "__main__":
    main()
