"""Tests for health check endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as redis_lib
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy.exc import InterfaceError, OperationalError


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    # Build a mock engine whose .begin() returns an async context manager
    mock_conn = AsyncMock()
    mock_conn.run_sync = AsyncMock()

    @asynccontextmanager
    async def _begin():
        yield mock_conn

    mock_async_engine = MagicMock()
    mock_async_engine.begin = _begin

    async def _get_engine():
        return mock_async_engine

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)

    with (
        patch("src.database.connection.get_engine", side_effect=_get_engine),
        patch("src.cache.redis_client.init_redis", new_callable=AsyncMock, return_value=mock_redis),
        patch("src.cache.redis_client.get_redis_client", return_value=mock_redis),
        patch("src.cache.redis_client.close_redis", new_callable=AsyncMock),
        patch("src.database.connection.close_engine", new_callable=AsyncMock),
        patch("src.services.storage_service.StorageService.initialize", new_callable=AsyncMock),
    ):
        from src.asgi import app

        with TestClient(app) as c:
            yield c


def _request(storage_service=None):
    """A stand-in ``Request`` exposing only what ``health_check`` reads.

    ``health_check`` now takes a ``Request`` solely to reach
    ``request.app.state.storage_service`` — the storage client the lifespan
    initialized and the job path writes through. Pass ``storage_service=None``
    (the default) to model an app with no initialized store: the minio branch
    must then report ``"unknown"``, never ``"healthy"``.
    """
    request = MagicMock()
    request.app.state.storage_service = storage_service
    return request


@pytest.fixture(autouse=True)
def _reset_health_cache():
    """Reset the module-level health cache before every test."""
    import src.api.health as h

    h._health_cache = {}
    h._health_cache_ts = 0.0
    yield
    h._health_cache = {}
    h._health_cache_ts = 0.0


class TestHealthEndpoints:
    def test_health_live(self, client):
        """Liveness probe should always return ok."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_endpoint(self, client):
        """Root endpoint should return service metadata."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "BDD Playwright Service"
        assert data["status"] == "running"
        assert "qa_suite" in data["endpoints"]


# ---------------------------------------------------------------------------
# Direct unit tests for health endpoint functions (covers all branches)
# ---------------------------------------------------------------------------


class TestHealthCheckFunction:
    """Tests for the /health endpoint handler covering all component branches."""

    @pytest.mark.asyncio
    async def test_all_healthy(self, tmp_path):
        """When DB, Redis, and storage are all healthy, status is 'healthy'."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.connect = _async_ctx(mock_conn)

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        mock_settings = MagicMock()
        mock_settings.storage.local_path = str(tmp_path)

        with (
            patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine),
            patch("src.api.health.get_redis_client", return_value=mock_redis),
            patch("src.api.health.get_settings", return_value=mock_settings),
        ):
            from src.api.health import health_check

            result = await health_check(_request())
            assert isinstance(result, dict)
            assert not isinstance(result, JSONResponse)
            assert result["status"] == "healthy"
            assert result["components"]["database"] == "healthy"
            assert result["components"]["redis"] == "healthy"
            assert result["components"]["storage"] == "healthy"
            assert result["components"]["api"] == "healthy"

    @pytest.mark.asyncio
    async def test_db_operational_error(self, tmp_path):
        """When DB raises OperationalError, database component is unhealthy."""
        mock_engine = AsyncMock()
        mock_engine.connect = _async_ctx_raising(
            OperationalError("conn refused", None, Exception("inner"))
        )

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        mock_settings = MagicMock()
        mock_settings.storage.local_path = str(tmp_path)

        with (
            patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine),
            patch("src.api.health.get_redis_client", return_value=mock_redis),
            patch("src.api.health.get_settings", return_value=mock_settings),
        ):
            from src.api.health import health_check

            result = await health_check(_request())
            assert isinstance(result, dict)
            assert not isinstance(result, JSONResponse)
            assert result["status"] == "degraded"
            assert "unhealthy" in result["components"]["database"]

    @pytest.mark.asyncio
    async def test_db_interface_error(self, tmp_path):
        """When DB raises InterfaceError, database component is unhealthy."""
        mock_engine = AsyncMock()
        mock_engine.connect = _async_ctx_raising(
            InterfaceError("bad iface", None, Exception("inner"))
        )

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        mock_settings = MagicMock()
        mock_settings.storage.local_path = str(tmp_path)

        with (
            patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine),
            patch("src.api.health.get_redis_client", return_value=mock_redis),
            patch("src.api.health.get_settings", return_value=mock_settings),
        ):
            from src.api.health import health_check

            result = await health_check(_request())
            assert isinstance(result, dict)
            assert not isinstance(result, JSONResponse)
            assert result["status"] == "degraded"
            assert "unhealthy" in result["components"]["database"]

    @pytest.mark.asyncio
    async def test_redis_not_configured(self, tmp_path):
        """When get_redis_client returns None, redis reports 'not configured'."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.connect = _async_ctx(mock_conn)

        mock_settings = MagicMock()
        mock_settings.storage.local_path = str(tmp_path)

        with (
            patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine),
            patch("src.api.health.get_redis_client", return_value=None),
            patch("src.api.health.get_settings", return_value=mock_settings),
        ):
            from src.api.health import health_check

            result = await health_check(_request())
            assert isinstance(result, dict)
            assert not isinstance(result, JSONResponse)
            assert result["status"] == "degraded"
            assert result["components"]["redis"] == "not configured"

    @pytest.mark.asyncio
    async def test_redis_error(self, tmp_path):
        """When Redis ping raises RedisError, redis component is unhealthy."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.connect = _async_ctx(mock_conn)

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=redis_lib.RedisError("timeout"))

        mock_settings = MagicMock()
        mock_settings.storage.local_path = str(tmp_path)

        with (
            patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine),
            patch("src.api.health.get_redis_client", return_value=mock_redis),
            patch("src.api.health.get_settings", return_value=mock_settings),
        ):
            from src.api.health import health_check

            result = await health_check(_request())
            assert isinstance(result, dict)
            assert not isinstance(result, JSONResponse)
            assert result["status"] == "degraded"
            assert "unhealthy" in result["components"]["redis"]

    @pytest.mark.asyncio
    async def test_storage_path_not_found(self):
        """When storage path does not exist, storage reports 'path not found'."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.connect = _async_ctx(mock_conn)

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        mock_settings = MagicMock()
        mock_settings.storage.local_path = "/nonexistent/path/bdd_test_xyz"

        with (
            patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine),
            patch("src.api.health.get_redis_client", return_value=mock_redis),
            patch("src.api.health.get_settings", return_value=mock_settings),
        ):
            from src.api.health import health_check

            result = await health_check(_request())
            assert isinstance(result, dict)
            assert not isinstance(result, JSONResponse)
            assert result["status"] == "degraded"
            assert result["components"]["storage"] == "path not found"

    @pytest.mark.asyncio
    async def test_storage_not_writable(self, tmp_path):
        """When storage path is not writable, storage reports 'not writable'."""
        read_only = tmp_path / "readonly"
        read_only.mkdir()

        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.connect = _async_ctx(mock_conn)

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        mock_settings = MagicMock()
        mock_settings.storage.local_path = str(read_only)

        with (
            patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine),
            patch("src.api.health.get_redis_client", return_value=mock_redis),
            patch("src.api.health.get_settings", return_value=mock_settings),
            patch("src.api.health.os.access", return_value=False),
        ):
            from src.api.health import health_check

            result = await health_check(_request())
            assert isinstance(result, dict)
            assert not isinstance(result, JSONResponse)
            assert result["status"] == "degraded"
            assert result["components"]["storage"] == "not writable"

    @pytest.mark.asyncio
    async def test_storage_os_error(self):
        """When storage check raises OSError, storage is unhealthy."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.connect = _async_ctx(mock_conn)

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with (
            patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine),
            patch("src.api.health.get_redis_client", return_value=mock_redis),
            patch("src.api.health.get_settings", side_effect=OSError("disk error")),
        ):
            from src.api.health import health_check

            result = await health_check(_request())
            assert isinstance(result, dict)
            assert not isinstance(result, JSONResponse)
            # The OSError from get_settings propagates to the storage try/except (OSError)
            # so storage becomes unhealthy, but overall status is degraded (not 503)
            assert result["status"] == "degraded"
            assert "unhealthy" in result["components"]["storage"]

    @pytest.mark.asyncio
    async def test_cache_returns_previous_result(self, tmp_path):
        """Calling health_check within TTL returns cached result."""
        import src.api.health as h

        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.connect = _async_ctx(mock_conn)

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        mock_settings = MagicMock()
        mock_settings.storage.local_path = str(tmp_path)

        mock_get_engine = AsyncMock(return_value=mock_engine)

        with (
            patch("src.api.health.get_engine", mock_get_engine),
            patch("src.api.health.get_redis_client", return_value=mock_redis),
            patch("src.api.health.get_settings", return_value=mock_settings),
        ):
            # First call populates cache
            result1 = await h.health_check(_request())
            assert isinstance(result1, dict)
            assert not isinstance(result1, JSONResponse)
            assert result1["status"] == "healthy"
            assert mock_get_engine.call_count == 1

            # Second call should return cached result (get_engine not called again)
            result2 = await h.health_check(_request())
            assert isinstance(result2, dict)
            assert not isinstance(result2, JSONResponse)
            assert result2["status"] == "healthy"
            # get_engine should still only have been called once (from first call)
            assert mock_get_engine.call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_cold_cache_is_single_flight(self, tmp_path):
        """Concurrent cold-cache callers run a single fill; others increment the stampede counter."""
        import asyncio

        import src.api.health as h

        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.connect = _async_ctx(mock_conn)

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        mock_settings = MagicMock()
        mock_settings.storage.local_path = str(tmp_path)

        # get_engine artificially slow so concurrent callers pile up on the lock.
        async def slow_get_engine():
            await asyncio.sleep(0.05)
            return mock_engine

        mock_get_engine = AsyncMock(side_effect=slow_get_engine)
        baseline = h.get_health_cache_stampede_avoided()

        with (
            patch("src.api.health.get_engine", mock_get_engine),
            patch("src.api.health.get_redis_client", return_value=mock_redis),
            patch("src.api.health.get_settings", return_value=mock_settings),
        ):
            results = await asyncio.gather(*(h.health_check(_request()) for _ in range(5)))

        for r in results:
            assert isinstance(r, dict)
            assert not isinstance(r, JSONResponse)
            assert r["status"] == "healthy"
        # Only one fill happened across 5 concurrent callers.
        assert mock_get_engine.call_count == 1
        # The other 4 callers awaited the lock and observed a fresh cache → counted as avoided.
        assert h.get_health_cache_stampede_avoided() - baseline == 4

    @pytest.mark.asyncio
    async def test_outer_exception_returns_503(self):
        """When get_engine raises an unexpected error, return 503."""
        with patch(
            "src.api.health.get_engine",
            new_callable=AsyncMock,
            side_effect=RuntimeError("catastrophic failure"),
        ):
            from src.api.health import health_check

            result = await health_check(_request())

        assert isinstance(result, JSONResponse)
        assert result.status_code == 503
        body = result.body
        assert b"unhealthy" in body
        assert b"catastrophic failure" in body


class TestHealthReadyFunction:
    """Tests for the /health/ready endpoint handler."""

    @pytest.mark.asyncio
    async def test_ready_healthy(self):
        """When the DB round trip succeeds, readiness returns healthy (200 shape)."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.connect = _async_ctx(mock_conn)
        mock_pool = MagicMock()
        mock_pool.status.return_value = "Pool size: 5"
        mock_engine.pool = mock_pool

        with patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine):
            from src.api.health import health_ready

            result = await health_ready()
            assert isinstance(result, dict)
            assert not isinstance(result, JSONResponse)
            assert result["status"] == "healthy"
            assert result["ready"] is True
            assert result["checks"]["database"]["status"] == "up"
            assert result["checks"]["database"]["pool"] == "Pool size: 5"
        # The probe must have executed a real statement, not merely read the pool.
        mock_conn.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ready_returns_200_over_http_when_roundtrip_succeeds(self):
        """Readiness returns HTTP 200 through the ASGI stack on a good round trip."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.connect = _async_ctx(mock_conn)
        mock_engine.pool = None

        from src.api.health import health_ready

        with patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine):
            result = await health_ready()

        response = result if isinstance(result, JSONResponse) else JSONResponse(content=result)
        assert response.status_code == 200
        assert b'"ready":true' in response.body

    @pytest.mark.asyncio
    async def test_ready_pool_none(self):
        """When engine.pool is None, pool status is 'unknown'."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.connect = _async_ctx(mock_conn)
        mock_engine.pool = None

        with patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine):
            from src.api.health import health_ready

            result = await health_ready()
            assert isinstance(result, dict)
            assert not isinstance(result, JSONResponse)
            assert result["status"] == "healthy"
            assert result["ready"] is True
            assert result["checks"]["database"]["pool"] == "unknown"

    @pytest.mark.asyncio
    async def test_ready_operational_error(self):
        """When get_engine raises OperationalError, readiness returns 503."""
        with patch(
            "src.api.health.get_engine",
            new_callable=AsyncMock,
            side_effect=OperationalError("conn refused", None, Exception("inner")),
        ):
            from src.api.health import health_ready

            result = await health_ready()

        assert isinstance(result, JSONResponse)
        assert result.status_code == 503
        body = result.body
        assert b"unhealthy" in body

    @pytest.mark.asyncio
    async def test_ready_interface_error(self):
        """When get_engine raises InterfaceError, readiness returns 503."""
        with patch(
            "src.api.health.get_engine",
            new_callable=AsyncMock,
            side_effect=InterfaceError("bad conn", None, Exception("inner")),
        ):
            from src.api.health import health_ready

            result = await health_ready()

        assert isinstance(result, JSONResponse)
        assert result.status_code == 503
        body = result.body
        assert b"unhealthy" in body
        assert b"down" in body

    @pytest.mark.asyncio
    async def test_ready_roundtrip_operational_error_is_503_not_500(self):
        """A dead DB surfaces on the ROUND TRIP (not on get_engine) -> 503, never 500/200.

        This is the regression guard for the lazy-engine defect: the old body only
        read ``engine.pool.status()``, which does no I/O and cannot raise, so this
        exact failure reported Ready.
        """
        mock_engine = AsyncMock()
        mock_engine.connect = _async_ctx_raising(
            OperationalError("connection refused", None, Exception("inner"))
        )
        mock_engine.pool = MagicMock()

        with patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine):
            from src.api.health import health_ready

            result = await health_ready()

        assert isinstance(result, JSONResponse), f"expected a 503 JSONResponse, got {result!r}"
        assert result.status_code == 503
        body = result.body
        assert b'"ready":false' in body
        # The body must name the failing check.
        assert b"database" in body
        assert b"down" in body
        assert b"connection refused" in body

    @pytest.mark.asyncio
    async def test_ready_roundtrip_timeout_is_503(self):
        """A hung DB round trip (asyncio.TimeoutError) fails readiness with 503."""

        mock_engine = AsyncMock()
        mock_engine.connect = _async_ctx_raising(TimeoutError("probe timed out"))
        mock_engine.pool = MagicMock()

        with patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine):
            from src.api.health import health_ready

            result = await health_ready()

        assert isinstance(result, JSONResponse), f"expected a 503 JSONResponse, got {result!r}"
        assert result.status_code == 503
        body = result.body
        assert b'"ready":false' in body
        assert b"database" in body
        assert b"down" in body

    @pytest.mark.asyncio
    async def test_ready_execute_error_is_503(self):
        """An error raised by the SELECT 1 itself (not the connect) is a 503."""
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(
            side_effect=OperationalError("server closed the connection", None, Exception("inner"))
        )
        mock_engine.connect = _async_ctx(mock_conn)
        mock_engine.pool = MagicMock()

        with patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine):
            from src.api.health import health_ready

            result = await health_ready()

        assert isinstance(result, JSONResponse), f"expected a 503 JSONResponse, got {result!r}"
        assert result.status_code == 503
        assert b"database" in result.body
        assert b"down" in result.body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _async_ctx(return_value):
    """Return a callable that produces an async context manager yielding *return_value*."""

    @asynccontextmanager
    async def _ctx():
        yield return_value

    return _ctx


def _async_ctx_raising(exc):
    """Return a callable that produces an async context manager raising *exc*."""

    @asynccontextmanager
    async def _ctx():
        raise exc
        yield  # pragma: no cover -- unreachable, but required for generator syntax

    return _ctx


# ---------------------------------------------------------------------------
# Startup log redaction -- CALL SITE, not the helper
# ---------------------------------------------------------------------------

_FAKE_PASSWORD = "not-a-real-password-xyzzy"  # pragma: allowlist secret
_FAKE_USER = "bdd_app"
_FAKE_HOST = "pgbouncer.pgbouncer"
_FAKE_DB = "bdd_playwright_service"
_FAKE_DSN = f"postgresql+psycopg://{_FAKE_USER}:{_FAKE_PASSWORD}@{_FAKE_HOST}:6432/{_FAKE_DB}"


class TestStartupLogDoesNotLeakDsnPassword:
    """The startup 'Database connected' line must never carry DSN userinfo.

    This is deliberately a test of the CALL SITE in ``src.asgi._init_database``
    rather than of ``w2v_common.observability.database_endpoint``: a
    helper-only test stays green for the entire duration of the leak.

    ``caplog`` is not used on purpose -- root handlers are reconfigured at
    import time, so caplog can capture nothing and the assertions would pass
    vacuously. We attach our own handler to the module logger instead.
    """

    @pytest.mark.asyncio
    async def test_database_connected_line_is_redacted(self, monkeypatch):
        import io
        import logging

        import src.asgi as asgi

        monkeypatch.setattr(asgi.settings.database, "url", _FAKE_DSN)

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        asgi.logger.addHandler(handler)
        previous_level = asgi.logger.level
        asgi.logger.setLevel(logging.INFO)
        try:
            with (
                patch("src.asgi.get_engine", new_callable=AsyncMock, return_value=MagicMock()),
                patch("src.database.rls_events.install_rls_tenant_rebind"),
            ):
                await asgi._init_database(MagicMock())
        finally:
            asgi.logger.removeHandler(handler)
            asgi.logger.setLevel(previous_level)

        captured = stream.getvalue()

        # Vacuity guard: without this the test passes when the line never emits.
        # (Matches "Database connected: <dsn>" specifically -- a later line reads
        # "Database connected (schema managed by alembic)" and must not satisfy this.)
        assert "Database connected: " in captured, f"startup line never emitted: {captured!r}"

        # The secret and the DB username must be gone.
        assert _FAKE_PASSWORD not in captured
        assert _FAKE_USER not in captured

        # ...but the line must stay useful: host and database still identify the target.
        assert _FAKE_HOST in captured
        assert _FAKE_DB in captured


# ---------------------------------------------------------------------------
# Startup Redis log redaction -- CALL SITE, not the helper
# ---------------------------------------------------------------------------

_FAKE_REDIS_PASSWORD = "not-a-real-password-xyzzy"  # pragma: allowlist secret
_FAKE_REDIS_USER = "default"
_FAKE_REDIS_HOST = "redis.w2v"
_FAKE_REDIS_URL = f"redis://{_FAKE_REDIS_USER}:{_FAKE_REDIS_PASSWORD}@{_FAKE_REDIS_HOST}:6379/3"


class TestStartupLogDoesNotLeakRedisPassword:
    """The startup 'Redis connected' line must never carry the URL password.

    A Redis URL is ``redis://[user]:<password>@host:port/db``, so logging it raw
    wrote the live Redis password to pod stdout on every pod start.

    This is deliberately a test of the CALL SITE in ``src.asgi._init_redis``
    rather than of ``w2v_common.observability.redis_endpoint``: a helper-only
    test stays green for the entire duration of the leak, and that helper
    already has its own unit tests in w2v_common.

    ``caplog`` is not used on purpose -- root handlers are reconfigured at
    import time, so caplog can capture nothing and the assertions would pass
    vacuously. We attach our own handler to the module logger instead.
    """

    @pytest.mark.asyncio
    async def test_redis_connected_line_is_redacted(self, monkeypatch):
        import io
        import logging

        import src.asgi as asgi

        monkeypatch.setattr(asgi.settings.redis, "url", _FAKE_REDIS_URL)

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        asgi.logger.addHandler(handler)
        # ``setLevel`` and not ``logger.level = ...``: Logger caches
        # isEnabledFor results and only setLevel clears that cache. Assigning
        # the attribute leaves a stale "INFO is disabled" entry, the record
        # never reaches the handler, and the negative assertions below pass on
        # EMPTY output while the leak is still live.
        previous_level = asgi.logger.level
        asgi.logger.setLevel(logging.INFO)

        redis_client = AsyncMock()
        try:
            with (
                patch("src.asgi.init_redis", new_callable=AsyncMock, return_value=redis_client),
                patch("src.asgi.get_redis_client", MagicMock(return_value=redis_client)),
            ):
                await asgi._init_redis(MagicMock())
        finally:
            asgi.logger.removeHandler(handler)
            asgi.logger.setLevel(previous_level)

        captured = stream.getvalue()

        # Vacuity guard: without this the test passes when the line never emits.
        assert "Redis connected: " in captured, f"startup line never emitted: {captured!r}"

        # The secret and the userinfo must be gone.
        assert _FAKE_REDIS_PASSWORD not in captured, "Redis password leaked into the startup log"
        assert f"{_FAKE_REDIS_USER}:" not in captured, "Redis userinfo leaked into the startup log"

        # ...but the line must stay useful: host and port still identify the target.
        assert f"{_FAKE_REDIS_HOST}:6379" in captured


class TestStorageProbeHonoursTheConfiguredBackend:
    """The storage line must describe the store the JOBS use, not the scratch dir.

    ⚠️ Every one of these cases has ``local_path`` pointing at a REAL, existing,
    WRITABLE directory — exactly the prod shape, where ``local_path`` is the
    pod-local ``emptyDir`` the package builder zips into and
    ``STORAGE_BACKEND=minio`` names the durable store. Under the old probe
    (``Path(local_path).exists()`` + ``os.access(W_OK)``, no backend branch) all
    four of these returned ``"storage": "healthy"`` regardless of MinIO, which is
    why a fleet of pods reported healthy while every job failed at its first
    store write.
    """

    @staticmethod
    def _minio_settings(local_path) -> MagicMock:
        settings = MagicMock()
        settings.storage.backend = "minio"
        settings.storage.local_path = str(local_path)
        settings.storage.minio_bucket = "w2v-bdd-playwright"
        return settings

    @staticmethod
    def _healthy_deps():
        mock_engine = AsyncMock()
        mock_engine.connect = _async_ctx(AsyncMock())
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        return mock_engine, mock_redis

    async def _run(self, tmp_path, storage_service):
        mock_engine, mock_redis = self._healthy_deps()
        with (
            patch("src.api.health.get_engine", new_callable=AsyncMock, return_value=mock_engine),
            patch("src.api.health.get_redis_client", return_value=mock_redis),
            patch("src.api.health.get_settings", return_value=self._minio_settings(tmp_path)),
        ):
            from src.api.health import health_check

            result = await health_check(_request(storage_service))
        assert isinstance(result, dict)
        return result

    @pytest.mark.asyncio
    async def test_unreachable_bucket_is_not_healthy(self, tmp_path):
        """Bucket stat says no → storage unhealthy even though local_path is fine."""
        service = MagicMock()
        service.check_health = AsyncMock(return_value=False)

        result = await self._run(tmp_path, service)

        assert result["components"]["storage"].startswith("unhealthy")
        assert "w2v-bdd-playwright" in result["components"]["storage"]
        assert result["status"] == "degraded"
        service.check_health.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raising_probe_is_not_healthy(self, tmp_path):
        """A connection error reaching MinIO is reported, not swallowed."""
        service = MagicMock()
        service.check_health = AsyncMock(side_effect=OSError("connection refused"))

        result = await self._run(tmp_path, service)

        assert "connection refused" in result["components"]["storage"]
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_hung_probe_times_out_instead_of_stalling(self, tmp_path):
        """A hung object store must FAIL the probe, not hold /health open."""
        import asyncio

        async def _hang() -> bool:
            await asyncio.sleep(30)
            return True

        service = MagicMock()
        service.check_health = AsyncMock(side_effect=_hang)

        with patch("src.api.health._STORAGE_PROBE_TIMEOUT", 0.05):
            result = await self._run(tmp_path, service)

        assert result["components"]["storage"].startswith("unhealthy")
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_unprobeable_store_reports_unknown_not_healthy(self, tmp_path):
        """No initialized client = UNMEASURED. Unmeasured is never 'healthy'."""
        result = await self._run(tmp_path, storage_service=None)

        assert result["components"]["storage"] == "unknown: storage service not initialized"
        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_reachable_bucket_is_healthy(self, tmp_path):
        """Control: a reachable bucket reports healthy on the minio backend."""
        service = MagicMock()
        service.check_health = AsyncMock(return_value=True)

        result = await self._run(tmp_path, service)

        assert result["components"]["storage"] == "healthy"
        assert result["status"] == "healthy"
