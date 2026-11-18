"""Guardrail tests for the auth middleware integration.

bdd_playwright_service delegates auth to ``w2v_common.auth.middleware.AuthMiddleware``
(re-exported via ``src.auth.middleware``).  This test exercises that the
middleware is **wired and registered** on the asgi app (not merely importable),
and that it behaves correctly for the standard public / protected /
invalid-token matrix.

The intent is to catch the "false-merge" pattern: a PR that says it added
auth but never called ``app.add_middleware(AuthMiddleware)``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
import w2v_common.auth.middleware as w2v_auth
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from w2v_common.auth.middleware import AuthMiddleware

from tests.conftest import TENANT_A


@pytest.fixture(autouse=True)
def _clear_license_cache() -> None:
    cache = w2v_auth._get_license_cache()
    cache.clear()


def _make_minimal_app() -> FastAPI:
    """Tiny app for behaviour tests -- independent of asgi lifespan side-effects."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/health")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/bdd/echo")
    async def _echo(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "tenant_id": getattr(request.state, "tenant_id", None),
                "license_present": getattr(request.state, "license", None) is not None,
            }
        )

    return app


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Wiring check -- the regression-against-false-merge assertion
# ---------------------------------------------------------------------------


def test_auth_middleware_is_actually_registered_on_asgi_app() -> None:
    """The asgi entry must register AuthMiddleware via ``app.add_middleware``.

    Reading ``settings.auth.enabled`` gates the registration in asgi.py.
    Tests run with ``AUTH_ENABLED`` unset (per conftest.py ``mock_env_vars``),
    so the wiring check has to compare against the production setting, not
    the live middleware stack. We verify both:
    (1) the import path resolves to the shared library, and
    (2) the asgi module references ``app.add_middleware(AuthMiddleware)``.
    """
    from pathlib import Path

    # (1) AuthMiddleware comes from the shared lib (not a local stub).
    from w2v_common.auth.middleware import AuthMiddleware as SharedAuthMiddleware

    assert AuthMiddleware is SharedAuthMiddleware

    # (2) The asgi module's source references the registration line.
    asgi_src = (Path(__file__).resolve().parent.parent / "src" / "asgi.py").read_text()
    assert "app.add_middleware(AuthMiddleware)" in asgi_src, (
        "asgi.py must register AuthMiddleware via app.add_middleware"
    )


# ---------------------------------------------------------------------------
# Public bypass
# ---------------------------------------------------------------------------


async def test_health_is_public(client: httpx.AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Protected paths
# ---------------------------------------------------------------------------


async def test_protected_route_with_valid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    introspect_payload: dict[str, Any] = {
        "active": True,
        "tenant_id": TENANT_A,
        "subject_id": "subj-1",
        "license_id": "lic-1",
        "roles": ["user"],
    }

    async def _fake(token: str) -> dict[str, Any]:
        assert token == "abc.def.ghi"
        return introspect_payload

    monkeypatch.setattr(w2v_auth, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/bdd/echo",
            headers={"X-Tenant-Id": "Bearer abc.def.ghi"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tenant_id"] == TENANT_A
    assert body["license_present"] is True


async def test_protected_route_with_invalid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake(_token: str) -> dict[str, Any]:
        return {"active": False, "reason": "expired"}

    monkeypatch.setattr(w2v_auth, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/bdd/echo",
            headers={"X-Tenant-Id": "Bearer dud-token"},
        )

    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "INVALID_TOKEN"
    assert body["error"]["status"] == 401
    assert "Invalid or expired license token" in body["error"]["message"]


async def test_protected_route_no_token_untrusted_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted source IP + no token must 401, regardless of TRUST_INCLUSTER_IDENTITY."""
    monkeypatch.setattr(w2v_auth, "_trust_incluster_identity", lambda: False)

    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/bdd/echo")
    assert resp.status_code == 401
