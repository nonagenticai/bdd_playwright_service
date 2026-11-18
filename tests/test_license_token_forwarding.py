"""The licence token the routes forward to ``dispatch_job`` must be the CANONICAL one.

Regression tests for a SILENT defect: ``src/services/background.py`` carried a local
``extract_license_token`` that re-implemented w2v_common's PRE-0.74.0 rule — it read
the token ONLY from ``X-Tenant-Id: Bearer <tok>``. w2v_common 0.74.0 widened the
canonical rule to ALSO accept an opaque ``Authorization: Bearer <tok>``, which is the
only shape the ``license-authz`` ext-authz gate forwards. Behind that gate the copy
returned ``None``, ``dispatch_job`` received ``license_token=None``, and every
background peer call silently fell back to trusted-network auth.

NO TEST FAILED. The copy had zero coverage; the suite was green and the code was wrong.

Every test here exercises the REAL ``w2v_common.auth.middleware.AuthMiddleware`` and
the REAL extractor the job routes call, so the pair cannot drift apart again without
going red.
"""

from __future__ import annotations

import ast
from typing import Any

import httpx
import pytest
import w2v_common.auth.middleware as w2v_auth
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from w2v_common.auth.middleware import AuthMiddleware

from src.api.routes._helpers import get_license_token
from tests.conftest import TENANT_A

#: An OPAQUE licence token: deliberately NOT three dot-separated segments, so
#: ``w2v_common.auth.keycloak._looks_like_jwt`` is False and the canonical rule is
#: allowed to read it off ``Authorization``.
OPAQUE_TOKEN = "opaque-licence-token-0123456789"

#: A JWT-SHAPED bearer: three non-empty dot-separated segments. The canonical rule
#: must NEVER submit this to licence introspection (the 2026-04-26 prod 401s).
JWT_SHAPED_BEARER = "aaaa.bbbb.cccc"

#: The caller's DELEGATED token, which ``BaseServiceClient`` puts on ``X-Tenant-Id``
#: while its own service-account token sits on ``Authorization``.
DELEGATED_TOKEN = "delegated-caller-token-abcdef"


@pytest.fixture(autouse=True)
def _clear_license_cache() -> None:
    w2v_auth._get_license_cache().clear()


@pytest.fixture(autouse=True)
def _no_trusted_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the trusted-network lane out of the way; it sets ``license_token = None``."""
    monkeypatch.setattr(w2v_auth, "_trust_incluster_identity", lambda: False)


@pytest.fixture(autouse=True)
def _stub_introspection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept any opaque token; record nothing. JWT-shaped input is a hard failure.

    ``_introspect_license`` receiving a JWT-shaped token would itself be the
    2026-04-26 regression, so assert rather than return a payload.
    """

    async def _fake(token: str) -> dict[str, Any]:
        assert token.count(".") != 2, f"JWT-shaped token reached licence introspection: {token!r}"
        return {
            "active": True,
            "tenant_id": TENANT_A,
            "subject_id": "subj-1",
            "license_id": "lic-1",
            "roles": ["user"],
        }

    monkeypatch.setattr(w2v_auth, "_introspect_license", _fake)


def _make_app() -> FastAPI:
    """Real AuthMiddleware + a route that forwards exactly what the job routes forward.

    ``/api/bdd/forwarded`` stands in for ``qa_suite`` / ``documentation`` /
    ``accessibility`` / ``demo_video``, each of which passes
    ``license_token=get_license_token(request)`` into ``dispatch_job``.
    """
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/api/bdd/forwarded")
    async def _forwarded(request: Request) -> JSONResponse:
        return JSONResponse({"license_token": get_license_token(request)})

    return app


async def _forwarded_token(headers: dict[str, str]) -> tuple[int, str | None]:
    transport = httpx.ASGITransport(app=_make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/bdd/forwarded", headers=headers)
    if resp.status_code != 200:
        return resp.status_code, None
    return resp.status_code, resp.json()["license_token"]


# ---------------------------------------------------------------------------
# The defect itself
# ---------------------------------------------------------------------------


async def test_opaque_authorization_bearer_is_forwarded() -> None:
    """THE regression. Behind ``license-authz`` the licence rides ``Authorization``.

    The deleted copy read only ``X-Tenant-Id`` and returned ``None`` here, so
    ``dispatch_job`` authenticated background peer calls as nobody.
    """
    status, token = await _forwarded_token({"Authorization": f"Bearer {OPAQUE_TOKEN}"})

    assert status == 200
    assert token == OPAQUE_TOKEN


async def test_jwt_shaped_authorization_bearer_is_not_a_licence_token() -> None:
    """A JWT bearer is a Keycloak credential, never a licence token to forward.

    With no validator configured the request is refused; what must NEVER happen is a
    200 that forwards the JWT as if it were an opaque licence token.
    """
    status, token = await _forwarded_token({"Authorization": f"Bearer {JWT_SHAPED_BEARER}"})

    assert token != JWT_SHAPED_BEARER
    if status == 200:
        assert token is None


# ---------------------------------------------------------------------------
# The legacy shape must keep working, and must keep WINNING
# ---------------------------------------------------------------------------


async def test_legacy_x_tenant_id_bearer_still_forwarded() -> None:
    """Every current w2v caller still sends this shape; widening must not drop it."""
    status, token = await _forwarded_token({"X-Tenant-Id": f"Bearer {DELEGATED_TOKEN}"})

    assert status == 200
    assert token == DELEGATED_TOKEN


async def test_x_tenant_id_wins_when_both_headers_present() -> None:
    """Precedence is load-bearing tenant isolation, not cosmetics.

    ``BaseServiceClient`` puts the CALLER's delegated token on ``X-Tenant-Id`` and its
    own SERVICE-ACCOUNT token on ``Authorization``. Preferring ``Authorization`` would
    re-attribute every delegated call to the service account's tenant.
    """
    status, token = await _forwarded_token(
        {
            "X-Tenant-Id": f"Bearer {DELEGATED_TOKEN}",
            "Authorization": f"Bearer {OPAQUE_TOKEN}",
        }
    )

    assert status == 200
    assert token == DELEGATED_TOKEN, "service-account Authorization bearer stole the attribution"


async def test_no_credential_forwards_none() -> None:
    """No licence presented -> ``None``, the documented fall-back signal."""
    _status, token = await _forwarded_token({})

    assert token is None


# ---------------------------------------------------------------------------
# Anti-drift guards
# ---------------------------------------------------------------------------


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Drop every docstring so prose that NAMES a header is not read as parsing it."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            # Explicit annotation: `body[1:] or [ast.Pass()]` widens to a union
            # (`list[stmt] | list[Pass]`) that ty will not assign to `.body`.
            stripped: list[ast.stmt] = list(body[1:]) if len(body) > 1 else [ast.Pass()]
            node.body = stripped
    return tree


def test_background_does_not_reimplement_the_extraction_rule() -> None:
    """``src/services/background.py`` must never parse licence headers again.

    This is the guard the original defect needed and did not have: the copy did not
    fail, it drifted. Any re-introduced header parsing here is the same bug returning.
    """
    from pathlib import Path

    import src.services.background as background

    source = Path(background.__file__).read_text()
    # Strip comments AND docstrings: both legitimately NAME the header shapes while
    # explaining why the copy is gone. Only executable code may not touch them.
    code = ast.unparse(_strip_docstrings(ast.parse(source)))

    assert "def extract_license_token" not in code
    assert "request.headers" not in code, "background.py is parsing licence headers again"
    assert "X-Tenant-Id" not in code, "background.py is parsing licence headers again"
    assert not hasattr(background, "extract_license_token")


def test_helper_reads_middleware_state_not_headers() -> None:
    """The helper must source the token from ``request.state``, set by AuthMiddleware.

    A request whose headers carry a token but whose state was never populated (auth
    disabled, or a public endpoint) must yield ``None`` — proving the helper is not
    quietly re-deriving the value from headers.
    """
    from pathlib import Path

    import src.api.routes._helpers as helpers

    source = Path(helpers.__file__).read_text()
    code = ast.unparse(_strip_docstrings(ast.parse(source)))
    assert "request.headers" not in code

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/bdd/x",
        "headers": [(b"authorization", f"Bearer {OPAQUE_TOKEN}".encode())],
    }
    assert get_license_token(Request(scope)) is None
