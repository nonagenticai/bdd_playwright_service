"""P7: a detached BDD job's in-cell outbound calls carry a FRESH cell token.

The job runs in an ``asyncio`` task long after the request that started it
returned. The in-cell auth hook (``w2v_common`` ``identity_event_hook``) mints
an ``X-W2V-Cell-Token`` per send, but ONLY from what is bound in the task's
context: a tenant with a VOUCHED provenance, plus the run / project / actor.
These tests prove the job binds exactly the identity the ORIGINATING request
was authenticated as — via the real ``AuthMiddleware`` cell lane, not a seeded
ContextVar — and that the binding does not depend on the forwarded licence
(on the cell lane there is none to forward).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import uuid
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI, Request
from jwt.algorithms import OKPAlgorithm
from w2v_common.auth import cell_jwt, clear_identity_context
from w2v_common.auth.cell_jwt import (
    CELL_TOKEN_HEADER,
    CellClaims,
    mint_cell_token,
    reset_cell_state,
    verify_cell_token,
)
from w2v_common.auth.identity_context import identity_event_hook, set_identity_context
from w2v_common.auth.middleware import AuthMiddleware
from w2v_common.discovery import service_url
from w2v_common.tenant import MissingTenantError, TenantSource

from src.api.routes._helpers import get_license_token, get_tenant_id
from src.services import background
from src.services.task_identity import TaskIdentity, capture_task_identity

ORCH = "w2v-orchestrator"
SELF = "bdd-playwright-service"
CALLEE = "app-inspector-service"
DEPLOYMENT = "strato-w2v-test"
TENANT = uuid.UUID("0198f1d2-0000-7000-8000-0000000c3111")
ACTOR = "0198f1d2-0000-7000-8000-0000000ac701"
PROJECT = "0198f1d2-0000-7000-8000-0000000b0001"
RUN = "0198f1d2-0000-7000-8000-00000000a0a1"
SUBJECT = "0198f1d2-0000-7000-8000-0000000c5501"
LICENSE = "0198f1d2-0000-7000-8000-0000000d1c01"
#: What the originating request is attributed to. On w2v_common 0.92.x the cell
#: lane takes these from the forwarded headers; from 0.93.0 they are SIGNED
#: claims of the inbound token and the outbound token must re-sign them.
_ATTRIBUTION_HEADERS = {"X-Subject-Id": SUBJECT, "X-License-Id": LICENSE}
_SIGNS_ATTRIBUTION = "license_id" in inspect.signature(mint_cell_token).parameters


def _jwk(private: Ed25519PrivateKey, kid: str, iss: str) -> dict[str, Any]:
    entry = json.loads(OKPAlgorithm.to_jwk(private.public_key()))
    entry.update({"kid": kid, "iss": iss})
    return entry


def _pem(private: Ed25519PrivateKey) -> str:
    return private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


@pytest.fixture
def keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Ed25519PrivateKey]]:
    """Orchestrator + this service keys in a JWKS file; this process signs as bdd."""
    orch, own = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps({"keys": [_jwk(orch, "orch-1", ORCH), _jwk(own, "bdd-1", SELF)]}))
    for name in ("W2V_LOCAL", "CELL_JWKS_RELOAD_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CELL_JWKS_PATH", str(jwks))
    monkeypatch.setenv("CELL_DEPLOYMENT_ID", DEPLOYMENT)
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", "*")
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    _sign_as(monkeypatch, own, "bdd-1", SELF)
    reset_cell_state()
    clear_identity_context()
    yield {"orch": orch, "own": own}
    clear_identity_context()
    reset_cell_state()


def _sign_as(monkeypatch: pytest.MonkeyPatch, key: Ed25519PrivateKey, kid: str, iss: str) -> None:
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", _pem(key))
    monkeypatch.setenv("CELL_JWT_KID", kid)
    monkeypatch.setenv("CELL_SERVICE_NAME", iss)


def _orchestrator_token(monkeypatch: pytest.MonkeyPatch, keys: dict[str, Ed25519PrivateKey]) -> str:
    _sign_as(monkeypatch, keys["orch"], "orch-1", ORCH)
    signed: dict[str, Any] = (
        {"subject_id": SUBJECT, "license_id": LICENSE} if _SIGNS_ATTRIBUTION else {}
    )
    token = mint_cell_token(SELF, tenant=TENANT, run=RUN, actor=ACTOR, project=PROJECT, **signed)
    _sign_as(monkeypatch, keys["own"], "bdd-1", SELF)
    return token


def _verify_at_callee(monkeypatch: pytest.MonkeyPatch, token: str, now: float) -> CellClaims:
    """Verify as the CALLEE would: aud must be the callee's own cell name."""
    with monkeypatch.context() as m:
        m.setenv("CELL_SERVICE_NAME", CALLEE)
        return verify_cell_token(token, now=now)


@contextlib.asynccontextmanager
async def _fake_db_session() -> AsyncGenerator[MagicMock]:
    yield MagicMock()


def _outbound_client(sent: list[httpx.Request]) -> httpx.AsyncClient:
    """The same request hook ``BaseServiceClient`` installs, on a recording transport."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"ok": True})

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        event_hooks={"request": [identity_event_hook]},
    )


def _app(processor: MagicMock) -> FastAPI:
    """A route that dispatches exactly as the real BDD routes do."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.post("/api/bdd/go")
    async def go(request: Request) -> dict[str, str]:
        background.dispatch_job(
            processor, "job-1", get_tenant_id(request), license_token=get_license_token(request)
        )
        return {"status": "accepted"}

    return app


async def _drain_jobs() -> None:
    pending = [t for t in background._background_tasks if not t.done()]
    if pending:
        await asyncio.wait_for(asyncio.gather(*pending), timeout=10)


async def test_detached_job_outbound_carries_cell_token_for_the_originating_request(
    monkeypatch: pytest.MonkeyPatch, keys: dict[str, Ed25519PrivateKey]
) -> None:
    """Inbound cell token → detached job → in-cell call: same tenant, run, actor, project."""
    sent: list[httpx.Request] = []
    in_job: dict[str, Any] = {}

    async def process_job(
        _db: object, _job_id: str, _tenant_id: str, license_token: Any = None
    ) -> None:
        in_job["license_token"] = license_token
        # Detach from the request for real: the job outlives the response.
        await asyncio.sleep(0)
        async with _outbound_client(sent) as client:
            await client.post(f"{service_url('app_inspector')}/api/inspect", json={})

    processor = MagicMock()
    processor.process_job = AsyncMock(side_effect=process_job)
    monkeypatch.setattr(background, "get_db_session", _fake_db_session)
    monkeypatch.setattr(background, "_acquire_concurrency_slot", AsyncMock(return_value=False))

    inbound = _orchestrator_token(monkeypatch, keys)
    transport = httpx.ASGITransport(app=_app(processor))
    async with httpx.AsyncClient(transport=transport, base_url="http://bdd") as client:
        resp = await client.post(
            "/api/bdd/go", headers={CELL_TOKEN_HEADER: inbound, **_ATTRIBUTION_HEADERS}
        )
    assert resp.status_code == 200, resp.text
    await _drain_jobs()

    # The cell lane forwards no licence: the cell binding must stand on its own.
    assert in_job["license_token"] is None
    assert len(sent) == 1
    token = sent[0].headers.get(CELL_TOKEN_HEADER)
    assert token, "the detached job's in-cell call went out with no cell token (enforce → 401)"
    assert token != inbound, "the inbound token was replayed instead of a fresh mint"
    claims = _verify_at_callee(monkeypatch, token, now=cell_jwt.time.time())
    assert claims.issuer == SELF
    assert claims.audience == CALLEE
    assert claims.tenant_id == TENANT
    assert claims.run == RUN
    assert claims.project == PROJECT
    assert claims.actor == ACTOR, "the human actor was dropped on the way into the job"
    # Who pays: the request's licence + subject reach the wire (headers on every
    # version; signed claims from 0.93.0), never NULL — a dropped licence is an
    # unattributable cost once receivers enforce.
    assert sent[0].headers.get("X-License-Id") == LICENSE
    assert sent[0].headers.get("X-Subject-Id") == SUBJECT
    if _SIGNS_ATTRIBUTION:
        assert getattr(claims, "license_id", None) == LICENSE
        assert getattr(claims, "subject", None) == SUBJECT


async def test_job_running_past_token_ttl_mints_fresh_per_call(
    monkeypatch: pytest.MonkeyPatch, keys: dict[str, Ed25519PrivateKey]
) -> None:
    """A job still running 61 s later mints a NEW token; nothing minted at start is held."""
    clock = {"now": 1_900_000_000.0}
    monkeypatch.setattr(cell_jwt, "time", SimpleNamespace(time=lambda: clock["now"]))
    sent: list[httpx.Request] = []

    async def process_job(
        _db: object, _job_id: str, _tenant_id: str, license_token: Any = None
    ) -> None:
        async with _outbound_client(sent) as client:
            await client.get(f"{service_url('app_inspector')}/api/inspect/x")
            clock["now"] += cell_jwt.CELL_TOKEN_TTL_SECONDS + 1
            await client.get(f"{service_url('app_inspector')}/api/inspect/x")

    processor = MagicMock()
    processor.process_job = AsyncMock(side_effect=process_job)
    monkeypatch.setattr(background, "get_db_session", _fake_db_session)
    monkeypatch.setattr(background, "_acquire_concurrency_slot", AsyncMock(return_value=False))

    identity = TaskIdentity(
        tenant_id=TENANT,
        tenant_source=TenantSource.CELL_JWT,
        run=RUN,
        project=PROJECT,
        actor=ACTOR,
        subject_id=SUBJECT,
        license_id=LICENSE,
    )
    await background._process_job_bg(processor, "job-1", str(TENANT), identity=identity)

    first, second = (r.headers[CELL_TOKEN_HEADER] for r in sent)
    assert first != second
    # Verified a little later still (past the callee's clock-skew allowance): the
    # token minted at job start is dead, the one minted on the late call is live.
    verify_at = clock["now"] + 10
    late = _verify_at_callee(monkeypatch, second, now=verify_at)
    assert late.tenant_id == TENANT
    assert late.run == RUN
    assert late.issued_at == int(clock["now"])
    with pytest.raises(cell_jwt.CellTokenError):
        _verify_at_callee(monkeypatch, first, now=verify_at)


def test_capture_without_a_bound_tenant_fails_loud(keys: dict[str, Ed25519PrivateKey]) -> None:
    """No tenant in request scope → refuse at dispatch, never a tenant-less job."""
    clear_identity_context()
    with pytest.raises(MissingTenantError):
        capture_task_identity(str(TENANT))


def test_dispatch_without_a_bound_tenant_fails_loud_and_schedules_nothing(
    keys: dict[str, Ed25519PrivateKey],
) -> None:
    async def drive() -> None:
        clear_identity_context()
        before = set(background._background_tasks)
        with pytest.raises(MissingTenantError):
            background.dispatch_job(MagicMock(), "job-1", str(TENANT))
        assert set(background._background_tasks) == before

    asyncio.run(drive())


def test_capture_refuses_a_tenant_that_is_not_the_authenticated_one(
    keys: dict[str, Ed25519PrivateKey],
) -> None:
    other = "0198f1d2-0000-7000-8000-0000000c3222"
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.CELL_JWT)
    with pytest.raises(MissingTenantError):
        capture_task_identity(other)
