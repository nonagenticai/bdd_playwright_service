"""The entitlement gate on ``/api/bdd`` actually refuses an unlicensed tenant.

``bdd_playwright_service`` is a BILLED destination — it is listed in
``w2v_common.auth.outbound.METERED_DESTINATION_SERVICES`` — yet until now it
carried no entitlement code at all. ``AuthMiddleware`` answers "who are you",
never "may you", so a tenant whose licence was ``expired`` or ``revoked`` still
presented a syntactically valid token and every route under ``/api/bdd`` served
them. ``LICENSE_GATE`` is mounted router-wide in ``src/api/routes/__init__.py``
to close that hole.

Every test here drives a REAL HTTP request through the REAL routers, because
the interesting question is not whether ``require_license`` works — w2v_common
owns that — but whether THIS service's routes actually carry it. FastAPI
includes sub-routers lazily (``_IncludedRouter``): a parent router's
``dependencies=`` are applied when a request is matched, not baked into the
route objects at import time, so static introspection of ``route.dependant``
reports every one of the seven sub-routers below as UNGATED — a false negative.
A request is the only honest probe.

The paired 200s (entitled, and flag-off) are what stop this file from passing
vacuously: if the gate were mounted wrongly, or the fixture were broken, those
cases would not be 200s either.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth import clear_license_status_cache
from w2v_common.auth.licenses_read_client import LicensesClientError

from src.api.health import router as health_router
from src.api.routes import router as api_router

TENANT = "3f2b1c9e-0a4d-4f6b-8c1e-7d5a9b2c4e60"
SUBJECT = "user-under-test"

#: The cheapest genuinely tenant-scoped route: once ``get_db_session`` and
#: ``JobManager.get_job`` are stubbed it does no I/O of its own, so a non-200
#: here can only be the gate — which is the whole point. It is also the route
#: used for the paired 200s.
GATED_PATH_JOBS = "/api/bdd/job-under-test"

#: One representative path per sub-router mounted under ``/api/bdd``. The 402
#: cases need no handler stubs at all: a router-level dependency is solved
#: BEFORE the endpoint runs, so the refusal happens without the handler ever
#: touching the database, storage, or the job dispatcher. Enumerating all seven
#: is the point — the gate is mounted on the PARENT, and this is what proves the
#: inheritance is real for every child rather than for the one we happened to
#: test.
GATED_PATHS: tuple[tuple[str, str, str], ...] = (
    ("jobs", "GET", "/api/bdd/job-under-test"),
    ("jobs_result", "GET", "/api/bdd/job-under-test/result"),
    ("packages", "GET", "/api/bdd/job-under-test/download"),
    ("qa_suite", "POST", "/api/bdd/qa-suite"),
    ("accessibility", "POST", "/api/bdd/accessibility"),
    ("documentation", "POST", "/api/bdd/documentation"),
    ("demo_video", "POST", "/api/bdd/demo-video"),
    ("gherkin", "POST", "/api/bdd/gherkin/upload"),
)

#: Probes. Must answer 200 for a revoked tenant: a 402/503 on a kubelet probe
#: CrashLoops the pod, and an unlicensed tenant cannot be billed by killing our
#: own deployment. These live on the health router, which is NOT gated.
PROBE_PATHS = ("/health/live", "/health/ready", "/healthz", "/readyz")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Record:
    """A licences wire record. Only ``model_dump`` is on the gate's path."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, Any]:
        return dict(self._fields)


class _Detail(_Record):
    """An ACTIVE licence detail (the 200 from ``GET /licenses/active``)."""

    conflict = False


class _FakeLicenses:
    """Stands in for ``LicensesReader``.

    ``active=None`` models the authoritative 404 ("no active licence"), which
    sends the gate to the subject inventory to refine none/expired/revoked.
    ``error`` models an upstream failure, which must 503, never allow.
    """

    def __init__(
        self,
        *,
        active: _Detail | None = None,
        inventory: list[_Record] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._active = active
        self._inventory = inventory or []
        self._error = error

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> _Detail | None:
        if self._error is not None:
            raise self._error
        return self._active

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[_Record]:
        if self._error is not None:
            raise self._error
        return list(self._inventory)


def _revoked() -> list[_Record]:
    return [_Record(app={"key": "w2v"}, tenant_id=TENANT, status="revoked")]


def _expired() -> list[_Record]:
    return [
        _Record(
            app={"key": "w2v"},
            tenant_id=TENANT,
            status="expired",
            expires_at="2020-01-01T00:00:00+00:00",
        )
    ]


def _entitled() -> _Detail:
    return _Detail(
        app={"key": "w2v"},
        tenant_id=TENANT,
        status="active",
        expires_at="2099-01-01T00:00:00+00:00",
        plan={"code": "pro"},
    )


class _FakeJob:
    """Just the attributes ``get_job_status`` projects into JobStatusResponse."""

    job_id = "job-under-test"
    status = "completed"
    stream = "qa_suite"
    features_generated = 1
    step_defs_generated = 1
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    completed_at = datetime(2026, 1, 1, tzinfo=UTC)
    error_message = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_cached_verdicts():
    """The gate caches 60s per (tenant, app, subject). Two tests in one process
    would otherwise share a verdict and the second would assert nothing."""
    clear_license_status_cache()
    yield
    clear_license_status_cache()


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch):
    """Default the flag OFF, exactly as it ships. Each test arms it explicitly."""
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    """Stub the ONLY I/O on the jobs route, so a non-200 can only be the gate.

    ``get_db_session`` is re-bound in ``src.api.routes.jobs`` (a ``from`` import),
    so it must be patched THERE, not at its definition site.
    """

    @asynccontextmanager
    async def _session():
        yield MagicMock()

    monkeypatch.setattr("src.api.routes.jobs.get_db_session", _session)
    monkeypatch.setattr(
        "src.api.routes.jobs.JobManager.get_job", AsyncMock(return_value=_FakeJob())
    )


@pytest.fixture
def app() -> FastAPI:
    """The REAL routers, with identity stamped the way AuthMiddleware stamps it."""
    application = FastAPI(redirect_slashes=False)

    class _StampIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = TENANT
            request.state.subject_id = SUBJECT
            return await call_next(request)

    application.add_middleware(_StampIdentity)
    application.include_router(health_router)
    application.include_router(api_router)
    return application


async def _request(app: FastAPI, method: str, path: str) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.request(method, path)


def _licenses(monkeypatch, fake: _FakeLicenses) -> None:
    """Patch the client the gate resolves through, in the module that calls it."""
    monkeypatch.setattr(
        "w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake
    )


# ---------------------------------------------------------------------------
# Armed: an unentitled tenant is refused, on EVERY sub-router
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "method", "path"), GATED_PATHS, ids=[g[0] for g in GATED_PATHS])
async def test_armed_gate_402s_an_unentitled_tenant_on_every_subrouter(
    app, monkeypatch, name, method, path
):
    """A revoked tenant is refused on all seven sub-routers under /api/bdd.

    Note the POSTs carry no body at all: the router-level gate is solved before
    the endpoint's request model is validated, so a 402 (not a 422) is the
    correct — and stricter — assertion. An unlicensed tenant should not get as
    far as having their payload inspected.
    """
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _request(app, method, path)

    assert response.status_code == 402, f"{name}: {response.status_code} {response.text}"
    assert response.json()["detail"]["error"] == "license_revoked"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inventory", "code"),
    [
        ([], "license_none"),
        (_revoked(), "license_revoked"),
        (_expired(), "license_expired"),
    ],
    ids=["none", "revoked", "expired"],
)
async def test_armed_gate_reports_the_state_specific_refusal(app, monkeypatch, inventory, code):
    """402 with the state-specific code — the tenant's licence is the problem."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))

    response = await _request(app, "GET", GATED_PATH_JOBS)

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == code


@pytest.mark.asyncio
async def test_armed_gate_503s_when_entitlement_cannot_be_established(app, monkeypatch):
    """An upstream failure is NOT an allow, and NOT a 402 either: we do not know."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))

    response = await _request(app, "GET", GATED_PATH_JOBS)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "license_unavailable"
    assert response.headers["Retry-After"]


# ---------------------------------------------------------------------------
# Armed: an entitled tenant is served (the non-vacuity pair)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_armed_gate_serves_an_entitled_tenant(app, monkeypatch):
    """Without this, every assertion above would still pass on a route that is
    simply broken. It is the control."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    response = await _request(app, "GET", GATED_PATH_JOBS)

    assert response.status_code == 200, response.text
    assert response.json()["job_id"] == "job-under-test"


# ---------------------------------------------------------------------------
# Dormant on merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_is_dormant_while_the_flag_is_unset(app, monkeypatch):
    """Merging this changes NOTHING until gitops arms the flag. A revoked tenant
    is still served — deliberately — so the rollout is a flag flip, not a deploy."""
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _request(app, "GET", GATED_PATH_JOBS)

    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_gate_is_dormant_when_the_flag_is_explicitly_false(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _request(app, "GET", GATED_PATH_JOBS)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Probes are never gated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("path", PROBE_PATHS)
async def test_probe_paths_are_never_gated(app, monkeypatch, path):
    """A gated probe path CrashLoops the pod. These must never answer 402/503
    BECAUSE OF THE GATE — /health/ready may legitimately 503 on a dead DB, so we
    assert on the body, not merely the code."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _request(app, "GET", path)

    body = response.text
    assert "license_revoked" not in body, f"{path} was gated: {body}"
    assert "license_unavailable" not in body, f"{path} was gated: {body}"


@pytest.mark.asyncio
async def test_liveness_probes_answer_200_for_a_revoked_tenant(app, monkeypatch):
    """The two probes that touch no dependency must be a flat 200."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    for path in ("/health/live", "/healthz"):
        response = await _request(app, "GET", path)
        assert response.status_code == 200, f"{path}: {response.text}"


@pytest.mark.asyncio
async def test_the_services_own_probe_paths_are_exempt_by_path():
    """Belt and braces: even if a future refactor mounted the gate on the health
    router, w2v_common would still refuse to enforce on these paths — and we do
    NOT hand-roll that list, which is what keeps it correct."""
    from w2v_common.auth import is_license_exempt_path

    for path in ("/health", "/health/live", "/health/ready", "/healthz", "/ready"):
        assert is_license_exempt_path(path), path
    for _name, _method, path in GATED_PATHS:
        assert not is_license_exempt_path(path), path


@pytest.mark.asyncio
async def test_this_service_is_a_metered_destination():
    """The reason the gate belongs here at all. If this service were ever removed
    from the metered list, this file's premise would need re-examining."""
    from w2v_common.auth.outbound import METERED_DESTINATION_SERVICES

    assert "bdd_playwright" in METERED_DESTINATION_SERVICES
