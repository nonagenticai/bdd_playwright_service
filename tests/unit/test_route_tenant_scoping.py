"""Tenant-scoping tests for the three by-id job routes.

GET /api/bdd/{job_id}, GET /api/bdd/{job_id}/result, and GET /api/bdd/{job_id}/download
must only return a job that belongs to the caller's tenant. A job that exists but
belongs to a *different* tenant must 404 -- byte-identical to a genuine miss -- never
403 (a 403 would be an enumeration oracle over uuid7 ids).

Uses a real postgres-backed session (via the shared ``db_session_factory`` fixture)
so the tenant filter is exercised against an actual SQL WHERE clause, not a mock.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.services.job_manager import JobManager
from src.services.storage_service import StorageService

TENANT_A = "01950000-0000-7000-8000-00000000000a"
TENANT_B = "01950000-0000-7000-8000-00000000000b"


@pytest.fixture
def client(db_session_factory):
    """Real app wiring the jobs + packages routers against a real DB session.

    A test-only middleware sets ``request.state.tenant_id`` from an
    ``X-Test-Tenant`` header, standing in for the production auth middleware
    that the by-id routes read ``request.state.tenant_id`` from.
    """
    from w2v_common.asgi import register_error_envelope_handlers

    @asynccontextmanager
    async def _get_db_session():
        async with db_session_factory() as session:
            yield session

    app = FastAPI()
    register_error_envelope_handlers(app)

    @app.middleware("http")
    async def _set_tenant(request: Request, call_next):
        tenant = request.headers.get("X-Test-Tenant")
        if tenant is not None:
            request.state.tenant_id = tenant
        return await call_next(request)

    from src.api.routes import router as api_router

    app.include_router(api_router)

    with (
        patch("src.api.routes.jobs.get_db_session", _get_db_session),
        patch("src.api.routes.packages.get_db_session", _get_db_session),
        TestClient(app, raise_server_exceptions=True) as c,
    ):
        yield c


@pytest.fixture
async def two_jobs(db_session_factory):
    """Seed job_a (tenant-a) and job_b (tenant-b), both completed with a package."""
    async with db_session_factory() as session:
        job_a = await JobManager.create_job(session, tenant_id=TENANT_A, stream="qa_suite")
        job_b = await JobManager.create_job(session, tenant_id=TENANT_B, stream="qa_suite")
        await session.commit()

        await JobManager.mark_processing(session, job_a.job_id)
        await JobManager.mark_completed(
            session, job_a.job_id, artifact_paths={"package_zip": f"{job_a.job_id}/pkg.zip"}
        )
        await JobManager.mark_processing(session, job_b.job_id)
        await JobManager.mark_completed(
            session, job_b.job_id, artifact_paths={"package_zip": f"{job_b.job_id}/pkg.zip"}
        )

        return job_a.job_id, job_b.job_id


def _random_job_id() -> str:
    return str(uuid.uuid7())


class TestGetJobStatusTenantScoping:
    def test_own_tenant_200(self, client, two_jobs):
        job_a, _job_b = two_jobs
        resp = client.get(f"/api/bdd/{job_a}", headers={"X-Test-Tenant": TENANT_A})
        assert resp.status_code == 200
        assert resp.json()["job_id"] == job_a

    def test_cross_tenant_404(self, client, two_jobs):
        _job_a, job_b = two_jobs
        resp = client.get(f"/api/bdd/{job_b}", headers={"X-Test-Tenant": TENANT_A})
        assert resp.status_code == 404
        assert resp.json()["error"]["message"] == f"Job not found: {job_b}"

    def test_nonexistent_id_404(self, client, two_jobs):
        missing = _random_job_id()
        resp = client.get(f"/api/bdd/{missing}", headers={"X-Test-Tenant": TENANT_A})
        assert resp.status_code == 404
        assert resp.json()["error"]["message"] == f"Job not found: {missing}"

    def test_cross_tenant_and_genuine_miss_share_the_same_not_found_shape(self, client, two_jobs):
        """The wrong-tenant 404 and a truly-nonexistent-id 404 must be produced by
        the exact same code branch -- proven here by both matching the identical
        template (status + error envelope keys + message format), never a
        distinguishing "forbidden" body that would leak existence."""
        _job_a, job_b = two_jobs
        missing = _random_job_id()

        cross = client.get(f"/api/bdd/{job_b}", headers={"X-Test-Tenant": TENANT_A})
        genuine = client.get(f"/api/bdd/{missing}", headers={"X-Test-Tenant": TENANT_A})

        assert cross.status_code == genuine.status_code == 404
        cross_body, genuine_body = cross.json(), genuine.json()
        assert set(cross_body.keys()) == set(genuine_body.keys())
        assert set(cross_body["error"].keys()) == set(genuine_body["error"].keys())
        assert cross_body["error"]["status"] == genuine_body["error"]["status"]
        assert cross_body["error"]["code"] == genuine_body["error"]["code"]


class TestGetJobResultTenantScoping:
    def test_own_tenant_200(self, client, two_jobs):
        job_a, _job_b = two_jobs
        resp = client.get(f"/api/bdd/{job_a}/result", headers={"X-Test-Tenant": TENANT_A})
        assert resp.status_code == 200
        assert resp.json()["job_id"] == job_a

    def test_cross_tenant_404(self, client, two_jobs):
        _job_a, job_b = two_jobs
        resp = client.get(f"/api/bdd/{job_b}/result", headers={"X-Test-Tenant": TENANT_A})
        assert resp.status_code == 404
        assert resp.json()["error"]["message"] == f"Job not found: {job_b}"

    def test_nonexistent_id_404(self, client, two_jobs):
        missing = _random_job_id()
        resp = client.get(f"/api/bdd/{missing}/result", headers={"X-Test-Tenant": TENANT_A})
        assert resp.status_code == 404
        assert resp.json()["error"]["message"] == f"Job not found: {missing}"


class TestDownloadPackageTenantScoping:
    class _FakeStream:
        status = 200
        headers: ClassVar[dict[str, str]] = {"content-length": "3", "etag": '"abc"'}

        async def iter_chunks(self, chunk_size: int = 65536) -> AsyncIterator[bytes]:
            yield b"zip"

        async def aclose(self) -> None:
            return None

    def test_own_tenant_200(self, client, two_jobs):
        job_a, _job_b = two_jobs
        with patch.object(
            StorageService,
            "open_stream",
            new_callable=AsyncMock,
            return_value=self._FakeStream(),
        ):
            resp = client.get(f"/api/bdd/{job_a}/download", headers={"X-Test-Tenant": TENANT_A})
        assert resp.status_code == 200
        assert resp.content == b"zip"

    def test_cross_tenant_404(self, client, two_jobs):
        _job_a, job_b = two_jobs
        resp = client.get(f"/api/bdd/{job_b}/download", headers={"X-Test-Tenant": TENANT_A})
        assert resp.status_code == 404
        assert resp.json()["error"]["message"] == f"Job not found: {job_b}"

    def test_nonexistent_id_404(self, client, two_jobs):
        missing = _random_job_id()
        resp = client.get(f"/api/bdd/{missing}/download", headers={"X-Test-Tenant": TENANT_A})
        assert resp.status_code == 404
        assert resp.json()["error"]["message"] == f"Job not found: {missing}"


class TestJobManagerGetJobNoTenantIsUnscoped:
    """Worker regression guard: the background job_processor calls
    ``JobManager.get_job(db, job_id)`` with NO tenant_id (cross-tenant by
    design, since a worker processes any tenant's job). That must keep
    working after tenant_id becomes an optional filter kwarg."""

    async def test_get_job_without_tenant_returns_row_regardless_of_owner(self, db_session):
        job = await JobManager.create_job(db_session, tenant_id=TENANT_B, stream="qa_suite")
        await db_session.commit()

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id

    async def test_get_job_with_matching_tenant_returns_row(self, db_session):
        job = await JobManager.create_job(db_session, tenant_id=TENANT_A, stream="qa_suite")
        await db_session.commit()

        fetched = await JobManager.get_job(db_session, job.job_id, tenant_id=TENANT_A)
        assert fetched is not None
        assert fetched.job_id == job.job_id

    async def test_get_job_with_mismatched_tenant_returns_none(self, db_session):
        job = await JobManager.create_job(db_session, tenant_id=TENANT_B, stream="qa_suite")
        await db_session.commit()

        fetched = await JobManager.get_job(db_session, job.job_id, tenant_id=TENANT_A)
        assert fetched is None
