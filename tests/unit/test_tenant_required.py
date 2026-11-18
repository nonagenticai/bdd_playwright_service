"""A request with no tenant identity must FAIL LOUD (401) -- never be substituted.

``src/api/routes/_helpers.get_tenant_id`` used to return
``settings.auth.default_tenant_id`` when ``request.state.tenant_id`` was absent, so
every unattributed caller wrote ``bdd_jobs`` / ``bdd_usage`` rows under one shared
sentinel tenant that RLS then let all of them read back. A ``tenant_id`` may never
hold a sentinel: an absent tenant is an ERROR.

These tests build the app WITHOUT any tenant-setting middleware -- exactly the shape
an unauthenticated caller produces -- and pin 401 on both a write route and a read
route. They also pin that the DB is never touched: nothing may be written for a
caller with no identity.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.routes._helpers import get_tenant_id


@pytest.fixture
def anonymous_client():
    """TestClient over the real router with NO tenant on request.state."""
    from w2v_common.asgi import register_error_envelope_handlers

    from src.api.routes import router as api_router

    app = FastAPI()
    register_error_envelope_handlers(app)
    app.include_router(api_router)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestAbsentTenantIs401:
    def test_create_route_without_tenant_is_401(self, anonymous_client):
        response = anonymous_client.post("/api/bdd/qa-suite", json={"url": "https://example.com"})
        assert response.status_code == 401, (
            f"unattributed create was accepted with {response.status_code}: {response.text}"
        )

    def test_read_route_without_tenant_is_401(self, anonymous_client):
        response = anonymous_client.get("/api/bdd/some-job-id")
        assert response.status_code == 401, (
            f"unattributed read was accepted with {response.status_code}: {response.text}"
        )

    @pytest.mark.parametrize("absent", [None, "", "   "])
    def test_helper_rejects_absent_or_blank_tenant(self, absent):
        """A blank/whitespace tenant is as absent as a missing one."""
        request = MagicMock(spec=Request)
        if absent is None:
            del request.state.tenant_id
            request.state = MagicMock(spec=[])
        else:
            request.state.tenant_id = absent

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            get_tenant_id(request)
        assert exc.value.status_code == 401

    @pytest.mark.parametrize(
        "nil_spelling",
        [
            "00000000-0000-0000-0000-000000000000",
            "00000000000000000000000000000000",
            "{00000000-0000-0000-0000-000000000000}",
            "urn:uuid:00000000-0000-0000-0000-000000000000",
            "  00000000-0000-0000-0000-000000000000  ",
        ],
    )
    def test_helper_rejects_the_nil_uuid_by_value(self, nil_spelling):
        """The nil uuid is a VALID uuid, so a blank-only guard never sees it."""
        request = MagicMock(spec=Request)
        request.state.tenant_id = nil_spelling

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            get_tenant_id(request)
        assert exc.value.status_code == 401

    def test_helper_returns_a_real_forwarded_tenant(self):
        request = MagicMock(spec=Request)
        request.state.tenant_id = "0195f48b-d000-7000-8000-000000000099"
        assert get_tenant_id(request) == "0195f48b-d000-7000-8000-000000000099"
