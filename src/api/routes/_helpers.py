"""Small shared helpers for route handlers."""

from __future__ import annotations

import uuid

from fastapi import HTTPException, Request


def get_tenant_id(request: Request) -> str:
    """Return the caller's real tenant id, or 401 -- never a sentinel default.

    ``request.state.tenant_id`` is set by ``w2v_common.auth.middleware.AuthMiddleware``
    (registered for this service in ``src/asgi.py``) from the JWT ``tenant_id`` claim,
    the licenses-service introspection payload, or -- only when
    ``AUTH_TRUST_INCLUSTER_IDENTITY=true`` -- the ``X-Tenant-Id`` header on a trusted
    in-cluster network.

    This used to fall back to ``settings.auth.default_tenant_id``, so every
    unattributed caller silently shared one bogus tenant and every ``bdd_jobs`` /
    ``bdd_usage`` / ``narration_templates`` row it wrote landed in a bucket that RLS
    then let all of them read. Absent identity is an ERROR: a request with no real
    tenant forwarded by the requestor has nothing legitimate to read or write.
    """
    tenant_id = (getattr(request.state, "tenant_id", None) or "").strip()
    # The nil UUID is the most dangerous spelling of "no tenant": it is a VALID
    # uuid, so it passes every cast and every dirty-row audit while billing
    # buckets it as unattributed and never charges it. Reject it BY VALUE, in
    # every spelling (dashed, undashed, braced, urn:, any case) -- a
    # ``None``/blank-only guard never sees it.
    if tenant_id:
        try:
            if uuid.UUID(tenant_id).int == 0:
                tenant_id = ""
        except ValueError:
            pass
    if not tenant_id:
        raise HTTPException(
            status_code=401,
            detail=(
                "No tenant identity on the request. bdd_playwright_service is "
                "tenant-scoped: authenticate with a license token or JWT carrying a "
                "tenant_id claim."
            ),
        )
    return tenant_id


def get_license_token(request: Request) -> str | None:
    """Return the caller's opaque licence token, or ``None`` when there is none.

    ⛔ DO NOT RE-DERIVE THIS FROM HEADERS. ``request.state.license_token`` is set by
    ``w2v_common.auth.middleware.AuthMiddleware`` on EVERY non-public dispatch branch,
    and it is the token the middleware already extracted, introspected and found
    ``active``. Reading it here is the only way this service stays on the canonical
    rule when that rule moves.

    This helper replaces a local ``extract_license_token`` in ``src/services/background``
    that re-implemented w2v_common's PRE-0.74.0 rule: it read the token ONLY from
    ``X-Tenant-Id: Bearer <token>``. 0.74.0 widened the canonical rule to ALSO accept
    an opaque ``Authorization: Bearer <token>`` — required so w2v can sit behind the
    ``license-authz`` ext-authz gate, which reads the token ONLY from ``Authorization``.
    Behind that gate the copy returned ``None``, ``dispatch_job`` got
    ``license_token=None``, and background peer calls silently fell back to
    trusted-network auth. No test failed; the copy simply drifted.

    Precedence (why a copy is dangerous, not merely redundant): the canonical rule
    checks ``X-Tenant-Id: Bearer`` FIRST and ``Authorization: Bearer`` only for a
    NON-JWT-shaped bearer. ``BaseServiceClient`` puts the CALLER's delegated token on
    ``X-Tenant-Id`` while its own SERVICE-ACCOUNT token sits on ``Authorization``, so
    preferring ``Authorization`` re-attributes every delegated call to the service
    account's tenant. Getting that order wrong in a copy is a tenant-isolation bug.

    Returns ``None`` when auth is disabled (``settings.auth.enabled=false``, so the
    middleware never ran and the attribute is absent) and on the JWT and
    trusted-network lanes, which carry no opaque token to forward. ``None`` is the
    documented "fall back to the service-account/trusted-network credential" signal.
    """
    return getattr(request.state, "license_token", None) or None
