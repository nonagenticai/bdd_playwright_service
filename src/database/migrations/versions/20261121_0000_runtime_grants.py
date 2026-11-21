"""Grant the RUNTIME role its table privileges (the role name moved off w2v).

``provision-service-db.sh`` deliberately creates the app role with CONNECT only:
"table/schema grants are the service's own migrations' job". But the chain never
granted them — ``0003_rls``/``0005_prune_pending`` hardcode
``w2v_bdd_playwright_app`` for the two pruning FUNCTIONS only, and the renamed
deployment role (``bdd_playwright_app``) got nothing. On the first in-cluster
deploy every ``INSERT INTO bdd_jobs`` answered
``permission denied for table bdd_jobs`` (HTTP 500), even though CI and the owner
path were green because both connect as the owner.

This migration makes the grant part of the schema, not an out-of-band step:

* tables: SELECT/INSERT/UPDATE/DELETE (RLS still restricts rows; the role has
  NOBYPASSRLS)
* sequences: USAGE/SELECT (uuid columns use gen_random_uuid, but sequences exist)
* functions: EXECUTE (RLS resolver + the two pruning functions)
* ALTER DEFAULT PRIVILEGES for the role running the migration (the owner), so
  tables/functions added by later migrations inherit the same grants.

WHICH ROLE. ``BDD_RUNTIME_DB_ROLE`` when set; otherwise the first of
``bdd_playwright_app`` / ``w2v_bdd_playwright_app`` that exists in ``pg_roles``.
None existing = no-op, so a dev/CI database with a single owner user is
untouched. A configured-but-missing role raises: silently skipping a role the
operator asked for is the "green while doing nothing" failure this migration
exists to end.
"""

from __future__ import annotations

import os
import re

from alembic import op

revision: str = "0007_runtime_grants"
down_revision: str | None = "0006_step_coverage"
branch_labels: str | None = None
depends_on: str | None = None

_ENV = "BDD_RUNTIME_DB_ROLE"
_FALLBACKS = ("bdd_playwright_app", "w2v_bdd_playwright_app")
#: Role names are interpolated into GRANT statements (DDL cannot bind them as
#: parameters), so accept only the shape provision-service-db.sh creates.
_ROLE_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")

_GRANTS = (
    "GRANT USAGE ON SCHEMA public TO {role}",
    "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {role}",
    "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}",
    "GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO {role}",
    "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT "
    "SELECT, INSERT, UPDATE, DELETE ON TABLES TO {role}",
    "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO {role}",
    "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO {role}",
)

_REVOKES = (
    "ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE EXECUTE ON FUNCTIONS FROM {role}",
    "ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE USAGE, SELECT ON SEQUENCES FROM {role}",
    "ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE "
    "SELECT, INSERT, UPDATE, DELETE ON TABLES FROM {role}",
    "REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA public FROM {role}",
    "REVOKE USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public FROM {role}",
    "REVOKE SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public FROM {role}",
    "REVOKE USAGE ON SCHEMA public FROM {role}",
)


def _ddl(template: str, role: str) -> str:
    """Render one statement for a role proven to match _ROLE_RE."""
    return template.format(role=f'"{role}"')


def _resolve_role(connection) -> str | None:
    configured = (os.environ.get(_ENV) or "").strip()
    if configured:
        if not _ROLE_RE.match(configured):
            raise RuntimeError(f"{_ENV}={configured!r} is not a valid Postgres role name")
        exists = connection.exec_driver_sql(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (configured,)
        ).scalar()
        if not exists:
            raise RuntimeError(f"{_ENV}={configured!r} does not exist in pg_roles")
        return configured
    for candidate in _FALLBACKS:
        exists = connection.exec_driver_sql(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (candidate,)
        ).scalar()
        if exists:
            return candidate
    return None


def upgrade() -> None:
    role = _resolve_role(op.get_bind())
    if role is None:
        return
    for template in _GRANTS:
        op.execute(_ddl(template, role))


def downgrade() -> None:
    """Revoke what upgrade granted. Never runs on the prod path (forward-only)."""
    role = _resolve_role(op.get_bind())
    if role is None:
        return
    for template in _REVOKES:
        op.execute(_ddl(template, role))
