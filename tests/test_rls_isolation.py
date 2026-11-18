"""Proves the Phase D RLS policies ISOLATE tenants (not just that they exist).

Connects AS a throwaway non-owner ``NOBYPASSRLS`` role — the posture the runtime
``w2v_bdd_playwright_app`` role has under enforce — and asserts the shipped policies
(applied by ``register_rls_ddl`` on ``create_all``) actually scope reads/writes.
Needs a superuser test DB (``CREATE ROLE`` + ``SET ROLE``); skips loudly in CI
where the role lacks that privilege.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

TENANT_A = "01950000-0000-7000-8000-0000000000aa"
TENANT_B = "01950000-0000-7000-8000-0000000000bb"
_ROLE = "w2v_bdd_playwright_app_rls_test"
_INS = (
    "INSERT INTO bdd_jobs (job_id, tenant_id, status, stream, created_at, updated_at) "
    "VALUES (:job_id, :tenant, 'pending', 'qa_suite', now(), now())"
)


@pytest.mark.asyncio
async def test_rls_isolates_tenants_under_restricted_role(db_engine) -> None:
    async with db_engine.begin() as conn:
        # skip-gate FIRST (before the RLS-affected seed): a non-superuser CI role
        # can't CREATE ROLE and its seed would hit the policy with no GUC.
        try:
            await conn.execute(text(f'CREATE ROLE "{_ROLE}" NOSUPERUSER NOBYPASSRLS NOINHERIT'))
        except ProgrammingError as exc:
            pytest.skip(f"cannot CREATE ROLE (needs superuser/CREATEROLE): {exc.orig}")
        await conn.execute(text(f'GRANT SELECT, INSERT, UPDATE, DELETE ON bdd_jobs TO "{_ROLE}"'))

        await conn.execute(text(_INS), {"job_id": "job-a", "tenant": TENANT_A})
        await conn.execute(text(_INS), {"job_id": "job-b", "tenant": TENANT_B})

        try:
            await conn.execute(text(f'SET LOCAL ROLE "{_ROLE}"'))

            n = (await conn.execute(text("SELECT count(*) FROM bdd_jobs"))).scalar()
            assert n == 0, f"leaked {n} rows with no GUC"

            await conn.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": TENANT_A}
            )
            # ``tenant_id`` is a real ``uuid`` column now, and a RAW ``text()`` SELECT is
            # NOT the ORM path — psycopg adapts it to ``uuid.UUID`` regardless of the
            # model's ``as_uuid=False``. Compare as text.
            rows = (await conn.execute(text("SELECT tenant_id FROM bdd_jobs"))).scalars().all()
            assert [str(r) for r in rows] == [TENANT_A], f"not isolated under GUC=A: {rows}"

            await conn.execute(text("SAVEPOINT s"))
            with pytest.raises(ProgrammingError):
                await conn.execute(text(_INS), {"job_id": "job-x", "tenant": TENANT_B})
            await conn.execute(text("ROLLBACK TO SAVEPOINT s"))
        finally:
            await conn.execute(text("RESET ROLE"))
            await conn.execute(text(f'DROP OWNED BY "{_ROLE}"'))
            await conn.execute(text(f'DROP ROLE IF EXISTS "{_ROLE}"'))
