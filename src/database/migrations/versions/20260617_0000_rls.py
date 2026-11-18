"""Phase D RLS — enable row-level security on the tenant-scoped tables.

Applies the SAME policies the ``create_all`` ``register_rls_ddl`` event applies in
tests, rendered from the single source ``w2v_common.database.rls`` (bdd_jobs +
bdd_usage + narration_templates; all varchar tenant_id → text predicate).
ENABLE-not-FORCE → inert until the runtime connects as the restricted non-owner role
``w2v_bdd_playwright_app`` with ``TENANT_RLS_MODE=enforce``.

Also creates two SECURITY DEFINER resolvers owned by the migration user (table owner
→ bypasses ENABLE) so the cross-tenant cleanup loop (``cleanup_service.py``, started
at app boot with no tenant context) still maintains jobs across ALL tenants under
enforce. EXECUTE is REVOKEd from PUBLIC here and GRANTed to the app role out-of-band
in Stage A (the app role does not exist in CI).

Revision ID: 0003_rls
Revises: 0002
"""

from alembic import op

from src.database.rls_resolvers import PRUNE_OLD_SQL, PRUNE_STUCK_SQL_0003

revision: str = "0003_rls"
down_revision: str | None = "0002"
branch_labels = None
depends_on = None


# ⛔ FROZEN 2026-09-02 — these statements used to be rendered at run time from the
# LIVE ``Base.metadata`` via ``w2v_common.database.rls.{up,down}grade_statements``.
# That makes a historical migration describe TODAY's ORM instead of the schema that
# existed when it ran: ``_predicate()`` renders ``tenant_id = <guc>`` for a
# ``String`` column and ``tenant_id = <guc>::uuid`` for anything else, so the moment
# the ORM column became ``uuid`` this migration started emitting a ``::uuid``
# predicate against a column that is still ``VARCHAR(128)`` at this point in the
# chain — ``operator does not exist: character varying = uuid`` — and
# ``alembic upgrade head`` from ZERO broke. The statements below are byte-for-byte
# what the helpers rendered at the time (varchar tenant_id, all NOT NULL → strict
# equality), pinned as literals. A migration is a historical record; it must not
# move when the models move. The type change itself lives in ``0004_tenant_uuid``.
_GUC = "NULLIF(current_setting('app.current_tenant', true), '')"
_PREDICATE = f"tenant_id = {_GUC}"  # varchar column → text-to-text comparison
_TENANT_TABLES: tuple[str, ...] = (
    "bdd_jobs",
    "bdd_usage",
    "narration_templates",
)
_ALL_POLICY_NAMES = (
    "tenant_isolation",
    "tenant_isolation_read",
    "tenant_isolation_ins",
    "tenant_isolation_upd",
    "tenant_isolation_del",
)


def _rls_upgrade_statements() -> list[str]:
    out: list[str] = []
    for table in _TENANT_TABLES:
        out += [
            f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
            f"DROP POLICY IF EXISTS tenant_isolation ON {table}",
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING ({_PREDICATE}) WITH CHECK ({_PREDICATE})",
        ]
    return out


def _rls_downgrade_statements() -> list[str]:
    out: list[str] = []
    for table in _TENANT_TABLES:
        out += [f"DROP POLICY IF EXISTS {policy} ON {table}" for policy in _ALL_POLICY_NAMES]
        out.append(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    return out


def upgrade() -> None:
    for stmt in _rls_upgrade_statements():
        op.execute(stmt)
    # ⛔ The ERA-CORRECT one-argument body. ``PRUNE_STUCK_SQL`` now renders the
    # two-argument function created by ``0005_prune_pending``; importing it here
    # would make the REVOKE below name a signature that does not exist yet and
    # break ``alembic upgrade head`` from zero — the same defect this file's
    # docstring already froze the RLS predicate against.
    op.execute(PRUNE_STUCK_SQL_0003)
    op.execute("REVOKE EXECUTE ON FUNCTION prune_stuck_bdd_jobs(timestamptz) FROM PUBLIC")
    op.execute(PRUNE_OLD_SQL)
    op.execute("REVOKE EXECUTE ON FUNCTION prune_old_bdd_jobs(timestamptz) FROM PUBLIC")


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS prune_old_bdd_jobs(timestamptz)")
    op.execute("DROP FUNCTION IF EXISTS prune_stuck_bdd_jobs(timestamptz)")
    for stmt in _rls_downgrade_statements():
        op.execute(stmt)
