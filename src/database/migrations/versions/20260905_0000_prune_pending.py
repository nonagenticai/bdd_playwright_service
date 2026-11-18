"""Teach ``prune_stuck_bdd_jobs`` that ``pending`` also strands.

``bdd_jobs`` is written in four states — ``pending``, ``processing``, ``completed``,
``failed`` — and exactly one of them was reclaimed by NOTHING.
``prune_stuck_bdd_jobs`` matched only ``status = 'processing'``; ``prune_old_bdd_jobs``
DELETEs only ``completed``/``failed``. So a row that the route COMMITS as ``pending``
before ``dispatch_job`` spawns its task (``qa_suite.py`` and its three siblings) is
unreclaimable and undeletable if the task never reaches ``mark_processing``: it
accumulates forever and the orchestrator polls it forever.

⚠️ **The two cutoffs are separate on purpose.** ``pending`` is a LEGITIMATE queued
state, so reusing the 60-minute processing cutoff would fail out live queued work —
a regression, not a fix. The new ``p_pending_cutoff`` argument defaults from
``CLEANUP_PENDING_JOB_TIMEOUT_MINUTES`` (6h) and runs off ``created_at``.

⚠️ **Arity change ⇒ an OVERLOAD, and that is what a rolling update needs.**
``CREATE OR REPLACE FUNCTION`` cannot change a function's argument list, so this
migration creates a SECOND ``prune_stuck_bdd_jobs`` and leaves the one-argument one
live. That is deliberate. The live rollout spec for ``bdd-playwright-service`` is
``replicas: 1`` / ``maxSurge: 25%`` / ``maxUnavailable: 25%``; Kubernetes rounds
maxSurge UP to 1 and maxUnavailable DOWN to 0, so the OLD pod keeps serving until
the NEW pod is Ready — **the two versions overlap**. This migration runs with the
new pod, while the old pod's cleanup loop is still calling
``func.prune_stuck_bdd_jobs(cutoff)`` on its periodic tick. Dropping the
one-argument signature here would make every one of those ticks fail with
``function prune_stuck_bdd_jobs(timestamp with time zone) does not exist`` for the
length of the rollout — a self-inflicted outage of the very reclaim path this
migration exists to fix.

So both arities stay resident, the one-argument one keeping its ORIGINAL
``status = 'processing'`` behaviour (``PRUNE_STUCK_SQL_0003``); the arities differ,
so calls stay unambiguous. ⛔ **Dropping the one-argument form is a SAFE FOLLOW-UP
PR**, once this revision is live on every replica and no pod calls it any more.

⚠️ **Both signatures need an in-band, ``pg_roles``-guarded GRANT.** A new signature
is a new function, born with an EMPTY ACL, so the out-of-band Stage A grant on
``prune_stuck_bdd_jobs(timestamptz)`` does not carry over to the two-argument form —
without the guarded in-band GRANT the reaper hits ``permission denied for function
prune_stuck_bdd_jobs`` in prod while CI, running as the owner, stays green and blind.
The one-argument grant is re-asserted, never revoked: the outgoing pod is using it.

Revision ID: 0005_prune_pending
Revises: 0004_tenant_uuid
"""

from __future__ import annotations

from alembic import op

from src.database.rls_resolvers import (
    DROP_PRUNE_STUCK_V1_SQL,
    PRUNE_STUCK_SQL_0003,
    upgrade_statements,
)

revision: str = "0005_prune_pending"
down_revision: str | None = "0004_tenant_uuid"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # This is the CURRENT head, so rendering from ``rls_resolvers`` is era-correct
    # here (and keeps that module the single source shared with the ``create_all``
    # test path). The next change to this function must freeze a literal, exactly as
    # ``0003_rls`` now does.
    for stmt in upgrade_statements():
        op.execute(stmt)


def downgrade() -> None:
    # Back to the 0003-era shape: ONLY the one-argument form exists. The DROP of the
    # one-argument function here is a re-create, not a removal — it is immediately
    # rebuilt from the frozen 0003 body and re-granted below, so the ACL that the
    # DROP discards is restored in the same transaction.
    op.execute("DROP FUNCTION IF EXISTS prune_stuck_bdd_jobs(timestamptz, timestamptz)")
    op.execute(DROP_PRUNE_STUCK_V1_SQL)
    op.execute(PRUNE_STUCK_SQL_0003)
    op.execute("REVOKE EXECUTE ON FUNCTION prune_stuck_bdd_jobs(timestamptz) FROM PUBLIC")
    op.execute(
        "DO $$\n"
        "BEGIN\n"
        "  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_bdd_playwright_app') THEN\n"
        "    GRANT EXECUTE ON FUNCTION prune_stuck_bdd_jobs(timestamptz)\n"
        "      TO w2v_bdd_playwright_app;\n"
        "  END IF;\n"
        "END $$"
    )
