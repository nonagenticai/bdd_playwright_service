"""Align chain with prod — drop unique constraint that prod never had.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-26 22:30:00.000000

Round-9 chain reconciliation. After Round 7 stamped prod at 0001, autogenerate
revealed that 0001's `sa.UniqueConstraint("job_id")` produces a constraint
`bdd_jobs_job_id_key` that the prod schema does not have (prod was originally
built by `Base.metadata.create_all` which emitted ONLY the unique index
`ix_bdd_jobs_job_id`, not a separate UniqueConstraint, despite the model also
having `unique=True`).

This migration drops the redundant UniqueConstraint idempotently so that:

- prod (no constraint): no-op, alembic_version advances to 0002.
- fresh CI / test-namespace DB (0001 just created the constraint): drops it,
  resulting schema matches prod.

The unique-on-job_id invariant remains enforced by `ix_bdd_jobs_job_id`
which is a unique index — semantically equivalent for PG's purposes.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE bdd_jobs DROP CONSTRAINT IF EXISTS bdd_jobs_job_id_key")


def downgrade() -> None:
    # Re-create the constraint. Will fail if there are duplicate job_id values,
    # but that's also true of the upgrade-path invariant — they are mutually
    # exclusive with the unique INDEX still in place. Wrapped in a savepoint
    # in alembic; an existing-constraint error here means downgrade-after-fresh-
    # upgrade is being attempted, which is an explicit non-goal of this revision.
    op.execute("ALTER TABLE bdd_jobs ADD CONSTRAINT bdd_jobs_job_id_key UNIQUE (job_id)")
