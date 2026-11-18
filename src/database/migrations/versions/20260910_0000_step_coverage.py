"""Persist how much of a generated suite PASSES WITHOUT ACTING.

``_ensure_step_coverage`` appends an auto-stub — ``waitForLoadState`` +
``stubDwell``, no action — for every uncovered Gherkin step. The stub PASSES, so a
heavily stubbed scenario films a page that never changed for the whole dwell, and
until now the only trace of that was a generator log WARNING: nothing downstream
could distinguish a stubbed scenario from a real one.

``bdd_jobs.step_coverage`` carries the measurement as structured data
(``steps_total``, ``steps_stubbed``, ``stub_ratio``, ``degraded``,
``degraded_threshold``, ``stubbed_steps``), which ``GET /api/bdd/{id}/result``
returns verbatim.

⚠️ **Additive and defaulted on purpose.** The live rollout is ``replicas: 1`` with
an overlapping surge, so the OLD pod keeps writing rows while this column exists.
A nullable JSONB with ``server_default '{}'`` means those rows are well-formed for
the new reader (an empty dict = "not measured"), and the old pod's INSERT/UPDATE
statements — which never name the column — keep working unchanged.

Revision ID: 0006_step_coverage
Revises: 0005_prune_pending
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0006_step_coverage"
down_revision: str | None = "0005_prune_pending"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "bdd_jobs",
        sa.Column("step_coverage", JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("bdd_jobs", "step_coverage")
