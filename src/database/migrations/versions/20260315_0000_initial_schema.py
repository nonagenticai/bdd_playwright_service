"""Initial schema for bdd_playwright_service.

Revision ID: 0001
Revises:
Create Date: 2026-03-15 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "bdd_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("stream", sa.String(32), nullable=False),
        sa.Column("input_type", sa.String(32), nullable=True),
        sa.Column("target_url", sa.Text(), nullable=True),
        sa.Column("app_path", sa.Text(), nullable=True),
        sa.Column(
            "inspection_data",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=True,
        ),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=True,
        ),
        sa.Column(
            "narration_config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=True,
        ),
        sa.Column("features_generated", sa.Integer(), server_default="0", nullable=True),
        sa.Column("step_defs_generated", sa.Integer(), server_default="0", nullable=True),
        sa.Column(
            "artifact_paths",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=True,
        ),
        sa.Column(
            "narration_script",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=True,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "downstream_jobs",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=True,
        ),
        sa.Column("generation_time_seconds", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index(op.f("ix_bdd_jobs_job_id"), "bdd_jobs", ["job_id"], unique=True)
    op.create_index(op.f("ix_bdd_jobs_tenant_id"), "bdd_jobs", ["tenant_id"], unique=False)

    op.create_table(
        "bdd_usage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("date", sa.String(10), nullable=False),
        sa.Column("jobs_started", sa.Integer(), server_default="0", nullable=True),
        sa.Column("jobs_completed", sa.Integer(), server_default="0", nullable=True),
        sa.Column("jobs_failed", sa.Integer(), server_default="0", nullable=True),
        sa.Column("features_generated", sa.Integer(), server_default="0", nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "date", name="uq_bdd_usage_tenant_date"),
    )

    op.create_table(
        "narration_templates",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("tone", sa.String(64), nullable=False),
        sa.Column(
            "given_patterns",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=True,
        ),
        sa.Column(
            "when_patterns",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=True,
        ),
        sa.Column(
            "then_patterns",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=True,
        ),
        sa.Column("is_default", sa.Boolean(), server_default="false", nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "name", name="uq_narration_template_tenant_name"),
    )


def downgrade() -> None:
    op.drop_table("narration_templates")
    op.drop_table("bdd_usage")
    op.drop_index(op.f("ix_bdd_jobs_tenant_id"), table_name="bdd_jobs")
    op.drop_index(op.f("ix_bdd_jobs_job_id"), table_name="bdd_jobs")
    op.drop_table("bdd_jobs")
