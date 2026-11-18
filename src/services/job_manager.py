"""Job lifecycle management for BDD generation jobs."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.constants import MAX_ERROR_LENGTH
from src.database.models import BDDJob, BDDUsage, StepCoverage

logger = logging.getLogger(__name__)


class JobStateConflictError(RuntimeError):
    """Raised when an atomic job-state transition loses a race.

    A caller should treat this as "another worker already transitioned
    this job" — log and bail cleanly, do not retry.
    """


class JobManager:
    """Static methods for BDD job CRUD and usage tracking."""

    @staticmethod
    async def create_job(
        db: AsyncSession,
        *,
        tenant_id: str,
        stream: str,
        input_type: str | None = None,
        target_url: str | None = None,
        app_path: str | None = None,
        config: dict[str, Any] | None = None,
        narration_config: dict[str, Any] | None = None,
    ) -> BDDJob:
        """Create a new BDD generation job."""
        job = BDDJob(
            job_id=str(uuid.uuid7()),
            tenant_id=tenant_id,
            stream=stream,
            input_type=input_type,
            target_url=target_url,
            app_path=app_path,
            config=config or {},
            narration_config=narration_config or {},
            status="pending",
        )
        db.add(job)
        await db.flush()
        logger.debug("Created BDD job %s for tenant=%s stream=%s", job.job_id, tenant_id, stream)
        return job

    @staticmethod
    async def get_job(db: AsyncSession, job_id: str, tenant_id: str | None = None) -> BDDJob | None:
        """Fetch a BDD job by its job_id.

        When ``tenant_id`` is given, the lookup is additionally scoped to that
        tenant -- a job owned by a different tenant behaves as not-found.
        ``tenant_id=None`` (the default) is unscoped, for background workers
        that legitimately process any tenant's job.
        """
        stmt = select(BDDJob).where(BDDJob.job_id == job_id)
        if tenant_id is not None:
            stmt = stmt.where(BDDJob.tenant_id == tenant_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def mark_processing(db: AsyncSession, job_id: str) -> None:
        """Atomically transition a pending job to the processing state.

        Raises :class:`JobStateConflictError` if the job is not currently pending
        (another worker already claimed it, or it is already terminal).
        """
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(BDDJob)
                .where(BDDJob.job_id == job_id, BDDJob.status == "pending")
                .values(status="processing", updated_at=datetime.now(UTC))
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"BDD job {job_id!r} is not in 'pending' state — cannot transition to 'processing'"
            )

    @staticmethod
    async def mark_completed(
        db: AsyncSession,
        job_id: str,
        *,
        features_generated: int = 0,
        step_defs_generated: int = 0,
        artifact_paths: dict[str, Any] | None = None,
        narration_script: dict[str, Any] | None = None,
        downstream_jobs: dict[str, Any] | None = None,
        generation_time_seconds: float | None = None,
        step_coverage: StepCoverage | None = None,
    ) -> None:
        """Atomically transition a processing job to the completed state.

        Raises :class:`JobStateConflictError` if the job is not currently
        in the ``processing`` state.
        """
        now = datetime.now(UTC)
        values: dict[str, Any] = {
            "status": "completed",
            "features_generated": features_generated,
            "step_defs_generated": step_defs_generated,
            "artifact_paths": artifact_paths or {},
            "generation_time_seconds": generation_time_seconds,
            "updated_at": now,
            "completed_at": now,
        }
        if narration_script is not None:
            values["narration_script"] = narration_script
        if downstream_jobs is not None:
            values["downstream_jobs"] = downstream_jobs
        if step_coverage is not None:
            values["step_coverage"] = step_coverage

        result = cast(
            CursorResult[Any],
            await db.execute(
                update(BDDJob)
                .where(BDDJob.job_id == job_id, BDDJob.status == "processing")
                .values(**values)
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"BDD job {job_id!r} is not in 'processing' state — cannot transition to 'completed'"
            )
        logger.debug(
            "BDD job %s completed: %d features, %d step defs in %.1fs",
            job_id,
            features_generated,
            step_defs_generated,
            generation_time_seconds or 0,
        )

    @staticmethod
    async def mark_failed(db: AsyncSession, job_id: str, error: str) -> None:
        """Atomically transition a non-terminal job to the failed state.

        Succeeds only when the job is currently ``pending`` or ``processing``.
        Raises :class:`JobStateConflictError` if the job is already terminal.
        """
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(BDDJob)
                .where(
                    BDDJob.job_id == job_id,
                    BDDJob.status.in_(["pending", "processing"]),
                )
                .values(
                    status="failed",
                    error_message=error[:MAX_ERROR_LENGTH],
                    updated_at=now,
                    completed_at=now,
                )
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"BDD job {job_id!r} is not pending/processing — cannot transition to 'failed'"
            )
        logger.debug("BDD job %s marked as failed", job_id)

    @staticmethod
    async def record_usage(
        db: AsyncSession,
        *,
        tenant_id: str,
        jobs_completed: int = 0,
        is_error: bool = False,
        features_generated: int = 0,
    ) -> None:
        """Upsert daily usage counters for a tenant."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        stmt = (
            pg_insert(BDDUsage)
            .values(
                tenant_id=tenant_id,
                date=today,
                jobs_started=1,
                jobs_completed=jobs_completed,
                jobs_failed=1 if is_error else 0,
                features_generated=features_generated,
            )
            .on_conflict_do_update(
                constraint="uq_bdd_usage_tenant_date",
                # Use ORM column arithmetic instead of `text(f"... + {val}")`
                # so SQLAlchemy emits the operands as bound parameters rather
                # than f-string-interpolated SQL — matches the
                # parameterisation pattern of `.values(...)` above and removes
                # the implicit trust that callers pass typed ints. The
                # `<Model>.<col>` references resolve to the EXISTING row's
                # value inside an `on_conflict_do_update` SET clause; use
                # `.excluded.<col>` to reference the candidate row.
                set_={
                    "jobs_started": BDDUsage.jobs_started + 1,
                    "jobs_completed": BDDUsage.jobs_completed + jobs_completed,
                    "jobs_failed": BDDUsage.jobs_failed + (1 if is_error else 0),
                    "features_generated": (BDDUsage.features_generated + features_generated),
                },
            )
        )
        await db.execute(stmt)
