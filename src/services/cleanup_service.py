"""Cleanup service for stale BDD jobs and expired artifacts.

These sweeps are CROSS-TENANT (the background loop runs at app boot with no tenant
context). Under Phase D RLS enforce the runtime is a non-bypass role with no
``app.current_tenant`` bound, so an ORM read/write would fail-closed to 0 rows.
The maintenance therefore runs through SECURITY DEFINER resolvers
(``prune_stuck_bdd_jobs`` / ``prune_old_bdd_jobs``, migration 0003) owned by the
table owner, which bypass ENABLE and operate across all tenants. The resolvers run
as the same owner under mode=off too, so this path is correct before and after
cutover.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.constants import (
    PENDING_JOB_TIMEOUT_MINUTES,
    RETENTION_DAYS,
    STUCK_JOB_TIMEOUT_MINUTES,
)

logger = logging.getLogger(__name__)


class CleanupService:
    """Periodic cleanup of stuck-processing and retention-expired jobs."""

    async def cleanup_stuck_jobs(
        self,
        db: AsyncSession,
        *,
        stuck_timeout_minutes: int = STUCK_JOB_TIMEOUT_MINUTES,
        pending_timeout_minutes: int = PENDING_JOB_TIMEOUT_MINUTES,
    ) -> int:
        """Mark stranded jobs failed (cross-tenant). Returns the count.

        Reaps BOTH non-terminal states, on TWO INDEPENDENT clocks:

        * ``processing`` older than ``stuck_timeout_minutes`` by ``updated_at`` — a
          job that started and stopped reporting.
        * ``pending`` older than ``pending_timeout_minutes`` by ``created_at`` — a
          job whose row committed in the route but whose background task never
          reached ``mark_processing`` (SIGKILL, eviction, or a shutdown whose rescue
          hit an already-disposed engine). Nothing reclaimed these; nothing deleted
          them either, so they accumulated forever and their callers polled forever.

        ⛔ Do NOT collapse the two windows onto one value. ``pending`` is a LEGITIMATE
        queued state, so the 60-minute processing cutoff applied to it would fail out
        live queued work.
        """
        now = datetime.now(UTC)
        cutoff = now - timedelta(minutes=stuck_timeout_minutes)
        pending_cutoff = now - timedelta(minutes=pending_timeout_minutes)
        count = (
            await db.execute(select(func.prune_stuck_bdd_jobs(cutoff, pending_cutoff)))
        ).scalar_one()
        await db.commit()
        if count:
            logger.info("Marked %d stranded BDD jobs failed", count)
        return int(count or 0)

    async def cleanup_old_jobs(
        self,
        db: AsyncSession,
        *,
        retention_days: int = RETENTION_DAYS,
    ) -> int:
        """Delete terminal BDD jobs older than retention (cross-tenant)."""
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        count = (await db.execute(select(func.prune_old_bdd_jobs(cutoff)))).scalar_one()
        await db.commit()
        if count:
            logger.info("Deleted %d old BDD jobs", count)
        return int(count or 0)
