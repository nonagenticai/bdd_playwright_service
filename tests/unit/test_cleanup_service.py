"""Integration tests for CleanupService.

Under Phase D RLS the cross-tenant cleanup runs through SECURITY DEFINER resolvers
(``prune_stuck_bdd_jobs`` / ``prune_old_bdd_jobs``). These tests exercise the real
resolver SQL against a live Postgres (the ``db_session`` fixture's ``create_all``
schema includes the resolvers via ``register_resolvers``), seeding real rows and
asserting both the returned count and the resulting DB state.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.constants import (
    PENDING_JOB_TIMEOUT_MINUTES,
    RETENTION_DAYS,
    STUCK_JOB_TIMEOUT_MINUTES,
)
from src.database import rls_resolvers
from src.database.models import BDDJob
from src.services.cleanup_service import CleanupService


async def _seed(
    db: AsyncSession,
    *,
    job_id: str,
    tenant_id: str = "01950000-0000-7000-8000-0000000ac0e5",
    status: str = "processing",
    stream: str = "qa_suite",
    updated_minutes_ago: float | None = None,
    created_days_ago: float | None = None,
) -> None:
    now = datetime.now(UTC)
    db.add(
        BDDJob(
            job_id=job_id,
            tenant_id=tenant_id,
            status=status,
            stream=stream,
            updated_at=(now - timedelta(minutes=updated_minutes_ago))
            if updated_minutes_ago is not None
            else now,
            created_at=(now - timedelta(days=created_days_ago))
            if created_days_ago is not None
            else now,
        )
    )
    await db.commit()


async def _status_of(db: AsyncSession, job_id: str) -> str | None:
    row = (await db.execute(select(BDDJob).where(BDDJob.job_id == job_id))).scalar_one_or_none()
    return row.status if row else None


class TestCleanupStuckJobs:
    async def test_no_stuck_jobs_returns_zero(self, db_session: AsyncSession) -> None:
        assert await CleanupService().cleanup_stuck_jobs(db_session) == 0

    async def test_marks_stuck_jobs_failed_cross_tenant(self, db_session: AsyncSession) -> None:
        await _seed(
            db_session,
            job_id="stuck-a",
            tenant_id="01950000-0000-7000-8000-00000000000a",
            updated_minutes_ago=120,
        )
        await _seed(
            db_session,
            job_id="stuck-b",
            tenant_id="01950000-0000-7000-8000-00000000000b",
            updated_minutes_ago=120,
        )
        await _seed(
            db_session,
            job_id="fresh",
            tenant_id="01950000-0000-7000-8000-00000000000a",
            updated_minutes_ago=1,
        )

        count = await CleanupService().cleanup_stuck_jobs(
            db_session, stuck_timeout_minutes=STUCK_JOB_TIMEOUT_MINUTES
        )

        assert count == 2
        assert await _status_of(db_session, "stuck-a") == "failed"
        assert await _status_of(db_session, "stuck-b") == "failed"
        assert await _status_of(db_session, "fresh") == "processing"

    async def test_custom_timeout_window(self, db_session: AsyncSession) -> None:
        await _seed(db_session, job_id="mid", updated_minutes_ago=90)
        assert await CleanupService().cleanup_stuck_jobs(db_session, stuck_timeout_minutes=120) == 0
        assert await _status_of(db_session, "mid") == "processing"
        assert await CleanupService().cleanup_stuck_jobs(db_session, stuck_timeout_minutes=60) == 1
        assert await _status_of(db_session, "mid") == "failed"


class TestCleanupOldJobs:
    async def test_no_old_jobs_returns_zero(self, db_session: AsyncSession) -> None:
        assert await CleanupService().cleanup_old_jobs(db_session) == 0

    async def test_deletes_old_terminal_jobs_cross_tenant(self, db_session: AsyncSession) -> None:
        await _seed(
            db_session,
            job_id="old-done",
            tenant_id="01950000-0000-7000-8000-00000000000a",
            status="completed",
            created_days_ago=60,
        )
        await _seed(
            db_session,
            job_id="old-fail",
            tenant_id="01950000-0000-7000-8000-00000000000b",
            status="failed",
            created_days_ago=45,
        )
        await _seed(db_session, job_id="old-running", status="processing", created_days_ago=60)
        await _seed(db_session, job_id="recent-done", status="completed", created_days_ago=0)

        count = await CleanupService().cleanup_old_jobs(db_session, retention_days=RETENTION_DAYS)

        assert count == 2
        assert await _status_of(db_session, "old-done") is None
        assert await _status_of(db_session, "old-fail") is None
        assert await _status_of(db_session, "old-running") == "processing"
        assert await _status_of(db_session, "recent-done") == "completed"

    async def test_custom_retention_days(self, db_session: AsyncSession) -> None:
        await _seed(db_session, job_id="ten-day", status="completed", created_days_ago=10)
        assert await CleanupService().cleanup_old_jobs(db_session, retention_days=30) == 0
        assert await _status_of(db_session, "ten-day") == "completed"
        assert await CleanupService().cleanup_old_jobs(db_session, retention_days=7) == 1
        assert await _status_of(db_session, "ten-day") is None


class TestCleanupStrandedPendingJobs:
    """``pending`` is the state NOTHING reclaimed.

    The route COMMITS the row as ``pending`` and only THEN spawns the background task
    (``qa_suite.py`` and its three siblings); the first state write is
    ``mark_processing``. If the task dies in that window — SIGKILL, kubelet eviction,
    or a SIGTERM whose ``mark_failed`` rescue met an already-disposed engine — the row
    stays ``pending``. ``prune_stuck_bdd_jobs`` matched only ``processing`` and
    ``prune_old_bdd_jobs`` DELETEs only terminal rows, so the row was neither reclaimed
    nor deleted: it accumulated forever and its caller polled forever.
    """

    async def test_reclaims_pending_older_than_the_pending_cutoff(
        self, db_session: AsyncSession
    ) -> None:
        await _seed(
            db_session,
            job_id="stranded-pending",
            status="pending",
            created_days_ago=1,  # 24h — well past the 6h pending cutoff
            updated_minutes_ago=60 * 24,
        )

        count = await CleanupService().cleanup_stuck_jobs(
            db_session,
            stuck_timeout_minutes=STUCK_JOB_TIMEOUT_MINUTES,
            pending_timeout_minutes=PENDING_JOB_TIMEOUT_MINUTES,
        )

        assert count == 1, (
            "a 24-hour-old `pending` row was not reclaimed — `prune_stuck_bdd_jobs` "
            "still matches only `status = 'processing'`, so the row is unreclaimable "
            "AND undeletable and its caller polls forever"
        )
        assert await _status_of(db_session, "stranded-pending") == "failed"

    async def test_pending_reclaim_is_cross_tenant(self, db_session: AsyncSession) -> None:
        await _seed(
            db_session,
            job_id="pend-a",
            tenant_id="01950000-0000-7000-8000-00000000000a",
            status="pending",
            created_days_ago=1,
        )
        await _seed(
            db_session,
            job_id="pend-b",
            tenant_id="01950000-0000-7000-8000-00000000000b",
            status="pending",
            created_days_ago=1,
        )

        assert (
            await CleanupService().cleanup_stuck_jobs(
                db_session, pending_timeout_minutes=PENDING_JOB_TIMEOUT_MINUTES
            )
            == 2
        )
        assert await _status_of(db_session, "pend-a") == "failed"
        assert await _status_of(db_session, "pend-b") == "failed"

    async def test_pending_carries_its_own_error_message(self, db_session: AsyncSession) -> None:
        """A never-dispatched job is not a timed-out one; the row must say which."""
        await _seed(db_session, job_id="never-started", status="pending", created_days_ago=1)
        await _seed(db_session, job_id="timed-out", status="processing", updated_minutes_ago=120)

        await CleanupService().cleanup_stuck_jobs(db_session)

        rows = {
            row.job_id: row.error_message
            for row in (await db_session.execute(select(BDDJob))).scalars()
        }
        assert rows["never-started"] == "Job never started (pending-job cleanup)"
        assert rows["timed-out"] == "Job timed out (stuck-job cleanup)"

    async def test_young_pending_is_spared_by_the_separate_cutoff(
        self, db_session: AsyncSession
    ) -> None:
        """⛔ REGRESSION GUARD: the two cutoffs must not collapse onto one another.

        ``pending`` is a LEGITIMATE queued state. This row is OLD by the 60-minute
        processing cutoff (created 2h ago) but YOUNG by the 6h pending cutoff, i.e. a
        job that is simply still queued. If the pending clause is wired to ``p_cutoff``
        instead of ``p_pending_cutoff`` — the obvious one-parameter "fix" — this live
        queued job is failed out from under its caller. That is a REGRESSION shipped as
        a bug fix, and it is strictly worse than the strand it replaces.
        """
        await _seed(
            db_session,
            job_id="still-queued",
            status="pending",
            created_days_ago=2 / 24,  # 2h old: > 60min, < 6h
            updated_minutes_ago=120,
        )
        # A genuinely stuck `processing` row of the SAME age, so the test also proves
        # the processing clock is still live at 60 minutes and the sweep is not simply
        # matching nothing.
        await _seed(db_session, job_id="really-stuck", status="processing", updated_minutes_ago=120)

        count = await CleanupService().cleanup_stuck_jobs(
            db_session,
            stuck_timeout_minutes=STUCK_JOB_TIMEOUT_MINUTES,
            pending_timeout_minutes=PENDING_JOB_TIMEOUT_MINUTES,
        )

        assert await _status_of(db_session, "still-queued") == "pending", (
            "a 2-hour-old `pending` job was failed out while it was still legitimately "
            "queued — the pending cutoff has been collapsed onto the 60-minute "
            "processing cutoff instead of using its own (6h) window"
        )
        assert count == 1, f"only the stuck `processing` row should have been reaped (got {count})"
        assert await _status_of(db_session, "really-stuck") == "failed"

    async def test_completed_and_failed_are_never_touched(self, db_session: AsyncSession) -> None:
        """Widening to `pending` must not widen to the terminal states."""
        await _seed(db_session, job_id="done", status="completed", created_days_ago=9)
        await _seed(db_session, job_id="dead", status="failed", created_days_ago=9)

        assert await CleanupService().cleanup_stuck_jobs(db_session) == 0
        assert await _status_of(db_session, "done") == "completed"
        assert await _status_of(db_session, "dead") == "failed"


class TestResolverGrants:
    """A NEW function signature is born with an EMPTY ACL.

    The Stage A grant was applied out-of-band to ``prune_stuck_bdd_jobs(timestamptz)``.
    Changing the arity mints a different function, so that grant does not carry over —
    the reaper would hit ``permission denied for function prune_stuck_bdd_jobs`` in
    prod while CI, which runs as the owner, stayed green and blind. The DDL therefore
    ships its own ``pg_roles``-guarded GRANT, and these assertions keep the granted
    signatures in step with the created ones.
    """

    def test_grant_names_the_app_role(self) -> None:
        assert rls_resolvers.APP_ROLE in rls_resolvers.GRANT_RESOLVERS_SQL
        assert "pg_roles" in rls_resolvers.GRANT_RESOLVERS_SQL, (
            "the GRANT is unguarded — it will fail the whole migration in CI, where "
            f"the role {rls_resolvers.APP_ROLE!r} does not exist"
        )

    def test_every_created_signature_is_revoked_and_granted(self) -> None:
        statements = rls_resolvers.upgrade_statements()
        grant_sql = " ".join(statements)
        for signature in (
            # The one-argument overload is created too (rollout overlap), so it needs
            # the same PUBLIC revoke + app-role grant as the signatures beside it.
            "prune_stuck_bdd_jobs(timestamptz)",
            "prune_stuck_bdd_jobs(timestamptz, timestamptz)",
            "prune_old_bdd_jobs(timestamptz)",
        ):
            assert f"REVOKE EXECUTE ON FUNCTION {signature} FROM PUBLIC" in statements, (
                f"{signature} is world-executable — EXECUTE was never revoked from PUBLIC"
            )
            assert f"GRANT EXECUTE ON FUNCTION {signature}" in grant_sql, (
                f"{signature} was created but never granted to {rls_resolvers.APP_ROLE} — "
                "the reaper will get `permission denied for function` in prod"
            )


class TestResolverOverloadsSurviveARollingUpdate:
    """⛔ ROLLOUT SAFETY: the ONE-argument overload must stay resident.

    ``bdd-playwright-service`` rolls with ``replicas: 1``, ``maxSurge: 25%`` and
    ``maxUnavailable: 25%``. Kubernetes rounds maxSurge UP to 1 and maxUnavailable
    DOWN to 0, so the OLD pod keeps serving until the NEW pod is Ready — the two
    versions OVERLAP. The migration runs with the new pod, while the old pod's
    cleanup loop is still calling ``func.prune_stuck_bdd_jobs(cutoff)`` on its
    periodic tick. If the migration dropped ``prune_stuck_bdd_jobs(timestamptz)``,
    every one of those ticks would fail ``function prune_stuck_bdd_jobs(timestamp
    with time zone) does not exist`` for the length of the rollout — a self-inflicted
    outage of the very reclaim path this change exists to fix.

    Dropping the one-argument form is a SAFE FOLLOW-UP PR, once no pod calls it.
    """

    def test_the_upgrade_never_drops_the_one_argument_signature(self) -> None:
        statements = rls_resolvers.upgrade_statements()
        assert rls_resolvers.DROP_PRUNE_STUCK_V1_SQL not in statements, (
            "the upgrade path drops prune_stuck_bdd_jobs(timestamptz) — the OLD pod, "
            "which keeps serving until the new pod is Ready (maxUnavailable 25% of 1 "
            "rounds DOWN to 0), calls exactly that signature every cleanup tick and "
            "will fail for the whole rollout"
        )
        assert not any("DROP FUNCTION" in stmt for stmt in statements), (
            f"unexpected DROP FUNCTION in the upgrade path: {statements}"
        )

    def test_the_upgrade_creates_both_arities(self) -> None:
        statements = rls_resolvers.upgrade_statements()
        assert rls_resolvers.PRUNE_STUCK_SQL_0003 in statements, (
            "the one-argument body is never created, so a database built from zero "
            "(or a fresh test schema) has only the two-argument form"
        )
        assert statements.index(rls_resolvers.PRUNE_STUCK_SQL_0003) < statements.index(
            rls_resolvers.PRUNE_STUCK_SQL
        )

    def test_the_one_argument_body_keeps_its_original_processing_only_behaviour(
        self,
    ) -> None:
        """The OLD pod must see the OLD semantics — widening is the new form's job."""
        body = rls_resolvers.PRUNE_STUCK_SQL_0003
        assert "status = 'processing'" in body
        assert "pending" not in body, (
            "the one-argument overload was taught about `pending`; the outgoing pod "
            "passes only a 60-minute cutoff, so this would fail out live queued work"
        )

    async def test_both_signatures_exist_in_pg_proc_and_are_callable(
        self, db_session: AsyncSession
    ) -> None:
        """(e) The rollout-safety guard, asserted against a REAL Postgres.

        The schema under test is built by the same ``upgrade_statements()`` that
        ``alembic upgrade head`` executes, so this reads the post-migration catalog.
        """
        arities = sorted(
            row[0]
            for row in (
                await db_session.execute(
                    text(
                        "SELECT pronargs FROM pg_proc p "
                        "JOIN pg_namespace n ON n.oid = p.pronamespace "
                        "WHERE p.proname = 'prune_stuck_bdd_jobs' "
                        "AND n.nspname = current_schema()"
                    )
                )
            ).all()
        )
        assert arities == [1, 2], (
            "after the migration `pg_proc` must hold BOTH prune_stuck_bdd_jobs "
            f"overloads — arity 1 (the OLD pod's signature) and arity 2 — got {arities}. "
            "A missing arity 1 means the migration dropped the signature the outgoing "
            "pod calls every cleanup tick, breaking it for the length of the rollout"
        )

        # ...and BOTH are callable. Overload resolution must stay unambiguous: the
        # arities differ, so PostgreSQL can never raise
        # `function prune_stuck_bdd_jobs(...) is not unique` — proven, not assumed.
        now = datetime.now(UTC)
        one_arg = (
            await db_session.execute(
                text("SELECT prune_stuck_bdd_jobs(:c)"), {"c": now - timedelta(minutes=60)}
            )
        ).scalar_one()
        two_arg = (
            await db_session.execute(
                text("SELECT prune_stuck_bdd_jobs(:c, :p)"),
                {"c": now - timedelta(minutes=60), "p": now - timedelta(hours=6)},
            )
        ).scalar_one()
        await db_session.commit()
        assert one_arg == 0
        assert two_arg == 0

    async def test_the_one_argument_overload_still_reaps_processing_only(
        self, db_session: AsyncSession
    ) -> None:
        """The old pod's call must keep working AND keep its old blast radius."""
        await _seed(
            db_session, job_id="old-pod-stuck", status="processing", updated_minutes_ago=120
        )
        await _seed(db_session, job_id="old-pod-pending", status="pending", created_days_ago=1)

        reaped = (
            await db_session.execute(
                text("SELECT prune_stuck_bdd_jobs(:c)"),
                {"c": datetime.now(UTC) - timedelta(minutes=STUCK_JOB_TIMEOUT_MINUTES)},
            )
        ).scalar_one()
        await db_session.commit()

        assert reaped == 1, (
            "the one-argument overload the OLD pod calls did not behave as it always "
            f"did (expected exactly the stuck `processing` row, got {reaped})"
        )
        assert await _status_of(db_session, "old-pod-stuck") == "failed"
        assert await _status_of(db_session, "old-pod-pending") == "pending"
