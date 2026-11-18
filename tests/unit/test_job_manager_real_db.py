"""Real-database tests for JobManager using postgres+psycopg via the shared db_engine fixture."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.constants import MAX_ERROR_LENGTH
from src.services.job_manager import JobManager, JobStateConflictError


class TestJobManager:
    """JobManager CRUD and lifecycle tests against a real postgres+psycopg database."""

    # ------------------------------------------------------------------
    # Basic CRUD
    # ------------------------------------------------------------------

    async def test_create_and_get_job(self, db_session: AsyncSession):
        """create_job persists a row that get_job can retrieve."""
        job = await JobManager.create_job(
            db_session,
            tenant_id="01950000-0000-7000-8000-000000000001",
            stream="qa_suite",
            input_type="web_url",
            target_url="https://example.com",
        )
        await db_session.commit()

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id
        assert fetched.tenant_id == "01950000-0000-7000-8000-000000000001"
        assert fetched.stream == "qa_suite"
        assert fetched.input_type == "web_url"
        assert fetched.target_url == "https://example.com"
        assert fetched.status == "pending"

    async def test_get_nonexistent_returns_none(self, db_session: AsyncSession):
        """get_job returns None for a job_id that does not exist."""
        result = await JobManager.get_job(db_session, "does-not-exist")
        assert result is None

    # ------------------------------------------------------------------
    # JSONB round-trips
    # ------------------------------------------------------------------

    async def test_config_round_trip(self, db_session: AsyncSession):
        """config dict survives a write/read cycle."""
        cfg = {"max_scenarios": 10, "browser": "chromium", "nested": {"a": [1, 2]}}
        job = await JobManager.create_job(
            db_session,
            tenant_id="01950000-0000-7000-8000-000000000001",
            stream="qa_suite",
            config=cfg,
        )
        await db_session.commit()

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.config == cfg

    async def test_narration_config_round_trip(self, db_session: AsyncSession):
        """narration_config dict survives a write/read cycle."""
        narr = {"tone": "professional", "wpm": 150, "voice_id": "alloy"}
        job = await JobManager.create_job(
            db_session,
            tenant_id="01950000-0000-7000-8000-000000000001",
            stream="demo_video",
            narration_config=narr,
        )
        await db_session.commit()

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.narration_config == narr

    async def test_artifact_paths_round_trip(self, db_session: AsyncSession):
        """artifact_paths written by mark_completed survives a read."""
        job = await JobManager.create_job(
            db_session, tenant_id="01950000-0000-7000-8000-000000000001", stream="qa_suite"
        )
        await db_session.commit()
        await JobManager.mark_processing(db_session, job.job_id)

        paths = {"features": ["login.feature"], "step_defs": ["login.steps.ts"]}
        await JobManager.mark_completed(db_session, job.job_id, artifact_paths=paths)

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.artifact_paths == paths

    async def test_narration_script_round_trip(self, db_session: AsyncSession):
        """narration_script written by mark_completed survives a read."""
        job = await JobManager.create_job(
            db_session, tenant_id="01950000-0000-7000-8000-000000000001", stream="demo_video"
        )
        await db_session.commit()
        await JobManager.mark_processing(db_session, job.job_id)

        script = {
            "scenes": [
                {"text": "Welcome to the app", "duration_ms": 3000},
                {"text": "Click the login button", "duration_ms": 2000},
            ]
        }
        await JobManager.mark_completed(db_session, job.job_id, narration_script=script)

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.narration_script == script

    async def test_downstream_jobs_round_trip(self, db_session: AsyncSession):
        """downstream_jobs written by mark_completed survives a read."""
        job = await JobManager.create_job(
            db_session, tenant_id="01950000-0000-7000-8000-000000000001", stream="demo_video"
        )
        await db_session.commit()
        await JobManager.mark_processing(db_session, job.job_id)

        downstream = {
            "tts_job_id": "tts-abc",
            "video_job_id": "vid-xyz",
        }
        await JobManager.mark_completed(db_session, job.job_id, downstream_jobs=downstream)

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.downstream_jobs == downstream

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    async def test_pending_to_processing_to_completed(self, db_session: AsyncSession):
        """Job transitions pending -> processing -> completed."""
        job = await JobManager.create_job(
            db_session, tenant_id="01950000-0000-7000-8000-000000000001", stream="qa_suite"
        )
        await db_session.commit()
        assert job.status == "pending"

        await JobManager.mark_processing(db_session, job.job_id)
        await db_session.commit()

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "processing"
        assert fetched.updated_at is not None

        await JobManager.mark_completed(db_session, job.job_id)
        await db_session.commit()

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "completed"
        assert fetched.completed_at is not None

    async def test_pending_to_processing_to_failed(self, db_session: AsyncSession):
        """Job transitions pending -> processing -> failed."""
        job = await JobManager.create_job(
            db_session, tenant_id="01950000-0000-7000-8000-000000000001", stream="accessibility"
        )
        await db_session.commit()

        await JobManager.mark_processing(db_session, job.job_id)
        await db_session.commit()

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "processing"

        await JobManager.mark_failed(db_session, job.job_id, "LLM timeout")
        await db_session.commit()

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "failed"
        assert fetched.error_message == "LLM timeout"
        assert fetched.completed_at is not None

    # ------------------------------------------------------------------
    # mark_completed with all output fields
    # ------------------------------------------------------------------

    async def test_mark_completed_all_output_fields(self, db_session: AsyncSession):
        """mark_completed populates every output column."""
        job = await JobManager.create_job(
            db_session, tenant_id="01950000-0000-7000-8000-000000000001", stream="demo_video"
        )
        await db_session.commit()
        await JobManager.mark_processing(db_session, job.job_id)

        artifact = {"features": ["a.feature"], "step_defs": ["a.steps.ts"]}
        narration = {"scenes": [{"text": "hello", "duration_ms": 1500}]}
        downstream = {"tts_job_id": "tts-1", "video_job_id": "vid-1"}

        await JobManager.mark_completed(
            db_session,
            job.job_id,
            features_generated=3,
            step_defs_generated=7,
            artifact_paths=artifact,
            narration_script=narration,
            downstream_jobs=downstream,
            generation_time_seconds=42.5,
        )

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "completed"
        assert fetched.features_generated == 3
        assert fetched.step_defs_generated == 7
        assert fetched.artifact_paths == artifact
        assert fetched.narration_script == narration
        assert fetched.downstream_jobs == downstream
        assert fetched.generation_time_seconds == pytest.approx(42.5)
        assert fetched.completed_at is not None

    # ------------------------------------------------------------------
    # Error truncation
    # ------------------------------------------------------------------

    async def test_mark_failed_truncates_long_error(self, db_session: AsyncSession):
        """mark_failed truncates error_message to MAX_ERROR_LENGTH."""
        job = await JobManager.create_job(
            db_session, tenant_id="01950000-0000-7000-8000-000000000001", stream="qa_suite"
        )
        await db_session.commit()

        long_error = "x" * (MAX_ERROR_LENGTH + 1000)
        await JobManager.mark_failed(db_session, job.job_id, long_error)

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.error_message is not None
        assert len(fetched.error_message) == MAX_ERROR_LENGTH

    async def test_mark_failed_short_error_not_truncated(self, db_session: AsyncSession):
        """mark_failed keeps short error messages intact."""
        job = await JobManager.create_job(
            db_session, tenant_id="01950000-0000-7000-8000-000000000001", stream="qa_suite"
        )
        await db_session.commit()

        short_error = "something went wrong"
        await JobManager.mark_failed(db_session, job.job_id, short_error)

        fetched = await JobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.error_message == short_error

    # ------------------------------------------------------------------
    # Atomic state transition / race conditions
    # ------------------------------------------------------------------

    async def test_mark_processing_twice_raises_conflict(self, db_session: AsyncSession):
        """Second mark_processing on an already-processing job raises conflict."""
        job = await JobManager.create_job(
            db_session, tenant_id="01950000-0000-7000-8000-000000000001", stream="qa_suite"
        )
        await db_session.commit()
        await JobManager.mark_processing(db_session, job.job_id)
        with pytest.raises(JobStateConflictError):
            await JobManager.mark_processing(db_session, job.job_id)

    async def test_mark_completed_on_pending_raises(self, db_session: AsyncSession):
        """mark_completed must not succeed on a pending (non-processing) job."""
        job = await JobManager.create_job(
            db_session, tenant_id="01950000-0000-7000-8000-000000000001", stream="qa_suite"
        )
        await db_session.commit()
        with pytest.raises(JobStateConflictError):
            await JobManager.mark_completed(db_session, job.job_id)

    async def test_mark_failed_on_completed_raises(self, db_session: AsyncSession):
        """mark_failed cannot move a terminal (completed) job to failed."""
        job = await JobManager.create_job(
            db_session, tenant_id="01950000-0000-7000-8000-000000000001", stream="qa_suite"
        )
        await db_session.commit()
        await JobManager.mark_processing(db_session, job.job_id)
        await JobManager.mark_completed(db_session, job.job_id)
        with pytest.raises(JobStateConflictError):
            await JobManager.mark_failed(db_session, job.job_id, "late error")

    async def test_mark_processing_race_exactly_one_winner(self, db_engine):
        """Two concurrent sessions racing on mark_processing: exactly one wins."""
        session_factory = async_sessionmaker(db_engine, expire_on_commit=False)
        async with session_factory() as seed:
            job = await JobManager.create_job(
                seed, tenant_id="01950000-0000-7000-8000-000000000001", stream="qa_suite"
            )
            await seed.commit()
            target_id = job.job_id

        async def _worker() -> str:
            async with session_factory() as session:
                try:
                    await JobManager.mark_processing(session, target_id)
                    return "won"
                except JobStateConflictError:
                    return "lost"

        results = await asyncio.gather(_worker(), _worker(), _worker())
        assert sorted(results) == ["lost", "lost", "won"]

        async with session_factory() as check:
            final = await JobManager.get_job(check, target_id)
            assert final is not None
            assert final.status == "processing"
