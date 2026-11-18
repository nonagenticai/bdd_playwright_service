"""Tests for job lifecycle management — mock-only edge cases.

Real-db state-transition coverage lives in ``test_job_manager_real_db.py``.
This file targets mock-side edge cases that don't require a live database
(e.g., asserting record_usage's postgres-only ON CONFLICT path against a
mocked session).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.database.models import BDDJob
from src.services.job_manager import JobManager


class TestJobManagerMockOnly:
    """Mock-based tests for paths that don't require a live database."""

    async def test_create_job_constructs_model_and_flushes(self, mock_db_session):
        """create_job builds a BDDJob and flushes it through the session."""
        job = await JobManager.create_job(
            mock_db_session,
            tenant_id="01950000-0000-7000-8000-000000000001",
            stream="qa_suite",
            input_type="web_url",
            target_url="https://example.com",
        )
        assert isinstance(job, BDDJob)
        assert job.tenant_id == "01950000-0000-7000-8000-000000000001"
        assert job.stream == "qa_suite"
        assert job.status == "pending"
        assert job.target_url == "https://example.com"
        mock_db_session.add.assert_called_once_with(job)
        mock_db_session.flush.assert_awaited_once()

    async def test_get_job_returns_row(self, mock_db_session):
        """get_job returns the scalar row from the session result."""
        mock_job = MagicMock(spec=BDDJob)
        mock_job.job_id = "test-job-id"
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        result = await JobManager.get_job(mock_db_session, "test-job-id")
        assert result is mock_job

    async def test_get_job_returns_none_when_missing(self, mock_db_session):
        """get_job returns None when the row is absent."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        result = await JobManager.get_job(mock_db_session, "nonexistent")
        assert result is None

    async def test_record_usage_issues_upsert(self, mock_db_session):
        """record_usage issues a single execute (postgres ON CONFLICT upsert).

        Cannot be exercised against sqlite because the statement uses
        ``sqlalchemy.dialects.postgresql.insert(...).on_conflict_do_update``,
        so a mock-based test is the only way to lock in the call.
        """
        await JobManager.record_usage(
            mock_db_session,
            tenant_id="01950000-0000-7000-8000-000000000001",
            jobs_completed=1,
            features_generated=5,
        )
        mock_db_session.execute.assert_awaited_once()
