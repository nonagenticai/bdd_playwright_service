"""Job status and result endpoints."""

from __future__ import annotations

import logging
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request

from src.api.routes._helpers import get_tenant_id
from src.api.schemas.common import JobStatus, JobStatusResponse, StreamType
from src.database.session import get_db_session
from src.services.job_manager import JobManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/{job_id}",
    operation_id="get_bdd_job_status",
    summary="Poll the status of a BDD generation job",
    responses={
        404: {"description": "Job not found"},
    },
)
async def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
    """Poll the status of a BDD generation job."""
    tenant_id = get_tenant_id(request)
    async with get_db_session() as db:
        job = await JobManager.get_job(db, job_id, tenant_id=tenant_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return JobStatusResponse(
            job_id=job.job_id,
            status=cast(JobStatus, job.status),
            stream=cast(StreamType, job.stream),
            features_generated=job.features_generated or 0,
            step_defs_generated=job.step_defs_generated or 0,
            created_at=job.created_at.isoformat() if job.created_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            error=job.error_message,
            download_url=f"/api/bdd/{job.job_id}/download" if job.status == "completed" else None,
        )


@router.get(
    "/{job_id}/result",
    operation_id="get_bdd_job_result",
    summary="Get the full result of a completed BDD generation job",
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job not completed yet"},
    },
)
async def get_job_result(job_id: str, request: Request) -> dict[str, Any]:
    """Get the full result of a completed BDD generation job."""
    tenant_id = get_tenant_id(request)
    async with get_db_session() as db:
        job = await JobManager.get_job(db, job_id, tenant_id=tenant_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        if job.status not in ("completed", "failed"):
            raise HTTPException(
                status_code=409,
                detail=f"Job is still {job.status}. Poll /api/bdd/{job_id} for status.",
                headers={"Retry-After": "30"},
            )
        return {
            "job_id": job.job_id,
            "status": job.status,
            "stream": job.stream,
            "features_generated": job.features_generated or 0,
            "step_defs_generated": job.step_defs_generated or 0,
            "artifact_paths": job.artifact_paths or {},
            "narration_script": job.narration_script or {},
            # Stub visibility for the demo_video stream: `degraded` is True when a
            # MAJORITY of the suite's steps are non-acting auto-stubs, i.e. the
            # recording largely films a static page. Structured on purpose — the
            # share used to exist only as a generator log line.
            "step_coverage": job.step_coverage or {},
            "downstream_jobs": job.downstream_jobs or {},
            "generation_time_seconds": job.generation_time_seconds,
            "error": job.error_message,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }
