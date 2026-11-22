"""QA Test Suite stream route."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from src.api.routes._helpers import get_license_token, get_tenant_id
from src.api.schemas.common import AsyncJobResponse
from src.api.schemas.qa_suite import QASuiteRequest
from src.database.session import get_db_session
from src.services.background import dispatch_job
from src.services.job_manager import JobManager
from src.services.job_processor import JobProcessor

logger = logging.getLogger(__name__)

router = APIRouter()


def _detect_input_type(request_body: QASuiteRequest) -> tuple[str, str | None, str | None]:
    """Determine input_type, target_url, and app_path from the request body.

    Precedence is order-based, not exactly-one: ``url`` wins over an uploaded
    snapshot, which wins over the legacy ``app_path``/``feature_files``/inspection
    inputs.
    """
    if request_body.url:
        return "web_url", request_body.url, None
    if request_body.html_snapshot_path:
        return "html_snapshot", None, None
    if request_body.app_path:
        return "mobile_app", None, request_body.app_path
    if request_body.feature_files:
        return "gherkin_upload", None, None
    if request_body.inspection_job_id:
        return "inspection_passthrough", None, None
    raise HTTPException(
        status_code=400,
        detail=(
            "At least one input source is required: url, html_snapshot_path, app_path, "
            "feature_files, or inspection_job_id"
        ),
    )


@router.post(
    "/qa-suite",
    operation_id="start_bdd_qa_suite",
    summary="Start a BDD QA test-suite generation job",
    responses={
        400: {"description": "No valid input source provided"},
        422: {"description": "Validation error"},
    },
)
async def start_qa_suite(request_body: QASuiteRequest, request: Request) -> AsyncJobResponse:
    """Accept a QA suite generation request and create a background job."""
    tenant_id = get_tenant_id(request)

    input_type, target_url, app_path = _detect_input_type(request_body)

    config = {
        "target_pages": request_body.target_pages,
        "javascript_enabled": request_body.javascript_enabled,
        "focus_areas": request_body.focus_areas,
        "include_negative_tests": request_body.include_negative_tests,
        "max_scenarios_per_feature": request_body.max_scenarios_per_feature,
        "generation_mode": request_body.generation_mode,
        "include_llm_scenarios": request_body.include_llm_scenarios,
        "feature_files": request_body.feature_files,
        "html_snapshot_path": request_body.html_snapshot_path,
        "inspection_job_id": request_body.inspection_job_id,
        "feed_to_video": request_body.feed_to_video,
        "auth_config": request_body.auth_config,
    }
    narration_config = request_body.narration.model_dump() if request_body.narration else {}

    async with get_db_session() as db:
        job = await JobManager.create_job(
            db,
            tenant_id=tenant_id,
            stream="qa_suite",
            input_type=input_type,
            target_url=target_url,
            app_path=app_path,
            config=config,
            narration_config=narration_config,
        )
        job_id = job.job_id

    logger.info("Created QA suite job %s for tenant %s", job_id, tenant_id)

    processor = JobProcessor()
    dispatch_job(processor, job_id, tenant_id, license_token=get_license_token(request))

    return AsyncJobResponse(
        job_id=job_id,
        status="pending",
        poll_url=f"/api/bdd/{job_id}",
    )
