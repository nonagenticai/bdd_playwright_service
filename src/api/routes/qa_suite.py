"""QA Test Suite stream route."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from src.api.constants import (
    INPUT_GHERKIN_UPLOAD,
    INPUT_HTML_SNAPSHOT,
    INPUT_INSPECTION_PASSTHROUGH,
    INPUT_MOBILE_APP,
    INPUT_WEB_URL,
)
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

    Precedence, first match wins: ``url`` → ``app_path`` → ``html_snapshot_path``
    → ``feature_files`` → ``inspection_job_id``. ``html_snapshot_path`` sits
    above ``feature_files`` because it needs a crawl-equivalent inspection: a
    request carrying both wants generated features, not step defs for supplied
    Gherkin.
    """
    if request_body.url:
        return INPUT_WEB_URL, request_body.url, None
    if request_body.app_path:
        return INPUT_MOBILE_APP, None, request_body.app_path
    if request_body.html_snapshot_path:
        return INPUT_HTML_SNAPSHOT, None, None
    if request_body.feature_files:
        return INPUT_GHERKIN_UPLOAD, None, None
    if request_body.inspection_job_id:
        return INPUT_INSPECTION_PASSTHROUGH, None, None
    raise HTTPException(
        status_code=400,
        detail=(
            "At least one input source is required: url, app_path, "
            "html_snapshot_path, feature_files, or inspection_job_id"
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
        "focus_areas": request_body.focus_areas,
        "include_negative_tests": request_body.include_negative_tests,
        "max_scenarios_per_feature": request_body.max_scenarios_per_feature,
        "html_snapshot_path": request_body.html_snapshot_path,
        "feature_files": request_body.feature_files,
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
