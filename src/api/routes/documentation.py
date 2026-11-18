"""Interactive documentation stream route."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from src.api.routes._helpers import get_license_token, get_tenant_id
from src.api.schemas.common import AsyncJobResponse
from src.api.schemas.documentation import DocumentationRequest
from src.database.session import get_db_session
from src.services.background import dispatch_job
from src.services.job_manager import JobManager
from src.services.job_processor import JobProcessor

logger = logging.getLogger(__name__)

router = APIRouter()


def _detect_input_type(
    request_body: DocumentationRequest,
) -> tuple[str, str | None, str | None]:
    """Determine input_type, target_url, and app_path from the request body."""
    if request_body.url:
        return "web_url", request_body.url, None
    if request_body.app_path:
        return "mobile_app", None, request_body.app_path
    if request_body.feature_files:
        return "gherkin_upload", None, None
    if request_body.inspection_job_id:
        return "inspection_passthrough", None, None
    raise HTTPException(
        status_code=400,
        detail="At least one input source is required: url, app_path, feature_files, or inspection_job_id",
    )


@router.post(
    "/documentation",
    operation_id="start_bdd_documentation",
    summary="Start a BDD interactive-documentation generation job",
    responses={
        400: {"description": "No valid input source provided"},
        422: {"description": "Validation error"},
    },
)
async def start_documentation(
    request_body: DocumentationRequest, request: Request
) -> AsyncJobResponse:
    """Accept a documentation generation request and create a background job."""
    tenant_id = get_tenant_id(request)

    input_type, target_url, app_path = _detect_input_type(request_body)

    config = {
        "output_format": request_body.output_format,
        "include_screenshots": request_body.include_screenshots,
        "max_scenarios_per_feature": request_body.max_scenarios_per_feature,
        "feature_files": request_body.feature_files,
        "inspection_job_id": request_body.inspection_job_id,
        "feed_to_video": request_body.feed_to_video,
    }
    narration_config = request_body.narration.model_dump() if request_body.narration else {}

    async with get_db_session() as db:
        job = await JobManager.create_job(
            db,
            tenant_id=tenant_id,
            stream="documentation",
            input_type=input_type,
            target_url=target_url,
            app_path=app_path,
            config=config,
            narration_config=narration_config,
        )
        job_id = job.job_id

    logger.info("Created documentation job %s for tenant %s", job_id, tenant_id)

    processor = JobProcessor()
    dispatch_job(processor, job_id, tenant_id, license_token=get_license_token(request))

    return AsyncJobResponse(
        job_id=job_id,
        status="pending",
        poll_url=f"/api/bdd/{job_id}",
    )
