"""Package download route.

Streams the generated BDD package zip from the SHARED object store, so a
download works no matter which replica serves it. The prior implementation
served a pod-local ``FileResponse`` guarded by ``Path.exists()``, which 404'd
when the request landed on a replica other than the one that produced the job.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from src.api.routes._helpers import get_tenant_id
from src.database.session import get_db_session
from src.services.job_manager import JobManager
from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)

router = APIRouter()

# Response headers forwarded from the store's streamed object (keeps the
# range/caching contract intact); the rest (X-Amz-*, server) are dropped.
_FORWARD_HEADERS = frozenset(
    {"content-length", "content-range", "accept-ranges", "etag", "last-modified"}
)
# Request headers forwarded to the store so partial/conditional GETs work.
_FORWARD_REQUEST_HEADERS = frozenset({"range", "if-range", "if-none-match", "if-modified-since"})


def _get_storage(request: Request) -> StorageService:
    """Reuse the app-lifespan storage service (already initialized) if present,
    else construct one (e.g. under the test client's bare app)."""
    storage = getattr(request.app.state, "storage_service", None)
    if isinstance(storage, StorageService):
        return storage
    return StorageService()


@router.get(
    "/{job_id}/download",
    response_model=None,
    operation_id="download_bdd_package",
    summary="Download the generated BDD package as a zip file",
    responses={
        404: {"description": "Job or package file not found"},
        409: {"description": "Job not completed yet"},
    },
)
async def download_package(job_id: str, request: Request) -> StreamingResponse:
    """Download the generated BDD package as a zip file."""
    tenant_id = get_tenant_id(request)
    async with get_db_session() as db:
        job = await JobManager.get_job(db, job_id, tenant_id=tenant_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.status}. Only completed jobs can be downloaded.",
            headers={"Retry-After": "30"},
        )

    artifact_paths = job.artifact_paths or {}
    package_ref = artifact_paths.get("package_zip")

    if not package_ref:
        raise HTTPException(
            status_code=404,
            detail=f"Package file not found for job {job_id}. It may not have been built yet.",
        )

    storage = _get_storage(request)
    req_headers = {
        k: v for k, v in request.headers.items() if k.lower() in _FORWARD_REQUEST_HEADERS
    }
    stream = await storage.open_stream(package_ref, request_headers=req_headers or None)

    if stream.status == 404:
        await stream.aclose()
        raise HTTPException(
            status_code=404,
            detail=f"Package file not found for job {job_id}. It may not have been built yet.",
        )

    headers = {
        "Content-Disposition": f'attachment; filename="bdd-{job.stream}-{job_id[:8]}.zip"',
    }
    headers.update({k: v for k, v in stream.headers.items() if k.lower() in _FORWARD_HEADERS})

    async def _body() -> AsyncIterator[bytes]:
        async for chunk in stream.iter_chunks():
            yield chunk

    return StreamingResponse(
        _body(),
        status_code=stream.status,
        media_type="application/zip",
        headers=headers,
        background=BackgroundTask(stream.aclose),
    )
