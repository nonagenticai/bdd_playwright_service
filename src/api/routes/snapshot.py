"""HTML-snapshot upload route.

The sibling of ``gherkin.py`` for the crawl's replacement: ``webscraper_service``
is retired, so a caller that can render the pages itself (Playwright, or curl
behind a login) uploads the DOM here and a ``qa-suite`` job with
``html_snapshot_path`` turns it into the same inspection a crawl produced.

Only the storage prefix matters for confinement — ``uploads/{tenant_id}/`` —
and that chokepoint lives in ``StorageService``, exactly as it does for
``.feature`` uploads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile

from src.api.routes._helpers import get_tenant_id
from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/snapshot/upload",
    operation_id="upload_bdd_html_snapshot",
    summary="Upload an HTML snapshot (JSON) for a qa-suite job",
    responses={
        400: {"description": "No files provided or invalid file type"},
        422: {"description": "Validation error"},
    },
)
async def upload_snapshot(files: list[UploadFile], request: Request) -> dict[str, Any]:
    """Accept a ``.json`` snapshot upload, store it, and return its storage path."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    tenant_id = get_tenant_id(request)

    storage = StorageService()
    stored_paths: list[str] = []
    try:
        for file in files:
            if not file.filename:
                continue
            if not file.filename.endswith(".json"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only .json snapshots are accepted. Got: {file.filename}",
                )

            # The multipart filename is fully caller-controlled; collapse it to a bare
            # basename so a traversal segment (``../../etc/x.json``) cannot escape
            # the tenant upload dir. Storage confinement is the backstop; this gives a
            # clean 400 instead of a 500.
            safe_name = Path(file.filename).name
            if safe_name != file.filename or not safe_name:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid filename (path components not allowed): {file.filename}",
                )

            content = await file.read()
            path = await storage.store_bytes(
                f"uploads/{tenant_id}",
                content,
                safe_name,
            )
            stored_paths.append(path)
            logger.debug("Stored uploaded HTML snapshot: %s", path)
    finally:
        # One StorageService per request: close its MinIO aiohttp session.
        await storage.aclose()

    if not stored_paths:
        raise HTTPException(status_code=400, detail="No valid .json snapshots provided")

    return {
        "uploaded": len(stored_paths),
        "paths": stored_paths,
        "tenant_id": tenant_id,
    }
