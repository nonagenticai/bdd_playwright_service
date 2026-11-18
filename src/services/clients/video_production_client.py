"""HTTP client for video_production_service at port 6308."""

from __future__ import annotations

from typing import Any

from w2v_common.clients.base import BaseServiceClient


class VideoProductionClient(BaseServiceClient):
    """Async HTTP client for the Video Production service."""

    def __init__(self, base_url: str, license_token: str | None = None) -> None:
        super().__init__(
            base_url=base_url,
            service_name="video_production_service",
            timeout=120.0,
            license_token=license_token,
        )

    async def start_recording(self, payload: dict[str, Any]) -> str:
        """Submit a recording job and return the job ID."""
        return await self.submit_job("/api/record/part", payload)

    async def get_recording(self, job_id: str) -> dict[str, Any]:
        """Get recording job status."""
        return await self._request("GET", f"/api/record/jobs/{job_id}")

    async def wait_for_recording(self, job_id: str, timeout: float = 600.0) -> dict[str, Any]:
        """Poll recording job until completion."""
        return await self.poll_job(
            endpoint="/api/record/jobs",
            job_id=job_id,
            timeout=timeout,
        )
