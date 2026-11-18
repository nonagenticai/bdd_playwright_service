"""HTTP client for app_inspector_service."""

from __future__ import annotations

import logging
from typing import Any

from w2v_common.clients.base import BaseServiceClient

logger = logging.getLogger(__name__)


class AppInspectorClient(BaseServiceClient):
    """Async HTTP client for the app_inspector_service.

    Inherits retry logic, adaptive polling, and connection management
    from :class:`BaseServiceClient`.
    """

    def __init__(
        self,
        base_url: str,
        license_token: str | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            service_name="app_inspector",
            timeout=60.0,
            license_token=license_token,
        )

    async def start_inspection(
        self,
        app_path: str,
        platform: str | None = None,
        max_screens: int = 20,
    ) -> str:
        """Start an inspection job on the app_inspector_service.

        Args:
            app_path: Path or URL of the application to inspect.
            platform: Platform hint (e.g. 'web', 'ios', 'android').
            max_screens: Maximum number of screens to inspect.

        Returns:
            The job_id of the started inspection.
        """
        payload: dict[str, Any] = {
            "app_path": app_path,
            "max_screens": max_screens,
        }
        if platform:
            payload["platform"] = platform

        return await self.submit_job("/api/inspect", payload)

    async def get_inspection(self, job_id: str) -> dict[str, Any]:
        """Get the current status and results of an inspection job.

        Args:
            job_id: The inspection job identifier.

        Returns:
            Dict with job status, results, and metadata.
        """
        return await self._request("GET", f"/api/inspect/{job_id}")

    async def wait_for_inspection(
        self,
        job_id: str,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Poll the inspection job until it completes or times out.

        Delegates to :meth:`BaseServiceClient.poll_job` which provides
        adaptive backoff and transport-error resilience.

        Args:
            job_id: The inspection job identifier.
            timeout: Maximum seconds to wait before raising TimeoutError.

        Returns:
            The completed inspection result dict.

        Raises:
            TimeoutError: If the job does not complete within the timeout.
            RuntimeError: If the downstream job reports failure.
        """
        return await self.poll_job("/api/inspect", job_id, timeout=timeout)
