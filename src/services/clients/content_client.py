"""HTTP client for content_service (optional narration enhancement)."""

from __future__ import annotations

from typing import Any

from w2v_common.clients.base import BaseServiceClient


class ContentClient(BaseServiceClient):
    """Async HTTP client for the Content service."""

    def __init__(self, base_url: str, license_token: str | None = None) -> None:
        super().__init__(
            base_url=base_url,
            service_name="content_service",
            timeout=60.0,
            license_token=license_token,
        )

    async def enhance_narration(
        self,
        shot_list: dict[str, Any],
        storyline: dict[str, Any],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Enhance shot list actions via content service."""
        payload = {
            "shot_list": shot_list,
            "storyline": storyline,
            "analysis": analysis,
        }
        return await self._request("POST", "/api/shotlist/enhance", json=payload)
