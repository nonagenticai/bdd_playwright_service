"""Thin adapter that routes BDD generation LLM calls through worker_management.

Per the cluster-wide invariant (memory ``wm-as-sole-llm-gateway.md``), w2v
services NEVER call openai / anthropic / inference-router directly. They post
to WM's ``POST /workers/{worker_name}/message`` endpoint, which routes to the
right provider, attributes cost to the caller's tenant, and supports per-call
``system_prompt`` overrides.

This module exposes a tiny ``LLMClient`` shim with a ``.generate(prompt,
system_prompt=...)`` API so call sites keep their existing shape, while the
underlying transport is ``WMClient.message(...)``. Each LLMClient instance
binds to a specific WM worker name (e.g. ``bdd-feature-generator``,
``bdd-step-def-generator``, ``bdd-narration-humanizer``) so cost / usage
metrics in WM are bucketed per subsystem.

A single ``WMClient`` is shared across all subsystems via the
``shared_wm_client()`` accessor — its underlying ``httpx.AsyncClient`` is
connection-pooled, and the identity event hook reads ContextVars per
outbound request, so the same client serves concurrent requests from
different end-users correctly.

Backend: ``WMClient`` honours w2v_common's ``LLM_BACKEND``. With ``wm`` the call
goes to worker_management; with ``vllm`` (live since 2026-10-18) it bypasses WM
and calls the pgx vLLM directly (``w2v_common.clients.vllm_direct``). Keep log
text here backend-neutral — "via WM" is false under ``vllm``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from w2v_common.clients import WMClient

logger = logging.getLogger(__name__)


_shared_wm_client: WMClient | None = None


def shared_wm_client() -> WMClient:
    """Return the process-wide ``WMClient`` instance, creating it on first use."""
    global _shared_wm_client
    if _shared_wm_client is None:
        _shared_wm_client = WMClient()
    return _shared_wm_client


async def aclose_shared_wm_client() -> None:
    """Close the shared ``WMClient`` (call from app shutdown)."""
    global _shared_wm_client
    if _shared_wm_client is not None:
        await _shared_wm_client.aclose()
        _shared_wm_client = None


class LLMClient:
    """Adapter that exposes a ``generate()`` API backed by a WM worker.

    Each subsystem (feature generation, step-def generation, narration
    humanization) constructs its own ``LLMClient`` bound to a stable
    ``worker_name``. WM auto-creates the worker on first call.
    """

    def __init__(self, worker_name: str, wm_client: WMClient | None = None) -> None:
        self.worker_name = worker_name
        self._wm = wm_client if wm_client is not None else shared_wm_client()

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        """Send the prompt to the bound WM worker and return the response text.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system prompt — lifted verbatim into WM's
                per-call ``system_prompt`` so the worker doesn't need a
                pre-configured config row.
            output_schema: Optional JSON Schema for the response. WM validates the
                model's output against it, so the caller gets a machine-checked
                shape instead of recovering structure from free text with a regex.
                Omitted entirely when ``None`` — every existing call site keeps
                today's plain-text behaviour byte for byte.

        Returns:
            The assistant's response text. A structured payload is returned as
            JSON (``json.dumps``), NOT ``str(dict)`` — Python's repr uses single
            quotes and is not parseable by any JSON reader, so the old fallback
            handed text-parsing call sites something no parser could read.
        """
        logger.debug("LLM request worker=%s prompt_len=%d", self.worker_name, len(prompt))
        result = await self._wm.message(
            worker_name=self.worker_name,
            message=prompt,
            system_prompt=system_prompt,
            output_schema=output_schema,
        )
        output = result.output
        if isinstance(output, str):
            content = output
        elif output is None:
            content = ""
        else:
            content = json.dumps(output)
        logger.debug("LLM response: %d chars", len(content))
        return content
