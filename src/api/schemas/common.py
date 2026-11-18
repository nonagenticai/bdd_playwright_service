"""Common schemas shared across all BDD streams."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

JobStatus = Literal["pending", "processing", "completed", "failed"]
StreamType = Literal["qa_suite", "accessibility", "documentation", "demo_video"]
InputType = Literal["web_url", "mobile_app", "gherkin_upload", "inspection_passthrough"]
NarrationTone = Literal["tutorial", "demo", "marketing", "technical"]


class AsyncJobResponse(BaseModel):
    """Immediate response when a BDD job is accepted."""

    job_id: str
    status: JobStatus = "pending"
    poll_url: str


class JobStatusResponse(BaseModel):
    """Status response for a BDD job."""

    job_id: str
    status: JobStatus
    stream: StreamType
    features_generated: int = 0
    step_defs_generated: int = 0
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    download_url: str | None = None


class NarrationConfig(BaseModel):
    """Narration configuration for humanizing Gherkin output."""

    enabled: bool = True
    tone: NarrationTone = "tutorial"
    custom_patterns: dict[str, list[str]] | None = None
    llm_polish: bool = True


class LLMConfigOverride(BaseModel):
    """Optional LLM configuration override per request."""

    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
