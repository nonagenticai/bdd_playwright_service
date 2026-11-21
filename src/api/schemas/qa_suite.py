"""QA Test Suite stream schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.api.schemas.common import LLMConfigOverride, NarrationConfig
from src.api.schemas.gherkin import FeatureFileInfo, StepDefFileInfo
from src.api.schemas.narration import NarrationScriptOutput


class QASuiteRequest(BaseModel):
    """Request to generate a QA test suite from a website or app."""

    # Input source (one required)
    url: str | None = None
    app_path: str | None = None
    # Caller-rendered HTML snapshot: a stored ``uploads/{tenant_id}/…json`` of
    # ``[{url, html, title?, text?, links?}]``. The crawl's replacement now that
    # webscraper is retired — the caller renders, this service extracts structure.
    html_snapshot_path: str | None = None
    feature_files: list[str] | None = None
    inspection_job_id: str | None = None

    # Configuration
    target_pages: list[str] | None = None
    focus_areas: list[str] | None = None
    include_negative_tests: bool = True
    max_scenarios_per_feature: int = Field(default=10, ge=1, le=50)

    # Authentication for login-gated sites
    auth_config: dict[str, Any] | None = None

    # Narration (optional)
    narration: NarrationConfig | None = None
    feed_to_video: bool = False

    # LLM override
    llm_config: LLMConfigOverride | None = None


class QASuiteResult(BaseModel):
    """Result of QA test suite generation."""

    job_id: str
    status: str
    features: list[FeatureFileInfo] = []
    step_definitions: list[StepDefFileInfo] = []
    config_file: str | None = None
    narration_script: NarrationScriptOutput | None = None
    package_download_url: str | None = None
    video_job_id: str | None = None
