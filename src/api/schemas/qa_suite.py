"""QA Test Suite stream schemas."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from src.api.schemas.common import LLMConfigOverride, NarrationConfig
from src.api.schemas.gherkin import FeatureFileInfo, StepDefFileInfo
from src.api.schemas.narration import NarrationScriptOutput
from src.services.input.page_fetcher import PageFetchError, validate_auth_config


class QASuiteRequest(BaseModel):
    """Request to generate a QA test suite from a website or app."""

    # Input source (one required)
    url: str | None = None
    html_snapshot_path: str | None = None
    app_path: str | None = None
    feature_files: list[str] | None = None
    inspection_job_id: str | None = None

    # Configuration
    target_pages: list[str] | None = None
    javascript_enabled: bool = True
    focus_areas: list[str] | None = None
    include_negative_tests: bool = True
    max_scenarios_per_feature: int = Field(default=10, ge=1, le=50)

    #: ``grounded`` (default) builds deterministic page contracts from the captured
    #: structure and validates every scenario against it; ``legacy`` keeps the old
    #: LLM-authors-everything path. A job with ``auth_config`` always uses the
    #: legacy path — the grounded vocabulary has no storage-state setup yet.
    generation_mode: Literal["grounded", "legacy"] = "grounded"
    #: Opt-in: let the LLM add scenarios to the grounded floor. They must use the
    #: same closed vocabulary and are dropped if the snapshot cannot ground them.
    include_llm_scenarios: bool = False

    # Authentication for login-gated sites
    auth_config: dict[str, Any] | None = None

    # Narration (optional)
    narration: NarrationConfig | None = None
    feed_to_video: bool = False

    # LLM override
    llm_config: LLMConfigOverride | None = None

    @model_validator(mode="after")
    def _reject_secret_bearing_auth_config(self) -> Self:
        """Reject credential-shaped keys before they are persisted on the job row."""
        try:
            validate_auth_config(self.auth_config)
        except PageFetchError as exc:
            raise ValueError(str(exc)) from exc
        return self


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
