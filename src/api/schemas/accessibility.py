"""Accessibility audit stream schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.api.schemas.common import LLMConfigOverride, NarrationConfig
from src.api.schemas.gherkin import FeatureFileInfo, StepDefFileInfo
from src.api.schemas.narration import NarrationScriptOutput


class AccessibilityRequest(BaseModel):
    """Request to generate accessibility audit BDD scenarios."""

    # Input source (one required)
    url: str | None = None
    app_path: str | None = None
    inspection_job_id: str | None = None

    # Configuration
    wcag_level: str = Field(default="AA", description="WCAG conformance level: A, AA, or AAA")
    focus_areas: list[str] | None = Field(
        default=None,
        description="Specific a11y areas: keyboard, screen_reader, contrast, focus, forms",
    )
    max_scenarios_per_feature: int = Field(default=10, ge=1, le=50)

    # Narration (optional)
    narration: NarrationConfig | None = None
    feed_to_video: bool = False

    # LLM override
    llm_config: LLMConfigOverride | None = None


class AccessibilityResult(BaseModel):
    """Result of accessibility audit scenario generation."""

    job_id: str
    status: str
    features: list[FeatureFileInfo] = []
    step_definitions: list[StepDefFileInfo] = []
    config_file: str | None = None
    narration_script: NarrationScriptOutput | None = None
    package_download_url: str | None = None
    video_job_id: str | None = None
