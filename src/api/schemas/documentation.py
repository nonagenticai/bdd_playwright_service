"""Interactive documentation stream schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.api.schemas.common import LLMConfigOverride, NarrationConfig
from src.api.schemas.gherkin import FeatureFileInfo, StepDefFileInfo
from src.api.schemas.narration import NarrationScriptOutput


class DocumentationRequest(BaseModel):
    """Request to generate interactive documentation from BDD scenarios."""

    # Input source (one required)
    url: str | None = None
    app_path: str | None = None
    feature_files: list[str] | None = None
    inspection_job_id: str | None = None

    # Configuration
    output_format: str = Field(default="markdown", description="Output format: markdown or html")
    include_screenshots: bool = True
    max_scenarios_per_feature: int = Field(default=10, ge=1, le=50)

    # Narration (optional)
    narration: NarrationConfig | None = None
    feed_to_video: bool = False

    # LLM override
    llm_config: LLMConfigOverride | None = None


class DocumentationResult(BaseModel):
    """Result of interactive documentation generation."""

    job_id: str
    status: str
    features: list[FeatureFileInfo] = []
    step_definitions: list[StepDefFileInfo] = []
    documentation_files: list[str] = []
    config_file: str | None = None
    narration_script: NarrationScriptOutput | None = None
    package_download_url: str | None = None
    video_job_id: str | None = None
