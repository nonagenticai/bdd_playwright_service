"""Product demo/tutorial video stream schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.api.schemas.common import LLMConfigOverride, NarrationConfig
from src.api.schemas.gherkin import FeatureFileInfo, StepDefFileInfo
from src.api.schemas.narration import NarrationScriptOutput


class DemoVideoRequest(BaseModel):
    """Request to generate a product demo/tutorial video via BDD."""

    # Input source (one required)
    url: str | None = None
    app_path: str | None = None
    feature_files: list[str] | None = None
    inspection_job_id: str | None = None
    # A knowledge pack — {access:{base_url, auth}, screens:[{id, route, label,
    # narration, capture:{click, scroll_to, wait_for, hold_ms}}]} — drives a
    # DETERMINISTIC showcase package: one @clip-<n> scenario per screen using the
    # fixed showcase step vocabulary (no LLM invents structure or steps; the LLM
    # only polishes narration). `access.auth` may carry `sso_button`/`success_url`
    # to drive the scaffold's SSO login branch.
    knowledge: dict[str, Any] | None = None

    # Configuration
    target_pages: list[str] | None = None
    focus_areas: list[str] | None = None
    max_scenarios_per_feature: int = Field(default=10, ge=1, le=50)
    # THE MINIMUM number of distinct recorded scenarios the caller needs, sent by the
    # orchestrator alongside the cap. OPTIONAL and backward compatible: absent means no
    # floor, exactly today's behaviour (the scenario count stays purely the model's
    # choice, and `focus_areas` remains the only thing that can imply a minimum).
    #
    # It exists because the cap is only a ceiling: a caller that orders many narration
    # beats has no way to demand enough distinct visuals to fill them, so a short draw
    # gets fanned onto the beat plan and the delivered video cycles the same footage.
    #
    # It is a FLOOR, never an extension: `_scenario_floor` clamps it by
    # `max_scenarios_per_feature`, so min_scenarios=9 under a cap of 2 still yields 2.
    min_scenarios: int | None = Field(default=None, ge=1)
    # THE ORDERED FILM DURATION, in seconds. OPTIONAL and backward compatible:
    # absent means today's behaviour exactly (every scenario records the fixed
    # HERO_DWELL_MS dwell and narration is written to no length).
    #
    # Present, it becomes a per-scenario second budget — ordered / scenarios
    # ACTUALLY generated — that bounds BOTH the narration word count and the
    # dwell ledger, because this service authors both and until now neither had
    # a budget. Reaching this service only as `max_scenarios_per_feature` was
    # not enough: one beat's footage (~24s) is comparable to a whole 30s order,
    # so NO integer scenario count lands within +-10% of it.
    #
    # It is a BOUND, never an extension: clamped above by HERO_DWELL_MS and
    # below by the per-scenario floor. A clamp that makes the order unmeetable is
    # RECORDED on `step_coverage.duration_budget` (`order_met: false`) — never
    # raised.
    target_duration_seconds: float | None = Field(default=None, gt=0)

    # THE NARRATION WORD BUDGET, sent by the orchestrator alongside the ordered
    # duration. OPTIONAL by contract, BOTH of them: any orchestrator that has not
    # deployed yet sends neither, and such a payload must behave exactly as it does
    # today (the word budget is then derived locally from the ordered duration and
    # the configured words-per-minute).
    #
    # They exist because this service can only GUESS the spoken length from a
    # configured `wpm`, while the orchestrator knows the delivered speech rate. The
    # budget is what the script is written to FILL, not merely a ceiling to trim
    # to — a demo that under-runs its order is the same defect as one that overruns.
    narration_words_target: int | None = Field(default=None, ge=1)
    #: Per-beat split, in the SAME document order as the `@clip-<n>` scenarios.
    #: Discarded (logged, never zipped short) when its length disagrees with the
    #: scenario count the LLM actually emitted; the total is then split evenly.
    narration_words_per_scenario: list[int] | None = None

    # Authentication for login-gated demo targets
    auth_config: dict[str, Any] | None = None

    # Narration (required for demo videos)
    narration: NarrationConfig = Field(default_factory=NarrationConfig)

    # Video feed
    feed_to_video: bool = True

    # LLM override
    llm_config: LLMConfigOverride | None = None


class DemoVideoResult(BaseModel):
    """Result of demo/tutorial video generation."""

    job_id: str
    status: str
    features: list[FeatureFileInfo] = []
    step_definitions: list[StepDefFileInfo] = []
    config_file: str | None = None
    narration_script: NarrationScriptOutput | None = None
    package_download_url: str | None = None
    video_job_id: str | None = None
