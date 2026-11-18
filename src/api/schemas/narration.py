"""Narration output schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class NarrationPart(BaseModel):
    """A narration segment corresponding to a Gherkin scenario."""

    part_id: str
    scenario_name: str
    text: str
    original_gherkin: str
    word_count: int
    estimated_duration_seconds: float
    #: Where the pre-polish line came from: ``"pattern"``, ``"authored"`` or
    #: ``"template_fallback"`` (canned per-tone sentence, no pattern matched).
    #: Optional and observability-only; ``None`` on older results.
    narration_source: str | None = None
    #: Steps in this line narrated by the fallback template.
    template_fallback_steps: int | None = None


class NarrationScriptOutput(BaseModel):
    """Complete narration script generated from BDD features."""

    parts: list[NarrationPart] = []
    total_words: int = 0
    estimated_total_duration_seconds: float = 0.0
    tone: str = "tutorial"
    #: Per-section budget outcome (see ``job_processor._word_budget_report``).
    #: Present only when a word budget applied or an LLM narration pass ran. It
    #: carries the RESIDUAL shortfall per section as data, so the orchestrator
    #: can read how short a beat is instead of stretching speech to fill the slot.
    word_budget_report: dict[str, Any] | None = None
