"""Gherkin and BDD artifact schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GherkinStep(BaseModel):
    """A single Gherkin step (Given/When/Then)."""

    keyword: str
    text: str
    data_table: list[list[str]] | None = None


class GherkinScenario(BaseModel):
    """A Gherkin scenario with steps."""

    name: str
    tags: list[str] = []
    steps: list[GherkinStep] = []
    #: How many of ``steps`` are the feature's ``Background`` steps, which the parser
    #: PREPENDS to every scenario (so they run before each one, as Gherkin requires).
    #: Narration needs to tell them apart: spoken over every clip they turn each beat
    #: into the same "First, I open the demo" sentence. Internal — excluded from any
    #: serialised payload so no API response changes.
    background_step_count: int = Field(default=0, ge=0, exclude=True)


class GherkinFeature(BaseModel):
    """A parsed Gherkin feature."""

    name: str
    description: str = ""
    tags: list[str] = []
    scenarios: list[GherkinScenario] = []


class FeatureFileInfo(BaseModel):
    """Metadata about a generated .feature file."""

    filename: str
    path: str
    scenario_count: int
    tags: list[str] = []


class StepDefFileInfo(BaseModel):
    """Metadata about a generated TypeScript step definition file."""

    filename: str
    path: str
    step_count: int
