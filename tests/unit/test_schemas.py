"""Tests for Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas.accessibility import AccessibilityRequest
from src.api.schemas.common import (
    AsyncJobResponse,
    JobStatusResponse,
    LLMConfigOverride,
    NarrationConfig,
)
from src.api.schemas.demo_video import DemoVideoRequest
from src.api.schemas.documentation import DocumentationRequest
from src.api.schemas.gherkin import (
    FeatureFileInfo,
    GherkinFeature,
    GherkinScenario,
    GherkinStep,
    StepDefFileInfo,
)
from src.api.schemas.narration import NarrationPart, NarrationScriptOutput
from src.api.schemas.qa_suite import QASuiteRequest


class TestCommonSchemas:
    def test_async_job_response(self):
        r = AsyncJobResponse(job_id="abc", status="pending", poll_url="/api/bdd/abc")
        assert r.job_id == "abc"
        assert r.status == "pending"

    def test_job_status_response(self):
        r = JobStatusResponse(
            job_id="abc",
            status="completed",
            stream="qa_suite",
            features_generated=5,
        )
        assert r.features_generated == 5
        assert r.stream == "qa_suite"

    def test_narration_config_defaults(self):
        n = NarrationConfig()
        assert n.enabled is True
        assert n.tone == "tutorial"
        assert n.llm_polish is True

    def test_llm_config_override(self):
        c = LLMConfigOverride(model="gpt-4o", temperature=0.5)
        assert c.model == "gpt-4o"
        assert c.max_tokens is None


class TestGherkinSchemas:
    def test_gherkin_step(self):
        s = GherkinStep(keyword="Given", text="I am on the login page")
        assert s.keyword == "Given"
        assert s.data_table is None

    def test_gherkin_scenario(self):
        sc = GherkinScenario(
            name="Login",
            tags=["@smoke"],
            steps=[GherkinStep(keyword="Given", text="I am on the login page")],
        )
        assert len(sc.steps) == 1
        assert sc.tags == ["@smoke"]

    def test_gherkin_feature(self):
        f = GherkinFeature(name="Login", description="User login flow")
        assert f.name == "Login"
        assert f.scenarios == []

    def test_feature_file_info(self):
        f = FeatureFileInfo(
            filename="login.feature", path="/features/login.feature", scenario_count=3
        )
        assert f.scenario_count == 3

    def test_step_def_file_info(self):
        s = StepDefFileInfo(filename="login.steps.ts", path="/steps/login.steps.ts", step_count=6)
        assert s.step_count == 6


class TestNarrationSchemas:
    def test_narration_part(self):
        p = NarrationPart(
            part_id="p1",
            scenario_name="Login",
            text="Let's start by navigating to the login page.",
            original_gherkin="Given I am on the login page",
            word_count=8,
            estimated_duration_seconds=3.2,
        )
        assert p.word_count == 8

    def test_narration_script_output(self):
        o = NarrationScriptOutput()
        assert o.parts == []
        assert o.total_words == 0


class TestQASuiteSchemas:
    def test_qa_suite_request_with_url(self):
        r = QASuiteRequest(url="https://example.com")
        assert r.url == "https://example.com"
        assert r.include_negative_tests is True
        assert r.feed_to_video is False

    def test_qa_suite_request_max_scenarios_validation(self):
        with pytest.raises(ValidationError):
            QASuiteRequest(url="https://example.com", max_scenarios_per_feature=0)

        with pytest.raises(ValidationError):
            QASuiteRequest(url="https://example.com", max_scenarios_per_feature=51)


class TestAccessibilitySchemas:
    def test_accessibility_request_defaults(self):
        r = AccessibilityRequest(url="https://example.com")
        assert r.wcag_level == "AA"

    def test_accessibility_request_with_focus_areas(self):
        r = AccessibilityRequest(
            url="https://example.com",
            focus_areas=["keyboard", "screen_reader"],
        )
        assert r.focus_areas is not None
        assert len(r.focus_areas) == 2


class TestDocumentationSchemas:
    def test_documentation_request_defaults(self):
        r = DocumentationRequest(url="https://example.com")
        assert r.output_format == "markdown"
        assert r.include_screenshots is True


class TestDemoVideoSchemas:
    def test_demo_video_request_defaults(self):
        r = DemoVideoRequest(url="https://example.com")
        assert r.feed_to_video is True
        assert r.narration.tone == "tutorial"
