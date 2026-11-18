"""Tests for the deterministic knowledge-pack → showcase-package generator (PR-F).

Covers the emitter (screens[] → @clip-N feature), the fixed step library, the
auth mapping (incl. SSO fields), the job-processor knowledge branch, and the
verbatim-narration override.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.services.generation.config_generator import ConfigGenerator
from src.services.generation.feature_generator import FeatureGenerator, _gherkin_quote
from src.services.generation.llm_client import LLMClient
from src.services.job_processor import JobProcessor, _auth_config_from_pack


def _pack() -> dict[str, Any]:
    """A small representative knowledge pack (CP-shaped, two interior screens)."""
    return {
        "app": {"name": "Common Practice"},
        "access": {
            "base_url": "https://app.example.com",
            "auth": {
                "login_url": "https://app.example.com/login",
                "username": "demo@example.com",
                "password_env": "CP_PASSWORD",  # pragma: allowlist secret
                "sso_button": "button.sso",
                "success_url": "https://app.example.com/",
            },
        },
        "screens": [
            {
                "id": "home",
                "route": "/",
                "label": "Home dashboard",
                "narration": "This is the pulse of the whole organisation.",
                "capture": {"wait_for": ".page section.hero h1", "hold_ms": 6000},
            },
            {
                "id": "challenges",
                "route": "/challenges",
                "label": "Challenge leaderboard",
                "narration": "Every friction anyone hits, ranked by votes.",
                "capture": {
                    "wait_for": ".page .board",
                    "scroll_to": '.board [data-rank="1"]',
                    "hold_ms": 5000,
                },
            },
        ],
    }


@pytest.fixture
def feature_gen() -> FeatureGenerator:
    return FeatureGenerator(AsyncMock(spec=LLMClient))


class TestGherkinQuote:
    def test_escapes_double_quotes_in_selectors(self):
        assert _gherkin_quote('input[name="title"]') == r'"input[name=\"title\"]"'

    def test_plain_route(self):
        assert _gherkin_quote("/challenges") == '"/challenges"'


class TestBuildShowcaseFeatures:
    def test_one_clip_scenario_per_screen(self, feature_gen: FeatureGenerator):
        pack = _pack()
        features = feature_gen.build_showcase_features(
            pack["screens"], {"knowledge_app_name": "Common Practice"}
        )
        assert len(features) == 1
        content = features[0]["content"]
        assert features[0]["scenario_count"] == 2
        assert features[0]["tags"] == ["clip-1", "clip-2"]
        assert "@clip-1" in content and "@clip-2" in content
        assert content.count("Scenario:") == 2
        assert "Feature: Common Practice — showcase tour" in content

    def test_maps_capture_directives_to_fixed_steps(self, feature_gen: FeatureGenerator):
        pack = _pack()
        content = feature_gen.build_showcase_features(pack["screens"], {})[0]["content"]
        # Screen 1: open+wait then dwell (no click/scroll).
        assert 'When I open "/" and wait for ".page section.hero h1"' in content
        assert "Then I dwell for 6000 ms" in content
        # Screen 2: open+wait, scroll, dwell.
        assert 'When I open "/challenges" and wait for ".page .board"' in content
        assert 'And I scroll to ".board [data-rank=\\"1\\"]"' in content
        assert "Then I dwell for 5000 ms" in content

    def test_emits_click_step_when_present(self, feature_gen: FeatureGenerator):
        screens = [
            {
                "id": "run",
                "route": "/build",
                "label": "Run",
                "narration": "n",
                "capture": {"wait_for": ".shell", "click": "button.run", "hold_ms": 3000},
            }
        ]
        content = feature_gen.build_showcase_features(screens, {})[0]["content"]
        assert 'And I click "button.run"' in content

    def test_defaults_route_wait_and_hold(self, feature_gen: FeatureGenerator):
        screens = [{"id": "x", "label": "X", "narration": "n"}]
        content = feature_gen.build_showcase_features(screens, {})[0]["content"]
        assert 'When I open "/" and wait for "body"' in content
        assert "Then I dwell for 4000 ms" in content


class TestReadShowcaseSteps:
    def test_returns_fixed_step_file(self):
        steps = ConfigGenerator().read_showcase_steps()
        assert len(steps) == 1
        assert steps[0]["filename"] == "showcase.steps.ts"
        content = steps[0]["content"]
        # The feature's four step phrases must be implemented by the fixed file.
        assert "I open {string} and wait for {string}" in content
        assert "I click {string}" in content
        assert "I scroll to {string}" in content
        assert "I dwell for {int} ms" in content
        assert "recording_helpers" in content

    def test_open_resolves_route_against_baseurl(self):
        """`page.goto(route)` with a RELATIVE route resolves against the ORIGIN and
        strips any sub-path off baseURL (https://host/app -> https://host/...), so
        the showcase clip records the wrong (usually blank) page."""
        content = ConfigGenerator().read_showcase_steps()[0]["content"]
        assert "await page.goto(absoluteNavUrl(baseURL, route), {" in content
        # The step must take `baseURL` from the fixture arg to have it in scope.
        assert "async ({ page, baseURL }, route: string, waitFor: string)" in content
        # `new URL(route, baseURL)` has the SAME sub-path-stripping defect for a
        # leading-slash route, so the helper CONCATENATES instead.
        assert "return rel ? `${base}/${rel}` : base;" in content

    def test_content_wait_is_bounded_by_the_per_test_timeout(self):
        """The orchestrator's `timeout_per_test_ms` (default 20000) reaches Playwright
        as `--timeout`, which OVERRIDES this package's config. The old hardcoded
        `timeout: 30_000` was therefore unreachable dead code: Playwright killed the
        test mid-step, the hero tail never recorded and the clip came out black."""
        content = ConfigGenerator().read_showcase_steps()[0]["content"]
        assert "30_000" not in content
        assert "Number(process.env.PW_TEST_TIMEOUT_MS) || 20000" in content
        assert "timeout: CONTENT_WAIT_MS" in content
        assert "await page.waitForTimeout(SETTLE_MS)" in content
        # Pin the derivation so the mirror below cannot silently drift from the TS.
        assert "Math.min(1_500, Math.floor(PW_TEST_TIMEOUT_MS * 0.1))" in content
        assert "Math.max(250, Math.floor(PW_TEST_TIMEOUT_MS * 0.5) - SETTLE_MS)" in content

    def test_the_two_waits_together_never_reach_the_per_test_budget(self):
        """The INVARIANT, not the expression: a wait longer than the per-test timeout is
        unreachable dead code (Playwright kills the test mid-step and the clip records
        black). An earlier version's comment claimed a ceiling that its own 1000ms floor
        silently outranked — at PW=1000 `min(PW-4000, ...)` went NEGATIVE and the floor
        won. So assert the property across the whole budget range FTR accepts."""

        def settle(budget: int) -> int:
            return min(1_500, budget // 10)

        def content_wait(budget: int) -> int:
            return max(250, budget // 2 - settle(budget))

        # FTR's schema accepts timeout_per_test_ms as low as 1000; check well past both ends.
        for budget in (500, 625, 1_000, 3_000, 5_000, 8_000, 20_000, 60_000):
            total = content_wait(budget) + settle(budget)
            assert total < budget, (budget, content_wait(budget), settle(budget), total)
        # And at the real default it still leaves the goto + networkidle half the budget.
        assert content_wait(20_000) + settle(20_000) <= 10_000


class TestAuthConfigFromPack:
    def test_carries_sso_and_success_url_omits_password(self):
        cfg = _auth_config_from_pack(_pack()["access"]["auth"])
        assert cfg["login_url"] == "https://app.example.com/login"
        assert cfg["username"] == "demo@example.com"
        assert cfg["sso_button"] == "button.sso"
        assert cfg["success_url"] == "https://app.example.com/"
        # The password is env-injected at record time — never carried here.
        assert "password" not in cfg
        assert "password_env" not in cfg

    def test_omits_absent_optional_selectors(self):
        cfg = _auth_config_from_pack({"login_url": "u", "username": "n"})
        assert cfg == {"login_url": "u", "username": "n"}


class TestResolveKnowledgeInput:
    def test_stashes_screens_appname_and_auth(self):
        processor = JobProcessor()
        config: dict[str, Any] = {"knowledge": _pack()}
        inspection = processor._resolve_knowledge_input(config)
        assert config["knowledge_screens"] == _pack()["screens"]
        assert config["knowledge_app_name"] == "Common Practice"
        assert config["auth_config"]["sso_button"] == "button.sso"
        assert inspection["base_url"] == "https://app.example.com"
        assert [p["url"] for p in inspection["pages"]] == ["/", "/challenges"]

    def test_does_not_override_explicit_auth_config(self):
        processor = JobProcessor()
        config: dict[str, Any] = {"knowledge": _pack(), "auth_config": {"login_url": "explicit"}}
        processor._resolve_knowledge_input(config)
        assert config["auth_config"] == {"login_url": "explicit"}


class TestGenerateFeaturesRouting:
    async def test_demo_with_screens_uses_deterministic_emitter(self):
        processor = JobProcessor()
        config = {"knowledge_screens": _pack()["screens"], "knowledge_app_name": "CP"}
        features = await processor._generate_features("demo_video", {"pages": []}, config)
        # Deterministic path: no LLM call, exactly the emitted feature.
        assert features[0]["tags"] == ["clip-1", "clip-2"]


class TestVerbatimNarration:
    async def test_authored_narration_wins_over_template(self):
        from src.services.generation.feature_generator import _parse_features

        processor = JobProcessor()
        features = _parse_features(
            """```gherkin
@clip-1
Feature: Tour
  Scenario: Home dashboard
    When I open "/" and wait for ".hero"
    Then I dwell for 6000 ms
```"""
        )
        result = await processor._generate_narration(
            features,
            {
                "enabled": True,
                "tone": "tutorial",
                "llm_polish": False,
                "authored_narration": {"clip-1": "This is the pulse of the whole organisation."},
            },
        )
        parts = result["parts"]
        assert parts[0]["part_id"] == "clip-1"
        assert parts[0]["text"] == "This is the pulse of the whole organisation."

    async def test_authored_expected_text_rides_on_part(self):
        from src.services.generation.feature_generator import _parse_features

        processor = JobProcessor()
        features = _parse_features(
            """```gherkin
@clip-1
Feature: Tour
  Scenario: Home dashboard
    When I open "/" and wait for ".hero"
    Then I dwell for 6000 ms
```"""
        )
        result = await processor._generate_narration(
            features,
            {
                "enabled": True,
                "tone": "tutorial",
                "llm_polish": False,
                "authored_expected_text": {"clip-1": "Welcome back, Alex"},
            },
        )
        parts = result["parts"]
        assert parts[0]["part_id"] == "clip-1"
        # O1b: expected on-screen text is carried verbatim on the narration part,
        # keyed by the clip id, for the demo-video record stage's OCR assertion.
        assert parts[0]["expected_text"] == "Welcome back, Alex"

    async def test_part_omits_expected_text_when_screen_has_none(self):
        from src.services.generation.feature_generator import _parse_features

        processor = JobProcessor()
        features = _parse_features(
            """```gherkin
@clip-1
Feature: Tour
  Scenario: Home dashboard
    When I open "/" and wait for ".hero"
    Then I dwell for 6000 ms
```"""
        )
        result = await processor._generate_narration(
            features,
            {"enabled": True, "tone": "tutorial", "llm_polish": False},
        )
        assert "expected_text" not in result["parts"][0]
