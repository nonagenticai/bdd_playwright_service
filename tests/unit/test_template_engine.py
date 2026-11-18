"""Tests for narration template engine pattern matching."""

from __future__ import annotations

from src.services.narration.template_engine import NarrationTemplateEngine


class TestNarrationTemplateEngine:
    """Tests for NarrationTemplateEngine step humanization."""

    def setup_method(self):
        self.engine = NarrationTemplateEngine(tone="tutorial")

    def test_given_page_pattern(self):
        """Given 'I am on the X page' should match the tutorial pattern."""
        result = self.engine.humanize_step("Given", "I am on the login page")
        assert "login" in result
        assert "navigating" in result.lower() or "start" in result.lower()

    def test_when_click_button_pattern(self):
        """When 'I click the X button' should match the tutorial pattern."""
        result = self.engine.humanize_step("When", "I click the Submit button")
        assert "Submit" in result
        assert "click" in result.lower()

    def test_when_fill_pattern(self):
        """When 'I fill X with Y' should match the tutorial pattern."""
        result = self.engine.humanize_step("When", "I fill username with admin")
        assert "admin" in result
        assert "username" in result

    def test_when_enter_field_pattern(self):
        """When 'I enter X in the Y field' should match the tutorial pattern."""
        result = self.engine.humanize_step("When", 'I enter "test" in the email field')
        assert "email" in result
        assert '"test"' in result

    def test_then_should_see_pattern(self):
        """Then 'I should see X' should match the tutorial pattern."""
        result = self.engine.humanize_step("Then", 'I should see "Welcome"')
        assert '"Welcome"' in result

    def test_then_redirected_pattern(self):
        """Then 'I should be redirected to the X' should match."""
        result = self.engine.humanize_step("Then", "I should be redirected to the dashboard")
        assert "dashboard" in result

    def test_then_error_message_pattern(self):
        """Then 'I should see an error message' should match."""
        result = self.engine.humanize_step("Then", "I should see an error message")
        assert "error" in result.lower()

    def test_fallback_for_unknown_step(self):
        """Unknown steps should use the fallback template."""
        result = self.engine.humanize_step("When", "I do something completely custom")
        assert "something completely custom" in result

    def test_demo_tone(self):
        """Demo tone should produce different text than tutorial."""
        demo_engine = NarrationTemplateEngine(tone="demo")
        result = demo_engine.humanize_step("Given", "I am on the settings page")
        assert "settings" in result
        # Demo uses "take a look" or similar phrasing
        assert result != self.engine.humanize_step("Given", "I am on the settings page")

    def test_marketing_tone(self):
        """Marketing tone should produce benefit-focused text."""
        marketing_engine = NarrationTemplateEngine(tone="marketing")
        result = marketing_engine.humanize_step("Given", "I am on the dashboard page")
        assert "dashboard" in result
        assert "power" in result.lower() or "discover" in result.lower()

    def test_technical_tone(self):
        """Technical tone should produce precise text."""
        tech_engine = NarrationTemplateEngine(tone="technical")
        result = tech_engine.humanize_step("Given", "I am on the login page")
        assert "login" in result
        assert "endpoint" in result.lower() or "navigate" in result.lower()

    def test_humanize_scenario(self):
        """humanize_scenario should combine multiple steps into text."""
        steps = [
            {"keyword": "Given", "text": "I am on the login page"},
            {"keyword": "When", "text": "I click the Submit button"},
            {"keyword": "Then", "text": "I should see an error message"},
        ]
        result = self.engine.humanize_scenario("Login test", steps)
        assert "login" in result
        assert "Submit" in result
        assert "error" in result.lower()

    def test_humanize_scenario_with_and_keyword(self):
        """And keyword should inherit the previous primary keyword."""
        steps = [
            {"keyword": "Given", "text": "I am on the login page"},
            {"keyword": "When", "text": "I click the Submit button"},
            {"keyword": "And", "text": "I submit the form"},
            {"keyword": "Then", "text": "I should see an error message"},
        ]
        result = self.engine.humanize_scenario("Login with And", steps)
        assert "login" in result
        assert "Submit" in result
        assert "error" in result.lower()

    def test_humanize_scenario_empty_steps(self):
        """humanize_scenario with empty steps should return empty string."""
        result = self.engine.humanize_scenario("Empty", [])
        assert result == ""

    def test_given_logged_in_pattern(self):
        """Given 'I am logged in as X' should match."""
        result = self.engine.humanize_step("Given", "I am logged in as admin")
        assert "admin" in result
        assert "log" in result.lower()

    def test_when_navigate_pattern(self):
        """When 'I navigate to X' should match."""
        result = self.engine.humanize_step("When", "I navigate to settings")
        assert "settings" in result
