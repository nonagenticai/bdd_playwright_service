"""Tests for Gherkin parser."""

from __future__ import annotations

import tempfile

import pytest

from src.services.input.gherkin_parser import GherkinParser


@pytest.fixture
def parser():
    """Create a GherkinParser instance."""
    return GherkinParser()


@pytest.fixture
def feature_with_tags():
    """Feature text with tags on feature and scenarios."""
    return """\
@regression @login
Feature: User Authentication
  Users should be able to log in securely

  @smoke @critical
  Scenario: Successful login
    Given I am on the login page
    When I enter valid credentials
    Then I should see the dashboard

  @negative
  Scenario: Failed login
    Given I am on the login page
    When I enter invalid credentials
    Then I should see an error message
"""


@pytest.fixture
def feature_with_background():
    """Feature text with a Background section."""
    return """\
Feature: Shopping Cart
  Manage items in the cart

  Background:
    Given I am logged in as a registered user
    And I have an empty cart

  Scenario: Add item to cart
    When I add "Widget" to the cart
    Then the cart should contain 1 item

  Scenario: Remove item from cart
    Given the cart contains "Widget"
    When I remove "Widget" from the cart
    Then the cart should be empty
"""


@pytest.fixture
def feature_with_scenario_outline():
    """Feature text with a Scenario Outline and Examples."""
    return """\
Feature: Login Validation
  Validate login with various inputs

  Scenario Outline: Login with different credentials
    Given I am on the login page
    When I enter "<username>" in the username field
    And I enter "<password>" in the password field
    And I click the "Sign In" button
    Then I should see "<result>"

  Examples:
    | username | password | result    |
    | admin    | admin123 | Dashboard |
    | user     | wrong    | Error     |
"""


@pytest.fixture
def feature_with_data_table():
    """Feature text with a data table in a step."""
    return """\
Feature: User Registration
  Register new users

  Scenario: Register with valid data
    Given the following users exist:
      | name  | email           |
      | Alice | alice@test.com  |
      | Bob   | bob@test.com    |
    When I register a new user
    Then the registration should succeed
"""


class TestParseFeatureText:
    def test_parse_complete_feature(self, parser, sample_gherkin_feature):
        """parse_feature_text should parse a complete feature with scenarios."""
        feature = parser.parse_feature_text(sample_gherkin_feature)
        assert feature.name == "User Login"
        assert len(feature.scenarios) == 2
        assert feature.scenarios[0].name == "Successful login"
        assert feature.scenarios[1].name == "Failed login with invalid credentials"

    def test_parse_feature_description(self, parser, sample_gherkin_feature):
        """parse_feature_text should capture the feature description."""
        feature = parser.parse_feature_text(sample_gherkin_feature)
        assert "authenticate" in feature.description.lower()

    def test_parse_scenario_steps(self, parser, sample_gherkin_feature):
        """parse_feature_text should capture all steps in a scenario."""
        feature = parser.parse_feature_text(sample_gherkin_feature)
        scenario = feature.scenarios[0]
        assert len(scenario.steps) == 6
        assert scenario.steps[0].keyword == "Given"
        assert scenario.steps[0].text == "I am on the login page"
        assert scenario.steps[1].keyword == "When"
        assert scenario.steps[2].keyword == "And"

    def test_parse_empty_text(self, parser):
        """parse_feature_text should return empty feature for empty input."""
        feature = parser.parse_feature_text("")
        assert feature.name == ""
        assert feature.scenarios == []

    def test_parse_whitespace_only(self, parser):
        """parse_feature_text should return empty feature for whitespace."""
        feature = parser.parse_feature_text("   \n\n  \n")
        assert feature.name == ""
        assert feature.scenarios == []

    def test_parse_malformed_no_feature_keyword(self, parser):
        """parse_feature_text should handle text without a Feature keyword."""
        text = "This is just some random text\nwithout any Gherkin structure."
        feature = parser.parse_feature_text(text)
        assert feature.name == ""
        assert feature.scenarios == []


class TestParseTags:
    def test_feature_tags(self, parser, feature_with_tags):
        """parse_feature_text should capture feature-level tags."""
        feature = parser.parse_feature_text(feature_with_tags)
        assert "@regression" in feature.tags
        assert "@login" in feature.tags

    def test_scenario_tags(self, parser, feature_with_tags):
        """parse_feature_text should capture scenario-level tags."""
        feature = parser.parse_feature_text(feature_with_tags)
        assert "@smoke" in feature.scenarios[0].tags
        assert "@critical" in feature.scenarios[0].tags
        assert "@negative" in feature.scenarios[1].tags


class TestParseBackground:
    def test_background_steps_prepended(self, parser, feature_with_background):
        """Background steps should be prepended to each scenario."""
        feature = parser.parse_feature_text(feature_with_background)
        assert len(feature.scenarios) == 2

        # First scenario should have background steps + its own steps
        first = feature.scenarios[0]
        assert first.steps[0].keyword == "Given"
        assert "logged in" in first.steps[0].text
        assert first.steps[1].keyword == "And"
        assert "empty cart" in first.steps[1].text

        # Second scenario also gets background steps
        second = feature.scenarios[1]
        assert second.steps[0].keyword == "Given"
        assert "logged in" in second.steps[0].text


class TestParseScenarioOutline:
    def test_scenario_outline_parsed(self, parser, feature_with_scenario_outline):
        """parse_feature_text should parse Scenario Outline as a scenario."""
        feature = parser.parse_feature_text(feature_with_scenario_outline)
        assert len(feature.scenarios) == 1
        assert feature.scenarios[0].name == "Login with different credentials"

    def test_scenario_outline_steps(self, parser, feature_with_scenario_outline):
        """Scenario Outline should have its steps parsed."""
        feature = parser.parse_feature_text(feature_with_scenario_outline)
        steps = feature.scenarios[0].steps
        assert len(steps) >= 4
        assert steps[0].keyword == "Given"


class TestParseDataTable:
    def test_data_table_captured(self, parser, feature_with_data_table):
        """Steps with data tables should have data_table populated."""
        feature = parser.parse_feature_text(feature_with_data_table)
        first_step = feature.scenarios[0].steps[0]
        assert first_step.data_table is not None
        assert len(first_step.data_table) == 3  # header + 2 rows
        assert "name" in first_step.data_table[0]
        assert "Alice" in first_step.data_table[1]


class TestExtractSteps:
    def test_extract_all_steps(self, parser, sample_gherkin_feature):
        """extract_steps should return a flat list of all steps."""
        steps = parser.extract_steps(sample_gherkin_feature)
        assert len(steps) > 0
        keywords = {s.keyword for s in steps}
        assert "Given" in keywords
        assert "When" in keywords
        assert "Then" in keywords

    def test_extract_steps_from_fragment(self, parser):
        """extract_steps should work on a text fragment (not a full feature)."""
        text = """\
    Given I am on the homepage
    When I click "Login"
    Then I should see the login form
"""
        steps = parser.extract_steps(text)
        assert len(steps) == 3
        assert steps[0].keyword == "Given"
        assert steps[1].keyword == "When"
        assert steps[2].keyword == "Then"

    def test_extract_steps_empty_text(self, parser):
        """extract_steps should return empty list for empty text."""
        steps = parser.extract_steps("")
        assert steps == []

    def test_extract_steps_preserves_and_but(self, parser):
        """extract_steps should preserve And/But keywords."""
        text = """\
    Given I am logged in
    And I am on the dashboard
    But I do not have admin rights
"""
        steps = parser.extract_steps(text)
        assert len(steps) == 3
        assert steps[1].keyword == "And"
        assert steps[2].keyword == "But"


class TestParseFeatureFile:
    def test_parse_feature_file(self, parser, sample_gherkin_feature):
        """parse_feature_file should read and parse a .feature file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".feature", delete=False, encoding="utf-8"
        ) as f:
            f.write(sample_gherkin_feature)
            f.flush()
            feature = parser.parse_feature_file(f.name)

        assert feature.name == "User Login"
        assert len(feature.scenarios) == 2

    def test_parse_feature_file_not_found(self, parser):
        """parse_feature_file should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            parser.parse_feature_file("/nonexistent/path/feature.feature")
