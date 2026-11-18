"""Tests for tone presets configuration."""

from __future__ import annotations

import pytest

from src.services.narration.tone_presets import TONE_PRESETS, ToneConfig, get_tone


class TestTonePresets:
    """Tests for tone preset definitions and retrieval."""

    def test_all_tones_exist(self):
        """All four tone presets should be defined."""
        expected_tones = {"tutorial", "demo", "marketing", "technical"}
        assert set(TONE_PRESETS.keys()) == expected_tones

    def test_tone_config_is_dataclass(self):
        """Each preset should be a ToneConfig dataclass."""
        for name, config in TONE_PRESETS.items():
            assert isinstance(config, ToneConfig), f"{name} is not a ToneConfig"

    def test_tutorial_has_required_fields(self):
        """Tutorial tone should have all required fields populated."""
        tone = TONE_PRESETS["tutorial"]
        assert tone.persona
        assert tone.sentence_style
        assert tone.vocabulary_level
        assert tone.transition_style
        assert tone.llm_system_prompt

    def test_demo_has_required_fields(self):
        """Demo tone should have all required fields populated."""
        tone = TONE_PRESETS["demo"]
        assert tone.persona
        assert tone.sentence_style
        assert tone.vocabulary_level
        assert tone.transition_style
        assert tone.llm_system_prompt

    def test_marketing_has_required_fields(self):
        """Marketing tone should have all required fields populated."""
        tone = TONE_PRESETS["marketing"]
        assert tone.persona
        assert tone.sentence_style
        assert tone.vocabulary_level
        assert tone.transition_style
        assert tone.llm_system_prompt

    def test_technical_has_required_fields(self):
        """Technical tone should have all required fields populated."""
        tone = TONE_PRESETS["technical"]
        assert tone.persona
        assert tone.sentence_style
        assert tone.vocabulary_level
        assert tone.transition_style
        assert tone.llm_system_prompt

    def test_get_tone_tutorial(self):
        """get_tone should return the tutorial preset."""
        tone = get_tone("tutorial")
        assert tone is TONE_PRESETS["tutorial"]

    def test_get_tone_demo(self):
        """get_tone should return the demo preset."""
        tone = get_tone("demo")
        assert tone is TONE_PRESETS["demo"]

    def test_get_tone_marketing(self):
        """get_tone should return the marketing preset."""
        tone = get_tone("marketing")
        assert tone is TONE_PRESETS["marketing"]

    def test_get_tone_technical(self):
        """get_tone should return the technical preset."""
        tone = get_tone("technical")
        assert tone is TONE_PRESETS["technical"]

    def test_get_tone_unknown_raises(self):
        """get_tone should raise ValueError for unknown tones."""
        with pytest.raises(ValueError, match="Unknown tone"):
            get_tone("nonexistent")

    def test_tone_config_is_frozen(self):
        """ToneConfig should be immutable (frozen dataclass)."""
        tone = TONE_PRESETS["tutorial"]
        with pytest.raises(AttributeError):
            tone.persona = "changed"  # type: ignore[misc]  # ty: ignore[invalid-assignment]

    def test_all_tones_have_distinct_personas(self):
        """Each tone should have a unique persona."""
        personas = [config.persona for config in TONE_PRESETS.values()]
        assert len(personas) == len(set(personas))

    def test_all_tones_have_llm_system_prompts(self):
        """Each tone's llm_system_prompt should be a non-empty string."""
        for name, config in TONE_PRESETS.items():
            assert isinstance(config.llm_system_prompt, str), f"{name} prompt is not a string"
            assert len(config.llm_system_prompt) > 20, f"{name} prompt is too short"
