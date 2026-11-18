"""Tone configuration presets for narration generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToneConfig:
    """Configuration for a narration tone/style."""

    persona: str
    sentence_style: str
    vocabulary_level: str
    transition_style: str
    llm_system_prompt: str


TONE_PRESETS: dict[str, ToneConfig] = {
    "tutorial": ToneConfig(
        persona="friendly instructor",
        sentence_style="clear, step-by-step with encouragement",
        vocabulary_level="beginner-friendly",
        transition_style="sequential with connecting phrases",
        llm_system_prompt=(
            "You are a friendly instructor guiding a beginner through a web application. "
            "Use clear, encouraging language. Explain each step as if the reader is "
            "following along for the first time."
        ),
    ),
    "demo": ToneConfig(
        persona="product specialist",
        sentence_style="concise, engaging, and confident",
        vocabulary_level="general audience",
        transition_style="smooth and flowing",
        llm_system_prompt=(
            "You are a product specialist giving a live demo. Keep the narration "
            "concise and engaging. Highlight key features and maintain a confident tone."
        ),
    ),
    "marketing": ToneConfig(
        persona="enthusiastic marketer",
        sentence_style="persuasive, benefit-focused, and energetic",
        vocabulary_level="general audience with emphasis on value",
        transition_style="benefit-driven with excitement",
        llm_system_prompt=(
            "You are an enthusiastic marketer showcasing a product's capabilities. "
            "Focus on benefits, ease of use, and the value delivered. "
            "Use persuasive and energetic language."
        ),
    ),
    "technical": ToneConfig(
        persona="technical documentation writer",
        sentence_style="precise, factual, and structured",
        vocabulary_level="technical audience",
        transition_style="procedural and logical",
        llm_system_prompt=(
            "You are a technical writer creating precise documentation. "
            "Use factual, structured language. Focus on accuracy and completeness."
        ),
    ),
}


def get_tone(name: str) -> ToneConfig:
    """Retrieve a tone preset by name.

    Args:
        name: Tone preset name (tutorial, demo, marketing, technical).

    Returns:
        The corresponding ToneConfig.

    Raises:
        ValueError: If the tone name is not recognized.
    """
    if name not in TONE_PRESETS:
        available = ", ".join(sorted(TONE_PRESETS.keys()))
        raise ValueError(f"Unknown tone '{name}'. Available tones: {available}")
    return TONE_PRESETS[name]
