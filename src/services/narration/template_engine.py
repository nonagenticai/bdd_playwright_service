"""Pattern-based narration engine for converting Gherkin steps to natural language."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from src.services.gherkin_keywords import PRIMARY_KEYWORDS, resolve_binding

logger = logging.getLogger(__name__)

_PATTERNS_DIR = Path(__file__).resolve().parent.parent.parent / "templates" / "narration_patterns"

#: A scenario-framing lead-in is only worth adding when the body is at least this
#: many words short of its budget; below it the template is already close enough
#: and the polish pass can cover the rest.
_LEAD_IN_MIN_GAP_WORDS = 6

#: Per-tone framing sentence. Real content — it names the scenario the viewer is
#: about to watch — not filler.
_LEAD_INS = {
    "tutorial": "In this step, let's walk through {scenario}.",
    "demo": "Here's {scenario} in action.",
    "marketing": "Take a look at how easy {scenario} is.",
    "technical": "This section covers {scenario}.",
}

# Keyword-to-pattern-file mapping
_KEYWORD_FILES = {
    "Given": "given_patterns.yaml",
    "When": "when_patterns.yaml",
    "Then": "then_patterns.yaml",
}


def _load_patterns() -> dict[str, dict[str, Any]]:
    """Load all narration patterns from YAML files."""
    all_patterns: dict[str, dict[str, Any]] = {}
    for keyword, filename in _KEYWORD_FILES.items():
        path = _PATTERNS_DIR / filename
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            all_patterns[keyword] = data or {}
        else:
            logger.warning("Pattern file not found: %s", path)
            all_patterns[keyword] = {"patterns": [], "fallback": {}}
    return all_patterns


def _match_pattern(step_text: str, patterns: list[dict[str, Any]], tone: str) -> str | None:
    """Try to match step text against a list of patterns.

    Pattern placeholders like {page_name} are converted to regex capture groups
    and substituted into the tone-specific template.
    """
    for pattern in patterns:
        match_template = pattern.get("match", "")
        tone_template = pattern.get(tone)
        if not tone_template:
            continue

        # Convert {placeholder} to named regex groups
        regex_str = match_template
        placeholders = re.findall(r"\{(\w+)\}", match_template)
        for ph in placeholders:
            regex_str = regex_str.replace(f"{{{ph}}}", f"(?P<{ph}>.+?)")

        match = re.fullmatch(regex_str, step_text, re.IGNORECASE)
        if match:
            result = tone_template
            for ph in placeholders:
                result = result.replace(f"{{{ph}}}", match.group(ph))
            return result

    return None


class NarrationTemplateEngine:
    """Converts Gherkin steps to natural language using pattern matching."""

    def __init__(self, tone: str = "tutorial") -> None:
        self.tone = tone
        self._patterns = _load_patterns()
        #: Running count of steps narrated by the fixed per-tone FALLBACK template
        #: (no pattern matched). Callers diff it around a call to learn whether a
        #: line contains fallback text — observability only, never read for output.
        self.fallback_steps = 0

    def humanize_step(self, keyword: str, text: str) -> str:
        """Convert a single Gherkin step to natural language.

        Args:
            keyword: The step keyword (Given, When, Then, And, But).
            text: The step text without the keyword.

        Returns:
            A natural-language sentence.
        """
        # Map And/But/* to the primary keyword category they follow. With no caller
        # context there is nothing to inherit from, so this default is "When" (a bare
        # narrated step reads as an action); `humanize_scenario` passes the real
        # preceding keyword.
        effective_keyword = resolve_binding(keyword, default="When")

        pattern_data = self._patterns.get(effective_keyword, {})
        patterns = pattern_data.get("patterns", [])
        fallback = pattern_data.get("fallback", {})

        result = _match_pattern(text, patterns, self.tone)
        if result:
            return result

        # Use fallback template. OBSERVABLE: a fallback line is canned
        # ("Next, {step_text}.") rather than real narration, and nothing recorded
        # that it happened. Length only, never the step text itself.
        self.fallback_steps += 1
        logger.warning(
            "narration_template_fallback tone=%s keyword=%s step_text_len=%d",
            self.tone,
            effective_keyword,
            len(text),
            extra={
                "narration_source": "template_fallback",
                "tone": self.tone,
                "step_keyword": effective_keyword,
                "step_text_len": len(text),
            },
        )
        fb_template = fallback.get(self.tone, "Next, {step_text}.")
        return fb_template.replace("{step_text}", text)

    def humanize_scenario(
        self,
        scenario_name: str,
        steps: list[dict[str, Any]],
        word_budget: int | None = None,
    ) -> str:
        """Convert a full Gherkin scenario to narrated text.

        Args:
            scenario_name: The scenario title.
            steps: List of dicts with 'keyword' and 'text' keys.
            word_budget: The words this scenario's narration is spoken over, when
                the caller knows it. The template is otherwise length-INDEPENDENT
                of the budget — it emits exactly one sentence per Gherkin step, so
                a four-step scenario produced ~14 words against a 24-word budget
                and the LLM polish pass was then asked to nearly double the text
                in a single shot, which is the shape that fails. A budget-aware
                template closes part of that gap deterministically, with real
                content (the scenario's own title as a framing line) rather than
                filler, and never writes past the budget. ``None`` keeps today's
                output byte for byte.

        Returns:
            Multi-sentence narration for the scenario.
        """
        parts: list[str] = []

        # Track last primary keyword for And/But/* resolution (same rule the step-def
        # generator binds by — `src.services.gherkin_keywords` is the single source).
        last_primary: str | None = None

        for step in steps:
            keyword = step.get("keyword", "When")
            text = step.get("text", "")

            effective_keyword = resolve_binding(keyword, last_primary, default="When")
            if keyword in PRIMARY_KEYWORDS:
                last_primary = keyword

            narrated = self.humanize_step(effective_keyword, text)
            parts.append(narrated)

        body = " ".join(parts)
        return self._lead_in_to_budget(scenario_name, body, word_budget)

    def _lead_in_to_budget(self, scenario_name: str, body: str, word_budget: int | None) -> str:
        """Prefix a scenario-framing line when the body under-runs its budget.

        Deterministic and content-bearing: the sentence names what the viewer is
        about to watch, which is true of every scenario by construction. Applied
        ONLY when it fits inside the budget, so this can never push a section
        over the ceiling the trim would then cut back.
        """
        if not word_budget or not scenario_name.strip():
            return body
        current = len(body.split())
        if word_budget - current < _LEAD_IN_MIN_GAP_WORDS:
            return body
        lead_in = _LEAD_INS.get(self.tone, _LEAD_INS["tutorial"]).format(
            scenario=scenario_name.strip().rstrip(".")
        )
        if current + len(lead_in.split()) > word_budget:
            return body
        return f"{lead_in} {body}".strip()
