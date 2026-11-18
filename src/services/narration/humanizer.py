"""LLM-based polish pass for narration text."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.services.generation.llm_client import LLMClient
from src.services.narration.tone_presets import ToneConfig, get_tone
from src.services.narration.types import NarrationPassReport, NarrationSection

logger = logging.getLogger(__name__)

#: How many times one pass is asked for. The response is accepted only on a
#: faithful echo (see ``_parse_polished``); before this constant existed a single
#: malformed response silently threw the whole polished script away and kept the
#: SHORT template text, with no log line at all. One retry is cheap next to a
#: narration that under-runs its slot and gets time-stretched downstream.
MAX_POLISH_ATTEMPTS = 2

#: A section is treated as SHORT — i.e. worth re-asking for — when it delivers
#: less than this share of its word budget. 0.85 leaves normal prose variation
#: alone; the live defect was ~58% of budget on every beat.
SHORTFALL_TRIGGER_RATIO = 0.85

#: Below this many missing words an expansion pass cannot buy anything a
#: listener would notice, so it is not worth an LLM round trip.
MIN_SHORTFALL_WORDS = 3

#: The response shape asked of WM. Passing it means the model's output is
#: machine-validated at the boundary instead of being recovered by an
#: echo-matching regex over free text; the ``[Name]`` echo parse stays as the
#: fallback for a WM that returns plain text.
POLISH_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scenario_name": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["scenario_name", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["sections"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class PolishReport:
    """What one LLM narration pass actually did — structured, not prose.

    ``parse_failures`` is the number that had NO observable existence before:
    a discarded response is indistinguishable from "the model returned the
    template text" in the artifact, and both routes end in a short script.
    """

    mode: str
    attempts: int = 0
    parse_failures: int = 0
    applied: bool = False

    def as_dict(self) -> NarrationPassReport:
        """Serialize for the job result payload."""
        return {
            "mode": self.mode,
            "attempts": self.attempts,
            "parse_failures": self.parse_failures,
            "applied": self.applied,
        }


class NarrationHumanizer:
    """Smooths template-generated narration using an LLM polish pass."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def polish(
        self,
        narration_parts: list[NarrationSection],
        tone: str,
        context: str = "",
        word_budgets: list[int] | None = None,
    ) -> list[NarrationSection]:
        """Polish narration parts with an LLM to smooth transitions.

        Args:
            narration_parts: List of dicts with 'scenario_name' and 'text' keys.
            tone: Tone preset name to use for the polish pass.
            context: Optional extra context about the application.
            word_budgets: The orchestrator's per-section word budget, in the same
                order as ``narration_parts``. When present it is stated in the
                prompt as a length to WRITE TO, not a ceiling to stay under — the
                deterministic trim downstream can only SHORTEN a line, so a script
                that under-runs its order can be fixed here and nowhere else.
                ``None`` (no such budget was sent) leaves the prompt byte for byte
                as it is today.

        Returns:
            Updated list of dicts with polished 'text' values.
        """
        parts, _ = await self.polish_with_report(
            narration_parts, tone, context=context, word_budgets=word_budgets
        )
        return parts

    async def polish_with_report(
        self,
        narration_parts: list[NarrationSection],
        tone: str,
        context: str = "",
        word_budgets: list[int] | None = None,
    ) -> tuple[list[NarrationSection], PolishReport]:
        """``polish`` plus the structured record of what the pass did."""
        return await self._run_pass(narration_parts, tone, context, word_budgets, mode="polish")

    async def expand_to_budget(
        self,
        narration_parts: list[NarrationSection],
        tone: str,
        word_budgets: list[int],
        context: str = "",
    ) -> tuple[list[NarrationSection], PolishReport]:
        """Re-ask for sections that came back UNDER budget.

        The deterministic ``trim_to_word_budget`` downstream is a ceiling and
        only a ceiling: nothing in the pipeline could ever LENGTHEN a section
        that under-ran its slot, so the orchestrator filled the remaining time
        by stretching the speech (its 0.80 rate floor) — which is the
        "narration weirdly slows down" the owner hears. This is the missing
        symmetric half, and it leads with the length requirement rather than
        appending it to a polish instruction.
        """
        return await self._run_pass(narration_parts, tone, context, word_budgets, mode="expand")

    async def _run_pass(
        self,
        narration_parts: list[NarrationSection],
        tone: str,
        context: str,
        word_budgets: list[int] | None,
        mode: str,
    ) -> tuple[list[NarrationSection], PolishReport]:
        """Run one LLM pass, retrying once when the response is unusable."""
        tone_config = get_tone(tone)
        prompt = self._build_prompt(narration_parts, tone, tone_config, context, word_budgets, mode)

        attempts = 0
        failures = 0
        for attempt in range(1, MAX_POLISH_ATTEMPTS + 1):
            attempts = attempt
            raw = await self._llm.generate(
                prompt,
                system_prompt=tone_config.llm_system_prompt,
                output_schema=POLISH_OUTPUT_SCHEMA,
            )
            blocks = self._polished_blocks(raw, narration_parts)
            if blocks is not None:
                polished: list[NarrationSection] = [
                    {**part, "text": body}
                    for part, (_, body) in zip(narration_parts, blocks, strict=True)
                ]
                logger.info("Polished %d narration parts (mode=%s)", len(polished), mode)
                return polished, PolishReport(
                    mode=mode, attempts=attempts, parse_failures=failures, applied=True
                )

            failures += 1
            self._log_parse_failure(raw, narration_parts, mode, attempt)

        logger.warning(
            "Narration %s pass produced NO usable response in %d attempt(s); keeping the "
            "template narration for all %d section(s)",
            mode,
            attempts,
            len(narration_parts),
        )
        return [{**part} for part in narration_parts], PolishReport(
            mode=mode, attempts=attempts, parse_failures=failures, applied=False
        )

    @staticmethod
    def _log_parse_failure(
        raw: str, narration_parts: list[NarrationSection], mode: str, attempt: int
    ) -> None:
        """Make a discarded response VISIBLE.

        Until this existed, an unfaithful response was thrown away in silence:
        the job kept the short template text and nothing anywhere recorded that
        an LLM answer had been rejected, so nobody could measure how often this
        second, independent route to a short script fires.
        """
        names = [p.get("scenario_name") for p in narration_parts]
        known: set[str] = {n for n in names if n is not None}
        parsed = [name for name, _ in NarrationHumanizer._parse_blocks(raw, known)]
        logger.warning(
            "Narration %s response DISCARDED (attempt %d/%d): expected %d section(s) %r, "
            "parsed %d %r; response was %d chars starting %r. Keeping template narration "
            "for this attempt.",
            mode,
            attempt,
            MAX_POLISH_ATTEMPTS,
            len(names),
            names,
            len(parsed),
            parsed,
            len(raw),
            raw[:200],
        )

    @staticmethod
    def _build_prompt(
        narration_parts: list[NarrationSection],
        tone: str,
        tone_config: ToneConfig,
        context: str,
        word_budgets: list[int] | None,
        mode: str,
    ) -> str:
        """Build the pass prompt, leading with LENGTH when length is the defect.

        The original prompt opened with "Polish … Preserve the meaning and
        structure" and appended the budget afterwards, so a model asked to
        nearly double a short section read "preserve the structure" first and
        returned ~58% of the budget on every beat. When any section is short the
        length requirement now LEADS and polish is the secondary instruction.
        """
        parts_text = "\n\n".join(
            f"[{part.get('scenario_name', 'Untitled')}]\n{part.get('text', '')}"
            for part in narration_parts
        )
        context_section = f"\nApplication context: {context}\n" if context else ""

        short = NarrationHumanizer._short_sections(narration_parts, word_budgets)
        expansion_led = mode == "expand" or bool(short)

        # THE LENGTH THE SCRIPT IS WRITTEN TO. Omitted entirely when the caller sent
        # no budget, so the no-budget prompt is unchanged.
        budget_section = ""
        if word_budgets:
            per_section = "\n".join(
                f"- [{part.get('scenario_name', 'Untitled')}]: {words} words"
                f"{NarrationHumanizer._delta_note(part, words)}"
                for part, words in zip(narration_parts, word_budgets, strict=False)
            )
            budget_section = (
                "\nLENGTH BUDGET — this narration is spoken over a film of a fixed "
                f"ordered length ({sum(word_budgets)} words in total). Write each "
                "section TO its budget below: expand a section that falls short with "
                "genuine detail about what is on screen, and tighten one that runs "
                "over. Do not pad with filler and do not merge or drop sections.\n"
                f"{per_section}\n"
            )

        if expansion_led:
            lead = (
                "REWRITE each narration section below so that it is AS LONG AS ITS WORD "
                "BUDGET. Length is the PRIMARY requirement of this task: a section that "
                "comes back short is spoken over a fixed-length shot and has to be "
                "slowed down to fill it, which sounds wrong. Every section must land "
                "within 10% of its budgeted word count — most of the sections below are "
                "currently well UNDER budget and must GROW substantially.\n\n"
                "Grow a short section with genuine, concrete detail about what is on "
                "screen and why it matters to the viewer — describe the interface, the "
                "action and its outcome. Do not repeat yourself, do not pad with filler "
                "words, and do not merge, drop, reorder or rename sections.\n\n"
                "Secondarily, once the length is right, polish for smooth natural "
                "transitions, remove awkward phrasing and keep the sections consistent "
                "with each other.\n\n"
            )
        else:
            lead = (
                "Polish the following narration script for smooth, natural transitions "
                "between sections. Preserve the meaning and structure but improve flow, "
                "remove awkward phrasing, and ensure consistency.\n\n"
            )

        return (
            f"{lead}"
            f"Tone: {tone} ({tone_config.persona})\n"
            f"Style: {tone_config.sentence_style}\n"
            f"{context_section}"
            f"{budget_section}\n"
            "Narration script:\n"
            f"{parts_text}\n\n"
            "Return the polished narration in the same format: "
            "[Scenario Name] followed by the polished text, "
            "separated by blank lines."
        )

    @staticmethod
    def _delta_note(part: NarrationSection, budget: int) -> str:
        """A per-section ``(currently N words — add M)`` note for the prompt."""
        current = len(str(part.get("text", "")).split())
        if budget <= 0:
            return ""
        missing = budget - current
        if missing >= MIN_SHORTFALL_WORDS:
            return f" (currently {current} words — ADD about {missing} more)"
        if missing <= -MIN_SHORTFALL_WORDS:
            return f" (currently {current} words — cut about {-missing})"
        return f" (currently {current} words — length is fine)"

    @staticmethod
    def _short_sections(
        narration_parts: list[NarrationSection], word_budgets: list[int] | None
    ) -> list[int]:
        """Indices of sections delivering less than their budget share."""
        if not word_budgets:
            return []
        short: list[int] = []
        for index, (part, budget) in enumerate(zip(narration_parts, word_budgets, strict=False)):
            words = len(str(part.get("text", "")).split())
            if (
                budget > 0
                and words < budget * SHORTFALL_TRIGGER_RATIO
                and budget - words >= MIN_SHORTFALL_WORDS
            ):
                short.append(index)
        return short

    @staticmethod
    def _parse_polished(raw: str, original_parts: list[NarrationSection]) -> list[NarrationSection]:
        """Parse the LLM polish response back into part dicts, or keep originals.

        Polishing is a cosmetic, optional pass; narrating a clip with the WRONG
        scenario's text is a serious pipeline defect (the ``@clip-<n>`` id is the
        load-bearing join spine). So the polished text is accepted ONLY when the
        response is a faithful, in-order echo of the requested format:

        * the parsed ``[Scenario Name]`` headers, in document order, exactly
          equal this run's ``scenario_name`` list (same names, same order, same
          count), and
        * every parsed body is non-empty.

        On that all-or-nothing match, block *i* is provably scenario *i*'s
        section, so each part is replaced by its own block. Any deviation —
        reordered, dropped, merged, duplicated, an unbracketed or glued header, a
        bracketed reference swallowed into a body — fails the exact match, and
        EVERY part keeps its original (correct, deterministic) narration. This
        makes silent cross-scenario contamination impossible rather than relying
        on per-scenario heuristics that an imperfect model can defeat.
        """
        blocks = NarrationHumanizer._polished_blocks(raw, original_parts)
        if blocks is None:
            return [{**part} for part in original_parts]
        return [
            {**part, "text": body} for part, (_, body) in zip(original_parts, blocks, strict=True)
        ]

    @staticmethod
    def _polished_blocks(
        raw: str, original_parts: list[NarrationSection]
    ) -> list[tuple[str, str]] | None:
        """The faithful ``(name, body)`` blocks, or ``None`` when unusable.

        Two shapes are accepted, in this order: the SCHEMA-validated JSON object
        asked for via ``POLISH_OUTPUT_SCHEMA`` (``{"sections": [...]}``), and the
        legacy ``[Name]`` free-text echo. ``None`` is the signal the caller needs
        in order to retry and to COUNT the failure — the old code folded that
        signal into "here are your originals back", which is why a discarded
        response was indistinguishable from a no-op polish.
        """
        names = [p.get("scenario_name") for p in original_parts]
        known: set[str] = {n for n in names if n is not None}
        blocks = NarrationHumanizer._parse_structured(raw, known)
        if blocks is None:
            blocks = NarrationHumanizer._parse_blocks(raw, known)

        faithful = (
            [name for name, _ in blocks] == names
            and all(name is not None for name in names)
            and all(body for _, body in blocks)
        )
        return blocks if faithful else None

    @staticmethod
    def _parse_structured(raw: str, known: set[str]) -> list[tuple[str, str]] | None:
        """Parse a schema-shaped ``{"sections": [...]}`` response, or ``None``.

        ``None`` means "this is not the structured shape" — NOT "unfaithful" —
        so the caller falls through to the text echo parse. The name filter is
        the same one the text parse applies, so a structured response cannot
        smuggle in a scenario this run never asked about.
        """
        text = raw.strip()
        if not text.startswith("{"):
            return None
        try:
            payload = json.loads(text)
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        sections = payload.get("sections")
        if not isinstance(sections, list):
            return None
        blocks: list[tuple[str, str]] = []
        for section in sections:
            if not isinstance(section, dict):
                return None
            name = section.get("scenario_name")
            body = section.get("text")
            if not isinstance(name, str) or not isinstance(body, str) or name not in known:
                return None
            blocks.append((name, " ".join(body.split())))
        return blocks

    @staticmethod
    def _parse_blocks(raw: str, known: set[str]) -> list[tuple[str, str]]:
        """Parse ``raw`` into an ordered list of ``(scenario_name, body)`` blocks.

        A header is a line whose stripped form is ``[Name]`` where ``Name`` is a
        KNOWN scenario name AND the line begins a block (it is the first content
        line, or is preceded by a blank line — the requested blank-line-separated
        format). Its body is every following line up to the next header, with
        internal paragraphs joined by single spaces. Lines before the first
        header (a preamble) are ignored. Restricting headers to known names that
        begin a block means a bracketed reference inside a body, or a header the
        model glued onto the prior line, is NOT promoted to a header — it stays
        in the current body, so the block list no longer matches the expected
        names and the caller safely keeps every original.
        """
        blocks: list[tuple[str, list[str]]] = []
        prev_blank = True  # start-of-text counts as a block boundary
        for line in raw.splitlines():
            stripped = line.strip()
            is_header = (
                prev_blank
                and stripped.startswith("[")
                and stripped.endswith("]")
                and stripped[1:-1].strip() in known
            )
            if is_header:
                blocks.append((stripped[1:-1].strip(), []))
            elif blocks:
                blocks[-1][1].append(line)
            prev_blank = not stripped
        return [
            (name, " ".join(line.strip() for line in body if line.strip()).strip())
            for name, body in blocks
        ]
