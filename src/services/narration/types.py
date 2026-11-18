"""Typed shapes for the narration pipeline's in-flight payloads.

The narration path moves plain JSON-shaped dicts between the template engine,
the humanizer and the job processor, and publishes them in the job result. They
were annotated ``dict[str, Any]``, which tells a reader (and ``ty``) nothing
about the keys that are actually load-bearing — ``scenario_name`` is the join
spine the polish pass matches on and ``word_count`` is what the budget report
is computed from.

These are ``TypedDict``s rather than pydantic models deliberately: the values
ARE dicts at runtime and are serialized straight onto the wire, so naming the
keys must not change a single byte of the published payload.
"""

from __future__ import annotations

from typing import Any, TypedDict

#: One narration segment as it travels through the pipeline.
#:
#: Deliberately an OPEN mapping rather than a ``TypedDict`` or a pydantic
#: model: the pipeline's contract is "preserve every key except a smoothed
#: ``text``" (see ``humanizer._parse_polished`` and the passthrough assertion
#: in ``tests/unit/test_humanizer.py::test_preserves_extra_keys_in_parts``), so
#: callers legitimately attach keys this layer must carry through untouched. A
#: closed type would state a contract the code does not have.
#:
#: The load-bearing keys are ``part_id``, ``scenario_name``, ``text`` and
#: ``word_count``; ``expected_text`` and ``expected_recording_seconds`` are
#: conditional. What the API actually publishes is the closed pydantic model
#: ``src.api.schemas.narration.NarrationPart``.
type NarrationSection = dict[str, Any]


class NarrationPassReport(TypedDict):
    """Serialized form of ``humanizer.PolishReport`` — what one LLM pass did."""

    mode: str
    attempts: int
    parse_failures: int
    applied: bool


class WordBudgetSectionRow(TypedDict):
    """One section that delivered fewer words than its budget."""

    part_id: str | None
    scenario_name: str | None
    word_count: int
    word_budget: int
    shortfall_words: int


class WordBudgetReport(TypedDict):
    """The per-section budget outcome published as ``word_budget_report``."""

    budgeted_words: int
    delivered_words: int
    shortfall_words: int
    sections_short: list[WordBudgetSectionRow]
    shortfall_trigger_ratio: float
    polish: NarrationPassReport | None
    expansion: NarrationPassReport | None
