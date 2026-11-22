"""The QA vocabulary and its TypeScript template must describe the same steps.

The grounded QA path's whole safety argument is that the static ``qa/steps.ts``
template defines exactly the steps the generator emits and the validator accepts.
These tests make that three-way agreement mechanical: add a step to one side only
and CI fails here, before a package is ever built.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.services.generation.qa_vocabulary import (
    QA_STEPS,
    VOCABULARY_KEYS,
    QaStepSpec,
    match_step,
    vocabulary_prompt,
)

_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "src" / "templates" / "scaffold" / "qa" / "steps.ts"
)
_STEP_CALL_RE = re.compile(r"^(Given|When|Then)\('([^']*)'", re.MULTILINE)


def test_template_and_registry_describe_the_same_steps():
    template = _TEMPLATE.read_text(encoding="utf-8")
    found = [(keyword, pattern) for keyword, pattern in _STEP_CALL_RE.findall(template)]
    assert found == list(VOCABULARY_KEYS)


def test_template_has_no_quoted_placeholder():
    """``"{string}"`` is a literal quote + parameter + quote: it can never match.

    This is the exact defect the old LLM step generator shipped 19 times. Only the
    step-call patterns are judged — the header comment quotes the broken shape to
    explain it, and that prose is not a step definition.
    """
    template = _TEMPLATE.read_text(encoding="utf-8")
    patterns = [pattern for _, pattern in _STEP_CALL_RE.findall(template)]
    assert patterns
    for pattern in patterns:
        assert '"{string}"' not in pattern
        assert "'{string}'" not in pattern


def test_template_uses_createbdd_not_a_typo():
    template = _TEMPLATE.read_text(encoding="utf-8")
    assert "createBdd()" in template
    assert "createBd()" not in template


def test_every_spec_matches_its_own_rendered_step():
    for spec in QA_STEPS:
        values = tuple(f"value{i}" for i in range(spec.arg_count))
        line = spec.format(*values)
        keyword, _, text = line.partition(" ")
        match = match_step(keyword, text)
        assert match is not None, line
        matched_spec, matched_values = match
        assert matched_spec == spec
        assert matched_values == values


def test_match_rejects_a_wrong_binding():
    assert match_step("When", 'the user opens the page "/x"') is None


def test_match_accepts_single_quotes_like_cucumber():
    match = match_step("Given", "the user opens the page '/x'")
    assert match is not None and match[1] == ("/x",)


def test_format_refuses_quote_bearing_arguments():
    spec = QaStepSpec("Given", "the user opens the page {string}", ("page_path",))
    with pytest.raises(ValueError, match="quote"):
        spec.format('"/x"')
    with pytest.raises(ValueError, match="argument"):
        spec.format("/x", "/y")


def test_prompt_lists_every_step():
    prompt = vocabulary_prompt()
    for spec in QA_STEPS:
        assert f"{spec.keyword} {spec.text}" in prompt
