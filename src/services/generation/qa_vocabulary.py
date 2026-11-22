"""The CLOSED step vocabulary a grounded QA suite may use.

A grounded QA package ships exactly one step-definition file,
``templates/scaffold/qa/steps.ts``, emitted verbatim. The deterministic feature
builder and the validator both speak to it through this registry, so the three
cannot drift: the emitter picks a spec, the validator recognises only a spec, and a
unit test compares this table against the TypeScript template step-for-step.

Why a closed vocabulary at all: the previous QA path asked an LLM to write the
TypeScript. It shipped ``createBd()``, an ``await`` in a non-async arrow, quoted
``"{string}"`` placeholders (which cucumber reads as literal text, so the step can
never match), and one Gherkin step with no definition at all. The service has no
Node runtime, so nothing could ever type-check or bddgen that output. A static
template, compiled once in CI, removes the entire class.

⚠️ The pattern text carries ``{string}`` UNQUOTED. ``{string}`` already matches the
surrounding quotes in a cucumber expression; writing ``"{string}"`` asks for a
literal quote plus the parameter plus another literal quote, which matches nothing.

The vocabulary is deliberately shallow: it can open a captured page, follow a link
whose target was captured, and assert headings/titles/links/buttons/text that the
snapshot proves exist. Business-state flows (expiry, retention, email confirmation)
are not expressible — they are hand-authored, not guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

QaArgKind = Literal["page_path", "heading", "title", "link", "button", "text"]
QaKeyword = Literal["Given", "When", "Then"]


@dataclass(frozen=True)
class QaStepSpec:
    """One bindable step. ``text`` is the cucumber-expression pattern."""

    keyword: QaKeyword
    text: str
    args: tuple[QaArgKind, ...]

    @property
    def arg_count(self) -> int:
        return len(self.args)

    def format(self, *values: str) -> str:
        """Render a Gherkin step line from this spec and its quoted arguments."""
        if len(values) != self.arg_count:
            raise ValueError(f"{self.text!r} takes {self.arg_count} argument(s), got {len(values)}")
        rendered = self.text
        for value in values:
            if '"' in value or "'" in value:
                raise ValueError(f"a step argument may not contain a quote: {value!r}")
            rendered = rendered.replace("{string}", f'"{value}"', 1)
        return f"{self.keyword} {rendered}"


QA_STEPS: Final[tuple[QaStepSpec, ...]] = (
    QaStepSpec("Given", "the user opens the page {string}", ("page_path",)),
    QaStepSpec("When", "the user clicks the link {string}", ("link",)),
    QaStepSpec("Then", "the page URL path is {string}", ("page_path",)),
    QaStepSpec("Then", "the page shows the heading {string}", ("heading",)),
    QaStepSpec("Then", "the page title contains {string}", ("title",)),
    QaStepSpec("Then", "the page contains the link {string}", ("link",)),
    QaStepSpec("Then", "the page contains the button {string}", ("button",)),
    QaStepSpec("Then", "the page contains the text {string}", ("text",)),
)

#: ``(keyword, text)`` pairs in template order — the shape the steps.ts template
#: must contain, and what the consistency test compares against.
VOCABULARY_KEYS: Final[tuple[tuple[str, str], ...]] = tuple(
    (spec.keyword, spec.text) for spec in QA_STEPS
)

_ARG_GROUP = "(?:\"(?P<dq{n}>[^\"]*)\"|'(?P<sq{n}>[^']*)')"


def _compile(spec: QaStepSpec) -> re.Pattern[str]:
    """Turn a spec into a matcher for the step text (keyword already stripped).

    Double- and single-quoted arguments are both accepted, matching cucumber's
    ``{string}``; anything else — including the old quoted ``"{string}"`` shape —
    does not match and is rejected by the validator.
    """
    parts = spec.text.split("{string}")
    if len(parts) != spec.arg_count + 1:
        raise ValueError(f"spec {spec.text!r} disagrees with its {spec.arg_count} arg(s)")
    pattern = re.escape(parts[0])
    for i, part in enumerate(parts[1:]):
        pattern += _ARG_GROUP.format(n=i) + re.escape(part)
    return re.compile("^" + pattern + "$")


_MATCHERS: Final[tuple[tuple[QaStepSpec, re.Pattern[str]], ...]] = tuple(
    (spec, _compile(spec)) for spec in QA_STEPS
)


def match_step(binding: str, text: str) -> tuple[QaStepSpec, tuple[str, ...]] | None:
    """Match one step against the vocabulary.

    ``binding`` is the RESOLVED keyword (``And``/``But`` already folded into the
    preceding Given/When/Then), so a step bound as ``Given`` in the feature and
    declared as ``Given`` in the template agree on the binding that playwright-bdd
    will look up.
    """
    for spec, matcher in _MATCHERS:
        if spec.keyword != binding:
            continue
        match = matcher.match(text.strip())
        if not match:
            continue
        values: list[str] = []
        for i in range(spec.arg_count):
            value = match.group(f"dq{i}")
            if value is None:
                value = match.group(f"sq{i}") or ""
            values.append(value)
        return spec, tuple(values)
    return None


def vocabulary_prompt() -> str:
    """The closed vocabulary as prompt lines for the opt-in LLM feature pass."""
    return "\n".join(f"- {spec.keyword} {spec.text}" for spec in QA_STEPS)
