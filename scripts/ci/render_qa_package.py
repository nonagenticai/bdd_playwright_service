#!/usr/bin/env python3
"""Render a grounded QA package into a directory — the CI compile gate's input.

The service image carries no Node runtime, so generated TypeScript is never
type-checked while the service runs. This script drives the SAME production
functions (``ConfigGenerator`` + ``qa_suite.build_grounded_features`` +
``validate_features``) over a representative inspection, and the ``package-compile``
CI job then runs ``bddgen`` and ``tsc --noEmit`` on the output. That is the gate
that catches the class of defect the old LLM step generator shipped: ``createBd()``,
an ``await`` in a non-async arrow, or a quoted ``"{string}"`` placeholder.

Usage: python scripts/ci/render_qa_package.py <output-dir>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.services.generation.config_generator import ConfigGenerator
from src.services.generation.qa_suite import build_grounded_features, validate_features


def _page(
    url: str,
    title: str,
    headings: list[str],
    elements: list[dict[str, Any]],
    text: str,
) -> dict[str, Any]:
    return {
        "url": url,
        "title": title,
        "headings": headings,
        "interactive_elements": elements,
        "text_content": text,
        "elements_truncated": 0,
        "headings_truncated": 0,
        "page_type": "content",
        "has_forms": False,
    }


def _link(label: str, href: str) -> dict[str, Any]:
    return {
        "tag": "a",
        "type": "",
        "name": "",
        "label": label,
        "selector": "",
        "text": label,
        "href": href,
        "role": "link",
    }


def _button(label: str) -> dict[str, Any]:
    return {
        "tag": "button",
        "type": "",
        "name": "",
        "label": label,
        "selector": "",
        "text": label,
        "href": "",
        "role": "button",
    }


#: Three pages, two captured-link targets, plural headings and a button: enough to
#: exercise every step in the vocabulary at least once.
INSPECTION: dict[str, Any] = {
    "pages": [
        _page(
            "https://example.com/",
            "Home - Example",
            ["Home", "Welcome"],
            [_link("Naar de inhoud", "#content"), _link("Docs", "/docs"), _button("Start")],
            "Welcome to the example application.",
        ),
        _page(
            "https://example.com/docs",
            "Docs - Example",
            ["Documentation"],
            [_link("Home", "/")],
            "Documentation body text.",
        ),
        _page(
            "https://example.com/account",
            "Account - Example",
            ["Account"],
            [_button("Save")],
            "Account settings.",
        ),
    ],
    "navigation_graph": [],
    "platform": "web",
    "base_url": "https://example.com",
}


def render(out_dir: Path) -> None:
    config_gen = ConfigGenerator()
    features, report = validate_features(build_grounded_features(INSPECTION, {}), INSPECTION)
    if report["rejected_count"]:
        raise SystemExit(f"grounded builder emitted ungrounded scenarios: {report['rejected']}")

    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    (out_dir / "steps").mkdir(parents=True, exist_ok=True)
    for feature in features:
        (out_dir / "features" / feature["filename"]).write_text(
            feature["content"], encoding="utf-8"
        )
    steps = config_gen.read_qa_steps()
    if len(steps) != 1:
        raise SystemExit(f"expected exactly one static step file, got {len(steps)}")
    (out_dir / "steps" / steps[0]["filename"]).write_text(steps[0]["content"], encoding="utf-8")
    (out_dir / "playwright.config.ts").write_text(
        config_gen.generate_qa_config("https://example.com"), encoding="utf-8"
    )
    (out_dir / "package.json").write_text(config_gen.generate_qa_package_json(), encoding="utf-8")
    (out_dir / "tsconfig.json").write_text(config_gen.generate_tsconfig(), encoding="utf-8")
    print(f"rendered {len(features)} feature file(s), {report['kept_scenarios']} scenario(s)")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    render(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
