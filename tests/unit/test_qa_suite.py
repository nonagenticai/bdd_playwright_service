"""Grounded QA builder + validator.

The builder is deterministic: same snapshot, same suite. The validator is the gate
that turns the old failure modes (a title asserted as a heading, an invented flow, a
link that is not on the page) into rejected scenarios instead of red tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.services.generation.qa_suite import (
    GroundedSuiteError,
    build_grounded_features,
    validate_features,
)


def _el(role: str, label: str, href: str = "") -> dict[str, Any]:
    return {
        "tag": "a" if role == "link" else "button",
        "type": "",
        "name": "",
        "label": label,
        "selector": "",
        "text": label,
        "href": href,
        "role": role,
    }


def _page(
    url: str,
    title: str,
    headings: list[str],
    elements: list[dict[str, Any]],
    text: str = "",
    *,
    elements_truncated: int = 0,
) -> dict[str, Any]:
    return {
        "url": url,
        "title": title,
        "headings": headings,
        "interactive_elements": elements,
        "text_content": text,
        "elements_truncated": elements_truncated,
        "headings_truncated": 0,
    }


@pytest.fixture
def inspection() -> dict[str, Any]:
    return {
        "pages": [
            _page(
                "https://example.com/",
                "Home - Example",
                ["Home", "Welcome"],
                [
                    _el("link", "Naar de inhoud", "#content"),
                    _el("link", "Docs", "/docs"),
                    _el("button", "Start"),
                ],
                text="Welcome to the example application.",
            ),
            _page(
                "https://example.com/docs",
                "Docs - Example",
                ["Documentation"],
                [_el("link", "Home", "/")],
                text="Documentation body text.",
            ),
        ],
        "base_url": "https://example.com",
        "platform": "web",
    }


def _feature(*scenarios: str, name: str = "Rejects") -> dict[str, Any]:
    body = "\n".join(scenarios)
    return {
        "filename": "rejects.feature",
        "content": f"Feature: {name}\n\n{body}\n",
        "scenario_count": len(scenarios),
        "tags": [],
    }


def _scenario(step_lines: list[str], name: str = "a scenario") -> str:
    return "\n  Scenario: " + name + "\n" + "\n".join(f"    {line}" for line in step_lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_build_is_deterministic_and_grounded(inspection):
    first = build_grounded_features(inspection, {})
    second = build_grounded_features(inspection, {})
    assert first == second
    assert [f["filename"] for f in first] == [
        "page_contract_root.feature",
        "page_contract_docs.feature",
    ]
    combined = "\n".join(f["content"] for f in first)
    assert "://" not in combined  # paths only; the suite is portable via BASE_URL
    assert 'Then the page shows the heading "Home"' in combined
    assert 'Then the page title contains "Home - Example"' in combined
    assert 'Then the page contains the button "Start"' in combined


def test_build_emits_navigation_only_to_captured_paths(inspection):
    features = build_grounded_features(inspection, {})
    docs = next(f for f in features if f["filename"] == "page_contract_root.feature")
    assert 'When the user clicks the link "Docs"' in docs["content"]
    assert 'Then the page URL path is "/docs"' in docs["content"]
    assert 'Then the page shows the heading "Documentation"' in docs["content"]


def test_build_raises_without_pages():
    with pytest.raises(GroundedSuiteError, match="at least one captured page"):
        build_grounded_features({"pages": []}, {})


def test_build_respects_zero_navigation_when_cap_is_one(inspection):
    features = build_grounded_features(inspection, {"max_scenarios_per_feature": 1})
    assert all(f["scenario_count"] == 1 for f in features)
    assert "@navigation" not in "\n".join(f["content"] for f in features)


# ---------------------------------------------------------------------------
# Validator — accepts what the builder emits
# ---------------------------------------------------------------------------


def test_builder_output_passes_its_own_validator(inspection):
    features = build_grounded_features(inspection, {})
    kept, report = validate_features(features, inspection)
    assert sum(f["scenario_count"] for f in kept) == sum(f["scenario_count"] for f in features)
    assert report["rejected_count"] == 0
    assert report["checked_scenarios"] == report["kept_scenarios"]


# ---------------------------------------------------------------------------
# Validator — one test per defect the NetCare run exposed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("step_lines", "reason_fragment"),
    [
        # The document <title> is not a heading — every NetCare scenario died here.
        (
            [
                'Given the user opens the page "/"',
                'Then the page shows the heading "Home - Example"',
            ],
            "heading",
        ),
        # Absolute capture URL baked into a feature.
        (
            ['Given the user opens the page "https://example.com/"'],
            "absolute URL",
        ),
        # A path the snapshot never captured.
        (
            ['Given the user opens the page "/admin"'],
            "was not captured",
        ),
        # A step outside the closed vocabulary (an invented form flow).
        (
            [
                'Given the user opens the page "/"',
                "When the user submits the booking form",
            ],
            "outside the closed QA vocabulary",
        ),
        # A control that is not on the captured page.
        (
            [
                'Given the user opens the page "/"',
                'Then the page contains the link "Ghost"',
            ],
            "not on the captured",
        ),
        # A claim made before any page was opened.
        (
            ['Then the page shows the heading "Home"'],
            "before any page was opened",
        ),
        # Text the snapshot does not contain.
        (
            [
                'Given the user opens the page "/"',
                'Then the page contains the text "Account deleted"',
            ],
            "not in the captured text",
        ),
    ],
)
def test_validator_rejects_defects(inspection, step_lines, reason_fragment):
    # A single bad scenario leaves nothing executable, so the validator raises with
    # the full report attached — keeping the scenario would be worse than deleting it.
    with pytest.raises(GroundedSuiteError) as exc:
        validate_features([_feature(_scenario(step_lines))], inspection)
    report = exc.value.report
    assert report["rejected_count"] == 1
    assert reason_fragment in report["rejected"][0]["reason"]


def test_all_rejected_raises(inspection):
    with pytest.raises(GroundedSuiteError, match="failed grounding"):
        validate_features(
            [_feature(_scenario(['Given the user opens the page "/admin"']))], inspection
        )


def test_mixed_feature_keeps_good_drops_bad(inspection):
    good = _scenario(
        [
            'Given the user opens the page "/"',
            'Then the page shows the heading "Home"',
        ],
        name="good",
    )
    bad = _scenario(
        [
            'Given the user opens the page "/"',
            'Then the page contains the link "Ghost"',
        ],
        name="bad",
    )
    kept, report = validate_features([_feature(good, bad)], inspection)
    assert len(kept) == 1
    assert kept[0]["scenario_count"] == 1
    assert "Ghost" not in kept[0]["content"]
    assert "Scenario: good" in kept[0]["content"]
    assert report["rejected_count"] == 1


def test_background_is_rejected_whole(inspection):
    feature = {
        "filename": "bg.feature",
        "content": (
            "Feature: Bg\n\n  Background:\n"
            '    Given the user opens the page "/"\n\n'
            '  Scenario: s\n    Then the page shows the heading "Home"\n'
        ),
        "scenario_count": 1,
    }
    with pytest.raises(GroundedSuiteError, match="failed grounding"):
        validate_features([feature], inspection)


def test_truncated_inventory_does_not_prove_absence():
    truncated = {
        "pages": [
            _page(
                "https://example.com/",
                "Home - Example",
                ["Home"],
                [],
                elements_truncated=5,
            )
        ]
    }
    scenario = _scenario(
        [
            'Given the user opens the page "/"',
            'Then the page contains the link "Maybe"',
        ]
    )
    kept, report = validate_features([_feature(scenario)], truncated)
    assert report["rejected_count"] == 0
    assert len(kept) == 1


def test_navigation_updates_the_grounded_page(inspection):
    # After following the captured Docs link, a Docs heading is grounded and a Home
    # heading is not — the validator must track the page under test.
    nav_ok = _scenario(
        [
            'Given the user opens the page "/"',
            'When the user clicks the link "Docs"',
            'Then the page URL path is "/docs"',
            'Then the page shows the heading "Documentation"',
        ],
        name="nav-ok",
    )
    nav_bad = _scenario(
        [
            'Given the user opens the page "/"',
            'When the user clicks the link "Docs"',
            'Then the page shows the heading "Welcome"',
        ],
        name="nav-bad",
    )
    kept, report = validate_features([_feature(nav_ok, nav_bad)], inspection)
    assert kept[0]["scenario_count"] == 1
    assert "nav-ok" in kept[0]["content"]
    assert report["rejected_count"] == 1
    assert "Welcome" in report["rejected"][0]["reason"]
