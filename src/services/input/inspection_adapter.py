"""Normalize crawl and inspection data into a common structure."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InteractiveElement:
    """An interactive element on a page (button, input, link, etc.)."""

    tag: str = ""
    type: str = ""
    name: str = ""
    label: str = ""
    selector: str = ""
    text: str = ""
    href: str = ""


@dataclass
class PageInfo:
    """Information about a single page or screen."""

    page_id: str = ""
    url: str = ""
    title: str = ""
    page_type: str = ""
    text_content: str = ""
    interactive_elements: list[InteractiveElement] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    has_forms: bool = False
    # How many interactive elements / headings the html-structure extractor DROPPED
    # at its caps. Carried, not derived: `job_processor._page_from_scrape` sets them
    # and `step_def_generator._collect_page_role_names` reads them to mark that
    # bucket non-exhaustive. Without these fields `asdict()` silently dropped the
    # keys on the `web_url` path — the only prod path — so a capped inventory read
    # as exhaustive and a valid `getByRole` assertion was DELETED (defeating #126).
    elements_truncated: int = 0
    headings_truncated: int = 0
    # id/class/data-testid inventory of the rendered DOM (`html_structure.DomTokensDict`).
    # Carried for the same reason as the counters above: `asdict()` drops any key the
    # dataclass does not declare, and the web_url path goes through this adapter.
    # `None` = not scanned, which the grounding pass must read as "unknown".
    dom_tokens: dict[str, Any] | None = None


@dataclass
class NavEdge:
    """An edge in the navigation graph between two pages."""

    from_page: str = ""
    to_page: str = ""
    action: str = ""
    element_label: str = ""


@dataclass
class InspectionResult:
    """Normalized result from any inspection source."""

    pages: list[PageInfo] = field(default_factory=list)
    navigation_graph: list[NavEdge] = field(default_factory=list)
    platform: str = ""
    base_url: str | None = None


def _as_count(value: object) -> int:
    """``value`` as a non-negative int count; 0 for anything that is not one.

    ``bool`` is excluded on purpose — a `True` here would read as 1 and mark a
    bucket untrusted on a type confusion rather than on a real cap.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


class InspectionAdapter:
    """Normalizes crawl and inspection data from various sources."""

    def from_web_crawl(self, data: dict[str, Any]) -> InspectionResult:
        """Normalize web crawl output into an InspectionResult.

        Args:
            data: Raw crawl data dict with pages, navigation_graph, etc.

        Returns:
            Normalized InspectionResult.
        """
        pages = self._parse_pages(data.get("pages", []))
        nav_graph = self._parse_nav_edges(data.get("navigation_graph", []))
        base_url = data.get("base_url") or data.get("url")

        logger.debug(
            "Normalized web crawl: %d pages, %d nav edges",
            len(pages),
            len(nav_graph),
        )

        return InspectionResult(
            pages=pages,
            navigation_graph=nav_graph,
            platform=data.get("platform", "web"),
            base_url=base_url,
        )

    def from_mobile_inspection(self, data: dict[str, Any]) -> InspectionResult:
        """Normalize mobile app inspection output into an InspectionResult.

        Mobile inspections use 'screens' instead of 'pages' and
        'transitions' instead of 'navigation_graph'.

        Args:
            data: Raw mobile inspection data dict.

        Returns:
            Normalized InspectionResult.
        """
        raw_screens = data.get("screens", data.get("pages", []))
        pages = self._parse_pages(raw_screens)

        raw_transitions = data.get("transitions", data.get("navigation_graph", []))
        nav_graph = self._parse_nav_edges(raw_transitions)

        platform = data.get("platform", "mobile")

        logger.debug(
            "Normalized mobile inspection: %d screens, %d transitions",
            len(pages),
            len(nav_graph),
        )

        return InspectionResult(
            pages=pages,
            navigation_graph=nav_graph,
            platform=platform,
            base_url=data.get("base_url"),
        )

    def from_dict(self, data: dict[str, Any]) -> InspectionResult:
        """Passthrough normalization — assumes data is already in the expected format.

        Args:
            data: Dict matching the InspectionResult structure.

        Returns:
            InspectionResult built directly from the dict.
        """
        pages = self._parse_pages(data.get("pages", []))
        nav_graph = self._parse_nav_edges(data.get("navigation_graph", []))

        return InspectionResult(
            pages=pages,
            navigation_graph=nav_graph,
            platform=data.get("platform", ""),
            base_url=data.get("base_url"),
        )

    def _parse_pages(self, raw_pages: list[dict[str, Any]]) -> list[PageInfo]:
        """Parse a list of raw page dicts into PageInfo instances."""
        pages: list[PageInfo] = []
        for raw in raw_pages:
            elements = [
                InteractiveElement(
                    tag=el.get("tag", ""),
                    type=el.get("type", ""),
                    name=el.get("name", ""),
                    label=el.get("label", ""),
                    selector=el.get("selector", ""),
                    text=el.get("text", ""),
                    href=el.get("href", ""),
                )
                for el in raw.get("interactive_elements", [])
            ]
            pages.append(
                PageInfo(
                    page_id=raw.get("page_id", raw.get("screen_id", "")),
                    url=raw.get("url", raw.get("screen_name", "")),
                    title=raw.get("title", ""),
                    page_type=raw.get("page_type", raw.get("screen_type", "")),
                    text_content=raw.get("text_content") or raw.get("text") or "",
                    interactive_elements=elements,
                    headings=raw.get("headings", []),
                    has_forms=raw.get("has_forms", False),
                    elements_truncated=_as_count(raw.get("elements_truncated")),
                    headings_truncated=_as_count(raw.get("headings_truncated")),
                    dom_tokens=(
                        raw.get("dom_tokens") if isinstance(raw.get("dom_tokens"), dict) else None
                    ),
                )
            )
        return pages

    def _parse_nav_edges(self, raw_edges: list[dict[str, Any]]) -> list[NavEdge]:
        """Parse a list of raw edge dicts into NavEdge instances."""
        edges: list[NavEdge] = []
        for raw in raw_edges:
            edges.append(
                NavEdge(
                    from_page=raw.get("from_page", raw.get("from_screen", "")),
                    to_page=raw.get("to_page", raw.get("to_screen", "")),
                    action=raw.get("action", ""),
                    element_label=raw.get("element_label", ""),
                )
            )
        return edges
