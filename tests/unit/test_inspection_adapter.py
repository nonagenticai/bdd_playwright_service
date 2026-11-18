"""Tests for inspection adapter."""

from __future__ import annotations

from src.services.input.inspection_adapter import (
    InspectionAdapter,
    InspectionResult,
    InteractiveElement,
    NavEdge,
    PageInfo,
)


class TestFromWebCrawl:
    def test_basic_web_crawl(self, sample_inspection_result):
        """from_web_crawl should normalize web crawl data."""
        adapter = InspectionAdapter()
        result = adapter.from_web_crawl(sample_inspection_result)

        assert isinstance(result, InspectionResult)
        assert result.platform == "web"
        assert result.base_url == "https://example.com"
        assert len(result.pages) == 2
        assert len(result.navigation_graph) == 1

    def test_page_info_fields(self, sample_inspection_result):
        """from_web_crawl should populate PageInfo fields correctly."""
        adapter = InspectionAdapter()
        result = adapter.from_web_crawl(sample_inspection_result)

        login_page = result.pages[0]
        assert isinstance(login_page, PageInfo)
        assert login_page.page_id == "page_0"
        assert login_page.url == "https://example.com/login"
        assert login_page.title == "Login"
        assert login_page.page_type == "login"
        assert login_page.has_forms is True
        assert "Welcome" in login_page.headings

    def test_interactive_elements(self, sample_inspection_result):
        """from_web_crawl should parse interactive elements."""
        adapter = InspectionAdapter()
        result = adapter.from_web_crawl(sample_inspection_result)

        elements = result.pages[0].interactive_elements
        assert len(elements) == 3
        assert isinstance(elements[0], InteractiveElement)
        assert elements[0].tag == "input"
        assert elements[0].name == "username"
        assert elements[2].tag == "button"
        assert elements[2].text == "Sign In"

    def test_navigation_graph(self, sample_inspection_result):
        """from_web_crawl should parse navigation edges."""
        adapter = InspectionAdapter()
        result = adapter.from_web_crawl(sample_inspection_result)

        assert len(result.navigation_graph) == 1
        edge = result.navigation_graph[0]
        assert isinstance(edge, NavEdge)
        assert edge.from_page == "page_0"
        assert edge.to_page == "page_1"
        assert edge.action == "submit"
        assert edge.element_label == "Sign In"

    def test_empty_crawl_data(self):
        """from_web_crawl should handle empty data gracefully."""
        adapter = InspectionAdapter()
        result = adapter.from_web_crawl({})

        assert result.pages == []
        assert result.navigation_graph == []
        assert result.platform == "web"
        assert result.base_url is None

    def test_url_fallback_for_base_url(self):
        """from_web_crawl should fall back to 'url' if 'base_url' is missing."""
        adapter = InspectionAdapter()
        data = {"url": "https://fallback.example.com", "pages": []}
        result = adapter.from_web_crawl(data)
        assert result.base_url == "https://fallback.example.com"

    def test_text_content_populated_from_canonical_key(self):
        """A page dict carrying the canonical 'text_content' populates PageInfo."""
        adapter = InspectionAdapter()
        data = {"pages": [{"url": "https://x.com", "text_content": "Clean body."}]}
        result = adapter.from_web_crawl(data)
        assert result.pages[0].text_content == "Clean body."

    def test_text_content_falls_back_to_legacy_text_key(self):
        """Durability: a producer still emitting the legacy 'text' key is honored,
        so neither side is brittle (regression for the silent-empty field mismatch
        between job_processor and the adapter)."""
        adapter = InspectionAdapter()
        data = {"pages": [{"url": "https://x.com", "text": "Legacy body."}]}
        result = adapter.from_web_crawl(data)
        assert result.pages[0].text_content == "Legacy body."

    def test_text_content_prefers_canonical_over_legacy(self):
        """When both keys are present the canonical 'text_content' wins."""
        adapter = InspectionAdapter()
        data = {
            "pages": [{"url": "https://x.com", "text_content": "Canonical.", "text": "Legacy."}]
        }
        result = adapter.from_web_crawl(data)
        assert result.pages[0].text_content == "Canonical."


class TestFromMobileInspection:
    def test_mobile_screens(self):
        """from_mobile_inspection should normalize mobile screen data."""
        adapter = InspectionAdapter()
        data = {
            "screens": [
                {
                    "screen_id": "screen_0",
                    "screen_name": "Login Screen",
                    "title": "Login",
                    "screen_type": "login",
                    "text_content": "Welcome",
                    "interactive_elements": [
                        {
                            "tag": "input",
                            "type": "text",
                            "name": "email",
                            "label": "Email",
                            "selector": "#email",
                        }
                    ],
                    "headings": ["Welcome"],
                    "has_forms": True,
                }
            ],
            "transitions": [
                {
                    "from_screen": "screen_0",
                    "to_screen": "screen_1",
                    "action": "tap",
                    "element_label": "Login Button",
                }
            ],
            "platform": "ios",
        }

        result = adapter.from_mobile_inspection(data)
        assert isinstance(result, InspectionResult)
        assert result.platform == "ios"
        assert len(result.pages) == 1
        assert result.pages[0].page_id == "screen_0"
        assert result.pages[0].url == "Login Screen"
        assert len(result.navigation_graph) == 1
        assert result.navigation_graph[0].from_page == "screen_0"
        assert result.navigation_graph[0].to_page == "screen_1"

    def test_mobile_defaults_platform(self):
        """from_mobile_inspection should default platform to 'mobile'."""
        adapter = InspectionAdapter()
        result = adapter.from_mobile_inspection({})
        assert result.platform == "mobile"

    def test_mobile_falls_back_to_pages(self):
        """from_mobile_inspection should fall back to 'pages' if 'screens' missing."""
        adapter = InspectionAdapter()
        data = {
            "pages": [
                {
                    "page_id": "p0",
                    "url": "/home",
                    "title": "Home",
                    "page_type": "content",
                    "text_content": "Home screen",
                    "interactive_elements": [],
                    "headings": [],
                    "has_forms": False,
                }
            ],
        }
        result = adapter.from_mobile_inspection(data)
        assert len(result.pages) == 1
        assert result.pages[0].page_id == "p0"


class TestFromDict:
    def test_passthrough(self, sample_inspection_result):
        """from_dict should pass data through with proper typing."""
        adapter = InspectionAdapter()
        result = adapter.from_dict(sample_inspection_result)

        assert isinstance(result, InspectionResult)
        assert result.platform == "web"
        assert result.base_url == "https://example.com"
        assert len(result.pages) == 2
        assert len(result.navigation_graph) == 1

    def test_passthrough_empty(self):
        """from_dict should handle empty dict."""
        adapter = InspectionAdapter()
        result = adapter.from_dict({})

        assert result.pages == []
        assert result.navigation_graph == []
        assert result.platform == ""
        assert result.base_url is None

    def test_passthrough_preserves_elements(self, sample_inspection_result):
        """from_dict should preserve interactive elements."""
        adapter = InspectionAdapter()
        result = adapter.from_dict(sample_inspection_result)

        login_page = result.pages[0]
        assert len(login_page.interactive_elements) == 3
        assert login_page.interactive_elements[0].selector == "#username"


class TestTruncationFlagsSurvive:
    """Bughunt 2026-08-21 #3: the truncation flags job_processor deliberately sets
    must survive `PageInfo` + `asdict()`, or PR #126's untrusted-bucket logic reads
    `None` on the only prod path (`web_url`) and deletes valid getByRole assertions.
    """

    def test_flags_round_trip_through_asdict(self):
        from dataclasses import asdict

        adapter = InspectionAdapter()
        result = adapter.from_web_crawl(
            {
                "url": "https://example.com",
                "pages": [
                    {
                        "url": "https://example.com",
                        "title": "Home",
                        "interactive_elements": [],
                        "headings": ["A"],
                        "elements_truncated": 7,
                        "headings_truncated": 3,
                    }
                ],
            }
        )
        assert result.pages[0].elements_truncated == 7
        assert result.pages[0].headings_truncated == 3
        page = asdict(result)["pages"][0]
        assert page["elements_truncated"] == 7
        assert page["headings_truncated"] == 3

    def test_flags_default_to_zero_when_absent(self):
        adapter = InspectionAdapter()
        result = adapter.from_dict({"pages": [{"url": "u"}]})
        assert result.pages[0].elements_truncated == 0
        assert result.pages[0].headings_truncated == 0

    def test_capped_inventory_keeps_the_bucket_untrusted(self):
        """End-to-end: the flag must reach `_inventory_role_names`'s untrust logic."""
        from dataclasses import asdict

        from src.services.generation.step_def_generator import _inventory_role_names

        adapter = InspectionAdapter()
        inspection = asdict(
            adapter.from_web_crawl(
                {
                    "url": "https://example.com",
                    "pages": [
                        {
                            "url": "https://example.com",
                            "headings": ["Dashboard"],
                            "interactive_elements": [],
                            "headings_truncated": 5,
                        }
                    ],
                }
            )
        )
        assert "heading" not in _inventory_role_names(inspection)
