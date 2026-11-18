"""Parse Gherkin .feature files into structured data."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.api.schemas.gherkin import GherkinFeature, GherkinScenario, GherkinStep
from src.services.gherkin_keywords import SCENARIO_OPENER_RE, STEP_LINE_RE

logger = logging.getLogger(__name__)

# Regex patterns for Gherkin parsing
_TAG_RE = re.compile(r"(@\S+)")
_DATA_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|$")


class GherkinParser:
    """Parses Gherkin .feature text into structured Pydantic models."""

    def __init__(self) -> None:
        self._background_steps: list[GherkinStep] = []

    def parse_feature_text(self, text: str) -> GherkinFeature:
        """Parse raw Gherkin text into a GherkinFeature model.

        Args:
            text: The raw Gherkin feature file content.

        Returns:
            A GherkinFeature with name, description, tags, and scenarios.
        """
        if not text or not text.strip():
            return GherkinFeature(name="")

        lines = text.split("\n")

        feature_name = ""
        feature_description = ""
        feature_tags: list[str] = []
        scenarios: list[GherkinScenario] = []

        # Track parsing state
        i = 0
        pending_tags: list[str] = []

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Collect tags
            if stripped.startswith("@"):
                pending_tags.extend(_TAG_RE.findall(stripped))
                i += 1
                continue

            # Feature line
            feature_match = re.match(r"^\s*Feature:\s*(.+)$", stripped)
            if feature_match:
                feature_name = feature_match.group(1).strip()
                feature_tags = pending_tags
                pending_tags = []
                i += 1
                # Collect description lines until we hit a keyword
                desc_lines: list[str] = []
                while i < len(lines):
                    desc_stripped = lines[i].strip()
                    if (
                        desc_stripped.startswith(("@", "Background:"))
                        # Same opener set the @clip-<n> stamper uses, so a description
                        # can never swallow an `Example:`/`Scenario Template:` scenario.
                        or SCENARIO_OPENER_RE.match(desc_stripped)
                    ):
                        break
                    if desc_stripped:
                        desc_lines.append(desc_stripped)
                    i += 1
                feature_description = "\n".join(desc_lines)
                continue

            # Background
            if stripped.startswith("Background:"):
                pending_tags = []
                i += 1
                bg_steps = self._parse_steps_block(lines, i)
                i += len(bg_steps)
                self._background_steps = bg_steps
                continue

            # A scenario opener, in EVERY keyword spelling the @clip-<n> stamper accepts
            # (`Scenario:`, `Scenario Outline:`, `Scenario Template:`, `Example:`).
            # This regex is imported, not restated: a scenario the stamper tags but this
            # parser skips is a video clip with NO narration part, and every later clip's
            # voiceover then describes the wrong screen.
            scenario_match = SCENARIO_OPENER_RE.match(stripped)
            if scenario_match:
                scenario_name = scenario_match.group("name").strip()
                scenario_tags = pending_tags
                pending_tags = []
                i += 1
                scenario_steps = self._parse_steps_block(lines, i)
                i += len(scenario_steps)

                # Prepend background steps if present
                all_steps = [s.model_copy() for s in self._background_steps] + scenario_steps

                scenarios.append(
                    GherkinScenario(
                        name=scenario_name,
                        tags=scenario_tags,
                        steps=all_steps,
                        background_step_count=len(self._background_steps),
                    )
                )
                continue

            i += 1

        # Reset background state for next parse call
        self._background_steps = []

        return GherkinFeature(
            name=feature_name,
            description=feature_description,
            tags=feature_tags,
            scenarios=scenarios,
        )

    def parse_feature_file(self, path: str) -> GherkinFeature:
        """Parse a .feature file from disk.

        Args:
            path: File system path to a .feature file.

        Returns:
            A GherkinFeature parsed from the file content.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

        text = file_path.read_text(encoding="utf-8")
        logger.debug("Parsing feature file: %s (%d chars)", path, len(text))
        return self.parse_feature_text(text)

    def extract_steps(self, text: str) -> list[GherkinStep]:
        """Extract all Gherkin steps from raw text without scenario grouping.

        Args:
            text: Raw Gherkin text (may be a full feature or a fragment).

        Returns:
            A flat list of all GherkinStep instances found.
        """
        steps: list[GherkinStep] = []
        lines = text.split("\n")
        i = 0

        while i < len(lines):
            stripped = lines[i].strip()
            step_match = STEP_LINE_RE.match(stripped)
            if step_match:
                keyword = step_match.group("keyword")
                step_text = step_match.group("text").strip()

                # Check for data table following the step
                data_table = self._try_parse_data_table(lines, i + 1)
                table_rows = len(data_table) if data_table else 0

                steps.append(
                    GherkinStep(
                        keyword=keyword,
                        text=step_text,
                        data_table=data_table,
                    )
                )
                i += 1 + table_rows
            else:
                i += 1

        return steps

    def _parse_steps_block(self, lines: list[str], start: int) -> list[GherkinStep]:
        """Parse consecutive step lines starting at the given index.

        Stops when hitting a non-step line (tag, scenario, background, Examples, or blank
        followed by a keyword).

        Returns:
            List of GherkinStep parsed from the block.
        """
        steps: list[GherkinStep] = []
        i = start

        while i < len(lines):
            stripped = lines[i].strip()

            # Stop at scenario boundaries
            if (
                stripped.startswith("Scenario:")
                or stripped.startswith("Scenario Outline:")
                or stripped.startswith("Background:")
                or stripped.startswith("@")
                or stripped.startswith("Examples:")
            ):
                break

            step_match = STEP_LINE_RE.match(stripped)
            if step_match:
                keyword = step_match.group("keyword")
                step_text = step_match.group("text").strip()

                # Check for data table
                data_table = self._try_parse_data_table(lines, i + 1)
                table_rows = len(data_table) if data_table else 0

                steps.append(
                    GherkinStep(
                        keyword=keyword,
                        text=step_text,
                        data_table=data_table,
                    )
                )
                i += 1 + table_rows
            elif stripped == "" or stripped.startswith("#"):
                # Skip blank lines and comments within a block
                i += 1
            elif _DATA_TABLE_ROW_RE.match(stripped):
                # Orphan data table row (shouldn't happen in well-formed Gherkin)
                i += 1
            else:
                break

        return steps

    def _try_parse_data_table(self, lines: list[str], start: int) -> list[list[str]] | None:
        """Try to parse a data table starting at the given line index.

        Returns:
            A list of rows (each a list of cell strings), or None if no table found.
        """
        rows: list[list[str]] = []
        i = start

        while i < len(lines):
            stripped = lines[i].strip()
            table_match = _DATA_TABLE_ROW_RE.match(stripped)
            if table_match:
                cells = [cell.strip() for cell in table_match.group(1).split("|")]
                rows.append(cells)
                i += 1
            else:
                break

        return rows if rows else None
