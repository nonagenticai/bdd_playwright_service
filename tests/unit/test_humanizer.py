"""Tests for narration humanizer with mocked LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.services.generation.llm_client import LLMClient
from src.services.narration.humanizer import NarrationHumanizer


@pytest.fixture
def mock_llm_client():
    """Create a mock LLMClient."""
    client = AsyncMock(spec=LLMClient)
    client.generate = AsyncMock()
    return client


@pytest.fixture
def humanizer(mock_llm_client):
    """Create a NarrationHumanizer with a mock LLM client."""
    return NarrationHumanizer(mock_llm_client)


@pytest.fixture
def sample_narration_parts():
    """Sample narration parts for polishing."""
    return [
        {
            "scenario_name": "User Login",
            "text": (
                "Let's start by navigating to the login page. "
                "Type testuser into the username field. "
                "You should now see Welcome on the page."
            ),
        },
        {
            "scenario_name": "Dashboard Navigation",
            "text": ("Let's start by navigating to the dashboard page. Now, click on Settings."),
        },
    ]


# ---------------------------------------------------------------------------
# _parse_polished — direct unit tests on the parsing logic
# ---------------------------------------------------------------------------


class TestParsePolished:
    """Direct tests of the static _parse_polished method — no LLM mock needed."""

    def test_parses_two_sections(self, sample_narration_parts):
        """Correctly splits two polished sections back into parts."""
        raw = (
            "[User Login]\n"
            "Head over to the login page to begin.\n\n"
            "[Dashboard Navigation]\n"
            "Now let's explore the dashboard and click Settings."
        )
        result = NarrationHumanizer._parse_polished(raw, sample_narration_parts)

        assert len(result) == 2
        assert result[0]["scenario_name"] == "User Login"
        assert result[0]["text"] == "Head over to the login page to begin."
        assert result[1]["scenario_name"] == "Dashboard Navigation"
        assert result[1]["text"] == "Now let's explore the dashboard and click Settings."

    def test_preserves_original_when_fewer_sections(self, sample_narration_parts):
        """Fewer sections than parts is not a faithful echo -> ALL keep original."""
        raw = "[User Login]\nOnly one polished section."
        result = NarrationHumanizer._parse_polished(raw, sample_narration_parts)

        assert len(result) == 2
        # Parsed headers ([User Login]) != the two expected names -> reject wholesale.
        assert result[0]["text"] == sample_narration_parts[0]["text"]
        assert result[1]["text"] == sample_narration_parts[1]["text"]

    def test_strips_bracket_headers(self):
        """Header lines starting with '[' are removed from the polished text."""
        parts = [{"scenario_name": "Test", "text": "original"}]
        raw = "[Test]\nPolished content only."
        result = NarrationHumanizer._parse_polished(raw, parts)

        assert result[0]["text"] == "Polished content only."
        # Confirm the bracket header isn't embedded in the text
        assert "[Test]" not in result[0]["text"]

    def test_joins_multiline_text(self):
        """Multi-line polished text is joined into a single string."""
        parts = [{"scenario_name": "Test", "text": "original"}]
        raw = "[Test]\nFirst line.\nSecond line.\nThird line."
        result = NarrationHumanizer._parse_polished(raw, parts)

        assert result[0]["text"] == "First line. Second line. Third line."

    def test_empty_raw_preserves_originals(self):
        """Empty LLM response preserves all original texts."""
        parts = [
            {"scenario_name": "A", "text": "alpha"},
            {"scenario_name": "B", "text": "beta"},
        ]
        result = NarrationHumanizer._parse_polished("", parts)

        assert len(result) == 2
        # Empty string splits into [""], so section 0 exists but has no non-bracket lines
        # The original part text should be preserved when no valid text_lines
        assert result[1]["text"] == "beta"

    def test_preserves_extra_keys_in_parts(self):
        """Extra keys in the original parts (besides scenario_name/text) are preserved."""
        parts = [{"scenario_name": "X", "text": "old", "word_count": 42, "audio_url": "/x.wav"}]
        raw = "[X]\nNew polished text."
        result = NarrationHumanizer._parse_polished(raw, parts)

        assert result[0]["word_count"] == 42
        assert result[0]["audio_url"] == "/x.wav"
        assert result[0]["text"] == "New polished text."


class TestParsePolishedHeaderAlignment:
    """Strict faithful-echo: polish is accepted only on an exact, in-order name match.

    Every case where the LLM output deviates from `[Name]` blocks in the exact
    scenario order -- reordered, dropped, merged, duplicated, glued, unbracketed,
    or an in-body bracket reference -- keeps EVERY part's original narration
    rather than risk pairing a clip with the wrong scenario's text.
    """

    @pytest.fixture
    def three_parts(self):
        return [
            {"scenario_name": "User Login", "text": "orig login"},
            {"scenario_name": "Dashboard Navigation", "text": "orig dashboard"},
            {"scenario_name": "User Logout", "text": "orig logout"},
        ]

    def test_faithful_echo_all_polished(self, three_parts):
        """All headers present, in order, one paragraph each -> all polished."""
        raw = (
            "[User Login]\n"
            "Begin at the login page.\n\n"
            "[Dashboard Navigation]\n"
            "Browse the dashboard.\n\n"
            "[User Logout]\n"
            "Sign out when finished."
        )
        result = NarrationHumanizer._parse_polished(raw, three_parts)

        assert result[0]["text"] == "Begin at the login page."
        assert result[1]["text"] == "Browse the dashboard."
        assert result[2]["text"] == "Sign out when finished."

    def test_multiparagraph_body_joins_within_faithful_echo(self, three_parts):
        """A scenario polished into TWO paragraphs joins fully; alignment holds."""
        raw = (
            "[User Login]\n"
            "Head to the login page.\n\n"
            "Then enter your credentials to continue.\n\n"
            "[Dashboard Navigation]\n"
            "Now explore the dashboard.\n\n"
            "[User Logout]\n"
            "Finally, sign out cleanly."
        )
        result = NarrationHumanizer._parse_polished(raw, three_parts)

        assert result[0]["text"] == (
            "Head to the login page. Then enter your credentials to continue."
        )
        assert result[1]["text"] == "Now explore the dashboard."
        assert result[2]["text"] == "Finally, sign out cleanly."

    def test_reordered_headers_keep_all_original(self, three_parts):
        """Out-of-order headers are not a faithful echo -> ALL keep original."""
        raw = (
            "[User Logout]\n"
            "Sign out when finished.\n\n"
            "[User Login]\n"
            "Begin at the login page.\n\n"
            "[Dashboard Navigation]\n"
            "Browse the dashboard."
        )
        result = NarrationHumanizer._parse_polished(raw, three_parts)

        assert [r["text"] for r in result] == ["orig login", "orig dashboard", "orig logout"]

    def test_dropped_scenario_keeps_all_original(self, three_parts):
        """A missing scenario means the echo is incomplete -> ALL keep original."""
        raw = "[User Login]\nBegin at the login page.\n\n[User Logout]\nSign out when finished."
        result = NarrationHumanizer._parse_polished(raw, three_parts)

        assert [r["text"] for r in result] == ["orig login", "orig dashboard", "orig logout"]

    def test_in_body_bracket_reference_keeps_all_original(self, three_parts):
        """Adversarial CASE1: a bracketed reference mid-body (no blank line before).

        `[Dashboard Navigation]` sits inside Login's body, so it is not a header;
        the parsed name list ([User Login, User Logout]) != expected -> ALL
        original. No fragment of one scenario ever lands on another.
        """
        raw = (
            "[User Login]\n"
            "Open the account page. To change these,\n"
            "[Dashboard Navigation]\n"
            "and adjust each option there.\n\n"
            "[User Logout]\n"
            "Sign out when finished."
        )
        result = NarrationHumanizer._parse_polished(raw, three_parts)

        assert [r["text"] for r in result] == ["orig login", "orig dashboard", "orig logout"]

    def test_glued_headers_no_blank_line_keep_all_original(self, three_parts):
        """Adversarial CASE2: headers glued with no blank line -> ALL original."""
        raw = (
            "[User Login]\n"
            "Begin at the login page.\n"
            "[Dashboard Navigation]\n"
            "Browse the dashboard.\n"
            "[User Logout]\n"
            "Sign out when finished."
        )
        result = NarrationHumanizer._parse_polished(raw, three_parts)

        assert [r["text"] for r in result] == ["orig login", "orig dashboard", "orig logout"]

    def test_glued_bracketed_scenario_name_keeps_all_original(self):
        """Adversarial CASE3: a bracketed scenario name (`Checkout [v2]`) glued -> original."""
        parts = [
            {"scenario_name": "Login", "text": "orig login"},
            {"scenario_name": "Checkout [v2]", "text": "orig checkout"},
        ]
        raw = "[Login]\nPolished login text.\n[Checkout [v2]]\nPolished checkout text."
        result = NarrationHumanizer._parse_polished(raw, parts)

        assert [r["text"] for r in result] == ["orig login", "orig checkout"]

    def test_unbracketed_glued_header_keeps_all_original(self):
        """Adversarial CASE4: a header the model emitted WITHOUT brackets.

        `Signup` on its own line is not a header (no brackets) and not a token,
        so it and its body used to bleed into the preceding Login part. Strict
        echo: parsed names ([Login]) != [Login, Signup] -> ALL keep original.
        """
        parts = [
            {"scenario_name": "Login", "text": "orig login"},
            {"scenario_name": "Signup", "text": "orig signup"},
        ]
        raw = "[Login]\nPolished login flow.\n\nSignup\nPolished signup flow."
        result = NarrationHumanizer._parse_polished(raw, parts)

        assert [r["text"] for r in result] == ["orig login", "orig signup"]

    def test_single_scenario_with_bracketed_phrase_polishes(self):
        """A lone faithful scenario polishes even if its body has a bracketed phrase."""
        parts = [{"scenario_name": "User Login", "text": "orig login"}]
        raw = "[User Login]\nOpen the page shown in [see the login screen] and sign in."
        result = NarrationHumanizer._parse_polished(raw, parts)

        assert result[0]["text"] == "Open the page shown in [see the login screen] and sign in."

    def test_duplicate_scenario_name_in_parts_keeps_all_original(self):
        """Two parts sharing a scenario_name cannot echo faithfully -> ALL original."""
        parts = [
            {"scenario_name": "Login", "text": "first original"},
            {"scenario_name": "Login", "text": "second original"},
            {"scenario_name": "Logout", "text": "orig logout"},
        ]
        raw = "[Login]\nPolished login body.\n\n[Logout]\nPolished logout body."
        result = NarrationHumanizer._parse_polished(raw, parts)

        assert [r["text"] for r in result] == ["first original", "second original", "orig logout"]

    def test_duplicate_header_keeps_all_original(self, three_parts):
        """A duplicated header makes the name list diverge -> ALL keep original."""
        raw = (
            "[User Login]\n"
            "First login body.\n\n"
            "[User Login]\n"
            "Second login body.\n\n"
            "[Dashboard Navigation]\n"
            "Browse the dashboard."
        )
        result = NarrationHumanizer._parse_polished(raw, three_parts)

        assert [r["text"] for r in result] == ["orig login", "orig dashboard", "orig logout"]

    def test_empty_body_in_echo_keeps_all_original(self, three_parts):
        """A header with no body is not a usable polish -> ALL keep original."""
        raw = "[User Login]\n\n[Dashboard Navigation]\nBrowse.\n\n[User Logout]\nSign out."
        result = NarrationHumanizer._parse_polished(raw, three_parts)

        assert [r["text"] for r in result] == ["orig login", "orig dashboard", "orig logout"]

    def test_extra_keys_preserved_on_faithful_polish(self):
        """Non-text keys survive a polish."""
        parts = [{"scenario_name": "X", "text": "old", "clip_id": "clip-0", "n": 3}]
        raw = "[X]\nNew polished text."
        result = NarrationHumanizer._parse_polished(raw, parts)

        assert result[0]["text"] == "New polished text."
        assert result[0]["clip_id"] == "clip-0"
        assert result[0]["n"] == 3

    async def test_polish_multiparagraph_via_mock_llm(self):
        """End-to-end through polish() with a stub LLM emitting a 2-paragraph scenario."""
        client = AsyncMock(spec=LLMClient)
        client.generate = AsyncMock(
            return_value=(
                "[User Login]\n"
                "Head to the login page.\n\n"
                "Then enter your credentials.\n\n"
                "[Dashboard Navigation]\n"
                "Now explore the dashboard."
            )
        )
        humanizer = NarrationHumanizer(client)
        parts = [
            {"scenario_name": "User Login", "text": "orig login"},
            {"scenario_name": "Dashboard Navigation", "text": "orig dashboard"},
        ]
        result = await humanizer.polish(parts, "tutorial")

        assert len(result) == 2
        assert result[0]["text"] == "Head to the login page. Then enter your credentials."
        assert result[1]["text"] == "Now explore the dashboard."


# ---------------------------------------------------------------------------
# polish — integration tests with mock LLM
# ---------------------------------------------------------------------------


class TestNarrationHumanizer:
    """Tests for NarrationHumanizer LLM polish pass."""

    async def test_polish_sends_prompt_containing_all_scenario_names(
        self, humanizer, mock_llm_client, sample_narration_parts
    ):
        """polish should include all scenario names in the LLM prompt."""
        mock_llm_client.generate.return_value = (
            "[User Login]\nPolished.\n\n[Dashboard Navigation]\nPolished."
        )

        await humanizer.polish(sample_narration_parts, "tutorial")

        call_args = mock_llm_client.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "User Login" in prompt
        assert "Dashboard Navigation" in prompt
        assert "tutorial" in prompt.lower() or "Tutorial" in prompt

    async def test_polish_preserves_scenario_names(
        self, humanizer, mock_llm_client, sample_narration_parts
    ):
        """polish should preserve scenario_name in output parts."""
        mock_llm_client.generate.return_value = (
            "[User Login]\nPolished login text.\n\n[Dashboard Navigation]\nPolished dashboard text."
        )

        result = await humanizer.polish(sample_narration_parts, "tutorial")
        assert result[0]["scenario_name"] == "User Login"
        assert result[1]["scenario_name"] == "Dashboard Navigation"

    async def test_polish_updates_text(self, humanizer, mock_llm_client, sample_narration_parts):
        """polish should update text content from LLM response."""
        mock_llm_client.generate.return_value = (
            "[User Login]\nSmoothed login narration.\n\n"
            "[Dashboard Navigation]\nSmoothed dashboard narration."
        )

        result = await humanizer.polish(sample_narration_parts, "tutorial")
        assert result[0]["text"] == "Smoothed login narration."
        assert result[1]["text"] == "Smoothed dashboard narration."

    async def test_polish_with_context(self, humanizer, mock_llm_client, sample_narration_parts):
        """polish with context should include it in the LLM prompt."""
        mock_llm_client.generate.return_value = (
            "[User Login]\nPolished text.\n\n[Dashboard Navigation]\nPolished text."
        )

        await humanizer.polish(sample_narration_parts, "tutorial", context="An e-commerce platform")

        call_args = mock_llm_client.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "e-commerce" in prompt

    async def test_polish_fallback_on_parse_failure(
        self, humanizer, mock_llm_client, sample_narration_parts
    ):
        """If the LLM echo is not faithful (fewer sections than parts), ALL originals stay."""
        mock_llm_client.generate.return_value = "[User Login]\nOnly one section polished."

        result = await humanizer.polish(sample_narration_parts, "tutorial")
        assert len(result) == 2
        # Not a faithful [Name]-per-part echo -> every part keeps its original text.
        assert result[0]["text"] == sample_narration_parts[0]["text"]
        assert result[1]["text"] == sample_narration_parts[1]["text"]

    async def test_polish_uses_tone_system_prompt(
        self, humanizer, mock_llm_client, sample_narration_parts
    ):
        """polish should use the tone preset's system prompt."""
        mock_llm_client.generate.return_value = (
            "[User Login]\nPolished.\n\n[Dashboard Navigation]\nPolished."
        )

        await humanizer.polish(sample_narration_parts, "demo")
        call_args = mock_llm_client.generate.call_args
        system_prompt = call_args.kwargs.get("system_prompt", "")
        assert "demo" in system_prompt.lower() or "product" in system_prompt.lower()

    async def test_polish_empty_parts(self, humanizer, mock_llm_client):
        """polish with empty parts list should return empty list."""
        mock_llm_client.generate.return_value = ""
        result = await humanizer.polish([], "tutorial")
        assert result == []

    async def test_polish_single_part(self, humanizer, mock_llm_client):
        """polish with a single part should work correctly."""
        mock_llm_client.generate.return_value = "[Test]\nSingle polished narration."

        parts = [{"scenario_name": "Test", "text": "Original text."}]
        result = await humanizer.polish(parts, "tutorial")
        assert len(result) == 1
        assert result[0]["text"] == "Single polished narration."


class TestTheNarrationPromptCarriesTheOrchestratorsWordBudget:
    """5c — the orchestrator now sends the word budget the script must FILL.

    The deterministic trim downstream can only SHORTEN a line, so a script that
    under-runs its ordered duration can be fixed HERE and nowhere else. Both keys
    are optional by contract: a payload from an orchestrator that has not deployed
    yet must produce the byte-identical prompt it produces today.
    """

    async def test_the_prompt_states_the_per_section_budget(
        self, humanizer, mock_llm_client, sample_narration_parts
    ):
        mock_llm_client.generate.return_value = ""
        await humanizer.polish(sample_narration_parts, "tutorial", word_budgets=[40, 25])

        prompt = mock_llm_client.generate.call_args.args[0]
        assert "LENGTH BUDGET" in prompt
        assert "65 words in total" in prompt
        assert "[User Login]: 40 words" in prompt
        assert "[Dashboard Navigation]: 25 words" in prompt
        # It must read as a length to WRITE TO, not a ceiling to stay under — a
        # "keep it under N words" prompt makes the under-run defect worse.
        assert "expand" in prompt

    async def test_the_prompt_is_unchanged_when_the_keys_are_absent(
        self, humanizer, mock_llm_client, sample_narration_parts
    ):
        """BACKWARD COMPATIBILITY, asserted by EQUALITY rather than by absence:
        an `assert "LENGTH BUDGET" not in prompt` would still pass if the no-budget
        prompt had drifted in some other way."""
        mock_llm_client.generate.return_value = ""
        await humanizer.polish(sample_narration_parts, "tutorial")
        without_kwarg = mock_llm_client.generate.call_args.args[0]

        mock_llm_client.generate.reset_mock()
        await humanizer.polish(sample_narration_parts, "tutorial", word_budgets=None)
        explicit_none = mock_llm_client.generate.call_args.args[0]

        assert without_kwarg == explicit_none
        assert "LENGTH BUDGET" not in without_kwarg
        # ...and it is genuinely the pre-change prompt: the budget block is the
        # ONLY thing inserted, so the surrounding text must still join directly.
        assert "Style:" in without_kwarg
        assert without_kwarg.count("\n\nNarration script:\n") == 1
