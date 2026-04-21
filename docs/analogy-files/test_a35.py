"""
tests/unit/agents/test_a35.py

Unit tests for the A3.5 Analogy Agent.
Follows the same pattern as your existing test_a1.py … test_a6.py.
"""

from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.a35_analogy_agent import (
    AnalogyAgent,
    Analogy,
    _extract_contested_positions,
    _parse_analogies,
    _mark_hook_candidate,
)
from src.utils.errors import PipelineWarning


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

A3_OUTPUT_GOOD = """
## Consensus zone
Remote work is now broadly accepted for knowledge workers.

## Contested zone
- Fully remote workers face invisible proximity bias in promotion decisions.
- Asynchronous communication erodes the tacit knowledge transfer that happens in offices.
- Managers cannot accurately assess productivity without physical presence.

## Outlier positions
A small body of research suggests outcomes depend entirely on role type.

## Evidence weight
Medium — longitudinal studies are scarce.

## Unresolved question
Whether documented engagement deficits are causal or correlational.
"""

A3_OUTPUT_NO_CONTESTED = """
## Consensus zone
Remote work is broadly positive.

## Contested zone

## Outlier positions
None notable.
"""

ANALOGY_JSON_RESPONSE = json.dumps({
    "analogies": [
        {
            "position": "Fully remote workers face invisible proximity bias in promotion decisions.",
            "title": "The Factory Floor Visibility Effect",
            "domain": "Industrial History",
            "structural_parallel": (
                "In early 20th-century factories, workers physically close to the foreman "
                "received more promotions regardless of output quality. Visibility acted as a "
                "proxy for effort because effort could not otherwise be measured."
            ),
            "maps_to": (
                "Remote workers are invisible to the default heuristics managers use. "
                "The mechanism — visibility substituting for unmeasurable output quality — "
                "is structurally identical to the factory floor dynamic."
            ),
            "breaks_down": (
                "Factory workers had no async record of their work; remote workers leave "
                "documented trails that could in principle replace visibility as a signal."
            ),
        },
        {
            "position": "Asynchronous communication erodes tacit knowledge transfer.",
            "title": "The Apprentice Ship Problem",
            "domain": "Medieval Craft Guilds",
            "structural_parallel": (
                "Guild apprentices could not learn glassblowing or goldsmithing from written "
                "manuals — the knowledge lived in physical co-presence and imitation. "
                "Codifying the tacit killed the transfer."
            ),
            "maps_to": (
                "The tacit dimension Polanyi describes — knowing which battles to pick with a "
                "client, reading a room — cannot survive async channels for the same reason "
                "glassblowing couldn't survive a manual."
            ),
            "breaks_down": (
                "Guild knowledge was never digitised; remote work tools like Loom and "
                "recorded calls create partial tacit-to-explicit conversion that guilds lacked."
            ),
        },
    ]
})


def make_state(a3_output: str = A3_OUTPUT_GOOD) -> MagicMock:
    state = MagicMock()
    state.topic = "Does remote work hurt career progression?"
    state.agents = {
        "A3": MagicMock(output=a3_output),
        "A35": MagicMock(output=None, warnings=[]),
    }
    state.analogies = None
    return state


# ---------------------------------------------------------------------------
# _extract_contested_positions
# ---------------------------------------------------------------------------

class TestExtractContestedPositions:
    def test_extracts_bullet_positions(self):
        positions = _extract_contested_positions(A3_OUTPUT_GOOD)
        assert len(positions) == 3
        assert "proximity bias" in positions[0]

    def test_returns_empty_on_missing_section(self):
        positions = _extract_contested_positions("## Consensus zone\nEveryone agrees.")
        assert positions == []

    def test_returns_empty_on_empty_contested_zone(self):
        positions = _extract_contested_positions(A3_OUTPUT_NO_CONTESTED)
        assert positions == []

    def test_caps_at_three(self):
        output = A3_OUTPUT_GOOD + "\n- A fourth contested point.\n- A fifth.\n"
        positions = _extract_contested_positions(output)
        assert len(positions) <= 3

    def test_strips_whitespace(self):
        positions = _extract_contested_positions(A3_OUTPUT_GOOD)
        for p in positions:
            assert p == p.strip()


# ---------------------------------------------------------------------------
# _parse_analogies
# ---------------------------------------------------------------------------

class TestParseAnalogies:
    def test_parses_valid_json(self):
        analogies = _parse_analogies(ANALOGY_JSON_RESPONSE, [])
        assert len(analogies) == 2
        assert analogies[0].title == "The Factory Floor Visibility Effect"
        assert analogies[0].domain == "Industrial History"

    def test_returns_empty_on_malformed_json(self):
        analogies = _parse_analogies("not json at all", [])
        assert analogies == []

    def test_returns_empty_on_missing_analogies_key(self):
        analogies = _parse_analogies('{"result": []}', [])
        assert analogies == []

    def test_skips_entries_with_missing_keys(self):
        bad_json = json.dumps({
            "analogies": [
                {"title": "Missing fields"},  # no position, domain, etc.
                {
                    "position": "p",
                    "title": "t",
                    "domain": "d",
                    "structural_parallel": "sp",
                    "maps_to": "mt",
                    "breaks_down": "bd",
                },
            ]
        })
        analogies = _parse_analogies(bad_json, [])
        assert len(analogies) == 1

    def test_strips_markdown_fences(self):
        fenced = f"```json\n{ANALOGY_JSON_RESPONSE}\n```"
        analogies = _parse_analogies(fenced, [])
        assert len(analogies) == 2

    def test_hook_candidate_defaults_false(self):
        analogies = _parse_analogies(ANALOGY_JSON_RESPONSE, [])
        for a in analogies:
            assert a.hook_candidate is False


# ---------------------------------------------------------------------------
# _mark_hook_candidate
# ---------------------------------------------------------------------------

class TestMarkHookCandidate:
    def test_prefers_industrial_history(self):
        analogies = [
            Analogy("p", "t1", "Biology", "sp", "mt", "bd"),
            Analogy("p", "t2", "Industrial History", "sp", "mt", "bd"),
        ]
        _mark_hook_candidate(analogies)
        assert analogies[1].hook_candidate is True
        assert analogies[0].hook_candidate is False

    def test_falls_back_to_first_if_no_preferred_domain(self):
        analogies = [
            Analogy("p", "t1", "Linguistics", "sp", "mt", "bd"),
            Analogy("p", "t2", "Philosophy", "sp", "mt", "bd"),
        ]
        _mark_hook_candidate(analogies)
        assert analogies[0].hook_candidate is True

    def test_only_one_hook_candidate(self):
        analogies = [
            Analogy("p", "t1", "Economics", "sp", "mt", "bd"),
            Analogy("p", "t2", "Technology", "sp", "mt", "bd"),
        ]
        _mark_hook_candidate(analogies)
        marked = [a for a in analogies if a.hook_candidate]
        assert len(marked) == 1

    def test_empty_list_does_not_raise(self):
        _mark_hook_candidate([])  # should not raise


# ---------------------------------------------------------------------------
# AnalogyAgent.run
# ---------------------------------------------------------------------------

class TestAnalogyAgentRun:
    @pytest.mark.asyncio
    async def test_happy_path_populates_state_analogies(self):
        agent = AnalogyAgent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value=ANALOGY_JSON_RESPONSE)

        state = make_state()
        await agent.run(state, signal=MagicMock())

        assert state.analogies is not None
        assert len(state.analogies) == 2
        assert state.agents["A35"].output == ANALOGY_JSON_RESPONSE

    @pytest.mark.asyncio
    async def test_no_a3_output_adds_warning_and_returns(self):
        agent = AnalogyAgent()
        agent.llm = AsyncMock()

        state = make_state()
        state.agents["A3"].output = None
        await agent.run(state, signal=MagicMock())

        agent.llm.complete.assert_not_called()
        assert PipelineWarning.NO_A3_OUTPUT in state.agents["A35"].warnings

    @pytest.mark.asyncio
    async def test_no_contested_positions_sets_empty_analogies(self):
        agent = AnalogyAgent()
        agent.llm = AsyncMock()

        state = make_state(a3_output=A3_OUTPUT_NO_CONTESTED)
        await agent.run(state, signal=MagicMock())

        agent.llm.complete.assert_not_called()
        assert state.analogies == []
        assert PipelineWarning.NO_CONTESTED_POSITIONS in state.agents["A35"].warnings

    @pytest.mark.asyncio
    async def test_malformed_llm_response_sets_empty_analogies(self):
        agent = AnalogyAgent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value="not valid json")

        state = make_state()
        await agent.run(state, signal=MagicMock())

        # Pipeline should not crash; analogies is empty list
        assert state.analogies == []

    @pytest.mark.asyncio
    async def test_hook_candidate_is_marked(self):
        agent = AnalogyAgent()
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value=ANALOGY_JSON_RESPONSE)

        state = make_state()
        await agent.run(state, signal=MagicMock())

        hook_count = sum(1 for a in state.analogies if a.hook_candidate)
        assert hook_count == 1
