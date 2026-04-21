"""
tests/unit/agents/test_a35.py

Unit tests for the A3.5 Analogy Agent.
Follows the same pattern as the existing test_a1.py … test_a6.py suite.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.a35_analogy_agent import (
    A35AnalogyAgent,
    _extract_contested_positions,
    _parse_analogies,
    _mark_hook_candidate,
)
from src.pipeline.state import Analogy, AgentStatus, PipelineState, TokenUsage
from src.utils.errors import PipelineWarning


# ---------------------------------------------------------------------------
# Shared fixtures
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

## Evidence weight summary
Medium — longitudinal studies are scarce.

## The unresolved question
Whether documented engagement deficits are causal or correlational.
"""

A3_OUTPUT_NO_CONTESTED = """
## Consensus zone
Remote work is broadly positive.

## Contested zone

## Outlier positions
None notable.
"""

ANALOGY_JSON = json.dumps({
    "analogies": [
        {
            "position": "Fully remote workers face invisible proximity bias in promotion decisions.",
            "title": "The Factory Floor Visibility Effect",
            "domain": "Industrial History",
            "structural_parallel": (
                "In early 20th-century factories, workers physically close to the foreman "
                "received more promotions regardless of output quality."
            ),
            "maps_to": (
                "Remote workers are invisible to the default heuristics managers use. "
                "Visibility substitutes for unmeasurable output quality."
            ),
            "breaks_down": (
                "Factory workers had no async record of their work; remote workers leave "
                "documented trails that could replace visibility as a signal."
            ),
        },
        {
            "position": "Asynchronous communication erodes tacit knowledge transfer.",
            "title": "The Apprentice Ship Problem",
            "domain": "Medieval Craft Guilds",
            "structural_parallel": (
                "Guild apprentices could not learn glassblowing from written manuals — "
                "the knowledge lived in physical co-presence and imitation."
            ),
            "maps_to": (
                "The tacit dimension — reading a room, knowing which battles to pick — "
                "cannot survive async channels for the same reason."
            ),
            "breaks_down": (
                "Guild knowledge was never digitised; remote tools like Loom create partial "
                "tacit-to-explicit conversion that guilds lacked."
            ),
        },
    ]
})

DUMMY_USAGE = TokenUsage(input_tokens=100, output_tokens=200, total_tokens=300)


def make_state(a3_output: str = A3_OUTPUT_GOOD) -> MagicMock:
    state = MagicMock(spec=PipelineState)
    state.topic = "Does remote work hurt career progression?"
    state.agents = {
        "A3": MagicMock(output=a3_output),
        "A35": MagicMock(output=None, warnings=[], input=""),
    }
    state.analogies = None
    return state


def make_agent() -> A35AnalogyAgent:
    mock_llm = MagicMock()
    mock_llm.call = AsyncMock(return_value=(ANALOGY_JSON, DUMMY_USAGE))
    return A35AnalogyAgent(llm_client=mock_llm)


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
        analogies = _parse_analogies(ANALOGY_JSON)
        assert len(analogies) == 2
        assert analogies[0].title == "The Factory Floor Visibility Effect"
        assert analogies[0].domain == "Industrial History"

    def test_returns_empty_on_malformed_json(self):
        assert _parse_analogies("not json at all") == []

    def test_returns_empty_on_missing_analogies_key(self):
        assert _parse_analogies('{"result": []}') == []

    def test_skips_entries_with_missing_keys(self):
        bad = json.dumps({
            "analogies": [
                {"title": "Missing fields"},
                {
                    "position": "p", "title": "t", "domain": "d",
                    "structural_parallel": "sp", "maps_to": "mt", "breaks_down": "bd",
                },
            ]
        })
        analogies = _parse_analogies(bad)
        assert len(analogies) == 1

    def test_strips_markdown_fences(self):
        fenced = f"```json\n{ANALOGY_JSON}\n```"
        assert len(_parse_analogies(fenced)) == 2

    def test_hook_candidate_defaults_false(self):
        for a in _parse_analogies(ANALOGY_JSON):
            assert a.hook_candidate is False


# ---------------------------------------------------------------------------
# _mark_hook_candidate
# ---------------------------------------------------------------------------

class TestMarkHookCandidate:
    def test_prefers_industrial_history(self):
        analogies = [
            Analogy(position="p", title="t1", domain="Biology",
                    structural_parallel="sp", maps_to="mt", breaks_down="bd"),
            Analogy(position="p", title="t2", domain="Industrial History",
                    structural_parallel="sp", maps_to="mt", breaks_down="bd"),
        ]
        _mark_hook_candidate(analogies)
        assert analogies[1].hook_candidate is True
        assert analogies[0].hook_candidate is False

    def test_falls_back_to_first_if_no_preferred_domain(self):
        analogies = [
            Analogy(position="p", title="t1", domain="Linguistics",
                    structural_parallel="sp", maps_to="mt", breaks_down="bd"),
            Analogy(position="p", title="t2", domain="Philosophy",
                    structural_parallel="sp", maps_to="mt", breaks_down="bd"),
        ]
        _mark_hook_candidate(analogies)
        assert analogies[0].hook_candidate is True

    def test_only_one_hook_candidate(self):
        analogies = [
            Analogy(position="p", title="t1", domain="Economics",
                    structural_parallel="sp", maps_to="mt", breaks_down="bd"),
            Analogy(position="p", title="t2", domain="Technology",
                    structural_parallel="sp", maps_to="mt", breaks_down="bd"),
        ]
        _mark_hook_candidate(analogies)
        assert sum(1 for a in analogies if a.hook_candidate) == 1

    def test_empty_list_does_not_raise(self):
        _mark_hook_candidate([])


# ---------------------------------------------------------------------------
# A35AnalogyAgent.run
# ---------------------------------------------------------------------------

class TestAnalogyAgentRun:
    @pytest.mark.asyncio
    async def test_happy_path_populates_state_analogies(self):
        agent = make_agent()
        state = make_state()
        await agent.run(state, signal=MagicMock())

        assert state.analogies is not None
        assert len(state.analogies) == 2
        assert state.agents["A35"].output == ANALOGY_JSON

    @pytest.mark.asyncio
    async def test_happy_path_makes_one_llm_call(self):
        agent = make_agent()
        state = make_state()
        await agent.run(state, signal=MagicMock())

        agent._llm.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_a3_output_adds_warning_and_skips_llm(self):
        agent = make_agent()
        state = make_state()
        state.agents["A3"].output = None
        await agent.run(state, signal=MagicMock())

        agent._llm.call.assert_not_called()
        assert PipelineWarning.NO_A3_OUTPUT in state.agents["A35"].warnings

    @pytest.mark.asyncio
    async def test_no_a3_output_sets_empty_analogies(self):
        agent = make_agent()
        state = make_state()
        state.agents["A3"].output = None
        await agent.run(state, signal=MagicMock())

        assert state.analogies == []

    @pytest.mark.asyncio
    async def test_no_contested_positions_sets_empty_analogies(self):
        agent = make_agent()
        state = make_state(a3_output=A3_OUTPUT_NO_CONTESTED)
        await agent.run(state, signal=MagicMock())

        agent._llm.call.assert_not_called()
        assert state.analogies == []
        assert PipelineWarning.NO_CONTESTED_POSITIONS in state.agents["A35"].warnings

    @pytest.mark.asyncio
    async def test_malformed_llm_response_sets_empty_analogies(self):
        agent = make_agent()
        agent._llm.call = AsyncMock(return_value=("not valid json", DUMMY_USAGE))
        state = make_state()
        await agent.run(state, signal=MagicMock())

        assert state.analogies == []

    @pytest.mark.asyncio
    async def test_malformed_response_does_not_raise(self):
        agent = make_agent()
        agent._llm.call = AsyncMock(return_value=("not valid json", DUMMY_USAGE))
        state = make_state()
        # Should not raise — graceful degradation
        await agent.run(state, signal=MagicMock())

    @pytest.mark.asyncio
    async def test_hook_candidate_is_marked(self):
        agent = make_agent()
        state = make_state()
        await agent.run(state, signal=MagicMock())

        assert sum(1 for a in state.analogies if a.hook_candidate) == 1

    @pytest.mark.asyncio
    async def test_token_usage_stored(self):
        agent = make_agent()
        state = make_state()
        await agent.run(state, signal=MagicMock())

        assert state.agents["A35"].token_usage == DUMMY_USAGE

    @pytest.mark.asyncio
    async def test_prompt_contains_contested_positions(self):
        agent = make_agent()
        state = make_state()
        await agent.run(state, signal=MagicMock())

        call_args = agent._llm.call.call_args
        prompt = call_args[0][1]  # positional: (agent_id, prompt, ...)
        assert "proximity bias" in prompt

    @pytest.mark.asyncio
    async def test_prompt_contains_topic(self):
        agent = make_agent()
        state = make_state()
        await agent.run(state, signal=MagicMock())

        prompt = agent._llm.call.call_args[0][1]
        assert "remote work" in prompt.lower()
