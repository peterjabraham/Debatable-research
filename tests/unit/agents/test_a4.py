"""Unit tests for A4DevilsAdvocate (spec §12 A4 section)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.a4_devils_advocate import A4DevilsAdvocate
from src.pipeline.state import AgentStatus, PipelineState, TokenUsage
from src.utils.errors import AgentValidationError, PipelineWarning

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

MOCK_TOKEN_USAGE = TokenUsage(input_tokens=300, output_tokens=600, total_tokens=900)

# A1 output with 3 numbered sources — source name fragments appear in A4 happy output
A1_OUTPUT = """\
1. URL: https://example.com/study Type: Academic Recency: 2024 Core claim: Email ROI is strong
2. URL: https://example.com/blog Type: Industry Recency: 2023 Core claim: Personalisation boosts rates
3. URL: https://example.com/report Type: Analyst Recency: 2024 Core claim: AI changes workflows
"""

# A3 output with exactly 3 numbered contested positions (inside ## Contested zone)
A3_OUTPUT_3_POSITIONS = """\
## Consensus zone
- Email remains relevant for B2B marketing

## Contested zone
1. AI personalisation is overhyped
2. Privacy regulations make email harder
3. Generic email is dying

## Outlier positions
- Email is dead for Gen Z

## Evidence weight summary
Most evidence supports moderate optimism.

## The unresolved question
Can AI personalisation overcome privacy friction?
"""

# A3 output with 5 numbered contested positions — triggers position cap
A3_OUTPUT_5_POSITIONS = """\
## Consensus zone
- Email remains relevant for B2B marketing

## Contested zone
1. AI personalisation is overhyped
2. Privacy regulations make email harder
3. Generic email is dying
4. SMBs cannot compete in email
5. Automation kills authenticity

## Outlier positions
- Email is dead for Gen Z

## Evidence weight summary
Mixed.

## The unresolved question
Can AI personalisation overcome privacy friction?
"""

# Happy-path A4 output: 3 blocks, each citing a source fragment from A1
A4_HAPPY_OUTPUT = """\
Position: AI personalisation is overhyped
Case: 1. Studies show marginal gains. 2. URL: https://example.com/study shows ROI concerns. 3. Implementation costs are high.
Hardest objection: Some companies do see strong results.
Response: Results are context-dependent and mostly for large enterprises.

Position: Privacy regulations make email harder
Case: 1. GDPR compliance overhead. 2. URL: https://example.com/blog cites open rate drops. 3. Consent fatigue is real.
Hardest objection: Email still has highest ROI per channel.
Response: Short-term friction may yield long-term trust.

Position: Generic email is dying
Case: 1. Unsubscribe rates rising. 2. URL: https://example.com/report data confirms. 3. Inbox competition is intense.
Hardest objection: Total email volume keeps growing.
Response: Quality over quantity is now mandatory.
"""

# A4 output where blocks do NOT cite any source from A1 (to trigger CP-07)
A4_NO_CITATION_OUTPUT = """\
Position: AI personalisation is overhyped
Case: 1. Studies show marginal gains. 2. Implementation costs are high. 3. Adoption is slow.
Hardest objection: Some companies do see strong results.
Response: Results are context-dependent.

Position: Privacy regulations make email harder
Case: 1. GDPR compliance overhead. 2. Consent fatigue is real. 3. Legal risk is high.
Hardest objection: Email still has highest ROI per channel.
Response: Short-term friction may yield long-term trust.

Position: Generic email is dying
Case: 1. Unsubscribe rates rising. 2. Inbox competition is intense. 3. Open rates declining.
Hardest objection: Total email volume keeps growing.
Response: Quality over quantity is now mandatory.
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    s = PipelineState(topic="Email marketing", audience="Marketers", tone="Direct")
    s.agents["A1"].status = AgentStatus.COMPLETED
    s.agents["A1"].output = A1_OUTPUT
    s.agents["A2"].status = AgentStatus.COMPLETED
    s.agents["A2"].output = "Some claim output"
    s.agents["A3"].status = AgentStatus.COMPLETED
    s.agents["A3"].output = A3_OUTPUT_3_POSITIONS
    return s


@pytest.fixture
def state_5_positions(state):
    state.agents["A3"].output = A3_OUTPUT_5_POSITIONS
    return state


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call = AsyncMock(return_value=(A4_HAPPY_OUTPUT, MOCK_TOKEN_USAGE))
    return llm


@pytest.fixture
def agent(mock_llm):
    return A4DevilsAdvocate(llm_client=mock_llm)


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_includes_contested_positions_from_a3(agent, state):
    prompt = agent.build_prompt(state)
    assert "AI personalisation is overhyped" in prompt
    assert "Privacy regulations make email harder" in prompt
    assert "Generic email is dying" in prompt


def test_build_prompt_includes_a1_source_list(agent, state):
    prompt = agent.build_prompt(state)
    assert A1_OUTPUT in prompt


def test_build_prompt_contains_steelman_instructions(agent, state):
    prompt = agent.build_prompt(state)
    assert "steelman" in prompt.lower()


# ---------------------------------------------------------------------------
# Position cap at 3 (MAX_POSITIONS)
# ---------------------------------------------------------------------------

async def test_positions_capped_at_3_when_a3_has_5(mock_llm, state_5_positions):
    agent = A4DevilsAdvocate(llm_client=mock_llm)
    await agent.run(state_5_positions)
    # The prompt passed to the LLM must list exactly 3 positions
    prompt_used = mock_llm.call.call_args_list[0][0][1]
    # Position 4 and 5 must not be in the prompt
    assert "SMBs cannot compete in email" not in prompt_used
    assert "Automation kills authenticity" not in prompt_used
    # Positions 1–3 must be present
    assert "AI personalisation is overhyped" in prompt_used
    assert "Privacy regulations make email harder" in prompt_used
    assert "Generic email is dying" in prompt_used


async def test_truncated_positions_warning_added_when_cap_applied(mock_llm, state_5_positions):
    agent = A4DevilsAdvocate(llm_client=mock_llm)
    await agent.run(state_5_positions)
    assert PipelineWarning.TRUNCATED_POSITIONS in state_5_positions.agents["A4"].warnings


async def test_no_truncated_warning_when_positions_within_limit(mock_llm, state):
    agent = A4DevilsAdvocate(llm_client=mock_llm)
    await agent.run(state)
    assert PipelineWarning.TRUNCATED_POSITIONS not in state.agents["A4"].warnings


async def test_dropped_positions_are_logged(mock_llm, state_5_positions):
    agent = A4DevilsAdvocate(llm_client=mock_llm)
    with patch("src.agents.a4_devils_advocate.logger") as mock_logger:
        await agent.run(state_5_positions)
    mock_logger.warning.assert_called_once()
    warning_args = mock_logger.warning.call_args
    # The log message or args must reference the dropped positions
    log_str = str(warning_args)
    assert "SMBs cannot compete in email" in log_str or "Automation kills authenticity" in log_str


# ---------------------------------------------------------------------------
# CP-07: re-prompt triggered when steelman block lacks source citation
# ---------------------------------------------------------------------------

async def test_cp07_reprompt_triggered_when_block_lacks_source_citation(mock_llm, state):
    """First LLM call returns output with no citations; second returns valid output."""
    mock_llm.call = AsyncMock(
        side_effect=[
            (A4_NO_CITATION_OUTPUT, MOCK_TOKEN_USAGE),
            (A4_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A4DevilsAdvocate(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 2


async def test_cp07_reprompt_not_triggered_when_citations_present(mock_llm, state):
    """Happy-path output has citations — only one LLM call should occur."""
    agent = A4DevilsAdvocate(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 1


async def test_cp07_reprompt_not_triggered_more_than_once(mock_llm, state):
    """Even if second output still lacks citations, agent must not call LLM a third time.
    It raises AgentValidationError instead."""
    mock_llm.call = AsyncMock(
        side_effect=[
            (A4_NO_CITATION_OUTPUT, MOCK_TOKEN_USAGE),
            (A4_NO_CITATION_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A4DevilsAdvocate(llm_client=mock_llm)
    with pytest.raises(AgentValidationError) as exc_info:
        await agent.run(state)
    assert mock_llm.call.call_count == 2
    assert exc_info.value.agent_id == "A4"


async def test_cp07_reprompt_contains_source_citation_instruction(mock_llm, state):
    """The re-prompt must instruct the LLM to cite named sources."""
    mock_llm.call = AsyncMock(
        side_effect=[
            (A4_NO_CITATION_OUTPUT, MOCK_TOKEN_USAGE),
            (A4_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A4DevilsAdvocate(llm_client=mock_llm)
    await agent.run(state)
    reprompt_text = mock_llm.call.call_args_list[1][0][1]
    assert "source" in reprompt_text.lower() or "cite" in reprompt_text.lower()


# ---------------------------------------------------------------------------
# run() — sets output, status=COMPLETED, token_usage
# ---------------------------------------------------------------------------

async def test_run_sets_output(agent, state):
    await agent.run(state)
    assert state.agents["A4"].output == A4_HAPPY_OUTPUT


async def test_run_sets_status_completed(agent, state):
    await agent.run(state)
    assert state.agents["A4"].status == AgentStatus.COMPLETED


async def test_run_sets_token_usage(agent, state):
    await agent.run(state)
    assert state.agents["A4"].token_usage == MOCK_TOKEN_USAGE


async def test_run_returns_output_string(agent, state):
    result = await agent.run(state)
    assert result == A4_HAPPY_OUTPUT
