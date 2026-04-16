"""Unit tests for A5EvidenceJudge (spec §12 A5 section)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.a5_evidence_judge import A5EvidenceJudge
from src.pipeline.state import AgentStatus, PipelineState, TokenUsage
from src.utils.errors import AgentValidationError, PipelineWarning

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

MOCK_TOKEN_USAGE = TokenUsage(input_tokens=200, output_tokens=500, total_tokens=700)

A1_OUTPUT = """\
1. URL: https://example.com/study Type: Academic Recency: 2024 Core claim: Email ROI is strong
2. URL: https://example.com/blog Type: Industry Recency: 2023 Core claim: Personalisation boosts rates
3. URL: https://example.com/report Type: Analyst Recency: 2024 Core claim: AI changes workflows
"""

A4_OUTPUT = """\
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

A5_HAPPY_OUTPUT = """\
## Verdict
Email marketing remains the highest-ROI channel for B2B when properly personalised.

## Three strongest reasons
1. Consistent ROI data across multiple studies
2. Direct channel ownership vs algorithm dependency
3. Personalisation technology now mature enough for SMBs

## Honest concession
Privacy regulations do create friction and compliance costs.

## The angle
Position email as the owned channel that compounds over time.

## What to avoid
Avoid generic batch-and-blast tactics that inflate unsubscribe rates.
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
    s.agents["A3"].output = "A3 landscape output"
    s.agents["A4"].status = AgentStatus.COMPLETED
    s.agents["A4"].output = A4_OUTPUT
    return s


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call = AsyncMock(return_value=(A5_HAPPY_OUTPUT, MOCK_TOKEN_USAGE))
    return llm


@pytest.fixture
def agent(mock_llm):
    return A5EvidenceJudge(llm_client=mock_llm)


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_includes_a4_output(agent, state):
    prompt = agent.build_prompt(state)
    assert A4_OUTPUT in prompt


def test_build_prompt_includes_a1_source_list(agent, state):
    prompt = agent.build_prompt(state)
    assert A1_OUTPUT in prompt


def test_build_prompt_includes_shallow_claims_caveat_when_warning_present(agent, state):
    state.agents["A2"].warnings.append(PipelineWarning.SHALLOW_CLAIMS)
    prompt = agent.build_prompt(state)
    assert "source material for this topic was thin" in prompt


def test_build_prompt_no_caveat_when_shallow_claims_absent(agent, state):
    # Ensure A2 has no SHALLOW_CLAIMS warning
    state.agents["A2"].warnings = []
    prompt = agent.build_prompt(state)
    assert "source material for this topic was thin" not in prompt


# ---------------------------------------------------------------------------
# validate_output — happy path
# ---------------------------------------------------------------------------

def test_validate_output_passes_on_happy_path(agent, state):
    agent.validate_output(A5_HAPPY_OUTPUT, state)  # should not raise


# ---------------------------------------------------------------------------
# validate_output — missing headings raise AgentValidationError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_heading", [
    "## Verdict",
    "## Three strongest reasons",
    "## Honest concession",
    "## The angle",
    "## What to avoid",
])
def test_validate_output_raises_when_heading_missing(agent, state, missing_heading):
    output_without_heading = A5_HAPPY_OUTPUT.replace(missing_heading, "## Replaced")
    with pytest.raises(AgentValidationError) as exc_info:
        agent.validate_output(output_without_heading, state)
    assert exc_info.value.agent_id == "A5"


# ---------------------------------------------------------------------------
# Hedge phrase detection — each hedge triggers re-prompt (CP-08)
# ---------------------------------------------------------------------------

def _output_with_hedge(phrase: str) -> str:
    """Replace the verdict sentence with one containing the given hedge phrase."""
    return A5_HAPPY_OUTPUT.replace(
        "Email marketing remains the highest-ROI channel for B2B when properly personalised.",
        f"Email marketing is {phrase} the right channel depending on context.",
    )


@pytest.mark.parametrize("hedge_phrase", [
    "it depends",
    "on the one hand",
    "nuanced",
    "both sides",
    "complex picture",
])
async def test_hedge_phrase_triggers_reprompt(mock_llm, state, hedge_phrase):
    hedged_output = _output_with_hedge(hedge_phrase)
    mock_llm.call = AsyncMock(
        side_effect=[
            (hedged_output, MOCK_TOKEN_USAGE),
            (A5_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A5EvidenceJudge(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 2


async def test_second_hedge_in_reprompt_raises_agent_validation_error(mock_llm, state):
    """If the re-prompted response still hedges, A5 must raise AgentValidationError."""
    hedged_output = _output_with_hedge("it depends")
    mock_llm.call = AsyncMock(
        side_effect=[
            (hedged_output, MOCK_TOKEN_USAGE),
            (hedged_output, MOCK_TOKEN_USAGE),  # still hedges after reprompt
        ]
    )
    agent = A5EvidenceJudge(llm_client=mock_llm)
    with pytest.raises(AgentValidationError) as exc_info:
        await agent.run(state)
    assert exc_info.value.agent_id == "A5"
    # Must be exactly 2 calls — no third attempt
    assert mock_llm.call.call_count == 2


# ---------------------------------------------------------------------------
# CP-09: missing section triggers re-prompt
# ---------------------------------------------------------------------------

async def test_cp09_missing_section_triggers_reprompt(mock_llm, state):
    incomplete_output = A5_HAPPY_OUTPUT.replace("## What to avoid\n", "")
    mock_llm.call = AsyncMock(
        side_effect=[
            (incomplete_output, MOCK_TOKEN_USAGE),
            (A5_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A5EvidenceJudge(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 2


# ---------------------------------------------------------------------------
# run() — sets output, status=COMPLETED, token_usage
# ---------------------------------------------------------------------------

async def test_run_sets_output(agent, state):
    await agent.run(state)
    assert state.agents["A5"].output == A5_HAPPY_OUTPUT


async def test_run_sets_status_completed(agent, state):
    await agent.run(state)
    assert state.agents["A5"].status == AgentStatus.COMPLETED


async def test_run_sets_token_usage(agent, state):
    await agent.run(state)
    assert state.agents["A5"].token_usage == MOCK_TOKEN_USAGE


async def test_run_returns_output_string(agent, state):
    result = await agent.run(state)
    assert result == A5_HAPPY_OUTPUT
