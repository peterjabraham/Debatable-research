"""Unit tests for A3LandscapeMapper (spec §12)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.a3_landscape_mapper import A3LandscapeMapper
from src.pipeline.state import AgentStatus, PipelineState, TokenUsage
from src.utils.errors import AgentValidationError, PipelineWarning

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

A2_HAPPY_OUTPUT = """\
Core claim: Email marketing continues to deliver strong and measurable ROI for B2B organisations
Key evidence: Multiple peer-reviewed studies show consistent returns above industry average
Caveats: Results vary by sector and list quality
Implicit assumption: Marketers have access to quality segmented lists

Core claim: Personalisation significantly boosts open rates across targeted campaigns with proper data hygiene
Key evidence: Large-sample industry studies confirm 20–30 % uplift in open rates
Caveats: Requires robust CRM data to be effective
Implicit assumption: Recipients consent to personalised messaging

Core claim: AI-powered tools are fundamentally transforming how email content is written and workflows are managed
Key evidence: Major analyst firms report widespread adoption across enterprise marketing teams
Caveats: Quality control remains a challenge for AI-generated content
Implicit assumption: Teams have budget to invest in AI tooling
"""

A3_HAPPY_OUTPUT = """\
## Consensus zone
- Email remains relevant for B2B marketing

## Contested zone
1. AI personalisation is overhyped
2. Privacy regulations make email harder

## Outlier positions
- Email is dead for Gen Z

## Evidence weight summary
Most evidence supports moderate optimism.

## The unresolved question
Can AI personalisation overcome privacy friction?
"""

MOCK_TOKEN_USAGE = TokenUsage(input_tokens=200, output_tokens=400, total_tokens=600)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    s = PipelineState(topic="Email marketing", audience="Marketers", tone="Direct")
    # Pre-populate A1 and A2 as completed
    s.agents["A1"].status = AgentStatus.COMPLETED
    s.agents["A1"].output = (
        "1. URL: https://example.com/study Type: Academic Recency: 2024 "
        "Core claim: Email ROI is strong Credibility signal: Peer-reviewed\n"
        "2. URL: https://example.com/blog Type: Industry Recency: 2023 "
        "Core claim: Personalisation boosts rates Credibility signal: Large sample\n"
        "3. URL: https://example.com/report Type: Analyst Recency: 2024 "
        "Core claim: AI changes workflows Credibility signal: Major firm\n"
    )
    s.agents["A2"].status = AgentStatus.COMPLETED
    s.agents["A2"].output = A2_HAPPY_OUTPUT
    return s


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call = AsyncMock(return_value=(A3_HAPPY_OUTPUT, MOCK_TOKEN_USAGE))
    return llm


@pytest.fixture
def agent(mock_llm):
    return A3LandscapeMapper(llm_client=mock_llm)


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

def test_agent_id():
    llm = MagicMock()
    a = A3LandscapeMapper(llm_client=llm)
    assert a.id == "A3"


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_includes_a2_output(agent, state):
    prompt = agent.build_prompt(state)
    assert A2_HAPPY_OUTPUT in prompt


def test_build_prompt_includes_all_required_section_names(agent, state):
    prompt = agent.build_prompt(state)
    for heading in [
        "## Consensus zone",
        "## Contested zone",
        "## Outlier positions",
        "## Evidence weight summary",
        "## The unresolved question",
    ]:
        assert heading in prompt


# ---------------------------------------------------------------------------
# validate_output — happy path
# ---------------------------------------------------------------------------

def test_validate_output_passes_on_happy_path(agent, state):
    # All 5 headings present, 2 numbered contested positions → no exception
    agent.validate_output(A3_HAPPY_OUTPUT, state)


def test_validate_output_happy_path_no_no_contest_warning(agent, state):
    agent.validate_output(A3_HAPPY_OUTPUT, state)
    assert PipelineWarning.NO_CONTEST not in state.agents["A3"].warnings


# ---------------------------------------------------------------------------
# validate_output — missing heading raises AgentValidationError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_heading", [
    "## Consensus zone",
    "## Contested zone",
    "## Outlier positions",
    "## Evidence weight summary",
    "## The unresolved question",
])
def test_validate_output_raises_when_heading_missing(agent, state, missing_heading):
    output_without_heading = A3_HAPPY_OUTPUT.replace(missing_heading, "## Replaced section")
    with pytest.raises(AgentValidationError) as exc_info:
        agent.validate_output(output_without_heading, state)
    assert exc_info.value.agent_id == "A3"


# ---------------------------------------------------------------------------
# NO_CONTEST warning + AgentValidationError when < 2 contested positions
# ---------------------------------------------------------------------------

def test_no_contest_warning_added_when_fewer_than_2_contested_positions(agent, state):
    one_position_output = """\
## Consensus zone
- Email remains relevant for B2B marketing

## Contested zone
1. AI personalisation is overhyped

## Outlier positions
- Email is dead for Gen Z

## Evidence weight summary
Most evidence supports moderate optimism.

## The unresolved question
Can AI personalisation overcome privacy friction?
"""
    with pytest.raises(AgentValidationError):
        agent.validate_output(one_position_output, state)
    assert PipelineWarning.NO_CONTEST in state.agents["A3"].warnings


def test_no_contest_raises_agent_validation_error_when_fewer_than_2_positions(agent, state):
    zero_position_output = """\
## Consensus zone
- Email remains relevant for B2B marketing

## Contested zone
No clear contested positions identified.

## Outlier positions
- Email is dead for Gen Z

## Evidence weight summary
Most evidence supports moderate optimism.

## The unresolved question
Can AI personalisation overcome privacy friction?
"""
    with pytest.raises(AgentValidationError) as exc_info:
        agent.validate_output(zero_position_output, state)
    assert exc_info.value.agent_id == "A3"


def test_no_contest_warning_not_added_when_2_or_more_positions(agent, state):
    agent.validate_output(A3_HAPPY_OUTPUT, state)
    assert PipelineWarning.NO_CONTEST not in state.agents["A3"].warnings


def test_validate_output_accepts_position_label_style_in_contested_zone(agent, state):
    position_label_output = """\
## Consensus zone
- Email remains relevant

## Contested zone
Position: AI personalisation is overhyped
Position: Privacy regulations are a major blocker

## Outlier positions
- Email is dead for Gen Z

## Evidence weight summary
Evidence is mixed.

## The unresolved question
Will privacy friction overcome AI gains?
"""
    # Should pass — 2 "Position:" labels satisfies the ≥ 2 requirement
    agent.validate_output(position_label_output, state)
    assert PipelineWarning.NO_CONTEST not in state.agents["A3"].warnings


# ---------------------------------------------------------------------------
# CP-05: conflation re-prompt triggered when position text appears in both zones
# ---------------------------------------------------------------------------

async def test_cp05_conflation_reprompt_triggered(mock_llm, state):
    """When the same bullet text appears in both Consensus and Contested zones,
    the agent must re-prompt once."""
    conflated_output = """\
## Consensus zone
- Email remains relevant for B2B marketing

## Contested zone
- Email remains relevant for B2B marketing
1. AI personalisation is overhyped

## Outlier positions
- Email is dead for Gen Z

## Evidence weight summary
Evidence is mixed.

## The unresolved question
Can AI overcome privacy friction?
"""
    mock_llm.call = AsyncMock(
        side_effect=[
            (conflated_output, MOCK_TOKEN_USAGE),
            (A3_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A3LandscapeMapper(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 2


async def test_cp05_reprompt_not_triggered_when_no_conflation(mock_llm, state, agent):
    await agent.run(state)
    assert mock_llm.call.call_count == 1


async def test_cp05_conflation_reprompt_not_triggered_more_than_once(mock_llm, state):
    """Even if the second response is still conflated, the agent must not re-prompt again."""
    conflated_output = """\
## Consensus zone
- Email remains relevant for B2B marketing

## Contested zone
- Email remains relevant for B2B marketing
1. AI personalisation is overhyped

## Outlier positions
- Email is dead for Gen Z

## Evidence weight summary
Evidence is mixed.

## The unresolved question
Can AI overcome privacy friction?
"""
    # Both calls return conflated output — but second call should be accepted as-is
    mock_llm.call = AsyncMock(
        side_effect=[
            (conflated_output, MOCK_TOKEN_USAGE),
            (A3_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A3LandscapeMapper(llm_client=mock_llm)
    await agent.run(state)
    # Must be exactly 2: initial call + one re-prompt (no further loops)
    assert mock_llm.call.call_count == 2


async def test_cp05_reprompt_prompt_instructs_zone_separation(mock_llm, state):
    """The re-prompt text must tell the LLM to keep zones distinct."""
    conflated_output = """\
## Consensus zone
- Email remains relevant for B2B marketing

## Contested zone
- Email remains relevant for B2B marketing
1. AI personalisation is overhyped

## Outlier positions
- Email is dead for Gen Z

## Evidence weight summary
Evidence is mixed.

## The unresolved question
Can AI overcome privacy friction?
"""
    mock_llm.call = AsyncMock(
        side_effect=[
            (conflated_output, MOCK_TOKEN_USAGE),
            (A3_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A3LandscapeMapper(llm_client=mock_llm)
    await agent.run(state)
    second_prompt = mock_llm.call.call_args_list[1][0][1]
    # Must mention the issue
    assert "Consensus" in second_prompt or "Contested" in second_prompt


# ---------------------------------------------------------------------------
# run() — sets output, token_usage, status on success
# ---------------------------------------------------------------------------

async def test_run_sets_output_on_success(agent, state):
    await agent.run(state)
    assert state.agents["A3"].output == A3_HAPPY_OUTPUT


async def test_run_sets_status_completed_on_success(agent, state):
    await agent.run(state)
    assert state.agents["A3"].status == AgentStatus.COMPLETED


async def test_run_sets_token_usage_on_success(agent, state):
    await agent.run(state)
    assert state.agents["A3"].token_usage == MOCK_TOKEN_USAGE


async def test_run_returns_output_string(agent, state):
    result = await agent.run(state)
    assert result == A3_HAPPY_OUTPUT
