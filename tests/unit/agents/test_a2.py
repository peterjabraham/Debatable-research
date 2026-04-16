"""Unit tests for A2ClaimExtractor (spec §12)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.a2_claim_extractor import A2ClaimExtractor
from src.pipeline.state import AgentStatus, PipelineState, TokenUsage
from src.utils.errors import AgentValidationError, PipelineWarning

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

A1_HAPPY_OUTPUT = """\
1. URL: https://example.com/study Type: Academic Recency: 2024 Core claim: Email still delivers strong ROI for B2B marketers Credibility signal: Peer-reviewed
2. URL: https://example.com/blog Type: Industry Recency: 2023 Core claim: Personalisation boosts open rates significantly for targeted campaigns Credibility signal: Large sample
3. URL: https://example.com/report Type: Analyst Recency: 2024 Core claim: AI tools are changing email workflow and content creation processes Credibility signal: Major firm
"""

# A2 happy path: 3 blocks, each claim has well over 15 words
A2_HAPPY_OUTPUT = """\
Core claim: Email marketing continues to deliver strong and measurable ROI for B2B organisations of all sizes
Key evidence: Multiple peer-reviewed studies show consistent returns above industry average
Caveats: Results vary by sector and list quality
Implicit assumption: Marketers have access to quality segmented lists

Core claim: Personalisation significantly boosts open rates across well-targeted campaigns when implemented with rigorous data hygiene practices
Key evidence: Large-sample industry studies confirm 20–30 % uplift in open rates
Caveats: Requires robust CRM data to be effective
Implicit assumption: Recipients consent to personalised messaging

Core claim: AI-powered tools are fundamentally transforming how email content is written and workflows are managed at scale
Key evidence: Major analyst firms report widespread adoption across enterprise marketing teams
Caveats: Quality control remains a challenge for AI-generated content
Implicit assumption: Teams have budget to invest in AI tooling
"""

MOCK_TOKEN_USAGE = TokenUsage(input_tokens=150, output_tokens=300, total_tokens=450)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    s = PipelineState(topic="Email marketing", audience="Marketers", tone="Direct")
    # Pre-populate A1 as completed
    s.agents["A1"].status = AgentStatus.COMPLETED
    s.agents["A1"].output = A1_HAPPY_OUTPUT
    return s


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call = AsyncMock(return_value=(A2_HAPPY_OUTPUT, MOCK_TOKEN_USAGE))
    return llm


@pytest.fixture
def agent(mock_llm):
    return A2ClaimExtractor(llm_client=mock_llm)


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

def test_agent_id():
    llm = MagicMock()
    a = A2ClaimExtractor(llm_client=llm)
    assert a.id == "A2"


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_includes_a1_output(agent, state):
    prompt = agent.build_prompt(state)
    assert A1_HAPPY_OUTPUT in prompt


def test_build_prompt_includes_extraction_labels(agent, state):
    prompt = agent.build_prompt(state)
    assert "Core claim" in prompt
    assert "Key evidence" in prompt
    assert "Caveats" in prompt
    assert "Implicit assumption" in prompt


# ---------------------------------------------------------------------------
# validate_output — happy path
# ---------------------------------------------------------------------------

def test_validate_output_passes_on_happy_path(agent, state):
    # 3 sources in A1, 3 "Core claim:" blocks in A2 → no exception
    agent.validate_output(A2_HAPPY_OUTPUT, state)


# ---------------------------------------------------------------------------
# validate_output — block count mismatch raises AgentValidationError
# ---------------------------------------------------------------------------

def test_validate_output_raises_when_block_count_mismatches(agent, state):
    # A1 has 3 entries; supply only 2 "Core claim:" blocks
    two_block_output = (
        "Core claim: Email drives measurable returns for B2B organisations consistently over time\n"
        "Key evidence: Studies confirm ROI above average\n"
        "Caveats: Sector dependent\n"
        "Implicit assumption: Good list hygiene\n\n"
        "Core claim: Personalisation lifts open rates in targeted campaigns with proper segmentation\n"
        "Key evidence: Large sample data\n"
        "Caveats: Needs CRM data\n"
        "Implicit assumption: Consent obtained\n"
    )
    with pytest.raises(AgentValidationError) as exc_info:
        agent.validate_output(two_block_output, state)
    assert exc_info.value.agent_id == "A2"


def test_validate_output_raises_with_too_many_blocks(agent, state):
    # 4 blocks for 3 sources → mismatch
    extra_block = A2_HAPPY_OUTPUT + (
        "\nCore claim: An extra unexpected claim block added beyond the source count\n"
        "Key evidence: N/A\n"
        "Caveats: None\n"
        "Implicit assumption: None\n"
    )
    with pytest.raises(AgentValidationError):
        agent.validate_output(extra_block, state)


# ---------------------------------------------------------------------------
# SHALLOW_CLAIMS warning — added when any claim < 15 words
# ---------------------------------------------------------------------------

def test_shallow_claims_warning_added_when_claim_under_15_words(agent, state):
    short_claim_output = (
        "Core claim: Email ROI is high\n"
        "Key evidence: Studies confirm it\n"
        "Caveats: Varies by sector\n"
        "Implicit assumption: Good data\n\n"
        "Core claim: Personalisation lifts open rates significantly in well-targeted campaigns\n"
        "Key evidence: Large sample data proves it\n"
        "Caveats: Requires CRM data\n"
        "Implicit assumption: Consent obtained from recipients\n\n"
        "Core claim: AI tools transform email workflows and content creation processes at scale\n"
        "Key evidence: Analyst reports confirm widespread adoption by enterprises\n"
        "Caveats: Quality control is a challenge for AI content\n"
        "Implicit assumption: Teams have budget for AI tools\n"
    )
    agent.validate_output(short_claim_output, state)
    assert PipelineWarning.SHALLOW_CLAIMS in state.agents["A2"].warnings


def test_shallow_claims_warning_not_added_when_all_claims_long_enough(agent, state):
    agent.validate_output(A2_HAPPY_OUTPUT, state)
    assert PipelineWarning.SHALLOW_CLAIMS not in state.agents["A2"].warnings


# ---------------------------------------------------------------------------
# SHALLOW_CLAIMS does not cause a retry (LLM called only once)
# ---------------------------------------------------------------------------

async def test_shallow_claims_does_not_cause_retry(mock_llm, state):
    short_claim_output = (
        "Core claim: Short claim\n"
        "Key evidence: Evidence\n"
        "Caveats: None\n"
        "Implicit assumption: None\n\n"
        "Core claim: Personalisation lifts open rates significantly in well-targeted campaigns\n"
        "Key evidence: Large sample data\n"
        "Caveats: Requires CRM data\n"
        "Implicit assumption: Consent obtained\n\n"
        "Core claim: AI tools fundamentally change how email workflows and content are managed\n"
        "Key evidence: Analyst firms report adoption\n"
        "Caveats: Quality issues exist\n"
        "Implicit assumption: Budget available\n"
    )
    mock_llm.call = AsyncMock(return_value=(short_claim_output, MOCK_TOKEN_USAGE))
    agent = A2ClaimExtractor(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 1


async def test_shallow_claims_warning_present_in_agent_record_after_run(mock_llm, state):
    short_claim_output = (
        "Core claim: Short\n"
        "Key evidence: Evidence\n"
        "Caveats: None\n"
        "Implicit assumption: None\n\n"
        "Core claim: Personalisation lifts open rates significantly in well-targeted email campaigns\n"
        "Key evidence: Large sample data\n"
        "Caveats: Requires CRM data\n"
        "Implicit assumption: Consent obtained\n\n"
        "Core claim: AI tools are transforming how email content and workflows are managed at scale\n"
        "Key evidence: Analyst firms confirm widespread adoption\n"
        "Caveats: Quality control remains a challenge\n"
        "Implicit assumption: Teams have the budget\n"
    )
    mock_llm.call = AsyncMock(return_value=(short_claim_output, MOCK_TOKEN_USAGE))
    agent = A2ClaimExtractor(llm_client=mock_llm)
    await agent.run(state)
    assert PipelineWarning.SHALLOW_CLAIMS in state.agents["A2"].warnings


# ---------------------------------------------------------------------------
# CP-02: re-prompt when block count mismatches
# ---------------------------------------------------------------------------

async def test_cp02_reprompt_triggered_when_block_count_mismatches(mock_llm, state):
    """First LLM response has wrong block count; second is correct."""
    two_block_output = (
        "Core claim: Email ROI is strong for B2B organisations in many sectors globally\n"
        "Key evidence: Peer-reviewed studies\n"
        "Caveats: Sector dependent\n"
        "Implicit assumption: Good list hygiene maintained\n\n"
        "Core claim: Personalisation lifts open rates significantly across well-segmented campaigns\n"
        "Key evidence: Large sample industry data\n"
        "Caveats: Needs CRM data\n"
        "Implicit assumption: Consent obtained from all recipients\n"
    )
    mock_llm.call = AsyncMock(
        side_effect=[
            (two_block_output, MOCK_TOKEN_USAGE),
            (A2_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A2ClaimExtractor(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 2


async def test_cp02_reprompt_contains_explicit_count(mock_llm, state):
    """The re-prompt passed to the LLM must mention the exact expected source count."""
    two_block_output = (
        "Core claim: Email ROI is strong for B2B organisations in many sectors globally\n"
        "Key evidence: Peer-reviewed studies\n"
        "Caveats: Sector dependent\n"
        "Implicit assumption: Good list hygiene maintained\n\n"
        "Core claim: Personalisation lifts open rates significantly across well-segmented campaigns\n"
        "Key evidence: Large sample industry data\n"
        "Caveats: Needs CRM data\n"
        "Implicit assumption: Consent obtained from all recipients\n"
    )
    mock_llm.call = AsyncMock(
        side_effect=[
            (two_block_output, MOCK_TOKEN_USAGE),
            (A2_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A2ClaimExtractor(llm_client=mock_llm)
    await agent.run(state)
    second_prompt = mock_llm.call.call_args_list[1][0][1]
    assert "3" in second_prompt  # A1 has 3 sources


async def test_cp02_no_reprompt_when_block_count_matches(mock_llm, state, agent):
    await agent.run(state)
    assert mock_llm.call.call_count == 1


# ---------------------------------------------------------------------------
# run() — sets output, token_usage, status on success
# ---------------------------------------------------------------------------

async def test_run_sets_output_on_success(agent, state):
    await agent.run(state)
    assert state.agents["A2"].output == A2_HAPPY_OUTPUT


async def test_run_sets_status_completed_on_success(agent, state):
    await agent.run(state)
    assert state.agents["A2"].status == AgentStatus.COMPLETED


async def test_run_sets_token_usage_on_success(agent, state):
    await agent.run(state)
    assert state.agents["A2"].token_usage == MOCK_TOKEN_USAGE
