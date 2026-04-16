"""Unit tests for A1ResearchCollector (spec §12)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.a1_research_collector import A1ResearchCollector
from src.pipeline.state import AgentStatus, PipelineState, TokenUsage
from src.utils.errors import AgentValidationError, PipelineWarning

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HAPPY_PATH_OUTPUT = """\
1. URL: https://example.com/study Type: Academic Recency: 2024 Core claim: Email still delivers strong ROI for B2B marketers Credibility signal: Peer-reviewed
2. URL: https://example.com/blog Type: Industry Recency: 2023 Core claim: Personalisation boosts open rates significantly for targeted campaigns Credibility signal: Large sample
3. URL: https://example.com/report Type: Analyst Recency: 2024 Core claim: AI tools are changing email workflow and content creation processes Credibility signal: Major firm
"""

MOCK_TOKEN_USAGE = TokenUsage(input_tokens=100, output_tokens=200, total_tokens=300)


@pytest.fixture
def state():
    return PipelineState(topic="Email marketing", audience="Marketers", tone="Direct")


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call = AsyncMock(return_value=(HAPPY_PATH_OUTPUT, MOCK_TOKEN_USAGE))
    return llm


@pytest.fixture
def agent(mock_llm):
    return A1ResearchCollector(llm_client=mock_llm)


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

def test_agent_id():
    llm = MagicMock()
    a = A1ResearchCollector(llm_client=llm)
    assert a.id == "A1"


def test_agent_timeout_ms():
    llm = MagicMock()
    a = A1ResearchCollector(llm_client=llm)
    assert a.timeout_ms == 90_000


def test_agent_max_retries():
    llm = MagicMock()
    a = A1ResearchCollector(llm_client=llm)
    assert a.max_retries == 2


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_includes_topic(agent, state):
    prompt = agent.build_prompt(state)
    assert "Email marketing" in prompt


def test_build_prompt_includes_provided_sources_when_present(agent, state):
    state.provided_sources = ["https://source1.com", "https://source2.com"]
    prompt = agent.build_prompt(state)
    assert "https://source1.com" in prompt
    assert "https://source2.com" in prompt


def test_build_prompt_no_sources_section_when_empty(agent, state):
    state.provided_sources = []
    prompt = agent.build_prompt(state)
    assert "Provided sources" not in prompt


# ---------------------------------------------------------------------------
# validate_output — happy path
# ---------------------------------------------------------------------------

def test_validate_output_passes_on_happy_path(agent, state):
    # 3 numbered entries + all 5 required fields → no exception
    agent.validate_output(HAPPY_PATH_OUTPUT, state)


# ---------------------------------------------------------------------------
# validate_output — fewer than 3 entries raises AgentValidationError
# ---------------------------------------------------------------------------

def test_validate_output_raises_when_fewer_than_3_entries(agent, state):
    two_entry_output = (
        "1. URL: https://a.com Type: Academic Recency: 2024 "
        "Core claim: Short claim here Credibility signal: Good\n"
        "2. URL: https://b.com Type: Industry Recency: 2023 "
        "Core claim: Another claim Credibility signal: OK\n"
    )
    with pytest.raises(AgentValidationError) as exc_info:
        agent.validate_output(two_entry_output, state)
    assert exc_info.value.agent_id == "A1"


def test_validate_output_raises_with_zero_entries(agent, state):
    with pytest.raises(AgentValidationError):
        agent.validate_output("No numbered sources here at all.", state)


# ---------------------------------------------------------------------------
# validate_output — CONTEXT_NEAR_LIMIT warning when < 6 entries
# ---------------------------------------------------------------------------

def test_validate_output_adds_context_near_limit_warning_when_fewer_than_6_entries(
    agent, state
):
    # HAPPY_PATH_OUTPUT has exactly 3 entries → warning added, no raise
    agent.validate_output(HAPPY_PATH_OUTPUT, state)
    assert PipelineWarning.CONTEXT_NEAR_LIMIT in state.agents["A1"].warnings


def test_validate_output_no_context_near_limit_warning_when_6_or_more_entries(
    agent, state
):
    six_entry_output = "\n".join(
        f"{i}. URL: https://example{i}.com Type: Academic Recency: 2024 "
        f"Core claim: Claim {i} here Credibility signal: Good"
        for i in range(1, 7)
    )
    agent.validate_output(six_entry_output, state)
    assert PipelineWarning.CONTEXT_NEAR_LIMIT not in state.agents["A1"].warnings


def test_validate_output_context_near_limit_does_not_raise(agent, state):
    # Warning must not cause an exception — just silently appended
    try:
        agent.validate_output(HAPPY_PATH_OUTPUT, state)
    except AgentValidationError:
        pytest.fail("CONTEXT_NEAR_LIMIT should not raise AgentValidationError")


# ---------------------------------------------------------------------------
# validate_output — missing required fields
# ---------------------------------------------------------------------------

def test_validate_output_raises_when_required_field_missing(agent, state):
    # Remove "Credibility signal" from an otherwise valid output
    bad_output = HAPPY_PATH_OUTPUT.replace("Credibility signal", "CS")
    with pytest.raises(AgentValidationError) as exc_info:
        agent.validate_output(bad_output, state)
    assert "Credibility signal" in exc_info.value.reason


# ---------------------------------------------------------------------------
# CP-01: fallback triggered when output contains "could not find"
# ---------------------------------------------------------------------------

async def test_cp01_fallback_triggered_on_could_not_find(mock_llm, state):
    no_sources_text = "I could not find any relevant sources for this topic."
    # First call returns "could not find", second returns happy path
    mock_llm.call = AsyncMock(
        side_effect=[
            (no_sources_text, MOCK_TOKEN_USAGE),
            (HAPPY_PATH_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A1ResearchCollector(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 2


async def test_cp01_fallback_triggered_on_no_sources(mock_llm, state):
    no_sources_text = "There are no sources available for this query."
    mock_llm.call = AsyncMock(
        side_effect=[
            (no_sources_text, MOCK_TOKEN_USAGE),
            (HAPPY_PATH_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A1ResearchCollector(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 2


async def test_cp01_fallback_triggered_case_insensitive_could_not_find(mock_llm, state):
    mixed_case_text = "I COULD NOT FIND any relevant sources."
    mock_llm.call = AsyncMock(
        side_effect=[
            (mixed_case_text, MOCK_TOKEN_USAGE),
            (HAPPY_PATH_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A1ResearchCollector(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 2


# ---------------------------------------------------------------------------
# CP-01: fallback output contains "[Training Knowledge - unverified]"
# ---------------------------------------------------------------------------

async def test_cp01_fallback_prompt_contains_training_knowledge_label(mock_llm, state):
    no_sources_text = "I could not find any relevant sources."
    fallback_output = (
        "1. URL: [Training Knowledge - unverified] Type: Training Recency: 2024 "
        "Core claim: Email marketing yields high ROI for B2B companies Credibility signal: Training\n"
        "2. URL: [Training Knowledge - unverified] Type: Training Recency: 2023 "
        "Core claim: Personalisation significantly improves open rates in campaigns Credibility signal: Training\n"
        "3. URL: [Training Knowledge - unverified] Type: Training Recency: 2024 "
        "Core claim: Automation tools change how email content is written and managed Credibility signal: Training\n"
    )
    mock_llm.call = AsyncMock(
        side_effect=[
            (no_sources_text, MOCK_TOKEN_USAGE),
            (fallback_output, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A1ResearchCollector(llm_client=mock_llm)
    result = await agent.run(state)
    assert "[Training Knowledge - unverified]" in result


async def test_cp01_fallback_prompt_includes_training_knowledge_instruction(mock_llm, state):
    """The fallback reprompt sent to the LLM should mention [Training Knowledge - unverified]."""
    no_sources_text = "could not find sources"
    mock_llm.call = AsyncMock(
        side_effect=[
            (no_sources_text, MOCK_TOKEN_USAGE),
            (HAPPY_PATH_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A1ResearchCollector(llm_client=mock_llm)
    await agent.run(state)
    # The second call's prompt argument should contain the fallback instruction
    second_call_prompt = mock_llm.call.call_args_list[1][0][1]  # positional arg index 1
    assert "[Training Knowledge - unverified]" in second_call_prompt


# ---------------------------------------------------------------------------
# CP-01: no fallback when output is clean
# ---------------------------------------------------------------------------

async def test_cp01_no_fallback_on_clean_output(mock_llm, state, agent):
    await agent.run(state)
    assert mock_llm.call.call_count == 1


# ---------------------------------------------------------------------------
# run() — sets output, token_usage, status=COMPLETED on success
# ---------------------------------------------------------------------------

async def test_run_sets_output_on_success(agent, state):
    await agent.run(state)
    assert state.agents["A1"].output == HAPPY_PATH_OUTPUT


async def test_run_sets_status_completed_on_success(agent, state):
    await agent.run(state)
    assert state.agents["A1"].status == AgentStatus.COMPLETED


async def test_run_sets_token_usage_on_success(agent, state):
    await agent.run(state)
    assert state.agents["A1"].token_usage == MOCK_TOKEN_USAGE


async def test_run_returns_output_string(agent, state):
    result = await agent.run(state)
    assert result == HAPPY_PATH_OUTPUT
