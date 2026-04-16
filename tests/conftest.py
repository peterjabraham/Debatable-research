import pytest
from unittest.mock import AsyncMock
from src.pipeline.state import PipelineState, AgentStatus

@pytest.fixture
def base_state():
    return PipelineState(
        topic="The future of email marketing",
        audience="Senior marketing leaders",
        tone="Direct and analytical",
        target_word_count=900,
    )

@pytest.fixture
def fixture_text():
    """Helper to load a fixture file by name."""
    from pathlib import Path
    def _load(name: str) -> str:
        path = Path("tests/fixtures/llm_responses") / f"{name}.txt"
        return path.read_text()
    return _load

@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    return client

@pytest.fixture
def state_after_a1(base_state, fixture_text):
    """State with A1 completed, A2–A6 pending."""
    base_state.agents["A1"].status = AgentStatus.COMPLETED
    base_state.agents["A1"].output = fixture_text("a1_happy_path")
    return base_state

@pytest.fixture
def state_after_a2(state_after_a1, fixture_text):
    state_after_a1.agents["A2"].status = AgentStatus.COMPLETED
    state_after_a1.agents["A2"].output = fixture_text("a2_happy_path")
    return state_after_a1

@pytest.fixture
def state_after_a3(state_after_a2, fixture_text):
    state_after_a2.agents["A3"].status = AgentStatus.COMPLETED
    state_after_a2.agents["A3"].output = fixture_text("a3_happy_path")
    return state_after_a2

@pytest.fixture
def state_after_a4(state_after_a3, fixture_text):
    state_after_a3.agents["A4"].status = AgentStatus.COMPLETED
    state_after_a3.agents["A4"].output = fixture_text("a4_happy_path")
    return state_after_a3

@pytest.fixture
def state_after_a5(state_after_a4, fixture_text):
    state_after_a4.agents["A5"].status = AgentStatus.COMPLETED
    state_after_a4.agents["A5"].output = fixture_text("a5_happy_path")
    return state_after_a4
