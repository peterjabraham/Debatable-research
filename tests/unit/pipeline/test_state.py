import time

import pytest

from src.pipeline.state import (
    AgentStatus,
    PipelineState,
    can_run,
    require_output,
    transition,
)
from src.utils.errors import InvalidStateTransitionError, PipelineDependencyError


@pytest.fixture
def state():
    return PipelineState(topic="t", audience="a", tone="x")


# --- can_run ---

def test_can_run_a1_always_true(state):
    assert can_run(state, "A1") is True


def test_can_run_a2_false_when_a1_pending(state):
    assert can_run(state, "A2") is False


def test_can_run_a2_false_when_a1_running(state):
    state.agents["A1"].status = AgentStatus.RUNNING
    assert can_run(state, "A2") is False


def test_can_run_a2_false_when_a1_failed(state):
    state.agents["A1"].status = AgentStatus.FAILED
    assert can_run(state, "A2") is False


def test_can_run_a2_true_when_a1_completed(state):
    state.agents["A1"].status = AgentStatus.COMPLETED
    assert can_run(state, "A2") is True


# --- require_output ---

def test_require_output_raises_when_pending(state):
    with pytest.raises(PipelineDependencyError):
        require_output(state, "A1")


def test_require_output_raises_when_failed(state):
    state.agents["A1"].status = AgentStatus.FAILED
    with pytest.raises(PipelineDependencyError):
        require_output(state, "A1")


def test_require_output_raises_when_output_none_despite_completed(state):
    state.agents["A1"].status = AgentStatus.COMPLETED
    state.agents["A1"].output = None
    with pytest.raises(PipelineDependencyError):
        require_output(state, "A1")


def test_require_output_returns_string_when_completed(state):
    state.agents["A1"].status = AgentStatus.COMPLETED
    state.agents["A1"].output = "hello"
    result = require_output(state, "A1")
    assert result == "hello"


# --- transition ---

def test_transition_pending_to_running(state):
    transition(state, "A1", AgentStatus.RUNNING)
    assert state.agents["A1"].status == AgentStatus.RUNNING


def test_transition_running_to_completed(state):
    state.agents["A1"].status = AgentStatus.RUNNING
    state.agents["A1"].started_at = time.time()
    transition(state, "A1", AgentStatus.COMPLETED)
    assert state.agents["A1"].status == AgentStatus.COMPLETED


def test_transition_running_to_failed(state):
    state.agents["A1"].status = AgentStatus.RUNNING
    state.agents["A1"].started_at = time.time()
    transition(state, "A1", AgentStatus.FAILED)
    assert state.agents["A1"].status == AgentStatus.FAILED


def test_transition_running_to_timed_out(state):
    state.agents["A1"].status = AgentStatus.RUNNING
    state.agents["A1"].started_at = time.time()
    transition(state, "A1", AgentStatus.TIMED_OUT)
    assert state.agents["A1"].status == AgentStatus.TIMED_OUT


def test_transition_failed_to_running_valid(state):
    state.agents["A1"].status = AgentStatus.FAILED
    transition(state, "A1", AgentStatus.RUNNING)
    assert state.agents["A1"].status == AgentStatus.RUNNING


def test_transition_timed_out_to_running_valid(state):
    state.agents["A1"].status = AgentStatus.TIMED_OUT
    transition(state, "A1", AgentStatus.RUNNING)
    assert state.agents["A1"].status == AgentStatus.RUNNING


def test_transition_completed_to_running_raises(state):
    state.agents["A1"].status = AgentStatus.COMPLETED
    with pytest.raises(InvalidStateTransitionError):
        transition(state, "A1", AgentStatus.RUNNING)


def test_transition_pending_to_completed_raises(state):
    with pytest.raises(InvalidStateTransitionError):
        transition(state, "A1", AgentStatus.COMPLETED)


def test_transition_pending_to_failed_raises(state):
    with pytest.raises(InvalidStateTransitionError):
        transition(state, "A1", AgentStatus.FAILED)


def test_transition_sets_started_at_on_running(state):
    before = time.time()
    transition(state, "A1", AgentStatus.RUNNING)
    assert state.agents["A1"].started_at is not None
    assert state.agents["A1"].started_at >= before


def test_transition_sets_completed_at_on_completed(state):
    state.agents["A1"].status = AgentStatus.RUNNING
    state.agents["A1"].started_at = time.time()
    before = time.time()
    transition(state, "A1", AgentStatus.COMPLETED)
    assert state.agents["A1"].completed_at is not None
    assert state.agents["A1"].completed_at >= before


def test_transition_calculates_duration_ms(state):
    transition(state, "A1", AgentStatus.RUNNING)
    time.sleep(0.01)
    transition(state, "A1", AgentStatus.COMPLETED)
    assert state.agents["A1"].duration_ms is not None
    assert state.agents["A1"].duration_ms >= 10
