from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from src.utils.errors import InvalidStateTransitionError, PipelineDependencyError

AgentId = Literal["A1", "A2", "A3", "A35", "A4", "A5", "A6"]
AGENT_ORDER: list[AgentId] = ["A1", "A2", "A3", "A35", "A4", "A5", "A6"]


class Analogy(BaseModel):
    """A structural analogy produced by A35, stored on PipelineState."""
    position: str
    title: str
    domain: str
    structural_parallel: str
    maps_to: str
    breaks_down: str
    hook_candidate: bool = False


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class AgentRecord(BaseModel):
    id: AgentId
    status: AgentStatus = AgentStatus.PENDING
    started_at: float | None = None
    completed_at: float | None = None
    duration_ms: float | None = None
    input: str = ""
    output: str | None = None
    token_usage: TokenUsage | None = None
    retry_count: int = 0
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PipelineStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class PipelineState(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    audience: str
    tone: str
    target_word_count: int = 900
    cluster_angle: str | None = None
    analogies: list[Analogy] | None = None
    provided_sources: list[str] = Field(default_factory=list)
    agents: dict[AgentId, AgentRecord] = Field(
        default_factory=lambda: {aid: AgentRecord(id=aid) for aid in AGENT_ORDER}
    )
    pipeline_status: PipelineStatus = PipelineStatus.RUNNING
    started_at: float = Field(default_factory=time.time)
    completed_at: float | None = None
    checkpoint_path: str = ""
    total_tokens: int = 0


def can_run(state: PipelineState, agent_id: AgentId) -> bool:
    """An agent can run only if its predecessor completed."""
    idx = AGENT_ORDER.index(agent_id)
    if idx == 0:
        return True
    predecessor = AGENT_ORDER[idx - 1]
    return state.agents[predecessor].status == AgentStatus.COMPLETED


def require_output(state: PipelineState, agent_id: AgentId) -> str:
    """Returns agent output or raises if unavailable."""
    record = state.agents[agent_id]
    if record.status != AgentStatus.COMPLETED or record.output is None:
        raise PipelineDependencyError(
            f"Agent {agent_id} output required but not available. Status: {record.status}"
        )
    return record.output


VALID_TRANSITIONS: set[tuple[AgentStatus, AgentStatus]] = {
    (AgentStatus.PENDING, AgentStatus.RUNNING),
    (AgentStatus.RUNNING, AgentStatus.COMPLETED),
    (AgentStatus.RUNNING, AgentStatus.FAILED),
    (AgentStatus.RUNNING, AgentStatus.TIMED_OUT),
    (AgentStatus.FAILED, AgentStatus.RUNNING),
    (AgentStatus.TIMED_OUT, AgentStatus.RUNNING),
}


def transition(state: PipelineState, agent_id: AgentId, to: AgentStatus) -> None:
    record = state.agents[agent_id]
    from_ = record.status
    if (from_, to) not in VALID_TRANSITIONS:
        raise InvalidStateTransitionError(agent_id, from_, to)
    record.status = to
    if to == AgentStatus.RUNNING:
        record.started_at = time.time()
    elif to in (AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.TIMED_OUT):
        record.completed_at = time.time()
        if record.started_at:
            record.duration_ms = (record.completed_at - record.started_at) * 1000
