import json
from pathlib import Path

from src.pipeline.state import AGENT_ORDER, AgentStatus, PipelineState
from src.utils.errors import CheckpointCorruptError

CHECKPOINT_DIR = Path(".pipeline-checkpoints")


def save(state: PipelineState) -> None:
    """Write state to disk after every agent status change."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"{state.run_id}.json"
    path.write_text(state.model_dump_json(indent=2))


def load(run_id: str) -> PipelineState:
    """Load and validate checkpoint. Raises CheckpointCorruptError if invalid."""
    path = CHECKPOINT_DIR / f"{run_id}.json"
    try:
        data = json.loads(path.read_text())
        return PipelineState.model_validate(data)
    except Exception as e:
        raise CheckpointCorruptError(str(path)) from e


def get_resume_point(state: PipelineState) -> str | None:
    """Return the ID of the first non-completed agent, or None if all done."""
    for agent_id in AGENT_ORDER:
        if state.agents[agent_id].status != AgentStatus.COMPLETED:
            return agent_id
    return None
