import json
import uuid

import pytest

from src.pipeline.checkpoints import get_resume_point, load, save
from src.pipeline.state import AgentStatus, PipelineState
from src.utils.errors import CheckpointCorruptError


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    s = PipelineState(topic="t", audience="a", tone="x")
    return s


@pytest.fixture
def tmp_checkpoint_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    return tmp_path


def test_save_writes_valid_json(state, tmp_checkpoint_dir):
    save(state)
    path = tmp_checkpoint_dir / f"{state.run_id}.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["topic"] == "t"


def test_save_is_idempotent(state, tmp_checkpoint_dir):
    save(state)
    state.agents["A1"].status = AgentStatus.COMPLETED
    save(state)
    path = tmp_checkpoint_dir / f"{state.run_id}.json"
    data = json.loads(path.read_text())
    assert data["agents"]["A1"]["status"] == "completed"


def test_load_returns_correct_state(state, tmp_checkpoint_dir):
    save(state)
    loaded = load(state.run_id)
    assert loaded.run_id == state.run_id
    assert loaded.topic == "t"


def test_load_raises_on_malformed_json(tmp_checkpoint_dir):
    run_id = str(uuid.uuid4())
    path = tmp_checkpoint_dir / f"{run_id}.json"
    path.write_text("not json{{")
    with pytest.raises(CheckpointCorruptError):
        load(run_id)


def test_load_raises_on_missing_required_fields(tmp_checkpoint_dir):
    run_id = str(uuid.uuid4())
    path = tmp_checkpoint_dir / f"{run_id}.json"
    path.write_text(json.dumps({"run_id": run_id}))
    with pytest.raises(CheckpointCorruptError):
        load(run_id)


def test_load_raises_on_wrong_field_types(tmp_checkpoint_dir):
    run_id = str(uuid.uuid4())
    path = tmp_checkpoint_dir / f"{run_id}.json"
    path.write_text(json.dumps({
        "run_id": run_id,
        "topic": 123,   # should be str
        "audience": "a",
        "tone": "x",
    }))
    with pytest.raises(CheckpointCorruptError):
        load(run_id)


# --- get_resume_point ---

def make_state_with_statuses(statuses: dict) -> PipelineState:
    s = PipelineState(topic="t", audience="a", tone="x")
    for aid, status in statuses.items():
        s.agents[aid].status = status
    return s


def test_resume_point_none_when_all_pending():
    s = make_state_with_statuses({})
    # All start as PENDING — get_resume_point should return A1
    assert get_resume_point(s) == "A1"


def test_resume_point_a1_when_a1_pending():
    s = PipelineState(topic="t", audience="a", tone="x")
    assert get_resume_point(s) == "A1"


def test_resume_point_a3_when_a1_a2_completed():
    s = make_state_with_statuses({
        "A1": AgentStatus.COMPLETED,
        "A2": AgentStatus.COMPLETED,
    })
    assert get_resume_point(s) == "A3"


def test_resume_point_a3_when_a1_a2_completed_a3_failed():
    s = make_state_with_statuses({
        "A1": AgentStatus.COMPLETED,
        "A2": AgentStatus.COMPLETED,
        "A3": AgentStatus.FAILED,
    })
    assert get_resume_point(s) == "A3"


def test_resume_point_a3_when_a1_a2_completed_a3_timed_out():
    s = make_state_with_statuses({
        "A1": AgentStatus.COMPLETED,
        "A2": AgentStatus.COMPLETED,
        "A3": AgentStatus.TIMED_OUT,
    })
    assert get_resume_point(s) == "A3"


def test_resume_point_none_when_all_completed():
    s = make_state_with_statuses({
        "A1": AgentStatus.COMPLETED,
        "A2": AgentStatus.COMPLETED,
        "A3": AgentStatus.COMPLETED,
        "A4": AgentStatus.COMPLETED,
        "A5": AgentStatus.COMPLETED,
        "A6": AgentStatus.COMPLETED,
    })
    assert get_resume_point(s) is None
