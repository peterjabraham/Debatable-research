"""
Integration tests for the full pipeline runner.
All LLM calls are mocked — no real API calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipeline.runner import PipelineRunner
from src.pipeline.state import AgentStatus, PipelineState, PipelineStatus, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fixture(name: str) -> str:
    return (Path("tests/fixtures/llm_responses") / f"{name}.txt").read_text()


def _make_token_usage(n: int = 100) -> TokenUsage:
    return TokenUsage(input_tokens=n // 2, output_tokens=n // 2, total_tokens=n)


def _make_mock_llm(responses: list[str]) -> AsyncMock:
    """Returns an LLMClient mock whose .call() yields responses in sequence."""
    client = MagicMock()
    side_effects = []
    for r in responses:
        side_effects.append((r, _make_token_usage(100)))
    client.call = AsyncMock(side_effect=side_effects)
    return client


def _happy_responses() -> list[str]:
    return [
        _load_fixture("a1_happy_path"),
        _load_fixture("a2_happy_path"),
        _load_fixture("a3_happy_path"),
        _load_fixture("a4_happy_path"),
        _load_fixture("a5_happy_path"),
        _load_fixture("a6_happy_path"),
        _load_fixture("a6_happy_path"),  # humanise pass
    ]


# ---------------------------------------------------------------------------
# Full pipeline happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline_runs_a1_to_a6(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr("src.pipeline.runner.Path", lambda *a: tmp_path.joinpath(*a))

    llm = _make_mock_llm(_happy_responses())
    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    runner = PipelineRunner(llm)
    result = await runner.run(state)

    assert result.pipeline_status == PipelineStatus.COMPLETED
    for aid in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        assert result.agents[aid].status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_each_agent_receives_correct_predecessor_output(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr("src.pipeline.runner.Path", lambda *a: tmp_path.joinpath(*a))

    llm = _make_mock_llm(_happy_responses())
    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    runner = PipelineRunner(llm)
    result = await runner.run(state)

    # A2's prompt should embed A1's output
    a1_out = result.agents["A1"].output
    a2_input = result.agents["A2"].input
    assert a1_out and a1_out[:50] in a2_input


@pytest.mark.asyncio
async def test_final_state_has_all_agents_completed(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr("src.pipeline.runner.Path", lambda *a: tmp_path.joinpath(*a))

    llm = _make_mock_llm(_happy_responses())
    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    runner = PipelineRunner(llm)
    result = await runner.run(state)

    for aid in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        assert result.agents[aid].status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_a6_output_is_non_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr("src.pipeline.runner.Path", lambda *a: tmp_path.joinpath(*a))

    llm = _make_mock_llm(_happy_responses())
    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    runner = PipelineRunner(llm)
    result = await runner.run(state)

    assert result.agents["A6"].output
    assert len(result.agents["A6"].output) > 100


@pytest.mark.asyncio
async def test_checkpoint_saved_after_each_agent(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr("src.pipeline.runner.Path", lambda *a: tmp_path.joinpath(*a))

    saved_after: list[str] = []
    original_save = __import__(
        "src.pipeline.checkpoints", fromlist=["save"]
    ).save

    def tracking_save(s: PipelineState) -> None:
        for aid in ["A1", "A2", "A3", "A4", "A5", "A6"]:
            if s.agents[aid].status == AgentStatus.COMPLETED:
                if aid not in saved_after:
                    saved_after.append(aid)
        original_save(s)

    monkeypatch.setattr("src.pipeline.runner.save", tracking_save)

    llm = _make_mock_llm(_happy_responses())
    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    runner = PipelineRunner(llm)
    await runner.run(state)

    assert "A1" in saved_after
    assert "A6" in saved_after


@pytest.mark.asyncio
async def test_total_tokens_accumulates(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr("src.pipeline.runner.Path", lambda *a: tmp_path.joinpath(*a))

    llm = _make_mock_llm(_happy_responses())
    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    runner = PipelineRunner(llm)
    result = await runner.run(state)

    # 6 agents × 100 tokens each = 600
    assert result.total_tokens == 600


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_resumes_from_a3(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr("src.pipeline.runner.Path", lambda *a: tmp_path.joinpath(*a))

    # Pre-populate A1+A2 as completed
    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    state.agents["A1"].status = AgentStatus.COMPLETED
    state.agents["A1"].output = _load_fixture("a1_happy_path")
    state.agents["A1"].token_usage = _make_token_usage(100)
    state.agents["A2"].status = AgentStatus.COMPLETED
    state.agents["A2"].output = _load_fixture("a2_happy_path")
    state.agents["A2"].token_usage = _make_token_usage(100)
    state.total_tokens = 200

    # Only A3-A6 responses needed (+ humanise)
    remaining = [
        _load_fixture("a3_happy_path"),
        _load_fixture("a4_happy_path"),
        _load_fixture("a5_happy_path"),
        _load_fixture("a6_happy_path"),
        _load_fixture("a6_happy_path"),  # humanise pass
    ]
    llm = _make_mock_llm(remaining)
    runner = PipelineRunner(llm)
    result = await runner.run(state)

    assert result.pipeline_status == PipelineStatus.COMPLETED
    assert llm.call.call_count == 5  # A3-A6 + humanise


@pytest.mark.asyncio
async def test_pipeline_resumes_from_a5(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr("src.pipeline.runner.Path", lambda *a: tmp_path.joinpath(*a))

    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    for aid, fname in [("A1", "a1_happy_path"), ("A2", "a2_happy_path"),
                       ("A3", "a3_happy_path"), ("A4", "a4_happy_path")]:
        state.agents[aid].status = AgentStatus.COMPLETED
        state.agents[aid].output = _load_fixture(fname)
        state.agents[aid].token_usage = _make_token_usage(100)
    state.total_tokens = 400

    llm = _make_mock_llm([
        _load_fixture("a5_happy_path"),
        _load_fixture("a6_happy_path"),
        _load_fixture("a6_happy_path"),  # humanise pass
    ])
    runner = PipelineRunner(llm)
    result = await runner.run(state)

    assert result.pipeline_status == PipelineStatus.COMPLETED
    assert llm.call.call_count == 3  # A5 + A6 + humanise


@pytest.mark.asyncio
async def test_skips_completed_agents_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr("src.pipeline.runner.Path", lambda *a: tmp_path.joinpath(*a))

    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    state.agents["A1"].status = AgentStatus.COMPLETED
    state.agents["A1"].output = _load_fixture("a1_happy_path")
    state.agents["A1"].token_usage = _make_token_usage(100)
    state.total_tokens = 100

    llm = _make_mock_llm([
        _load_fixture("a2_happy_path"),
        _load_fixture("a3_happy_path"),
        _load_fixture("a4_happy_path"),
        _load_fixture("a5_happy_path"),
        _load_fixture("a6_happy_path"),
        _load_fixture("a6_happy_path"),  # humanise pass
    ])
    runner = PipelineRunner(llm)
    await runner.run(state)

    # A1 was completed and must NOT be re-run
    assert llm.call.call_count == 6  # A2-A6 + humanise


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_halts_when_a3_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr("src.pipeline.runner.Path", lambda *a: tmp_path.joinpath(*a))

    # A3 returns no-contest fixture → raises AgentValidationError
    responses = [
        _load_fixture("a1_happy_path"),
        _load_fixture("a2_happy_path"),
        _load_fixture("a3_no_contest"),
    ]
    llm = _make_mock_llm(responses)
    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    runner = PipelineRunner(llm, user_input_fn=lambda _: "n")

    result = await runner.run(state)
    assert result.pipeline_status in (PipelineStatus.FAILED, PipelineStatus.ABORTED)


@pytest.mark.asyncio
async def test_checkpoint_saved_when_agent_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr("src.pipeline.runner.Path", lambda *a: tmp_path.joinpath(*a))

    responses = [
        _load_fixture("a1_happy_path"),
        _load_fixture("a2_happy_path"),
        _load_fixture("a3_no_contest"),
    ]
    llm = _make_mock_llm(responses)
    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    runner = PipelineRunner(llm, user_input_fn=lambda _: "n")
    await runner.run(state)

    # Checkpoint file should exist
    checkpoint_file = tmp_path / f"{state.run_id}.json"
    assert checkpoint_file.exists()


@pytest.mark.asyncio
async def test_pipeline_does_not_run_a4_when_a3_failed(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr("src.pipeline.runner.Path", lambda *a: tmp_path.joinpath(*a))

    responses = [
        _load_fixture("a1_happy_path"),
        _load_fixture("a2_happy_path"),
        _load_fixture("a3_no_contest"),
    ]
    llm = _make_mock_llm(responses)
    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    runner = PipelineRunner(llm, user_input_fn=lambda _: "n")
    await runner.run(state)

    # A4 should still be PENDING
    assert state.agents["A4"].status == AgentStatus.PENDING


def _patch_output_path(tmp_path, monkeypatch):
    """Redirect output/ directory writes into tmp_path for isolation."""
    import src.pipeline.runner as runner_mod
    original_path = runner_mod.Path

    def patched_path(*args):
        p = original_path(*args)
        s = str(p)
        # Redirect anything under "output" to tmp_path
        if s == "output" or s.startswith("output/") or s.startswith("output\\"):
            return tmp_path / s
        return p

    monkeypatch.setattr(runner_mod, "Path", patched_path)


@pytest.mark.asyncio
async def test_warnings_log_written_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output_path(tmp_path, monkeypatch)

    responses = [
        _load_fixture("a1_happy_path"),
        _load_fixture("a2_happy_path"),
        _load_fixture("a3_no_contest"),
    ]
    llm = _make_mock_llm(responses)
    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    runner = PipelineRunner(llm, user_input_fn=lambda _: "n")
    await runner.run(state)

    logs = list(tmp_path.rglob("warnings.log"))
    assert len(logs) >= 1


@pytest.mark.asyncio
async def test_post_md_written_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output_path(tmp_path, monkeypatch)

    llm = _make_mock_llm(_happy_responses())
    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    runner = PipelineRunner(llm)
    await runner.run(state)

    posts = list(tmp_path.rglob("post.md"))
    assert len(posts) >= 1
    assert len(posts[0].read_text()) > 100


@pytest.mark.asyncio
async def test_audit_json_contains_token_usage(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output_path(tmp_path, monkeypatch)

    llm = _make_mock_llm(_happy_responses())
    state = PipelineState(topic="Email", audience="Marketers", tone="Direct")
    runner = PipelineRunner(llm)
    await runner.run(state)

    audits = list(tmp_path.rglob("audit.json"))
    assert len(audits) >= 1
    audit = json.loads(audits[0].read_text())
    assert "total_tokens" in audit
    assert audit["total_tokens"] == 600
    assert "agents" in audit
    for aid in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        assert aid in audit["agents"]
        assert audit["agents"][aid]["token_usage"] is not None
