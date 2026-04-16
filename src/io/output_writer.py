"""
Write final pipeline outputs: post.md, audit.json, sources.md,
verdict.md, warnings.log.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.state import PipelineState


def write_outputs(state: PipelineState, output_dir: Path | None = None) -> Path:
    """
    Write all output files for a completed pipeline run.
    Returns the output directory path.
    """
    if output_dir is None:
        output_dir = Path("output") / state.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # post.md
    a6_output = state.agents["A6"].output or ""
    (output_dir / "post.md").write_text(a6_output)

    # verdict.md
    a5_output = state.agents["A5"].output or ""
    (output_dir / "verdict.md").write_text(a5_output)

    # sources.md
    a1_output = state.agents["A1"].output or ""
    (output_dir / "sources.md").write_text(a1_output)

    # audit.json
    audit = _build_audit(state)
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2))

    return output_dir


def write_warnings(
    state: PipelineState,
    warnings_log: list[str],
    output_dir: Path | None = None,
) -> Path:
    """Write warnings.log — always called, even on failure."""
    if output_dir is None:
        output_dir = Path("output") / state.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "warnings.log"
    content = "\n".join(warnings_log) if warnings_log else "(no warnings)\n"
    path.write_text(content)
    return path


def _build_audit(state: PipelineState) -> dict:
    return {
        "run_id": state.run_id,
        "topic": state.topic,
        "audience": state.audience,
        "tone": state.tone,
        "target_word_count": state.target_word_count,
        "pipeline_status": state.pipeline_status.value,
        "started_at": state.started_at,
        "completed_at": state.completed_at,
        "total_tokens": state.total_tokens,
        "agents": {
            aid: {
                "status": rec.status.value,
                "started_at": rec.started_at,
                "completed_at": rec.completed_at,
                "duration_ms": rec.duration_ms,
                "retry_count": rec.retry_count,
                "token_usage": rec.token_usage.model_dump() if rec.token_usage else None,
                "warnings": rec.warnings,
                "error": rec.error,
            }
            for aid, rec in state.agents.items()
        },
    }
