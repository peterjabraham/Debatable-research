"""
Parse CLI arguments and construct a PipelineState for a new run.
"""
from __future__ import annotations

from src.pipeline.state import PipelineState


def parse_run_args(
    topic: str,
    audience: str,
    tone: str,
    words: int = 900,
    sources: list[str] | None = None,
    cluster_angle: str | None = None,
) -> PipelineState:
    """Build a fresh PipelineState from CLI arguments."""
    return PipelineState(
        topic=topic,
        audience=audience,
        tone=tone,
        target_word_count=words,
        provided_sources=sources or [],
        cluster_angle=cluster_angle,
    )
