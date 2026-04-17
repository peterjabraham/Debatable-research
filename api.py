"""
FastAPI backend for the Debatable Research pipeline.

Endpoints:
  POST /topics/refine     — GPT-4.1 topic expansion (pillar + clusters)
  POST /jobs              — start a new pipeline run (async background task)
  GET  /jobs/{run_id}     — polling: read checkpoint, return status + results
  GET  /health            — Railway health check
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv(override=True)

logging.basicConfig(level=os.getenv("PIPELINE_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(title="Debatable Research API", version="1.0.0")

# ---------------------------------------------------------------------------
# CORS — allow Cloudflare Pages domain (and localhost for dev)
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RefineRequest(BaseModel):
    query: str


class JobRequest(BaseModel):
    pillar_topic: str
    cluster_topics: list[str]        # 1+ selected clusters
    audience: str
    tone: str
    words: int = 900
    sources: list[str] = []


class AgentStatusOut(BaseModel):
    id: str
    status: str
    token_usage: dict[str, int] | None = None
    warnings: list[str] = []
    error: str | None = None
    duration_ms: float | None = None


class JobStatusOut(BaseModel):
    run_id: str
    pipeline_status: str
    total_tokens: int
    agents: list[AgentStatusOut]
    post: str | None = None
    verdict: str | None = None
    sources: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_topic(pillar: str, clusters: list[str]) -> str:
    """
    Combine pillar + cluster selections into a rich topic string for A1.
    1 cluster → use cluster directly (focused).
    2+ clusters → prefix with pillar and list research angles.
    """
    if len(clusters) == 1:
        return clusters[0]
    angles = "\n".join(f"- {c}" for c in clusters)
    return f"{pillar}\n\nResearch angles:\n{angles}"


def _get_llm_client() -> Any:
    from src.llm.client import LLMClient
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
    return LLMClient(api_key=api_key)


def _get_research_client() -> Any | None:
    """Instantiate Perplexity client if PERPLEXITY_API_KEY is set, else None."""
    api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    if not api_key:
        logger.info("PERPLEXITY_API_KEY not set — A1 will use Claude for research")
        return None
    from src.llm.perplexity import PerplexityClient
    return PerplexityClient(api_key=api_key)


async def _run_pipeline(run_id: str) -> None:
    """Background task: load checkpoint and run the pipeline to completion."""
    from src.pipeline.checkpoints import load
    from src.pipeline.runner import PipelineRunner
    try:
        state = load(run_id)
        llm = _get_llm_client()
        research = _get_research_client()
        runner = PipelineRunner(llm, research_client=research)
        await runner.run(state)
    except Exception:
        logger.exception("Pipeline run %s failed", run_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    """Serve the UI with RAILWAY_API_URL injected as a meta tag."""
    ui_path = Path(__file__).parent / "ui" / "index.html"
    html = ui_path.read_text()
    api_base = os.getenv("RAILWAY_API_URL", "")
    meta_tag = f'<meta name="api-base" content="{api_base}">'
    html = html.replace("<head>", f"<head>\n  {meta_tag}", 1)
    return HTMLResponse(content=html)


@app.post("/topics/refine")
async def refine_topic(body: RefineRequest) -> dict[str, Any]:
    """Call GPT-4.1 to expand a rough query into a pillar + cluster topics."""
    from src.topic_refiner import refine
    try:
        options = await refine(body.query.strip())
        return options.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/jobs", status_code=202)
async def create_job(body: JobRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    """
    Start a new pipeline run.
    Returns immediately with run_id; poll GET /jobs/{run_id} for status.
    """
    from src.io.input_parser import parse_run_args
    from src.pipeline.checkpoints import save

    topic = _build_topic(body.pillar_topic, body.cluster_topics)

    state = parse_run_args(
        topic=topic,
        audience=body.audience,
        tone=body.tone,
        words=body.words,
        sources=body.sources or [],
        cluster_angle=body.pillar_topic,   # pillar → SEO framing hint for A6
    )

    # Save initial checkpoint so GET /jobs/{id} works immediately
    save(state)

    background_tasks.add_task(_run_pipeline, state.run_id)

    return {
        "run_id": state.run_id,
        "status_url": f"/jobs/{state.run_id}",
    }


@app.get("/jobs/{run_id}", response_model=JobStatusOut)
async def get_job(run_id: str) -> JobStatusOut:
    """Read checkpoint and return current pipeline status."""
    from src.pipeline.checkpoints import load
    from src.utils.errors import CheckpointCorruptError

    try:
        state = load(run_id)
    except CheckpointCorruptError:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    agents_out = [
        AgentStatusOut(
            id=rec.id,
            status=rec.status.value,
            token_usage=rec.token_usage.model_dump() if rec.token_usage else None,
            warnings=rec.warnings,
            error=rec.error,
            duration_ms=rec.duration_ms,
        )
        for rec in state.agents.values()
    ]

    completed = state.pipeline_status.value == "completed"

    return JobStatusOut(
        run_id=state.run_id,
        pipeline_status=state.pipeline_status.value,
        total_tokens=state.total_tokens,
        agents=agents_out,
        post=state.agents["A6"].output if completed else None,
        verdict=state.agents["A5"].output if completed else None,
        sources=state.agents["A1"].output if completed else None,
    )


@app.post("/jobs/{run_id}/resume", status_code=202)
async def resume_job(run_id: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Resume a paused or failed pipeline run from its checkpoint."""
    from src.pipeline.checkpoints import load
    from src.utils.errors import CheckpointCorruptError

    try:
        load(run_id)  # validate it exists
    except CheckpointCorruptError:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    background_tasks.add_task(_run_pipeline, run_id)
    return {"run_id": run_id, "status_url": f"/jobs/{run_id}"}
