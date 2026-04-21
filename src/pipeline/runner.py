"""
Orchestrates the six agents in sequence. Responsibilities:
- Enforce state transitions via transition()
- Wrap every agent in with_watchdog()
- Persist checkpoint after every agent completion or failure
- Surface PipelineWarnings to the terminal without halting
- Halt and surface errors cleanly on failures
- Handle NO_CONTEST pause (A3)
- Handle CONTEXT_NEAR_LIMIT warning (any agent)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.agents.a1_research_collector import A1ResearchCollector
from src.agents.a2_claim_extractor import A2ClaimExtractor
from src.agents.a3_landscape_mapper import A3LandscapeMapper
from src.agents.a35_analogy_agent import A35AnalogyAgent
from src.agents.a4_devils_advocate import A4DevilsAdvocate
from src.agents.a5_evidence_judge import A5EvidenceJudge
from src.agents.a6_blog_writer import A6BlogWriter
from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.pipeline.checkpoints import get_resume_point, save
from src.pipeline.state import (
    AGENT_ORDER,
    AgentId,
    AgentStatus,
    PipelineState,
    PipelineStatus,
    can_run,
    transition,
)
from src.pipeline.watchdog import with_watchdog
from src.utils.errors import (
    AgentTimeoutError,
    AgentValidationError,
    PipelineDependencyError,
    PipelineWarning,
)

logger = logging.getLogger(__name__)

PIPELINE_MAX_TOTAL_TOKENS = int(os.getenv("PIPELINE_MAX_TOTAL_TOKENS", "150000"))
CONTEXT_WARN_THRESHOLD = PIPELINE_MAX_TOTAL_TOKENS * 0.8


def _make_agents(
    llm_client: LLMClient,
    research_client: object | None = None,
) -> dict[AgentId, BaseAgent]:
    return {
        "A1": A1ResearchCollector(llm_client, research_client=research_client),
        "A2": A2ClaimExtractor(llm_client),
        "A3": A3LandscapeMapper(llm_client),
        "A35": A35AnalogyAgent(llm_client),
        "A4": A4DevilsAdvocate(llm_client),
        "A5": A5EvidenceJudge(llm_client),
        "A6": A6BlogWriter(llm_client),
    }


class PipelineRunner:
    def __init__(
        self,
        llm_client: LLMClient,
        user_input_fn: Callable[[str], str] | None = None,
        research_client: object | None = None,
    ):
        self._llm = llm_client
        self._agents: dict[AgentId, BaseAgent] = _make_agents(
            llm_client, research_client=research_client
        )
        # user_input_fn used for NO_CONTEST pause prompt
        self._user_input_fn = user_input_fn or input

    async def run(self, state: PipelineState) -> PipelineState:
        """Run the pipeline from the current resume point."""
        resume_from = get_resume_point(state)
        if resume_from is None:
            logger.info("All agents already completed — nothing to do.")
            return state

        warnings_log: list[str] = []

        try:
            for agent_id in AGENT_ORDER:
                record = state.agents[agent_id]

                # Skip completed agents on resume
                if record.status == AgentStatus.COMPLETED:
                    continue

                # Dependency check
                if not can_run(state, agent_id):
                    raise PipelineDependencyError(
                        f"Cannot run {agent_id}: predecessor not completed"
                    )

                agent = self._agents[agent_id]

                # Context near limit warning — add to this agent's record
                if state.total_tokens > CONTEXT_WARN_THRESHOLD:
                    record.warnings.append(PipelineWarning.CONTEXT_NEAR_LIMIT)
                    msg = (
                        f"[{agent_id}] Token budget at "
                        f"{state.total_tokens}/{PIPELINE_MAX_TOTAL_TOKENS} "
                        f"(>{int(CONTEXT_WARN_THRESHOLD)} threshold)"
                    )
                    logger.warning(msg)
                    warnings_log.append(f"{datetime.utcnow().isoformat()} WARNING {msg}")

                # Transition to RUNNING — requires watchdog active
                transition(state, agent_id, AgentStatus.RUNNING)
                save(state)

                _timeout_ms = agent.timeout_ms

                def _on_timeout(aid: str, _tms: int = _timeout_ms) -> None:
                    _aid = aid  # type: ignore[assignment]
                    transition(state, _aid, AgentStatus.TIMED_OUT)  # type: ignore[arg-type]
                    state.agents[_aid].error = f"Timed out after {_tms}ms"  # type: ignore[index]
                    save(state)
                    warnings_log.append(
                        f"{datetime.utcnow().isoformat()} TIMEOUT Agent {aid} timed out"
                    )

                _current_agent = agent

                async def _run_agent(
                    cancel_event: asyncio.Event,
                    _ag: BaseAgent = _current_agent,
                ) -> str:
                    return await _ag.run(state, signal=cancel_event)

                try:
                    await with_watchdog(
                        agent_id,
                        agent.timeout_ms,
                        _run_agent,
                        _on_timeout,
                    )
                except AgentTimeoutError:
                    state.pipeline_status = PipelineStatus.FAILED
                    save(state)
                    self._write_warnings(state, warnings_log)
                    raise

                except AgentValidationError as exc:
                    # CP-04: NO_CONTEST — pause and ask user
                    if agent_id == "A3" and PipelineWarning.NO_CONTEST in state.agents["A3"].warnings:
                        transition(state, agent_id, AgentStatus.FAILED)
                        state.agents[agent_id].error = str(exc)
                        save(state)
                        answer = self._user_input_fn(
                            "No genuine disagreement found on this topic. Continue anyway? (y/n): "
                        ).strip().lower()
                        if answer != "y":
                            state.pipeline_status = PipelineStatus.ABORTED
                            self._write_warnings(state, warnings_log)
                            return state
                        # User said yes — re-try from A3 (they'll need to resume)
                        self._write_warnings(state, warnings_log)
                        return state

                    transition(state, agent_id, AgentStatus.FAILED)
                    state.agents[agent_id].error = str(exc)
                    state.pipeline_status = PipelineStatus.FAILED
                    save(state)
                    self._write_warnings(state, warnings_log)
                    raise

                except Exception as exc:
                    if state.agents[agent_id].status == AgentStatus.RUNNING:
                        transition(state, agent_id, AgentStatus.FAILED)
                        state.agents[agent_id].error = str(exc)
                    state.pipeline_status = PipelineStatus.FAILED
                    save(state)
                    self._write_warnings(state, warnings_log)
                    raise

                # Success — accumulate tokens
                usage = state.agents[agent_id].token_usage
                if usage:
                    state.total_tokens += usage.total_tokens

                # Propagate warnings to log
                for w in record.warnings:
                    msg = f"[{agent_id}] Warning: {w}"
                    logger.warning(msg)
                    warnings_log.append(f"{datetime.utcnow().isoformat()} WARNING {msg}")

                save(state)

            # All agents done
            state.pipeline_status = PipelineStatus.COMPLETED
            state.completed_at = time.time()
            save(state)
            self._write_output(state)

        finally:
            self._write_warnings(state, warnings_log)

        return state

    def _write_warnings(self, state: PipelineState, warnings_log: list[str]) -> None:
        """Write warnings.log — always, even on failure."""
        output_dir = Path("output") / state.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "warnings.log"
        content = "\n".join(warnings_log) if warnings_log else "(no warnings)\n"
        path.write_text(content)

    def _write_output(self, state: PipelineState) -> None:
        """Write final output files on successful completion."""
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

        # analogies.md (optional — only written when A35 produced results)
        if state.analogies:
            lines = ["# Structural Analogies\n"]
            for a in state.analogies:
                lines.append(f"## {a.title}  [{a.domain}]")
                lines.append(f"**Position:** {a.position}\n")
                lines.append(f"**Structural parallel:** {a.structural_parallel}\n")
                lines.append(f"**Maps to topic:** {a.maps_to}\n")
                lines.append(f"**Where it breaks down:** {a.breaks_down}\n")
                if a.hook_candidate:
                    lines.append("_↑ Used as post illustration_\n")
            (output_dir / "analogies.md").write_text("\n".join(lines))

        # audit.json
        import json

        audit = {
            "run_id": state.run_id,
            "topic": state.topic,
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
        (output_dir / "audit.json").write_text(json.dumps(audit, indent=2))
