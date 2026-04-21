"""
A3.5 — Analogy Agent

Sits between A3 (Landscape Mapper) and A4 (Devil's Advocate).

Reads the contested positions from A3's output, finds 1-2 structural
analogies per position from history or other domains, and writes them
into PipelineState.analogies.

A4 uses the analogies to ground steelman arguments.
A6 uses the hook_candidate analogy as a post illustration.
"""
from __future__ import annotations

import json
import logging
import re

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.pipeline.state import Analogy, AgentId, AgentStatus, PipelineState
from src.utils.errors import AgentValidationError, PipelineWarning

logger = logging.getLogger(__name__)


class A35AnalogyAgent(BaseAgent):
    id: AgentId = "A35"
    timeout_ms: int = 60_000
    max_retries: int = 1

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def build_prompt(self, state: PipelineState) -> str:
        from src.llm.prompts import build_a35_prompt
        positions = _extract_contested_positions(state.agents["A3"].output or "")
        return build_a35_prompt(state, positions)

    def validate_output(self, output: str, state: PipelineState) -> None:
        pass  # Graceful degradation — no hard failures; empty analogies is valid

    async def run(self, state: PipelineState, signal=None) -> str:
        a3_output = state.agents["A3"].output
        if not a3_output:
            state.agents["A35"].warnings.append(PipelineWarning.NO_A3_OUTPUT)
            state.agents["A35"].status = AgentStatus.COMPLETED
            state.analogies = []
            return ""

        positions = _extract_contested_positions(a3_output)
        if not positions:
            state.agents["A35"].warnings.append(PipelineWarning.NO_CONTESTED_POSITIONS)
            logger.warning("A35: No contested positions found in A3 output — skipping analogies")
            state.agents["A35"].status = AgentStatus.COMPLETED
            state.analogies = []
            return ""

        from src.llm.prompts import build_a35_prompt
        prompt = build_a35_prompt(state, positions)
        state.agents["A35"].input = prompt

        response, usage = await self._llm.call("A35", prompt, signal=signal)
        state.agents["A35"].output = response

        analogies = _parse_analogies(response)
        _mark_hook_candidate(analogies)
        state.analogies = analogies

        if not analogies:
            logger.warning("A35: LLM returned no parseable analogies — continuing without them")

        state.agents["A35"].token_usage = usage
        state.agents["A35"].status = AgentStatus.COMPLETED
        return response


# ---------------------------------------------------------------------------
# Parsing helpers (importable for tests)
# ---------------------------------------------------------------------------

def _extract_contested_positions(a3_output: str) -> list[str]:
    """Pull contested positions out of A3's structured output."""
    pattern = re.compile(
        r"contested zone[:\s]*\n(.*?)(?=\n##|\nOUTLIER|\nEVIDENCE|\nUNRESOLVED|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(a3_output)
    if not match:
        return []

    block = match.group(1).strip()
    # Bullet or numbered lines
    positions = re.findall(r"(?:^[-*\d.]+\s+)(.+)", block, re.MULTILINE)
    return [p.strip() for p in positions if len(p.strip()) > 10][:3]


def _parse_analogies(response: str) -> list[Analogy]:
    """Parse the JSON block the LLM returns into Analogy objects."""
    clean = re.sub(r"```(?:json)?|```", "", response).strip()
    json_match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not json_match:
        return []

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    analogies: list[Analogy] = []
    for item in data.get("analogies", []):
        try:
            analogies.append(Analogy(
                position=item["position"],
                title=item["title"],
                domain=item["domain"],
                structural_parallel=item["structural_parallel"],
                maps_to=item["maps_to"],
                breaks_down=item["breaks_down"],
            ))
        except KeyError:
            continue

    return analogies


def _mark_hook_candidate(analogies: list[Analogy]) -> None:
    """Flag the single analogy most suitable as a blog post hook."""
    PREFERRED_DOMAINS = {"industrial history", "economics", "technology", "social history"}
    for analogy in analogies:
        if analogy.domain.lower() in PREFERRED_DOMAINS:
            analogy.hook_candidate = True
            return
    if analogies:
        analogies[0].hook_candidate = True
