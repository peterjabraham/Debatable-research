"""
A3.5 — Analogy Agent

Sits between A3 (Landscape Mapper) and A4 (Devil's Advocate).

Reads the contested positions from A3's output, finds 1-2 structural analogies
per position from history or other domains, and writes them into PipelineState.

A4 uses the analogies to ground steelman arguments.
A6 uses the hook_candidate analogy as a post illustration.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.agents.base import BaseAgent
from src.pipeline.state import PipelineState
from src.llm.prompts import build_a35_prompt
from src.utils.errors import PipelineWarning


AGENT_ID = "A35"
MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0.5
MAX_TOKENS = 2_000


@dataclass
class Analogy:
    position: str           # which contested position this analogy belongs to
    title: str              # short punchy name, e.g. "The Steam-to-Dynamo Swap"
    domain: str             # e.g. "Industrial History", "Biology", "Military Strategy"
    structural_parallel: str   # the mechanism that is structurally identical
    maps_to: str            # how it maps to this position in context of the topic
    breaks_down: str        # where the analogy fails — keeps usage honest
    hook_candidate: bool = False  # True for the single strongest cross-run analogy


class AnalogyAgent(BaseAgent):
    """
    A3.5 — finds unexpected structural analogies for each contested position.
    Stores results in state.analogies for use by A4 and A6.
    """

    agent_id = AGENT_ID

    async def run(self, state: PipelineState, signal) -> None:
        a3_output = state.agents["A3"].output
        if not a3_output:
            state.agents[AGENT_ID].warnings.append(
                PipelineWarning.NO_A3_OUTPUT
            )
            return

        contested_positions = _extract_contested_positions(a3_output)

        if not contested_positions:
            state.agents[AGENT_ID].warnings.append(
                PipelineWarning.NO_CONTESTED_POSITIONS
            )
            # Graceful degradation — pipeline continues without analogies
            state.analogies = []
            return

        prompt = build_a35_prompt(state, contested_positions)
        response = await self.llm.complete(
            prompt=prompt,
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        state.agents[AGENT_ID].output = response
        analogies = _parse_analogies(response, contested_positions)
        _mark_hook_candidate(analogies)
        state.analogies = analogies


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_contested_positions(a3_output: str) -> list[str]:
    """
    Pull the contested positions out of A3's structured output.

    A3 produces five sections; we want 'Contested zone'.
    Handles both '## Contested zone' and 'CONTESTED ZONE:' heading styles.
    """
    pattern = re.compile(
        r"contested zone[:\s]*\n(.*?)(?=\n##|\nOUTLIER|\nEVIDENCE|\nUNRESOLVED|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(a3_output)
    if not match:
        return []

    block = match.group(1).strip()
    # Each position is a bullet or numbered line
    positions = re.findall(r"(?:^[-*\d.]+\s+)(.+)", block, re.MULTILINE)
    return [p.strip() for p in positions if len(p.strip()) > 10][:3]  # cap at 3


def _parse_analogies(response: str, positions: list[str]) -> list[Analogy]:
    """
    Parse the JSON block the LLM returns.
    Falls back to an empty list rather than crashing the pipeline.
    """
    # Strip any accidental markdown fences
    clean = re.sub(r"```(?:json)?|```", "", response).strip()

    # Find the JSON object
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
            analogies.append(
                Analogy(
                    position=item["position"],
                    title=item["title"],
                    domain=item["domain"],
                    structural_parallel=item["structural_parallel"],
                    maps_to=item["maps_to"],
                    breaks_down=item["breaks_down"],
                )
            )
        except KeyError:
            continue  # skip malformed entries silently

    return analogies


def _mark_hook_candidate(analogies: list[Analogy]) -> None:
    """
    Flag the single analogy most suitable as a blog post hook.

    Heuristic: prefer analogies whose domain is 'Industrial History',
    'Economics', or 'Technology' — they tend to be accessible to general
    audiences. Fall back to the first analogy if none match.
    """
    PREFERRED_DOMAINS = {"industrial history", "economics", "technology", "social history"}
    for analogy in analogies:
        if analogy.domain.lower() in PREFERRED_DOMAINS:
            analogy.hook_candidate = True
            return
    if analogies:
        analogies[0].hook_candidate = True
