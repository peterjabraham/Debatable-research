import logging
import re

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.pipeline.state import AgentId, AgentStatus, PipelineState
from src.utils.errors import AgentValidationError, PipelineWarning

logger = logging.getLogger(__name__)

MAX_POSITIONS = 3


def _extract_contested_positions(a3_output: str) -> list[str]:
    """Extract contested positions from A3 output."""
    match = re.search(
        r"## Contested zone\s*(.*?)(?=##|$)", a3_output, re.DOTALL | re.IGNORECASE
    )
    if not match:
        return []
    section = match.group(1).strip()
    # Numbered items: "1. text" or "1." on its own line followed by bold text
    numbered = re.findall(r"^\d+\.\s*\*{0,2}(.+?)\*{0,2}\s*$", section, re.MULTILINE)
    if numbered:
        return [p.strip() for p in numbered if p.strip()]
    # Table rows: extract first column
    table_rows = []
    for line in section.split("\n"):
        if line.strip().startswith("|") and not re.match(r"^\s*\|[-: |]+\|\s*$", line):
            cols = [c.strip().strip("*") for c in line.split("|") if c.strip()]
            if cols and "Position" not in cols[0]:
                table_rows.append(cols[0])
    if table_rows:
        return table_rows
    # Position: labels
    labeled = re.findall(r"Position:\s*(.+)", section)
    return labeled


def _extract_source_names(a1_output: str) -> list[str]:
    """Extract source identifiers from A1 output for citation matching.

    Returns short identifiers (domain names from URLs, or text prefixes)
    that are likely to appear in natural-language citation text.
    """
    # Collect domain names from any URL in the output
    domain_names = []
    for line in a1_output.split("\n"):
        url_match = re.search(r"https?://(?:www\.)?([^/\s\)>]+)", line)
        if url_match:
            domain_names.append(url_match.group(1).lower().strip())
    if domain_names:
        return domain_names
    # Fallback: text content from numbered source lines
    text_names = []
    for line in a1_output.split("\n"):
        m = re.match(r"^\d+\.\s+(.+)", line.strip())
        if m:
            content = m.group(1)
            text_names.append(content[:40].strip())
    return text_names


def _count_blocks(output: str) -> int:
    """Count steelman blocks — matches 'Position:' with optional bold markdown."""
    return len(re.findall(r"^\*{0,2}Position:\*{0,2}", output, re.MULTILINE))


class A4DevilsAdvocate(BaseAgent):
    id: AgentId = "A4"
    timeout_ms: int = 90_000
    max_retries: int = 3

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def build_prompt(self, state: PipelineState, positions: list[str] | None = None) -> str:
        from src.pipeline.state import require_output
        a3_output = require_output(state, "A3")
        a1_output = require_output(state, "A1")
        if positions is None:
            positions = _extract_contested_positions(a3_output)
        positions_str = "\n".join(f"{i+1}. {p}" for i, p in enumerate(positions))
        return (
            f"For each contested position below, write a steelman block with:\n"
            f"Position: / Case: (3 points) / Hardest objection: / Response:\n\n"
            f"Each block must cite at least one named source from the source list.\n\n"
            f"Contested positions:\n{positions_str}\n\n"
            f"Source list:\n{a1_output}"
        )

    def validate_output(self, output: str, state: PipelineState) -> None:
        pass  # validation done inline in run() with reprompt logic

    def _validate_with_positions(
        self, output: str, positions: list[str], source_names: list[str], state: PipelineState
    ) -> None:
        block_count = _count_blocks(output)
        if block_count != len(positions):
            raise AgentValidationError(
                "A4",
                f"Block count mismatch: expected {len(positions)}, got {block_count}",
            )
        # Each block must reference at least one source
        for name in source_names:
            # We just check the whole output contains at least some source references
            break
        # Split output into blocks — handles plain and bold '**Position:**'
        blocks = re.split(r"(?=^\*{0,2}Position:\*{0,2})", output, flags=re.MULTILINE)
        # Keep only blocks that actually contain a Position: label (skip preambles/headers)
        blocks = [
            b.strip() for b in blocks
            if b.strip() and re.search(r"\*{0,2}Position:\*{0,2}", b)
        ]
        for block in blocks:
            block_lower = block.lower()
            # Check domain names or text names appear in block
            has_source = any(
                name[:20].lower() in block_lower for name in source_names if name
            )
            # Also accept "(Source N)" style citations Claude commonly uses
            if not has_source:
                has_source = bool(re.search(r"\(Source \d+\)", block))
            if not has_source:
                raise AgentValidationError(
                    "A4",
                    "A steelman block lacks a source citation from the source list",
                )

    async def run(self, state: PipelineState, signal=None) -> str:
        from src.pipeline.state import require_output
        a3_output = require_output(state, "A3")
        a1_output = require_output(state, "A1")

        positions = _extract_contested_positions(a3_output)

        # CP-06: cap at MAX_POSITIONS
        if len(positions) > MAX_POSITIONS:
            dropped = positions[MAX_POSITIONS:]
            positions = positions[:MAX_POSITIONS]
            state.agents["A4"].warnings.append(PipelineWarning.TRUNCATED_POSITIONS)
            logger.warning("A4: Dropped positions: %s", dropped)

        prompt = self.build_prompt(state, positions)
        state.agents["A4"].input = prompt

        text, usage = await self._llm.call("A4", prompt, signal=signal)

        source_names = _extract_source_names(a1_output)

        # CP-07: check each block has a source reference
        try:
            self._validate_with_positions(text, positions, source_names, state)
        except AgentValidationError:
            reprompt = (
                f"{prompt}\n\n"
                f"Each steelman block must cite at least one named source from the source list."
            )
            state.agents["A4"].input = reprompt
            text, usage = await self._llm.call("A4", reprompt, signal=signal)
            self._validate_with_positions(text, positions, source_names, state)

        state.agents["A4"].output = text
        state.agents["A4"].token_usage = usage
        state.agents["A4"].status = AgentStatus.COMPLETED
        return text
