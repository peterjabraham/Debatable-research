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

    Returns multiple identifier variants per source (domain names, readable
    names derived from URLs, and Core-claim text) so that validation can
    match however the LLM chooses to cite.
    """
    names: list[str] = []

    for line in a1_output.split("\n"):
        url_match = re.search(r"https?://(?:www\.)?([^/\s\)>]+)", line)
        if url_match:
            domain = url_match.group(1).lower().strip()
            names.append(domain)
            # Also add the human-readable site name (e.g. hbr.org -> hbr)
            site = domain.split(".")[0]
            if site and site not in ("com", "org", "net", "io", "co"):
                names.append(site)

    # "Core claim:" lines contain recognisable phrases the LLM may echo
    for line in a1_output.split("\n"):
        m = re.match(r"\s*Core claim:\s*(.+)", line, re.IGNORECASE)
        if m:
            claim = m.group(1).strip()
            if len(claim) > 10:
                names.append(claim[:50])

    # Numbered entry header lines (e.g. "1. Gartner Report on AI")
    for line in a1_output.split("\n"):
        m = re.match(r"^\*{0,2}\d+[\.\)]\*{0,2}\s+(.+)", line.strip())
        if m:
            title = m.group(1).strip().strip("*")
            if title:
                names.append(title[:50])

    if not names:
        for line in a1_output.split("\n"):
            m = re.match(r"^\d+\.\s+(.+)", line.strip())
            if m:
                names.append(m.group(1)[:40].strip())

    return names


def _format_source_ids(source_names: list[str]) -> str:
    """Build a deduplicated, readable list of citation identifiers."""
    seen: set[str] = set()
    unique: list[str] = []
    for n in source_names:
        key = n.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(n)
    return ", ".join(unique[:20])


def _count_blocks(output: str) -> int:
    """Count steelman blocks — matches 'Position:' with optional bold markdown."""
    return len(re.findall(r"^\*{0,2}Position:\*{0,2}", output, re.MULTILINE))


class A4DevilsAdvocate(BaseAgent):
    id: AgentId = "A4"
    timeout_ms: int = 90_000
    max_retries: int = 3

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def build_prompt(
        self,
        state: PipelineState,
        positions: list[str] | None = None,
        source_names: list[str] | None = None,
    ) -> str:
        from src.pipeline.state import require_output
        a3_output = require_output(state, "A3")
        a1_output = require_output(state, "A1")
        if positions is None:
            positions = _extract_contested_positions(a3_output)
        if source_names is None:
            source_names = _extract_source_names(a1_output)
        positions_str = "\n".join(f"{i+1}. {p}" for i, p in enumerate(positions))
        source_ids = _format_source_ids(source_names)
        return (
            f"For each contested position below, write a steelman block with:\n"
            f"Position: / Case: (3 points) / Hardest objection: / Response:\n\n"
            f"Each block MUST cite at least one source by name or domain. "
            f"Valid citation identifiers include: {source_ids}\n"
            f"Use the identifier directly in your text "
            f'(e.g. "according to gartner.com" or "Gartner research shows").\n\n'
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
        blocks = re.split(r"(?=^\*{0,2}Position:\*{0,2})", output, flags=re.MULTILINE)
        blocks = [
            b.strip() for b in blocks
            if b.strip() and re.search(r"\*{0,2}Position:\*{0,2}", b)
        ]
        for block in blocks:
            block_lower = block.lower()
            has_source = any(
                name[:20].lower() in block_lower
                for name in source_names
                if name and len(name.strip()) >= 3
            )
            if not has_source:
                has_source = bool(re.search(r"\(Source \d+\)", block))
            if not has_source:
                has_source = bool(re.search(
                    r"(according to|cited by|per|as reported by|research from|"
                    r"data from|study by|analysis by|report by|findings from)\s+\S+",
                    block_lower,
                ))
            if not has_source:
                has_source = bool(re.search(r"https?://\S+", block))
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

        source_names = _extract_source_names(a1_output)

        prompt = self.build_prompt(state, positions, source_names)
        state.agents["A4"].input = prompt

        text, usage = await self._llm.call("A4", prompt, signal=signal)

        # CP-07: check each block has a source reference
        try:
            self._validate_with_positions(text, positions, source_names, state)
        except AgentValidationError:
            source_ids = _format_source_ids(source_names)
            reprompt = (
                f"{prompt}\n\n"
                f"IMPORTANT — your previous response failed validation because one or "
                f"more steelman blocks did not contain a recognisable source citation.\n"
                f"You MUST mention at least one of these identifiers in EVERY block: "
                f"{source_ids}\n"
                f"Rewrite all blocks now, keeping the same structure."
            )
            state.agents["A4"].input = reprompt
            text, usage = await self._llm.call("A4", reprompt, signal=signal)
            self._validate_with_positions(text, positions, source_names, state)

        state.agents["A4"].output = text
        state.agents["A4"].token_usage = usage
        state.agents["A4"].status = AgentStatus.COMPLETED
        return text
