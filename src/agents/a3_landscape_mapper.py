import re

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.pipeline.state import AgentId, AgentStatus, PipelineState
from src.utils.errors import AgentValidationError, PipelineWarning

REQUIRED_HEADINGS = [
    "## Consensus zone",
    "## Contested zone",
    "## Outlier positions",
    "## Evidence weight summary",
    "## The unresolved question",
]


def _extract_section(output: str, heading: str) -> str:
    pattern = re.escape(heading) + r"\s*(.*?)(?=##|$)"
    match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _count_distinct_positions(section: str) -> int:
    """Count distinct positions in a section.

    Handles multiple formats Claude may use:
    - Numbered: '1. Position text' or '**1.**' (bold markdown)
    - Bullets: '- Position text' or '* Position text'
    - Markdown table rows (| col | col |) — counts data rows, not header/separator
    - Labeled: 'Position: ...'
    """
    # Numbered items (plain or bold markdown)
    numbered = re.findall(r"^\*{0,2}\d+[\.\)]\*{0,2}", section, re.MULTILINE)
    if numbered:
        return len(numbered)
    # Markdown table data rows: lines starting with '|' that are not separators
    table_rows = [
        line for line in section.split("\n")
        if line.strip().startswith("|")
        and not re.match(r"^\s*\|[-: |]+\|\s*$", line)  # skip separator rows
        and not re.search(r"\bPosition\b", line)         # skip header row
    ]
    if table_rows:
        return len(table_rows)
    # Explicit position labels
    labeled = re.findall(r"Position:", section)
    if labeled:
        return len(labeled)
    # Bullet points as fallback
    bullets = re.findall(r"^[-*•]\s+\S", section, re.MULTILINE)
    if bullets:
        return len(bullets)
    # Final fallback: substantive prose with no explicit structure
    # If the section has meaningful content, count paragraphs (separated by blank lines)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if len(p.strip()) > 40]
    return len(paragraphs)


def _has_conflation(output: str) -> bool:
    """Check if any position text appears in both Consensus and Contested zones."""
    consensus = _extract_section(output, "## Consensus zone")
    contested = _extract_section(output, "## Contested zone")
    if not consensus or not contested:
        return False
    # Extract bullet points or position labels from each zone
    consensus_items = set(re.findall(r"[-*]\s*(.+)", consensus))
    contested_items = set(re.findall(r"[-*]\s*(.+)", contested))
    return bool(consensus_items & contested_items)


class A3LandscapeMapper(BaseAgent):
    id: AgentId = "A3"
    timeout_ms: int = 60_000
    max_retries: int = 3

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def build_prompt(self, state: PipelineState) -> str:
        from src.pipeline.state import require_output
        a2_output = require_output(state, "A2")
        return (
            f"Map the research landscape from these claims.\n\n"
            f"Output must contain exactly these five sections:\n"
            f"## Consensus zone\n## Contested zone\n## Outlier positions\n"
            f"## Evidence weight summary\n## The unresolved question\n\n"
            f"In the Contested zone, number each distinct position as a list:\n"
            f"1. Position one description\n"
            f"2. Position two description\n"
            f"(and so on)\n\n"
            f"Claims:\n{a2_output}"
        )

    def validate_output(self, output: str, state: PipelineState) -> None:
        for heading in REQUIRED_HEADINGS:
            if heading.lower() not in output.lower():
                raise AgentValidationError("A3", f"Missing section: {heading}")

        contested = _extract_section(output, "## Contested zone")
        position_count = _count_distinct_positions(contested)
        if position_count < 2:
            state.agents["A3"].warnings.append(PipelineWarning.NO_CONTEST)
            raise AgentValidationError(
                "A3", "Contested zone contains fewer than 2 distinct positions"
            )

    async def run(self, state: PipelineState, signal=None) -> str:
        prompt = self.build_prompt(state)
        state.agents["A3"].input = prompt

        text, usage = await self._llm.call("A3", prompt, signal=signal)

        # CP-05: conflation check — re-prompt once
        if _has_conflation(text):
            reprompt = (
                f"{prompt}\n\n"
                f"Important: Do not place the same position in both the Consensus zone "
                f"and the Contested zone. Each position must appear in exactly one zone."
            )
            state.agents["A3"].input = reprompt
            text, usage = await self._llm.call("A3", reprompt, signal=signal)

        self.validate_output(text, state)

        state.agents["A3"].output = text
        state.agents["A3"].token_usage = usage
        state.agents["A3"].status = AgentStatus.COMPLETED
        return text
