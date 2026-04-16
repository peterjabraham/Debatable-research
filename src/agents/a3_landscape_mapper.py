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
    numbered = re.findall(r"^\d+\.", section, re.MULTILINE)
    if numbered:
        return len(numbered)
    labeled = re.findall(r"Position:", section)
    return len(labeled)


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
