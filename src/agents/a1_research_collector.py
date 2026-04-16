import re

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.pipeline.state import AgentId, AgentStatus, PipelineState
from src.utils.errors import AgentValidationError, PipelineWarning

FALLBACK_A1 = (
    "No live sources were found for this topic. Using training knowledge instead.\n"
    "Please provide at least 3 sources from your training knowledge. "
    "Label each source as [Training Knowledge - unverified].\n\n"
    "Format each source as a numbered list entry with: URL, Type, Recency, "
    "Core claim, Credibility signal."
)

REQUIRED_FIELDS = ["URL", "Type", "Recency", "Core claim", "Credibility signal"]


def _count_numbered_entries(text: str) -> int:
    """Count numbered source entries, handling both '1.' and '**1.**' formats."""
    count = 0
    for line in text.split("\n"):
        stripped = line.strip()
        # Plain: "1. text" or "1) text"
        # Bold markdown: "**1.**" or "**1)**"
        if re.match(r"^\d+[\.\)]", stripped) or re.match(r"^\*{1,2}\d+[\.\)]\*{0,2}", stripped):
            count += 1
    return count


class A1ResearchCollector(BaseAgent):
    id: AgentId = "A1"
    timeout_ms: int = 90_000
    max_retries: int = 2

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def build_prompt(self, state: PipelineState) -> str:
        sources_str = ""
        if state.provided_sources:
            sources_str = "\n\nProvided sources to include:\n" + "\n".join(
                f"- {s}" for s in state.provided_sources
            )
        return (
            f"Research the following topic and compile a numbered list of sources.\n\n"
            f"Topic: {state.topic}{sources_str}\n\n"
            f"For each source, provide the following five fields on separate lines:\n"
            f"  URL: <url>\n"
            f"  Type: <Academic|Industry|Analyst|News|Case study>\n"
            f"  Recency: <year>\n"
            f"  Core claim: <one sentence>\n"
            f"  Credibility signal: <why credible>\n\n"
            f"Format each entry as a numbered item (1., 2., 3., ...). "
            f"Aim for at least 6 sources."
        )

    def validate_output(self, output: str, state: PipelineState) -> None:
        entry_count = _count_numbered_entries(output)
        if entry_count < 3:
            raise AgentValidationError("A1", f"Only {entry_count} source entries found (need ≥ 3)")
        output_lower = output.lower()
        for field in REQUIRED_FIELDS:
            if field.lower() not in output_lower:
                raise AgentValidationError("A1", f"Missing required field: {field}")
        if entry_count < 6:
            state.agents["A1"].warnings.append(PipelineWarning.CONTEXT_NEAR_LIMIT)

    def _no_sources_detected(self, output: str) -> bool:
        lower = output.lower()
        return "could not find" in lower or "no sources" in lower

    async def run(self, state: PipelineState, signal=None) -> str:
        prompt = self.build_prompt(state)
        state.agents["A1"].input = prompt

        text, usage = await self._llm.call("A1", prompt, signal=signal)

        # CP-01: fallback if no sources found
        if self._no_sources_detected(text):
            fallback_prompt = f"{prompt}\n\n{FALLBACK_A1}"
            state.agents["A1"].input = fallback_prompt
            text, usage = await self._llm.call("A1", fallback_prompt, signal=signal)

        self.validate_output(text, state)

        state.agents["A1"].output = text
        state.agents["A1"].token_usage = usage
        state.agents["A1"].status = AgentStatus.COMPLETED
        return text
