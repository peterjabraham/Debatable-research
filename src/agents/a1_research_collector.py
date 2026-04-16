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
            f"For each source provide: URL, Type, Recency, Core claim, Credibility signal.\n"
            f"Format as a numbered list."
        )

    def validate_output(self, output: str, state: PipelineState) -> None:
        lines = output.split("\n")
        entries = [line for line in lines if re.match(r"^\d+\.", line.strip())]
        if len(entries) < 3:
            raise AgentValidationError("A1", f"Only {len(entries)} source entries found (need ≥ 3)")
        for field in REQUIRED_FIELDS:
            if field not in output:
                raise AgentValidationError("A1", f"Missing required field: {field}")
        if len(entries) < 6:
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
