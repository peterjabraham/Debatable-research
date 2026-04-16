import re

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.pipeline.state import AgentId, AgentStatus, PipelineState
from src.utils.errors import AgentValidationError, PipelineWarning


def _count_source_entries(text: str) -> int:
    """Count numbered source entries, handling both '1.' and '**1.**' formats."""
    count = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if re.match(r"^\d+[\.\)]", stripped) or re.match(r"^\*{1,2}\d+[\.\)]\*{0,2}", stripped):
            count += 1
    return count


def _count_claim_blocks(text: str) -> int:
    return len(re.findall(r"Core claim:", text))


class A2ClaimExtractor(BaseAgent):
    id: AgentId = "A2"
    timeout_ms: int = 60_000
    max_retries: int = 3

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def build_prompt(self, state: PipelineState) -> str:
        from src.pipeline.state import require_output
        a1_output = require_output(state, "A1")
        return (
            f"For each source below, extract the following in labelled blocks:\n"
            f"Core claim: / Key evidence: / Caveats: / Implicit assumption:\n\n"
            f"Sources:\n{a1_output}"
        )

    def validate_output(self, output: str, state: PipelineState) -> None:
        a1_output = state.agents["A1"].output or ""
        source_count = _count_source_entries(a1_output)
        block_count = _count_claim_blocks(output)
        if block_count != source_count:
            raise AgentValidationError(
                "A2",
                f"Block count mismatch: expected {source_count}, got {block_count}",
            )
        # CP-03: check each Core claim length
        for match in re.finditer(r"Core claim:\s*(.+?)(?=Key evidence:|Caveats:|$)", output, re.DOTALL):
            claim_text = match.group(1).strip()
            word_count = len(claim_text.split())
            if word_count < 15:
                state.agents["A2"].warnings.append(PipelineWarning.SHALLOW_CLAIMS)
                break  # one warning is enough

    async def run(self, state: PipelineState, signal=None) -> str:
        from src.pipeline.state import require_output
        a1_output = require_output(state, "A1")
        source_count = _count_source_entries(a1_output)

        prompt = self.build_prompt(state)
        state.agents["A2"].input = prompt

        text, usage = await self._llm.call("A2", prompt, signal=signal)

        # CP-02: source count mismatch re-prompt
        if _count_claim_blocks(text) != source_count:
            reprompt = (
                f"{prompt}\n\n"
                f"The previous source list contained {source_count} sources. "
                f"Your output must contain exactly {source_count} claim blocks."
            )
            state.agents["A2"].input = reprompt
            text, usage = await self._llm.call("A2", reprompt, signal=signal)

        self.validate_output(text, state)

        state.agents["A2"].output = text
        state.agents["A2"].token_usage = usage
        state.agents["A2"].status = AgentStatus.COMPLETED
        return text
