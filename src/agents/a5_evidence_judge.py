import re

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.pipeline.state import AgentId, AgentStatus, PipelineState
from src.utils.errors import AgentValidationError, PipelineWarning

REQUIRED_HEADINGS = [
    "## Verdict",
    "## Three strongest reasons",
    "## Honest concession",
    "## The angle",
    "## What to avoid",
]

HEDGE_PATTERN = re.compile(
    r"\bit depends\b|\bon the one hand\b|\bnuanced\b|\bboth sides\b|\bcomplex picture\b",
    re.IGNORECASE,
)


def _extract_verdict(output: str) -> str:
    match = re.search(
        r"## Verdict\s*(.*?)(?=##|$)", output, re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else ""


class A5EvidenceJudge(BaseAgent):
    id: AgentId = "A5"
    timeout_ms: int = 60_000
    max_retries: int = 3

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def build_prompt(self, state: PipelineState) -> str:
        from src.pipeline.state import require_output
        a4_output = require_output(state, "A4")
        a1_output = require_output(state, "A1")

        caveat = ""
        if PipelineWarning.SHALLOW_CLAIMS in state.agents["A2"].warnings:
            caveat = (
                "Note: the source material for this topic was thin. "
                "Factor this uncertainty into your verdict.\n\n"
            )

        return (
            f"{caveat}"
            f"Based on the steelman arguments and source list below, deliver a verdict.\n\n"
            f"Output must contain exactly:\n"
            f"## Verdict\n## Three strongest reasons\n## Honest concession\n"
            f"## The angle\n## What to avoid\n\n"
            f"## Verdict must be a single sentence committing to one position.\n\n"
            f"Steelman arguments:\n{a4_output}\n\n"
            f"Source list:\n{a1_output}"
        )

    def validate_output(self, output: str, state: PipelineState) -> None:
        for heading in REQUIRED_HEADINGS:
            if heading.lower() not in output.lower():
                raise AgentValidationError("A5", f"Missing section: {heading}")
        verdict = _extract_verdict(output)
        if HEDGE_PATTERN.search(verdict):
            raise AgentValidationError("A5", "Verdict contains hedge phrase")

    async def run(self, state: PipelineState, signal=None) -> str:
        prompt = self.build_prompt(state)
        state.agents["A5"].input = prompt

        text, usage = await self._llm.call("A5", prompt, signal=signal)

        # CP-09: check all headings present
        for heading in REQUIRED_HEADINGS:
            if heading.lower() not in text.lower():
                reprompt = (
                    f"{prompt}\n\nYour previous response was missing required sections. "
                    f"Include all five sections: "
                    + ", ".join(REQUIRED_HEADINGS)
                )
                state.agents["A5"].input = reprompt
                text, usage = await self._llm.call("A5", reprompt, signal=signal)
                break

        # CP-08: hedge detection — max one re-prompt
        verdict = _extract_verdict(text)
        if HEDGE_PATTERN.search(verdict):
            reprompt = (
                f"{prompt}\n\n"
                f"Commit to the position the weight of evidence most supports. "
                f"A single clear sentence. No hedging."
            )
            state.agents["A5"].input = reprompt
            text, usage = await self._llm.call("A5", reprompt, signal=signal)
            verdict2 = _extract_verdict(text)
            if HEDGE_PATTERN.search(verdict2):
                raise AgentValidationError(
                    "A5", "Verdict still contains hedge phrase after re-prompt"
                )

        # Final full validation
        self.validate_output(text, state)

        state.agents["A5"].output = text
        state.agents["A5"].token_usage = usage
        state.agents["A5"].status = AgentStatus.COMPLETED
        return text
