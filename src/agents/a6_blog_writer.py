import re

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.pipeline.state import AgentId, AgentStatus, PipelineState
from src.utils.errors import AgentValidationError

HEDGE_PATTERN = re.compile(
    r"\bit depends\b|\bon the one hand\b|\bnuanced\b|\bboth sides\b|\bcomplex picture\b",
    re.IGNORECASE,
)


def _word_count(text: str) -> int:
    return len(text.split())


def _extract_source_names(a1_output: str) -> list[str]:
    """Extract source identifiers from A1 output for citation matching.

    Prefers URLs (handles block format where URL is on its own line).
    Falls back to text content from numbered lines when no URLs are present.
    """
    # First pass: collect all URLs from any line
    url_names = []
    for line in a1_output.split("\n"):
        url_match = re.search(r"https?://([^\s\)>]+)", line)
        if url_match:
            url_names.append(url_match.group(1).rstrip("/.,").strip())
    if url_names:
        return url_names
    # Fallback: text content from numbered source lines
    text_names = []
    for line in a1_output.split("\n"):
        m = re.match(r"^\d+\.\s+(.+)", line.strip())
        if m:
            content = m.group(1)
            text_names.append(content[:40].strip())
    return text_names


def _count_citations(post: str, source_names: list[str]) -> tuple[int, list[str]]:
    found = 0
    missing = []
    for name in source_names:
        if name[:15].lower() in post.lower():
            found += 1
        else:
            missing.append(name)
    return found, missing


class A6BlogWriter(BaseAgent):
    id: AgentId = "A6"
    timeout_ms: int = 120_000
    max_retries: int = 2

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def build_prompt(self, state: PipelineState) -> str:
        from src.pipeline.state import require_output
        a5_output = require_output(state, "A5")
        a1_output = require_output(state, "A1")
        a2_output = require_output(state, "A2")
        angle = f"\nCluster angle: {state.cluster_angle}" if state.cluster_angle else ""
        return (
            f"Write a blog post based on the verdict and research below.\n\n"
            f"Target audience: {state.audience}\n"
            f"Tone: {state.tone}\n"
            f"Target word count: {state.target_word_count}{angle}\n\n"
            f"The post must include: hook, concession paragraph (acknowledging opposing view), "
            f"and conclusion. Cite at least 3 sources by name.\n\n"
            f"Verdict:\n{a5_output}\n\n"
            f"Claims:\n{a2_output}\n\n"
            f"Sources:\n{a1_output}"
        )

    def validate_output(self, output: str, state: PipelineState) -> None:
        target = state.target_word_count
        wc = _word_count(output)
        if wc < target * 0.8:
            raise AgentValidationError("A6", f"Word count {wc} below 80% of target {target}")
        if wc > target * 1.2:
            raise AgentValidationError("A6", f"Word count {wc} above 120% of target {target}")
        lower = output.lower()
        for kw in ["hook", "concession", "conclusion"]:
            if kw not in lower:
                raise AgentValidationError("A6", f"Missing required section keyword: {kw}")

    async def run(self, state: PipelineState, signal=None) -> str:
        from src.pipeline.state import require_output
        a1_output = require_output(state, "A1")

        prompt = self.build_prompt(state)
        state.agents["A6"].input = prompt

        text, usage = await self._llm.call("A6", prompt, signal=signal)
        target = state.target_word_count
        source_names = _extract_source_names(a1_output)

        # CP-10a: too short
        if _word_count(text) < target * 0.8:
            reprompt = (
                f"The post is too short. Expand the weakest section to reach ~{target} words. "
                f"Do not regenerate the full post — only expand.\n\nCurrent post:\n{text}"
            )
            state.agents["A6"].input = reprompt
            expanded, usage = await self._llm.call("A6", reprompt, signal=signal)
            text = expanded

        # CP-10b: too long
        elif _word_count(text) > target * 1.2:
            reprompt = (
                f"The post is too long ({_word_count(text)} words). "
                f"Cut it to ~{target} words. Do not regenerate — only trim.\n\nCurrent post:\n{text}"
            )
            state.agents["A6"].input = reprompt
            trimmed, usage = await self._llm.call("A6", reprompt, signal=signal)
            text = trimmed

        # CP-11: missing concession
        if "concession" not in text.lower():
            reprompt = (
                f"The post is missing a concession paragraph. "
                f"Write a concession paragraph that acknowledges the opposing view and append it "
                f"after the current post. Do not regenerate.\n\nCurrent post:\n{text}"
            )
            state.agents["A6"].input = reprompt
            addition, usage = await self._llm.call("A6", reprompt, signal=signal)
            text = text + "\n\n" + addition

        # CP-12: source citations
        citation_count, missing_sources = _count_citations(text, source_names)
        if citation_count < 3:
            missing_str = ", ".join(missing_sources[:5])
            reprompt = (
                f"Weave in references to these sources: {missing_str}. "
                f"Do not regenerate the full post.\n\nCurrent post:\n{text}"
            )
            state.agents["A6"].input = reprompt
            with_citations, usage = await self._llm.call("A6", reprompt, signal=signal)
            text = with_citations

        # CP: hedge phrases — targeted reprompt for specific paragraph
        if HEDGE_PATTERN.search(text):
            reprompt = (
                f"Remove hedge phrases from the post. "
                f"Replace any hedging language with a clear direct statement. "
                f"Do not regenerate.\n\nCurrent post:\n{text}"
            )
            state.agents["A6"].input = reprompt
            dehedged, usage = await self._llm.call("A6", reprompt, signal=signal)
            text = dehedged

        state.agents["A6"].output = text
        state.agents["A6"].token_usage = usage
        state.agents["A6"].status = AgentStatus.COMPLETED
        return text
