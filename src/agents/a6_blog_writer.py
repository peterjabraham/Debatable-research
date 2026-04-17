import logging
import re

from src.agents.base import BaseAgent
from src.llm.client import LLMClient
from src.pipeline.state import AgentId, AgentStatus, PipelineState
from src.utils.errors import AgentValidationError

logger = logging.getLogger(__name__)

HEDGE_PATTERN = re.compile(
    r"\bit depends\b|\bon the one hand\b|\bnuanced\b|\bboth sides\b|\bcomplex picture\b",
    re.IGNORECASE,
)

EM_DASH = re.compile(r"\s*—\s*")


def _word_count(text: str) -> int:
    return len(text.split())


def _extract_headings(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip().startswith("#")]


def _extract_urls(text: str) -> set[str]:
    return set(re.findall(r"https?://[^\s\)>]+", text))


def _replace_em_dashes(text: str) -> str:
    """Replace em dashes with commas, preserving surrounding whitespace."""
    return EM_DASH.sub(", ", text)


PREAMBLE_RE = re.compile(
    r"^(?:Here (?:is|are) (?:the|my|a) (?:revised|updated|rewritten|complete|full|final)"
    r"|(?:I(?:'ve| have) (?:revised|updated|rewritten|woven|added|expanded|trimmed|removed))"
    r"|(?:Below is (?:the|my|a))"
    r"|(?:The (?:revised|updated|rewritten) (?:post|version|text))"
    r"|(?:Only the affected))"
    r"[^\n]*\n+",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_preamble(text: str) -> str:
    """Remove LLM meta-commentary that precedes the actual post content."""
    cleaned = text
    matched = False
    for _ in range(3):
        m = PREAMBLE_RE.match(cleaned)
        if not m:
            break
        cleaned = cleaned[m.end():]
        matched = True
    return cleaned.strip() if matched else text


def _guard_humanised(
    original: str, humanised: str, target_wc: int
) -> tuple[bool, str]:
    """Check that humanisation preserved structural integrity.

    Returns (passed, reason). If passed is False, caller should fall back
    to the original text.
    """
    orig_headings = _extract_headings(original)
    new_headings = _extract_headings(humanised)
    if len(new_headings) < len(orig_headings):
        return False, f"Lost headings: had {len(orig_headings)}, now {len(new_headings)}"

    orig_urls = _extract_urls(original)
    new_urls = _extract_urls(humanised)
    lost_urls = orig_urls - new_urls
    if lost_urls:
        return False, f"Lost URLs: {lost_urls}"

    wc = _word_count(humanised)
    if wc < target_wc * 0.75 or wc > target_wc * 1.25:
        return False, f"Word count drifted to {wc} (target {target_wc})"

    if HEDGE_PATTERN.search(humanised):
        return False, "Humanisation re-introduced hedge phrases"

    return True, ""


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
        text = _strip_preamble(text)
        target = state.target_word_count
        source_names = _extract_source_names(a1_output)

        _NO_PREAMBLE = (
            "IMPORTANT: Output ONLY the complete blog post from the first heading "
            "to the final sentence. Do NOT include any preamble, commentary, or "
            "notes about what you changed."
        )

        # CP-10a: too short
        if _word_count(text) < target * 0.8:
            reprompt = (
                f"The post below is too short. Expand the weakest section to "
                f"reach ~{target} words, then output the COMPLETE post.\n"
                f"{_NO_PREAMBLE}\n\nCurrent post:\n{text}"
            )
            state.agents["A6"].input = reprompt
            expanded, usage = await self._llm.call("A6", reprompt, signal=signal)
            text = _strip_preamble(expanded)

        # CP-10b: too long
        elif _word_count(text) > target * 1.2:
            reprompt = (
                f"The post below is too long ({_word_count(text)} words). "
                f"Trim it to ~{target} words, then output the COMPLETE post.\n"
                f"{_NO_PREAMBLE}\n\nCurrent post:\n{text}"
            )
            state.agents["A6"].input = reprompt
            trimmed, usage = await self._llm.call("A6", reprompt, signal=signal)
            text = _strip_preamble(trimmed)

        # CP-11: missing concession
        if "concession" not in text.lower():
            reprompt = (
                f"The post below is missing a concession paragraph. Add a "
                f"concession paragraph acknowledging the opposing view, then "
                f"output the COMPLETE post with the concession included.\n"
                f"{_NO_PREAMBLE}\n\nCurrent post:\n{text}"
            )
            state.agents["A6"].input = reprompt
            with_concession, usage = await self._llm.call("A6", reprompt, signal=signal)
            text = _strip_preamble(with_concession)

        # CP-12: source citations
        citation_count, missing_sources = _count_citations(text, source_names)
        if citation_count < 3:
            missing_str = ", ".join(missing_sources[:5])
            reprompt = (
                f"The post below is missing references to these sources: "
                f"{missing_str}. Weave them into the existing text naturally, "
                f"then output the COMPLETE post.\n"
                f"{_NO_PREAMBLE}\n\nCurrent post:\n{text}"
            )
            state.agents["A6"].input = reprompt
            with_citations, usage = await self._llm.call("A6", reprompt, signal=signal)
            text = _strip_preamble(with_citations)

        # CP: hedge phrases
        if HEDGE_PATTERN.search(text):
            reprompt = (
                f"The post below contains hedge phrases. Replace every hedge "
                f"with a clear direct statement, then output the COMPLETE post.\n"
                f"{_NO_PREAMBLE}\n\nCurrent post:\n{text}"
            )
            state.agents["A6"].input = reprompt
            dehedged, usage = await self._llm.call("A6", reprompt, signal=signal)
            text = _strip_preamble(dehedged)

        # ── Humanise + rhythm break ──────────────────────────────────
        pre_humanised = text
        humanise_prompt = (
            "Rewrite the blog post below to read like it was written by an "
            "experienced practitioner, not an AI. Output the COMPLETE rewritten "
            "post and nothing else.\n\n"
            "Rules:\n"
            "- Inject first-person perspective where natural "
            '("We see this during audits", "Most teams hit this wall")\n'
            "- Vary sentence length: mix short punchy sentences with longer "
            "explanatory ones. Add at least two one-sentence paragraphs.\n"
            "- Use conversational transitions and fragments where they add punch\n"
            "- Break any pattern of same-length paragraphs\n"
            "- Replace every em dash (—) with a comma\n"
            "- Keep ALL headings exactly as they are (lines starting with #)\n"
            "- Keep ALL URLs and source citations exactly as they are\n"
            "- Stay within 10% of the current word count\n"
            "- Do NOT add new sections, remove sections, or change the argument\n"
            "- Do NOT introduce hedge phrases: 'it depends', 'on the one hand', "
            "'nuanced', 'both sides', 'complex picture'\n"
            "- Do NOT include any preamble or commentary about changes\n\n"
            f"Current post:\n{text}"
        )
        state.agents["A6"].input = humanise_prompt
        humanised, h_usage = await self._llm.call("A6", humanise_prompt, signal=signal)
        humanised = _strip_preamble(humanised)

        humanised = _replace_em_dashes(humanised)

        passed, reason = _guard_humanised(pre_humanised, humanised, target)
        if passed:
            text = humanised
            usage = h_usage
            logger.info("A6 humanisation passed guard checks")
        else:
            logger.warning("A6 humanisation failed guard: %s — using pre-humanised text", reason)
            text = _replace_em_dashes(pre_humanised)

        state.agents["A6"].output = text
        state.agents["A6"].token_usage = usage
        state.agents["A6"].status = AgentStatus.COMPLETED
        return text
