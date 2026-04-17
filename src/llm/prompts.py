"""
All 6 agent prompt template functions.
Each accepts a PipelineState and returns the formatted prompt string.
"""
from __future__ import annotations

from src.pipeline.state import PipelineState


def a1_prompt(state: PipelineState) -> str:
    """A1 Research Collector — gather sources for the topic."""
    sources_str = ""
    if state.provided_sources:
        sources_str = "\n\nProvided sources to include:\n" + "\n".join(
            f"- {s}" for s in state.provided_sources
        )
    return (
        f"Research the following topic and compile a numbered list of sources.\n\n"
        f"Topic: {state.topic}{sources_str}\n\n"
        f"For each source provide these five fields:\n"
        f"  URL: <url>\n"
        f"  Type: <Academic|Industry|Analyst|News|Case study>\n"
        f"  Recency: <year>\n"
        f"  Core claim: <one sentence>\n"
        f"  Credibility signal: <why this source is credible>\n\n"
        f"Format as a numbered list. Aim for at least 6 sources."
    )


def a2_prompt(state: PipelineState) -> str:
    """A2 Claim Extractor — extract structured claims from A1 sources."""
    a1_output = state.agents["A1"].output or ""
    return (
        f"For each source in the list below, extract the following in labelled blocks.\n\n"
        f"Required block format:\n"
        f"Core claim: <the central argument the source makes>\n"
        f"Key evidence: <the specific data or evidence cited>\n"
        f"Caveats: <limitations, sample issues, or methodological concerns>\n"
        f"Implicit assumption: <the unstated premise the source relies on>\n\n"
        f"Produce exactly one block per source, in the same order.\n\n"
        f"Sources:\n{a1_output}"
    )


def a3_prompt(state: PipelineState) -> str:
    """A3 Landscape Mapper — map the intellectual terrain of the debate."""
    a2_output = state.agents["A2"].output or ""
    return (
        f"Map the research landscape from the claims below.\n\n"
        f"Your output must contain exactly these five sections in order:\n\n"
        f"## Consensus zone\n"
        f"(positions where the evidence is broadly aligned)\n\n"
        f"## Contested zone\n"
        f"(positions where evidence conflicts — number each distinct position)\n\n"
        f"## Outlier positions\n"
        f"(minority views with some evidential basis)\n\n"
        f"## Evidence weight summary\n"
        f"(which side has stronger or more numerous sources)\n\n"
        f"## The unresolved question\n"
        f"(the core question the evidence cannot yet settle)\n\n"
        f"Claims:\n{a2_output}"
    )


def a4_prompt(state: PipelineState) -> str:
    """A4 Devil's Advocate — steelman each contested position."""
    a3_output = state.agents["A3"].output or ""
    a1_output = state.agents["A1"].output or ""
    return (
        f"For each contested position listed below, write a steelman block.\n\n"
        f"Required block format:\n"
        f"Position: <restate the position clearly>\n"
        f"Case: <three numbered points making the strongest possible case>\n"
        f"Hardest objection: <the best counterargument against this position>\n"
        f"Response: <how a defender of this position would answer that objection>\n\n"
        f"Each block MUST cite at least one source by name or domain from the "
        f"source list below. Use the identifier directly in your text "
        f'(e.g. "according to gartner.com" or "Gartner research shows").\n\n'
        f"Contested positions (from landscape map):\n{a3_output}\n\n"
        f"Source list:\n{a1_output}"
    )


def a5_prompt(state: PipelineState) -> str:
    """A5 Evidence Judge — deliver a clear, committed verdict."""
    a4_output = state.agents["A4"].output or ""
    a1_output = state.agents["A1"].output or ""
    from src.utils.errors import PipelineWarning
    caveat = ""
    if PipelineWarning.SHALLOW_CLAIMS in state.agents["A2"].warnings:
        caveat = (
            "Note: the source material for this topic was thin. "
            "Factor this uncertainty into your verdict.\n\n"
        )
    return (
        f"{caveat}"
        f"Based on the steelman arguments and source list below, deliver a verdict.\n\n"
        f"Your output must contain exactly these five sections:\n\n"
        f"## Verdict\n"
        f"(a single sentence committing to one position — no hedging)\n\n"
        f"## Three strongest reasons\n"
        f"(numbered list of the three pieces of evidence most supporting the verdict)\n\n"
        f"## Honest concession\n"
        f"(the strongest point against the verdict — acknowledge it directly)\n\n"
        f"## The angle\n"
        f"(the positioning lens for the blog post — what makes this verdict interesting)\n\n"
        f"## What to avoid\n"
        f"(arguments or framings that would undermine the verdict's credibility)\n\n"
        f"Steelman arguments:\n{a4_output}\n\n"
        f"Source list:\n{a1_output}"
    )


def a6_prompt(state: PipelineState) -> str:
    """A6 Blog Writer — write the positioned blog post."""
    a5_output = state.agents["A5"].output or ""
    a1_output = state.agents["A1"].output or ""
    a2_output = state.agents["A2"].output or ""
    angle = f"\nCluster angle: {state.cluster_angle}" if state.cluster_angle else ""
    return (
        f"Write a blog post based on the verdict and research below.\n\n"
        f"Target audience: {state.audience}\n"
        f"Tone: {state.tone}\n"
        f"Target word count: {state.target_word_count}{angle}\n\n"
        f"The post must include:\n"
        f"- An engaging hook opening\n"
        f"- A concession paragraph that genuinely acknowledges the opposing view\n"
        f"- A conclusion that reinforces the verdict\n"
        f"- At least 3 named source citations woven into the text\n\n"
        f"Verdict and positioning:\n{a5_output}\n\n"
        f"Source claims:\n{a2_output}\n\n"
        f"Source list:\n{a1_output}"
    )


ALL_PROMPT_FUNCTIONS = {
    "A1": a1_prompt,
    "A2": a2_prompt,
    "A3": a3_prompt,
    "A4": a4_prompt,
    "A5": a5_prompt,
    "A6": a6_prompt,
}
