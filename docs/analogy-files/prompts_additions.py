"""
Additions and modifications to src/llm/prompts.py
==================================================

1. ADD  build_a35_prompt()          — new function, paste in alongside the others
2. MODIFY build_a4_prompt()         — add analogies block to the prompt
3. MODIFY build_a6_prompt()         — add hook_candidate analogy to the prompt

The modifications to A4 and A6 are additive — they inject an optional
{{ANALOGIES}} section that is simply omitted when state.analogies is empty,
so the pipeline degrades gracefully if A3.5 produces nothing.
"""

from __future__ import annotations
import json
from src.pipeline.state import PipelineState


# ---------------------------------------------------------------------------
# 1. NEW — build_a35_prompt
# ---------------------------------------------------------------------------

def build_a35_prompt(state: PipelineState, contested_positions: list[str]) -> str:
    positions_block = "\n".join(
        f"{i+1}. {pos}" for i, pos in enumerate(contested_positions)
    )

    return f"""You are an expert in structural analogy and cross-domain pattern matching.

TOPIC (pillar): {state.topic}

CONTESTED POSITIONS (clusters):
{positions_block}

---

For each contested position, find exactly 2 structural analogies — cases from history,
biology, economics, military strategy, or any other domain that share the SAME UNDERLYING
MECHANISM as that position, even if the surface domain looks completely unrelated.

Requirements:
- Analogies must be surprising and non-obvious. Avoid the first thing that comes to mind.
- Match on STRUCTURE (the mechanism), not surface similarity.
- Each analogy must include an honest account of where it breaks down.
- Across all analogies, vary the domains — don't use the same field twice.

Respond ONLY with valid JSON, no markdown, no backticks, no preamble:

{{
  "analogies": [
    {{
      "position": "exact text of the contested position this analogy belongs to",
      "title": "Short punchy name for the analogy (max 8 words)",
      "domain": "Domain/field (e.g. Industrial History, Biology, Military Strategy)",
      "structural_parallel": "2-3 sentences: what mechanism makes this structurally identical to the position",
      "maps_to": "2-3 sentences: how this maps specifically to the contested position in context of '{state.topic}'",
      "breaks_down": "1 sentence: where this analogy fails"
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# 2. MODIFICATION — build_a4_prompt (Devil's Advocate)
#
# Find your existing build_a4_prompt function and add the ANALOGIES block
# to the prompt string. The change is the _format_analogies_for_a4() call
# injected into the prompt near the end of the instructions.
# ---------------------------------------------------------------------------

def _format_analogies_for_a4(state: PipelineState) -> str:
    """Returns a formatted analogies block for A4, or empty string if none."""
    if not state.analogies:
        return ""

    lines = ["STRUCTURAL ANALOGIES (optional grounding for your steelman arguments):",
             "Where it strengthens the argument, anchor a steelman point in one of these.",
             "Do not force it — only use an analogy if it genuinely clarifies the mechanism.\n"]

    for a in state.analogies:
        lines.append(f"Position: {a.position}")
        lines.append(f"  Analogy: {a.title} [{a.domain}]")
        lines.append(f"  Parallel: {a.structural_parallel}")
        lines.append(f"  Maps to: {a.maps_to}")
        lines.append(f"  Breaks down: {a.breaks_down}\n")

    return "\n".join(lines)


# In your existing build_a4_prompt, add this line near the end of the prompt,
# before the output format instructions:
#
#   {_format_analogies_for_a4(state)}
#
# Example of where it fits in a typical A4 prompt structure:
#
#   ...steelman each of the following contested positions (max 3)...
#   ...each block must cite a named source...
#
#   {_format_analogies_for_a4(state)}      <-- ADD THIS LINE
#
#   Output format:
#   POSITION: ...
#   CASE: ...


# ---------------------------------------------------------------------------
# 3. MODIFICATION — build_a6_prompt (Blog Writer)
#
# Injects the single hook_candidate analogy so A6 can use it as an
# illustration. A6 is instructed to use it — not to force it, but to favour
# the unexpected over the obvious.
# ---------------------------------------------------------------------------

def _format_hook_analogy_for_a6(state: PipelineState) -> str:
    """Returns the hook candidate analogy for A6, or empty string if none."""
    if not state.analogies:
        return ""

    hook = next((a for a in state.analogies if a.hook_candidate), None)
    if not hook:
        return ""

    return f"""ILLUSTRATION ANALOGY (use this in the post):
The post should use the following structural analogy as an illustration — it makes
the argument concrete and memorable. Work it in naturally; don't announce it as an analogy.

Analogy: {hook.title}
Domain: {hook.domain}
The parallel: {hook.structural_parallel}
How it maps here: {hook.maps_to}
Where it breaks down (be honest if you use it): {hook.breaks_down}
"""


# In your existing build_a6_prompt, add this line after the verdict/angle
# section and before the word count / citation instructions:
#
#   {_format_hook_analogy_for_a6(state)}   <-- ADD THIS LINE
#
# Example of where it fits:
#
#   VERDICT: {state.agents["A5"].output}
#
#   {_format_hook_analogy_for_a6(state)}   <-- ADD THIS LINE
#
#   Write a {state.target_word_count}-word blog post...
#   Requirements: hook / body / concession / conclusion
#   >= 3 source citations required
