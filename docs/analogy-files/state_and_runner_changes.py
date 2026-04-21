"""
Changes to src/pipeline/state.py and src/pipeline/runner.py
============================================================
"""

# ===========================================================================
# src/pipeline/state.py — ADD analogies field
# ===========================================================================
#
# 1. Import the Analogy dataclass at the top of state.py:
#
#    from src.agents.a35_analogy_agent import Analogy
#
# 2. Add AgentId literal (find the existing Literal and add "A35"):
#
#    BEFORE:
#    AgentId = Literal["A1", "A2", "A3", "A4", "A5", "A6"]
#
#    AFTER:
#    AgentId = Literal["A1", "A2", "A3", "A35", "A4", "A5", "A6"]
#
# 3. Add the analogies field to PipelineState (alongside cluster_angle):
#
#    BEFORE:
#    cluster_angle     : str | None   (optional SEO angle)
#
#    AFTER:
#    cluster_angle     : str | None            = None
#    analogies         : list[Analogy] | None  = None
#
# That's it for state.py. The field is optional and defaults to None so
# existing checkpoints deserialise without errors.


# ===========================================================================
# src/pipeline/runner.py — INSERT A35 between A3 and A4
# ===========================================================================
#
# Find the agent sequence list (likely something like):
#
#    AGENT_SEQUENCE = [
#        A1ResearchCollector,
#        A2ClaimExtractor,
#        A3LandscapeMapper,
#        A4DevilsAdvocate,
#        A5EvidenceJudge,
#        A6BlogWriter,
#    ]
#
# Change to:
#
#    from src.agents.a35_analogy_agent import AnalogyAgent   # ADD THIS IMPORT
#
#    AGENT_SEQUENCE = [
#        A1ResearchCollector,
#        A2ClaimExtractor,
#        A3LandscapeMapper,
#        AnalogyAgent,           # <-- INSERT HERE
#        A4DevilsAdvocate,
#        A5EvidenceJudge,
#        A6BlogWriter,
#    ]
#
# Also add A35 to the initial agent record setup (wherever A1-A6 are
# initialised into state.agents):
#
#    BEFORE:
#    agent_ids = ["A1", "A2", "A3", "A4", "A5", "A6"]
#
#    AFTER:
#    agent_ids = ["A1", "A2", "A3", "A35", "A4", "A5", "A6"]


# ===========================================================================
# src/utils/errors.py — ADD two new PipelineWarning values
# ===========================================================================
#
# Find the PipelineWarning enum/constants and add:
#
#    NO_A3_OUTPUT          = "NO_A3_OUTPUT"
#    NO_CONTESTED_POSITIONS = "NO_CONTESTED_POSITIONS"
#
# These allow A3.5 to degrade gracefully without crashing the pipeline.


# ===========================================================================
# LLM config table update (src/llm/client.py or wherever model routing lives)
# ===========================================================================
#
# Add A35 to the model routing config:
#
#    "A35": ModelConfig(
#        model="claude-sonnet-4-6",
#        temperature=0.5,
#        max_tokens=2_000,
#    ),
#
# Full updated table:
#
#    Agent  Model               Temp  Max Tokens  Role
#    A1     claude-sonnet-4-6   0.2   2,000       Research
#    A2     claude-sonnet-4-6   0.1   3,000       Extraction
#    A3     claude-opus-4-5     0.3   2,000       Mapping
#    A35    claude-sonnet-4-6   0.5   2,000       Analogy Finding   <- NEW
#    A4     claude-opus-4-5     0.4   3,000       Argumentation
#    A5     claude-opus-4-5     0.2   1,500       Judgement
#    A6     claude-sonnet-4-6   0.7   4,000       Creative writing
#
# Sonnet rather than Opus for A35 — the task is generative/creative
# rather than analytical, and Sonnet handles far-transfer analogy well
# at lower cost. Temp 0.5 balances creativity with structured JSON output.


# ===========================================================================
# CLI dry-run output — no change needed
# ===========================================================================
#
# The dry-run path in main.py calls build_aX_prompt() for each agent.
# Once build_a35_prompt() is in prompts.py and AnalogyAgent is in the
# sequence, it will be included automatically.


# ===========================================================================
# output_writer.py — optional: write analogies.md
# ===========================================================================
#
# If you want analogies surfaced as a separate output file alongside
# post.md and verdict.md, add to OutputWriter:
#
#    def _write_analogies(self, state: PipelineState) -> None:
#        if not state.analogies:
#            return
#        lines = ["# Structural Analogies\n"]
#        for a in state.analogies:
#            lines.append(f"## {a.title}  [{a.domain}]")
#            lines.append(f"**Position:** {a.position}\n")
#            lines.append(f"**Structural parallel:** {a.structural_parallel}\n")
#            lines.append(f"**Maps to topic:** {a.maps_to}\n")
#            lines.append(f"**Where it breaks down:** {a.breaks_down}\n")
#            if a.hook_candidate:
#                lines.append("_↑ Used as post illustration_\n")
#        path = self.run_dir / "analogies.md"
#        path.write_text("\n".join(lines))
#
# Then call _write_analogies(state) alongside the existing write calls.
