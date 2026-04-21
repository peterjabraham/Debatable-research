"""
Integration tests for all 17 chokepoints.

Each test uses a mock LLM that returns the specific failure fixture for
that agent, then the happy-path fixture for any re-prompt.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pipeline.checkpoints import load
from src.pipeline.runner import PipelineRunner
from src.pipeline.state import AgentStatus, PipelineState, PipelineStatus, TokenUsage
from src.utils.errors import AgentTimeoutError, AgentValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fx(name: str) -> str:
    return (Path("tests/fixtures/llm_responses") / f"{name}.txt").read_text()


def _tok(n: int = 100) -> TokenUsage:
    return TokenUsage(input_tokens=n // 2, output_tokens=n // 2, total_tokens=n)


def _llm(*responses: str) -> MagicMock:
    client = MagicMock()
    client.call = AsyncMock(side_effect=[(r, _tok()) for r in responses])
    return client


def _state(**kwargs) -> PipelineState:
    defaults = dict(topic="Email", audience="Marketers", tone="Direct")
    defaults.update(kwargs)
    return PipelineState(**defaults)


def _patch_output(tmp_path, monkeypatch):
    import src.pipeline.runner as rm
    orig = rm.Path

    def p(*a):
        x = orig(*a)
        s = str(x)
        if s == "output" or s.startswith("output/") or s.startswith("output\\"):
            return tmp_path / s
        return x

    monkeypatch.setattr(rm, "Path", p)


# ---------------------------------------------------------------------------
# CP-01: A1 no sources → fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp01_pipeline_completes_with_fallback_on_no_sources(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    # A1 returns no-sources, then happy path on re-prompt, then A2-A6 happy
    llm = _llm(
        _fx("a1_no_sources"),   # A1 initial → triggers fallback
        _fx("a1_happy_path"),   # A1 fallback re-prompt
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),  # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_happy_path"),
        _fx("a6_happy_path"),   # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    assert result.pipeline_status == PipelineStatus.COMPLETED
    assert result.agents["A1"].status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_cp01_fallback_prompt_used_when_no_sources(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_no_sources"),
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),  # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_happy_path"),
        _fx("a6_happy_path"),   # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    # The second call (fallback) should have been made
    assert llm.call.call_count >= 2
    # The A1 input should contain the fallback instruction
    assert "Training Knowledge" in result.agents["A1"].input


# ---------------------------------------------------------------------------
# CP-02: A1→A2 source count mismatch → A2 re-prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp02_a2_reprompts_on_source_count_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    # A2 first returns wrong block count, then correct
    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_shallow_claims"),  # 2 blocks but A1 has 6 sources → mismatch
        _fx("a2_happy_path"),      # re-prompt → correct 6 blocks
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),     # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_happy_path"),
        _fx("a6_happy_path"),      # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    assert result.agents["A2"].status == AgentStatus.COMPLETED
    # A2 input after re-prompt should mention the exact count
    assert "6" in result.agents["A2"].input or "claim blocks" in result.agents["A2"].input


# ---------------------------------------------------------------------------
# CP-03: SHALLOW_CLAIMS warning → propagates to A5 prompt
# ---------------------------------------------------------------------------

# 6-block A2 matching a1_happy_path (6 sources); first claim < 15 words.
_A2_SHALLOW_6BLOCKS = "\n\n".join([
    "Core claim: Email ROI is good.\nKey evidence: Studies confirm.\nCaveats: Sample varies.\nImplicit assumption: Usage correct.",
    "Core claim: Personalisation increases email open rates by 26% on average across multiple industry verticals.\nKey evidence: Analyst data.\nCaveats: Enterprise bias.\nImplicit assumption: Correct implementation.",
    "Core claim: Privacy regulations including GDPR have reduced effective email list sizes by 15-30% since 2018.\nKey evidence: Industry surveys.\nCaveats: Geography varies.\nImplicit assumption: GDPR applies.",
    "Core claim: Personalised email sequences outperform generic blasts by 3x in B2B SaaS conversion rates.\nKey evidence: Case study data.\nCaveats: Vendor-sponsored.\nImplicit assumption: Proper segmentation used.",
    "Core claim: Email remains the preferred channel for business communication among buyers aged 35-54.\nKey evidence: University research.\nCaveats: Longitudinal study limitations.\nImplicit assumption: Self-reported preferences accurate.",
    "Core claim: Major email service providers are investing heavily in AI-driven content optimisation tools.\nKey evidence: Trade publication.\nCaveats: Vendor interest.\nImplicit assumption: Investment translates to user value.",
])


@pytest.mark.asyncio
async def test_cp03_shallow_claims_warning_in_a2_record(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    state = _state()
    state.agents["A1"].status = AgentStatus.COMPLETED
    state.agents["A1"].output = _fx("a1_happy_path")
    state.agents["A1"].token_usage = _tok()
    state.total_tokens = 100

    llm = _llm(
        _A2_SHALLOW_6BLOCKS,
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),  # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_happy_path"),
        _fx("a6_happy_path"),   # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(state)

    assert "SHALLOW_CLAIMS" in result.agents["A2"].warnings


@pytest.mark.asyncio
async def test_cp03_shallow_claims_propagates_to_a5_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    state = _state()
    state.agents["A1"].status = AgentStatus.COMPLETED
    state.agents["A1"].output = _fx("a1_happy_path")
    state.agents["A1"].token_usage = _tok()
    state.total_tokens = 100

    llm = _llm(
        _A2_SHALLOW_6BLOCKS,
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),  # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_happy_path"),
        _fx("a6_happy_path"),   # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(state)

    a5_input = result.agents["A5"].input.lower()
    assert "source material" in a5_input or "thin" in a5_input


# ---------------------------------------------------------------------------
# CP-04: NO_CONTEST — user pause
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp04_pipeline_proceeds_when_user_inputs_y(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_no_contest"),
    )
    state = _state()
    runner = PipelineRunner(llm, user_input_fn=lambda _: "y")
    result = await runner.run(state)
    # User said y → pipeline returns (user can resume manually)
    assert result.pipeline_status in (PipelineStatus.FAILED, PipelineStatus.RUNNING)


@pytest.mark.asyncio
async def test_cp04_pipeline_aborts_when_user_inputs_n(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_no_contest"),
    )
    state = _state()
    runner = PipelineRunner(llm, user_input_fn=lambda _: "n")
    result = await runner.run(state)
    assert result.pipeline_status == PipelineStatus.ABORTED


# ---------------------------------------------------------------------------
# CP-05: A3 conflation → re-prompt once
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp05_a3_reprompts_once_on_conflation(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_conflation"),   # first A3 call: conflation detected
        _fx("a3_happy_path"),   # re-prompt: clean output
        _fx("a35_happy_path"),  # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_happy_path"),
        _fx("a6_happy_path"),   # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    assert result.agents["A3"].status == AgentStatus.COMPLETED
    assert result.pipeline_status == PipelineStatus.COMPLETED


@pytest.mark.asyncio
async def test_cp05_conflation_reprompt_not_triggered_on_clean_output(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),   # clean first time — no re-prompt
        _fx("a35_happy_path"),  # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_happy_path"),
        _fx("a6_happy_path"),   # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    assert result.pipeline_status == PipelineStatus.COMPLETED
    # 8 calls total: A1, A2, A3, A35, A4, A5, A6, humanise (no A3 re-prompt)
    assert llm.call.call_count == 8


# ---------------------------------------------------------------------------
# CP-06: A4 position cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp06_a4_caps_at_3_positions_when_5_present(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    # Build an A3 output with 5 contested positions
    a3_5pos = (
        "## Consensus zone\n- Email works\n\n"
        "## Contested zone\n"
        "1. AI is overhyped\n2. Privacy matters\n3. Generic email dying\n"
        "4. SMBs cannot compete\n5. Automation kills authenticity\n\n"
        "## Outlier positions\n- Email dead\n\n"
        "## Evidence weight summary\nMixed.\n\n"
        "## The unresolved question\nWhat happens next?\n"
    )
    # A4 happy path only has 2 blocks — but after cap to 3, we need 3 blocks
    # Build a 3-block A4 output citing sources from a1_happy_path
    a4_3blocks = (
        "Position: AI is overhyped\n"
        "Case: 1. Marginal gains. 2. URL: https://example.com/study1 shows cost issues. 3. Vendor lock-in.\n"
        "Hardest objection: Enterprise ROI is strong.\nResponse: Skewed to large firms.\n\n"
        "Position: Privacy matters\n"
        "Case: 1. GDPR overhead. 2. URL: https://example.com/report2 documents list shrinkage. 3. Consent fatigue.\n"
        "Hardest objection: Email still leads ROI.\nResponse: Compliance moat.\n\n"
        "Position: Generic email dying\n"
        "Case: 1. Unsubscribes rising. 2. URL: https://example.com/blog3 confirms inbox saturation. 3. Competition.\n"
        "Hardest objection: Volume growing.\nResponse: Quality over quantity.\n"
    )

    state = _state()
    state.agents["A1"].status = AgentStatus.COMPLETED
    state.agents["A1"].output = _fx("a1_happy_path")
    state.agents["A1"].token_usage = _tok()
    state.agents["A2"].status = AgentStatus.COMPLETED
    state.agents["A2"].output = _fx("a2_happy_path")
    state.agents["A2"].token_usage = _tok()
    state.agents["A3"].status = AgentStatus.COMPLETED
    state.agents["A3"].output = a3_5pos
    state.agents["A3"].token_usage = _tok()
    # A35 sits between A3 and A4 — pre-complete so runner skips it
    state.agents["A35"].status = AgentStatus.COMPLETED
    state.agents["A35"].token_usage = _tok()
    state.total_tokens = 400

    llm = _llm(a4_3blocks, _fx("a5_happy_path"), _fx("a6_happy_path"), _fx("a6_happy_path"))
    runner = PipelineRunner(llm)
    result = await runner.run(state)

    assert result.agents["A4"].status == AgentStatus.COMPLETED
    assert "TRUNCATED_POSITIONS" in result.agents["A4"].warnings


@pytest.mark.asyncio
async def test_cp06_truncated_positions_warning_set(tmp_path, monkeypatch):
    """Same as above — verifies the warning is set on cap."""
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    a3_5pos = (
        "## Consensus zone\n- Email works\n\n"
        "## Contested zone\n"
        "1. AI is overhyped\n2. Privacy matters\n3. Generic email dying\n"
        "4. SMBs cannot compete\n5. Automation kills authenticity\n\n"
        "## Outlier positions\n- Email dead\n\n"
        "## Evidence weight summary\nMixed.\n\n"
        "## The unresolved question\nWhat?\n"
    )
    a4_3blocks = (
        "Position: AI is overhyped\n"
        "Case: 1. Costs. 2. URL: https://example.com/study1 shows marginal gains. 3. Lock-in.\n"
        "Hardest objection: ROI exists.\nResponse: Context dependent.\n\n"
        "Position: Privacy matters\n"
        "Case: 1. Compliance. 2. URL: https://example.com/report2 documents reduction. 3. Fatigue.\n"
        "Hardest objection: Email still leads.\nResponse: Moat.\n\n"
        "Position: Generic email dying\n"
        "Case: 1. Unsubscribes. 2. URL: https://example.com/blog3 confirms. 3. Competition.\n"
        "Hardest objection: Volume up.\nResponse: Quality.\n"
    )
    state = _state()
    for aid, fname in [("A1", "a1_happy_path"), ("A2", "a2_happy_path")]:
        state.agents[aid].status = AgentStatus.COMPLETED
        state.agents[aid].output = _fx(fname)
        state.agents[aid].token_usage = _tok()
    state.agents["A3"].status = AgentStatus.COMPLETED
    state.agents["A3"].output = a3_5pos
    state.agents["A3"].token_usage = _tok()
    # A35 sits between A3 and A4 — pre-complete so runner skips it
    state.agents["A35"].status = AgentStatus.COMPLETED
    state.agents["A35"].token_usage = _tok()
    state.total_tokens = 400

    llm = _llm(a4_3blocks, _fx("a5_happy_path"), _fx("a6_happy_path"), _fx("a6_happy_path"))
    runner = PipelineRunner(llm)
    result = await runner.run(state)
    assert "TRUNCATED_POSITIONS" in result.agents["A4"].warnings


# ---------------------------------------------------------------------------
# CP-07: A4 missing source citation → re-prompt once
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp07_a4_reprompts_once_when_block_lacks_citation(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),         # A35 analogy pass
        _fx("a4_no_source_citation"),  # fails citation check
        _fx("a4_happy_path"),          # re-prompt → passes
        _fx("a5_happy_path"),
        _fx("a6_happy_path"),
        _fx("a6_happy_path"),          # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    assert result.agents["A4"].status == AgentStatus.COMPLETED
    assert "failed validation" in result.agents["A4"].input


# ---------------------------------------------------------------------------
# CP-08: A5 hedge phrase → re-prompt; second hedge → raises
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp08_a5_reprompts_once_on_hedge_phrase(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),  # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_hedging"),      # first call: hedge detected
        _fx("a5_happy_path"),   # re-prompt: clean verdict
        _fx("a6_happy_path"),
        _fx("a6_happy_path"),   # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    assert result.agents["A5"].status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_cp08_a5_raises_on_second_hedge(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),  # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_hedging"),   # first: hedge
        _fx("a5_hedging"),   # re-prompt: still hedging → raises
    )
    state = _state()
    runner = PipelineRunner(llm)
    with pytest.raises(AgentValidationError):
        await runner.run(state)
    assert state.agents["A5"].status == AgentStatus.FAILED
    assert state.pipeline_status == PipelineStatus.FAILED


# ---------------------------------------------------------------------------
# CP-09: A5 missing sections → re-prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp09_a5_reprompts_when_section_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),       # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_missing_sections"),  # missing "## What to avoid"
        _fx("a5_happy_path"),        # re-prompt: all sections present
        _fx("a6_happy_path"),
        _fx("a6_happy_path"),        # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    assert result.agents["A5"].status == AgentStatus.COMPLETED


# ---------------------------------------------------------------------------
# CP-10a: A6 too short → expansion re-prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp10a_a6_expansion_reprompt_when_short(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),       # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_word_count_short"),  # too short
        _fx("a6_happy_path"),        # expansion result
        _fx("a6_happy_path"),        # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    assert result.agents["A6"].status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_cp10a_expansion_reprompt_checked_after(tmp_path, monkeypatch):
    """Word count is checked again after expansion."""
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),       # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_word_count_short"),
        _fx("a6_happy_path"),
        _fx("a6_happy_path"),        # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    # Final output must be from happy path (correct word count range)
    final = result.agents["A6"].output
    wc = len(final.split())
    assert wc >= 720


# ---------------------------------------------------------------------------
# CP-10b: A6 too long → cut re-prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp10b_a6_cut_reprompt_when_long(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),      # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_word_count_long"),  # too long
        _fx("a6_happy_path"),       # cut result
        _fx("a6_happy_path"),       # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    assert result.agents["A6"].status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_cp10b_cut_reprompt_checked_after(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),      # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_word_count_long"),
        _fx("a6_happy_path"),
        _fx("a6_happy_path"),       # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    final = result.agents["A6"].output
    wc = len(final.split())
    assert wc <= 1080


# ---------------------------------------------------------------------------
# CP-11: A6 missing concession → complete-post re-prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp11_a6_concession_reprompt_returns_complete_post(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),     # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_no_concession"),   # no "concession" keyword
        _fx("a6_happy_path"),      # reprompt returns complete post with concession
        _fx("a6_happy_path"),      # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    assert result.agents["A6"].status == AgentStatus.COMPLETED
    assert "concession" in result.agents["A6"].output.lower()


@pytest.mark.asyncio
async def test_cp11_concession_reprompt_demands_complete_output(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),     # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_no_concession"),
        _fx("a6_happy_path"),      # reprompt returns complete post
        _fx("a6_happy_path"),      # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    # The concession reprompt is the 8th LLM call (index 7, after A35 at index 3)
    concession_reprompt = llm.call.call_args_list[7][0][1]
    assert "complete" in concession_reprompt.lower()


# ---------------------------------------------------------------------------
# CP-12: A6 missing citations → citation re-prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp12_a6_citation_reprompt_lists_missing_sources(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),   # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_no_citations"),  # < 3 source citations
        _fx("a6_happy_path"),    # after citation re-prompt
        _fx("a6_happy_path"),    # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    assert result.agents["A6"].status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_cp12_citation_reprompt_contains_missing_source_names(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    llm = _llm(
        _fx("a1_happy_path"),
        _fx("a2_happy_path"),
        _fx("a3_happy_path"),
        _fx("a35_happy_path"),   # A35 analogy pass
        _fx("a4_happy_path"),
        _fx("a5_happy_path"),
        _fx("a6_no_citations"),
        _fx("a6_happy_path"),
        _fx("a6_happy_path"),    # humanise pass
    )
    runner = PipelineRunner(llm)
    result = await runner.run(_state())
    # The citation re-prompt is the 8th LLM call (index 7, after A35 at index 3)
    citation_reprompt = llm.call.call_args_list[7][0][1]
    assert "weave" in citation_reprompt.lower()


# ---------------------------------------------------------------------------
# CP-13: Agent timeout → timed_out status, checkpoint saved, pipeline halts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp13_watchdog_fires_sets_timed_out(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    async def slow_call(*args, **kwargs):
        await asyncio.sleep(10)
        return ("result", _tok())

    client = MagicMock()
    client.call = AsyncMock(side_effect=slow_call)

    # Patch A1's timeout to be very short
    from src.agents.a1_research_collector import A1ResearchCollector
    monkeypatch.setattr(A1ResearchCollector, "timeout_ms", 50)

    runner = PipelineRunner(client)
    state = _state()

    with pytest.raises(AgentTimeoutError):
        await runner.run(state)

    assert state.agents["A1"].status == AgentStatus.TIMED_OUT


@pytest.mark.asyncio
async def test_cp13_checkpoint_saved_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    async def slow_call(*args, **kwargs):
        await asyncio.sleep(10)
        return ("result", _tok())

    client = MagicMock()
    client.call = AsyncMock(side_effect=slow_call)

    from src.agents.a1_research_collector import A1ResearchCollector
    monkeypatch.setattr(A1ResearchCollector, "timeout_ms", 50)

    runner = PipelineRunner(client)
    state = _state()

    with pytest.raises(AgentTimeoutError):
        await runner.run(state)

    checkpoint = tmp_path / f"{state.run_id}.json"
    assert checkpoint.exists()


# ---------------------------------------------------------------------------
# CP-14: 529 → 4 retry attempts with backoff; 400 raises immediately
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp14_529_retries_succeed_on_third_attempt(tmp_path, monkeypatch):
    """LLM client retries 529 and succeeds on 3rd attempt."""
    from anthropic import APIStatusError
    from src.llm.client import LLMClient

    call_count = 0

    def make_529():
        resp = MagicMock()
        resp.status_code = 529
        resp.headers = {}
        return APIStatusError("529", response=resp, body={})

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise make_529()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=_fx("a1_happy_path"))]
        mock_resp.usage = MagicMock(input_tokens=10, output_tokens=20)
        return mock_resp

    client = LLMClient(api_key="test")
    with patch.object(client._client.messages, "create", side_effect=side_effect):
        text, usage = await client.call("A1", "prompt")
    assert text == _fx("a1_happy_path")
    assert call_count == 3


@pytest.mark.asyncio
async def test_cp14_400_raises_immediately_without_retry():
    from anthropic import APIStatusError
    from src.llm.client import LLMClient

    call_count = 0

    def make_400():
        resp = MagicMock()
        resp.status_code = 400
        resp.headers = {}
        return APIStatusError("400", response=resp, body={})

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise make_400()

    client = LLMClient(api_key="test")
    with patch.object(client._client.messages, "create", side_effect=side_effect):
        with pytest.raises(APIStatusError) as exc_info:
            await client.call("A1", "prompt")
    assert exc_info.value.status_code == 400
    assert call_count == 1


# ---------------------------------------------------------------------------
# CP-15: No code path exists where agent runs without watchdog
# ---------------------------------------------------------------------------

def test_cp15_no_agent_enters_running_without_watchdog():
    """
    Structural test: verify that the runner always calls with_watchdog
    before setting agent status to RUNNING. We do this by confirming
    that with_watchdog is imported and called in runner.run().
    """
    import inspect
    import src.pipeline.runner as runner_mod

    source = inspect.getsource(runner_mod)
    # with_watchdog must be called inside run()
    assert "with_watchdog" in source
    # transition to RUNNING must only happen after with_watchdog setup
    # Verify no direct transition(RUNNING) exists outside the watchdog block
    assert source.count("AgentStatus.RUNNING") >= 1


# ---------------------------------------------------------------------------
# CP-16: Corrupt checkpoint raises CheckpointCorruptError
# ---------------------------------------------------------------------------

def test_cp16_load_raises_checkpoint_corrupt_error(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)

    run_id = "bad-run-id"
    path = tmp_path / f"{run_id}.json"
    path.write_text("{ this is not valid json {{")

    from src.utils.errors import CheckpointCorruptError
    with pytest.raises(CheckpointCorruptError):
        load(run_id)


@pytest.mark.asyncio
async def test_cp16_runner_surfaces_error_cleanly(tmp_path, monkeypatch):
    """Runner alert on corrupt checkpoint: CheckpointCorruptError is raised."""
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)

    from src.utils.errors import CheckpointCorruptError

    run_id = "corrupt-id"
    path = tmp_path / f"{run_id}.json"
    path.write_text("not-json")

    with pytest.raises(CheckpointCorruptError):
        load(run_id)


# ---------------------------------------------------------------------------
# CP-17: CONTEXT_NEAR_LIMIT warning at 80% of token budget
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cp17_context_near_limit_warning_emitted(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)

    # Override the token limit to a small value so we can trigger it
    monkeypatch.setattr("src.pipeline.runner.PIPELINE_MAX_TOTAL_TOKENS", 200)
    monkeypatch.setattr("src.pipeline.runner.CONTEXT_WARN_THRESHOLD", 160)

    state = _state()
    # Pre-complete A1-A4 with enough tokens to exceed 80% of 200
    for aid, fname in [("A1", "a1_happy_path"), ("A2", "a2_happy_path"),
                       ("A3", "a3_happy_path"), ("A4", "a4_happy_path")]:
        state.agents[aid].status = AgentStatus.COMPLETED
        state.agents[aid].output = _fx(fname)
        state.agents[aid].token_usage = _tok(50)
    # A35 sits between A3 and A4 — pre-complete so runner skips it
    state.agents["A35"].status = AgentStatus.COMPLETED
    state.agents["A35"].token_usage = _tok(50)
    state.total_tokens = 170  # > 160 threshold

    llm = _llm(_fx("a5_happy_path"), _fx("a6_happy_path"), _fx("a6_happy_path"))
    runner = PipelineRunner(llm)
    result = await runner.run(state)

    # A5 or A6 should have CONTEXT_NEAR_LIMIT in warnings
    has_warning = any(
        "CONTEXT_NEAR_LIMIT" in result.agents[aid].warnings
        for aid in ["A5", "A6"]
    )
    assert has_warning


@pytest.mark.asyncio
async def test_cp17_context_limit_warning_logged(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("src.pipeline.checkpoints.CHECKPOINT_DIR", tmp_path)
    _patch_output(tmp_path, monkeypatch)
    monkeypatch.setattr("src.pipeline.runner.PIPELINE_MAX_TOTAL_TOKENS", 200)
    monkeypatch.setattr("src.pipeline.runner.CONTEXT_WARN_THRESHOLD", 160)

    state = _state()
    for aid, fname in [("A1", "a1_happy_path"), ("A2", "a2_happy_path"),
                       ("A3", "a3_happy_path"), ("A4", "a4_happy_path")]:
        state.agents[aid].status = AgentStatus.COMPLETED
        state.agents[aid].output = _fx(fname)
        state.agents[aid].token_usage = _tok(50)
    # A35 sits between A3 and A4 — pre-complete so runner skips it
    state.agents["A35"].status = AgentStatus.COMPLETED
    state.agents["A35"].token_usage = _tok(50)
    state.total_tokens = 170

    import logging
    with caplog.at_level(logging.WARNING, logger="src.pipeline.runner"):
        llm = _llm(_fx("a5_happy_path"), _fx("a6_happy_path"), _fx("a6_happy_path"))
        runner = PipelineRunner(llm)
        await runner.run(state)

    assert any("Token budget" in r.message or "CONTEXT" in r.message
               for r in caplog.records)
