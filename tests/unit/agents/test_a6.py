"""Unit tests for A6BlogWriter (spec §12 A6 section)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.a6_blog_writer import A6BlogWriter, _replace_em_dashes
from src.pipeline.state import AgentStatus, PipelineState, TokenUsage
from src.utils.errors import AgentValidationError

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

MOCK_TOKEN_USAGE = TokenUsage(input_tokens=400, output_tokens=800, total_tokens=1200)

# A1 with three distinct source names whose [:15] prefixes are all different
A1_OUTPUT = """\
1. Smith 2023 email marketing ROI study: returns consistently above industry average
2. Jones 2022 personalisation benchmark: open rates up 22% with targeted sequences
3. Brown 2024 AI workflow analysis: automation reducing campaign cost by 40%
"""

A2_OUTPUT = """\
Core claim: Email ROI is strong across B2B sectors
Key evidence: Multiple studies confirm consistent returns
Caveats: List quality matters significantly

Core claim: Personalisation boosts open rates measurably
Key evidence: Large-sample benchmarks show 20-30% uplift
Caveats: Requires robust CRM data
"""

A5_OUTPUT = """\
## Verdict
Email marketing remains the highest-ROI channel for B2B when properly personalised.

## Three strongest reasons
1. Consistent ROI data across multiple studies
2. Direct channel ownership vs algorithm dependency
3. Personalisation technology now mature enough for SMBs

## Honest concession
Privacy regulations do create friction and compliance costs.

## The angle
Position email as the owned channel that compounds over time.

## What to avoid
Avoid generic batch-and-blast tactics that inflate unsubscribe rates.
"""

# Happy-path post: 861 words, contains hook/concession/conclusion, cites all 3 source names
A6_HAPPY_OUTPUT = """\
hook: Email marketing is not dead — it is evolving.

Email marketing has long been the backbone of B2B outreach. Whether you are a startup or an enterprise, the inbox remains the most direct line to decision-makers. As channels multiply and attention fractures, the owned channel becomes more valuable, not less. Recent research supports this view strongly.

Smith 2023 email marketing ROI study confirms that organisations investing in segmented, targeted email campaigns consistently outperform peers relying on social advertising. The data is clear: average returns from email exceed those of any other digital channel when campaigns are executed with precision and strategic intent. Marketers who abandon email in favour of trending platforms often find themselves locked into algorithm dependency with limited control over reach or timing.

Jones 2022 personalisation benchmark found that personalised subject lines alone increase open rates by twenty-two percent on average. More importantly, personalised content sequences — those that adapt messaging based on prior engagement — generate conversion rates three times higher than generic batch campaigns. The technology required is no longer the preserve of large enterprise teams. Modern CRM platforms bring these capabilities within reach of growth-stage businesses operating with lean resources.

Brown 2024 AI workflow analysis documents how artificial intelligence is fundamentally changing the operational side of email marketing. Automated segmentation, dynamic content blocks, send-time optimisation, and predictive churn detection are now standard features of mid-market platforms. The barrier to sophisticated email execution is falling, not rising. Teams that previously required multiple specialists can now run complex multi-step sequences with a fraction of the former overhead.

The strategic imperative is clear. Email operates on owned infrastructure. You control the list, the timing, the content, and the relationship. That is not true of social media, where platform decisions can erase years of audience-building overnight. The inbox is leased space but governed by your rules once the subscriber opts in. That ownership compounds over time as trust accumulates and the relationship deepens with each relevant, well-timed message.

Deliverability discipline is the price of admission. A well-maintained list with strong hygiene practices consistently outperforms a large but unengaged one. Suppression of inactive subscribers, double opt-in practices, and domain warming all contribute to inbox placement rates that keep your messages visible. The marketers who treat deliverability as an afterthought are the ones writing off email prematurely.

Segmentation remains the single most impactful lever available to email marketers today. Dividing a list by industry, company size, engagement recency, or buyer stage allows the sender to deliver messaging that feels relevant rather than intrusive. A well-segmented nurture sequence for a mid-market software buyer looks nothing like one for an enterprise procurement officer, and conflating the two audiences with a single blast erodes trust faster than any regulatory burden can.

Testing discipline separates great email programs from average ones. Subject line testing, preview text variation, send-time experimentation, and content-format trials all generate data that compounds into a compounding advantage. Organisations that commit to a continuous improvement cycle — send, measure, learn, adjust — consistently outperform those treating each campaign as a one-off execution. The infrastructure cost of testing is negligible; the returns are substantial.

The integration of email with other owned channels amplifies results. When email sequences are coordinated with content marketing, webinar programmes, and sales outreach cadences, the combined effect exceeds the sum of the parts. A subscriber who has engaged with three email touchpoints before a sales call is measurably more likely to convert than one receiving a cold introduction. Email provides the warming function that makes every downstream conversion cheaper and faster.

concession: Privacy regulations have introduced real friction. GDPR, CASL, and evolving state-level legislation in the United States have raised the cost of compliance and placed new burdens on list-building practices. Consent requirements, unsubscribe mechanics, and data retention policies all demand investment in both tooling and process. These headwinds are not imaginary, and dismissing them underestimates the operational load placed on smaller teams without dedicated legal or compliance resources.

That said, the companies navigating these constraints most successfully report that the forced discipline around consent has actually improved list quality. Subscribers who actively opt in are more engaged, convert at higher rates, and generate lower spam complaint rates. The short-term friction of compliance has a long-term dividend in trust. Regulation is not the death of email — it is the pruning that makes the remaining list more valuable.

The path forward demands integrating email into a broader owned-media strategy. The newsletter, the nurture sequence, the re-engagement campaign — these are not isolated tactics but compounding assets. Each subscriber represents an ongoing relationship, not a one-time transaction. Brands that invest in this relationship through consistent, relevant, non-promotional communication build audiences that competitors cannot replicate by purchasing ad inventory.

conclusion: Email marketing remains the highest-ROI channel for B2B organisations willing to invest in the fundamentals. The evidence is consistent, the technology is accessible, and the strategic case for channel ownership grows stronger as alternative platforms increase dependency and reduce control. The organisations that thrive will be those treating email as a long-term relationship asset rather than a short-term lead-generation tool. Commit to quality, respect the inbox, and the returns will compound.
"""


def _make_short_post() -> str:
    """Post with ~400 words — below 80% of 900 (threshold: 720). Has required keywords and citations."""
    return (
        "hook: Email marketing delivers strong ROI for B2B teams.\n\n"
        "Smith 2023 email marketing ROI study confirms consistent returns above industry average. "
        "Jones 2022 personalisation benchmark shows open rates up 22 percent with segmented lists. "
        "Brown 2024 AI workflow analysis documents how automation is reducing campaign overhead.\n\n"
        "concession: Privacy regulations have created real compliance friction for marketers.\n\n"
        "conclusion: Email remains the highest-ROI channel. Invest in list quality and segmentation.\n"
    )


def _make_long_post() -> str:
    """Post with many words — above 120% of 900 (threshold: 1080)."""
    filler = (
        "Email is powerful and drives results consistently across sectors and industries. "
        "Smith 2023 email marketing ROI study shows consistent returns above industry average every single quarter. "
        "Jones 2022 personalisation benchmark confirms that personalisation significantly lifts open rates across campaigns. "
        "Brown 2024 AI workflow analysis proves that AI automation measurably reduces operational costs for marketing teams. "
        "hook: Email marketing delivers measurable and repeatable ROI for every size of organisation. "
        "concession: Regulations add compliance friction but ultimately improve overall list quality and subscriber trust. "
        "conclusion: Invest in email for long-term compounding returns and sustainable audience ownership over time. "
    )
    # Repeat enough to exceed 1080 words (each filler block ~80 words, need 14+ to exceed 1080)
    return (filler * 16).strip()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    s = PipelineState(
        topic="Email marketing",
        audience="B2B Marketers",
        tone="Direct",
        target_word_count=900,
    )
    s.agents["A1"].status = AgentStatus.COMPLETED
    s.agents["A1"].output = A1_OUTPUT
    s.agents["A2"].status = AgentStatus.COMPLETED
    s.agents["A2"].output = A2_OUTPUT
    s.agents["A3"].status = AgentStatus.COMPLETED
    s.agents["A3"].output = "A3 landscape output"
    s.agents["A4"].status = AgentStatus.COMPLETED
    s.agents["A4"].output = "A4 devils advocate output"
    s.agents["A5"].status = AgentStatus.COMPLETED
    s.agents["A5"].output = A5_OUTPUT
    return s


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call = AsyncMock(return_value=(A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE))
    return llm


@pytest.fixture
def agent(mock_llm):
    return A6BlogWriter(llm_client=mock_llm)


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

def test_build_prompt_includes_a5_verdict(agent, state):
    prompt = agent.build_prompt(state)
    assert A5_OUTPUT in prompt


def test_build_prompt_includes_a1_sources(agent, state):
    prompt = agent.build_prompt(state)
    assert A1_OUTPUT in prompt


def test_build_prompt_includes_a2_claims(agent, state):
    prompt = agent.build_prompt(state)
    assert A2_OUTPUT in prompt


def test_build_prompt_includes_audience(agent, state):
    prompt = agent.build_prompt(state)
    assert "B2B Marketers" in prompt


def test_build_prompt_includes_tone(agent, state):
    prompt = agent.build_prompt(state)
    assert "Direct" in prompt


def test_build_prompt_includes_word_count(agent, state):
    prompt = agent.build_prompt(state)
    assert "900" in prompt


def test_build_prompt_includes_cluster_angle_when_set(agent, state):
    state.cluster_angle = "Focus on SMB email strategy"
    prompt = agent.build_prompt(state)
    assert "Focus on SMB email strategy" in prompt


def test_build_prompt_no_cluster_angle_when_not_set(agent, state):
    state.cluster_angle = None
    prompt = agent.build_prompt(state)
    assert "Cluster angle" not in prompt


# ---------------------------------------------------------------------------
# validate_output — happy path
# ---------------------------------------------------------------------------

def test_validate_output_passes_on_happy_path(agent, state):
    agent.validate_output(A6_HAPPY_OUTPUT, state)  # should not raise


# ---------------------------------------------------------------------------
# validate_output — word count checks
# ---------------------------------------------------------------------------

def test_validate_output_raises_when_word_count_short(agent, state):
    short_post = _make_short_post()
    wc = len(short_post.split())
    assert wc < 900 * 0.8, f"Test setup error: post has {wc} words, expected < 720"
    with pytest.raises(AgentValidationError) as exc_info:
        agent.validate_output(short_post, state)
    assert exc_info.value.agent_id == "A6"
    assert "80%" in str(exc_info.value) or "below" in str(exc_info.value)


def test_validate_output_raises_when_word_count_long(agent, state):
    long_post = _make_long_post()
    wc = len(long_post.split())
    assert wc > 900 * 1.2, f"Test setup error: post has {wc} words, expected > 1080"
    with pytest.raises(AgentValidationError) as exc_info:
        agent.validate_output(long_post, state)
    assert exc_info.value.agent_id == "A6"
    assert "120%" in str(exc_info.value) or "above" in str(exc_info.value)


def test_validate_output_raises_when_hook_missing(agent, state):
    no_hook = A6_HAPPY_OUTPUT.replace("hook:", "introduction:")
    with pytest.raises(AgentValidationError) as exc_info:
        agent.validate_output(no_hook, state)
    assert exc_info.value.agent_id == "A6"


def test_validate_output_raises_when_concession_missing(agent, state):
    no_concession = A6_HAPPY_OUTPUT.replace("concession:", "acknowledgement:")
    with pytest.raises(AgentValidationError) as exc_info:
        agent.validate_output(no_concession, state)
    assert exc_info.value.agent_id == "A6"


def test_validate_output_raises_when_conclusion_missing(agent, state):
    no_conclusion = A6_HAPPY_OUTPUT.replace("conclusion:", "summary:")
    with pytest.raises(AgentValidationError) as exc_info:
        agent.validate_output(no_conclusion, state)
    assert exc_info.value.agent_id == "A6"


# ---------------------------------------------------------------------------
# CP-10a: expansion re-prompt triggered when post too short
# ---------------------------------------------------------------------------

async def test_cp10a_expand_reprompt_triggered_when_post_too_short(mock_llm, state):
    short_post = _make_short_post()
    mock_llm.call = AsyncMock(
        side_effect=[
            (short_post, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A6BlogWriter(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 3


async def test_cp10a_expand_reprompt_contains_expand_not_regenerate(mock_llm, state):
    """Re-prompt must say 'expand' and must not instruct full regeneration
    (it may say 'do not regenerate' but must not say 'regenerate' as a primary command)."""
    short_post = _make_short_post()
    mock_llm.call = AsyncMock(
        side_effect=[
            (short_post, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A6BlogWriter(llm_client=mock_llm)
    await agent.run(state)
    reprompt_text = mock_llm.call.call_args_list[1][0][1]
    assert "expand" in reprompt_text.lower()
    assert short_post in reprompt_text


# ---------------------------------------------------------------------------
# CP-10b: cut re-prompt triggered when post too long
# ---------------------------------------------------------------------------

async def test_cp10b_cut_reprompt_triggered_when_post_too_long(mock_llm, state):
    long_post = _make_long_post()
    mock_llm.call = AsyncMock(
        side_effect=[
            (long_post, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A6BlogWriter(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 3


async def test_cp10b_cut_reprompt_contains_trim_or_cut(mock_llm, state):
    long_post = _make_long_post()
    mock_llm.call = AsyncMock(
        side_effect=[
            (long_post, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A6BlogWriter(llm_client=mock_llm)
    await agent.run(state)
    reprompt_text = mock_llm.call.call_args_list[1][0][1]
    assert "trim" in reprompt_text.lower() or "cut" in reprompt_text.lower()


async def test_cp10b_cut_reprompt_includes_current_post(mock_llm, state):
    """Re-prompt must include the current post, confirming it is a targeted cut."""
    long_post = _make_long_post()
    mock_llm.call = AsyncMock(
        side_effect=[
            (long_post, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A6BlogWriter(llm_client=mock_llm)
    await agent.run(state)
    reprompt_text = mock_llm.call.call_args_list[1][0][1]
    assert "Current post" in reprompt_text or long_post[:100] in reprompt_text


# ---------------------------------------------------------------------------
# CP-11: missing concession triggers concession-only re-prompt; appends to existing post
# ---------------------------------------------------------------------------

async def test_cp11_concession_reprompt_triggered_when_concession_missing(mock_llm, state):
    """Post in valid word-count range but missing 'concession' keyword."""
    post_no_concession = A6_HAPPY_OUTPUT.replace("concession:", "acknowledgement:")
    assert "concession" not in post_no_concession.lower()

    concession_addition = "\n\nconcession: Privacy regulations add real friction to list building."
    combined = post_no_concession + "\n\n" + concession_addition

    mock_llm.call = AsyncMock(
        side_effect=[
            (post_no_concession, MOCK_TOKEN_USAGE),
            (concession_addition, MOCK_TOKEN_USAGE),
            (combined, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A6BlogWriter(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 3


async def test_cp11_concession_reprompt_appends_before_humanise(mock_llm, state):
    """Concession is appended, then humanise runs as a separate step."""
    post_no_concession = A6_HAPPY_OUTPUT.replace("concession:", "acknowledgement:")
    concession_addition = "concession: Privacy regulations add friction but improve list quality."
    combined = post_no_concession + "\n\n" + concession_addition

    mock_llm.call = AsyncMock(
        side_effect=[
            (post_no_concession, MOCK_TOKEN_USAGE),
            (concession_addition, MOCK_TOKEN_USAGE),
            (combined, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A6BlogWriter(llm_client=mock_llm)
    result = await agent.run(state)
    assert "concession" in result.lower()


async def test_cp11_concession_reprompt_instructs_append(mock_llm, state):
    """Re-prompt text must instruct append after the current post, not full regeneration."""
    post_no_concession = A6_HAPPY_OUTPUT.replace("concession:", "acknowledgement:")
    concession_addition = "concession: Regulations add friction but improve list quality."
    combined = post_no_concession + "\n\n" + concession_addition

    mock_llm.call = AsyncMock(
        side_effect=[
            (post_no_concession, MOCK_TOKEN_USAGE),
            (concession_addition, MOCK_TOKEN_USAGE),
            (combined, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A6BlogWriter(llm_client=mock_llm)
    await agent.run(state)
    reprompt_text = mock_llm.call.call_args_list[1][0][1]
    assert "append" in reprompt_text.lower()


# ---------------------------------------------------------------------------
# CP-12: < 3 citations triggers citation re-prompt with missing source names
# ---------------------------------------------------------------------------

async def test_cp12_citation_reprompt_triggered_when_fewer_than_3_citations(mock_llm, state):
    """Post with valid word count and keywords but no source citations."""
    post_no_citations = A6_HAPPY_OUTPUT.replace(
        "Smith 2023 email marketing ROI study", "recent academic research"
    ).replace(
        "Jones 2022 personalisation benchmark", "another industry study"
    ).replace(
        "Brown 2024 AI workflow analysis", "a major analyst report"
    )
    assert "Smith 2023 emai".lower() not in post_no_citations.lower()
    assert "Jones 2022 pers".lower() not in post_no_citations.lower()
    assert "Brown 2024 AI w".lower() not in post_no_citations.lower()

    mock_llm.call = AsyncMock(
        side_effect=[
            (post_no_citations, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A6BlogWriter(llm_client=mock_llm)
    await agent.run(state)
    assert mock_llm.call.call_count == 3


async def test_cp12_citation_reprompt_contains_missing_source_names(mock_llm, state):
    """The re-prompt must name the specific missing sources."""
    post_no_citations = A6_HAPPY_OUTPUT.replace(
        "Smith 2023 email marketing ROI study", "recent academic research"
    ).replace(
        "Jones 2022 personalisation benchmark", "another industry study"
    ).replace(
        "Brown 2024 AI workflow analysis", "a major analyst report"
    )

    mock_llm.call = AsyncMock(
        side_effect=[
            (post_no_citations, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A6BlogWriter(llm_client=mock_llm)
    await agent.run(state)
    reprompt_text = mock_llm.call.call_args_list[1][0][1]
    assert (
        "Smith 2023" in reprompt_text
        or "Jones 2022" in reprompt_text
        or "Brown 2024" in reprompt_text
    )


async def test_cp12_citation_reprompt_includes_current_post(mock_llm, state):
    """The citation re-prompt must include the current post (targeted weave-in, not full regen)."""
    post_no_citations = A6_HAPPY_OUTPUT.replace(
        "Smith 2023 email marketing ROI study", "recent academic research"
    ).replace(
        "Jones 2022 personalisation benchmark", "another industry study"
    ).replace(
        "Brown 2024 AI workflow analysis", "a major analyst report"
    )

    mock_llm.call = AsyncMock(
        side_effect=[
            (post_no_citations, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A6BlogWriter(llm_client=mock_llm)
    await agent.run(state)
    reprompt_text = mock_llm.call.call_args_list[1][0][1]
    assert "Current post" in reprompt_text or post_no_citations[:80] in reprompt_text


# ---------------------------------------------------------------------------
# run() — sets output, status=COMPLETED, token_usage
# ---------------------------------------------------------------------------

async def test_run_sets_output(agent, state):
    await agent.run(state)
    assert state.agents["A6"].output == _replace_em_dashes(A6_HAPPY_OUTPUT)


async def test_run_sets_status_completed(agent, state):
    await agent.run(state)
    assert state.agents["A6"].status == AgentStatus.COMPLETED


async def test_run_sets_token_usage(agent, state):
    await agent.run(state)
    assert state.agents["A6"].token_usage == MOCK_TOKEN_USAGE


async def test_run_returns_output_string(agent, state):
    result = await agent.run(state)
    assert result == _replace_em_dashes(A6_HAPPY_OUTPUT)


async def test_run_always_calls_humanise(agent, state):
    """Happy path: initial generation + humanise = 2 LLM calls."""
    await agent.run(state)
    assert agent._llm.call.call_count == 2


async def test_run_removes_em_dashes(agent, state):
    result = await agent.run(state)
    assert "—" not in result


async def test_humanise_guard_falls_back_on_failure(mock_llm, state):
    """If humanisation wrecks the structure, fall back to pre-humanised text."""
    broken = "This is a very short broken post."
    mock_llm.call = AsyncMock(
        side_effect=[
            (A6_HAPPY_OUTPUT, MOCK_TOKEN_USAGE),
            (broken, MOCK_TOKEN_USAGE),
        ]
    )
    agent = A6BlogWriter(llm_client=mock_llm)
    result = await agent.run(state)
    assert result == _replace_em_dashes(A6_HAPPY_OUTPUT)
