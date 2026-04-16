# Blog Post Agent Pipeline — Claude Code Spec
**Version:** 2.0 (Python)
**Stack:** Python 3.12 · Anthropic Python SDK · Pydantic · pytest
**Purpose:** A CLI-driven, 6-agent sequential pipeline that researches a topic, debates it internally, reaches an evidence-based verdict, and writes a positioned blog post. Runs entirely in Claude via the Anthropic API.

---

## 1. Project Structure

```
blog-agent-pipeline/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base class all agents implement
│   │   ├── a1_research_collector.py
│   │   ├── a2_claim_extractor.py
│   │   ├── a3_landscape_mapper.py
│   │   ├── a4_devils_advocate.py
│   │   ├── a5_evidence_judge.py
│   │   └── a6_blog_writer.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── runner.py            # Orchestrates agent execution in sequence
│   │   ├── state.py             # PipelineState Pydantic model + transitions
│   │   ├── checkpoints.py       # Persist/resume state to/from disk
│   │   └── watchdog.py          # Async timeout + stall detection per agent
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py            # Anthropic SDK wrapper + retry logic
│   │   └── prompts.py           # All 6 agent prompt templates
│   ├── io/
│   │   ├── __init__.py
│   │   ├── input_parser.py      # Parse CLI args, URLs, file paths
│   │   └── output_writer.py     # Write final post + audit trail
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Structured logging
│       └── errors.py            # Custom exception classes
├── tests/
│   ├── conftest.py              # Shared fixtures (mock LLM client, base state)
│   ├── unit/
│   │   ├── agents/
│   │   │   ├── test_a1.py
│   │   │   ├── test_a2.py
│   │   │   ├── test_a3.py
│   │   │   ├── test_a4.py
│   │   │   ├── test_a5.py
│   │   │   └── test_a6.py
│   │   ├── pipeline/
│   │   │   ├── test_state.py
│   │   │   ├── test_watchdog.py
│   │   │   └── test_checkpoints.py
│   │   └── llm/
│   │       └── test_client.py
│   ├── integration/
│   │   ├── test_pipeline.py     # Full pipeline with mocked LLM
│   │   └── test_chokepoints.py  # All 17 chokepoints exercised
│   └── fixtures/
│       ├── __init__.py
│       ├── sources.py
│       ├── topics.py
│       └── llm_responses/       # Canned LLM outputs per agent + scenario
│           ├── a1_happy_path.txt
│           ├── a1_no_sources.txt
│           ├── a2_happy_path.txt
│           ├── a2_shallow_claims.txt
│           ├── a3_happy_path.txt
│           ├── a3_no_contest.txt
│           ├── a3_conflation.txt
│           ├── a4_happy_path.txt
│           ├── a4_no_source_citation.txt
│           ├── a5_happy_path.txt
│           ├── a5_hedging.txt
│           ├── a5_missing_sections.txt
│           ├── a6_happy_path.txt
│           ├── a6_word_count_short.txt
│           ├── a6_word_count_long.txt
│           ├── a6_no_concession.txt
│           └── a6_no_citations.txt
├── output/                      # Created at runtime, gitignored
├── .pipeline-checkpoints/       # Created at runtime, gitignored
├── .env.example
├── pyproject.toml
└── main.py                      # CLI entry point
```

---

## 2. Dependencies

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "blog-agent-pipeline"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.28.0",
    "pydantic>=2.7.0",
    "python-dotenv>=1.0.0",
    "typer>=0.12.0",          # CLI argument parsing
    "httpx>=0.27.0",          # Anthropic SDK dependency, also used in tests
    "tenacity>=8.3.0",        # Retry logic with backoff
    "rich>=13.7.0",           # Terminal output formatting
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "pytest-mock>=3.14.0",
    "respx>=0.21.0",          # Mock httpx requests (used by Anthropic SDK)
    "ruff>=0.4.0",            # Linting + formatting
    "mypy>=1.10.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

Install everything:
```bash
pip install -e ".[dev]"
```

---

## 3. Data Flow & State Machine

### PipelineState

Every agent reads from and writes to a single Pydantic model. This is the only mechanism for passing data between agents. No agent calls another agent directly.

```python
# src/pipeline/state.py
from __future__ import annotations
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field
import time, uuid

AgentId = Literal["A1", "A2", "A3", "A4", "A5", "A6"]
AGENT_ORDER: list[AgentId] = ["A1", "A2", "A3", "A4", "A5", "A6"]

class AgentStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    TIMED_OUT  = "timed_out"
    SKIPPED    = "skipped"

class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int

class AgentRecord(BaseModel):
    id: AgentId
    status: AgentStatus = AgentStatus.PENDING
    started_at: float | None = None       # unix timestamp
    completed_at: float | None = None
    duration_ms: float | None = None
    input: str = ""                        # exact string sent to LLM
    output: str | None = None             # exact string received
    token_usage: TokenUsage | None = None
    retry_count: int = 0
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)

class PipelineStatus(str, Enum):
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    ABORTED   = "aborted"

class PipelineState(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    audience: str
    tone: str
    target_word_count: int = 900
    cluster_angle: str | None = None
    provided_sources: list[str] = Field(default_factory=list)
    agents: dict[AgentId, AgentRecord] = Field(
        default_factory=lambda: {
            aid: AgentRecord(id=aid) for aid in AGENT_ORDER
        }
    )
    pipeline_status: PipelineStatus = PipelineStatus.RUNNING
    started_at: float = Field(default_factory=time.time)
    completed_at: float | None = None
    checkpoint_path: str = ""
    total_tokens: int = 0


def can_run(state: PipelineState, agent_id: AgentId) -> bool:
    """An agent can run only if its predecessor completed."""
    idx = AGENT_ORDER.index(agent_id)
    if idx == 0:
        return True
    predecessor = AGENT_ORDER[idx - 1]
    return state.agents[predecessor].status == AgentStatus.COMPLETED


def require_output(state: PipelineState, agent_id: AgentId) -> str:
    """Returns predecessor output or raises if unavailable."""
    record = state.agents[agent_id]
    if record.status != AgentStatus.COMPLETED or record.output is None:
        raise PipelineDependencyError(
            f"Agent {agent_id} output required but not available. "
            f"Status: {record.status}"
        )
    return record.output


# Valid state transitions only
VALID_TRANSITIONS: set[tuple[AgentStatus, AgentStatus]] = {
    (AgentStatus.PENDING,    AgentStatus.RUNNING),
    (AgentStatus.RUNNING,    AgentStatus.COMPLETED),
    (AgentStatus.RUNNING,    AgentStatus.FAILED),
    (AgentStatus.RUNNING,    AgentStatus.TIMED_OUT),
    (AgentStatus.FAILED,     AgentStatus.RUNNING),    # manual resume
    (AgentStatus.TIMED_OUT,  AgentStatus.RUNNING),    # manual resume
}

def transition(state: PipelineState, agent_id: AgentId, to: AgentStatus) -> None:
    record = state.agents[agent_id]
    from_ = record.status
    if (from_, to) not in VALID_TRANSITIONS:
        raise InvalidStateTransitionError(agent_id, from_, to)
    record.status = to
    if to == AgentStatus.RUNNING:
        record.started_at = time.time()
    elif to in (AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.TIMED_OUT):
        record.completed_at = time.time()
        if record.started_at:
            record.duration_ms = (record.completed_at - record.started_at) * 1000
```

---

## 4. Error Classes

```python
# src/utils/errors.py

class AgentTimeoutError(Exception):
    def __init__(self, agent_id: str, timeout_ms: int):
        super().__init__(f"Agent {agent_id} timed out after {timeout_ms}ms")
        self.agent_id = agent_id
        self.timeout_ms = timeout_ms

class AgentValidationError(Exception):
    def __init__(self, agent_id: str, reason: str):
        super().__init__(f"Agent {agent_id} output failed validation: {reason}")
        self.agent_id = agent_id
        self.reason = reason

class PipelineDependencyError(Exception):
    pass

class InvalidStateTransitionError(Exception):
    def __init__(self, agent_id: str, from_: object, to: object):
        super().__init__(
            f"Invalid transition for {agent_id}: {from_} → {to}"
        )

class CheckpointCorruptError(Exception):
    def __init__(self, path: str):
        super().__init__(f"Checkpoint at {path} failed schema validation")
        self.path = path

class LLMRetryExhaustedError(Exception):
    def __init__(self, agent_id: str, attempts: int):
        super().__init__(
            f"Agent {agent_id} exhausted {attempts} retry attempts"
        )


# Pipeline warnings — not exceptions, stored in agent record
class PipelineWarning:
    NO_CONTEST           = "NO_CONTEST"           # A3: no genuine disagreement
    SHALLOW_CLAIMS       = "SHALLOW_CLAIMS"       # A2: claims too thin
    TRUNCATED_POSITIONS  = "TRUNCATED_POSITIONS"  # A4: >4 positions, capped at 3
    CONTEXT_NEAR_LIMIT   = "CONTEXT_NEAR_LIMIT"   # 80% of token budget used
```

---

## 5. LLM Client

```python
# src/llm/client.py
import asyncio
from anthropic import AsyncAnthropic, APIStatusError, APIConnectionError
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception, before_sleep_log
)
import logging

logger = logging.getLogger(__name__)

# Model selection per agent — defined here, referenced by agents
AGENT_MODELS: dict[str, str] = {
    "A1": "claude-sonnet-4-6",   # Fast, web-aware
    "A2": "claude-sonnet-4-6",   # Structured extraction
    "A3": "claude-opus-4-5",     # Complex synthesis
    "A4": "claude-opus-4-5",     # Requires genuine reasoning
    "A5": "claude-opus-4-5",     # Most critical verdict step
    "A6": "claude-sonnet-4-6",   # Long-form generation
}

# Temperature per agent
AGENT_TEMPERATURES: dict[str, float] = {
    "A1": 0.2, "A2": 0.1, "A3": 0.3,
    "A4": 0.4, "A5": 0.2, "A6": 0.7,
}

MAX_TOKENS_PER_AGENT: dict[str, int] = {
    "A1": 2000, "A2": 3000, "A3": 2000,
    "A4": 3000, "A5": 1500, "A6": 4000,
}


def _is_retryable(exc: BaseException) -> bool:
    """Only retry on server errors and connection issues.
    Never retry 400 (bad request) or 401 (auth)."""
    if isinstance(exc, APIStatusError):
        return exc.status_code in (429, 500, 529)
    return isinstance(exc, APIConnectionError)


class LLMClient:
    def __init__(self, api_key: str):
        self._client = AsyncAnthropic(api_key=api_key)

    async def call(
        self,
        agent_id: str,
        prompt: str,
        signal: asyncio.Event | None = None,   # watchdog cancellation
    ) -> tuple[str, "TokenUsage"]:
        """
        Call the LLM for a given agent. Returns (output_text, token_usage).
        Raises LLMRetryExhaustedError after 4 failed attempts.
        Respects cancellation via signal.
        """

        @retry(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=2, max=20),
            retry=retry_if_exception(_is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _call_with_retry() -> tuple[str, object]:
            if signal and signal.is_set():
                raise AgentTimeoutError(agent_id, 0)

            response = await self._client.messages.create(
                model=AGENT_MODELS[agent_id],
                max_tokens=MAX_TOKENS_PER_AGENT[agent_id],
                temperature=AGENT_TEMPERATURES[agent_id],
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            usage = response.usage
            return text, usage

        try:
            text, usage = await _call_with_retry()
        except APIStatusError as e:
            if e.status_code not in (429, 500, 529):
                raise   # 400, 401 etc — hard failure, do not retry
            raise LLMRetryExhaustedError(agent_id, 4) from e

        from src.pipeline.state import TokenUsage
        token_usage = TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )
        return text, token_usage
```

---

## 6. Abstract Base Agent

```python
# src/agents/base.py
import abc
from src.pipeline.state import PipelineState, AgentId

class BaseAgent(abc.ABC):
    id: AgentId
    timeout_ms: int
    max_retries: int

    @abc.abstractmethod
    async def run(self, state: PipelineState) -> str:
        """Execute the agent. Returns raw LLM output string."""
        ...

    @abc.abstractmethod
    def build_prompt(self, state: PipelineState) -> str:
        """Construct the full prompt string from current state."""
        ...

    @abc.abstractmethod
    def validate_output(self, output: str, state: PipelineState) -> None:
        """
        Validate the LLM output. Raises AgentValidationError if invalid.
        May mutate state to add warnings (e.g. SHALLOW_CLAIMS).
        Should NOT raise on warnings — only on hard failures.
        """
        ...
```

---

## 7. Agent Specifications

### A1 — Research Collector
- **timeout_ms:** 90_000
- **max_retries:** 2
- **Input:** `state.topic` + `state.provided_sources`
- **Output format:** Numbered list. Each entry must contain: URL, Type, Recency, Core claim, Credibility signal
- **validate_output:**
  - Count entries using regex `r'^\d+\.'` on each line (multiline)
  - Raise `AgentValidationError` if < 3 entries found
  - Check each entry contains all five required field labels
  - If < 6 entries total, set `CONTEXT_NEAR_LIMIT` warning (not a hard failure — the landscape may just be sparse)
- **Chokepoint CP-01:** If output signals no sources found (detect phrase `"could not find"` or `"no sources"` case-insensitive), re-prompt with `FALLBACK_A1` prompt instructing Claude to use training knowledge and flag sources as `[Training Knowledge - unverified]`

### A2 — Claim Extractor
- **timeout_ms:** 60_000
- **max_retries:** 3
- **Input:** A1 output via `require_output(state, "A1")`
- **Output format:** Per-source blocks. Each block has four labelled sections: `Core claim:` / `Key evidence:` / `Caveats:` / `Implicit assumption:`
- **validate_output:**
  - Count source entries in A1 output (reuse count logic)
  - Count blocks in A2 output — raise if mismatch
  - For each `Core claim:` field, count words — if any < 15 words, add `PipelineWarning.SHALLOW_CLAIMS` to agent record. **Do not raise. Do not retry.** The warning is propagated to A5's prompt.
- **Chokepoint CP-02:** Source count mismatch → re-prompt once with explicit count: `"The previous source list contained {n} sources. Your output must contain exactly {n} claim blocks."`

### A3 — Landscape Mapper
- **timeout_ms:** 60_000
- **max_retries:** 3
- **Input:** A2 output
- **Output format:** Five labelled sections: `## Consensus zone` / `## Contested zone` / `## Outlier positions` / `## Evidence weight summary` / `## The unresolved question`
- **validate_output:**
  - All five `##` headings must be present — raise `AgentValidationError` if any missing
  - Extract content under `## Contested zone`
  - Count distinct positions (look for numbered items or `Position:` labels) — if < 2, add `PipelineWarning.NO_CONTEST` and raise `AgentValidationError("A3", "Contested zone contains fewer than 2 distinct positions")`
  - Check for positions appearing in both Consensus and Contested zones (conflation) — if found, re-prompt once
- **Chokepoint CP-04:** Runner catches `NO_CONTEST` specifically and pauses pipeline to prompt user: `"No genuine disagreement found on this topic. Continue anyway? (y/n)"` If n, set `pipeline_status = ABORTED`

### A4 — Devil's Advocate
- **timeout_ms:** 90_000
- **max_retries:** 3
- **Input:** A3 output + A1 source list (both injected into prompt)
- **Pre-run check (before LLM call):** Count positions in A3 Contested zone. If > 4, cap prompt at 3 and add `PipelineWarning.TRUNCATED_POSITIONS` to record. Log which positions were dropped.
- **Output format:** One block per contested position: `Position:` / `Case:` (3 points) / `Hardest objection:` / `Response:`
- **validate_output:**
  - Block count must match positions sent (after cap)
  - Each block must reference at least one source name from A1's source list (extract source names, check each block contains ≥ 1)
  - If a block lacks a source reference, re-prompt once: `"Each steelman block must cite at least one named source from the source list."`

### A5 — Evidence Judge
- **timeout_ms:** 60_000
- **max_retries:** 3
- **Input:** A4 output + A1 source list. If `SHALLOW_CLAIMS` warning is in A2 record, prepend a notice to the prompt: `"Note: the source material for this topic was thin. Factor this uncertainty into your verdict."`
- **Output format:** Five labelled sections: `## Verdict` / `## Three strongest reasons` / `## Honest concession` / `## The angle` / `## What to avoid`
- **validate_output:**
  - All five `##` headings present — raise if any missing
  - Extract `## Verdict` content — must be a single sentence (no `.` mid-section except at the end)
  - Hedge detection: `re.search(r'\bit depends\b|\bon the one hand\b|\bnuanced\b|\bboth sides\b|\bcomplex picture\b', verdict, re.IGNORECASE)` — if match, re-prompt once with: `"Commit to the position the weight of evidence most supports. A single clear sentence. No hedging."`
  - Max one re-prompt for hedging — if second output still hedges, raise `AgentValidationError`

### A6 — Blog Writer
- **timeout_ms:** 120_000
- **max_retries:** 2
- **Input:** A5 output + A1 source list + A2 claims + user brief (audience, tone, word count, cluster angle)
- **validate_output (run in sequence — fix each before checking next):**
  1. **Word count:** `len(output.split())`. Target ± 20%. If short, targeted re-prompt for the weakest section. If long, targeted re-prompt to cut. **Do not regenerate the full post.**
  2. **Sections present:** Check for `hook`, `concession`, `conclusion` keywords (case-insensitive) — raise if any absent
  3. **Concession paragraph:** Check that the concession section contains a reference to an opposing view. If absent, targeted re-prompt for concession only — append to existing post, do not regenerate.
  4. **Source citations:** Extract all source names from A1. Count how many appear in the post. If < 3, re-prompt with: `"Weave in references to these sources: {missing_sources}"` — targeted re-prompt only.
  5. **Hedge phrases:** Same regex as A5. If found, targeted re-prompt for the specific paragraph.

---

## 8. Watchdog

```python
# src/pipeline/watchdog.py
import asyncio
from src.utils.errors import AgentTimeoutError

async def with_watchdog(
    agent_id: str,
    timeout_ms: int,
    coro,                   # the agent's run() coroutine
    on_timeout,             # callable(agent_id) — called before raising
):
    """
    Race the coroutine against a timeout.
    If timeout fires: calls on_timeout, then raises AgentTimeoutError.
    If coroutine raises: propagates the exception unchanged.
    Always cancels the losing side cleanly.
    """
    timeout_s = timeout_ms / 1000
    cancel_event = asyncio.Event()

    async def _run():
        return await coro(cancel_event)

    try:
        result = await asyncio.wait_for(_run(), timeout=timeout_s)
        return result
    except asyncio.TimeoutError:
        cancel_event.set()
        on_timeout(agent_id)
        raise AgentTimeoutError(agent_id, timeout_ms)
```

The `cancel_event` is passed into each agent's `run()` method and forwarded to `LLMClient.call()`. This ensures the in-flight HTTP request is abandoned immediately when the watchdog fires — not just after it returns.

---

## 9. Checkpointing

```python
# src/pipeline/checkpoints.py
import json
from pathlib import Path
from src.pipeline.state import PipelineState, AGENT_ORDER, AgentStatus
from src.utils.errors import CheckpointCorruptError

CHECKPOINT_DIR = Path(".pipeline-checkpoints")

def save(state: PipelineState) -> None:
    """Write state to disk after every agent status change."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"{state.run_id}.json"
    path.write_text(state.model_dump_json(indent=2))

def load(run_id: str) -> PipelineState:
    """Load and validate checkpoint. Raises CheckpointCorruptError if invalid."""
    path = CHECKPOINT_DIR / f"{run_id}.json"
    try:
        data = json.loads(path.read_text())
        return PipelineState.model_validate(data)
    except Exception as e:
        raise CheckpointCorruptError(str(path)) from e

def get_resume_point(state: PipelineState) -> str | None:
    """Return the ID of the first non-completed agent, or None if all done."""
    for agent_id in AGENT_ORDER:
        if state.agents[agent_id].status != AgentStatus.COMPLETED:
            return agent_id
    return None
```

---

## 10. Pipeline Runner

```python
# src/pipeline/runner.py
"""
Orchestrates the six agents in sequence. Responsibilities:
- Enforce state transitions via transition()
- Wrap every agent in with_watchdog()
- Persist checkpoint after every agent completion or failure
- Surface PipelineWarnings to the terminal without halting
- Halt and surface errors cleanly on failures
- Handle NO_CONTEST pause (A3)
- Handle CONTEXT_NEAR_LIMIT warning (any agent)
"""
```

The runner must:
1. On startup, check if a checkpoint exists for the given run_id. If yes, call `get_resume_point()` and skip already-completed agents.
2. Before running each agent, call `can_run()`. If False, raise `PipelineDependencyError`.
3. After each agent completes, update `state.total_tokens` with the agent's `token_usage.total_tokens`.
4. After each agent completes or fails, call `save(state)` immediately — before any further processing.
5. After every agent, check `state.total_tokens` against `PIPELINE_MAX_TOTAL_TOKENS * 0.8`. If exceeded, add `PipelineWarning.CONTEXT_NEAR_LIMIT` to the next agent's record and log a warning.
6. If A3 raises `AgentValidationError` with `NO_CONTEST` in the message, pause and prompt the user. Do not silently continue.

---

## 11. Chokepoint Register

All 17 chokepoints, their detection mechanism, and resolution:

| # | Location | Failure Mode | Detection | Resolution |
|---|----------|--------------|-----------|------------|
| CP-01 | A1 web search | No sources found for niche topic | Output contains `"could not find"` or `"no sources"` (case-insensitive) | Re-prompt with fallback using training knowledge; flag sources as unverified |
| CP-02 | A1→A2 handoff | Source count mismatch | Count sources in A1, expect same count of blocks in A2 | Re-prompt A2 with explicit count |
| CP-03 | A2 claim quality | Shallow claims (< 15 words) | Word count check on each `Core claim:` field | Add `SHALLOW_CLAIMS` warning; propagate to A5 prompt |
| CP-04 | A3 contest detection | No interesting angle | Contested zone has < 2 distinct positions | Raise `AgentValidationError`; runner surfaces `NO_CONTEST`; pause for user |
| CP-05 | A3 conflation | Consensus/contested conflated | Same position text appears in both zones | Re-prompt A3 once with explicit separation instruction |
| CP-06 | A4 context overflow | Too many positions to steelman | Count positions in A3 before building prompt; > 4 is risk | Cap at 3 positions; add `TRUNCATED_POSITIONS` warning; log dropped positions |
| CP-07 | A4 source grounding | Arguments not grounded in sources | Check each steelman block for a named source from A1 | Re-prompt once: "Each block must cite at least one named source" |
| CP-08 | A5 hedging | Judge refuses to commit | Hedge phrase regex on `## Verdict` section | Re-prompt once with firm instruction; raise on second failure |
| CP-09 | A5 missing sections | Malformed output | Check all five `##` headings present | Re-prompt with explicit format instruction |
| CP-10a | A6 word count short | Post < 80% of target | `len(output.split()) < target * 0.8` | Targeted re-prompt: expand weakest section only |
| CP-10b | A6 word count long | Post > 120% of target | `len(output.split()) > target * 1.2` | Targeted re-prompt: cut to target |
| CP-11 | A6 missing concession | Writer omitted concession | Check concession section contains opposing view reference | Targeted re-prompt for concession only; append, don't regenerate |
| CP-12 | A6 missing citations | < 3 named sources in post | Check source names from A1 appear in post | Targeted re-prompt with list of missing sources to weave in |
| CP-13 | Any agent timeout | LLM call stalls | `asyncio.wait_for` timeout fires | Cancel coroutine, set `timed_out`, save checkpoint, halt pipeline |
| CP-14 | Any agent API error | 500/529 from Anthropic | `APIStatusError.status_code` check | Exponential backoff retry via tenacity (max 4 attempts) |
| CP-15 | Silent stall | Agent stuck in `running` with no notification | Watchdog wraps every agent — no code path exists without a timer | Structurally prevented by `with_watchdog()` |
| CP-16 | Checkpoint corruption | Invalid JSON on resume | `PipelineState.model_validate()` raises | Raise `CheckpointCorruptError`; alert user; offer restart from last valid agent |
| CP-17 | Context growth | Accumulated tokens near model limit | Track `state.total_tokens`; warn at 80% of `PIPELINE_MAX_TOTAL_TOKENS` | Truncate A2 output to top N claims; log what was cut |

---

## 12. Test Suite

### conftest.py (shared fixtures)

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock
from src.pipeline.state import PipelineState, AgentStatus, AgentRecord, AGENT_ORDER

@pytest.fixture
def base_state():
    return PipelineState(
        topic="The future of email marketing",
        audience="Senior marketing leaders",
        tone="Direct and analytical",
        target_word_count=900,
    )

@pytest.fixture
def state_after_a1(base_state, fixture_text):
    """State with A1 completed, A2–A6 pending."""
    base_state.agents["A1"].status = AgentStatus.COMPLETED
    base_state.agents["A1"].output = fixture_text("a1_happy_path")
    return base_state

# Add state_after_a2 through state_after_a5 following the same pattern

@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    return client

@pytest.fixture
def fixture_text():
    """Helper to load a fixture file by name."""
    from pathlib import Path
    def _load(name: str) -> str:
        path = Path("tests/fixtures/llm_responses") / f"{name}.txt"
        return path.read_text()
    return _load
```

### Unit Tests — State Machine

**`tests/unit/pipeline/test_state.py`**
```
✓ can_run returns False for A2 when A1 is pending
✓ can_run returns False for A2 when A1 is running
✓ can_run returns False for A2 when A1 is failed
✓ can_run returns True for A2 when A1 is completed
✓ can_run always returns True for A1
✓ require_output raises PipelineDependencyError when agent is pending
✓ require_output raises PipelineDependencyError when agent is failed
✓ require_output raises PipelineDependencyError when output is None despite completed status
✓ require_output returns string when agent is completed with output
✓ transition pending→running is valid
✓ transition running→completed is valid
✓ transition running→failed is valid
✓ transition running→timed_out is valid
✓ transition failed→running is valid (resume)
✓ transition timed_out→running is valid (resume)
✓ transition completed→running raises InvalidStateTransitionError
✓ transition pending→completed raises InvalidStateTransitionError
✓ transition pending→failed raises InvalidStateTransitionError
✓ transition sets started_at on →running
✓ transition sets completed_at on →completed
✓ transition calculates duration_ms correctly
```

### Unit Tests — Watchdog

**`tests/unit/pipeline/test_watchdog.py`**
```
✓ resolves with correct value when coroutine completes within timeout
✓ calls on_timeout callback when coroutine exceeds timeout
✓ raises AgentTimeoutError when coroutine exceeds timeout
✓ sets cancel_event when timeout fires
✓ does NOT call on_timeout when coroutine completes in time
✓ propagates coroutine exceptions unchanged (not wrapped in TimeoutError)
✓ handles coroutine that completes 1ms before timeout (no race condition)
```

### Unit Tests — Checkpoints

**`tests/unit/pipeline/test_checkpoints.py`**
```
✓ save writes valid JSON to .pipeline-checkpoints/{run_id}.json
✓ save is idempotent — second save overwrites cleanly
✓ load returns correct PipelineState
✓ load raises CheckpointCorruptError on malformed JSON
✓ load raises CheckpointCorruptError on JSON missing required fields
✓ load raises CheckpointCorruptError on wrong field types
✓ get_resume_point returns None when all agents pending (start fresh)
✓ get_resume_point returns A1 when A1 is pending
✓ get_resume_point returns A3 when A1+A2 completed, A3 pending
✓ get_resume_point returns A3 when A1+A2 completed, A3 failed
✓ get_resume_point returns A3 when A1+A2 completed, A3 timed_out
✓ get_resume_point returns None when all agents completed
```

### Unit Tests — LLM Client

**`tests/unit/llm/test_client.py`**
```
✓ calls Anthropic SDK with correct model for each agent ID
✓ calls Anthropic SDK with correct temperature for each agent ID
✓ returns (output_text, TokenUsage) on success
✓ retries on 529 status
✓ retries on 500 status
✓ retries on 429 status
✓ does NOT retry on 400 — raises immediately
✓ does NOT retry on 401 — raises immediately
✓ raises LLMRetryExhaustedError after 4 failed 529 attempts
✓ respects cancel_event: raises AgentTimeoutError if event set before call
✓ exponential backoff: second attempt waits ~2s, third ~8s, fourth ~20s
```

### Unit Tests — Per Agent

Each agent test file covers:
```
✓ build_prompt includes the correct state fields
✓ build_prompt includes predecessor output
✓ validate_output passes on happy path fixture
✓ validate_output raises AgentValidationError on [failure scenario]
✓ validate_output adds correct warning to state (not raises) on [warning scenario]
✓ re-prompt is triggered on [specific condition]
✓ re-prompt is NOT triggered more than once
✓ output written to state.agents[id].output on success
✓ status set to COMPLETED on success
✓ status set to FAILED after max_retries exhausted
```

Plus agent-specific tests:

**A1 additionally:**
```
✓ falls back to training knowledge when output signals no sources found
✓ fallback output contains [Training Knowledge - unverified] markers
```

**A2 additionally:**
```
✓ SHALLOW_CLAIMS warning added when any claim < 15 words
✓ SHALLOW_CLAIMS does not cause a retry
✓ SHALLOW_CLAIMS warning present in agent record
```

**A3 additionally:**
```
✓ NO_CONTEST warning added and AgentValidationError raised when < 2 positions
✓ conflation re-prompt triggered when position appears in both zones
✓ conflation re-prompt not triggered more than once
```

**A4 additionally:**
```
✓ positions capped at 3 when A3 has 5 contested positions
✓ TRUNCATED_POSITIONS warning added when cap applied
✓ dropped positions are logged (check logger call)
```

**A5 additionally:**
```
✓ hedge phrase "it depends" triggers re-prompt
✓ hedge phrase "on the one hand" triggers re-prompt
✓ hedge phrase "nuanced" triggers re-prompt
✓ hedge phrase "both sides" triggers re-prompt
✓ hedge phrase "complex picture" triggers re-prompt
✓ SHALLOW_CLAIMS warning in A2 record adds caveat to A5 prompt
✓ second hedge in re-prompt output raises AgentValidationError
```

**A6 additionally:**
```
✓ word count short triggers expansion re-prompt (not full regeneration)
✓ word count long triggers cut re-prompt (not full regeneration)
✓ missing concession triggers concession-only re-prompt
✓ concession re-prompt appends to existing post
✓ < 3 citations triggers citation re-prompt with specific missing sources
✓ each targeted re-prompt does not regenerate the full post
```

### Integration Tests — Full Pipeline

**`tests/integration/test_pipeline.py`**
```
✓ full pipeline runs A1→A6 in sequence with mocked LLM
✓ each agent receives the correct predecessor output as input
✓ final state has all agents set to COMPLETED
✓ final state contains a non-empty blog post in A6 output
✓ state is saved to checkpoint after each agent completes
✓ total_tokens accumulates correctly across all agents
✓ pipeline can resume from A3 (checkpoint with A1+A2 completed)
✓ pipeline can resume from A5 (checkpoint with A1–A4 completed)
✓ pipeline skips completed agents on resume without re-running them
✓ pipeline halts and sets pipeline_status=FAILED when A3 raises
✓ checkpoint is saved even when an agent fails
✓ pipeline does not run A4 when A3 failed
✓ warnings.log is written even when pipeline fails
✓ output/post.md is written on successful completion
✓ output/audit.json contains per-agent token usage and timings
```

### Integration Tests — Chokepoints

**`tests/integration/test_chokepoints.py`**

Each test uses a mock LLM that returns the specific failure fixture for that agent, then the happy-path fixture for the re-prompt.

```
CP-01: ✓ pipeline completes when A1 returns no-sources fixture; fallback used
CP-02: ✓ A2 re-prompts when source count in output mismatches A1
CP-03: ✓ SHALLOW_CLAIMS warning in A2 record; A5 prompt contains caveat
CP-04: ✓ pipeline pauses on NO_CONTEST; proceeds when user inputs 'y'
CP-04: ✓ pipeline aborts on NO_CONTEST when user inputs 'n'
CP-05: ✓ A3 re-prompts once on conflation; succeeds on second attempt
CP-06: ✓ A4 caps at 3 positions when A3 has 5; TRUNCATED_POSITIONS warning set
CP-07: ✓ A4 re-prompts once when steelman block has no source citation
CP-08: ✓ A5 re-prompts once on hedge phrase in verdict section
CP-08: ✓ A5 raises AgentValidationError on second hedging output
CP-09: ✓ A5 re-prompts when any of the five section headings is missing
CP-10a: ✓ A6 expansion re-prompt triggered; word count checked again after
CP-10b: ✓ A6 cut re-prompt triggered; word count checked again after
CP-11: ✓ A6 concession re-prompt appends text; does not regenerate full post
CP-12: ✓ A6 citation re-prompt lists exactly the missing source names
CP-13: ✓ watchdog fires; agent status set to timed_out; checkpoint saved; pipeline halts
CP-14: ✓ 529 triggers 4 retry attempts with backoff; succeeds on 3rd
CP-14: ✓ 400 raises immediately without retry
CP-15: ✓ no code path exists where agent is running without watchdog active
CP-16: ✓ load raises CheckpointCorruptError; runner alerts user cleanly
CP-17: ✓ CONTEXT_NEAR_LIMIT warning emitted when total_tokens > 80% of limit
CP-17: ✓ A2 output truncated when context limit approached; truncation logged
```

---

## 13. CLI Entry Point

```python
# main.py
"""
Usage:
  python main.py run \
    --topic "The future of email marketing" \
    --audience "Senior marketing leaders" \
    --tone "Direct and analytical" \
    --words 900 \
    --sources "https://example.com/research1" "https://example.com/blog2" \
    --cluster-angle "AI personalisation in email"

  python main.py resume <run-id>

  python main.py dry-run \
    --topic "..." --audience "..." --tone "..."
    # Prints all 6 prompts with state filled in. No API calls.

Options:
  run          Start a new pipeline run
  resume       Resume from checkpoint by run_id
  dry-run      Print all prompts and exit — no LLM calls
"""
```

---

## 14. Output Files

On completion, the runner writes to `./output/{run_id}/`:

| File | Contents |
|------|----------|
| `post.md` | Final blog post in Markdown |
| `audit.json` | Full pipeline state: all inputs, outputs, timings, token usage per agent and total |
| `sources.md` | Formatted reference list from A1 |
| `verdict.md` | A5 judge output: position, reasons, concession, angle |
| `warnings.log` | All `PipelineWarning` events with timestamps — written even on failure |

---

## 15. Environment Variables

```bash
# .env.example
ANTHROPIC_API_KEY=sk-ant-...

# Optional overrides
PIPELINE_MODEL_FAST=claude-sonnet-4-6
PIPELINE_MODEL_DEEP=claude-opus-4-5
PIPELINE_CHECKPOINT_DIR=.pipeline-checkpoints
PIPELINE_LOG_LEVEL=INFO           # DEBUG | INFO | WARNING | ERROR
PIPELINE_MAX_TOTAL_TOKENS=150000  # warn at 80% (120k tokens), hard cap at 100%
```

---

## 16. Implementation Order for Claude Code

Build and pass tests in this sequence. Do not proceed to the next step until all tests for that step pass.

```
Step 1   src/utils/errors.py
         tests/unit/pipeline/test_state.py  →  src/pipeline/state.py

Step 2   tests/unit/pipeline/test_watchdog.py  →  src/pipeline/watchdog.py

Step 3   tests/unit/pipeline/test_checkpoints.py  →  src/pipeline/checkpoints.py

Step 4   tests/unit/llm/test_client.py  →  src/llm/client.py

Step 5   src/agents/base.py
         tests/unit/agents/test_a1.py  →  src/agents/a1_research_collector.py
         tests/unit/agents/test_a2.py  →  src/agents/a2_claim_extractor.py
         tests/unit/agents/test_a3.py  →  src/agents/a3_landscape_mapper.py
         tests/unit/agents/test_a4.py  →  src/agents/a4_devils_advocate.py
         tests/unit/agents/test_a5.py  →  src/agents/a5_evidence_judge.py
         tests/unit/agents/test_a6.py  →  src/agents/a6_blog_writer.py

Step 6   tests/fixtures/  — build all .txt fixture files

Step 7   tests/conftest.py — all shared fixtures

Step 8   tests/integration/test_pipeline.py  →  src/pipeline/runner.py

Step 9   tests/integration/test_chokepoints.py  — all 17 chokepoints

Step 10  src/llm/prompts.py  — all 6 prompt templates as functions
         src/io/input_parser.py
         src/io/output_writer.py

Step 11  main.py  — CLI entry point

Step 12  Full test run: pytest --tb=short -q
         All tests must pass before delivery.
```

---

## 17. Acceptance Criteria

The build is complete when:

- [ ] `pytest --tb=short -q` exits 0 with all tests passing
- [ ] `mypy src/` exits 0 with no type errors
- [ ] `ruff check src/ tests/` exits 0
- [ ] All 17 chokepoints have: a fixture that triggers the failure, a test that fails without the fix, and a test that passes with the fix
- [ ] `python main.py dry-run --topic "x" --audience "y" --tone "z"` prints all 6 prompts and exits 0 without calling the API
- [ ] `python main.py resume <run_id>` skips completed agents and continues from the first non-completed agent
- [ ] A timed-out agent leaves a readable checkpoint JSON file that can be resumed
- [ ] `warnings.log` is written even when the pipeline fails mid-run
- [ ] `audit.json` contains per-agent token usage plus pipeline total
- [ ] No agent can enter `running` state without the watchdog active
