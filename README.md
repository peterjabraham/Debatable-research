# Debatable Research — Blog Agent Pipeline

A six-agent pipeline that researches a contested topic, maps the evidence landscape, steelmans opposing positions, delivers a verdict, and writes a publication-ready blog post — all powered by Claude.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/peterjabraham/Debatable-research.git
cd Debatable-research
pip install -e ".[dev]"
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and set:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Preview prompts (no API calls)

```bash
python main.py dry-run \
  --topic "Does remote work hurt career progression?" \
  --audience "knowledge workers and managers" \
  --tone "evidence-based and direct"
```

### 4. Run the pipeline

```bash
python main.py run \
  --topic "Does remote work hurt career progression?" \
  --audience "knowledge workers and managers" \
  --tone "evidence-based and direct" \
  --words 700
```

Output lands in `output/<run-id>/`:

```
output/<run-id>/
├── post.md        ← the blog post
├── verdict.md     ← A5 evidence verdict
├── sources.md     ← A1 source list
├── audit.json     ← per-agent token usage, timing, warnings
└── warnings.log   ← pipeline warnings (written even on failure)
```

### 5. Resume a failed run

```bash
python main.py resume <run-id>
```

Skips completed agents and picks up from the first non-completed one.

---

## Data Flow

```
User Input
  topic / audience / tone / words / sources
          │
          ▼
    ┌─────────────┐
    │ PipelineState│  ◄── single shared object, all agents read/write via it
    └──────┬──────┘
           │
     ┌─────▼──────────────────────────────────────────────────────────────┐
     │                        PipelineRunner                              │
     │                                                                    │
     │  for each agent in [A1 → A2 → A3 → A4 → A5 → A6]:               │
     │    1. transition(state, agent_id, RUNNING)                         │
     │    2. save checkpoint                                              │
     │    3. with_watchdog(timeout_ms, run_agent, on_timeout)            │
     │    4. agent.run(state, signal)                                     │
     │    5. accumulate tokens                                            │
     │    6. save checkpoint                                              │
     └─────┬──────────────────────────────────────────────────────────────┘
           │
     ┌─────▼──────────────────────────────────────────────────────────────┐
     │                          Agents                                    │
     │                                                                    │
     │  A1 ──► A1.output ──► A2 ──► A2.output ──► A3 ──► A3.output ──►  │
     │  A4 ──► A4.output ──► A5 ──► A5.output ──► A6 ──► A6.output      │
     │                                                                    │
     │  (no agent calls another agent directly — all via PipelineState)  │
     └─────┬──────────────────────────────────────────────────────────────┘
           │
     ┌─────▼──────────────────────────────────────────────────────────────┐
     │                     Output Writer                                  │
     │  post.md · verdict.md · sources.md · audit.json · warnings.log    │
     └────────────────────────────────────────────────────────────────────┘
```

---

## Agent Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Six-Agent Sequence                               │
├──────────┬───────────────────────────────────────────────────────────────┤
│  Agent   │  Role                                                         │
├──────────┼───────────────────────────────────────────────────────────────┤
│  A1      │  Research Collector                                           │
│          │  Gathers >= 6 sources with URL / Type / Recency /            │
│          │  Core claim / Credibility signal                              │
│          │  Fallback: training knowledge if no live sources found        │
├──────────┼───────────────────────────────────────────────────────────────┤
│  A2      │  Claim Extractor                                              │
│          │  For each source: Core claim / Key evidence /                │
│          │  Caveats / Implicit assumption                                │
│          │  Re-prompts if block count != source count                   │
├──────────┼───────────────────────────────────────────────────────────────┤
│  A3      │  Landscape Mapper                                             │
│          │  Five sections: Consensus zone / Contested zone /            │
│          │  Outlier positions / Evidence weight / Unresolved question   │
│          │  Pauses for user input if < 2 contested positions found      │
├──────────┼───────────────────────────────────────────────────────────────┤
│  A4      │  Devil's Advocate                                             │
│          │  Steelmans each contested position (max 3):                  │
│          │  Position / Case (3 points) / Hardest objection / Response   │
│          │  Each block must cite a named source                         │
├──────────┼───────────────────────────────────────────────────────────────┤
│  A5      │  Evidence Judge                                               │
│          │  Delivers a verdict — no hedging allowed:                    │
│          │  Verdict / Three reasons / Concession / Angle / What to avoid│
├──────────┼───────────────────────────────────────────────────────────────┤
│  A6      │  Blog Writer                                                  │
│          │  Writes the post: hook / body / concession / conclusion      │
│          │  Word count enforced (+-20%); >= 3 source citations required │
└──────────┴───────────────────────────────────────────────────────────────┘

         topic
           │
    ┌──────▼──────┐     sources
    │     A1      │──────────────────────────────────────────────────────┐
    │  Collector  │                                                       │
    └──────┬──────┘                                                       │
           │ source list                                                  │
    ┌──────▼──────┐                                                       │
    │     A2      │                                                       │
    │  Extractor  │                                                       │
    └──────┬──────┘                                                       │
           │ claims                                                        │
    ┌──────▼──────┐                                                       │
    │     A3      │                                                       │
    │   Mapper    │                                                       │
    └──────┬──────┘                                                       │
           │ contested positions                                           │
    ┌──────▼──────┐                                                       │
    │     A4      │◄──────────────────────────────────────────────────────┘
    │  Advocate   │  (also reads A1 sources for citation checking)
    └──────┬──────┘
           │ steelman blocks
    ┌──────▼──────┐
    │     A5      │◄── A1 sources (for context)
    │    Judge    │
    └──────┬──────┘
           │ verdict + angle
    ┌──────▼──────┐
    │     A6      │◄── A1 sources (for citations), A5 verdict
    │   Writer    │
    └──────┬──────┘
           │
     post.md / verdict.md / sources.md / audit.json
```

---

## State Schema

```
PipelineState
├── run_id            : str          (UUID, auto-generated)
├── topic             : str
├── audience          : str
├── tone              : str
├── target_word_count : int          (default 900)
├── cluster_angle     : str | None   (optional SEO angle)
├── provided_sources  : list[str]    (optional seed URLs)
├── pipeline_status   : PipelineStatus
│     RUNNING | COMPLETED | FAILED | ABORTED
├── started_at        : float        (unix timestamp)
├── completed_at      : float | None
├── total_tokens      : int          (accumulated across all agents)
└── agents            : dict[AgentId, AgentRecord]
      │
      └── AgentRecord
            ├── id            : "A1" | "A2" | ... | "A6"
            ├── status        : AgentStatus
            │     PENDING | RUNNING | COMPLETED | FAILED | TIMED_OUT | SKIPPED
            ├── started_at    : float | None
            ├── completed_at  : float | None
            ├── duration_ms   : float | None
            ├── input         : str           (prompt sent to LLM)
            ├── output        : str | None    (LLM response)
            ├── token_usage   : TokenUsage | None
            │     ├── input_tokens  : int
            │     ├── output_tokens : int
            │     └── total_tokens  : int
            ├── retry_count   : int
            ├── error         : str | None
            └── warnings      : list[str]
                  SHALLOW_CLAIMS | NO_CONTEST | TRUNCATED_POSITIONS | CONTEXT_NEAR_LIMIT
```

Checkpoints are written to `.pipeline-checkpoints/<run-id>.json` after every agent
completion or failure — including on timeout — so any run can be resumed.

---

## Checkpoint & Resume Flow

```
Agent starts
     │
     ▼
transition(RUNNING) ──► save checkpoint
     │
     ▼
with_watchdog(timeout_ms)
     │
     ├── success ──► transition(COMPLETED) ──► save checkpoint
     │
     ├── timeout ──► transition(TIMED_OUT) ──► save checkpoint
     │                    (watchdog sets cancel_event, agent sees signal)
     │
     └── error ───► transition(FAILED) ──► save checkpoint
                         write warnings.log
                         raise (pipeline halts)

Resume:
  python main.py resume <run-id>
       │
       ▼
  load .pipeline-checkpoints/<run-id>.json
       │
       ▼
  get_resume_point(state)  ← first non-COMPLETED agent
       │
       ▼
  continue from that agent (completed agents are skipped)
```

---

## LLM Configuration

| Agent | Model             | Temp | Max Tokens | Role             |
|-------|-------------------|------|------------|------------------|
| A1    | claude-sonnet-4-6 | 0.2  | 2,000      | Research         |
| A2    | claude-sonnet-4-6 | 0.1  | 3,000      | Extraction       |
| A3    | claude-opus-4-5   | 0.3  | 2,000      | Mapping          |
| A4    | claude-opus-4-5   | 0.4  | 3,000      | Argumentation    |
| A5    | claude-opus-4-5   | 0.2  | 1,500      | Judgement        |
| A6    | claude-sonnet-4-6 | 0.7  | 4,000      | Creative writing |

Retries: up to 4 attempts with exponential backoff (2–20 s) on `429 / 500 / 529`.

---

## Project Structure

```
debating-researcher/
├── main.py                        ← CLI entry point (run / resume / dry-run)
├── pyproject.toml
├── .env.example
│
├── src/
│   ├── agents/
│   │   ├── base.py                ← abstract BaseAgent
│   │   ├── a1_research_collector.py
│   │   ├── a2_claim_extractor.py
│   │   ├── a3_landscape_mapper.py
│   │   ├── a4_devils_advocate.py
│   │   ├── a5_evidence_judge.py
│   │   └── a6_blog_writer.py
│   │
│   ├── pipeline/
│   │   ├── state.py               ← PipelineState, AgentRecord, enums
│   │   ├── runner.py              ← PipelineRunner, orchestration loop
│   │   ├── checkpoints.py         ← save / load / get_resume_point
│   │   └── watchdog.py            ← timeout wrapper (asyncio)
│   │
│   ├── llm/
│   │   ├── client.py              ← LLMClient, retry logic, model routing
│   │   └── prompts.py             ← all 6 prompt-builder functions
│   │
│   ├── io/
│   │   ├── input_parser.py        ← CLI args -> PipelineState
│   │   └── output_writer.py       ← state -> output files
│   │
│   └── utils/
│       ├── errors.py              ← custom exceptions + PipelineWarning
│       └── logger.py
│
├── tests/
│   ├── unit/
│   │   ├── pipeline/              ← state / watchdog / checkpoints / client
│   │   └── agents/                ← test_a1.py ... test_a6.py (126 tests)
│   ├── integration/
│   │   ├── test_pipeline.py       ← end-to-end pipeline tests
│   │   └── test_chokepoints.py    ← all 17 chokepoint scenarios
│   └── fixtures/
│       └── llm_responses/         ← 17 .txt fixture files
│
├── .pipeline-checkpoints/         ← auto-created, one JSON per run
└── output/                        ← auto-created, one folder per run
    └── <run-id>/
        ├── post.md
        ├── verdict.md
        ├── sources.md
        ├── audit.json
        └── warnings.log
```

---

## CLI Reference

```
python main.py run
  --topic         TEXT    Topic to research (required)
  --audience      TEXT    Target audience for the blog post (required)
  --tone          TEXT    Writing tone e.g. "Direct and analytical" (required)
  --words         INT     Target word count [default: 900]
  --sources       TEXT    Seed URLs to include (repeatable)
  --cluster-angle TEXT    Optional SEO content cluster angle

python main.py resume <run-id>
  Loads checkpoint and continues from the first non-completed agent.

python main.py dry-run
  --topic / --audience / --tone  (same as run)
  Prints all 6 prompts with state filled in. Zero API calls.
```

---

## Development

```bash
# Run all 223 tests
pytest --tb=short -q

# Type check
mypy src/

# Lint
ruff check src/ tests/

# Install in editable mode
pip install -e ".[dev]"
```

Environment variables (`.env`):

```
ANTHROPIC_API_KEY=sk-ant-...          # required
PIPELINE_MAX_TOTAL_TOKENS=150000      # optional, default 150,000
PIPELINE_LOG_LEVEL=INFO               # optional
```

---

## Chokepoints

The pipeline handles 17 edge cases (CP-01 through CP-17). Key ones:

| CP    | Agent | Trigger                              | Response                                   |
|-------|-------|--------------------------------------|--------------------------------------------|
| CP-01 | A1    | "no sources found" in output         | Re-prompt with training-knowledge fallback |
| CP-02 | A2    | Block count != source count          | Re-prompt with explicit count              |
| CP-03 | A2    | Core claim < 15 words                | Add SHALLOW_CLAIMS warning                 |
| CP-04 | A3    | < 2 contested positions              | Pause, ask user to continue or abort       |
| CP-05 | A3    | Same bullet in Consensus + Contested | Re-prompt to separate zones                |
| CP-06 | A4    | > 3 contested positions              | Cap at 3, log dropped, add warning         |
| CP-07 | A4    | Block missing source citation        | Re-prompt once                             |
| CP-08 | A5    | Hedge phrase in verdict              | Re-prompt once; raise on second hedge      |
| CP-09 | A5    | Missing required section heading     | Re-prompt once                             |
| CP-10 | A6    | Word count outside +-20% of target   | Expand or cut re-prompt                    |
| CP-11 | A6    | No "concession" in post              | Append-only re-prompt                      |
| CP-12 | A6    | < 3 source citations                 | "Weave in: [missing sources]" re-prompt    |

---

## Example Output

**Topic:** Does remote work hurt career progression?  
**Audience:** Knowledge workers and managers  
**Tone:** Evidence-based and direct  
**Words:** 700  
**Total tokens:** ~15,000  
**Models:** claude-sonnet-4-6 (A1/A2/A6) + claude-opus-4-5 (A3/A4/A5)

> **Verdict (A5):** Hybrid remote work arrangements, when properly structured with clear
> expectations and measurable performance criteria, do not produce career penalties — but
> fully remote work likely does create meaningful disadvantage through documented engagement
> deficits and manager bias that most organizations have not adequately addressed.

> **Post headline (A6):** *Your Remote Work Policy Is the Career Risk — Not Remote Work Itself*

---

## License

MIT
