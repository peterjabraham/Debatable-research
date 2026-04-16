# Blog Agent Pipeline — Claude Code Build Instructions

**Spec:** `docs/AGENT_PIPELINE_SPEC_PYTHON.md`
**Stack:** Python 3.12 · Anthropic Python SDK · Pydantic · pytest
**GitHub:** https://github.com/peterjabraham/Debatable-research.git

---

## Prime directive

Build and pass tests in the sequence defined in §16 of the spec.
**Do not proceed to the next step until all tests for the current step pass.**

---

## Implementation sequence (§16)

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

## Acceptance criteria (§17)

- `pytest --tb=short -q` exits 0 with all tests passing
- `mypy src/` exits 0 with no type errors
- `ruff check src/ tests/` exits 0
- All 17 chokepoints have: a fixture that triggers the failure, a test that fails without the fix, and a test that passes with the fix
- `python main.py dry-run --topic "x" --audience "y" --tone "z"` prints all 6 prompts and exits 0 without calling the API
- `python main.py resume <run_id>` skips completed agents and continues from the first non-completed agent
- A timed-out agent leaves a readable checkpoint JSON file that can be resumed
- `warnings.log` is written even when the pipeline fails mid-run
- `audit.json` contains per-agent token usage plus pipeline total
- No agent can enter `running` state without the watchdog active

---

## Environment

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest --tb=short -q

# Lint
ruff check src/ tests/
mypy src/
```

Required env var: `ANTHROPIC_API_KEY` (see `.env.example`)

---

## Key constraints

- Python 3.12+
- No agent calls another agent directly — all communication via `PipelineState`
- Every agent run must be wrapped in `with_watchdog()` — no exceptions
- Checkpoint written after every agent completion or failure, before any further processing
- `warnings.log` written even on pipeline failure
- Targeted re-prompts only (never full regeneration) for A6 word count, concession, citations
- Models: A1/A2/A6 use `claude-sonnet-4-6`; A3/A4/A5 use `claude-opus-4-5`
