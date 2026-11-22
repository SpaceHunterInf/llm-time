# Single-life adaptation

Active research workspace, established 28 September 2026. Status: tasks 001–003 are accepted as the offline causal kernel; task 005 is the active first-Gemini-diagnostic handoff. No adaptation or method result is claimed.

**Question:** Can an LLM agent improve its cumulative performance in one changing lifetime by attaching testable conditions to learned procedures, investigating when those conditions fail, and revising only the affected procedures?

The agent cannot reset the world or validate a revision using its future. Frozen model weights and a fixed improvement procedure are the first setting. Recursive improvement of the improver is a later question, conditional on evidence.

## Start here

1. [Plain-language research brief](research/BRIEF-commander.md), then [current state and next task](STATE-commander.md).
2. [Research question, hypotheses, and falsifiers](research/question-commander.md).
3. [Causal protocol and data contracts](research/protocol-commander.md).
4. [Commander–Grok workflow](program-commander.md).
5. [First Gemini diagnostic](tasks/005-gemini-lifetime-pilot-commander.md) and [latest review](reviews/004-review-commander.md).

Consult [related work](research/related-work-commander.md) for claim boundaries and [decisions](research/decisions-commander.md) for why this workspace exists. Read these documents instead of treating every historical file in the parent directory as active instructions.

## Layout

```text
single_life/
  AGENTS.md                     operational entrypoint for coding agents
  README.md                     project map
  STATE-commander.md            current evidence, status, one next task
  program-commander.md          working agreement and research loop
  research/                    question, protocol, literature, decisions
  tasks/                       numbered implementation briefs and worker returns
  reviews/                     commander reviews and independent verification
  templates/                   experiment registration and worker return formats
  src/single_life/              causal kernel (contracts, fixture, baselines, runner, CLI)
  tests/                       causal invariants and implementation checks
  configs/                     versioned development experiment configurations
  experiments/                 registrations and compact result summaries
  runs/                        generated logs/artifacts; ignored by Git
  pyproject.toml               independent, initially dependency-free uv project
```

Use `uv` from this directory. The kernel has no runtime dependencies. Hatchling is a build backend so `uv run` can import the package without the parent environment.

```bash
cd single_life
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv sync
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python -m unittest discover -s tests -v
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync single-life run \
  --config configs/dev_continuing.json \
  --run-dir runs/003-preservation-smoke \
  --summary experiments/003-preservation/summary.json \
  --example-decision 61 \
  --example-out experiments/003-preservation/public_input_t61.json
```

The active protocol identity is `single-life-v0.2`. Earlier smoke directories under `runs/001-causal-kernel-smoke/`, `runs/002-kernel-integrity-smoke/`, and `experiments/001-causal-kernel/` and `experiments/002-kernel-integrity/` are historical and are not overwritten. A fresh run refuses a nonempty run directory, and summary or example paths that resolve to the run's logs, the input config, or each other.

`configs/dev_stationary.json` is the paired no-change fixture. It shares the continuing fixture's lifetime id, innovation seed, and initial probabilities so pre-change rewards can be matched. That shared identity is for the pairing check, not an evaluation replicate. No API experiment, GPU job, or trained model is part of this kernel.

## Relationship to the old repository

This is a clean subproject in the existing Git checkout, not a nested Git repository. The parent engine, reports, models, and old tasks remain historical reference material. The new implementation must not import `llm_rl_engine`, depend on the parent environment, or add the parent to `PYTHONPATH`. If a utility is worth porting, copy only the reviewed unit, record provenance, and test its contract here.

The parent `reuse-audit-commander.md` records the legacy audit. Its key blockers are repeated history counting in classical agents, an overly broad agent information interface, coupled schedule/reward randomness, and overwritten retry usage. None has been fixed merely by creating this workspace.

Keep the directory portable. A separate repository can be created when the kernel is reviewed and we want its own release history or collaborators. Extraction must include local research records and an explicit provenance note for any copied code; there should be no runtime dependency on the parent checkout.
