# Task 001 — Build and verify the causal lifetime kernel

Owner: Grok. Commander: Codex. Status: ready. Date: 28 September 2026.

## Read first

Read [README](../README.md), [operating agreement](../program-commander.md), [question](../research/question-commander.md), and [protocol](../research/protocol-commander.md). Work from the `single_life/` root. Parent tasks 001–012 are historical assignments; do not continue them.

## Objective and scope

Implement a minimal runner demonstrating chronological, prefix-only adaptation in one lifetime. Enforce the public/evaluator boundary and retain artifacts across hidden changes. This establishes infrastructure validity, not a novel adaptive algorithm or LLM result.

CPU only, offline, no provider calls/model downloads/GPU training. Prefer standard-library Python. Do not modify the parent engine, environment, reports, or root lockfile. Use uv here; justify any added dependencies.

## Deliverables

1. **Independent package/command.** Implementation in `src/single_life/`; documented uv invocation working here without parent imports. A small CLI runs a fixture, writes manifest/public log/private log/artifacts, and prints a short summary. Add a local lockfile if needed by the chosen uv setup.
2. **Explicit records.** Implement protocol contracts for public view, observed event, artifact version, private event, and run manifest. Strict serialization excludes extra private fields. Defer unused assumption/API schemas.
3. **Evaluator-owned fixture.** Two or three Bernoulli actions, sealed piecewise probabilities, a no-change condition, and at least two hidden changes in a continuing condition. Use one small development fixture (e.g. 180 actions); no schedule/horizon change hints in agent inputs. Separate parameter, innovation, and agent randomness. Action IDs stay stable within each lifetime.
4. **Simple agents.** Random and one recent-evidence baseline (e.g. sliding-window UCB). Document equation, startup, ties, and whether its window counts global time or per-arm pulls. Consume each outcome once; do not reuse legacy baseline objects.
5. **Persistent artifacts.** Deterministic test updater creates/deploys a small configuration artifact from observed evidence only. Record parents, evidence cutoff, and deployment; demonstrate actual use in subsequent decisions. This is a scripted interface fixture, not self-improvement evidence. No evaluator labels or schedule access.
6. **Checkpoint/resume.** Save world/agent/RNG/artifact state and committed event index at completed-step boundaries. Resume exactly. Reject incomplete checkpoints rather than replaying interactions. This offline fixture makes no claim about external-service transaction recovery.
7. **Evidence bundle.** Offline smoke, compact summary under `experiments/001-causal-kernel/`, raw artifacts under `runs/`. Include config, source identity including dirty changes, seeds, log hashes, and failure status. Do not claim adaptation performance from the smoke.

Suggested modules: `contracts.py`, `runner.py`, `fixtures.py`, `baselines.py`, `artifacts.py`, `cli.py`. Adjust organization for clarity. No plugin registry, dashboard, scheduler, tree search, or arbitrary code execution.

## Acceptance checks

- Test all seven protocol invariants, asserting both actor and updater inputs. Public prompt inspection is insufficient if the updater receives hidden objects.
- Future perturbation: two sealed futures share observations through outcome 60. Action 61 and artifacts deployed for it must match with fixed agent randomness. Later outcomes may differ. Also perturb unreachable future metadata.
- Regression for old counting bug: three distinct observations count as three even if a full prefix is supplied repeatedly.
- No-change/change fixtures with equal initial probabilities and innovations have identical pre-change outcomes under identical actions.
- Reject future/nonexistent evidence references and invalid creation/deployment order. Verify lifetime/artifact persistence through boundaries.
- Explicit behavior for invalid actions, budget exhaustion, and evaluator failures. Document abort versus tick consumption; no free alternative-arm query or silent resampling.
- Resumed/uninterrupted semantic streams match, ignoring wall-clock timestamps/directory names. Incomplete checkpoints are rejected.
- Include intentionally invalid input at core boundaries. Tests must expose failures rather than only validate constructors.
- Package works without parent imports, and loads no credential/provider code.

Use `unittest` or a justified test runner. Report exact command and actual pass/fail count. A short fixture is sufficient; no sweep or significance test.

## Return and stop

Write `tasks/001-causal-kernel-worker.md` using [worker-return template](../templates/worker-return.md). Include changed files, commands, artifacts, evidence per invariant, deviations, and limits. Include a small public-input example at the first post-change action showing its cutoff; do not leak evaluator fields into runtime inputs.

Stop for commander review after completing this task. No LLM agent or next experiment. If an ambiguity affects causal validity, report a proposed resolution while completing independent work. Routine module/serialization choices are yours.
