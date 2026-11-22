# Task 002 — implementation return

Status: ready for review. Date: 28 September 2026.

Commander acceptance is pending. Task 001 stays unaccepted. No next experiment was started.

## Implemented

Protocol identity is now `single-life-v0.2`. Checkpoint format is `single-life-checkpoint-v2`. A v0.1 checkpoint is rejected before any reveal. Historical v0.1 smoke files were not overwritten.

Changed files:

- `src/single_life/__init__.py` — protocol version `single-life-v0.2`.
- `src/single_life/contracts.py` — causal checks on the serialized public view. `ArtifactVersion` records `creation_decision_index` separately from `creation_cutoff`.
- `src/single_life/artifacts.py` — the store is bound to the host decision and the observed catalog. Caller deployment times that are not the current decision are rejected. Prior versions are immutable. Stale evidence can still support a new artifact.
- `src/single_life/runner.py` — updater, actor, evaluator, and post-observation failures stop the lifetime without a second reveal. Compute left in a checkpoint must equal `compute_budget - committed_index`. Supported resume is `random`, `ucb`, and `artifact`, with `scripted-mean-updater-v1` or `null-updater-v1`.
- `src/single_life/cli.py` — preflight before interactions, refuse a nonempty run directory, write identity and flush each committed step, keep a failed manifest.
- `configs/dev_continuing.json`, `configs/dev_stationary.json` — protocol version only.
- `tests/test_causal_kernel.py` — version and host-owned store calls. The original 15 tests still pass.
- `tests/test_kernel_integrity.py` — R1–R4 regressions.
- `README.md` — v0.2 command and the new smoke path.

`reviews/` and the task 001 smoke were not modified.

Supported checkpoint components are the three supplied policies and the two stateless updaters above. Any other updater id is rejected at checkpoint time. This is not general improver-state recovery.

## Executed

Working directory: `/home/xz479/llm-time/single_life`. No parent `uv sync`.

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync python -m unittest discover -s tests -v
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync single-life run \
  --config configs/dev_continuing.json \
  --run-dir runs/002-kernel-integrity-smoke \
  --summary experiments/002-kernel-integrity/summary.json \
  --example-decision 61 \
  --example-out experiments/002-kernel-integrity/public_input_t61.json
```

Tests: `Ran 21 tests in 2.095s`, `OK`. Failures: 0. Errors: 0. The original 15 are included. Six new tests cover the review probes.

Smoke: `status=completed`, protocol `single-life-v0.2`, 180 events, 17 artifacts, `performance_claim=none`.

| Item | Value |
|---|---|
| Git commit | `1e76874fc54d0fec9104c9f3a7977dbdd6db0903` |
| Worktree | dirty |
| Source sha256 | `97e5a0f5d6848c4b25648b7ef893ccc686ad045e89cb81e2b3fdc4afabf83bc7` |
| Config sha256 | `c45a0226e1d03e303d4b7980cde90dc609216d7a37a4e2de78221257b05a7831` |
| Spec sha256 | `9231d313479181ed3b361a6693c978691501779bb6463d751e9b748193fe4a69` |
| Public log sha256 | `2bffb1b38b5331d1fcab5ee9ff2270deb4c89d314b43ae0afcb2564934e2b31b` |
| Private log sha256 | `7a96e8fde8cf4753f7539faf149bfe33d3b22b431dc06d86900467e063ba2b0a` |
| Artifacts sha256 | `4fb93cd9af41a1c6e32e4739e518e0118e52534c99b4f5ea5da892fc61c9d088` |
| Manifest sha256 | `ffeb95c281e0249276ca32a65b6c447624e491ca7c3edc2f2ae1266b416fe69c` |
| Decision-61 input sha256 | `91183f2e52dfdf0277e7be06230ae5d12602b91980b0cf8987609b19812280e1` |
| API calls / dollars / tokens | 0 / null / null |

The private-log hash matches the historical v0.1 smoke. The public log changed because artifact records now include `creation_decision_index`. At decision 61 the newest artifact is `v6`, cutoff 60, creation decision 61, deployment 61. The actor input has 60 observed events and does not contain the sentinel, `regime_id`, or `schedule_seed`.

`experiments/001-causal-kernel/summary.json`, `runs/001-causal-kernel-smoke/manifest.json`, and `reviews/001-kernel-probes.json` kept their previous mtimes.

## Acceptance evidence

| Criterion | Test |
|---|---|
| R1 future observation rejected at the serialized actor/updater boundary | `test_future_observation_rejected_at_serialized_boundary` |
| Legal invalid-action gap still parses | same test, indexes 1 then 3 at decision 4 |
| Unordered events, duplicate indexes, and a non-permitted observed action are rejected | same test |
| Runner-produced actor and updater payloads re-parse; inserting action 60 into the decision-2 payload does not | same test |
| R2 backdated `deploy("v1", 2)` at real decision 4 is rejected, prefix events 1–3 remain, no reward at 4 | `test_backdated_deployment_rejected_and_stale_evidence_kept` |
| Stale evidence may support a new artifact, recorded at creation/deployment decision 4 with cutoff 1 | same test, `HonestLateUpdater` |
| Forged support id and a directly appended deployment-at-2 artifact are rejected and rolled back | same test |
| R3 compute balance restored to the full budget is rejected before `reveal` | `test_inconsistent_checkpoint_rejected_before_reveal` |
| Valid budget-5 checkpoint after 4 commits resumes only through commit 5 | same test (`compute_budget_exhausted`, 5 units) |
| v0.1 checkpoint format is rejected before `reveal` | same test |
| R4 nonempty directory is refused and the prior log is unchanged | `test_cli_preserves_prior_directory_and_failed_prefix` |
| Updater crash at decision 3 leaves events 1–2 on disk, a `running` manifest that then finalizes as `updater_failure`, and the exception text only in the private log | same test |
| A second fresh invocation does not replace that failed run | same test, exit 2 |
| Failure before the action reveals nothing for that decision | `test_failures_before_action_and_after_reward` |
| Failure after reward 2 keeps that reward and does not call `reveal` again | same test, and the CLI file check in the preservation test |
| Resume matches for random, UCB, and artifact after a hidden boundary (61), after deployment (5), and after an invalid tick (3) | `test_resume_matrix_and_independent_process` |
| One random resume runs in a separate process | same test |

The historical `reviews/001-kernel-probes.py` script still targets the v0.1 parser and a caller-chosen deployment time. It was left unchanged. The rows above are the v0.2 replacements. Re-running that script against this tree is not a reproduction of the review record.

The original 15 tests still cover future perturbation, sentinel separation, the counting regression, persistence, pairing, ordinary resume, budgets, artifact use, and the package boundary.

## Findings and limitations

These repairs enforce the v0.2 causal interface. They do not establish adaptation or an LLM result.

The actor's public view still shows an invalid action as a gap in `observed_events` plus a lower `interactions_remaining`. The explicit failure record is in the public log, not in `PublicView`. Actionable invalid-action feedback is still undefined for a later method.

One abstract compute unit is charged when a decision starts, including an attempt that then aborts. A checkpoint is only a completed prefix, so its remaining compute is `compute_budget - committed_index`.

Deployments are kept when the decision commits. An updater, actor, or evaluator failure before commit rolls the store back to the versions from the previous commit, so a failed attempt does not publish a lifecycle. The reward path commits the observation before `policy.observe`. If that hook fails, the reward stays and the run stops.

A machine crash can leave `status=running` with the steps flushed so far. That is local file handling, not recovery of an external service.

Resume covers the three policies and the two stateless updaters named above. A stateful updater is rejected at checkpoint time rather than reconstructed.

## Handoff

Review the diff, the 21 tests, and `experiments/002-kernel-integrity/summary.json`. Task 001 remains revise until this repair is accepted.
