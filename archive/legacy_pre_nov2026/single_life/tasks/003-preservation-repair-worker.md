# Task 003 — implementation return

Status: ready for review. Date: 28 September 2026.

Commander acceptance is pending. Tasks 001 and 002 stay unaccepted. No next experiment was started.

Protocol identity remains `single-life-v0.2`. Checkpoint format is now `single-life-checkpoint-v3`. Formats `single-life-checkpoint-v1` and `single-life-checkpoint-v2` are rejected before any reveal. Historical smoke directories and `reviews/` were not modified.

## Implemented

- `src/single_life/contracts.py` — artifact content is stored as a deep, read-only copy. `to_dict` returns a detached copy. Content hashes are computed from that plain structure.
- `src/single_life/artifacts.py` — `ArtifactStore.snapshot` copies versions through `to_dict`/`from_dict`, so rollback does not share nested dictionaries with the updater.
- `src/single_life/baselines.py` — the artifact agent reads mapping content, including the read-only form.
- `src/single_life/runner.py` — exact class and configuration checks. A `ScriptedMeanUpdater` period/window mismatch, a UCB window mismatch, and any subclass of a supported component are rejected before reveal. Unrelated in-process test doubles may run, but they cannot be checkpointed. Invalid-action resume coverage uses `forced_actions` on the real policies.
- `src/single_life/cli.py` — authoritative files are replaced by writing a temporary file and `os.replace`. Preflight resolves summary, example, run files, and the input config, including symlink aliases, and rejects collisions, impossible parents, and example indices outside `1..horizon`.
- `tests/test_kernel_integrity.py` — invalid-action resume no longer wraps the policy.
- `tests/test_preservation.py` — R2a, R3a, R4a, and R4b regressions.
- `README.md` — smoke path for this repair.

## Supported checkpoint components

Resume accepts only these exact classes, not subclasses:

| Role | Class | Configuration that must match |
|---|---|---|
| Policy | `RandomAgent` | spec agent may select it; no extra window |
| Policy | `SlidingWindowUCB` | `window == spec.ucb_window` |
| Policy | `ArtifactConfigAgent` | none beyond the class |
| Updater | `ScriptedMeanUpdater` | `period == spec.artifact_period` and `window == spec.artifact_window` |
| Updater | `NullUpdater` | none |

The checkpoint records `policy_type`, `policy_window`, `updater_type`, `updater_period`, and `updater_window`. Restore requires those fields to match the live objects. A test double that is not one of these classes can affect one process, and `save_checkpoint` rejects it.

## Executed

Working directory: `/home/xz479/llm-time/single_life`. No parent environment change. No API calls.

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync python -m unittest discover -s tests -q
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync single-life run \
  --config configs/dev_continuing.json \
  --run-dir runs/003-preservation-smoke \
  --summary experiments/003-preservation/summary.json \
  --example-decision 61 \
  --example-out experiments/003-preservation/public_input_t61.json
```

Tests: `Ran 25 tests in 2.589s`, `OK`. Failures: 0. Errors: 0. That is the previous 21 plus 4 preservation regressions.

Smoke: `status=completed`, protocol `single-life-v0.2`, 180 events, 17 artifacts, `performance_claim=none`. API calls 0, API dollars null, tokens null.

| Item | Value |
|---|---|
| Git commit | `1e76874fc54d0fec9104c9f3a7977dbdd6db0903` |
| Source sha256 | `91c2686972205f1522d2d1c316c34f275d74edc5e0e675bce7c4e992a1159a8b` |
| Config sha256 | `c45a0226e1d03e303d4b7980cde90dc609216d7a37a4e2de78221257b05a7831` |
| Spec sha256 | `9231d313479181ed3b361a6693c978691501779bb6463d751e9b748193fe4a69` |
| Public log sha256 | `2bffb1b38b5331d1fcab5ee9ff2270deb4c89d314b43ae0afcb2564934e2b31b` |
| Private log sha256 | `7a96e8fde8cf4753f7539faf149bfe33d3b22b431dc06d86900467e063ba2b0a` |
| Artifacts sha256 | `4fb93cd9af41a1c6e32e4739e518e0118e52534c99b4f5ea5da892fc61c9d088` |
| Manifest sha256 | `64f936736c73a23df034106a823a239f72348547aa8a7e51714913f3c92610a4` |
| Decision-61 input sha256 | `91183f2e52dfdf0277e7be06230ae5d12602b91980b0cf8987609b19812280e1` |

The public log, private log, artifact file, config, spec, and decision-61 input hashes match the task 002 v0.2 smoke. The manifest hash differs because the preservation note changed. Each log hash matches the bytes in `runs/003-preservation-smoke/`. At decision 61 the newest artifact is still `v6`, cutoff 60, creation decision 61, deployment 61.

Mtimes were unchanged for `experiments/001-causal-kernel/summary.json`, `experiments/002-kernel-integrity/summary.json`, both older run manifests, and `reviews/002-kernel-probes.json`.

## Acceptance evidence

| Criterion | Test |
|---|---|
| R2a nested mutation then exception leaves the actor's artifact bytes and hash intact | `test_historical_artifact_bytes_survive_mutation` |
| R2a replacement caught by update validation, then rolled back to the same bytes | same test, `ReplaceThenContinue` |
| R2a caller mutation of the dict passed to `create` does not change stored content | same test |
| R3a period/window mismatch rejected before reveal | `test_component_identity_rejects_mismatch_before_reveal` |
| R3a subclass of the scripted updater, and a policy subclass, rejected before reveal | same test |
| R3a matching components preserve the deployment sequence across resume | same test |
| v2 checkpoint format rejected before reveal | same test |
| R4a interruption during the event-2 public-log write leaves event 1, `status=running`, and matching file hashes | `test_interrupted_write_keeps_the_committed_prefix` |
| R4b symlink and relative collisions with `public_log.jsonl` refuse before `run_lifetime` and do not change existing bytes | `test_preflight_rejects_collisions_without_running` |
| Example index 0 and a summary path under a file are refused before `run_lifetime` | same test |
| A valid summary repeats the authoritative log hashes | same test |

`reviews/002-kernel-probes.py` was not edited. Those four probes are the cases above: in-place nested mutation, `ScriptedMeanUpdater(3, 1)` against period 2/window 4, truncation during the event-2 write, and `--summary <run-dir>/public_log.jsonl`.

The previous 21 tests still pass, including forced-action resume for random, UCB, and the artifact policy after a hidden boundary, a deployment, and an invalid tick.

## Preservation limits

The authoritative committed prefix is the last manifest whose hashes match `public_log.jsonl`, `private_log.jsonl`, and `artifacts.json`. Each of those files is replaced atomically. An interruption while the next temporary file is being written leaves the previous file in place. A crash after one replacement and before the next can make the directory inconsistent. That state is not treated as success and is not replayed. This does not cover arbitrary disk or power failure, and it does not recover external side effects.

Invalid actions remain a gap in the public view. The explicit failure record stays in the public log.

## Handoff

Review the diff, the 25 tests, and `experiments/003-preservation/summary.json`.
