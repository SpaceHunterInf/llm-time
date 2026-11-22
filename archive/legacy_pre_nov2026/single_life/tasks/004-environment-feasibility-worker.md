# Task 004 — implementation return

Status: ready for review. Date: 29 September 2026.

Commander acceptance is pending. No kernel change, fixture, API call, or next experiment was started.

## Implemented

Design and exact arithmetic only. Production source, tests, historical smokes, reviews, and commander files were not modified.

- `experiments/004-environment-feasibility/dossier.md` — feasibility dossier for a factored desk, compared on paper with the dev bandit and with a queued job-service.
- `experiments/004-environment-feasibility/reference_witness.py` — standard-library witness. Hand-specified rules, exact rationals. Does not import `single_life`.
- `experiments/004-environment-feasibility/witness.json` — totals, preservation counts, confound table, and the 36-tick shared-change trajectory.

The dossier recommends the factored desk as the sketch to discuss, and recommends against building the job-service simulator. Under equal weights, per-type recency outscores selective cross-type transfer. That illustration is not a population result.

## Executed

Working directory: `/home/xz479/llm-time/single_life`. No API calls. No paid calls. No GPU work. No parent `uv sync`.

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync python experiments/004-environment-feasibility/reference_witness.py
```

The script exited 0. It checks the response totals named in the dossier and writes `witness.json`.

SHA-256:

| File | SHA-256 |
|---|---|
| `dossier.md` | `ea7586d895eeb4707e3f3c157cac39a33a029201b576db219c1b90db4e21fd41` |
| `reference_witness.py` | `11583de1d763d5cd97cfad3e3b59497a2a03f0c49209b31e89f99f51032b6c2f` |
| `witness.json` | `e2f674129aeb8d999f87e819c892206eb5992db9ccd7bcd2c54ee4c260588fab` |

Parent `pyproject.toml` and `uv.lock` mtimes stayed 2026-09-11. Smoke manifests for tasks 001–003 were not rewritten.

## Acceptance evidence

| Criterion | Where it is checked |
|---|---|
| Feedback and cost accounting are unambiguous | Dossier §1. One tick, one request, one scalar. Probe is an ordinary action. Reward formula and units are explicit. |
| At least one checkable numerical example | Dossier §2–§3. `reference_witness.py` exits nonzero if a checked total changes. Shared-change transfer response is `73/10`. |
| A competent simple baseline is named | `per_type_stop`. On the type-local case it matches the oracle at `72/5`. |
| A plausible negative outcome is identified | Equal-weight response means: per-type `59/5`, transfer `87/8`. The trap alone is `121/10` versus `37/5`. |
| An unsuitable environment may be recommended | The queued job-service is not recommended. The bandit stays a causality fixture. |

Observed response totals, ticks 19–36:

```text
shared_batch        keep 9/5   per_type 63/10   transfer 73/10   reset 63/10   oracle 9/1
type_local_route    keep 57/5  per_type 72/5    transfer 72/5    reset 56/5    oracle 72/5
stable              keep 72/5  per_type 72/5    transfer 72/5    reset 48/5    oracle 72/5
nonadditive_trap    keep 6/1   per_type 121/10  transfer 37/5    reset 21/2    oracle 63/5
```

## Findings and limitations

The acquired desk procedure earns `4/5` per tick before the change and `1/10` after a shared pair-effect change. A route-first probe can return `-2/5`. Copying a successful mode change scores `73/10` on that case, `1` above per-type search and `17/10` below an oracle that already knows the factor. The same copy scores `37/5` on the non-additive trap, below per-type search at `121/10`. Whole reset on the stable case scores `48/5` against `72/5` for doing nothing.

These rows are hand-specified illustrations. Oracle and wrong-factor rows read the hidden table. The rules were written while looking at the four cases, so this directory is not a held-out evaluation. Rewards are deterministic. The equal-weight mean is not a registered prior. No claim is made about LLMs, open-world adaptation, recurrence, or job-shop congestion.

Kernel semantics that a later desk fixture would have to change are listed in dossier §6. They were not changed here. Checkpoint resume is still limited to the accepted exact policy and updater classes.

## Handoff

Codex review only. Suggested decision, not taken here: accept or revise the recommendation to stop at the factored desk and not to build a queue simulator or call a model. If a follow-up is issued, the smallest implementation is an offline information-boundary fixture for this contract, still without a method claim.

Commander acceptance of task 004 is pending.
