# Task 001 — implementation return

Status: ready for review. Date: 28 September 2026.

Commander acceptance is pending. This return stops at the causal kernel.

## Implemented

New code under `single_life/`. Nothing was ported from the parent engine. No commander file was edited.

Working paths:

- `src/single_life/contracts.py` — strict `PublicView`, `ObservedEvent`, `ArtifactVersion`, `PrivateEvent`, and canonical JSON. Extra fields and non-finite numbers are rejected.
- `src/single_life/fixtures.py` — evaluator-owned Bernoulli fixture. Schedule metadata, outcome innovations, and agent RNG are separate. Innovations are keyed by `(innovation_seed, lifetime_id, action_index, action_id)`. `reveal` scores the chosen arm only.
- `src/single_life/baselines.py` — `random-v1` and `sliding-window-ucb-v1`.
- `src/single_life/artifacts.py` — `scripted-mean-updater-v1` and `scripted-artifact-agent-v1`.
- `src/single_life/runner.py` — one interaction-indexed lifetime, checkpoints, resume.
- `src/single_life/cli.py`, `__main__.py` — `single-life run`.
- `src/single_life/identity.py` — git commit, dirty-worktree flag, source hash.
- `tests/test_causal_kernel.py` — protocol checks.
- `configs/dev_continuing.json` — 180 actions, three arms, hidden boundaries at 61 and 121.
- `configs/dev_stationary.json` — 180 actions, one regime, same initial probabilities and innovation index as the continuing fixture.
- `pyproject.toml`, `uv.lock` — package `single-life-research`, runtime dependencies empty. Hatchling is the build backend so `uv run` imports this tree without the parent environment.
- `README.md` — run commands.

`AssumptionRecord` and `AttemptRecord` are not implemented. The brief defers them.

### Sliding-window UCB

At decision `t`, with window length `W`, the window is global time: unique observed events with `action_index` in `[t - W, t - 1]`. It is not the last `W` pulls of each arm.

`N` is the number of those events whose action is permitted. `N_a` is the number that chose arm `a`. `mean(a)` is their arithmetic mean.

Startup: if any permitted arm has `N_a = 0`, choose the lexicographically smallest such arm. That includes an arm that has slid out of the window.

Otherwise:

```text
score(a) = mean(a) + sqrt(2 * ln(N) / N_a)
```

`ln` is natural. Ties keep the highest score, then the lexicographically smallest action id. The agent RNG is not used.

Each `event_id` is stored once. Supplying the same prefix again does not increase counts. After three observations with true counts arm_a=2 and arm_b=1, repeating that prefix leaves the counts at 2 and 1.

### Scripted artifact

Before the action at decision `t`, if the latest observed action index `c` is a positive multiple of `period` and no artifact already has that cutoff, the updater writes empirical means from the last `window` global-time observations. Parent is the previous version. Supporting ids must exist and have `action_index <= c`. Deployment is for decision `t`, which must be strictly after `c`. The artifact agent then chooses the permitted arm with the highest recorded mean, breaking ties by action id. With no means, it chooses the lexicographically smallest permitted id.

### Tick rules

- Invalid action: the interaction tick is consumed, the attempt is a committed step, and no reward is drawn. `select` is not called again. `reveal` is not called.
- Evaluator failure: the run aborts. The interaction tick is not consumed, no `ObservedEvent` is written, and no other arm is scored. The compute unit reserved at the start of that decision stays consumed.
- Interaction budget exhausted: the runner stops before the next decision and does not call `reveal`.
- Compute budget exhausted: the runner stops before that decision. No interaction tick is consumed and `reveal` is not called.

One chosen action consumes one simulated tick. Host time does not advance the clock. The public view's `compute_budget_remaining` is the balance after reserving the current decision's compute unit. `interactions_remaining` still includes the current tick until it commits.

Checkpoints are written only at completed-step boundaries. Resume loads one checkpoint and continues. An incomplete, truncated, hash-mismatched, or ambiguous checkpoint raises `CheckpointError` before any reveal.

## Executed

Working directory: `/home/xz479/llm-time/single_life`.

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv lock
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv sync
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python -m unittest discover -s tests -v
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync single-life run \
  --config configs/dev_continuing.json \
  --run-dir runs/001-causal-kernel-smoke \
  --summary experiments/001-causal-kernel/summary.json \
  --example-decision 61 \
  --example-out experiments/001-causal-kernel/public_input_t61.json
```

The first `uv sync` failed because Hatch could not see `src/single_life`. `pyproject.toml` then gained `[tool.hatch.build.targets.wheel] packages = ["src/single_life"]`. The commands above are the successful retry.

Tests: `Ran 15 tests in 1.673s`, `OK`. Failures: 0. Errors: 0. Subtests for evidence order, invalid actions, budgets, evaluator failure, and resume of `random`, `ucb`, and `artifact` are inside those 15.

Smoke: `status=completed`, `events=180`, `artifacts=17`, `committed_index=180`, `performance_claim=none`.

| Item | Value |
|---|---|
| Git commit | `1e76874fc54d0fec9104c9f3a7977dbdd6db0903` |
| Worktree | dirty (`single_life/` is untracked; parent `pyproject.toml` and `uv.lock` were already dirty) |
| Source sha256 | `8353e493e99d0448152cd707de47b2ffb07a0a298bf7c4adaaf4df0e6cd36b13` |
| Config sha256 | `debb197dfb3597d660e9e733b06a76b619ac735ea6e97a728bd30751371364ef` |
| Public log sha256 | `1d088bad9f837f9174b0a0903be0ba42e449f22dffe3fbd1c12643bdf7c755dd` |
| Private log sha256 | `7a96e8fde8cf4753f7539faf149bfe33d3b22b431dc06d86900467e063ba2b0a` |
| Artifacts sha256 | `5fae60d49912217fb009b63909f01fec1bb5df9382a6ede26566c76f7efd7e85` |
| Manifest sha256 | `dcf77deeb1763324fc187fae86995ae2d127901583a85e077a875f0b85fcab16` |
| Decision-61 input sha256 | `e496dc1d911ad6565257b14c5539f10edfc8ac8237b05351e4bacd74bac234ce` |
| uv lock sha256 | `85287db42113c1aa43175bdb6aab027b87726e2235b38d3a7ef6771d51c6ac7b` |
| Python | 3.11.5 |
| API calls | 0 |
| API dollars | null |
| Tokens | null |

Raw files, gitignored: `runs/001-causal-kernel-smoke/` (`manifest.json`, `config.json`, `public_log.jsonl` 214 lines, `private_log.jsonl` 180 lines, `artifacts.json`). Compact copies: `experiments/001-causal-kernel/summary.json` and `public_input_t61.json`.

The public log has 180 observed-event lines and 34 artifact lines (17 creations and 17 deployments). The private log has one record per observed event. The public log does not contain `SENTINEL_PRIVATE_do_not_leak` or `regime_id`. The private log does contain the sentinel.

No provider was contacted. Planned spend and actual spend are both zero API dollars.

While locating the project I once ran `uv lock` and `uv sync` in the parent checkout. That sync removed the parent's `train` dependency-group packages from the parent `.venv`. I restored them with `uv sync --group train` (the same 25 packages, including `torch==2.14.0+cpu`). Parent `pyproject.toml` and `uv.lock` were not rewritten; both still have mtime 2026-09-11 21:34.

The porcelain list stored in the smoke manifest was taken before `summary.json` and `public_input_t61.json` were written, so those two paths are absent from that list. `source_sha256` covers `src/single_life/*.py` after the final source edit.

## Acceptance evidence

1. **Future perturbation.** `test_future_perturbation_and_unreachable_metadata`. Two sealed futures share probabilities through action 60 and differ from 61 onward (`p = 0` versus `p = 1` on every arm), with different schedule seeds, private notes, and future-only metadata. Actor inputs, updater inputs, actions, and artifacts deployed at decision 61 or earlier match. Rewards through outcome 60 match. Innovation ids and `innovation_u` through outcome 60 match. Outcome 61 differs. A third run changes only unreachable metadata, the private note, and the schedule seed. Its actor inputs, updater inputs, actions, rewards, and artifacts match the first run for the whole horizon. Its private events differ, so the metadata was stored on the evaluator side.

2. **Information boundary.** `test_information_boundary_reaches_actor_and_updater` wraps both the updater and the actor and requires each received object to be a `PublicView`. Every actor and updater payload through the first hidden boundary was checked for the sentinel and for evaluator keys (`regime_id`, `probabilities`, `innovation_u`, `schedule_seed`, `oracle_best_arm`, manifest fields, and the rest of `BANNED_KEYS`). The private event log contains the sentinel, `regime_id`, and `oracle_best_arm`. The smoke public log and decision-61 input do not.

3. **Evidence counting.** `test_counting_bug_prefix_supplied_repeatedly`. Three distinct events, supplied as a full prefix three more times and then concatenated with itself, stay at counts `{arm_a: 2, arm_b: 1}`. A fresh UCB and the repeated-prefix UCB select the same arm. A 3-step runner that passes the growing prefix on every `select` finishes with `unique_count == 3`. Conflicting payloads for one event id raise `ContractError`.

4. **Persistence.** `test_hidden_change_persists_without_a_boundary_signal` runs the 180-step continuing fixture. At decisions 61 and 121 the lifetime id is unchanged, `public_current_observation` is null, permitted ids are still `arm_a`, `arm_b`, `arm_c`, the observed prefix length is 60 and 120, and the artifact deployed at decision 11 is still accessible. The UCB cache still holds event 1 at the end, with 180 unique events. A later artifact records parent `v1`.

5. **Pairing.** `test_pairing_prechange_outcomes_ignore_future_schedule`. The continuing and stationary configs share `lifetime_id=life-dev-paired`, `innovation_seed=innovation-dev-shared-v1`, and initial probabilities `{arm_a: 0.8, arm_b: 0.2, arm_c: 0.5}`. Their schedule seeds differ. Under one forced 180-action sequence, rewards at actions 1–60 match, and at least one reward in 61–120 differs. Direct `reveal` calls for that prefix agree and each call scores only the requested arm. Changing `schedule_seed` alone leaves the innovation units and the full forced reward tape unchanged. Changing `lifetime_id` changes the innovation unit.

6. **Resume.** `test_resume_matches_uninterrupted_stream` for `random`, `ucb`, and `artifact`. Stop at committed index 8, resume from a new policy object, and compare the semantic stream with an uninterrupted run. The stream has actions, rewards, private events, failures, and artifacts. It has no wall-clock field and no checkpoint directory. `test_incomplete_checkpoint_is_rejected_without_replay` rejects `complete: false`, truncated JSON, a flipped hash, a missing `agent_rng`, and `next_decision_index` that is not `committed_index + 1`. In each case `reveal_calls` stays empty.

7. **Budget and failures.** `test_invalid_action_budget_and_evaluator_failure`.
   - Invalid string and non-string actions at decision 2: reveal indexes are `[1, 3, 4]`. The public failure has `tick_consumed: true` and `reward_revealed: false`.
   - Interaction budget 3 and horizon 10: status `budget_exhausted`, 3 reveals, committed index 3.
   - Compute budget 2: status `compute_budget_exhausted`, 2 reveals, committed index 2.
   - Injected evaluator failure at decision 3: status `evaluator_failure`, committed index 2, reveals `[1, 2]`, public abort has `tick_consumed: false` and `reward_revealed: false`.

8. **Artifact use.** `test_artifact_content_changes_the_next_action`. Four forced pulls (`arm_b`, `arm_b`, `arm_b`, `arm_a`) with probabilities 0, 1, and 0. At decision 5 the updater input has no artifact. The actor input has version `v1`, `creation_cutoff` 4, `deployment_decision_index` 5, and supporting ids from that prefix. The agent selects `arm_b`. The same view with the artifact removed selects `arm_a`.

9. **Core boundary failures.** `test_contracts_reject_invalid_boundary_inputs` and `test_evidence_references_and_deployment_order` raise on extra public fields, NaN/Infinity/boolean rewards, a tampered content hash, deployment at the creation cutoff, missing evidence, evidence after the cutoff, an unknown version, a missing parent, and a parent that is not strictly older. These go through `from_dict` and `ArtifactStore.create` / `deploy`, not only through a constructor that stores the bad value.

10. **Package boundary.** `test_package_has_no_parent_or_provider_imports` and `test_cli_writes_separated_logs`. Source under `src/single_life/` has no `llm_rl_engine`, provider, dotenv, or ToolSandbox reference. The CLI writes `api_calls: 0`, `api_usd: null`, `tokens: null`, and `performance_claim: null`.

### Public input at the first post-change action

Decision 61 is the first action under the second regime. The actor payload is `experiments/001-causal-kernel/public_input_t61.json` (sha256 `e496dc1d911ad6565257b14c5539f10edfc8ac8237b05351e4bacd74bac234ce`). The object below is that payload. The canonical file is one line with sorted keys.

- `decision_index`: 61
- `public_current_observation`: null
- observed events: 60, last event `life-dev-paired:event:60`
- resources at decision time: `interactions_remaining` 120, `compute_budget_remaining` 99939
- newest artifact: `v6`, parent `v5`, `creation_cutoff` 60, `deployment_decision_index` 61
- supporting events: `life-dev-paired:event:31` through `life-dev-paired:event:60`
- accessible versions still present: `v1` through `v6`

```json
{
  "protocol_version": "single-life-v0.1",
  "lifetime_id": "life-dev-paired",
  "decision_index": 61,
  "permitted_action_ids": ["arm_a", "arm_b", "arm_c"],
  "public_current_observation": null,
  "observed_event_count": 60,
  "last_observed_event": {
    "event_id": "life-dev-paired:event:60",
    "action_index": 60,
    "chosen_action": "arm_a",
    "observed_outcome": 1,
    "public_observation": null,
    "actual_interaction_cost": 1
  },
  "artifact_deployed_for_this_decision": {
    "artifact_id": "life-dev-paired:means",
    "version": "v6",
    "kind": "empirical-arm-means",
    "parent_version": "v5",
    "creation_cutoff": 60,
    "deployment_decision_index": 61,
    "content_hash": "8796f78c6ed17d3d7c0da535006f36be9ebc30cafd60283a09e3a2860380eb4d",
    "content": {
      "arm_means": {"arm_a": 0.6666666666666666},
      "evidence_count": 30,
      "window": 30,
      "window_definition": "global-time"
    }
  },
  "remaining_public_resources": {
    "interactions_remaining": 120,
    "compute_budget_remaining": 99939
  }
}
```

`arm_means` contains only `arm_a` because this scripted agent, before any artifact exists, always picks the smallest action id, and the resulting artifact then agrees with that choice. The mixed-prefix test is the demonstration that a deployed mean changes the action. The number above is the fixture input, not a performance result.

## Findings and limitations

The tests support protocol integrity for this offline fixture: prefix-only views, separated randomness, persistence across the two scripted boundaries, exact resume, and explicit failure accounting. They do not support an adaptation claim, a method comparison, or an LLM result. The smoke reward sum is not reported.

Resolutions used where the draft left an operational choice:

- `PublicView.accessible_artifacts` carries the public artifact record, including content, rather than a bare id. The envelope is still strict. Content for this updater is limited to `arm_means`, `evidence_count`, `window`, and `window_definition`.
- Invalid-action records live in the public log and the checkpoint. They are not a new `PublicView` field. The agent sees the gap because `decision_index` moves past the last observed event and `interactions_remaining` drops.
- Evaluator-failure text stays in the private log. The public abort record names `evaluator_failure` and says the tick was not consumed and no reward was revealed.
- The two development configs share a lifetime id and innovation seed so the pairing check is exact. They are one counterfactual pair, not two evaluation replicates.
- The stationary fixture's extra "future segment" sits inside `unreachable_metadata`. It is not installed as a schedule segment.

Untested or out of scope: service/resource environments, partial observability beyond the null bandit observation, adversarial isolation, external-service transaction recovery, provider attempts, statistical replication, and any claim about cumulative utility. Resume was checked inside one process against an uninterrupted run of the same spec. It was not checked across machines or interpreter versions. `random.Random.getstate` is Python 3.11.5's state.

The scripted updater and artifact agent are an interface fixture. On the 180-step smoke they only ever select `arm_a`.

## Handoff

Review the diff, the 15 tests, and `experiments/001-causal-kernel/summary.json`. The kernel gate in `STATE-commander.md` is for commander acceptance after that review.

Particular review points: the public-failure gap, compute reserved before the action, full artifact records on the public view, and the shared lifetime id on the paired fixtures. If those should be protocol text rather than runner choices, the protocol version needs a commander edit before any comparison that depends on them.

No LLM agent and no next experiment were started.
