# Task 005 — Gemini lifetime pilot

Status: ready for review. Date: 29 September 2026. Commander acceptance is pending.

## What Gemini did

Gemini 3.5 Flash, temperature 0, thinking level `low`, completed every scheduled call. The smoke lifetime and all four pilot lifetimes returned a permitted action and a short note on the first attempt. There were 196 live requests, 196 parser successes, 0 retries, 0 transport failures, and 0 invalid actions. The model-list check was not counted as a decision.

Each lifetime began the same way: pull `arm_a`, then `arm_b`, then `arm_c`, then settle on `arm_a`. The notes describe that as a three-arm look followed by staying with the arm that had paid. On the stationary world the persistent-note run then stayed on `arm_a` for the remaining 45 pulls (46 of 48). The recent-only run on that same world kept leaving `arm_a` to test the other two and scored 26, against 34 for the persistent-note run. On the world whose probabilities move, both runs spent most of actions 4–16 on `arm_a` (recent 12 of 16, persistent 14 of 16, both with reward sum 10). The persistent note at decision 17 still said to keep using `arm_a` because the last 12 rewards looked strong. The first reward from the middle probability table is the outcome of decision 17, so it is not in that prompt. The persistent run stayed on `arm_a` through decision 20 and moved to `arm_b` at decision 21, when its new note said the 12-trial rate had fallen to 6/12. The note carried into decision 21 still said to keep `arm_a`; the model switched anyway. It later left `arm_b` at decision 36 after three failures, while the note carried into that decision still said to keep `arm_b`. The recent-only run on that world probed earlier (it left `arm_a` at decision 15, still inside the first probability segment) and returned to `arm_a` at decision 34. Reward sums on that world were 35 recent-only and 32 persistent-note.

Those four numbers are one trajectory each. They are not a population effect, not a harmful-memory result, and not evidence of self-modification. The notes were rewritten on almost every persistent-note decision and mostly restated the last 12 rewards, which the recent-only condition also sees. A carried sentence did not bind the next action when the recent window moved. Offline sliding-window UCB (window 12) scored 29 on both worlds. Random scored 28 on both. Those references made no API calls.

## How this differs from the legacy work

This is an integration baseline plus a narrow note-retention comparison. It is not a new scientific setting and not an RSI result.

The older bandit runner already chose actions and saw rewards in time order. Calling the unit a lifetime does not change that. The usual legacy history renderer replayed actions, rewards, observations, or engine statistics. Its hypothesis-surprise variant did feed back a previously declared arm, so this run is not the first time any prior declaration was visible. Legacy sequential tasks carried a numerical joint state, and a later condition added an observation suffix. Their evidence tapes were fixed: the evaluated choice did not pick the next observation. Task 006's review already cut back the broad carried-state-harm claim after the large penalty failed to repeat (the F−R loss there was about +0.0016).

Here the model keeps a free-form note plus the last 12 completed observations, and the chosen action determines the next reward. Recent-only records a note and does not carry it. Persistent-note deploys that note for the next decision through the host artifact store. The two conditions therefore do not have identical historical information. They do have the same instruction, the same response schema, the same recent-observation window, and the same token ceilings. Public and private logs, indexed innovations, and per-attempt response files are measurement choices. They do not show that older prompts leaked future rewards, and they do not by themselves make a new method.

## What is uncertain

One pair of lifetimes cannot say whether the 34 versus 26 gap, or the 35 versus 32 gap, would recur. The runs are matched on innovations only when the actions match. After the third pull they often chose differently, so later rewards are not a controlled contrast of the same evidence. Temperature 0 did not remove the need for replicates; there was one call per decision and no second seed. The notes track a 12-step window, so this pilot barely separates "a stored sentence" from "the recent rewards the other condition already sees." Nothing here justifies a larger sweep or training.

If a sharper retention check is wanted, the cheap version is another pair on these same two configs and the same frozen model settings, with the persistent note written during the first few pulls and then held fixed (or rewritten only every 12 decisions). That asks whether an older sentence still changes actions after the recent window has moved. It is not a new method and it should not be launched from this return.

## Implemented

Working path, used for the live run:

- `src/single_life/gemini_adapter.py` — stdlib `urllib` client, prompt builder, parser, note handoff. No parent import and no new runtime dependency.
- `src/single_life/progress.py` — newline progress lines and atomic `progress.json`.
- `src/single_life/pilot.py` — registration, offline references, smoke, four pilot lifetimes, summary and timeline.
- `configs/005_world_w1.json`, `configs/005_world_w2.json` — private 48-step configs. Regime ids do not use the words stable or change.
- `tests/test_gemini_pilot.py` — fake-client checks.
- `src/single_life/fixtures.py` accepts agent names `llm-recent` and `llm-note`.
- `src/single_life/baselines.py` refuses to build those names in `make_policy`, so a CLI run cannot silently substitute another policy.
- `src/single_life/cli.py` records their agent ids. Checkpoint allowlists are unchanged. `save_checkpoint` rejects `GeminiNotePolicy`.

Accounting idea, reimplemented rather than imported: a missing usage object keeps the per-attempt reserve and is not priced as zero. Legacy `DEFAULT_MODEL_PRICING` was not used.

Equivalent token setting, fixed before any call and not retuned after the smoke: Gemini 3.x exposes `thinkingLevel`, not a separate integer thinking budget. This run sends `thinkingLevel: "low"` and `maxOutputTokens: 2048`, which caps visible output and thinking together. The 2,048 reserve stands in for 1,024 visible tokens plus 1,024 thinking tokens. `thinking_budget=-1` is not used. Measured thoughts averaged about 227 tokens per call (44,502 / 196). Every finish reason was `STOP`. Settings stayed at the frozen values.

Parser policy, logged in `registration.json` before the pilot: one JSON object `{"action","strategy_note"}`; a note longer than 1,000 characters is truncated and kept; a bad parse is retried once and then consumes the tick as an invalid action with no reward and no second environment action; transport errors retry once; a repeated auth, permission, or quota error stops the task; any other repeated transport error aborts that lifetime and does not start a replacement. None of those failure branches fired in the live run.

The note proposed at decision t is stored before that decision's reward and deployed at decision t+1. Its creation cutoff is t−1. Recent-only logs the note and does not deploy it. Smoke used lifetime `life-005-smoke` and was not copied into the pilot. Pilot decision 1 in every condition has `retained_strategy_note` null.

## Executed

Working directory: `/home/xz479/llm-time/single_life`.

```text
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync python -m unittest discover -s tests -q
```

42 tests, OK. That is the previous 25 plus the new adapter tests. The fake client covered an allowlisted prompt, an empty starting note, carry only in the persistent condition, truncation to 1,000 characters, a retry that is billed twice and scored once, an invalid action with no reward, a transport abort with no environment step, unknown usage kept at the per-attempt reserve, identical prompts through a shared schedule prefix, a progress sequence with success, retry, unknown usage, and failure, a heartbeat that does not send another request, checkpoint rejection, key redaction, and no `llm_rl_engine` import.

```text
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python -m single_life.pilot
```

Exit 0. Wall time about 411 seconds. Parent `pyproject.toml` and `uv.lock` mtimes are still 2026-09-11 21:34:51 and 21:34:52. No `uv sync`, no GPU work, no model download.

Git commit `1e76874fc54d0fec9104c9f3a7977dbdd6db0903`, worktree dirty, recorded as dirty. Source sha256 at registration: `1e1772057b66e48a00ec1b54a411694cd7fc2fe694eec67926fdf560426d9265`. Runtime dependencies: `[]`.

Model check: `models/gemini-3.5-flash`, display name Gemini 3.5 Flash, not counted as a decision.

Price source: https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash , page updated 2026-09-24 UTC, checked 2026-09-29. Standard synchronous row used here: input $1.50 / 1M, output including thinking $9.00 / 1M. The same section also lists $0.75/$4.50 and $2.70/$16.20 rows. This run did not request batch, flex, or priority.

Preflight, printed and saved before the first `generateContent`:

- 196 decisions (4 smoke + 192 pilot), 392 attempts including one retry each.
- Input cap 4,096. Output cap 2,048, including thinking.
- Per-attempt cap $0.024576 = 4096 × 1.50/1e6 + 2048 × 9.00/1e6.
- Conservative completion cap $9.633792 = that amount × 392. This is the configured spend cap. It is under the about-$20 user ceiling.
- Illustrative short-run figure $0.7644 assumes 196 single attempts at 800 input and 300 output tokens. That figure is not a measurement.
- Before four completed decisions, the remaining-cost line uses two full-cap attempts per unfinished decision. After that it uses the observed mean per completed decision. The $9.633792 cap is not replaced by the projection.
- Unknown usage would stay at the per-attempt reserve. None occurred.
- The dollar figures are a usage estimate from the published rates, not a provider invoice.

Example progress line from the run:

```text
PROGRESS [----------]   6% smoke 4/4 pilot 9/192 world=w1 condition=recent step=9 attempts=13 retries=0 failures=0 elapsed=29s eta=413s spent=0.0328125 unknown_reserved=0 inflight=0 remaining=0.4618990384615384615384615385 projected_final=0.4947115384615384615384615385 cap=9.633792 headroom=9.6009795 basis=observed_mean_per_completed_decision status=running
```

The final line is 100% only with `status=complete`, after smoke 4/4 and pilot 192/192.

Live usage estimate: **$0.6078675**. Known input tokens 73,487. Billed output including thinking 55,293, of which thoughts are 44,502. Unknown reserved $0. Headroom under the cap about $9.03. Per-lifetime estimates: smoke $0.0117705 (reward 2), w1 recent $0.1603080 (reward 26), w1 persistent $0.1494135 (reward 34), w2 recent $0.1445670 (reward 35), w2 persistent $0.1418085 (reward 32). Reference cost is 0, status `offline_reference`, not unknown.

## Acceptance evidence

| Check | Where it shows up |
| --- | --- |
| Allowlisted prompts; no regime ids, sentinels, innovation seeds, or progress fields | Fake-client test, and a scan of all 196 saved prompts with zero hits |
| Memory starts empty; carry only if persistent | Decision 1 `retained_strategy_note` is null in every live lifetime. Recent-only excerpts stay null. Persistent decision 17 quotes the note written at 16 |
| Retry, invalid tick, transport abort, unknown reserve, no second environment action | `tests/test_gemini_pilot.py` (the live run did not exercise these branches) |
| Sealed futures do not change the shared prefix | Fake-client test. Live prompts also lack the private regime ids |
| Checkpoint disabled | `save_checkpoint` raises; manifests say `llm_checkpoint: disabled` for Gemini and `not_used` for references |
| Progress ledger: retry billed once, unknown not zero, no 100% while incomplete, heartbeat does not call the provider | Unit tests. Live file `experiments/005-gemini-lifetime/progress.json` |
| UCB and random on the same innovation keys, labeled references, cost 0 | `runs/005-gemini-references/`, updater `null-updater-v1`, window 12 |
| Responses flushed before the reward | Fake-client order check. Live files are `responses/dNNN-a1.json` under each lifetime (196 files) |

Reward sums, descriptive only:

| Run | Actions 1–48 | 1–16 | 17–34 | 35–48 |
| --- | --- | --- | --- | --- |
| w1 recent | 26 (modal `arm_a`, 24/48) |  |  |  |
| w1 persistent | 34 (modal `arm_a`, 46/48) |  |  |  |
| w1 UCB | 29 |  |  |  |
| w1 random | 28 |  |  |  |
| w2 recent | 35 | 10, modal `arm_a` 12/16 | 13, modal `arm_b` 14/18 | 12, modal `arm_a` 14/14 |
| w2 persistent | 32 | 10, modal `arm_a` 14/16 | 11, modal `arm_b` 14/18 | 11, modal `arm_a` 13/14 |
| w2 UCB | 29 | 9, modal `arm_a` | 11, modal `arm_c` | 9, modal `arm_a` |
| w2 random | 28 | 7, modal `arm_a` | 11, modal `arm_c` | 10, modal `arm_c` |

The evaluator-only higher arm is `arm_a` on the stationary world; on the other world it is `arm_a`, then `arm_b`, then `arm_a`. Those labels were not in the prompts. Both Gemini runs on the moving world did spend the first window mostly on `arm_a`, the middle window mostly on `arm_b`, and the last window mostly on `arm_a`. UCB's middle window did not. That is a description of these four trajectories, not a finding that the note caused the switch or that recurrence was recognized. The persistent notes at the switches talk about the latest 12 rewards, not about a schedule.

Compact timelines and a strip plot: `experiments/005-gemini-lifetime/timeline.svg`, plus `timeline.json` in each run directory. A black dot on the plot marks a deployed-note text change. Recent-only has none. Persistent-note changes on nearly every step because the sentence is rewritten with a new fraction.

Excerpts (model text only; cutoffs are host observations already inside the prompt):

- Decision 1, every pilot lifetime: no observations, retained note null, action `arm_a`, proposed note "Starting exploration phase. Testing arm_a first to establish a baseline."
- w2 persistent, decision 16: 12 observations, latest action index 15. Action `arm_a`. Retained note: "Arm A remains highly successful with 8 rewards out of 11 trials (72.7%). I will continue to exploit Arm A." Proposed note updates that count to 9/12.
- w2 persistent, decision 17: 12 observations, latest action index 16, which is still before any middle-table reward is visible. Action `arm_a`. Retained note: "Arm A continues to perform well with 9 rewards out of 12 trials (75%) in the current window. I will continue to exploit Arm A."
- w2 recent, decision 17: retained note null, action `arm_c`, proposed note says `arm_a` looks strong and `arm_c` is untested in the window.

Full prompts and notes are in the response files. `summary.json` quotes are shortened at 180 or 300 characters and marked when shortened.

## Artifact paths

| Path | sha256 |
| --- | --- |
| `experiments/005-gemini-lifetime/summary.json` | `b64bd6c94484f5aa19db233d6e6b1c15cf547c484a3f3f562e13f2695a3220bb` |
| `experiments/005-gemini-lifetime/registration.json` | `39b0317929df4fcf5cf8b14d48742809bf7db01b115aa8ed6da7397cc0716a77` |
| `experiments/005-gemini-lifetime/preflight.txt` | `ef0421e7218cc376fcc78c9b08e04075bba580ed1445c1009b7fa3b9925accc3` |
| `experiments/005-gemini-lifetime/progress.json` | `6d0d9e4abcd9999c1ab3a2b941bbe1e2e5ae664d1319e265b87aacfef46d5df8` |
| `experiments/005-gemini-lifetime/frozen.json` | `4c5f7b35d9d6204f5d7054dd05594062403c090cb4209997dfa50d04276a4185` |
| `experiments/005-gemini-lifetime/model_check.json` | `c930391ea15ca3211bfe0e8e6307f57c5bcbb82ab6489801aa798bb33d01c71d` |
| `experiments/005-gemini-lifetime/timeline.svg` | `07ac6866cc883486cb471745c2abb0d47becb9083b54c916286685b31b590f58` |
| `experiments/005-gemini-lifetime/summary.md` | (narrative table; hash not required beyond the json) |
| `configs/005_world_w1.json` | `da0bc8d29f05217a20313e1fc4b1163af9c1215fd73f7f4f192b3eaf247e1070` |
| `configs/005_world_w2.json` | `d9e6167c9e98458df44f1244e23534275ec6752bdf287f84c19189566e9add34` |
| `runs/005-gemini-smoke/manifest.json` | `af0b7e1c72a7533aa7657381fc6e6bd1dec617090c36f3c862e22322e919e089` |
| `runs/005-gemini-pilot/w1/recent/manifest.json` | `c9a674aad8d91951cc50d2b61aebce372724677fedb517576faffbd4628fdd7f` |
| `runs/005-gemini-pilot/w1/persistent/manifest.json` | `93e7b6873f9aaa2f747e996724f448c8dc2790c762e674d368c94bac83517c6c` |
| `runs/005-gemini-pilot/w2/recent/manifest.json` | `9fa8c5246dd4ccd1494a9406756e9e0d85abbb8ac64fda728bfab4cd9dc57ba3` |
| `runs/005-gemini-pilot/w2/persistent/manifest.json` | `6fa3f4547e887943ad4890d1fb0631323e5c5639aab760d5db4490b31b31cb36` |
| `runs/005-gemini-references/w1/ucb/manifest.json` | `34718d9d18aa9839da2982cdfb32aae0909785a411e6a0f83db06c38296f2db9` |
| `runs/005-gemini-references/w1/random/manifest.json` | `af79aaa1d2ff5168fa28d39641e2d1b1a2c7a63e09d1931c83226204dbf38f2d` |
| `runs/005-gemini-references/w2/ucb/manifest.json` | `bb4be2edcff92c84c9dabd96c81fc6541ea6b79ced77892b9cfb706bc1cde411` |
| `runs/005-gemini-references/w2/random/manifest.json` | `80c1ee9032e6b8bef3b36507bd67162d543f5b8e857d80b069c4d5c96c7077af` |

Each Gemini run directory also has `config.json`, `public_log.jsonl`, `private_log.jsonl`, `artifacts.json`, `responses/`, and `timeline.json`. Private logs hold the evaluator record. They were not part of the prompt.

## Limitations

- Four lifetimes, one per cell. No second innovation seed, so no variance estimate.
- Reward gaps mix the note rule with different later observations. They are not an effect of memory by themselves.
- Persistent notes were updated every decision. `note_changes` is 46 on w1 and 47 on w2. The intervention is "carry the latest self-written sentence," and that sentence usually repeated the recent window.
- The 12-step window and the 48-step horizon were fixed before the run and not tuned afterward. UCB uses the same window length of 12, which matches the observation window and is not the older development fixture's window of 30.
- The provider's reported token counts were present on every call. The estimate uses those counts and the standard published rates. It is not an invoice. Thoughts are inside the billed output figure, not added a second time.
- LLM checkpoints stay disabled. A crashed lifetime would be left in place and not restarted. This run completed, so that path was not needed live.
- Smoke is a separate lifetime. Its notes were not loaded into the pilot.
- No DeepSeek call, no training, no factored desk, and no queue simulator.

## Handoff

Codex can review the trajectories and the accounting. This return does not mark the result accepted and does not start another paid batch. The useful next question, if any, is whether a note that is held fixed after the first few pulls still changes later actions. That is a small follow-up on this fixture, not a training run and not a sweep.
