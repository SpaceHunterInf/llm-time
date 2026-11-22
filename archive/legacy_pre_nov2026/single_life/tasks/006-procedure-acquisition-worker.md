# Task 006 — procedure acquisition

Status: ready for review. Date: 29 September 2026. Commander acceptance is pending. This return does not start another paid batch.

## What Gemini did

Gemini 3.5 Flash, temperature 0, thinking level `low`, completed every scheduled call on the first attempt. There were 46 generation requests, 46 `countTokens` requests, 0 retries, 0 failures, and 0 over-cap refusals. Every finish reason was `STOP`. The model-list check was not counted as a generation.

Each lifetime started empty: no observations and no procedure. The call before action 1 deployed the same exploration procedure on every type: `A|single`, `A|pair`, `B|single`, `B|pair`, in that order. The host then executed one full pass. Actions 1–12 were therefore three of each action, grouped by the type cycle. That is the same sample order as the offline local relearner. The call before action 13, which could see those 12 utilities, replaced it on every lifetime with one action per type:

- T1: `A|pair`
- T2: `B|pair`
- T3: `A|pair`

That is the pre-change optimum. Content hash `94752996b25b9779b5cf9d13fc7686f4fab72906be64df0de17548c1d95cfa23`. All eight Gemini lifetimes still held those three lists at actions 25 and 37. The call at 25 was a rationale update with the same lists, so the occurrence counters stayed. Evaluator-only cycle means for that artifact under the pre-change table, saved in `summary.json` and never given to the model: each type has cycle mean 8/1, the uniform mean is 16/4, and the table optimum is 8. The acquired-policy window, actions 25–42, scored 144 utility tenths on every Gemini lifetime (18 actions at 8).

Actions 1–42 also matched the local relearner on all 42 steps in every world. The later window is where the trajectories separate.

Stable revision never changed the three lists after action 13. Calls at 25, 37, 49, 61, 73, and 85 rewrote or restated the rationale and left the lists alone. From the call at 49 through the call at 85 the rationale text itself was identical. Total utility tenths: 720, the same as local relearning and whole relearning, and the same 96 actions.

Shared revision kept the old pairs through the call at 37. The call at 49 had seen actions 43–48, where those pairs had fallen from 8 tenths to 1. It put the four-action rotation back on all three types. The call at 61 then wrote T1 `A|single`, T2 `B|single`, T3 `A|single`, and actions 61–96 executed exactly those, at 5 tenths each. That is the post-change preferred-single rule. Later-window total: 192, against 216 for local and whole relearning and 54 for the frozen procedure, which kept pulling the old pairs at 1 tenth. Full-horizon totals: revision 480, local 504, whole 504, frozen 342, random 210. Action agreement with local relearning: 82 of 96, all 14 disagreements after action 42.

Route revision left T2 `B|pair` and T3 `A|pair` unchanged at every checkpoint after acquisition. The call at 49 changed only T1, from `A|pair` to `A|single`. Its rationale says `A|pair` had fallen from 0.8 to 0.3 on indexes 43 and 46, and that `A|single` had previously yielded 0.5 at index 1. Those later T1 pulls scored 0. The call at 61 changed only T1 again, to `B|pair`, citing the new zeros at indexes 49, 52, 55, and 58. Actions 61–96 then executed T1 `B|pair` at 8, with T2 and T3 still on their acquired pairs at 8. The first post-change edit was a stale acquisition cell; the second edit corrected T1 and did not copy that correction onto T2 or T3. Lists after action 49 had length 1, so the probe count stays 12 and does not show this second look. Full-horizon totals: revision 678, local 704, whole 672, frozen 630, random 365. Action agreement with local relearning: 91 of 96. On this one route trajectory the revision total is above whole relearning and below local relearning. At action 44 the local relearner pulled `B|pair` and the whole relearner pulled `A|single`.

Trap revision left T3 on `A|pair` at every checkpoint. The call at 49 saw T1 `A|pair` and T2 `B|pair` fall to 1 tenth and replaced T1 and T2 with four-action rotations, leaving T3 alone. The call at 61 wrote T1 `A|single`, T2 `A|pair`, T3 `A|pair`. Actions 61–96 executed exactly that: T1 at 5 tenths, T2 and T3 at 8. That is the post-change trap optimum. The T1 single-mode conclusion was not copied onto T2. Probe count 20: the original 12 plus 8 executions from the two length-4 lists. Full-horizon totals: revision 609, local 631, whole 615, frozen 468, random 312. Action agreement with local relearning: 87 of 96. On this one trap trajectory the revision total is above the frozen procedure and below both relearners.

Frozen lifetimes made calls only before actions 1, 13, and 25, then executed the action-25 artifact through action 96. In this run that artifact was the same three lists as revision, hash prefix `94752996b25b`. The schedule can produce different acquired lists; this run happened to match. Frozen's lower API spend is the shorter schedule. Stable frozen also scored 720, because the table did not move. Shared, route, and trap frozen stayed on the pre-change pairs and scored 342, 630, and 468.

Local relearning, cost 0, tied Gemini on the stable world and scored higher on every changing world. Whole relearning tied local on stable and shared. On this route trajectory it scored 672 against the revision total of 678. On this trap trajectory it scored 615 against the revision total of 609. A revision total above the frozen procedure, or above whole reset on one route trajectory, does not identify an LLM-specific gain. These are one trajectory per cell. The gaps are not a population effect, not harmful memory, not novelty, and not RSI.

## Scores

Utility is integer tenths. Windows 1–24, 25–42, and 43–96 are evaluator labels. The model was not told where they fall, and there was no call at action 43. Dollar figures are usage estimates from the published standard rates, not an invoice. Reference cost is 0.

| World | Run | 1–96 | 1–24 | 25–42 | 43–96 | USD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stable | revision | 720 | 144 | 144 | 432 | 0.0467505 |
| stable | frozen | 720 | 144 | 144 | 432 | 0.0171885 |
| stable | uniform-random-v1 | 385 | 112 | 48 | 225 | 0 |
| stable | local-relearning-v1 | 720 | 144 | 144 | 432 | 0 |
| stable | whole-relearning-v1 | 720 | 144 | 144 | 432 | 0 |
| shared | revision | 480 | 144 | 144 | 192 | 0.0601170 |
| shared | frozen | 342 | 144 | 144 | 54 | 0.017481 |
| shared | uniform-random-v1 | 210 | 112 | 48 | 50 | 0 |
| shared | local-relearning-v1 | 504 | 144 | 144 | 216 | 0 |
| shared | whole-relearning-v1 | 504 | 144 | 144 | 216 | 0 |
| route | revision | 678 | 144 | 144 | 390 | 0.0631710 |
| route | frozen | 630 | 144 | 144 | 342 | 0.017868 |
| route | uniform-random-v1 | 365 | 112 | 48 | 205 | 0 |
| route | local-relearning-v1 | 704 | 144 | 144 | 416 | 0 |
| route | whole-relearning-v1 | 672 | 144 | 144 | 384 | 0 |
| trap | revision | 609 | 144 | 144 | 321 | 0.0672165 |
| trap | frozen | 468 | 144 | 144 | 180 | 0.016869 |
| trap | uniform-random-v1 | 312 | 112 | 48 | 152 | 0 |
| trap | local-relearning-v1 | 631 | 144 | 144 | 343 | 0 |
| trap | whole-relearning-v1 | 615 | 144 | 144 | 327 | 0 |

Gemini window sums above were recomputed from the public logs. The reference rows are the saved `summary.md` table.

Semantic deployments (a list change, not a rationale rewrite):

| World | Revision | Frozen |
| --- | --- | --- |
| stable | 1, 13 | 1, 13 |
| shared | 1, 13, 49, 61 | 1, 13 |
| route | 1, 13, 49, 61 | 1, 13 |
| trap | 1, 13, 49, 61 | 1, 13 |

Type lists unchanged between revision checkpoints: stable, all three types across 25→37, 37→49, and 49→61. Shared, all three across 25→37, then none. Route, all three across 25→37, then T2 and T3 across both later gaps. Trap, all three across 25→37, then T3 across both later gaps. Frozen checkpoints at 37, 49, and 61 are the artifact deployed at 25.

Probes, counted as actions whose deployed list for that type had length greater than 1: 12 on every frozen lifetime and on stable and route revision; 24 on shared revision; 20 on trap revision.

## Implemented

Working path used for the live run, from `/home/xz479/llm-time/single_life`:

- `src/single_life/procedure_world.py` — evaluator-only tables in tenths. The prompt builder does not import it.
- `src/single_life/procedure_dsl.py` — public instruction, parser, host interpreter, and the three offline references.
- `src/single_life/procedure_client.py` — `countTokens` then `generateContent`. Input cap 8192, output cap 2048 including thinking, two attempts.
- `src/single_life/procedure_progress.py` — separate tick, call, and attempt bars, atomic `progress.json`.
- `src/single_life/procedure_runner.py` — one lifetime. The response file is written before that action's utility is computed.
- `src/single_life/procedure_pilot.py` — registration, references, preflight, smoke, eight pilot lifetimes, summary, and the researcher SVG.
- `configs/006_worlds.json` — private world document, sha256 `05301f2ebea37450f79fb7654ec56115a4503f8b9432303f15d5c1449d08298a`.
- `tests/test_procedure_acquisition.py` — seven offline tests.

The client imports model id, rates, usage parsing, and key loading from the existing `gemini_adapter.py`. Task 006 does not edit that adapter, `identity.py`, the v0.2 checkpoint allowlists, the parent `pyproject.toml` or `uv.lock`, or task 004/005 outputs. `dependency_identity()` records `runtime_dependencies: []`.

Protocol string on every manifest: `single-life-v0.3-specified`. LLM manifests have `baseline_definition: null` and `llm_checkpoint: disabled`. Reference manifests have `llm_checkpoint: not_used` and usage estimate `0`. This string is the specified extension in `research/protocol-commander.md`. It is not an accepted protocol until the commander says so.

Host rule: each type uses `list[counter % length]`, then increments that counter. Identical action lists keep the counters. Any list change sets all three counters to 0. The rationale is logged and does not select the action. Evidence ids must be completed indexes through t−1. A second invalid procedure, or a repeated transport failure, would abort the lifetime and keep the committed prefix. Repeated HTTP 401, 403, or 429 would stop the task. None of those branches fired.

The same instruction, schema, temperature, thinking level, and token caps were used for smoke, revision, and frozen. Smoke lifetime `life-006-smoke` ran on the stable world for two actions and was not copied into the pilot.

## Executed

Working directory: `/home/xz479/llm-time/single_life`.

```text
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --offline --no-sync python -m unittest discover -s tests -q
```

49 tests, OK, about 4.1 seconds, rechecked after the live run. That is the previous suite plus the seven procedure tests. The new tests cover table arithmetic against the task-004 fractions, the action-43 switch, parser rejection of future evidence, boolean evidence, duplicate actions, and a 1001-character rationale, counter reset when the lists change, frozen immutability after action 25 with exactly three generations, identical prompts through action 42 when only the hidden post table differs, invalid and transport abort, an input count of 8193 with zero `generateContent` calls, unknown usage held at the per-attempt reserve, local relearning's stable total of 720, the route split at action 44 (`B|pair` local, `A|single` whole), a hidden improvement that does not move the local cache, and the progress cap. The sentinel check is on the instruction and on a built prompt. The source guard that rejects a leaked sentinel is allowed to contain the marker.

```text
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python -m single_life.procedure_pilot
```

Exit 0. Wall time 146.098 seconds. `PILOT complete`. Order: twelve offline references, smoke on stable, then stable, shared, route, and trap, revision before frozen inside each world. No second live run.

Parent `pyproject.toml` and `uv.lock` mtimes, rechecked after the live run and again after the offline retest: `2026-09-11 21:34:51.465776170` and `2026-09-11 21:34:52.269776669`. No `uv sync`, no GPU work, no model download.

Git commit `1e76874fc54d0fec9104c9f3a7977dbdd6db0903`, worktree dirty, recorded as dirty. Source sha256 at registration: `262406600071bec7d5f857f0bcd44f280b720c72c6778213b926c8777ae95e2c`. The hash differs from task 005 because these modules were added. `uv.lock` sha256 `85287db42113c1aa43175bdb6aab027b87726e2235b38d3a7ef6771d51c6ac7b`.

Model check: `models/gemini-3.5-flash`, display name Gemini 3.5 Flash, `counted_as_generation: false`.

Price source: https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash , checked 2026-09-29. Standard synchronous row: input US$1.50 / 1M, output including thinking US$9.00 / 1M. The same section also lists US$0.75/US$4.50 and US$2.70/US$16.20. This run did not request batch, flex, or priority. All 46 saved `usageMetadata.serviceTier` values are `standard`.

Preflight, printed and saved before the first `generateContent`:

- 46 calls (2 smoke + 32 revision + 12 frozen), 92 attempts including one retry each.
- Input cap 8192, enforced by provider `totalTokens` before generation. Characters/4 is not an acceptance rule.
- Output cap 2048, including thinking. `thinkingLevel: "low"`. No integer thinking budget, and `thinking_budget=-1` is not sent.
- Per-attempt cap US$0.03072. Completion cap US$2.82624.
- `countTokens` is not added to that cap. The standard row used here does not list a `countTokens` price.
- Unknown generation usage would keep the per-attempt reserve. None occurred.
- Before four completed calls, remaining cost uses two full attempts per unfinished call. After that it uses the observed mean per completed call.
- Settings were not retuned after the smoke.

The first progress line, with references already finished and no generation yet:

```text
PROGRESS [----------]   0% ticks 0/770 (smoke 0/2 rev 0/384 frz 0/384) calls 0/46 (smoke 0/2 rev 0/32 frz 0/12) references 12/12 world=- condition=- tick=0 attempts=0 retries=0 failures=0 token_counts=0 elapsed=0s eta=unknown spent=0 unknown_reserved=0 inflight=0 remaining=2.82624 projected_final=2.82624 cap=2.82624 headroom=2.82624 basis=two_full_attempts_per_remaining_call status=running
```

Percent reached 100 only when status was `complete`, ticks were 770/770, and calls were 46/46. A line at 99% still had four frozen ticks left.

Live accounting from `progress.json`, updated `2026-09-29T11:59:28Z`:

- Attempts 46, retries 0, failures 0. Token-count requests 46, token-count failures 0.
- Known input tokens 59,472. Billed output including thinking 25,293, of which thoughts are 17,782.
- `countTokens` matched `promptTokenCount` on all 46 calls. Maximum counted input 2,480. Maximum billed output 1,033. Maximum thoughts 826.
- Known spend US$0.3168450. Check: 59472 × 1.50/1e6 + 25293 × 9/1e6 = 0.316845. Cap US$2.82624. Headroom US$2.5093950.
- Smoke usage estimate US$0.0101835. The eight pilot estimates sum to US$0.3066615.

Prompt scan of the 46 saved prompts: `banned_hits: []`. The private sentinel, table phase, change index, oracle best action, and progress fields are absent from those prompts. Public log fields are index, type, action, utility tenths, cycle slot, procedure version, procedure hash, and semantic deployment. World and table phase stay in the private log and `private_table.json`.

## Artifact paths

| Path | sha256 |
| --- | --- |
| `experiments/006-procedure-acquisition/summary.json` | `9cff282818edf7d5e0094e924c9917e5fceedaa004d60ad933d90796e1a206f9` |
| `experiments/006-procedure-acquisition/summary.md` | `a00ff0fc71b95e5b30ec85e446d8561b01d098e321e567754f689d3e05722041` |
| `experiments/006-procedure-acquisition/registration.json` | `50824ba93eb61761462e7776023aaae97bf05edb337a3465552f1f6568d5109e` |
| `experiments/006-procedure-acquisition/preflight.txt` | `d6578b1b357ba49ae767700fd216454a7d0384be46c74b14bf267a6179569af1` |
| `experiments/006-procedure-acquisition/progress.json` | `038cc384de8151e9164be76ca91e5e210f3a248e0c015357b636737272502830` |
| `experiments/006-procedure-acquisition/frozen.json` | `cc880fe87c0024645b5927262d7e924a24e01501519065d8e2eeda280803b9df` |
| `experiments/006-procedure-acquisition/model_check.json` | `2b9d3019f10c2ea24654332244554da404450634e753867290c88559198aa9c5` |
| `experiments/006-procedure-acquisition/timeline.svg` | `1ec9cbc0e24825ac4471631b8e67483ac9582942ecb0c55fe77497932c1d29a7` |
| `configs/006_worlds.json` | `05301f2ebea37450f79fb7654ec56115a4503f8b9432303f15d5c1449d08298a` |
| `runs/006-procedure-smoke/manifest.json` | `4df5eeaab156493489a58f3bf113be83ab4c94f18d3130a8d34076c2b484d924` |
| `runs/006-procedure-pilot/stable/revision/manifest.json` | `d049c8c082f0abd3b88ebab1ebfe3ded4e4a062f6f07c7edf77937e11cc64198` |
| `runs/006-procedure-pilot/stable/frozen/manifest.json` | `0e1582ce3a6f9fd54d10e83fffec2e30b506e5b65312686de1f433dd8b75d108` |
| `runs/006-procedure-pilot/shared/revision/manifest.json` | `96846d2d75c944d78046b008d741c011b167586b528f7e00058d504179720fad` |
| `runs/006-procedure-pilot/shared/frozen/manifest.json` | `7dbf48762536a7e0d1cdb8fb37c04fad03d3adfadc56db9fb3ab008e9868272a` |
| `runs/006-procedure-pilot/route/revision/manifest.json` | `2bba13d2a49da0f026ccc3e7c6094e85fc4361c1e36dba98c203d4739f356d8c` |
| `runs/006-procedure-pilot/route/frozen/manifest.json` | `1bd02491269cb6b17bd7678be1a5eb7e036290abd783dae8d50ebbe488a11029` |
| `runs/006-procedure-pilot/trap/revision/manifest.json` | `52f45a4f5a587e6f47cbddfc4e25ecc7addb82ad513475469f9792a9dfb42a0d` |
| `runs/006-procedure-pilot/trap/frozen/manifest.json` | `3585b7bb68fd794d3039b02164405e6cb7a0004afe82642c9b2bdccf0194b4f5` |
| `runs/006-procedure-references/stable/uniform-random-v1/manifest.json` | `aee828d2d2a8e5e982877de9c30122f5ad9bcd410d27ca7f3c1c217f8f1951e1` |
| `runs/006-procedure-references/stable/local-relearning-v1/manifest.json` | `03afb973660f7973e45abace065979836d07334f1c0dbc2576b87a96801bc54f` |
| `runs/006-procedure-references/stable/whole-relearning-v1/manifest.json` | `d4a63a7de613062c9d76c44aa67b90588f535775be003eabbd6d8cf77da719ee` |
| `runs/006-procedure-references/shared/uniform-random-v1/manifest.json` | `b0c755a497d3e32b771c982da62d66d4fb2e873dfa4c559808383c542cd2e694` |
| `runs/006-procedure-references/shared/local-relearning-v1/manifest.json` | `f3afb72a9c18f45672987d3f47e521e24f626d830338b067df4f3023c69d14e3` |
| `runs/006-procedure-references/shared/whole-relearning-v1/manifest.json` | `201b089d9c0e3b961cd00fa84bad14bff0b43ae2d335562695098eb9a42f9f92` |
| `runs/006-procedure-references/route/uniform-random-v1/manifest.json` | `7423d47dfafe96bf55dbc9dcac3b081a9cffea6436c96b99a3e8359b5cd379e1` |
| `runs/006-procedure-references/route/local-relearning-v1/manifest.json` | `f84b7a3e00f6fa0d8ef7e5d96fd6be48b5bd1c536a3a695a677e2a343f630a5b` |
| `runs/006-procedure-references/route/whole-relearning-v1/manifest.json` | `87e0a00d0a6c60aa00e23c583b5e8b797ddacc3e116505d3649d1777c2f20ef8` |
| `runs/006-procedure-references/trap/uniform-random-v1/manifest.json` | `d78d43e0b64e34182d0b0b7a5d304b2d8d67622d4dcdbbaf24d6a168b694c28f` |
| `runs/006-procedure-references/trap/local-relearning-v1/manifest.json` | `1ab4473dfbaf51c9a0b8b3ff774fae18eb86c74d2dbf0e4f28a9f50cf339a259` |
| `runs/006-procedure-references/trap/whole-relearning-v1/manifest.json` | `0d302fb994443c18e1d674fd79d00a619c3cf30eb49d2f6c19dd43baa6d2ecb3` |

Each run directory also has `config.json`, `public_log.jsonl`, `private_log.jsonl`, `artifacts.json`, `private_table.json`, and `responses/`. Response names are `cNNN-a1.json`. The SVG marks the hidden boundary at action 43; that mark is researcher output.

## Limitations

- One lifetime per cell. Temperature 0 did not remove the need for replicates. No second seed was run.
- Utilities are deterministic. A cached action changes only when its new observation disagrees with the cache. That is a property of this toy.
- The pre-change behavior is sample-then-exploit, written as JSON. It matched the local relearner for 42 actions in every world.
- Revision opportunities are a fixed schedule. They are not a learned change detector. Frozen stops calling after action 25 by construction.
- In this run the action-25 lists matched across revision and frozen. A later run can acquire different lists before the freeze. Compare the saved artifacts if that happens.
- The route call at 49 cited index 1, an acquisition sample of 5 tenths, and the following T1 actions scored 0. That is this trajectory. It is not a rate.
- The worlds are known development examples. They are not a withheld test, not an applicability-record method, and not a paper benchmark.
- The dollar figure uses provider token counts and the standard published rates. It is not an invoice. Thoughts are inside the billed output figure.
- LLM checkpoints stay disabled. A crashed lifetime would be left in place. This run completed, so that path was not needed.
- Smoke was a separate two-action lifetime. Its procedure was not loaded into the pilot.

## Handoff

Acquisition worked on these eight trajectories: the call at action 13 locked the pre-change optimum, and actions 25–42 executed it. Observable revision also happened on the three changing worlds. On shared and trap the second post-change procedure matched the new preferred cells, including the trap's refusal to copy T1's single onto T2. On route the first post-change procedure was wrong and the second corrected only T1. Local relearning, with no API cost, tied the stable result and beat every changing-world revision total.

That is enough to close this toy as an engineering fixture. The environment should not be made harder to manufacture a gap. The next commander decision is whether any stronger procedure task justifies applicability testing. This return does not launch that task, does not rerun task 005, and does not mark the result accepted.
