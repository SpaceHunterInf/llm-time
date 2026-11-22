# Task 006 review: close the paid numerical diagnostic

Author: Codex (commander), 11 September 2026. Reviewed the worker return, handoff implementation, all 55 saved responses and 48 results. No model calls or training in this review. Historical files were not rewritten.

## Decision

**Close the current sequence of paid numeric prompt experiments.** The large full-history-plus-state penalty from task 005 did not recur on this six-tape slice. Suffix continuation remains unreliable, but formatting failures and one variable response dominate its excess loss. Correct current-posterior assistance works. These results do not demonstrate a selective-revision learning mechanism or justify training one yet.

Next is [task 007: CPU feasibility of the proposed paired revision objective](../research_tasks/007_paired_revision_feasibility-commander.md). Its purpose is to make the method hypothesis concrete and falsifiable relative to ordinary posterior distillation. No APIs, model downloads, or training in 007. Do not commission another numerical grid simply because inference errors remain.

## Independently verified results

Audit: [task006_audit-commander.json](task006_audit-commander.json).

- **141 tests passed**: `UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q`.
- 48 distinct jobs, 55 provider attempts, **41/48 first-attempt valid**, **43/48 final valid**. Seven repairs, two successful. All finish reasons STOP; all usage metadata available.
- Recomputed cost **$3.7327365**, including smoke, matches ledger. No dangling or interrupted reservations. Tasks 004–006 together cost approximately $21.52 in logged token estimates, including task-004 smoke; this excludes task 002.
- Raw numeric schema, request hashes, complete redraw law, actual prompt-to-job matches, tape fingerprints, and recorded implementation hashes check out. All 24 pairs of base repeat prompts are byte-identical.
- Exact n20 posterior plus observations 21..32 matches reconstruction through n32 within 1e-12 on all six tapes. Checked decisions and own-forecast regrets using independent utility formulas.

| Interface | Valid / jobs | Errors among valid outputs | Errors including safe fallback | Mean loss, all jobs |
|---|---:|---:|---:|---:|
| R: full history | 12/12 | 6 | 6 | .0096521 |
| F: full history + previous reference | 10/12 | 6 | 8 | .0112407 |
| S: previous reference + suffix | 9/12 | 5 | 8 | .0364064 |
| C: current reference | 12/12 | 0 | 0 | 0 |

The worker's 17 decision errors are valid-only. **There are 22 suboptimal decisions including the five safe fallbacks**, all of which occur on diagnose-optimal jobs. This is a denominator clarification, not a disagreement with the worker's reported losses. Do not compare valid-only error counts while hiding failures.

C copies the joint to numerical precision and selects the reference-optimal action in 12/12 jobs. Across the grid there are 32 diagnose-optimal and 16 linked-optimal jobs, with no safe-optimal coverage. Eleven of 17 valid reference errors also conflict with the emitted forecast's optimum. Replacing actions mechanically using emitted probabilities reduces valid errors from 17 to 6; this is an offline diagnostic, not a tested online agent.

## What changes in our interpretation

Tape-averaged F−R loss is **+.0015886**, with only independent_s3 contributing. This is much smaller than task 005's +.0225038 previous-reference penalty. Downgrade the broad redundant-state-harm hypothesis. This is not proof that the original result was random: seeds and checkpoint coverage differ, and the handoff renderer also omits a v2 explanatory paragraph. It is a failure to obtain a comparably large effect in this follow-up.

S−R is **+.0267543**, but independent_s4 contributes **78.26%** of that difference. On identical S base inputs, its first repeat yields an almost exact linked forecast and the wrong action; the second, after schema repair, yields a linked forecast near .999 versus reference .331. Removing that tape leaves +.0069803. Three of five final schema failures are S. This establishes instability on these inputs, not a clean structural-revision deficit.

R/F/C repeat loss agreement is not equivalent to identical forecasts; for example independent_s3 F has different joint errors across repeats. S repeat divergence is notable, but comparing F−R only against S variability is not an uncertainty estimate for F−R. With six tapes and two repeats we do not have a population memory-effect estimate.

All 12 current-posterior cases work, so the simple decision problem is solvable under this assistance. This does not prove that every R/F/S error is purely inference: those conditions require a different computation, and some emitted actions conflict with their emitted probabilities. Freeze deterministic utility maximization as a cheap baseline for future prediction-method comparisons.

## Implementation and operational limits

Own-forecast scoring and raw numeric type rejection are corrected. The two previously reported recovery cases have targeted tests. The completed run's actual records and charges reconcile.

Acceptance remains partial for general interruption safety. By code inspection, `classify_saved_request` permits any appended text containing the repair marker, rather than reconstructing the exact allowed repair from the preceding attempt; identity fields remain optional. A manually saved valid response can be recovered without ledger evidence, with no charge reconstruction. If interrupted attempts exist without an unscored response, retry accounting is not used to set the loop's starting attempt. The scientific manifest also still lists the old three conditions and four checkpoints for this custom-job experiment; per-job prompt hashes protect the recorded requests but do not make that manifest an accurate experiment definition. These limitations do not justify redoing this completed grid. Resolve them before any future paid runner is reused.

Provider observations also matter: **32/55 attempts report more than the configured 8192 thinking tokens**, maximum 9831, although all billed outputs stay below 10240 (maximum 9992). Report the configured setting and actual usage separately. This run does not establish a strictly enforced 8192-token reasoning budget. No cap increase or new provider experiment is requested now.

## Why the next step is method feasibility

We have an expensive-versus-exact inference comparison, interface effects, and reliable decision references. We do not yet have a new learning algorithm. Generic Bayesian teaching already has transfer evidence; its main implementation trains on Bayesian assistant responses, and it also examines explicit preference supervision. See [Qiu et al., §§3 and A.5/C.2](https://arxiv.org/html/2503.17523v3). Learned memory updating for a frozen player is also established by [MemoPilot](https://arxiv.org/html/2606.08656). These primary sources were rechecked; this is not an exhaustive novelty clearance.

The candidate we discussed with the user is matched training examples where identical evidence should produce different revision scopes under different dependency structures. Task 007 specifies a concrete paired loss and asks whether it does anything beyond reweighting ordinary prediction errors. A negative algebra/data result is useful: reject that objective before paying for training. A positive feasibility result only earns a model/data/compute plan and a further novelty comparison, not a novelty claim or an automatic training job.
