# Task 009 review: first learning result, inadequate absolute fit

Author: Codex (commander), 15 September 2026. Read the release, worker feedback, live GPU report, actual training/evaluation code, logs, manifests and checkpoints. The worker records subsequent user authorization for the nine-run local GPU pilot. That historical run is complete; this review launches no further GPU training or paid calls.

## Decision

**Accept the nine completed runs as an exploratory learning pilot. Do not extend the language-model runs yet.** LoRA improves on the other learned controls, but all learned controls remain worse than simple statistical references even on training data. This is not a clean demonstration of successful in-domain updating followed by structural failure.

Next is [010: one bounded numeric fitting diagnosis](../research_tasks/010_numeric_fit_diagnosis-commander.md), using the existing 8,272-parameter MLP on CPU. Repair the small set of recovery/audit problems needed for that run, freeze one longer schedule, and stop after reporting train/development results. No new architecture or objective and no automatic LoRA extension.

## What I verified

- **174 tests passed**. Tests include tiny numeric optimizer checks; no new research fitting occurred in this review.
- All four v2 splits preserve the v1 current/previous posterior targets and evidence families; all model/audit file hashes match. There are still 2,816 records.
- All nine runs have exactly **192 logged training updates**, three dev evaluations, and best checkpoints selected at step 192. Each epoch uses each training example once. Batch order matches across all three modes for each seed.
- Recorded code/data hashes match current files. The list of code hashes unfortunately omits the training module itself; that limits provenance and needs correction.
- All **36 split evaluations / 25,344 example records** are present. Driver commands place every fit before evaluation; checkpoint/evaluation modification times agree. There are no resume commands in the saved live driver log.
- Independently recomputed marginals from saved `pred_d` plus transition-only marginals, checked reference actions/regret with direct utility formulas, and reaggregated metrics. Maximum per-component error against saved fields: 4.44e-16; summary error: 5.55e-16.
- Recomputed the three cheap reference KLs and decision regrets independently. Reloaded all three numeric checkpoints on CPU and reproduced all their split KL records to a maximum difference below 7.1e-7.
- Reloaded head-only and LoRA seed-0 checkpoints on CPU for eight fixed training examples each. These are spot checks, not a replay of all language-model inference. Results are recorded in the audit artifact.
- Compared saved actual-model verification checkpoints directly: uninterrupted and resumed final trainable parameters match exactly for all three controls. The verification script itself does not assert this comparison.

Audit: [task009_audit-commander.json](task009_audit-commander.json). No full predicted joints were saved by the worker evaluator; thus every saved neural KL cannot be independently reconstructed from the evaluation files alone. The numeric replay and language-model spot checks give additional support without pretending the missing artifact exists.

## The result that matters

Mean teacher-to-student joint KL across three training seeds, lower is better:

| Predictor | Train | Same-topology holdout | Structural holdout |
|---|---:|---:|---:|
| Numeric MLP | 0.2854 | 0.2779 | 0.3281 |
| Frozen backbone + trained head | 0.2843 | 0.2780 | 0.3331 |
| LoRA + trained head | **0.2304** | **0.2225** | **0.2951** |
| Keep the previous joint | 0.1578 | 0.1537 | 0.1541 |
| Apply transitions, ignore new evidence | 0.1202 | 0.1197 | 0.1162 |
| Same-information independent filter | **0.0168** | **0.0198** | **0.0806** |

Exact filtering is zero by construction and remains a hand-coded known-model reference, not a learned competitor. Its continuation can use the same supplied state, dynamics and suffix as the learner. The independent filter starts from the supplied prior marginals and uses the suffix under singleton resets; it ignores the supplied coupling and still beats every learned control substantially.

This ordering also holds for aggregate structural decision regret. LoRA: **0.008568**; stale: **0.006462**; transition-only: **0.005608**; independent: **0.000753**. These use all four decisions per example, including diagnosis cost. The ordinary-KL pilot has not improved actual decisions over the cheap references.

## Important qualifications to the worker interpretation

**Most of the apparent structural improvement is static-case performance.** LoRA's structural KL improvement over head-only is 0.037987. About **77.8%** comes from the quarter of examples with h=0; against the MLP the fraction is **80.1%**. On h=.03 structural examples, LoRA's advantage is only 0.011254 over head-only and 0.008737 over MLP. The ordering is consistent, but it does not show that most of the gain comes from learning coupled resets. At h=0 the reset grouping has no dynamical effect.

**Lower unrelated error than another learner is not successful preservation.** Structural unrelated marginal MAE is about 0.0553 for LoRA, versus effectively zero for transition-only and independent filtering. Related-unobserved MAE is about 0.0639 for LoRA versus 0.0326 for the independent filter. We cannot claim that the central selective-revision behavior has been learned adequately.

**Longer training is a hypothesis, not an established repair.** Every run's dev KL fell through epoch three, and train/test errors are close. This supports investigating insufficient fitting. It does not establish convergence would beat the filters, that model capacity is adequate, or that the remaining gap is purely optimization. The schedule reaches zero learning rate at step 192, so extending `--max-steps` alone would not continue learning.

**There is little average decision headroom over independence in this distribution.** Its structural regret is only 0.000753. This does not bound effects on every informative subgroup or prove selective revision is unimportant. It does mean a large average decision gain cannot be extracted here merely by fixing the remaining coupling approximation. Do not change the sampling distribution or utilities after seeing results and present that as the original experiment.

## Remaining implementation issues

These do not by themselves invalidate the completed uninterrupted runs, but matter before reuse:

1. `load_run_checkpoint` checks only mode, seed, model revision, renderer version and data hashes. I confirmed it accepts a changed learning rate and effective batch. Full config/code/state-order identities are not enforced; evaluation compares the checkpoint to the saved identity rather than independently checking current config/data.
2. `CODE_HASH_PATHS` omits `learning_pilot_train.py`. Identity also lacks the full resolved config and relevant package/tokenizer identities. Future manifests must include them before fitting.
3. If interrupted after the final batch checkpoint but before epoch-end validation, resume enters the batch loop with `cursor >= n`. It can take an empty optimizer/scheduler step before completing validation. A final-step resume may skip pending validation entirely because the loop has already reached its step limit. Recover pending epoch work before testing/entering the next batch. This path was not exercised by the current tests.
4. Logs are appended before checkpoints; a crash between them can leave rows ahead of the last committed state. Resume must reconcile them without silently duplicating steps or selection records. Starting a non-resume fit in an existing run directory should fail instead of overwriting its identity/logs.
5. The reported verification count is wrong: the script executes two uninterrupted updates, a separate one-update run, then one resumed update—**four**, before the additional probe. Thus the main verification-plus-probe sequence is **five per control**, not four; test updates are additional. Preserve historical artifacts and correct the accounting.
6. Evaluation saves scores, not the required 16-state predictions. Save raw keyed joints with checkpoint/config/data hashes. Paired comparisons and role denominators should be produced by the aggregation command, which currently only collects structural summaries.

Do not rerun the nine models to repair paperwork. Preserve this result and fix the paths needed for the cheap next diagnosis. A complete archival replay of all language-model predictions can wait; its scope and cost must be stated if later requested.

## Research direction

There is still **no new method contribution**. This pilot establishes that the current short ordinary-KL recipe learns something, with LoRA stronger than head-only, but does not learn the supplied filter well enough to justify a transfer-method claim. Numerical approximation of a known 16-state filter is not the eventual paper.

The next limited investment tests whether the numeric control can fit substantially better with a longer fixed schedule. If it cannot, stop this implementation branch rather than buy more language-model training. If it can, that provides an optimization baseline, not automatic justification to train Qwen longer. Any continued LLM project must require language-based attribution of which shared mechanism changed and demonstrate an advantage on unobserved linked tasks without collateral harm, as already required by the repository's novelty audit. Designing that task is a separate decision, not an automatic reward for a lower KL.
