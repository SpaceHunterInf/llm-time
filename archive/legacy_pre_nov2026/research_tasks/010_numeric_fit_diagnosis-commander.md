# Task 010: one bounded numeric fitting diagnosis

Author: Codex (commander), 15 September 2026. Worker: Grok. Read `auto_research_rules.md` and [009 review](../reports/task009_review-commander.md). Preserve the completed nine-run pilot, all v1/v2 data, and commander plans.

## Decision and question

The learned models lose to transition-only and independent references even on training data. The immediate question is **whether the existing small numeric MLP can fit the training/development distribution substantially better with one longer, fixed schedule**. We are not yet testing a new revision method or interpreting structural failure.

Implement the recovery/audit corrections below and run only this inexpensive numeric diagnosis. No Qwen/LoRA optimization, GPU use, paid API calls, new model downloads, architecture/loss changes, new data or hyperparameter search. This is an 8,272-parameter CPU development experiment; it does not authorize further language-model training on the departmental GPU. Use uv, at most four CPU threads, and no sudo. Do not contact anyone or book compute.

## 1. Repair only what this diagnosis needs

Use the existing trainer/MLP. Before fitting:

- Persist full resolved config, train/dev model and target hashes, encoder/state order, dependency versions and all execution-source hashes, including `learning_pilot_train.py`. Include fixture/subset identities when testing, not the hashes of a different full dataset.
- Strictly validate resume identity and tensors; reject changed optimizer settings, schedule, batch, architecture, encoder, data or code. Keep original experiment identity and append provenance for an allowed device/backend migration rather than ignoring differences. A different training schedule is a new run, never an in-place resume of 009.
- Complete pending epoch validation/selection before drawing another batch, including a resume already at the final optimizer step. Never perform an update on an empty batch. Reconcile metrics to the committed checkpoint so an interrupted write does not duplicate steps/dev records. Refuse a non-resume start in an existing run directory.
- Save keyed **full predicted joints**, selected-checkpoint identity, evaluation data hashes and scores. Verify that recomputing KL/marginals/regret from these files reproduces the summary. Evaluation must validate the live config/data against the intended run, not merely compare two saved copies of identity.
- Add small numeric tests for uninterrupted versus resumed parameters, scheduler, batch order and dev selection; cover interruption immediately before/after epoch validation, final-step interruption and log/checkpoint disagreement. Exercise changed learning-rate/batch/code rejection. These are development checks, not extra research seeds.

Do not turn this into a general training framework rewrite or rerun pretrained verification. Correct the 009 optimizer-count explanation in your new worker return: its actual verification path is 2+1+1 updates, then a probe. Keep historical reports intact and explicitly mark the correction.

## 2. Freeze one CPU experiment before execution

Keep the numeric encoder, MLP (47→64→64→16), ordinary joint KL, sampled train/dev examples and AdamW settings from 009. Precompute numeric features in memory if helpful; verify equality with the original encoder. Do not change example weights or use hidden prefixes.

New runs: seeds **0, 1, 2**, fresh initialization, **30 epochs = 1,920 optimizer updates per seed**, effective batch 32, microbatch 8, LR 1e-3, clip 1, weight decay 0.01 non-bias/0 bias, betas (.9,.999), epsilon 1e-8. Keep the initial warmup at **19 updates**, followed by linear decay over the new 1,920-update schedule. Make the warmup count explicit and test it; do not accidentally stretch the warmup to 192 updates. The longer decay schedule is an intentional change as well as the larger update budget, so this does not isolate update count alone.

Use the same per-seed epoch permutations as 009 for the first three epochs, then deterministic continuation. Select checkpoints using epoch-end dev KL only. Save train/dev learning curves, actual update counts, throughput, peak RSS, and last/best checkpoints. Do not modify `runs/learning_pilot/`; use a new task-specific output directory and config.

Total planned research updates: **5,760**, all on the tiny numeric network. Bound the combined fitting wall time to **10 minutes on CPU with four threads**. This is a resource ceiling, not an early-stopping rule based on a desired score. Save a resumable checkpoint at a completed update boundary if the limit is reached, mark the result incomplete and stop; do not switch to GPU or automatically increase the limit. Prepare the full config/commands and test the code before starting. No additional rate, width, seed or duration variants if results disappoint.

This fixed ceiling is appropriate to a cheap numeric development check. It is not a new arbitrary API dollar cap or permission for serious local language-model training.

## 3. Train/development analysis only

Do not evaluate these newly trained models on `same_topology` or `structural` during this task. Those pilot holdouts have already been inspected in 009; they are not a fresh sealed test set. Resolve fitting first using train/dev.

For each selected seed, report:

- Joint KL, marginal MAE and utility regret, with h=0/h=.03 and observed/related-unobserved/unrelated roles separated.
- Exactly the same metrics for 009's selected MLP, stale state, transition-only and same-information independent references. Use the full 16-state targets and all four components, with eligible component/example denominators. Derive reference numbers from the saved data rather than copying rounded report values.
- Paired per-example differences and each seed separately. Do not treat components or model seeds as independent new environments. Label any intervals exploratory.
- Train/dev curves versus update count and actual LR, best-step location, and whether train/development improvement agrees. Check whether the trained model beats transition-only on both train and dev; separately show its remaining gap to the stronger independent filter. Lower than the 009 MLP alone is not enough to call the filter well learned.

Retain raw prediction files so the commander can verify every reported KL without another network run. Recomputing predictions for the tiny existing numeric checkpoint is permitted. No language-model replay is required.

## 4. Interpretation and stop

- Better training and development fit: larger optimization budget/schedule helped this numeric control. This does not prove that Qwen would benefit equally or establish structural transfer.
- Better training but flat/worse development: an overfitting/generalization issue, not a reason to add still more epochs.
- Both still worse than transition-only: end this fixed-recipe numerical fitting branch. Do not introduce residual heads, new losses or a larger MLP automatically. This is a resource/prioritization decision, not proof that neural filtering is impossible.
- Numeric predictions approach the independent/exact references: useful baseline evidence; still no LLM novelty. Commander decides whether a separate language-based dependency-attribution problem is worth pursuing.

No outcome automatically authorizes a Qwen extension, a fresh benchmark, a new objective, or more runs. The research project must eventually justify why language interpretation or learned attribution is necessary; this supplied numerical filter does not establish that by itself.

## Return

Return `research_tasks/010_numeric_fit_diagnosis-worker.md`, `reports/numeric_fit_diagnosis.md`, config/manifests, raw predictions and machine-readable metrics. Include actual commands, tests, research versus test update counts, runtime/resources, completed/incomplete status, identity/recovery evidence and an explicit interpretation against the above cases.

Stop after this one CPU diagnosis. Preserve all negative results and all seeds.
