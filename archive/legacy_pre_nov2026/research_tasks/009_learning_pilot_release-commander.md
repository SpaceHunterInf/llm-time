# Task 009: finish the learning pilot release

Author: Codex (commander), 14 September 2026. Worker: Grok. Read `auto_research_rules.md`, [008 review](../reports/task008_review-commander.md), and the existing 008 implementation. Preserve historical data, reports and commander plans.

## Objective and boundary

Deliver a runnable, resumable and evaluable external learning job. Most data/model work already exists: repair the actual pipeline, do not replace it. This is the last implementation gate before the planned learning pilot, subject to passing the checks below.

The scientific question is whether ordinary posterior supervision transfers to unseen dependency structures while preserving unrelated forecasts. We have no novel method yet. No paired loss, new benchmark, paid APIs, model-family downloads, hyperparameter sweep or full local training. Reuse the cached Qwen checkpoint. Use uv and at most four CPU threads; no sudo or departmental driver changes.

**Local optimization is verification only: at most four executed optimizer updates per control in this task**, counting interrupted/replayed updates and any local hardware probe. Controls are head-only, LoRA+head and the small numeric MLP below. Track the counts explicitly. Never run a full 2,048-example epoch here. Lightweight unit tests may use stub models; the intended pretrained model must also pass the bounded actual-path check. Stop after returning the package; the commander will review it before the user arranges external training.

## 1. Freeze the data and make timing explicit

Keep the 008 sampled trajectories, families, targets, split membership and sizes: 2,048 train and 256 each dev/same-topology/structural. Do not regenerate until favorable coverage appears.

Create a new versioned rendering directory/manifest, preserving v1. Explicitly state: before each subsequent observation, sample each group's reset event independently of other groups and previous events; apply resets, then generate/condition on that observation. Keep independent member redraws and the no-extra-final-transition clause.

Re-render the same records, record parent hashes, and verify numerical targets are unchanged. Hash both model and audit/target files. Assert joins by stable record identity, not merely row counts. Recheck whitelist, family separation and every input's token length. Do not score learned predictions on the real holdouts in this task.

## 2. One executable configuration and matched initialization

Make the YAML the actual input to probe, fit, resume and evaluate. Reject unknown fields and incompatible overrides. Record the fully resolved configuration before training. Keep the pinned model/tokenizer revision and lexicographic state order from 008.

Head-only and LoRA+head must start with identical base/head weights for each paired seed. Use an explicitly separated head seed or copy the initialized head; do not rely on reseeding before two different construction paths. Verify equal head tensors and evaluation-mode initial predictions, with a stated floating-point tolerance. Verify LoRA gradients/updates separately from head updates.

Retain the six planned language-model runs: modes head/lora, seeds 0/1/2, three epochs, effective batch 32, 192 optimizer steps each. Preserve the 008 head/adapter rates, rank, clipping and schedule. Specify all remaining optimizer defaults explicitly, including weight decay for non-bias weights. Support physical microbatch size with correctly weighted accumulation; clip/step/schedule once per effective batch. Float32 is the initial specification. Do not silently change precision or optimization to accommodate hardware.

## 3. Actual training interruption and identity

Implement resume in the real fit path. Save atomic `last` checkpoints at completed optimizer-step boundaries and separate `best` checkpoints selected solely by epoch-end dev KL. Append/flush metrics as work completes; do not wait until the end to write the run record.

Persist model/head/adapters, optimizer, scheduler, global and private RNG states, epoch/permutation/batch cursor, optimizer-step count, best-selection state and run identity. If checkpointing is restricted to completed accumulated batches, say so and replay only incomplete batches. Handle an interruption at an epoch boundary without skipping or duplicating development selection.

The manifest must include code content hashes (uncommitted code included), model/tokenizer revisions, data/model/target hashes, state order, renderer/numeric-encoder versions, all settings, seed, dependencies, dtype/device/backend and trainable parameter counts. Fail before updating on mismatch or missing/unexpected checkpoint tensors. Do not silently skip unmatched names. A resume should preserve the original experiment identity and append a continuation record.

Test the actual fit/resume path, not a parallel miniature optimizer loop. A small training fixture may run two uninterrupted updates, retain step-one checkpoint, then reload and execute the second update again: three executed updates per control. Use distinct batches so sampler errors cannot pass by replaying one constant minibatch. Compare batch identities, learning rates, scheduler state, parameters, predictions and logs. Exercise epoch-boundary and mismatch handling with inexpensive stub tests. Use separate train-derived verification/dev fixtures; real held-out splits are not plumbing fixtures.

## 4. Add the cheap numeric control

Implement a small two-hidden-layer MLP (64 units per layer, ReLU, 16-logit softmax output) with ordinary joint KL. This is a baseline, not a proposed architecture contribution. No pretrained backbone.

Encode only permitted information in a deterministic numeric representation: 16 previous-joint probabilities in the displayed component order, six pairwise same-group indicators, one hazard, and up to four ordered suffix observations, each represented by a four-way component one-hot, outcome and presence mask. This gives 47 inputs; outcome/redraw rates are fixed constants in this dataset. Relative suffix order is sufficient because these dynamics are time homogeneous. Do not feed hidden prefixes, split labels or targets. Use the displayed previous probabilities consistently and record any rounding/normalization. Add semantic-permutation and metadata-exclusion checks.

Use seeds 0/1/2, the same examples and effective batch order, three epochs/192 updates, AdamW LR 1e-3, the same clipping/warmup schedule and explicit weight-decay policy. Select by dev KL. No tuning. These three cheap runs join the six language-model runs in the future pilot; **do not launch them in this local preflight**. Their unequal architecture/capacity is deliberate: they test whether a language backbone is necessary, not a capacity-matched causal effect of pretraining.

## 5. Implement the evaluation that answers the question

Add checkpoint evaluation and aggregation commands. Load selected checkpoints strictly, run in evaluation mode, save keyed per-example predicted joints and metric records, and produce summaries reproducible from those records. Fit must never read the held-out splits. Freeze/select all nine runs using dev only before evaluating the real holdouts. An incomplete run is not silently omitted from aggregate results.

Use existing exact utilities to derive decisions from predicted marginals. Resolve ties deterministically and compute regret entirely under the reference posterior. Evaluate **all four components**, not only the stored query (which is always observed). Report:

- Joint KL and joint error; marginal absolute error and reference-utility regret.
- Observed, unobserved-related (same reset group as a suffix-observed component), and unobserved-unrelated roles. State the exact partition-based definitions.
- h=0 and h=.03 separately, within train/dev/same-topology/structural and topology types. h=.03 means resets are possible, not that a change occurred. Do not present h=0 structure labels as active structural transfer.
- Transition-only reference r over the suffix duration, and evidence-induced predicted/true marginal changes relative to r. Unrelated retention does not mean blindly keeping q_previous when hazard causes legitimate drift.
- Optimal-action denominators, worst examples, and per-role eligible-example and component counts. Empty cells are NA, never zero performance.

Average component metrics within each eligible example/role before averaging examples, so long suffixes and different group sizes do not silently change weighting. Also provide raw sums/counts. Report each seed plus paired method differences on identical examples. For exploratory uncertainty, bootstrap evidence families with a fixed analysis seed and report intervals separately per training seed; do not count four components or three trained seeds as independent new worlds. This pilot cannot supply a strong significance claim from three training seeds.

Include exact, stale, transition-only and **same-information independent** references. For the latter, take marginals of the supplied q_previous, form their product, then apply singleton-group transitions and the same suffix. Name this projection explicitly; do not replay the hidden prefix under another model. Preserve 008's old full-prefix reference number as historical, not a matched comparison.

Verify evaluation with train-derived fixtures: exact predictions give zero regret/KL to tolerance; deliberately wrong joints produce independently calculable errors; a related-unobserved example is included; all four components are scored; h=0 and unavailable-role counts are correct. Exercise checkpoint loading on a verification artifact. No actual held-out network inference during 009.

## 6. Consistent CPU/external CUDA execution

Provide a uv-managed CPU path and a coherent external CUDA path that normal documented commands will not replace with CPU torch. An isolated task uv project or explicit compatible dependency profiles are acceptable. Keep the existing default environment working. Consult the official [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) if needed; do not reuse the manual-CUDA-install-then-CPU-sync sequence from 008.

External commands must check actual torch build, CUDA availability/device and model placement, failing if CUDA was requested but is unavailable. Verify CPU setup here. If no GPU is usable locally, mark CUDA execution unverified and provide an external verification command; do not claim it passed from dependency resolution alone.

Prepare a probe for both language-model modes with the same microbatch/accumulation, dtype and sequence lengths intended for the pilot. Measure loading, a warmed optimizer step, dev-evaluation and checkpoint overhead, peak memory and throughput. Probe outputs are disposable verification artifacts and never initialize a scientific run. Numeric CPU timing can be measured within the same local update budget. Report actual measured local costs, and leave external GPU hours unknown until measured there.

Package commands for setup, backend check, probe, each fit/resume, selected-checkpoint evaluation and aggregation. State nine runs at 192 steps each, of which six use the pretrained model. Include retry/resume instructions and a disk estimate. Do not book compute or launch the pilot. If hardware requires a material configuration change, return that concrete tradeoff for commander review rather than silently modifying the study.

## 7. Interpretation fixed before learning results

Use dev KL for checkpoint selection only. No settings, data selection or architecture changes based on holdouts. The eventual review compares each trained control against the cheap references and each other, checking related improvement and unrelated harm separately.

- Numeric MLP competitive: numerical filtering alone does not justify an LLM method project.
- LoRA transfers with ordinary KL: do not invent a special loss to claim novelty.
- Good training/same-topology fit but reproducible structural weakness: inspect compositional structure/representation before proposing a method.
- Poor fit: a bounded optimization diagnosis, not a reasoning claim or automatic scale-up.
- Mixed seeds/weak denominators: report inconclusive; no automatic extra runs.

Do not use a single arbitrary KL cutoff or select the best seed to declare success. Keep the pilot exploratory and bounded. A later language-based task requires its own hypothesis and literature comparison; it is not authorized by this brief.

## Return

Return `research_tasks/009_learning_pilot_release-worker.md` and `reports/learning_pilot_release.md` with resolved configs/hashes, actual commands, test results, precise per-control optimizer counts, real fit/resume evidence, matched initialization evidence, numeric/evaluator checks, local resources, external setup/probe/run/resume instructions, and remaining limitations.

Stop after this release. No full local training, API calls, next-task launch or claim of generalization from verification minibatches.
