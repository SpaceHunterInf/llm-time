# Task 008: clean posterior-distillation baseline and bounded preflight

Author: Codex (commander), 11 September 2026. Worker: Grok. Read `auto_research_rules.md` and [the task-007 review](../reports/task007_review-commander.md). Preserve earlier artifacts and commander plans.

## Decision and scope

Retire the paired loss. Prepare **ordinary joint posterior KL distillation** as the first learning baseline. This task delivers a clean data pipeline, a working small-model forward/backward/checkpoint path, and an external-run command with measured local resource requirements. It does not establish a new method or launch the full learning pilot.

No paid API calls. No full dataset training, hyperparameter sweep or benchmark expansion. Downloading the one public model and installing task-scoped dependencies with uv is permitted for verification. Do not download additional model families, change departmental drivers, use sudo or alter global Python. Model files/checkpoints must stay out of git.

Local accelerator use is verification only under the user's rules. At commander review `nvidia-smi` failed; use CPU if this persists. Report the limitation and complete CPU-compatible work without waiting for a driver repair. End with a concrete external job and hardware probe, not a speculative GPU-hour promise.

## A. Clean data and independent teacher checks

Reuse the four-component reset-group model (.99 initial high, .2/.8 outcome rates, .5 reset redraw; every group has an independent event). Keep h=0 controls and h=.03 cases. No paired objective or pair weighting.

Create a NEW versioned dataset for a small future learning pilot:

- 2,048 training examples, 256 development examples, 256 held-out same-topology examples, and 256 structural-holdout examples.
- Training/development/same-topology: balance four singleton groups versus one pair plus two singletons; sample the six pair memberships evenly where possible.
- Structural holdout: balance two-pair versus triple-plus-singleton types; cover all memberships. Do not treat a label permutation as a topology holdout.
- In each split reserve 25% h=0 controls, with the rest h=.03. Generate short trajectories, prefix length 4..8 and suffix length 1..4. Sample observation components independently of hidden state and sample outcomes from the declared generative model. Record the sampling distribution and seeds; do not select records based on model performance.
- Assign an entire evidence family to one split before any relabeling or augmentation. Define family identity from the actual ordered prefix/suffix and boundary, canonicalized under component renaming, ignoring topology, hazard and bookkeeping names. Detect collisions and regenerate deterministically. Hash both exact model inputs and family identities. Keep a clear distinction between same evidence under different models and duplicate individual training rows.
- These are new pilot data, not a final sealed publication test set. Do not evaluate learned models on the held-out splits during preflight or tune generation using their model outputs.

Serialize an explicit model-input whitelist:

```
component names/order
reset groups and hazards
outcome rates and reset-redraw distribution
previous-state timestamp and keyed joint
subsequent timestamped observations ONLY
```

Exclude pair IDs, partner structures, side, family ID, split name, topology-type labels, prefix observations, current posterior, r, d, likelihoods, and all evaluator metadata. Group membership itself is permitted information. Keep prefix/metadata/targets in separate audit records. Test the renderer, not merely JSON key names. Distinct metadata must never change the model-facing text.

Use a fixed lexicographic 16-state head order 0000..1111; characters correspond to the displayed component order. Explicitly convert to/from the legacy module's integer bit order. Test semantic component permutations, not just alternative printed names.

Add a truly independent product-of-group-kernels reference check; the existing brute reference shares its redraw helper with the implementation. Check transitions, replay, likelihoods, no-observation drift, normalized targets, clean model inputs, deduplication and split separation. Historical 128-pair fixtures remain unchanged.

Generate and hash data once. Keep all objective/control variants on the same examples and future sampler order. Report coverage of observed / suffix-unobserved related / unrelated components, h=0 versus h>0, and all three optimal actions under the existing utilities. Do not search away an inconvenient denominator.

## B. Small model and two ordinary-KL controls

Use **Qwen/Qwen3-0.6B-Base** ([official model card](https://huggingface.co/Qwen/Qwen3-0.6B-Base)), pinned to an exact public revision. It is a pretrained base model; use plain text tokenization, not an assumed chat template or a reasoning-mode switch. Obtain only model/tokenizer/config files needed for this checkpoint. Record licenses, revision, package versions and download size.

Use a separate optional uv dependency group or an isolated uv project for torch/transformers/peft. Pin a mutually compatible stack and preserve the existing benchmark environment. A CPU-compatible install is sufficient for this machine; prepare explicit GPU installation instructions for the external environment. Do not install a system CUDA toolkit or rely on FlashAttention/custom compilation.

Architecture: use the pretrained decoder's last non-padding token representation with a linear 16-logit head and float32 log-softmax. Feed only the clean input above. Teacher joint KL is the sole training loss. Keep normalization and target conversion explicit. Do not ask the language model to generate JSON probability strings for this learning baseline.

Two configurations:

1. **Head-only:** frozen backbone, train the 16-state linear head.
2. **LoRA + head:** same initial head and backbone, rank 8 / alpha 16 / dropout 0 adapters on q_proj and v_proj, plus the same head. Ordinary KL, no pair term or extra marginal term.

Head-only is necessary because an untrained random readout cannot serve as a meaningful untreated model baseline. The two configurations differ in trainable capacity; document that. This is not yet a capacity-matched comparison against a new method. Keep exact filtering, transition-only state, stale state, and independent-component filtering as cheap non-neural references.

Use deterministic utility maximization from predicted marginals for action metrics in both configurations. This isolates forecast learning from action arithmetic. Include posterior KL, joint/marginal error, reference decision regret, and h=0/changed-structure results. A 16-state readout is restricted to this four-component task and does not demonstrate language-workflow transfer.

## C. Verification only on this machine

First pass CPU data tests. Then use the ACTUAL selected pretrained checkpoint for the following short checks; a random miniature model alone does not validate the intended path.

- Forward pass on a few training examples, finite KL, normalized probabilities, correct last-token selection with mixed-length padding, no truncation of state/observations.
- One optimizer step for each configuration on the same small TRAINING minibatch (batch 1 or 2). Verify frozen parameters stay unchanged, intended head/adapters receive gradients and update, and outputs are finite. Do not interpret this loss change as research evidence.
- Save/load the head and adapters plus base revision; after reload, evaluation-mode predictions match to a stated dtype tolerance.
- Save/load optimizer and RNG/sampler state; verify that the next training update agrees with an uninterrupted continuation within an appropriate tolerance. Keep scratch verification outputs separate from eventual research checkpoints.
- At most **four optimizer steps per configuration total**, including resume verification. No fitting to the 128 fixtures, no full epoch, no generalization evaluation, no repeated optimizer tuning to force a decrease.

Use float32 on CPU. If a GPU becomes available, inspect actual dtype support rather than assuming native BF16 on the older card. Avoid quantization for this first path. Report visible device, actual threads used (limit CPU threads to a small explicit count), peak resident/accelerator memory, examples/tokens per second, token-length distribution, and checkpoint sizes. A maximum input length of 1024 is a starting bound; fail explicitly on overflow and shorten the serialization without dropping evidence before verification. Do not silently truncate.

If the public download or package access fails, retain the working CPU data module, exact error and ready command. Do not substitute another checkpoint or call a paid API. Do not claim pretrained-model verification succeeded if only a dummy fixture ran.

## D. Prepare, but do not launch, the external pilot

Produce configs and explicit commands for both controls on the new dataset with seeds 0/1/2. Initial bounded pilot specification: 3 epochs, effective batch 32, AdamW, head LR 1e-3, LoRA LR 1e-4, no weight decay on biases, gradient clip 1.0, linear schedule with 10% warmup. These are initial settings, not optimized claims. Select checkpoints using development KL only, and defer all held-out evaluation until that selection is complete. Do not sweep rates in this task.

Freeze hashes of data splits, renderer, state order, code, tokenizer/model revision, adapters and settings in each run manifest. The output must be sufficient to reproduce/resume a run. Preserve examples, seed order, metrics and checkpoints rather than only summary means.

Provide a short external GPU probe command using the same batch/sequence/model configuration. It should measure throughput and peak memory before calculating total training time. CPU timings are not GPU estimates. State the number of optimizer steps (2048/32 × 3 = 192 per run; six runs total) and distinguish head-only from adapter-run costs. Include evaluation and checkpoint overhead when projecting completion. Do not book/rent compute or start the six runs automatically.

A later result would mean:

- LoRA+KL transfers well: ordinary distillation is a strong baseline; do not claim a new loss is needed.
- Training fit succeeds but structural transfer fails: investigate a structural inductive bias only after controlling data coverage, state representation and model capacity.
- Neither configuration fits adequately: debug optimization/representation before interpreting a reasoning limitation.

No result from this small baseline alone establishes novelty. Bayesian teaching and learned memory remain prior art; a future contribution needs a distinct, tested improvement beyond these controls.

## E. Return and stop

Return `research_tasks/008_posterior_baseline_preflight-worker.md` and `reports/posterior_baseline_preflight.md`, with data hashes/coverage, tests, model revision/packages, actual forward/backward/save-load evidence, local resource measurements, unresolved limitations, and ready external commands/configs.

Stop after preflight. Commander will review the concrete script and resource needs with the user before arranging serious training on another machine, as required by `auto_research_rules.md`. Do not ask for external compute before making the job reviewable. No automatic API experiment, full local training or new architecture.
