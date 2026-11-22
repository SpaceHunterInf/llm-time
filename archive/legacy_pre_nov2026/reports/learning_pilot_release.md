# Learning-pilot release (task 009)

14 September 2026. No paid APIs. No full training. No holdout network scores. The paired loss is not used. 008 sampled trajectories were **not** regenerated.

This package makes ordinary joint-KL distillation runnable, resumable, and evaluable. It is not a learning result.

## Data (v2 render of frozen v1 records)

Parent 008 hashes are recorded in `reports/learning_pilot/data_v2/manifest.json`. Numerical `q_current` matches v1 to 1e-12, joined by `example_id`. Renderer v2 adds: each group’s reset is sampled independently of other groups and previous events **before** each subsequent observation; then generate/condition. Independent redraw 0.5 and no extra final transition remain.

Token lengths (pinned Qwen tokenizer, no chat template): train 491–591, structural 456–556. None exceed 1024.

Same-information independent reference: product of supplied `q_previous` marginals, then singleton-group transitions on the suffix only. 008’s prefix-replay independent joint is kept as `independent_fullprefix_lex` (historical, different interface).

## Executable config

`llm_rl_engine/configs/learning_pilot.yaml` is the input to probe/fit/resume/eval/aggregate. Unknown fields are rejected. Float32, effective batch 32, microbatch 8 with loss-weighted accumulation, clip/step/schedule once per effective batch. AdamW β=(0.9, 0.999), ε=1e-8, weight decay 0.01 on non-bias, 0 on bias. Head LR 1e-3, LoRA LR 1e-4, MLP LR 1e-3. Warmup 10% of 192 steps. Checkpoints: atomic `last` after each completed optimizer step; `best` by epoch-end **dev KL only**.

## Matched initialization and resume

Head-only and LoRA+head share an explicit `head_seed` (the run seed). LoRA uses `adapter_seed = seed + 10007`. Measured max abs difference: **weight 0, bias 0, eval log-prob 0** (tolerance 1e-6).

Actual `fit_run` path (not a parallel toy loop): two distinct minibatches, save after step 1, reload, second update. Batch IDs and LRs match the uninterrupted run. Identity mismatch refuses to update. Epoch-boundary state stores `last_dev_epoch` so development selection is not skipped or repeated.

## Numeric MLP

Two hidden layers, 64 units, ReLU, 16-logit softmax. 47 inputs: 16 lex previous joint, 6 pairwise same-group flags, 1 hazard, 4×(4-way one-hot + outcome + presence). No prefix, split, or targets. Semantic permutation is tested.

## Evaluation (not run on holdouts here)

`eval` loads `best.pt` strictly, writes keyed per-example joints and a summary. Fit never opens `same_topology` or `structural`. CLI refuses those splits unless `--allow-holdout` after all nine runs are selected on dev. Incomplete predictions raise rather than drop runs.

Metrics: joint KL/MSE; per-component MAE and reference-utility regret (all four components); roles = observed / same-group unobserved / other groups; h=0 vs h=.03 separately; evidence-induced change vs transition-only `r`. Empty roles are `null`, not zero. Family bootstrap uses analysis seed 909; three training seeds are not extra worlds.

## Local verification (CPU)

| Control | Unique steps | Executed updates (incl. resume replay) | Probe step | **Total** |
|---|---:|---:|---:|---:|
| mlp | 2 | 3 | 1 | **4** |
| head | 2 | 3 | 1 | **4** |
| lora | 2 | 3 | 1 | **4** |

Cap is 4 executed updates per control. Tests: **174 passed**.

CPU probe (batch 32, micro 8, float32, 4 threads): MLP 0.09 s; head 179 s / 8.7 GB RSS; LoRA 271 s / 24.6 GB RSS. **Not GPU-hour estimates.**

`backend-check --device cuda` on this CPU torch build **fails** as required (`torch.cuda.is_available() is False`). CUDA execution is **unverified** here.

## External job (prepared, not launched)

Isolated project `learning_pilot/` with extras `cpu` and `cu124` so `uv sync --extra cu124` does not rewrite the default repo env to CPU torch.

Nine runs × 192 steps: `head`/`lora`/`mlp` × seeds 0/1/2. Six use Qwen3-0.6B-Base `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Disk: ~19 MB v2 data; checkpoints on the order of 15 MB × 2 (`last`+`best`) × 3 LoRA seeds plus tiny head/MLP files; plus one 1.15 GB model cache. Retry: same `fit --resume` command.

```bash
# External CUDA (from learning_pilot/)
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv sync --extra cu124
uv run --extra cu124 python ../scripts/learning_pilot.py backend-check --device cuda
uv run --extra cu124 python ../scripts/learning_pilot.py probe --mode head --device cuda
uv run --extra cu124 python ../scripts/learning_pilot.py probe --mode lora --device cuda
for seed in 0 1 2; do
  for mode in head lora mlp; do
    uv run --extra cu124 python ../scripts/learning_pilot.py fit --mode $mode --seed $seed --device cuda
  done
done
# After all nine best checkpoints exist (dev KL only):
for seed in 0 1 2; do
  for mode in head lora mlp; do
    uv run --extra cu124 python ../scripts/learning_pilot.py eval \
      --run-dir ../runs/learning_pilot/${mode}_s${seed} --split structural --device cuda --allow-holdout
  done
done
uv run --extra cu124 python ../scripts/learning_pilot.py aggregate --runs-root ../runs/learning_pilot
```

Measure GPU probe throughput/memory **before** quoting wall-clock. If hardware cannot run float32 microbatch 8, stop and return the tradeoff; do not silently change dtype.

## Interpretation (fixed before results)

- MLP competitive → this is not an LLM-method project.
- LoRA+KL transfers and preserves unrelated forecasts → do not invent a special loss.
- Train/same-topology fit, structural fail → inspect representation/coverage/capacity first.
- Poor fit → bounded optimization diagnosis.
- Mixed seeds / thin cells (safe is rare) → inconclusive; do not pick a seed.

## Stop

No nine-run launch in task 009. Commander/user review before external training.

## Later (same day)

The user authorized the nine-run job on this GPU. Live numbers: [learning_pilot_live.md](learning_pilot_live.md).
