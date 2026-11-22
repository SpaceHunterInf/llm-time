# Task 009 return

Author: Grok. No APIs. No full training. No holdout network inference. 008 trajectories frozen. Commander plans not edited.

## Tests

`UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q` → **174 passed**.

Includes v2 timing sentence, same-info independent (no prefix replay), 47-d numeric permutation, exact-zero scoring, empty roles as null, incomplete-eval error, stub fit/resume with distinct batches, identity mismatch refusal, v2 target/id preservation.

## Optimizer counts (cap 4 executed updates / control)

| Control | verify-path (2 unique + 1 replay) | probe | total |
|---|---:|---:|---:|
| mlp | 3 | 1 | 4 |
| head | 3 | 1 | 4 |
| lora | 3 | 1 | 4 |

Matched heads: weight/bias/logprob max abs **0**. CUDA request on CPU torch **rejected**.

## Commands actually run

```bash
python scripts/learning_pilot.py render
uv run --group train python scripts/learning_pilot.py backend-check --device cpu
uv run --group train python scripts/learning_pilot.py backend-check --device cuda  # fails
uv run --group train python scripts/learning_pilot.py verify-path --device cpu
uv run --group train python scripts/learning_pilot.py probe --mode {mlp,head,lora} --device cpu
cd learning_pilot && uv lock
```

Spend **$0**. CPU probe: MLP 0.09 s; head 179 s; LoRA 271 s (not GPU estimates).

## Files

- `llm_rl_engine/analysis/learning_pilot.py`
- `llm_rl_engine/analysis/learning_pilot_train.py`
- `scripts/learning_pilot.py`
- `llm_rl_engine/configs/learning_pilot.yaml`
- `learning_pilot/pyproject.toml` (extras `cpu` / `cu124`)
- `reports/learning_pilot/{data_v2,verify_path.json,token_lengths.json,optimizer_counts.json,live_tables.json,gpu_pilot.log}`
- `reports/learning_pilot_release.md`
- `reports/learning_pilot_live.md` (nine-run GPU results; after user authorization)
- `runs/learning_pilot/` (gitignored checkpoints + evals)

## Limitations / stop (package, 14 Sep)

CUDA unverified on this machine at package time. Safe-optimal remains rare. 16-state supplied-filter task. Fit/eval exist but the nine runs were **not** started in task 009. Stopped for commander/user review.

## GPU pilot (user-authorized after 009)

User: “yes, run the pilot on this GPU.” Isolated `learning_pilot` extra `cu124`, `torch==2.6.0+cu124`, float32, microbatch 8 unchanged. Quadro RTX 8000.

```bash
cd learning_pilot
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv sync --extra cu124
uv run --extra cu124 python ../scripts/learning_pilot.py backend-check --device cuda
uv run --extra cu124 python ../scripts/learning_pilot.py probe --mode head --device cuda
uv run --extra cu124 python ../scripts/learning_pilot.py probe --mode lora --device cuda
UV_CACHE_DIR=/tmp/llm-time-uv-cache python ../scripts/run_learning_pilot_local_gpu.py
```

Nine fits 192/192; all `best` at step 192 by epoch-end dev KL; holdout eval only after selection (`--allow-holdout`). Spend **$0**. Wall ~1 h 52 min. Results: `reports/learning_pilot_live.md`. No further experiment started.
