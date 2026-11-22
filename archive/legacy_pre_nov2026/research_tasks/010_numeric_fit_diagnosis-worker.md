# Task 010 return

Author: Grok. CPU only. No Qwen/GPU/API/holdout. 009 GPU runs preserved. Commander plans not edited.

## Tests

`UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q` → **183 passed**.

010 tests: warmup 19 vs stretched-192, fixture row hashes, refuse existing dir, reject LR/batch/code, uninterrupted vs resume tensors/scheduler/batches/dev selection, interrupt before/after epoch validation and at the final step, log-ahead reconcile, eval joints recompute.

## Update counts

| Kind | Count |
|---|---:|
| Research MLP (seeds 0/1/2 × 1920) | **5760** |
| 010 tiny-MLP unit tests | 12 |
| Warmup dummy (1 parameter, not MLP) | 192 |

**009 correction (do not rewrite old reports):** verify-path is 2+1+1 executed updates, then a probe → five per control including probe, not four.

## Commands

```bash
uv run --no-sync pytest -q
uv run --no-sync python scripts/numeric_fit_diagnosis.py \
  --config llm_rl_engine/configs/numeric_fit_diagnosis.yaml \
  --out-root runs/numeric_fit_diagnosis run
# eval failed once on set_num_interop_threads after fit; retried eval+analyze
uv run --no-sync python scripts/numeric_fit_diagnosis.py ... eval
uv run --no-sync python scripts/numeric_fit_diagnosis.py ... analyze
```

Fit wall **74.8 s** / budget 600 s. RSS 432 MB. 4 threads. Spend **$0**. All three runs complete (best = step 1920).

## Files

- `llm_rl_engine/analysis/learning_pilot.py` (identity, hashes, eval joints, reconcile)
- `llm_rl_engine/analysis/learning_pilot_train.py` (strict resume, epoch recovery, warmup_steps, empty-batch guard)
- `llm_rl_engine/configs/numeric_fit_diagnosis.yaml`
- `scripts/numeric_fit_diagnosis.py`
- `tests/test_numeric_fit_diagnosis.py`
- `reports/numeric_fit_diagnosis.md`
- `reports/numeric_fit_diagnosis/{analysis.json,fit_status.json,eval_status.json,run.log}`
- `runs/numeric_fit_diagnosis/mlp_s{0,1,2}/` (checkpoints, metrics, `eval_*_preds.jsonl`)

`warmup_steps: 19` was also recorded on the 009 YAML (same integer as `int(0.1*192)`). 009 run artifacts were not rewritten.

## Result vs pre-fixed cases

Train/dev KL mean 0.1048 / 0.1239 vs 009 MLP 0.2854 / 0.2935 vs transition-only 0.1202 / 0.1338 vs independent 0.0168 / 0.0176. All seeds beat r on train and dev; none approach independent. Train and dev both improved. First-192 batches match 009.

## Stop

No further run. No Qwen extension.
