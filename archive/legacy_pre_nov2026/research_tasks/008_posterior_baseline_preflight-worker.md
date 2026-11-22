# Task 008 return

Author: Grok. No paid APIs. No full training. 007 pair fixtures unused. Commander plans not edited.

## Tests

`UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q` → **164 passed**.

Includes independent product-kernel vs matrix, lex/legacy conversion, family-id renaming, renderer leakage, metadata-invariant text, semantic permutation, generated-file hashes.

## Commands actually run

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python scripts/posterior_baseline_data.py
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv lock
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv sync --group train
HF_HOME=models/hf uv run --group train python -c 'snapshot_download(...)'
HF_HOME=models/hf UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --group train \
  python scripts/posterior_baseline_preflight.py
```

Spend **$0**. Dataset gen 31.5 s. Preflight ~107 s on CPU. Model snapshot 1148 MB (gitignored).

## Files

- `llm_rl_engine/analysis/posterior_baseline.py`
- `llm_rl_engine/analysis/posterior_baseline_model.py`
- `scripts/posterior_baseline_{data,preflight,train}.py`
- `llm_rl_engine/configs/posterior_baseline_pilot.yaml`
- `reports/posterior_baseline_preflight.md`
- `reports/posterior_baseline_preflight/{data,verification.json}`
- uv group `train` (CPU torch 2.14.0+cpu, transformers 5.17.0, peft 0.20.0)

## Verification

Qwen3-0.6B-Base `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, Apache-2.0, no chat template. Device: CPU (driver sees RTX 8000; this torch build cannot use it). Head and LoRA: frozen Δ=0, resume Δ=0, 2 unique steps each (3 executed). Token lengths 474–531. Forward KL 0.760 on 4 train rows.

## External package

192 steps × 6 runs specified, not launched. Probe/fit commands are in the report. GPU probe must be measured on CUDA torch before quoting time.

## Limitations / stop

Safe-optimal rare. RSS 22 GB during multi-copy resume. No holdout network eval. No new method. Stopped for commander/user review before any serious training job.
