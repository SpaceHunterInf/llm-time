# Ordinary posterior-KL baseline preflight (task 008)

11 September 2026. No paid APIs. No full training run. The paired loss is not used. The 007 128-pair fixtures were not trained on.

This is a **data + path verification** for ordinary joint-KL distillation, not a learning result and not a new method.

## Data

New sampled dataset `posterior_baseline_v1`, seed **808**. Observation components are uniform and independent of the hidden state; outcomes are Bernoulli from the latent rate after group resets. Prefix length 4–8, suffix 1–4. 25% h=0 in every split.

| Split | n | Families | Unique inputs | Topologies | h=0 / h=.03 |
|---|---:|---:|---:|---|---|
| train | 2048 | 2030 | 2048 | 1024 four_singletons, 1024 one_pair | 512 / 1536 |
| dev | 256 | 256 | 256 | 128 / 128 | 64 / 192 |
| same_topology holdout | 256 | 256 | 256 | 128 / 128 | 64 / 192 |
| structural holdout | 256 | 256 | 256 | 128 two_pairs, 128 triple | 64 / 192 |

Family-split collisions: **0**. Family identity is the ordered prefix/suffix plus boundary, canonicalized by first-appearance renaming; hazard and topology are ignored. Same evidence under different partitions is a different model input (partition is in the prompt).

Generation: 2838 attempts, 31.5 s. Files under `reports/posterior_baseline_preflight/data/`. Model-facing jsonl has **no** prefix, `q_current`, family/split/topology labels, or pair IDs.

Train action denominators under existing utilities (query = first suffix component). **Not searched:**

| Action | Train | Dev | Same-topo | Structural |
|---|---:|---:|---:|---:|
| choose_linked | 1545 | 178 | 187 | 195 |
| diagnose_linked | 467 | 71 | 61 | 59 |
| choose_safe | **36** | 7 | 8 | **2** |

Safe-optimal is rare. Related-unobserved present in 556/2048 train and 199/256 structural. Unrelated present in 1890/2048 train.

File hashes:

| File | sha256 |
|---|---|
| train_model.jsonl | `12fea13909664efe0110eb8b646d80e4555ede0ca7d7a083098236259e6cd25f` |
| train_audit.jsonl | `8d8dd5763c5f1ae50923c0ba339a260e53906c523ebb14dc6306929397d7a734` |
| dev_model.jsonl | `d1e9e67a8c724dbc2b1997acbba27439200535bd6f5db6d87fc5f77460fff432` |
| same_topology_model.jsonl | `0893eeefca3b0fba3f4f0b047c8a3d2fa53a8e8438a5c27c85ad211ba7dbcdc1` |
| structural_model.jsonl | `8b85d11deeb3c4e199b4521e852bc593c5bdc826f598da1a26441e985b1277e1` |

Independent product-of-group-kernels (no shared redraw helper) matches `transition_matrix` to 1e-12 for all topology types and h in {0, .03, .4, 1}. Prefix+suffix replay is in the example builder. Lexicographic head order is `0000..1111` in displayed name order; conversion to the legacy integer bit order is tested.

Non-neural mean teacher KL on **train only** (not a holdout model eval): exact ~0, independent 0.0204, transition-only 0.120, stale 0.158.

## Model path

Pinned **Qwen/Qwen3-0.6B-Base** revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Apache-2.0. Snapshot size **1148 MB** (`model.safetensors` 1192_135_096 bytes). Tokenizer has a chat template in `tokenizer_config.json`; **it was not used**. Plain `tokenizer(text, add_special_tokens=True)`.

Packages (uv group `train`, CPU torch index): torch 2.14.0+cpu, transformers 5.17.0, peft 0.20.0. Isolated from the default benchmark env via the optional group. No system CUDA toolkit, no FlashAttention.

Architecture: last non-padding token, linear 16-logit head, float32 log-softmax, teacher joint KL only. Head-only: 2 trainable tensors, 310 frozen. LoRA+head: r=8, α=16, dropout 0, `q_proj`/`v_proj`, 114 trainable tensors. These differ in capacity; they are not a new-method comparison.

## Verification (this machine)

`nvidia-smi` lists Quadro RTX 8000. This preflight used **CPU torch**, so `torch.cuda.is_available() == False`. Do not treat CPU step times as GPU estimates. Dtype: float32. CPU threads: 4. Peak RSS **22019 MB** (several 0.6B copies during resume). Token lengths on 64 train prompts: min 474, mean 500, max 531 (cap 1024; overflow fails rather than truncates).

Forward on 4 train examples: finite log-probs, row sums 1, KL 0.760. Mixed-length last-token vs unbatched: max abs **6.2e-6** (float32). Chat template not applied.

| Mode | Unique steps | Executed updates | Frozen Δ | Trainable Δ | Resume Δ | ckpt bytes |
|---|---:|---:|---:|---:|---:|---:|
| head | 2 | 3 | 0 | 0.00100 | **0** | 276_341 |
| lora | 2 | 3 | 0 | 0.00100 | **0** | 14_169_948 |

Step-loss numbers are optimizer-path checks, **not** research evidence. Max 4 steps/config respected. Scratch under `reports/posterior_baseline_preflight/scratch/` (gitignored). Hold-out splits were not scored with the network.

Raw: `reports/posterior_baseline_preflight/verification.json`.

## External job (prepared, not launched)

Config: `llm_rl_engine/configs/posterior_baseline_pilot.yaml`. 3 epochs, batch 32, AdamW, head LR 1e-3, LoRA LR 1e-4, clip 1.0, 10% linear warmup. **192 optimizer steps per run** (2048/32 × 3). Six runs: `{head,lora} × seeds {0,1,2}`. Select checkpoints by **dev KL only**. Do not score same-topology or structural until that selection is done.

GPU probe (external CUDA torch; CPU timings here are not a substitute):

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --group train \
  python scripts/posterior_baseline_train.py probe --mode head --device cuda --batch-size 32
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --group train \
  python scripts/posterior_baseline_train.py probe --mode lora --device cuda --batch-size 32
```

Install CUDA torch on the external machine instead of the CPU index used here, for example:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then measure examples/s, tokens/s, and peak allocator memory **before** quoting wall-clock. Include eval/checkpoint overhead. Head-only should be cheaper than LoRA (2 vs 114 trainable tensors; LoRA still runs the full backward through adapted layers).

Fit commands after commander/user review (not run in 008):

```bash
for seed in 0 1 2; do
  UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --group train \
    python scripts/posterior_baseline_train.py fit --mode head --seed $seed \
    --data-dir reports/posterior_baseline_preflight/data \
    --out-dir runs/posterior_baseline/head_s${seed} --device cuda --cache-dir models/hf
  UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --group train \
    python scripts/posterior_baseline_train.py fit --mode lora --seed $seed \
    --data-dir reports/posterior_baseline_preflight/data \
    --out-dir runs/posterior_baseline/lora_s${seed} --device cuda --cache-dir models/hf
done
```

A later result would mean: LoRA+KL transfers → ordinary distillation is a strong baseline, do not claim a new loss; fit without structural transfer → check coverage/representation/capacity before a new inductive bias; neither fits → debug optimization first. None of that is a novelty claim. Bayesian teaching and learned memory remain prior art.

## Limitations

- This machine’s preflight torch is CPU-only; the GPU is visible to the driver but unused.
- Peak RSS ~22 GB from multiple loaded copies; a single-copy external run should be much smaller.
- Mixed-length last-token match is 6e-6, not bit-identical.
- Safe-optimal coverage is thin (36 train, 2 structural).
- transformers 5.17 (Qwen3 requires ≥4.51).
- 16-state readout is this four-component task only.
- No MLP-on-numeric-state control yet.

## Stop

No six-run launch. No API batch. Commander reviews the script and a GPU probe with the user before serious training on another machine.
