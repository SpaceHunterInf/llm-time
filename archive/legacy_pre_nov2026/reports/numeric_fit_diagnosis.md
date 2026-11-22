# Numeric fitting diagnosis (task 010)

15 September 2026. CPU only. No Qwen/LoRA optimization, no GPU, no paid APIs, no holdout scoring, no architecture or loss change. 008/009 data and the nine-run GPU pilot were not modified.

Question: can the existing 8,272-parameter MLP fit train/dev substantially better with one longer fixed schedule?

## Protocol

Same encoder and MLP (47→64→64→16), ordinary teacher→student joint KL, AdamW, float32, batch 32 / micro 8, LR 1e-3, clip 1, wd 0.01 non-bias. Seeds **0/1/2**, fresh init. **30 epochs = 1,920 updates/seed**. Warmup **19 updates** (not 10% of 1,920), then linear decay over 1,920. First three epoch permutations match 009 (verified 192/192 batches per seed). Select by epoch-end **dev KL only**. New dir `runs/numeric_fit_diagnosis/`; `runs/learning_pilot/` was read, not written.

Wall ceiling: 10 minutes combined, 4 CPU threads. Actual fit **74.8 s**, RSS **432 MB**, torch `2.14.0+cpu`. All 5,760 research updates completed. Incomplete=false.

## Joint KL (selected `best.pt`, all seeds)

Mean of three seeds (min–max). Teacher joint KL.

| Predictor | train | dev |
|---|---:|---:|
| 010 MLP | **0.1048** (0.0974–0.1098) | **0.1239** (0.1168–0.1297) |
| 009 MLP (CPU replay of GPU `best`) | 0.2854 (0.2782–0.2940) | 0.2935 (0.2851–0.3038) |
| stale | 0.1578 | 0.1811 |
| transition-only r | 0.1202 | 0.1338 |
| independent_sameinfo | **0.0168** | **0.0176** |
| exact | 0 | 0 |

Per seed (010):

| Seed | train KL | dev KL | train MAE | dev MAE | train regret | dev regret | best step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0974 | 0.1168 | 0.0448 | 0.0492 | 0.00302 | 0.00364 | 1920 |
| 1 | 0.1070 | 0.1252 | 0.0486 | 0.0529 | 0.00356 | 0.00416 | 1920 |
| 2 | 0.1098 | 0.1297 | 0.0491 | 0.0548 | 0.00361 | 0.00404 | 1920 |

Every seed beats transition-only on **both** train and dev. Every seed is still far from independent (~+0.08 to +0.11 KL). Every `best` is the last step; epoch-end dev KL fell through epoch 30.

## Roles and hazard (010 mean of 3)

| Split | related MAE | unrelated MAE | observed MAE | h=0 KL | h=.03 KL |
|---|---:|---:|---:|---:|---:|
| train | 0.0357 | 0.0359 | 0.0597 | 0.0535 | 0.1218 |
| dev | 0.0294 | 0.0374 | 0.0671 | 0.0708 | 0.1417 |

Independent unrelated MAE is ~0; transition-only unrelated MAE is ~0. 010 unrelated MAE ~0.036 is better than 009 MLP (~0.064) and not a successful independent filter.

## Paired per-example differences

010 minus reference, mean over examples (negative = 010 better). All seeds kept.

| Seed | split | vs r | vs stale | vs independent | vs 009 MLP |
|---:|---|---:|---:|---:|---:|
| 0 | train | −0.0228 (1033/2048 better) | −0.0604 | +0.0806 (83 better) | −0.1966 (1931 better) |
| 0 | dev | −0.0169 (123/256) | −0.0642 | +0.0993 (10) | −0.1870 (232) |
| 1 | train | −0.0132 (948) | −0.0507 | +0.0903 (75) | −0.1769 (1887) |
| 1 | dev | −0.0085 (121) | −0.0558 | +0.1077 (14) | −0.1663 (227) |
| 2 | train | −0.0104 (953) | −0.0479 | +0.0931 (53) | −0.1684 (1879) |
| 2 | dev | −0.0040 (118) | −0.0513 | +0.1122 (9) | −0.1553 (228) |

Family-bootstrap intervals are exploratory and per training seed, in `eval_{split}.json`.

## Curves

Seed 0 (others similar): step-1 LR 1.05e-4 and train KL 2.1433 match 009, as required by the shared first-epoch permutation. Step 19 LR = 1e-3 (warmup finished). Step 192 LR = 9.09e-4 (009 was already 0). Step 1920 LR = 0. Train first-epoch mean KL ~1.24 → last-epoch ~0.098; dev 0.419 → 0.117. Train and dev improved together.

## Identity / recovery (needed for this run)

- Code hashes include `learning_pilot_train.py` and `numeric_fit_diagnosis.yaml`.
- Resume rejects changed LR, batch, and code hashes; non-resume start in an existing dir is refused.
- Pending epoch-end selection runs before the next batch, including resume already at the step limit. Empty batches raise. Metrics are reconciled to the committed checkpoint.
- Eval saves keyed 16-state `pred_lex` joints; recomputing KL from those files matches the summary.
- Device/backend is provenance, not a silent identity match. 009 MLP comparison used tensor-only load of the historical GPU checkpoints.

## Against the table fixed before results

- **Better train and development fit.** The longer schedule/budget helped this numeric control (all seeds, train and dev). This does not prove Qwen would benefit equally and does not establish structural transfer. Holdouts were not re-scored.
- **Not overfitting.** Dev fell through the last epoch with train.
- **Not “still worse than transition-only.”** All three seeds beat r on train and on dev. Do **not** apply the “end this recipe” branch from that clause.
- **Does not approach independent/exact.** Remaining gap to independent is large. Not LLM novelty.

No Qwen extension, new loss, larger MLP, extra seeds, or holdout pass was started.

## Commands

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q   # 183 passed
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync \
  python scripts/numeric_fit_diagnosis.py --config llm_rl_engine/configs/numeric_fit_diagnosis.yaml \
  --out-root runs/numeric_fit_diagnosis run
# fit finished; eval needed a one-line interop-threads guard, then:
python scripts/numeric_fit_diagnosis.py ... eval
python scripts/numeric_fit_diagnosis.py ... analyze
```

Spend **$0**. Research updates **5,760**. 010 MLP unit-test updates **12** (tiny fixtures). Warmup unit test: 192 AdamW steps on a 1-parameter dummy, not the research MLP.

## Correction to the 009 update count (historical reports unchanged)

009 `verify-path` executes a 2-step run, a separate 1-step run, then one resumed step: **2+1+1**, then a probe. Verification-plus-probe is **five** executed updates per control, not four. `reports/learning_pilot_release.md` and `optimizer_counts.json` are left as written.

## Stop

One CPU diagnosis, complete. Commander/user review. No automatic next experiment.
