# State-handoff localizer v1 (task 006)

11 September 2026. Frozen `gemini-3.5-flash`, temperature 0, thinking_budget 8192, visible output cap 2048, provider output bound 10240. Six new-seed tapes × four interfaces × two byte-identical repeats = **48** planned decisions. One checkpoint: after observation 32. Previous-state timestamp: after observation 20. **Not a learned intervention.** Preserve v1/v2 sequential runs; do not overwrite them.

Effective hash: `0dc35143ef5a69e8a1e246d5ad8118e370eb89734ed2c6da234253a889f02929`

Pricing rechecked 11 September 2026 from https://ai.google.dev/gemini-api/docs/pricing (Gemini 3.5 Flash, Standard): **$1.50 input / $9.00 output per 1M**, output includes thinking. Unchanged from the 9 September `token_cost.py` entry.

## Spend, schema, recovery

| | Jobs | Valid schema | Decision errors (valid) | Parse failures | Attempts | Spent |
|---|---:|---:|---:|---:|---:|---:|
| Smoke (shared R r0, independent S r0, stable C r0) | 3 | 2 | 1 | 1 | 4 | $0.292 |
| Full 48 (smoke included, not repaid) | 48 | 43 | 17 | 5 | 55 | **$3.733** |

All 55 provider finishes report `STOP` and `usage_status=available`. First-attempt schema success **41/48**. Seven jobs used the one permitted repair; **2** repairs became valid (`independent_s4` F r1, `independent_s4` S r1); **5** remained invalid after the repair (safe ITT fallback). No API/quota stop. Interrupted reservations: 0. Ledger attempts charged once: 55. Cumulative input 50,205; billed output (includes thinking) 406,381; thoughts_tokens 397,013. Recomputed cost at 1.50/9.00 = **$3.7327365**, matching the ledger.

Planning reserve (96 attempts × 4000 in / 10240 out): **$9.42336**. Operational cap $20 / 96 calls. Actual spend is under both.

Output: `runs/selective_revision_state_handoff_v1/` (gitignored). Dry-run fixtures: `reports/selective_revision_state_handoff_v1_dryrun/` (24 unique prompts, shuffle seed 6006, smoke-first order).

### Per-attempt finishes

- Finish reason: STOP × 55. No truncation finish.
- Thoughts tokens: min 1055 (C), median 9826, max 9831. C never exceeded 1972. R/F/S often report thoughts_tokens **> configured 8192**; billed output still ≤ 9992 < 10240.
- First/final validity: 41 jobs valid on attempt 0; 2 valid only after repair; 5 invalid on both attempts.

Parse-failure finals (schema, not API):

| Job | att0 sum | att1 sum | Typical failure |
|---|---:|---:|---|
| `independent_s3` S r0 and r1 | 0 | 1.0954 | all-zero then overshoot |
| `shared_s3` F r0 and r1 | 0 | 0 | all-zero both attempts |
| `shared_s4` S r1 | 0.999669 | 1.943679 | off by >1e-4, then duplicated masses |

Invalid rows are **not** counted as revision results. ITT loss uses `choose_safe`. `decision_error` is defined only on valid schema.

### Identity / recovery

Scientific identity includes tape fingerprints (not only IDs), previous-state hash, model, temperature, thinking budget, schema v2, and implementation hashes of `llm.py`, `selective_revision.py`, `selective_revision_harness.py`, `selective_revision_pilot.py`, `selective_revision_sequential.py`, `selective_revision_sequential_run.py`, `token_cost.py`. Operational `max_usd` / `max_calls` are not in the scientific hash.

Resume used the same directory after smoke: completed jobs were not regenerated. Repair hashes are classified against the saved repair request text; they are not compared to the base prompt hash. The five exhausted jobs stopped after the remaining allowance.

Offline v2 own-forecast audit on the retained v2 run (rows not rewritten): reconstruct **2**, self_state **10**, reference_state **7**, overall **19**, **18** of 31 reference errors.

## Denominators

No **safe-optimal** cell in this six-tape set. Commander selected the seeds before inspecting outcomes; this limitation is preserved, not searched away.

| Tape | Structure | n=32 optimal | Jobs |
|---|---|---|---:|
| shared_s3 | shared | diagnose_linked | 8 |
| independent_s3 | independent | diagnose_linked | 8 |
| shared_s4 | shared | diagnose_linked | 8 |
| independent_s4 | independent | diagnose_linked | 8 |
| stable_s12 | stable h=0 | choose_linked | 8 |
| stable_s13 | stable h=0 | choose_linked | 8 |

Diagnose-optimal jobs: 32. Linked-optimal: 16. Safe-optimal: 0.

## Interface summary

ITT = all 12 jobs per interface, parse failures charged as `choose_safe`. Valid-only excludes those five rows.

| Interface | n | Valid | Ref-optimal errors (valid) | Own-forecast errors (valid) | Mean ITT loss | Mean valid loss | Mean joint MSE (valid) | Mean linked abs (valid) | Mean unrel abs (valid) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R reconstruct | 12 | 12 | 6 | 4 | 0.009652 | 0.009652 | 5.54e-4 | 0.0262 | 0.00217 |
| F full+prev | 12 | 10 | 6 | 3 | 0.011241 | 0.010733 | 3.09e-3 | 0.0489 | 1.94e-4 |
| S prev+suffix | 12 | 9 | 5 | 4 | 0.036406 | 0.036959 | 0.0124 | 0.0827 | 7.37e-7 |
| C current joint | 12 | 12 | **0** | 0 | **0** | 0 | ≤1.6e-33 | ≤1.1e-16 | ≤1.1e-16 |

Diagnose hits: **10/32** (ITT and valid coincide: all hits are valid). Breakdown: C 8/8 diagnose jobs; R 2/8 (`independent_s3` both repeats); F 0/8; S 0/8. Linked hits: **16/16**.

Of 17 valid reference-optimal errors: **11** are also own-forecast suboptimal (valuation-like under the emitted p); **6** are consistent with the model's own linked probability (inference-like). Do not label every missed diagnostic as VOI failure.

Unrelated-component absolute error stays small on valid rows except `independent_s4` R (0.013). The large S linked error is `independent_s4` S r1 (0.668).

## C: copy versus action

C is posterior assistance, not revealed simulator state. All 12 C jobs are schema-valid.

| Tape | r | joint MSE | linked abs | action | opt | copy ok | action ok |
|---|---:|---:|---:|---|---|:-:|:-:|
| shared_s3 | 0,1 | 0 | 0 | diagnose_linked | diagnose_linked | Y | Y |
| independent_s3 | 0,1 | 0 | 0 | diagnose_linked | diagnose_linked | Y | Y |
| shared_s4 | 0,1 | 1.6e-33 | 1.1e-16 | diagnose_linked | diagnose_linked | Y | Y |
| independent_s4 | 0,1 | 0 | 0 | diagnose_linked | diagnose_linked | Y | Y |
| stable_s12 | 0,1 | 0 | 0 | choose_linked | choose_linked | Y | Y |
| stable_s13 | 0,1 | 1.5e-33 | 1.1e-16 | choose_linked | choose_linked | Y | Y |

Copying succeeded. There is no C action error to attribute to a failed copy. Conclusion (5) does not apply.

## Tape-level effects

Average the two repeats first, then difference. ITT (parse-fail = choose_safe). Do not treat 48 jobs as 48 independent environments.

| Tape | R | F | S | C | F−R | S−F | S−R | C−R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_s3 | 0.013779 | 0.013779 | 0.013779 | 0 | 0 | 0 | 0 | −0.013779 |
| independent_s3 | 0 | 0.009531 | 0.009531 | 0 | 0.009531 | 0 | 0.009531 | 0 |
| shared_s4 | 0.034445 | 0.034445 | 0.059815 | 0 | 0 | 0.025370 | 0.025370 | −0.034445 |
| independent_s4 | 0.009688 | 0.009688 | 0.135312 | 0 | 0 | 0.125624 | 0.125624 | −0.009688 |
| stable_s12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| stable_s13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **mean of 6 tapes** | 0.009652 | 0.011241 | 0.036406 | 0 | **+0.001589** | **+0.025166** | **+0.026754** | **−0.009652** |

Valid-only tape means are identical to ITT except: `shared_s3` F both invalid (F mean undefined); `independent_s3` S both invalid (S undefined); `shared_s4` S r1 invalid (S valid-only = 0.034445, so S−R = 0 on that tape). Tapes with all four interfaces defined valid-only: shared_s4, independent_s4, stable_s12, stable_s13. Their mean F−R is **0**; mean S−R **+0.0314** (still dominated by `independent_s4`).

### Leave-one-tape-out (ITT means)

| Excluded tape | F−R | S−F | S−R | C−R |
|---|---:|---:|---:|---:|
| shared_s3 | +0.0019 | +0.0302 | +0.0321 | −0.0088 |
| independent_s3 | **0** | +0.0302 | +0.0302 | −0.0116 |
| shared_s4 | +0.0019 | +0.0251 | +0.0270 | −0.0047 |
| independent_s4 | +0.0019 | +0.0051 | **+0.0070** | −0.0096 |
| stable_s12 | +0.0019 | +0.0302 | +0.0321 | −0.0116 |
| stable_s13 | +0.0019 | +0.0302 | +0.0321 | −0.0116 |

F−R is one-tape (`independent_s3`). S−R is outlier-heavy (`independent_s4`). Stables contribute zeros, not a hidden F-harm.

### Repeat disagreement

R, F, and C: both repeats agree on ITT loss (F's two invalid `shared_s3` rows agree with each other). S disagrees on two tapes; both are reported:

| Tape | r0 | r1 |
|---|---|---|
| `independent_s4` S | valid, choose_safe, loss 0.009688, joint MSE 1.7e-11, linked abs 8e-6 | valid, choose_linked, loss 0.260937, joint MSE 0.111, linked abs 0.668 |
| `shared_s4` S | valid, choose_linked, loss 0.034445, joint MSE 1.7e-4 | parse_fail, choose_safe fallback, loss 0.085185 |

Within-input |r0−r1| mean ITT loss: R 0, F 0, C 0, **S 0.050**. Across-tape stdev of tape-means: R 0.012, F 0.012, S 0.049, C 0. Mean F−R (+0.0016) is smaller than S repeat variability. Do not cherry-pick the S repeat that looks like a clean continuation.

## Offline action from the emitted joint

Known utilities, model's parsed linked probability: replace the emitted action with `argmax_a U(a; p_emitted)`. This is not a hidden-state oracle.

| Interface | Valid n | Actual ref errors | Actual mean loss | Offline ref errors | Offline mean regret |
|---|---:|---:|---:|---:|---:|
| R | 12 | 6 | 0.009652 | **2** | 0.002297 |
| F | 10 | 6 | 0.010733 | **3** | 0.002875 |
| S | 9 | 5 | 0.036959 | **1** | 0.028993 |
| C | 12 | 0 | 0 | 0 | 0 |

The two remaining R offline errors are both `shared_s3` R (emitted p_linked = 0.24 vs reference 0.359; own-optimal is also `choose_safe`). The three remaining F offline errors are `independent_s3` F r0/r1 and `independent_s4` F r1. The one remaining S offline error is `independent_s4` S r1 (p_linked 0.999 vs 0.331). On several diagnose cells the joint is close enough that deterministic action choice would have hit `diagnose_linked` (`independent_s4` R both, `independent_s4` S r0, `shared_s4` R/F/S r0, `independent_s4` F r0).

## All 48 jobs

`ref_err` is valid-only. Invalid rows show ITT loss for `choose_safe` and `—` for joint metrics. `own` / `offline` are the own-forecast action (identical to joint-argmax under the known utilities).

| tape | I | r | valid | action | opt | own | offline | ref_err | own_err | loss | off_regret | mse | linked_abs | unrel_abs | att |
|---|:-:|:-:|:-:|---|---|---|---|:-:|:-:|---:|---:|---:|---:|---:|---:|
| shared_s3 | C | 0 | Y | diagnose_linked | diagnose_linked | diagnose_linked | diagnose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| shared_s3 | C | 1 | Y | diagnose_linked | diagnose_linked | diagnose_linked | diagnose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| shared_s3 | F | 0 | N | choose_safe | diagnose_linked | — | — | N | N | 0.013779 | — | — | — | — | 2 |
| shared_s3 | F | 1 | N | choose_safe | diagnose_linked | — | — | N | N | 0.013779 | — | — | — | — | 2 |
| shared_s3 | R | 0 | Y | choose_safe | diagnose_linked | choose_safe | choose_safe | Y | N | 0.013779 | 0.013779 | 0.002911 | 0.118529 | 0.000002 | 1 |
| shared_s3 | R | 1 | Y | choose_safe | diagnose_linked | choose_safe | choose_safe | Y | N | 0.013779 | 0.013779 | 0.002911 | 0.118529 | 0.000002 | 1 |
| shared_s3 | S | 0 | Y | choose_safe | diagnose_linked | diagnose_linked | diagnose_linked | Y | Y | 0.013779 | 0 | 0.000230 | 0.025179 | 9.46e-07 | 1 |
| shared_s3 | S | 1 | Y | choose_safe | diagnose_linked | diagnose_linked | diagnose_linked | Y | Y | 0.013779 | 0 | 0.000230 | 0.025179 | 9.46e-07 | 1 |
| independent_s3 | C | 0 | Y | diagnose_linked | diagnose_linked | diagnose_linked | diagnose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| independent_s3 | C | 1 | Y | diagnose_linked | diagnose_linked | diagnose_linked | diagnose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| independent_s3 | F | 0 | Y | choose_safe | diagnose_linked | choose_safe | choose_safe | Y | N | 0.009531 | 0.009531 | 0.007054 | 0.176210 | 0.000002 | 1 |
| independent_s3 | F | 1 | Y | choose_safe | diagnose_linked | choose_safe | choose_safe | Y | N | 0.009531 | 0.009531 | 0.020463 | 0.080209 | 0.000002 | 1 |
| independent_s3 | R | 0 | Y | diagnose_linked | diagnose_linked | diagnose_linked | diagnose_linked | N | N | 0 | 0 | 1.08e-12 | 9.00e-07 | 0.000002 | 1 |
| independent_s3 | R | 1 | Y | diagnose_linked | diagnose_linked | diagnose_linked | diagnose_linked | N | N | 0 | 0 | 3.10e-09 | 0.000110 | 0.000002 | 1 |
| independent_s3 | S | 0 | N | choose_safe | diagnose_linked | — | — | N | N | 0.009531 | — | — | — | — | 2 |
| independent_s3 | S | 1 | N | choose_safe | diagnose_linked | — | — | N | N | 0.009531 | — | — | — | — | 2 |
| shared_s4 | C | 0 | Y | diagnose_linked | diagnose_linked | diagnose_linked | diagnose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| shared_s4 | C | 1 | Y | diagnose_linked | diagnose_linked | diagnose_linked | diagnose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| shared_s4 | F | 0 | Y | choose_linked | diagnose_linked | diagnose_linked | diagnose_linked | Y | Y | 0.034445 | 0 | 0.000753 | 0.055223 | 0.000963 | 1 |
| shared_s4 | F | 1 | Y | choose_linked | diagnose_linked | diagnose_linked | diagnose_linked | Y | Y | 0.034445 | 0 | 0.000753 | 0.055223 | 0.000963 | 1 |
| shared_s4 | R | 0 | Y | choose_linked | diagnose_linked | diagnose_linked | diagnose_linked | Y | Y | 0.034445 | 0 | 0.000372 | 0.038566 | 0.000039 | 1 |
| shared_s4 | R | 1 | Y | choose_linked | diagnose_linked | diagnose_linked | diagnose_linked | Y | Y | 0.034445 | 0 | 0.000372 | 0.038566 | 0.000039 | 1 |
| shared_s4 | S | 0 | Y | choose_linked | diagnose_linked | diagnose_linked | diagnose_linked | Y | Y | 0.034445 | 0 | 0.000174 | 0.026230 | 4.51e-09 | 1 |
| shared_s4 | S | 1 | N | choose_safe | diagnose_linked | — | — | N | N | 0.085185 | — | — | — | — | 2 |
| independent_s4 | C | 0 | Y | diagnose_linked | diagnose_linked | diagnose_linked | diagnose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| independent_s4 | C | 1 | Y | diagnose_linked | diagnose_linked | diagnose_linked | diagnose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| independent_s4 | F | 0 | Y | choose_safe | diagnose_linked | diagnose_linked | diagnose_linked | Y | Y | 0.009688 | 0 | 0.000738 | 0.056250 | 0.000007 | 1 |
| independent_s4 | F | 1 | Y | choose_safe | diagnose_linked | choose_safe | choose_safe | Y | N | 0.009688 | 0.009688 | 0.001121 | 0.066252 | 0.000003 | 2 |
| independent_s4 | R | 0 | Y | choose_safe | diagnose_linked | diagnose_linked | diagnose_linked | Y | Y | 0.009688 | 0 | 0.000042 | 0.000006 | 0.012987 | 1 |
| independent_s4 | R | 1 | Y | choose_safe | diagnose_linked | diagnose_linked | diagnose_linked | Y | Y | 0.009688 | 0 | 0.000042 | 0.000006 | 0.012987 | 1 |
| independent_s4 | S | 0 | Y | choose_safe | diagnose_linked | diagnose_linked | diagnose_linked | Y | Y | 0.009688 | 0 | 1.73e-11 | 0.000008 | 3.35e-10 | 1 |
| independent_s4 | S | 1 | Y | choose_linked | diagnose_linked | choose_linked | choose_linked | Y | N | 0.260937 | 0.260937 | 0.111194 | 0.667707 | 0.000005 | 2 |
| stable_s12 | C | 0 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| stable_s12 | C | 1 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| stable_s12 | F | 0 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 1.95e-12 | 1.53e-12 | 1 |
| stable_s12 | F | 1 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 1.95e-12 | 1.53e-12 | 1 |
| stable_s12 | R | 0 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 0 | 8.52e-08 | 1 |
| stable_s12 | R | 1 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 0 | 8.52e-08 | 1 |
| stable_s12 | S | 0 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 9.80e-10 | 8.25e-10 | 1 |
| stable_s12 | S | 1 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 5.19e-11 | 2.16e-11 | 1 |
| stable_s13 | C | 0 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| stable_s13 | C | 1 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| stable_s13 | F | 0 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| stable_s13 | F | 1 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| stable_s13 | R | 0 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 3.39e-08 | 3.39e-08 | 1 |
| stable_s13 | R | 1 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 6.08e-11 | 6.08e-11 | 1 |
| stable_s13 | S | 0 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |
| stable_s13 | S | 1 | Y | choose_linked | choose_linked | choose_linked | choose_linked | N | N | 0 | 0 | 0 | 0 | 0 | 1 |

Reference linked probabilities at n=32 (from the exact filter; C copies them): shared_s3 0.3585, independent_s3 0.3302, shared_s4 0.8346, independent_s4 0.3313, stable_s12 0.9900, stable_s13 1.0000.

## Conditional conclusions (task-006 list)

These are mappings onto the brief's five branches, not a method claim and not a reason to train.

1. **F underperforms R but S approaches R/C.** Not this run. S is worse than R (tape-mean S−R +0.027) and does not approach C. F−R is +0.0016.
2. **R/S have accurate forecasts but action errors; C or deterministic choice removes them.** Partial. C removes all action errors. Offline joint-argmax cuts R 6→2 and F 6→3 valid errors. Forecasts are not uniformly accurate (`shared_s3` R linked abs 0.119; `independent_s3` F 0.08–0.18; `independent_s4` S r1 0.668).
3. **S retains material forecast/schema error while C works.** Matches. S has 3/5 parse failures, mean valid joint MSE 0.012, and the only large linked miss. Recursive updating remains a candidate. It still needs a precise comparison against ordinary Bayesian teaching and against prompt-length controls before any training argument.
4. **F−R does not recur, or is smaller than repeat variability.** Matches on these tapes. Mean F−R +0.0016, one tape, R/F repeats agree, smaller than S within-input variability. Downgrade the v2 “redundant state harms reconstruction” hypothesis for this seed/checkpoint slice. Do not treat that as a population reversal of v2.
5. **C itself fails.** Does not match. Copy MSE is numerical zero and actions are reference-optimal.

## Reproducible offline commands

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python scripts/selective_revision_handoff.py \
  --output-dir reports/selective_revision_state_handoff_v1_dryrun
```

Live (already completed; resume, do not overwrite):

```bash
UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python scripts/selective_revision_handoff.py \
  --live --smoke --max-usd 20 --max-calls 96
# then the same without --smoke to finish the remaining 45 of the same 48
```

v2 own-forecast audit (does not rewrite v2 rows): `audit_v2_own_forecast("runs/selective_revision_sequential_v2")` → 2/10/7, 19, 18/31.

## Stop

No training. No further paid grid from this worker. Six tapes and two repeats are a diagnostic localizer, not a memory evaluation. Commander decides whether to close the numerical branch or write a structural-transfer/training comparison. The cheapest baselines still in front of any learner are **C**, **R**, and **argmax U(a; p_emitted)**.
