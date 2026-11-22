# Task 006 return

Author: Grok. v1 and v2 sequential runs were not rewritten. No training. No extra grid after this report.

## Tests

`UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q` → **141 passed**.

Includes the task-006 CPU checks: own-forecast utilities under the emitted linked probability (not reference utilities); v2 offline own-error audit **2/10/7, 19 overall, 18 of 31 reference errors** without rewriting v2 rows; boolean/string joint-mass rejection; invalid first attempt still consumes only its remaining repair; valid repair request hash checked against its own text, not the base prompt; changed previous-state identity rejection; n20→n32 continuation matches reconstruction to **1e-12** on all six handoff tapes; 48 jobs, 24 unique byte-identical repeat prompts.

## CPU / protocol

- Own-forecast: `U(linked)=.2+.6p`, `U(safe)=.65`, `U(diagnose)=.61+.15p`. Reference regret kept as a separate field.
- Schema: reject bool/str/null/nonfinite/out-of-range/missing/extra keys and `|sum-1|>1e-4` as a schema error. Retain 1e-4 sum tolerance.
- Identity hashes tape fingerprints, previous-state, harness/llm/token_cost/pilot, and sequential modules. `effective_hash` `0dc35143ef5a69e8a1e246d5ad8118e370eb89734ed2c6da234253a889f02929`.
- Interfaces R/F/S/C; seeds 3/4/12/13; checkpoint n=32; previous state n=20; shuffle seed 6006; two byte-identical repeats.

## Live

- Dir: `runs/selective_revision_state_handoff_v1` (gitignored). Dry-run: `reports/selective_revision_state_handoff_v1_dryrun/`.
- Smoke is the first three of the same 48 (shared R r0, independent S r0, stable C r0); not paid twice.
- 48 jobs, 55 STOP attempts, **$3.7327365**, 43 valid, 5 parse failures, 17 valid reference-optimal errors.
- Pricing rechecked 11 September 2026: Gemini 3.5 Flash standard **$1.50 / $9.00** per 1M (output includes thinking). Unchanged from the 9 September entry. Planning reserve at 96 attempts × 4000 in / 10240 out: **$9.42336**. Cap $20.
- Report: `reports/selective_revision_state_handoff_v1_live.md`.

## Headlines (descriptive)

- **C copies and acts.** 12/12 valid, joint MSE ≤ 1.6e-33, 12/12 reference-optimal actions. Copying did not fail; C action errors are not a residual valuation problem.
- **F−R does not recur as a tape-level penalty.** Mean tape-averaged ITT F−R **+0.0016**, and only `independent_s3` is nonzero. R and F repeats agree on ITT loss. This is smaller than S within-input variability.
- **S is the hard interface.** Mean tape-averaged ITT S−R **+0.027**. Three of five parse failures are S. Repeat disagreement on `independent_s4` S and `shared_s4` S. Leave-one-tape-out: excluding `independent_s4` drops mean S−R to +0.007.
- **No safe-optimal cell** in this six-tape set (32 diagnose / 16 linked). Diagnoses hit **10/32** (all 8 C diagnose jobs plus `independent_s3` reconstruct). Linked-optimal stables: **16/16**.
- Offline joint-argmax (known utilities, emitted joint) cuts valid reference errors R 6→2, F 6→3, S 5→1. Remaining misses are cells whose own p already points at the wrong action.

Commander branch mapping (not a method claim): numbers match **(4)** F−R downgrade on these tapes, and **(3)** S forecast/schema error while C works. They do not match **(1)** (S does not approach R/C) or **(5)** (C does not fail). **(2)** holds on some diagnose cells and as an offline baseline, not uniformly.

## Evidence table

| This experiment establishes | It cannot establish | Simplest baseline a future method must beat |
|---|---|---|
| On these six development tapes, supplying the exact n=32 joint (C) yields a copy-accurate joint and the reference-optimal action. | A general claim that Gemini can use an arbitrary posterior, or that copying would survive a different schema. Two repeats, six tapes. | C itself, and deterministic `argmax_a U(a; p_emitted)`. |
| Reconstruct (R) already has small ITT loss (0.0097) and zero parse failures; F is nearly interchangeable with R here. | That v2 F-harm was a fluke in the population. New seeds, one checkpoint. | R (full history, no previous state). |
| Suffix-only continuation (S) is worse: higher loss, more parse failures, the only repeat disagreements. | Internal double-counting, or that S would remain hard after ordinary Bayesian teaching / shorter prompts. | S versus R with matched prompt length, and versus C. |
| Own-forecast and reference regret are separable; 11/17 valid reference errors are also own-suboptimal. | That remaining misses are a revision-scope learning problem. | Action arithmetic on the emitted linked probability. |

## Limitations / stop

Development diagnostic, not a holdout or powered evaluation. No safe-optimal coverage. Five parse failures use choose_safe ITT fallback. Thinking tokens often exceed the configured 8192 (measured max 9831; all finishes still STOP, billed output ≤ 9992 < 10240). Stopped here. Commander chooses whether to close the numerical diagnostic.
