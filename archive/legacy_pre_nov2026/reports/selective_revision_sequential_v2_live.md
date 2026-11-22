# Sequential selective revision v2 (protocol correction)

11 September 2026. Frozen `gemini-3.5-flash`, temperature 0, thinking_budget 8192. Same eight development tapes and four checkpoints as v1. **v1→v2 is a protocol/schema correction, not a learned intervention.** Preserve v1 runs; do not overwrite them.

Effective hash: `a85781f4e3542a56c0279ff456ef433041ac5b9cf7a8ef328d2cb2f66235375d`

## Spend and schema

| | Jobs | Valid schema | Decision errors (valid) | Parse failures | Attempts | Spent |
|---|---:|---:|---:|---:|---:|---:|
| Smoke (shared/independent/stable reconstruct n=12) | 3 | 3 | 0 | 0 | 3 | $0.216 |
| Full v2 grid (smoke included) | 96 | 95 | 31 | 1 | 104 | **$8.511** |

All 104 attempted provider finishes report `STOP`. First-attempt schema success 88/96; 8 jobs used a retry; one remained invalid (`shared_s2` n=32 self_state) and used the predeclared safe fallback. v1 offline audit on retained finals: **15** raw joint sums off by >1e-4 and **13** joint/marginal disagreements >.001 (unchanged; not rewritten).

Cap $20. Output: `runs/selective_revision_sequential_v2/` (gitignored).

## Results under the corrected protocol

| Condition | n | Valid | Reference-optimal errors | Mean ITT loss | Own-forecast errors |
|---|---:|---:|---:|---:|---:|
| reconstruct | 32 | 32 | **4** | **0.0043** | 2 |
| self_state | 32 | 31 | 14 | 0.0389 | 8 |
| reference_state | 32 | 32 | 13 | 0.0268 | 7 |

| Structure | n | Errors | Mean loss |
|---|---:|---:|---:|
| shared | 36 | 11 | 0.036 |
| independent | 36 | 20 | 0.026 |
| stable | 24 | **0** | 0 |

Diagnose-optimal cells (42): hit **17/42** overall; reconstruct **11/14**; self **3/14**; reference-state **3/14**; shared 7/12; independent **10/30**.

Of 31 valid reference-optimal errors, **17** are also suboptimal under the model's own linked probability (valuation-like). The other 14 are consistent with its own forecast (inference-like). Do not label every missed diagnostic as VOI failure.

Unrelated-component absolute error vs the exact conditional reference is small (reconstruct mean ~7e-6, self max 0.0094). That is not the v1 “joint vs reported marginal ~1” artifact.

n=12 self-state and reference-state still receive the same initial prior. Action agreed on 6/8 tapes and disagreed on `shared_s1` and `shared_s2`. Temperature 0 does not make those repeats identical.

Self-minus-reconstruct mean loss **+0.035** (32 pairs, ITT on the parse-failed cell). Exclude `shared_s2`: **+0.015** (28 pairs). Largest single cell: `shared_s2` n=44, +0.370. This remains development-tape evidence dominated by outliers, not a general memory penalty.

## v1 comparison (not an intervention effect)

Independent prompts in v1 omitted the 0.5 redraw law; v2 states it in every structure. Reconstruct independent errors fell (v1 29/36 independent → v2 20/36 overall independent; reconstruct-only independent diagnose hits rose). Schema is keyed and rejects bad sums. Those changes move reconstruct a lot. Self-state did not similarly recover.

## Stop

No training. No further grid from this worker. Commander branch: reconstruct is close to solved with thinking; self-state is not, but the gap is outlier-heavy and mixed inference/valuation. Next is Astra's call (new seeds / controlled state replacement / current-posterior localizer), not another automatic 96.
