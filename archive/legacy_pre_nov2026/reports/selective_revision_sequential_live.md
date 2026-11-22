# Sequential selective revision: live grid

10 September 2026. Frozen `gemini-3.5-flash`, temperature 0, thinking_budget 8192. Eight development tapes × 4 checkpoints × 3 conditions = 96 decisions. Descriptive, not a powered sample.

## Spend and integrity

| Run | Jobs | Valid JSON | Decision errors | Spent |
|---|---:|---:|---:|---:|
| Smoke (shared_s0 n=12, 3 conditions) | 3 | 3 | 0 | $0.456 |
| Full grid | 96 | 96 | **37** | **$8.819** |

101 attempts (5 parse retries). No API failures, no quota stop, no unfinished JSON. Cap was $20. Artifacts: `runs/selective_revision_sequential_v1/` (gitignored).

## Result

Mean ITT decision loss **0.021**. Stationary h=0 tapes: **0/24 errors**. Independent-change tapes: **29/36 errors**. Shared: **8/36**. Almost every error is missing `diagnose_linked` and taking `choose_linked` instead (36/37). One cell took safe instead of diagnose.

| Condition | n | Errors | Mean loss |
|---|---:|---:|---:|
| reconstruct | 32 | 12 | 0.017 |
| self_state | 32 | 14 | 0.032 |
| reference_state | 32 | 11 | 0.014 |

Paired **self − reconstruct** mean loss: **+0.015** (32 pairs). Independent tapes contribute 0 to that difference (same action errors in all three conditions). Self-only extra misses: `shared_s0` n=32, `shared_s1` n=12, `shared_s2` n=32 (the last of those has loss 0.38). Reconstruct-only miss: `shared_s0` n=44. Reference-state repairs reconstruct on `independent_s1` n=32 and `shared_s1` n=20, and adds `shared_s1` n=12.

When diagnose is optimal (42 cells): hit rate **6/42** overall, **1/30** on independent, **5/12** on shared, **2/14** reconstruct, **1/14** self-state, **3/14** reference-state. Mean `joint_mse`: reconstruct 0.016, self-state 0.027, reference-state 0.015. Self-state is worse on both actions and joint.

## What this is allowed to mean

- The one-event worksheet with thinking was a **ceiling**. This sequential grid is **not**.
- Carrying the model's own joint does **not** beat rebuilding from the same observations on these tapes. On independent change it is interchangeable; on shared it can be worse.
- Handing the exact previous posterior (reference-state) is a computational assist, not a memory-free baseline, and only modestly helps.
- The dominant miss is still **not buying a cheap diagnostic** when the linked bit is in the uncertain band — now under recurring hidden transitions, with thinking on.
- Eight tapes are the independent development units. Checkpoints and conditions are repeated measures. Do not treat 96 as n=96.

## What not to do next

Do not train on the one-event grid. Do not claim a method. A targeted next probe, if any, is why diagnose is skipped on independent tapes even with the true previous posterior, or a new seed/role holdout — not SFT of the 8-state filter.
