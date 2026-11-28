# Untuned `Qwen/Qwen3-8B` Keep-or-Apply Reviewer Gatekeeper: `gate8b_s731`

- **Reviewer & Gatekeeper:** `Qwen/Qwen3-8B` (`$0.00` local RTX 8000 FP16) | **Fixed Actor:** `gemini-3.8-flash` | **Held-Out Third Setup Untouched:** `coalition_ops`
- **Fresh Streams:** `Seeds [731]` (`N = 1`) | **Preselected Reviews:** `[16, 28, 36, 44]` (`M = 4` opportunities) | **Eval Trials / Version:** `K = 6` | **Total Spend:** `$1.4142`

## 1. Gatekeeper Policy Comparison (Identical History, Actor, & 4-Session After-Fee Budget)

| Slice | Opportunities (`M`) | `8B` Apply Rate | `Always-Keep` DP (%) | `Always-Apply` DP (%) | **Untuned `8B` Policy DP (%)** | `Oracle` Ceiling (%) | `8B − Keep` (95% CI) | `8B − Apply` (95% CI) | `Oracle − 8B` Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All Preselected Opportunities | `4` | `75.0%` | 49.33% | 46.98% | **46.98%** | 49.33% | **`-2.35`** `[-5.11, -0.04]` | **`+0.00`** `[+0.00, +0.00]` | `+2.35 pts` |
| Non-Identical Rule Proposals (`R_prop != R_cur`) | `3` | `100.0%` | 49.57% | 46.45% | **46.45%** | 49.57% | **`-3.13`** `[-6.79, -0.07]` | **`+0.00`** `[+0.00, +0.00]` | `+3.13 pts` |
| Mode: `drift_meta_60` | `4` | `75.0%` | 49.33% | 46.98% | **46.98%** | 49.33% | **`-2.35`** `[-5.11, -0.04]` | **`+0.00`** `[+0.00, +0.00]` | `+2.35 pts` |
| Step: `ep_16_to_20` | `1` | `100.0%` | 64.12% | 61.60% | **61.60%** | 64.12% | **`-2.52`** `[-2.52, -2.52]` | **`+0.00`** `[+0.00, +0.00]` | `+2.52 pts` |
| Step: `ep_28_to_32` | `1` | `0.0%` | 48.59% | 48.59% | **48.59%** | 48.59% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_36_to_40` | `1` | `100.0%` | 48.83% | 42.04% | **42.04%** | 48.83% | **`-6.79`** `[-6.79, -6.79]` | **`+0.00`** `[+0.00, +0.00]` | `+6.79 pts` |
| Step: `ep_44_to_48` | `1` | `100.0%` | 35.77% | 35.70% | **35.70%** | 35.77% | **`-0.07`** `[-0.07, -0.07]` | **`+0.00`** `[+0.00, +0.00]` | `+0.07 pts` |

## 2. Classification of Material Edits (`|V_apply − V_keep| >= 2.0 pts`)

- **Material Proposals:** `2` (`0` helpful `>= +2.0 pts`, `2` harmful `<= -2.0 pts`)
- **Untuned `8B` Gatekeeper Accuracy:** **`0.0%`** (`TP_apply_helpful = 0`, `TN_keep_harmful = 0`, `FP_apply_harmful = 2`, `FN_keep_helpful = 0`)

## 3. Per-Opportunity Pre-Outcome Choices & Next-Review (`t_use`) Scores

| Seed | Mode | `t_edit -> t_use` | `8B` Choice | `Keep` DP (`σ`) | `Apply` DP (`σ`) | `Apply − Keep` | `8B` Achieved | Distinct Rule? |
|---:|---|---|---:|---:|---:|---:|---:|:---:|
| `731` | `drift_meta_60` | `16 -> 20` | **`APPLY`** | 64.1% (`4.8`) | 61.6% (`4.8`) | **`-2.52`** | 61.6% | `True` |
| `731` | `drift_meta_60` | `28 -> 32` | **`KEEP`** | 48.6% (`6.9`) | 48.6% (`6.9`) | **`+0.00`** | 48.6% | `False` |
| `731` | `drift_meta_60` | `36 -> 40` | **`APPLY`** | 48.8% (`11.3`) | 42.0% (`0.0`) | **`-6.79`** | 42.0% | `True` |
| `731` | `drift_meta_60` | `44 -> 48` | **`APPLY`** | 35.8% (`3.1`) | 35.7% (`1.9`) | **`-0.07`** | 35.7% | `True` |

