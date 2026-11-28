# Untuned `Qwen/Qwen3-8B` Keep-or-Apply Reviewer Gatekeeper: `gate8b_s737`

- **Reviewer & Gatekeeper:** `Qwen/Qwen3-8B` (`$0.00` local RTX 8000 FP16) | **Fixed Actor:** `gemini-3.8-flash` | **Held-Out Third Setup Untouched:** `coalition_ops`
- **Fresh Streams:** `Seeds [737]` (`N = 1`) | **Preselected Reviews:** `[16, 28, 36, 44]` (`M = 4` opportunities) | **Eval Trials / Version:** `K = 6` | **Total Spend:** `$1.4499`

## 1. Gatekeeper Policy Comparison (Identical History, Actor, & 4-Session After-Fee Budget)

| Slice | Opportunities (`M`) | `8B` Apply Rate | `Always-Keep` DP (%) | `Always-Apply` DP (%) | **Untuned `8B` Policy DP (%)** | `Oracle` Ceiling (%) | `8B − Keep` (95% CI) | `8B − Apply` (95% CI) | `Oracle − 8B` Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All Preselected Opportunities | `4` | `100.0%` | 61.56% | 61.91% | **61.91%** | 61.91% | **`+0.35`** `[+0.01, +0.71]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Non-Identical Rule Proposals (`R_prop != R_cur`) | `4` | `100.0%` | 61.56% | 61.91% | **61.91%** | 61.91% | **`+0.35`** `[+0.01, +0.71]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Mode: `drift_dip_c_60` | `4` | `100.0%` | 61.56% | 61.91% | **61.91%** | 61.91% | **`+0.35`** `[+0.01, +0.71]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_16_to_20` | `1` | `100.0%` | 66.39% | 66.85% | **66.85%** | 66.85% | **`+0.46`** `[+0.46, +0.46]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_28_to_32` | `1` | `100.0%` | 46.38% | 47.33% | **47.33%** | 47.33% | **`+0.95`** `[+0.95, +0.95]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_36_to_40` | `1` | `100.0%` | 70.70% | 70.71% | **70.71%** | 70.71% | **`+0.01`** `[+0.01, +0.01]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_44_to_48` | `1` | `100.0%` | 62.76% | 62.76% | **62.76%** | 62.76% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |

## 2. Classification of Material Edits (`|V_apply − V_keep| >= 2.0 pts`)

- **Material Proposals:** `0` (`0` helpful `>= +2.0 pts`, `0` harmful `<= -2.0 pts`)
- **Untuned `8B` Gatekeeper Accuracy:** **`0.0%`** (`TP_apply_helpful = 0`, `TN_keep_harmful = 0`, `FP_apply_harmful = 0`, `FN_keep_helpful = 0`)

## 3. Per-Opportunity Pre-Outcome Choices & Next-Review (`t_use`) Scores

| Seed | Mode | `t_edit -> t_use` | `8B` Choice | `Keep` DP (`σ`) | `Apply` DP (`σ`) | `Apply − Keep` | `8B` Achieved | Distinct Rule? |
|---:|---|---|---:|---:|---:|---:|---:|:---:|
| `737` | `drift_dip_c_60` | `16 -> 20` | **`APPLY`** | 66.4% (`0.2`) | 66.8% (`1.1`) | **`+0.46`** | 66.8% | `True` |
| `737` | `drift_dip_c_60` | `28 -> 32` | **`APPLY`** | 46.4% (`1.2`) | 47.3% (`3.2`) | **`+0.95`** | 47.3% | `True` |
| `737` | `drift_dip_c_60` | `36 -> 40` | **`APPLY`** | 70.7% (`0.0`) | 70.7% (`0.0`) | **`+0.01`** | 70.7% | `True` |
| `737` | `drift_dip_c_60` | `44 -> 48` | **`APPLY`** | 62.8% (`0.0`) | 62.8% (`0.0`) | **`+0.00`** | 62.8% | `True` |

