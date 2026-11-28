# Untuned `Qwen/Qwen3-8B` Keep-or-Apply Reviewer Gatekeeper: `gate8b_s738`

- **Reviewer & Gatekeeper:** `Qwen/Qwen3-8B` (`$0.00` local RTX 8000 FP16) | **Fixed Actor:** `gemini-3.8-flash` | **Held-Out Third Setup Untouched:** `coalition_ops`
- **Fresh Streams:** `Seeds [738]` (`N = 1`) | **Preselected Reviews:** `[16, 28, 36, 44]` (`M = 4` opportunities) | **Eval Trials / Version:** `K = 6` | **Total Spend:** `$1.3197`

## 1. Gatekeeper Policy Comparison (Identical History, Actor, & 4-Session After-Fee Budget)

| Slice | Opportunities (`M`) | `8B` Apply Rate | `Always-Keep` DP (%) | `Always-Apply` DP (%) | **Untuned `8B` Policy DP (%)** | `Oracle` Ceiling (%) | `8B − Keep` (95% CI) | `8B − Apply` (95% CI) | `Oracle − 8B` Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All Preselected Opportunities | `4` | `75.0%` | 50.55% | 50.66% | **50.66%** | 50.87% | **`+0.12`** `[-0.63, +0.98]` | **`+0.00`** `[+0.00, +0.00]` | `+0.21 pts` |
| Non-Identical Rule Proposals (`R_prop != R_cur`) | `4` | `75.0%` | 50.55% | 50.66% | **50.66%** | 50.87% | **`+0.12`** `[-0.63, +0.98]` | **`+0.00`** `[+0.00, +0.00]` | `+0.21 pts` |
| Mode: `drift_dip_c_60` | `4` | `75.0%` | 50.55% | 50.66% | **50.66%** | 50.87% | **`+0.12`** `[-0.63, +0.98]` | **`+0.00`** `[+0.00, +0.00]` | `+0.21 pts` |
| Step: `ep_16_to_20` | `1` | `100.0%` | 66.62% | 66.62% | **66.62%** | 66.62% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_28_to_32` | `1` | `100.0%` | 35.44% | 36.74% | **36.74%** | 36.74% | **`+1.30`** `[+1.30, +1.30]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_36_to_40` | `1` | `0.0%` | 40.34% | 40.34% | **40.34%** | 40.34% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_44_to_48` | `1` | `100.0%` | 59.78% | 58.94% | **58.94%** | 59.78% | **`-0.84`** `[-0.84, -0.84]` | **`+0.00`** `[+0.00, +0.00]` | `+0.84 pts` |

## 2. Classification of Material Edits (`|V_apply − V_keep| >= 2.0 pts`)

- **Material Proposals:** `0` (`0` helpful `>= +2.0 pts`, `0` harmful `<= -2.0 pts`)
- **Untuned `8B` Gatekeeper Accuracy:** **`0.0%`** (`TP_apply_helpful = 0`, `TN_keep_harmful = 0`, `FP_apply_harmful = 0`, `FN_keep_helpful = 0`)

## 3. Per-Opportunity Pre-Outcome Choices & Next-Review (`t_use`) Scores

| Seed | Mode | `t_edit -> t_use` | `8B` Choice | `Keep` DP (`σ`) | `Apply` DP (`σ`) | `Apply − Keep` | `8B` Achieved | Distinct Rule? |
|---:|---|---|---:|---:|---:|---:|---:|:---:|
| `738` | `drift_dip_c_60` | `16 -> 20` | **`APPLY`** | 66.6% (`0.0`) | 66.6% (`0.0`) | **`+0.00`** | 66.6% | `True` |
| `738` | `drift_dip_c_60` | `28 -> 32` | **`APPLY`** | 35.4% (`4.4`) | 36.7% (`1.2`) | **`+1.30`** | 36.7% | `True` |
| `738` | `drift_dip_c_60` | `36 -> 40` | **`KEEP`** | 40.3% (`0.0`) | 40.3% (`0.0`) | **`+0.00`** | 40.3% | `True` |
| `738` | `drift_dip_c_60` | `44 -> 48` | **`APPLY`** | 59.8% (`0.0`) | 58.9% (`1.3`) | **`-0.84`** | 58.9% | `True` |

