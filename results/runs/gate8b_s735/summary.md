# Untuned `Qwen/Qwen3-8B` Keep-or-Apply Reviewer Gatekeeper: `gate8b_s735`

- **Reviewer & Gatekeeper:** `Qwen/Qwen3-8B` (`$0.00` local RTX 8000 FP16) | **Fixed Actor:** `gemini-3.8-flash` | **Held-Out Third Setup Untouched:** `coalition_ops`
- **Fresh Streams:** `Seeds [735]` (`N = 1`) | **Preselected Reviews:** `[16, 28, 36, 44]` (`M = 4` opportunities) | **Eval Trials / Version:** `K = 6` | **Total Spend:** `$1.2024`

## 1. Gatekeeper Policy Comparison (Identical History, Actor, & 4-Session After-Fee Budget)

| Slice | Opportunities (`M`) | `8B` Apply Rate | `Always-Keep` DP (%) | `Always-Apply` DP (%) | **Untuned `8B` Policy DP (%)** | `Oracle` Ceiling (%) | `8B − Keep` (95% CI) | `8B − Apply` (95% CI) | `Oracle − 8B` Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All Preselected Opportunities | `4` | `100.0%` | 54.80% | 54.31% | **54.31%** | 54.80% | **`-0.49`** `[-1.36, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.49 pts` |
| Non-Identical Rule Proposals (`R_prop != R_cur`) | `4` | `100.0%` | 54.80% | 54.31% | **54.31%** | 54.80% | **`-0.49`** `[-1.36, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.49 pts` |
| Mode: `drift_dip_c_60` | `4` | `100.0%` | 54.80% | 54.31% | **54.31%** | 54.80% | **`-0.49`** `[-1.36, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.49 pts` |
| Step: `ep_16_to_20` | `1` | `100.0%` | 68.72% | 68.72% | **68.72%** | 68.72% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_28_to_32` | `1` | `100.0%` | 36.39% | 34.57% | **34.57%** | 36.39% | **`-1.82`** `[-1.82, -1.82]` | **`+0.00`** `[+0.00, +0.00]` | `+1.82 pts` |
| Step: `ep_36_to_40` | `1` | `100.0%` | 52.41% | 52.41% | **52.41%** | 52.41% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_44_to_48` | `1` | `100.0%` | 61.67% | 61.53% | **61.53%** | 61.67% | **`-0.14`** `[-0.14, -0.14]` | **`+0.00`** `[+0.00, +0.00]` | `+0.14 pts` |

## 2. Classification of Material Edits (`|V_apply − V_keep| >= 2.0 pts`)

- **Material Proposals:** `0` (`0` helpful `>= +2.0 pts`, `0` harmful `<= -2.0 pts`)
- **Untuned `8B` Gatekeeper Accuracy:** **`0.0%`** (`TP_apply_helpful = 0`, `TN_keep_harmful = 0`, `FP_apply_harmful = 0`, `FN_keep_helpful = 0`)

## 3. Per-Opportunity Pre-Outcome Choices & Next-Review (`t_use`) Scores

| Seed | Mode | `t_edit -> t_use` | `8B` Choice | `Keep` DP (`σ`) | `Apply` DP (`σ`) | `Apply − Keep` | `8B` Achieved | Distinct Rule? |
|---:|---|---|---:|---:|---:|---:|---:|:---:|
| `735` | `drift_dip_c_60` | `16 -> 20` | **`APPLY`** | 68.7% (`0.0`) | 68.7% (`0.0`) | **`+0.00`** | 68.7% | `True` |
| `735` | `drift_dip_c_60` | `28 -> 32` | **`APPLY`** | 36.4% (`7.2`) | 34.6% (`2.8`) | **`-1.82`** | 34.6% | `True` |
| `735` | `drift_dip_c_60` | `36 -> 40` | **`APPLY`** | 52.4% (`0.0`) | 52.4% (`0.0`) | **`+0.00`** | 52.4% | `True` |
| `735` | `drift_dip_c_60` | `44 -> 48` | **`APPLY`** | 61.7% (`1.8`) | 61.5% (`0.0`) | **`-0.14`** | 61.5% | `True` |

