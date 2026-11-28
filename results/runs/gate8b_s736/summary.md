# Untuned `Qwen/Qwen3-8B` Keep-or-Apply Reviewer Gatekeeper: `gate8b_s736`

- **Reviewer & Gatekeeper:** `Qwen/Qwen3-8B` (`$0.00` local RTX 8000 FP16) | **Fixed Actor:** `gemini-3.8-flash` | **Held-Out Third Setup Untouched:** `coalition_ops`
- **Fresh Streams:** `Seeds [736]` (`N = 1`) | **Preselected Reviews:** `[16, 28, 36, 44]` (`M = 4` opportunities) | **Eval Trials / Version:** `K = 6` | **Total Spend:** `$1.4443`

## 1. Gatekeeper Policy Comparison (Identical History, Actor, & 4-Session After-Fee Budget)

| Slice | Opportunities (`M`) | `8B` Apply Rate | `Always-Keep` DP (%) | `Always-Apply` DP (%) | **Untuned `8B` Policy DP (%)** | `Oracle` Ceiling (%) | `8B − Keep` (95% CI) | `8B − Apply` (95% CI) | `Oracle − 8B` Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All Preselected Opportunities | `4` | `75.0%` | 88.04% | 88.41% | **88.41%** | 88.41% | **`+0.36`** `[+0.00, +0.73]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Non-Identical Rule Proposals (`R_prop != R_cur`) | `4` | `75.0%` | 88.04% | 88.41% | **88.41%** | 88.41% | **`+0.36`** `[+0.00, +0.73]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Mode: `drift_dip_c_60` | `4` | `75.0%` | 88.04% | 88.41% | **88.41%** | 88.41% | **`+0.36`** `[+0.00, +0.73]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_16_to_20` | `1` | `100.0%` | 96.73% | 97.22% | **97.22%** | 97.22% | **`+0.49`** `[+0.49, +0.49]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_28_to_32` | `1` | `100.0%` | 97.70% | 98.67% | **98.67%** | 98.67% | **`+0.97`** `[+0.97, +0.97]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_36_to_40` | `1` | `100.0%` | 100.59% | 100.59% | **100.59%** | 100.59% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_44_to_48` | `1` | `0.0%` | 57.14% | 57.14% | **57.14%** | 57.14% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |

## 2. Classification of Material Edits (`|V_apply − V_keep| >= 2.0 pts`)

- **Material Proposals:** `0` (`0` helpful `>= +2.0 pts`, `0` harmful `<= -2.0 pts`)
- **Untuned `8B` Gatekeeper Accuracy:** **`0.0%`** (`TP_apply_helpful = 0`, `TN_keep_harmful = 0`, `FP_apply_harmful = 0`, `FN_keep_helpful = 0`)

## 3. Per-Opportunity Pre-Outcome Choices & Next-Review (`t_use`) Scores

| Seed | Mode | `t_edit -> t_use` | `8B` Choice | `Keep` DP (`σ`) | `Apply` DP (`σ`) | `Apply − Keep` | `8B` Achieved | Distinct Rule? |
|---:|---|---|---:|---:|---:|---:|---:|:---:|
| `736` | `drift_dip_c_60` | `16 -> 20` | **`APPLY`** | 96.7% (`0.0`) | 97.2% (`1.2`) | **`+0.49`** | 97.2% | `True` |
| `736` | `drift_dip_c_60` | `28 -> 32` | **`APPLY`** | 97.7% (`1.5`) | 98.7% (`1.5`) | **`+0.97`** | 98.7% | `True` |
| `736` | `drift_dip_c_60` | `36 -> 40` | **`APPLY`** | 100.6% (`0.0`) | 100.6% (`0.0`) | **`+0.00`** | 100.6% | `True` |
| `736` | `drift_dip_c_60` | `44 -> 48` | **`KEEP`** | 57.1% (`0.0`) | 57.1% (`0.0`) | **`+0.00`** | 57.1% | `True` |

