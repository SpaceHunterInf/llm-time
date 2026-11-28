# Untuned `Qwen/Qwen3-8B` Keep-or-Apply Reviewer Gatekeeper: `gate8b_s732`

- **Reviewer & Gatekeeper:** `Qwen/Qwen3-8B` (`$0.00` local RTX 8000 FP16) | **Fixed Actor:** `gemini-3.8-flash` | **Held-Out Third Setup Untouched:** `coalition_ops`
- **Fresh Streams:** `Seeds [732]` (`N = 1`) | **Preselected Reviews:** `[16, 28, 36, 44]` (`M = 4` opportunities) | **Eval Trials / Version:** `K = 6` | **Total Spend:** `$1.3528`

## 1. Gatekeeper Policy Comparison (Identical History, Actor, & 4-Session After-Fee Budget)

| Slice | Opportunities (`M`) | `8B` Apply Rate | `Always-Keep` DP (%) | `Always-Apply` DP (%) | **Untuned `8B` Policy DP (%)** | `Oracle` Ceiling (%) | `8B − Keep` (95% CI) | `8B − Apply` (95% CI) | `Oracle − 8B` Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All Preselected Opportunities | `4` | `100.0%` | 62.41% | 61.60% | **61.60%** | 62.41% | **`-0.80`** `[-2.32, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.80 pts` |
| Non-Identical Rule Proposals (`R_prop != R_cur`) | `4` | `100.0%` | 62.41% | 61.60% | **61.60%** | 62.41% | **`-0.80`** `[-2.32, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.80 pts` |
| Mode: `drift_meta_60` | `4` | `100.0%` | 62.41% | 61.60% | **61.60%** | 62.41% | **`-0.80`** `[-2.32, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.80 pts` |
| Step: `ep_16_to_20` | `1` | `100.0%` | 70.39% | 67.30% | **67.30%** | 70.39% | **`-3.09`** `[-3.09, -3.09]` | **`+0.00`** `[+0.00, +0.00]` | `+3.09 pts` |
| Step: `ep_28_to_32` | `1` | `100.0%` | 41.92% | 41.80% | **41.80%** | 41.92% | **`-0.12`** `[-0.12, -0.12]` | **`+0.00`** `[+0.00, +0.00]` | `+0.12 pts` |
| Step: `ep_36_to_40` | `1` | `100.0%` | 65.38% | 65.38% | **65.38%** | 65.38% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_44_to_48` | `1` | `100.0%` | 71.94% | 71.94% | **71.94%** | 71.94% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |

## 2. Classification of Material Edits (`|V_apply − V_keep| >= 2.0 pts`)

- **Material Proposals:** `1` (`0` helpful `>= +2.0 pts`, `1` harmful `<= -2.0 pts`)
- **Untuned `8B` Gatekeeper Accuracy:** **`0.0%`** (`TP_apply_helpful = 0`, `TN_keep_harmful = 0`, `FP_apply_harmful = 1`, `FN_keep_helpful = 0`)

## 3. Per-Opportunity Pre-Outcome Choices & Next-Review (`t_use`) Scores

| Seed | Mode | `t_edit -> t_use` | `8B` Choice | `Keep` DP (`σ`) | `Apply` DP (`σ`) | `Apply − Keep` | `8B` Achieved | Distinct Rule? |
|---:|---|---|---:|---:|---:|---:|---:|:---:|
| `732` | `drift_meta_60` | `16 -> 20` | **`APPLY`** | 70.4% (`1.3`) | 67.3% (`0.2`) | **`-3.09`** | 67.3% | `True` |
| `732` | `drift_meta_60` | `28 -> 32` | **`APPLY`** | 41.9% (`3.7`) | 41.8% (`4.0`) | **`-0.12`** | 41.8% | `True` |
| `732` | `drift_meta_60` | `36 -> 40` | **`APPLY`** | 65.4% (`0.0`) | 65.4% (`0.0`) | **`+0.00`** | 65.4% | `True` |
| `732` | `drift_meta_60` | `44 -> 48` | **`APPLY`** | 71.9% (`0.0`) | 71.9% (`0.0`) | **`+0.00`** | 71.9% | `True` |

