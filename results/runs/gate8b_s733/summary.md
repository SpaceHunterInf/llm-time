# Untuned `Qwen/Qwen3-8B` Keep-or-Apply Reviewer Gatekeeper: `gate8b_s733`

- **Reviewer & Gatekeeper:** `Qwen/Qwen3-8B` (`$0.00` local RTX 8000 FP16) | **Fixed Actor:** `gemini-3.8-flash` | **Held-Out Third Setup Untouched:** `coalition_ops`
- **Fresh Streams:** `Seeds [733]` (`N = 1`) | **Preselected Reviews:** `[16, 28, 36, 44]` (`M = 4` opportunities) | **Eval Trials / Version:** `K = 6` | **Total Spend:** `$1.3293`

## 1. Gatekeeper Policy Comparison (Identical History, Actor, & 4-Session After-Fee Budget)

| Slice | Opportunities (`M`) | `8B` Apply Rate | `Always-Keep` DP (%) | `Always-Apply` DP (%) | **Untuned `8B` Policy DP (%)** | `Oracle` Ceiling (%) | `8B − Keep` (95% CI) | `8B − Apply` (95% CI) | `Oracle − 8B` Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All Preselected Opportunities | `4` | `75.0%` | 82.90% | 82.52% | **82.52%** | 82.90% | **`-0.38`** `[-1.15, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.38 pts` |
| Non-Identical Rule Proposals (`R_prop != R_cur`) | `3` | `100.0%` | 79.22% | 78.71% | **78.71%** | 79.22% | **`-0.51`** `[-1.53, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.51 pts` |
| Mode: `drift_meta_60` | `4` | `75.0%` | 82.90% | 82.52% | **82.52%** | 82.90% | **`-0.38`** `[-1.15, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.38 pts` |
| Step: `ep_16_to_20` | `1` | `100.0%` | 79.41% | 79.41% | **79.41%** | 79.41% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_28_to_32` | `1` | `100.0%` | 93.92% | 92.39% | **92.39%** | 93.92% | **`-1.53`** `[-1.53, -1.53]` | **`+0.00`** `[+0.00, +0.00]` | `+1.53 pts` |
| Step: `ep_36_to_40` | `1` | `0.0%` | 93.94% | 93.94% | **93.94%** | 93.94% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_44_to_48` | `1` | `100.0%` | 64.33% | 64.33% | **64.33%** | 64.33% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |

## 2. Classification of Material Edits (`|V_apply − V_keep| >= 2.0 pts`)

- **Material Proposals:** `0` (`0` helpful `>= +2.0 pts`, `0` harmful `<= -2.0 pts`)
- **Untuned `8B` Gatekeeper Accuracy:** **`0.0%`** (`TP_apply_helpful = 0`, `TN_keep_harmful = 0`, `FP_apply_harmful = 0`, `FN_keep_helpful = 0`)

## 3. Per-Opportunity Pre-Outcome Choices & Next-Review (`t_use`) Scores

| Seed | Mode | `t_edit -> t_use` | `8B` Choice | `Keep` DP (`σ`) | `Apply` DP (`σ`) | `Apply − Keep` | `8B` Achieved | Distinct Rule? |
|---:|---|---|---:|---:|---:|---:|---:|:---:|
| `733` | `drift_meta_60` | `16 -> 20` | **`APPLY`** | 79.4% (`0.0`) | 79.4% (`0.0`) | **`+0.00`** | 79.4% | `True` |
| `733` | `drift_meta_60` | `28 -> 32` | **`APPLY`** | 93.9% (`3.8`) | 92.4% (`2.1`) | **`-1.53`** | 92.4% | `True` |
| `733` | `drift_meta_60` | `36 -> 40` | **`KEEP`** | 93.9% (`0.0`) | 93.9% (`0.0`) | **`+0.00`** | 93.9% | `False` |
| `733` | `drift_meta_60` | `44 -> 48` | **`APPLY`** | 64.3% (`0.0`) | 64.3% (`0.0`) | **`+0.00`** | 64.3% | `True` |

