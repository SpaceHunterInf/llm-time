# Untuned `Qwen/Qwen3-8B` Keep-or-Apply Reviewer Gatekeeper: `gate8b_s721`

- **Reviewer & Gatekeeper:** `Qwen/Qwen3-8B` (`$0.00` local RTX 8000 FP16) | **Fixed Actor:** `gemini-3.8-flash` | **Held-Out Third Setup Untouched:** `coalition_ops`
- **Fresh Streams:** `Seeds [721]` (`N = 1`) | **Preselected Reviews:** `[16, 28, 36, 44]` (`M = 4` opportunities) | **Eval Trials / Version:** `K = 6` | **Total Spend:** `$0.9659`

## 1. Gatekeeper Policy Comparison (Identical History, Actor, & 4-Session After-Fee Budget)

| Slice | Opportunities (`M`) | `8B` Apply Rate | `Always-Keep` DP (%) | `Always-Apply` DP (%) | **Untuned `8B` Policy DP (%)** | `Oracle` Ceiling (%) | `8B − Keep` (95% CI) | `8B − Apply` (95% CI) | `Oracle − 8B` Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All Preselected Opportunities | `4` | `25.0%` | 57.08% | 57.08% | **57.08%** | 57.08% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Non-Identical Rule Proposals (`R_prop != R_cur`) | `1` | `100.0%` | 76.84% | 76.84% | **76.84%** | 76.84% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Mode: `drift_meta_60` | `4` | `25.0%` | 57.08% | 57.08% | **57.08%** | 57.08% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_16_to_20` | `1` | `100.0%` | 76.84% | 76.84% | **76.84%** | 76.84% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_28_to_32` | `1` | `0.0%` | 47.13% | 47.13% | **47.13%** | 47.13% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_36_to_40` | `1` | `0.0%` | 37.33% | 37.33% | **37.33%** | 37.33% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |
| Step: `ep_44_to_48` | `1` | `0.0%` | 67.02% | 67.02% | **67.02%** | 67.02% | **`+0.00`** `[+0.00, +0.00]` | **`+0.00`** `[+0.00, +0.00]` | `+0.00 pts` |

## 2. Classification of Material Edits (`|V_apply − V_keep| >= 2.0 pts`)

- **Material Proposals:** `0` (`0` helpful `>= +2.0 pts`, `0` harmful `<= -2.0 pts`)
- **Untuned `8B` Gatekeeper Accuracy:** **`0.0%`** (`TP_apply_helpful = 0`, `TN_keep_harmful = 0`, `FP_apply_harmful = 0`, `FN_keep_helpful = 0`)

## 3. Per-Opportunity Pre-Outcome Choices & Next-Review (`t_use`) Scores

| Seed | Mode | `t_edit -> t_use` | `8B` Choice | `Keep` DP (`σ`) | `Apply` DP (`σ`) | `Apply − Keep` | `8B` Achieved | Distinct Rule? |
|---:|---|---|---:|---:|---:|---:|---:|:---:|
| `721` | `drift_meta_60` | `16 -> 20` | **`APPLY`** | 76.8% (`0.0`) | 76.8% (`0.0`) | **`+0.00`** | 76.8% | `True` |
| `721` | `drift_meta_60` | `28 -> 32` | **`KEEP`** | 47.1% (`0.0`) | 47.1% (`0.0`) | **`+0.00`** | 47.1% | `False` |
| `721` | `drift_meta_60` | `36 -> 40` | **`KEEP`** | 37.3% (`0.0`) | 37.3% (`0.0`) | **`+0.00`** | 37.3% | `False` |
| `721` | `drift_meta_60` | `44 -> 48` | **`KEEP`** | 67.0% (`3.2`) | 67.0% (`3.2`) | **`+0.00`** | 67.0% | `False` |

