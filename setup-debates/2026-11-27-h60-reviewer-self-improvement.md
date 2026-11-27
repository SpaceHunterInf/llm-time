# Single-Lifetime `H = 60` Two-Level Reviewer Self-Improvement & Paired `W4` Counterfactual Replay (2026-11-27)

Following Astra/Codex's 27 November review and our pre-registered protocol ([2026-11-26-decomposed-procedure-and-vbq-resolution.md](2026-11-26-decomposed-procedure-and-vbq-resolution.md#2026-11-27-codex-review-of-the-single-lifetime-horizon)), we implemented `AlphaDeskV3MetaHorizonEnv` (`H = 60` sessions, private seed-jittered 4-window schedules across fresh Latin-square `Seeds 701–706`) and `TwoLevelReviewerHarness` (`src/harnesses/procedure_harness.py`), and executed the full paired benchmark across **`gemini-3.8-flash` (`N = 6`, `$13.59`, [summary.md](../results/runs/meta_reviewer_h60_flash_n6/summary.md))** and **`gemini-4-hydrogen` (`N = 3`, `$15.89`, [summary.md](../results/runs/meta_reviewer_h60_hydrogen_n3/summary.md))**, with every shard capped strictly at `--max-usd 1.95`.

---

## 1. Clean Two-Level Isolation (`task_note` $N_t$ vs. Reviewer-Only `review_rule` $R_t$)

- **Zero Researcher Hints or Regex Scrubbers:** Both `task_note` (`<= 120` words) and `review_rule` (`<= 100` words) initialize blank (`""`) at Episode 1 with **zero** strategy-ID scrubbers, **zero** schema-separation rules, and **zero** researcher-supplied yield/probing hints.
- **Strict Actor Isolation (`Verified in pytest`):** `TwoLevelReviewerHarness.build_context(spec)` exposes **only** `Factual-Ledger` + `task_note` $N_t$ to the actor (`review_rule` $R_t$ is strictly excluded from actor prompts).
- **Common Live `W1` Checkpoint Fork (`t_ckpt = 12 ± 1`):** From identical live `W1` `(N_ckpt, R_ckpt, observations, holdings)`, both `Editable-Reviewer` and `Frozen-Reviewer` rewrite `task_note` every `4` sessions (`11–12` post-fork review cycles) under identical review and tool budgets; **only `Editable-Reviewer` applies proposed updates to `review_rule` $R_t$**, so any downstream difference must pass through `ΔR_t -> ΔN_{t+4} -> Δaction`.
- **Private 4-Window Drift Schedule (`drift_meta_60` vs. Matched `stationary_meta_60`):**
  - `W1` (`11–13` sessions): Calm acquisition `A = (92, 80, 22, 20, 24, 18)`.
  - `W2` (`12–15` sessions — *Temporary Loss*): Opens with a 2-session unannounced `hidden_transient_dip` (`18, 20, 19..19` under `orderly_normal`), then returns to `A`.
  - `W3` (`14–18` sessions — *Lasting Change*): Permanent shift to `B = (18, 20, 93, 81, 24, 20)`.
  - `W4` (`15–20` sessions — *Later Surprise: Revision + Selective Reuse*): Shifts to Partial Recurrence `A' = (92, 18, 81, 20, 24, 18)`, where `W3`'s #2 (`r3=20`) breaks (`87.6 -> 53.6`), `W3`'s #1 (`r2=81`) stays optimal runner-up, and `W1`'s #1 (`r0=92`) returns.

---

## 2. Empirical Results: Honest Negative on Untuned `Flash` (`N = 6`) vs. Action-Agnostic Discipline on `Hydrogen` (`N = 3`)

### 2.1 `gemini-3.8-flash` (`N = 6`, `Seeds 701–706`, [summary.md](../results/runs/meta_reviewer_h60_flash_n6/summary.md))

| Arm (`drift_meta_60`) | **W4 Surprise DP (%)** | Post-Ckpt DP (%) | W2 / W3 DP (%) | W4 CalmOpt (%) | Probes (W4) | Note / Rule Edits | Stationary Post DP (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Facts-Only` (Autonomy Baseline — No Notes) | **72.7%** | 75.1% | `79.7%` / `75.6%` | 28.3% | 5.0 (`1.3`) | `0.0` / `0.0` | 79.8% |
| `Frozen-Reviewer` (Control — `R_t ≡ R_ckpt`) | **70.7%** | 68.3% | `80.7%` / `56.7%` | 21.1% | 2.8 (`0.8`) | `10.0` / `0.0` | 79.4% |
| **`Editable-Reviewer` (Ours — Both Editable)** | **69.5%** | 66.3% | `78.1%` / `54.6%` | 22.2% | 3.5 (`1.0`) | `9.5` / `8.5` | 78.7% |
| `Directive-SGEH-VBQ` (Reference Controller) | **71.5%** | **78.4%** | `83.2%` / `83.8%` | 24.2% | 18.8 (`7.2`) | `0.0` / `0.0` | **94.0%** |

- **Aggregate Live Finding (Negative for Zero-Shot Self-Editing on `Flash`):**
  - **`W4 Editable − Frozen Reviewer`:** **`-1.23 pts [-14.08, +11.62]`** (per-seed: `[-23.97, 0.00, +5.15, +7.64, +22.08, -18.28]`; Stationary-adjusted DiD **`+0.83 pts [-10.71, +12.12]`**).
  - **`Post-Ckpt Editable − Facts-Only`:** **`-8.84 pts [-14.84, -2.85]`** (DiD **`-7.79 pts [-14.51, -0.09]`**; `Frozen − Facts = -6.86 pts`). Both self-authored reviewer arms stalled in `W3` (`54.6%` / `56.7%` vs. `75.6%` `Facts-Only` and `83.8%` `Directive-SGEH-VBQ`).
- **Paired `W4` Counterfactual Replay (`R_ckpt` Old Rule vs. `R_pre_W4` New Rule on Identical Observations, [counterfactual_w4_replay.json](../results/runs/meta_reviewer_h60_flash_n6/counterfactual_w4_replay.json)):**
  - Across all `6/6` seeds, `Editable-Reviewer` modified `review_rule` before `W4` (`8.5` edits/life) and produced a distinct `task_note` revision, with **`0/6` negative counterfactual replays** (**mean 4-episode net-of-cost DP gain: `+13.48 pts [+0.00, +30.86]`**, per-seed `[0.00, 0.00, 0.00, +52.15, +28.73, 0.00]`):
    - **When `Flash` Evolved a Falsification Rule (`2/6` Seeds — `704` & `705`), the Full Causal Chain Fired Strongly:**
      - *Seed `704`:* `R_ckpt` locked `{S4: 0.55, S6: 0.45}` with `probe=none` (`23.7%` 4-ep DP, `0` probes). After `W3`, `R_pre_W4` added *"If calm performance drops below 25 net for 3+ consecutive sessions, consider probing alternatives"*, triggering a `STRAT_S3` (`r0=92`) probe in Session #49 and jumping to **`75.8%` 4-ep DP (`+52.15%` / `+177.15 pts` net of `-10 pt` probe & turnover fees)**.
      - *Seed `705`:* `R_pre_W4` added *"as long as session gross remains >= 55; if calm gross drops below 45, probe S2 or S3"*, lifting 4-ep replay DP from **`52.6% -> 81.3%` (`+28.73%` / `+100.05 pts`)** and lifting live `W4` DP to **`85.0%` (`+22.08 pts` over `Frozen-Reviewer`, `+14.02 pts` over `Facts-Only`, and `+23.40 pts` over `Directive-SGEH-VBQ`)**.
    - **Why `4/6` Seeds Failed (`Second-Order Satisficing & Object-Binding Ratchet`):** On Seeds `701, 702, 703, 706`, `Flash` wrote concrete strategy weights + *"confirm the actor remains fully passive... with zero probing"* directly into `review_rule` and lowered its subjective reference anchor during `W3` (e.g., `s703` labeled `~28–34` calm net *"consistently high"*, locking `W3` at `36%` and yielding `+0.00%` replay delta).

### 2.2 `gemini-4-hydrogen` (`N = 3`, `Seeds 701–703`, [summary.md](../results/runs/meta_reviewer_h60_hydrogen_n3/summary.md))

| Arm (`drift_meta_60`) | **W4 Surprise DP (%)** | Post-Ckpt DP (%) | W2 / W3 DP (%) | W4 CalmOpt (%) | Probes (W4) | Note / Rule Edits | Stationary Post DP (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Facts-Only` (Autonomy Baseline — No Notes) | **73.1%** | 80.2% | `84.6%` / `85.2%` | 28.9% | 6.3 (`2.3`) | `0.0` / `0.0` | **97.2%** |
| `Frozen-Reviewer` (Control — `R_t ≡ R_ckpt`) | **80.3%** | 80.8% | `89.2%` / `74.6%` | 47.6% | 6.3 (`2.7`) | `10.3` / `0.0` | 91.6% |
| **`Editable-Reviewer` (Ours — Both Editable)** | **84.0%** | **84.3%** | `89.9%` / `79.8%` | **53.9%** | 6.7 (`2.7`) | `9.7` / `1.3` | 91.6% |
| `Directive-SGEH-VBQ` (Reference Controller) | **61.9%** | 77.6% | `88.9%` / `86.3%` | 0.0% | 21.3 (`9.7`) | `0.0` / `0.0` | 96.0% |

- **Spontaneous Action-Agnostic `review_rule`s (`0` Strategy IDs in `review_rule` Across `3/3` Seeds):**
  - Unlike `Flash`, `Hydrogen` kept `review_rule` 100% strategy-agnostic at `R_ckpt` (*"Evaluate strategy yields separately by (vol, liquidity) regime... Recommend probing only in under-explored regimes..."*) and made **`1.3` surgical rule edits/life**—specifically at `ep 28` (`W3` onset) on **`3/3` seeds**, adding *"or when established leaders suffer consecutive sharp drops"* ([reviewer_versions_audit.json](../results/runs/meta_reviewer_h60_hydrogen_n3/reviewer_versions_audit.json)).
- **Live Performance vs. Honest Counterfactual Replay Attribution:**
  - Live `Editable-Reviewer` ranked **`#1` on both `W4` Surprise (`84.0%`, `53.9%` CalmOpt) and Post-Ckpt (`84.3%`)**, beating `Facts-Only` in `W4` by **`+10.94 pts [+0.00, +29.28]`** (`+18.15 pts` DiD), edging `Frozen-Reviewer` (`+3.70 pts` `W4` DiD; **`+3.55 pts [+0.54, +8.62]` Post DiD, `3/3` seeds positive**; `0.00 pts` difference on `stationary_meta_60`), and beating `Directive-SGEH-VBQ` in `W4` by **`+22.09 pts [+4.34, +33.35]` (`3/3` seeds)** (where `VBQ`'s fixed threshold kept `W3`'s `r2=81` primary and spent `9.7` `W4` probes = `-97 pts` searching for a runner-up, whereas `Editable-Reviewer` cleanly reused `W1`'s `r0=92` + `W3`'s `r2=81` in `2.7` probes!).
  - **Crucial Replay Caveat (`-3.09 pts [-4.74, +0.00]` 4-ep replay delta):** On the controlled `ep 44/48` `W4` counterfactual replay, `Hydrogen` *already* ordered a `W4` diagnostic probe even under `R_ckpt` (`Old 4-ep DP = 82.5%` vs. `New = 79.4%`; on `s702`, `Old` probed winner `S1` first in 1 probe while `New` probed unprobed `S6` before `S1` in 2 probes). Thus `Hydrogen`'s `+10.94 pt` `W4` advantage over `Facts-Only` and `+22.09 pt` `W4` advantage over `VBQ` come primarily from **maintaining a clean regime-scoped `task_note` + action-agnostic reviewer policy**, while much of its `+3.70 pt` live margin over `Frozen-Reviewer` originated at the earlier `W3` (`ep 28–32`) transition rather than `R_pre_W4` at `ep 44`.

---

## 3. Implications & Critical Next Decision (`Awaiting Explicit PI GO`)

1. **What Is Now Settled:**
   - Within a single `60`-session lifetime, **prompt-only reviewer self-editing does not reliably self-improve an untuned smaller model (`Flash`)**: `Editable-Reviewer` ties `Frozen-Reviewer` (`-1.23 pts` `W4`) and trails `Facts-Only` (`-8.84 pts` Post) because on `4/6` seeds it writes concrete strategy lock-in + `"zero probing"` into `review_rule` and lowers its expectation anchor during `W3`.
   - **Yet the two-level reviewer representation itself is NOT the bottleneck:** When `Flash` *does* write a drop-triggered falsification `review_rule` (`Seeds 704 & 705`), counterfactual replay proves **`+52.15%` and `+28.73%`** net-of-cost gains; and when a stronger model (`Hydrogen`) naturally writes action-agnostic `review_rule`s, `Editable-Reviewer` (`84.0%` `W4`, `84.3%` Post) beats **both** `Facts-Only` (`+10.94 pts` `W4`) **and** our hand-written `Directive-SGEH-VBQ` controller (`+22.09 pts` `W4` on Partial Recurrence `A'`)!
2. **Proposed Next Step (`Awaiting Explicit PI GO` — Act 3 Cross-Lifetime Small-Model Reviewer Training):**
   - Execute Astra's held-out small-model reviewer protocol (`Qwen3-8B` LoRA on our 48 GB RTX 8000 $\to$ CSD3): train only the `8B` **reviewer** across lifetimes in two setups using downstream net-of-checking-cost return, freeze its weights, clear `task_note` & `review_rule` (`""`), and evaluate zero-shot in the held-out third setup against the untuned `8B` reviewer under identical actor, tools, and budget.
