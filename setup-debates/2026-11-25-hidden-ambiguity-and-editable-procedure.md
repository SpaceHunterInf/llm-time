# Post-Meeting Empirical Execution: Hidden-Ambiguity VOI, Self-Authored Editable vs. Frozen Procedures, and Zero-Shot Cross-Domain Transfer (2026-11-25)

Following the 23 November group meeting, we executed all three pre-registered priorities from [2026-11-22-group-meeting-brief.md](2026-11-22-group-meeting-brief.md) and [RESEARCH_STATE.md](../RESEARCH_STATE.md) §4.1 under strict `<$1.95`/shard guardrails (`28/28` Tier-0 `pytest` checks green, `0` `role_to_strat` scripting in `W1` acquisition, and `100%` verbatim pre-episode rule/procedure trace logging).

---

## 1. Priority 1 — Unscripted Hidden-Ambiguity VOI Fork (`voi_hidden_ambiguity_flash_n6` & `voi_hidden_ambiguity_hydrogen_n3`)

- **Design (`AlphaDeskV3ForkEnv`, `hidden_transient_dip` vs. `abrupt_reversal`):**
  - Shared `W1` checkpoints (`t_fork = 14` on `Flash`, `t_fork = 12` on `Hydrogen`) were acquired via **live model rollouts using only public observations** ([w1_live_checkpoints.json](../results/runs/voi_hidden_ambiguity_flash_n6/w1_live_checkpoints.json)).
  - At `t_fork` and `t_fork + 1`, `hidden_transient_dip` (`r0=18, r1=20, r2..r5=19` for 2 sessions before `A` returns) and `abrupt_reversal` (`B = 18, 20, 93, 81, 24, 20`) expose **100% bit-for-bit identical free `inspect_desk_brief` covariates** (`calm_volatility`, `orderly_normal`, normal spreads) and **bit-for-bit identical `t_fork` realized portfolio loss (`18.9` + matched noise)**.
  - Only paying `-10.0 pts` for `probe_strategy_execution` separates them via anonymized cross-sectional tape diagnostics (`desk_best_yield_estimate ~ 20.4` under systemic dip vs. `~ 93.0` under strategy reversal).

| Model (`Seeds`) | Arm | `stationary` (%) | `hidden_transient_dip` (%) | `abrupt_reversal` (%) | Dip Probes / Abandon% |
|---|---|---:|---:|---:|---:|
| **`gemini-3.8-flash`** (`N=6`, `601–606`) | `ScopedCardBook (Baseline)` | `84.6%` | `84.6%` | **`49.7%`** | `0.0` / `0%` |
| **`gemini-3.8-flash`** (`N=6`, `601–606`) | **`No-Procedure / Factual-Ledger (Ours)`** | `83.4%` | **`80.1%`** | **`78.1%`** | `1.0` / `17%` |
| **`gemini-3.8-flash`** (`N=6`, `601–606`) | **`Editable-Procedure (Ours)`** | **`86.3%`** | **`83.6%`** | **`77.3%`** | `0.8` / **`0%`** |
| **`gemini-3.8-flash`** (`N=6`, `601–606`) | `Directive-SGEH (Controller Ref)` | **`88.6%`** | `59.0%` | `73.1%` | `1.0` / `100%` |
| **`gemini-4-hydrogen`** (`N=3`, `601–603`) | `ScopedCardBook (Baseline)` | `98.4%` | `98.2%` | **`56.9%`** | `0.0` / `0%` |
| **`gemini-4-hydrogen`** (`N=3`, `601–603`) | **`No-Procedure / Factual-Ledger (Ours)`** | `97.3%` | **`95.9%`** | **`82.0%`** | `1.0` / **`0%`** |
| **`gemini-4-hydrogen`** (`N=3`, `601–603`) | **`Editable-Procedure (Ours)`** | **`97.8%`** | **`96.0%`** | **`78.7%`** | `1.0` / **`0%`** |

- **Key Findings:**
  1. `ScopedCardBook (Baseline)` never pays to check (`0.0` probes): it survives `hidden_transient_dip` by blind inertia, **collapsing by `-28.34 pts` (`Flash`, `6/6` seeds) and `-25.12 pts` (`Hydrogen`, `3/3` seeds) on `abrupt_reversal`**.
  2. Rigid CUSUM quarantine (`Directive-SGEH` = `59.0%` on `hidden_transient_dip`) suffers **false incumbent quarantine** when the 2-session unannounced dip ends right as the CUSUM alarm triggers (`ep 16 -> ep 17`).
  3. **`Editable-Procedure` (`83.6%` / `96.0%` dip, `0%` abandonment; `77.3%` / `78.7%` reversal) and `Factual-Ledger` (`80.1%` / `95.9%` dip; `78.1%` / `82.0%` reversal)** autonomously pay `-10 pts` for `~1.0` diagnostic check on the second dip day, read `desk_max ~ 20.4`, preserve their calm holdings (`0.0` dip turnover), and still recover `+25` to `+28 pts` above `ScopedCardBook` when the shift is permanent.

---

## 2. Priority 2 — Self-Authored `Editable-Procedure` vs. Matched-Budget `Frozen-Procedure` (`proc3w_flash_n4`)

- **Design (`AlphaDeskV3ThreeWindowEnv`, `W1 [12] -> W2 [10] -> W3 [10]`, `Seeds 601–604`):**
  - Agents start at `ep = 0` with `self_procedure = ""` (blank) and neutral public mechanics (`0` hints). At `t_ckpt = 12`, we fork `No-Procedure` (`Factual-Ledger`), `Frozen-Procedure` (same `every 4 eps` review call schedule & token budget, edits logged counterfactually while `self_procedure` stays locked at `P_W1` — testing Sundaram et al. `arXiv:2611.23219`), `Editable-Procedure` (edits take effect), and `Directive-SGEH` ([summary.md](../results/runs/proc3w_flash_n4/summary.md), [procedure_versions_audit.json](../results/runs/proc3w_flash_n4/procedure_versions_audit.json)).

| 3-Window Stream (`W2 -> W3`) | `No-Procedure (Factual-Ledger)` | `Frozen-Procedure (Control)` | `Editable-Procedure (Ours)` | `Directive-SGEH (Controller)` | Primary Contrast (`95% CI`) |
|---|---:|---:|---:|---:|---|
| `stationary_3w` (`A -> A`) | `92.2%` | `89.2%` | `89.8%` (`0.2` prb) | `93.7%` | `Editable − Frozen = +0.65 pts [+0.00, +1.93]` |
| `double_reversal_3w` (`B -> C`) | **`70.2%`** | `59.5%` | **`69.6%`** | `67.8%` | **`Editable − Frozen = +10.12 pts [+5.24, +15.00]` (`4/4` seeds)** |
| `dip_then_reversal_3w` (`Dip+A -> B`) | **`83.6%`** (`89/79`) | `73.8%` (`89/61`) | `68.4%` (`90/51`) | `78.2%` (`72/84`) | **`Editable − NoProc = -15.19 pts [-28.91, -1.46]`** |
| `opp_stationary_3w` (`A_opp -> A_opp`) | `83.5%` | `85.0%` | `85.2%` (`0.2` prb) | `93.8%` | `Editable − Frozen = +0.21 pts [+0.00, +0.64]` |
| `opp_double_rise_3w` (`B_opp -> D_opp`) | `71.9%` (`0` non-inc) | `73.0%` (`0` non-inc) | `73.2%` (`0` non-inc) | **`86.3%`** | **`Directive − Editable = +13.09 pts [+10.42, +15.32]` (`4/4` seeds)** |

- **Three Sharp Mechanistic Takeaways:**
  1. **Why `Frozen-Procedure` Fails Under Repeated Reversals (`+10.12 pts` for `Editable`):** During stationary `W1`, the frozen LLM spontaneously writes `W1` strategy names and `"never probe (-10 pts)"` into `P_W1` (successfully keeping its money on `stationary_3w`!). Locking `P_W1` (`Frozen-Procedure`) turns the agent's own procedure into a stale instruction file across `B -> C` (`59.5%`), whereas `Editable-Procedure` revises it (`69.6%`, `4/4` seeds positive).
  2. **Candid Negative Finding — *Self-Authored Lesson Over-Generalization* (`dip_then_reversal_3w`):** After surviving the `W2` hidden transient dip (`89.5%` in `W2`), `Editable-Procedure` writes a lesson discounting calm yield drops as temporary market compression—causing it to stall at `50.7%` when `W3` brings a *true* permanent reversal (`68.4%` vs. `83.6%` `No-Procedure`). Thus `Lessons Outlive Their Failures` (`arXiv:2610.32578`) emerges endogenously inside self-editing procedures, while pure `Factual-Ledger` (`89.1%` `W2` / `79.0%` `W3`) avoids sticky text commitments.
  3. **Zero Spontaneous Type-II Hazard Discovery (`opp_double_rise_3w`):** Across all `8` self-edits (`v8.0`), `Editable-Procedure` took **`0.0` non-incumbent probes** (`73.2%` vs. `86.3%` `Directive-SGEH`, `-13.09 pts`), proving that prompt-level procedure self-modification still inherits the Type-II zero-exploration blindspot when incumbent rewards stay flat.

---

## 3. Priority 3 — Zero-Shot Cross-Domain Procedure Transfer on `ServeLearnBench L3` (`slb_banking_l3_transfer_6win`)

- Froze the domain-agnostic procedure distilled from `AlphaDesk` (`SHA-256: 1a9e347720f75148...`, [frozen_transferred_procedure.json](../results/runs/proc3w_flash_n4/frozen_transferred_procedure.json)) prior to touching `ServeLearnBench`, and evaluated `banking_l3` (`W0..W5`, sparser `9`-serving-case/window slice `serve_adapt=6, serve_general=3`, `100%` verbatim pre-episode state/context JSONL logging in [results/raw/](../results/raw)):
  - `Oracle = 100.0%`, `Blind = 60.0%` (`40.0%` `Adapt`, `93.3%` `General`), **`Transferred-Procedure + FactualLog = 55.0%`** (`36.0%` `Adapt`, `86.7%` `General`, `13` ref calls), **`SRN = 55.0%`** (`40.0%` `Adapt`, `80.0%` `General`, `27` ref calls), `Factual-SLB-Log = 52.5%` (`32.0%` `Adapt`, `86.7%` `General`, `0` ref calls), `ScopedSGEH+Shelf = 40.0%` (`17` ref calls).
  - **Candid Interpretation (Confirming Codex's Nov 22 Sample-Size Warning):** Dropping from `12` serving cases/window (`slb_banking_l3_6win`, where `ScopedSGEH+Shelf` hit `70.0%` vs `62.5%` `SRN`) to `9` binary `0/1` cases/window with zero diagnostic probe tool leaves multi-clause policy shifts statistically underdetermined (`<= 60%` `Blind`), though `Transferred-Procedure` matched `SRN` (`55.0%`) at `2×` fewer reflection calls (`13` vs `27`) and higher `General` invariant compliance (`86.7%` vs `80.0%`). Full-stream (`50–100` serving cases/window) Tier-2 evaluation is required for decisive binary-feedback separation on `ServeLearnBench L3`.
