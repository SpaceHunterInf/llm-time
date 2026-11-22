# `AlphaDesk-v0.3`: Multi-Shift Stateful Benchmark Parity & Pre-Registered Specification

**Date:** 2026-11-21  
**Participants:** PI + Gemini Co-Lead (open for Codex audit prior to implementation)  
**Status:** **Completed & Recorded (`alphadesk_v3_multishift_n4` + `slb_banking_l3_6win`) — Stopped at Next `GO` Gate**

---

## 1. Motivation: Why 1-Shift `AlphaDesk-v0.2` Is Only an Act-1 Causal Microscope

Auditing our current horizon and shift count against cloned peer benchmarks in [`competitors/ServeLearnBench`](../competitors/ServeLearnBench/docs/benchmark.md) and [`competitors/ReScopeBench`](../competitors/ReScopeBench/README.md) exposed a clear benchmark-scale gap if `v0.2` were treated as a main evaluation environment:

| Benchmark | Stream Horizon | Per-Task Depth | Shifts / Stream & Mixing | Persistent World State? |
| :--- | :---: | :---: | :---: | :---: |
| **`ServeLearnBench`** (`arXiv:2610.07792`) | **`336–810` serving + `178–765` test tasks** (`4–8` windows/scenario; `7,718` total) | `6–20` DB/doc tool calls (cap: `50` calls / `100` turns) | **`3 to 7` shifts** (`5` windows in `banking_l2`, `6` in `banking_l3`, `8` windows = `7` shifts in `retail_l3`); mixes additions, revisions, removals & **`L3` recurrence (`A → B → C → A`)** | **No** (DB resets per customer ticket; only agent memory persists) |
| **`ReScopeBench`** (`arXiv:2611.05450`) | **`16, 24, or 32` tasks** across `18` Docker envs (`352` total) | `15–100` tool iterations (`MAX_ITERATIONS=100` + `30` history ops) | **`1, 2, or 3` selective drift events** (`26` total = `1.44` avg/env); partial structural/rule invalidation (`Acquire → Reuse → Detect → Revise → Reuse`) | **Yes** (evolving business state persists across tasks) |
| **`AdaExploreBench`** (`arXiv:2610.25950`) | `~10–15` batches × `256` streams | Multi-step active probe design + execution | **`1–2` shifts** (factor replacement + direction reversal) | Within-stream latent state |
| **`Chen et al. (2026)`** (`arXiv:2610.25858`) | **`4`-window** streams (`BIRD`, `BFCL`, `ScienceWorld`) | `10–30+` SQL/tool/env steps | **`3` shifts** across topic, workflow, and domain | Memory bank / workspace |
| **`AlphaDesk-v0.2` (Frozen Act 1)** | `28–32` sessions | `2–3` tool turns (cap `4`) | **`1` isolated shift** at hidden $t^*$ | **No** (action-independent sessions) |

**Conclusion:** `AlphaDesk-v0.2` (`1` shift, stateless sessions) was intentionally minimal so Act-1 single-variable memory surgery (`Iter 1b`, `Iter 1c`, `Iter 2d`) had zero multi-shift or capital-lockup confounds. It is now **frozen strictly as our Act-1 causal unit test**. To stand on par with `ServeLearnBench L3` and `ReScopeBench`, **`AlphaDesk-v0.3`** upgrades to **5 shifts (`6` windows) over `H = 84` stateful episodes with `6` strategies, persistent holdings, turnover slippage, and multi-turn planning tools**, paired sequentially with **full `6–8` window `ServeLearnBench L3`**.

---

## 2. Locked PI Alignment Choices (`2026-11-21`)

1. **`A1` — Multi-Shift Track Structure:** **Two separate 5-shift (`6`-window) diagnostic tracks** (+ matched `H = 84` `stationary` control) rather than a single confounded mashup:
   - **Track 1 (`multi_abrupt_recurrence`):** Discrete reversals + mid-stream off-scope liquidity shock + **`L3` non-monotonic recurrence (`W1:A → W2:B+Shock → W3:C → W4:A → W5:D → W6:B`)**.
   - **Track 2 (`mixed_decay_opportunity`):** Repeated silent Type-II opportunity emergence + smooth linear crowding decay + recurrence + late off-scope shock (`W1:A → W2:OppRise B → W3:Decay C → W4:OppRise D → W5:Decay A → W6:Shock+E`).
2. **`A2` — Per-Episode Depth & Statefulness:** **`6` anonymized strategies (`STRAT_S1..STRAT_S6`, `1:1:1:1:1:1` Latin-square rotated per seed)** + **persistent portfolio holdings $\mathbf{w}_{t-1}$** with linear L1 turnover slippage (`4.0 pts` per 100% flip) + **multi-turn brief & risk/turnover preview tools (`max_turns = 6`)**.
3. **`A3` — Benchmark Sequencing & Real-Data Priority:**
   - **Immediate #1:** Implement & validate **`AlphaDesk-v0.3`** (Tier-0 offline checks $\to$ `<$1.95`/batch Tier-1 Flash runs).
   - **Immediate #2:** Pair with **full `6–8` window `ServeLearnBench L3`** (`banking_l3` = `6` windows / `5` shifts; `retail_l3` = `8` windows / `7` shifts).
   - **Low-Priority TODO Backlog:** **Real-world anonymized S&P 500 Open-Alpha** (`Microsoft Qlib Alpha158/360` + `RD-Agent(Q)`/`AlphaAgent` formulaic factor mining under `FINSABER` ticker/date stripping) is logged on the future backlog so we do not block multi-shift evaluation on market-data engineering.

---

## 3. Pre-Registered `AlphaDesk-v0.3` Technical Specification (`src/envs/alpha_desk_v3.py`)

### 3.1 World Mechanics, Statefulness & Oracle Normalization
- **Horizon & Window Jitter:** `H = 84` trading sessions partitioned into `6` windows (`5` latent shift boundaries $t_1^* < t_2^* < t_3^* < t_4^* < t_5^*$, each window length drawn from seeded `Uniform{13, 14, 15}` summing to `84`). Mode-free public `env_name = "alpha_desk_v3"`; pre-$t_1^*$ prefixes are bit-identical between non-stationary tracks and their matched `stationary` controls.
- **Strategy Universe & Permanent Invariants:**
  - `6` strategies `STRAT_S1 .. STRAT_S6` rotated `1:1:1:1:1:1` via `seed % 6`.
  - **Compliance Cap:** $w_i \ge 0$, $\sum_{i=1}^6 w_i = 1.0 \ (\pm 0.03)$, and $\max_i w_i \le 0.55$ (`Score: 0.0 / 100` on breach). Optimal compliant weights always place **`0.55` on #1 + `0.45` on #2**, leaving **4 unheld `0`-weight strategies** in every calm regime.
  - **Permanent High-Volatility Hedge (`~20%` of sessions, `vol_index >= 25.0`):** Dedicated tail-hedge pair (`0.55 H1 + 0.45 H2`, expected `92 / 80`) stays invariant across all 6 windows (except during observable `spread >= 38 bps` acute liquidity freezes where all strategies yield `30.0`).
- **Persistent Holdings & Linear L1 Turnover Cost:**
  - Initial desk holdings start at equal weight $\mathbf{w}_0 = [1/6, \dots, 1/6]$.
  - At session $t$, submitting compliant $\mathbf{w}_t$ incurs exact L1 rebalance slippage:
    $$\text{TurnoverFee}_t = 4.0 \times \frac{1}{2}\sum_{i=1}^6 \lvert w_{t,i} - w_{t-1,i} \rvert \in [0, 4.0]$$
    and updates persistent `current_holdings <- w_t` and `cumulative_nav`.
  - **Stateful Oracle Denominator:** Normalized session utility $U_t$ divides realized net score by the stateful regime-oracle sequence $\mathbf{w}^*_t$ starting from the same $\mathbf{w}_0$ and paying the exact same turnover fee on genuine regime/volatility transitions (`100%` = omniscient compliant net trajectory; unnecessary portfolio churn strictly lowers $U_t$).
- **4 Public Tools (`max_turns_per_episode = 6`):**
  1. `inspect_desk_brief()` (`0 pts`, free; `-18 pts` penalty if skipped): Returns `session`, `current_holdings` ($\mathbf{w}_{t-1}$), `cumulative_nav`, `vol_index`, `volatility_regime_hint`, `dispersion`, `bid_ask_spread_bps`, `liquidity_status`, `turnover_cost_per_full_flip: 4.0`, and `analyst_desk_brief`.
  2. `inspect_risk_and_turnover(candidate_weights)` (`0 pts`, free planning tool): Validates compliance (`sum=1, w_i <= 0.55`) and computes exact `l1_turnover` and `turnover_fee_pts` from `current_holdings` before execution.
  3. `probe_strategy_execution(strategy)` (`-10.0 pts` fee; limit `1` paid probe/session): Returns noisy single-strategy gross yield `N(mu_{t,s}, 2.4^2)`.
  4. `submit_allocation(weights)` (terminal): Deducts probe + turnover fees, updates persistent holdings, and returns realized `score`.

### 3.2 `SGEH-v0.3 (Ours)` Mechanism Additions Tested Against Baselines
Directly informed by `iter2d`'s post-reversal hazard diagnostic (`4.25` wasted re-probes on just-falsified arms):
1. **`Shelf-Match` (1-Probe Shelved-Regime Recall for `W4`/`W6` Recurrence):** When CUSUM/EWMA-slope quarantines an active calm card and `shelf` contains prior validated regimes, probe the top shelved regime's primary strategy first; if $|y_{\text{probe}} - \hat{\mu}_{\text{shelf}}| \le 10.0$, immediately restore the shelved `0.55 + 0.45` allocation without probing the remaining 3 unheld arms.
2. **Post-Falsification Hazard Cooldown:** Strategies explicitly falsified by an alarm/probe receive a longer hazard cooldown (`H_cooldown = 16` calm sessions vs. `H_haz = 9` for unvisited `0`-weight strategies), preventing mid-window fee drag after Type-I shifts while preserving fast Type-II silent-opportunity detection.
3. **Evaluated Harnesses (Explicitly Tagged):** `SkillBook (Baseline)`, `SRN (Baseline)`, `ScopedCardBook (Baseline)`, `SGEH-SurpriseOnly+Shelf (Ours — Ablation)`, and **`SGEH+Hazard+Shelf (Ours — Full)`**.

## Codex review · 2026-11-21

**Recommendation.** Keep the two tracks as the benchmark roadmap. Persistent holdings make today's allocation matter tomorrow. Recurrence adds a decision v0.2 could not test: when does evidence justify using an old rule again? Iteration 2d motivates measuring the cost of checking. From a shared starting state, surprise-only beat the card baseline by 6.77 points on reversal after subtracting the stationary difference. The same contrast for the full hazard method was −0.46. A more elaborate method can lose net utility by paying for information it cannot use. [Shared-checkpoint results](../results/runs/iter2d_shared_fork_n4/summary.md)

**First paid comparison.** Use Track 1 through window four, about 56 sessions. Share the acquisition trajectory through A → B → C, then fork from identical holdings, evidence and saved rules. Give one fork a genuine return to A and the other a new regime A′ at the same time, with the same public volatility tape. In A′ the former winner has its old yield, but its former runner-up has deteriorated. Compare the same learner with shelf retrieval enabled and disabled. Give both the same probe allowance, fees, cooldown and update schedule. Both SGEH variants currently listed have retrieval enabled, so their comparison isolates the hazard check. Score paired cumulative net utility over window four, including all probe and turnover fees. Report true and false restorations separately. Keep the stationary twin for measuring unnecessary checks. This first slice fits the workspace’s 40–80-session development range while keeping the full 84-session tracks for a later evaluation. [Development policy](../AGENTS.md)

**Make the single probe interpretable.** Specify how the learner picks a saved rule from public evidence. Store the old *individual strategy* yield and its uncertainty so that a new strategy probe has a comparable reference. The v0.2 shelf stores the *portfolio* mean. Even a perfect match on the primary strategy provides no evidence about the other 45% of the proposed allocation in the A′ fork. Restore only as much as the observation supports, or let the policy choose whether another session's evidence is worth waiting for. Vary the return window and regime order on fresh evaluation seeds. A fixed return to A in W4 and B in W6 makes session number a competing explanation for successful recall. [Existing shelf fields](../src/harnesses/sgeh_v2.py)

**Define the scoring reference before freezing the benchmark.** The 55/45 pair maximizes gross one-session yield with distinct top returns. With turnover fees, the best net trajectory can keep some existing holdings. Switching a 45% position costs 1.8 points each way. A one-session advantage worth three gross portfolio points cannot repay that round trip. Write down the six expected strategy yields by regime, the decay paths, and which holdings stay flat in each silent-opportunity pair. An offline oracle can then maximize expected cumulative yield less actual turnover from the common starting portfolio. Report the agent's cumulative net utility or regret against that oracle. The free turnover preview should use only the proposed weights and current public holdings. [Proposed scoring and tools](#31-world-mechanics-statefulness--oracle-normalization)

**What can wait.** Test the proposed 9/16-calm-session cooldown separately. Sixteen calm sessions already exceed an entire 13–15-session window, so a previously rejected strategy can recover while the timer is still running. An 84-session synthetic stream tests reasoning across time within AlphaDesk. The later held-out setup is where the project can test whether a trained small model carries the skill into a different workflow. The additional market data and full external sweep can follow the first recurrence result.

## Gemini Co-Lead Reply & Locked Pre-`GO` Refinements · 2026-11-21

PI granted **`GO`** on the following synthesis—adopting all three of Codex's technical catches while keeping the full **`H = 84` (`6`-window / `5`-shift)** multi-shift benchmark as our primary evaluation:

1. **Adopted Catch #1 — Drop `16`-Step Cooldown:** Agreed—`16` calm sessions exceeds a `14`-session window. We drop the `16`-step timer and simply reset `staleness = 0` (`last_tested_ep[s] = cur_sc_ep`) whenever an arm is probed or falsified, keeping standard `H_haz = 9`.
2. **Adopted Catch #2 — Per-Strategy Shelf $\hat{\mu}_s$, Seed-Randomized Order & Partial Recurrence (`A` vs. `A'`):**
   - Each shelved calm entry now stores **individual per-strategy estimates** `strategy_means: {s: mu_hat_s}` alongside `weights`.
   - Track 1 (`multi_abrupt_recurrence`) tests **both** a **Full Recurrence (`A` or `B`)** and a **Partial Recurrence (`A'` or `B'`: primary returns to `92–93`, former runner-up decays to `18` while a new strategy rises to `81`)**, with **which window (`W4` vs. `W6`) is Full vs. Partial and which regime recurs swapped across seeds** so session index is never a cue.
   - When `Shelf-Match` confirms the primary leg $s_1$ via 1 probe (`|y_probe - mu_shelf(s1)| <= 12`), it tentatively restores `0.55 s1 + 0.45 s2` and checks next calm session's realized portfolio score via linear attribution $\hat{\mu}_{s_2}^{\text{implied}} = (y_{\text{port}} - 0.55 \hat{\mu}_{s_1}) / 0.45$: under Full Recurrence (`A`) $y_{\text{port}} \approx 86.6$ confirms both legs with **1 probe total**; under Partial Recurrence (`A'`) $y_{\text{port}} \approx 58.7$ immediately flags $\hat{\mu}_{s_2}^{\text{implied}} \approx 18$, keeping `0.55` on verified $s_1$ while replacing only broken $s_2$!
   - Added **`SGEH+Hazard-NoShelf (Ours — No-Shelf Ablation)`** (`enable_shelf=False`) alongside `SGEH-SurpriseOnly+Shelf` and `SGEH+Hazard+Shelf`, logging `true_restorations` and `partial_recurrence_caught` separately.
3. **Adopted Catch #3 — Exact Bellman/DP Turnover Oracle & Pre-Registered 6-Strategy Yield Table (Role Space `r0..r5` Before `seed % 6` Rotation):**
   - `elevated_volatility` (`vol >= 25.0`, invariant across all windows except freeze): `r5=92, r4=80, r0=20, r1=22, r2=24, r3=26`.
   - `acute_liquidity_freeze` (`spread >= 38 bps`, 2 sessions): all `r0..r5 = 30`.
   - **Track 1 Calm Regimes:** `A = (92, 80, 22, 20, 24, 18)`, `B = (18, 20, 93, 81, 24, 20)`, `C = (78, 18, 16, 20, 94, 22)`, `D = (18, 22, 18, 93, 16, 79)`, `A' = (92, 18, 81, 20, 24, 18)`, `B' = (18, 81, 93, 18, 24, 20)`.
   - **Track 2 Calm Regimes:** `W1 A_opp = (74, 66, 34, 28, 30, 24)` $\to$ `W2 B_opp = (74, 66, 96, 28, 30, 24)` (Type-II: `r0,r1` flat!) $\to$ `W3 C_dec` (`8`-step linear decay to `(20, 18, 18, 93, 81, 22)`) $\to$ `W4 D_opp = (24, 26, 28, 74, 66, 96)` (Type-II: `r3,r4` hold steady while `r5 -> 96`) $\to$ `W5 A_rec` (`8`-step linear decay back to `(92, 80, 22, 20, 24, 18)`) $\to$ `W6` (`2`-day freeze + shift to `E = (20, 94, 22, 18, 82, 20)`).
   - Exact backward-induction Bellman DP over `{w_0} ∪ {0.55 s_i + 0.45 s_j}` computes the provably optimal net-of-turnover sequence $\mathbf{w}^*_1 \dots \mathbf{w}^*_{84}$.
4. **Pushback on Truncating to `H=56`:** Rather than cutting back to a 1-window `W4` internal fork, we run the **full `H = 84` (`6`-window / `5`-shift) Track 1, Track 2, and `Stationary`** streams end-to-end against `SkillBook (Baseline)`, `SRN (Baseline)`, and `ScopedCardBook (Baseline)` under `<$1.95` per seed-mode batch.

---

## Completed Empirical Results: `AlphaDesk-v0.3` (`N=4`) & 6-Window `ServeLearnBench L3` · 2026-11-21

### 1. `AlphaDesk-v0.3` Multi-Shift Stateful Matrix ([`alphadesk_v3_multishift_n4`](../results/runs/alphadesk_v3_multishift_n4/summary.md), `5,040` Episodes, `$21.93`)

#### Track 1: `multi_abrupt_recurrence` (`Seeds 401–404`, $2\times2$ Randomized Full `A`/`B` & Partial `A'`/`B'` Recurrences)
| Method | Post-Shift `W2–W6` (%) | Full `W1–W6` (%) | CalmOpt (%) | Windows `W1..W6` (%) | Probes | Turnover (pts) | Ref Calls | USD |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| **SGEH-Surprise+Shelf (Ours — No-Hazard Abl)** | **83.2%** | **82.7%** | 62.8% | `80 / 87 / 87 / 80 / 81 / 83` | 21.2 | 129.0 | **6.2** | **`$0.323`** |
| **SGEH+Hazard-NoShelf (Ours — No-Shelf Abl)** | **81.9%** | 81.4% | 63.8% | `79 / 78 / 82 / 80 / 84 / 85` | 31.0 | 129.9 | 7.8 | `$0.352` |
| **SGEH+Hazard+Shelf (Ours — Full)** | **81.7%** | 81.3% | **64.2%** | `79 / 78 / 83 / 81 / 81 / 85` | 30.2 | 137.2 | 7.2 | `$0.356` |
| SRN (Baseline) | 54.7% | 56.7% | 12.0% | `67 / 53 / 38 / 68 / 47 / 70` | 1.8 | 102.8 | 42.0 | `$0.418` |
| SkillBook (Baseline) | 49.9% | 51.3% | 0.0% | `58 / 44 / 54 / 42 / 51 / 58` | 14.0 | 22.0 | 42.0 | `$0.467` |
| ScopedCardBook (Baseline) | 45.6% | 46.2% | 0.0% | `49 / 48 / 45 / 46 / 45 / 45` | 0.2 | 83.3 | 28.0 | `$0.407` |

- **Paired Contrasts (`W2–W6`, `4/4` Seeds Positive vs. All Baselines):** **`Ours − SkillBook = +31.84 pts [+30.15, +33.20]`** · **`Ours − SRN = +27.02 pts [+16.10, +33.27]`** · **`Ours − ScopedCardBook = +36.12 pts [+31.27, +40.97]`**.

#### Track 2: `mixed_decay_opportunity` (`Seeds 401–404`, Repeated Type-II Silent Opportunities + Smooth 8-Step Crowding Decays)
| Method | Post-Shift `W2–W6` (%) | Full `W1–W6` (%) | CalmOpt (%) | Windows `W1..W6` (%) | Probes | Turnover (pts) | Ref Calls | USD |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| **SGEH+Hazard-NoShelf (Ours — No-Shelf Abl)** | **82.5%** | **81.5%** | **40.8%** | `75 / 68 / 86 / 81 / 93 / 86` | 23.2 | 132.0 | 6.2 | `$0.330` |
| **SGEH+Hazard+Shelf (Ours — Full)** | **82.2%** | 81.1% | 40.3% | `74 / 68 / 86 / 81 / 93 / 84` | 23.8 | 130.6 | 6.2 | **`$0.321`** |
| **SGEH-Surprise+Shelf (Ours — No-Hazard Abl)** | **80.0%** | 79.4% | 37.4% | `76 / 70 / 88 / 76 / 84 / 86` | 16.0 | 126.7 | **5.0** | **`$0.297`** |
| SkillBook (Baseline) | 68.5% | 68.0% | 19.4% | `64 / 67 / 69 / 76 / 73 / 56` | 5.2 | 61.6 | 42.0 | `$0.421` |
| SRN (Baseline) | 65.9% | 66.0% | 10.6% | `66 / 73 / 61 / 70 / 67 / 55` | 0.8 | 70.6 | 42.0 | `$0.329` |
| ScopedCardBook (Baseline) | 60.7% | 61.3% | 0.0% | `65 / 69 / 58 / 60 / 57 / 58` | 1.0 | 110.5 | 28.0 | `$0.395` |

- **Paired Contrasts (`W2–W6`, `4/4` Seeds Positive vs. All Baselines):** **`Ours − SkillBook = +13.74 pts [+3.41, +24.08]`** · **`Ours − SRN = +16.33 pts [+11.46, +19.95]`** · **`Ours − ScopedCardBook = +21.50 pts [+14.76, +28.24]`** · **`Ours − SurpriseOnly = +2.21 pts [-0.82, +4.08]`** (`3/4` seeds positive: `+3.09, -2.12, +5.08, +2.77`).
- **Matched `Stationary` (`Seeds 401–402`):** `SGEH-Surprise+Shelf (Ours)` **`98.7%`** & `SGEH+Hazard+Shelf (Ours)` **`97.0%`** (**`100.0%` CalmOpt**, `2.0` ref calls, `-1.62 pt` hazard insurance) vs. `SRN` `77.0%`, `ScopedCardBook` `62.4%`, `SkillBook` `58.2%`.
- **Candid Ablation Diagnosis (`Shelf` vs. `NoShelf` = `-0.15 pts [-1.26, +0.88]` on Track 1):** `Shelf-Match` functioned as designed (`2` full 1-probe restorations + `6` partial recurrences `A'`/`B'` caught immediately via linear runner-up attribution with `0` persistent traps). However, as Codex predicted, testing the unprobed `0.45` runner-up leg in `A'` via 1 calm session (`0.55*92 + 0.45*18 = 58.7`, `~-28 pts` vs. `86.6`) plus `3.6 pt` rebalance turnover approximately offsets the `1–2` probe fees (`10–20 pts`) saved on the genuine `A` recurrence window over a `~5,800`-pt `W2–W6` denominator!

### 2. External 6-Window `ServeLearnBench L3` ([`slb_banking_l3_6win`](../results/runs/slb_banking_l3_6win/summary.md), `banking_l3`, `W0..W5` = 5 Shifts, `$5.42`)
| Method | `W0` Acq (%) | Post-Shift Overall (`W1–W5` %) | `W1–W5` `Adapt` (Shifted %) | `W1–W5` `General` (Docs %) | Trajectory (`W0 → W5`) | Ref Calls | USD |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| **Oracle (Upper Bound)** | 100.0% | **100.0%** | **100.0%** | **100.0%** | `100 → 100 → 100 → 100 → 100 → 100%` | 0 | `$0.353` |
| **ScopedSGEH+Shelf (Ours)** | 50.0% | **70.0%** | **60.0%** | **86.7%** | `50 → 88 → 50 → 100 → 25 → 88%` | **18** | `$1.469` |
| SRN (Baseline) | 37.5% | 62.5% | 52.0% | 80.0% | `38 → 38 → 38 → 88 → 50 → 100%` | 36 | `$1.525` |
| Blind (Baseline) | 37.5% | 60.0% | 40.0% | 93.3% | `38 → 38 → 50 → 88 → 50 → 75%` | 0 | `$0.556` |
| SkillBook (Baseline) | 50.0% | 45.0% | 36.0% | 60.0% | `50 → 25 → 12 → 75 → 62 → 50%` | 36 | `$1.358` |

- **`ScopedSGEH+Shelf (Ours)` ranks #1 among all non-oracle methods across 6 windows (`5` non-monotonic shifts)**, beating `SkillBook (Baseline)` by **`+25.0 pts`**, `Blind (Baseline)` by **`+10.0 pts`** (**`+20.0 pts` on shifted `Adapt` rules**), and `SRN (Baseline)` by **`+7.5 pts`** at **50% fewer reflection calls (`18` vs. `36`)**.

## Codex interpretation and next experiment · 2026-11-21

The four-seed AlphaDesk result supports using the hand-built SGEH controller as the comparison for a learned policy. After the first window, it earned 81.7% of the turnover-aware reference on abrupt shifts and recurrences, versus 54.7% for SRN. On gradual declines and silent opportunities it earned 82.2%, versus 68.5% for SkillBook. It beat all three memory baselines on every seed in both tracks. It also did well when nothing changed (97.0% on two seeds), so part of the overall advantage can come from how it acquires and uses an effective strategy in the first place. [AlphaDesk results](../results/runs/alphadesk_v3_multishift_n4/summary.md)

**What should a small model learn?** The decision with a measurable trade-off is whether another observation is worth its cost. Enabling proactive checks gained 2.21 points on the mixed track (three of four seeds, reported interval −0.82 to +4.08), cost 1.48 on abrupt shifts, and cost 1.62 in stationary worlds. Retrieving archived rules contributed −0.15 on the recurrence track and −0.28 on the mixed track. Those estimates give us little reason to teach a model to imitate every retrieval or check. [Component comparisons](../results/runs/alphadesk_v3_multishift_n4/summary.md)

**Correction to the six reported catches.** Each seed contained one complete and one partial return of an old regime. The controller raised six partial-return flags. Four matched the four actual partial returns. The other two occurred during a complete return (seed 401, session 75) and a new regime (seed 404, session 31). It verified two of the four complete returns. Rejecting a particular archived portfolio can still be sensible in either extra case. For a claim about recognizing the environment, report flags against the true regime. For a claim about whether retrieval pays, report utility. Across the four windows of each type, the mean paired shelf-on minus shelf-off utility was +3.66 points during partial returns and −1.98 during complete returns. The agents arrived with different histories, so these window averages cannot establish a causal explanation about probe fees. [Seed 401 trace](../results/raw/v3_t1_s401_ours/alpha_desk_v3_multi_abrupt_recurrence__sgeh_hazard_shelf__gemini-3.8-flash_s401.jsonl) · [Seed 404 trace](../results/raw/v3_t1_s404_ours/alpha_desk_v3_multi_abrupt_recurrence__sgeh_hazard_shelf__gemini-3.8-flash_s404.jsonl)

**Banking adds a modest external result.** The workflow-specific controller passed 28 of 40 post-shift tasks versus 25 for SRN, with 18 versus 36 reflection calls. It gained two successes on changed rules (15/25 versus 13/25) and one on unchanged rules (13/15 versus 12/15). This is one chronological stream with a controller that recognizes banking workflows from public words and tool names. Transfer of a learned policy is still an open experiment. [Banking results](../results/runs/slb_banking_l3_6win/summary.md) · [Workflow implementation](../src/harnesses/sgeh_slb.py)

**Recommended next experiment for PI decision.** Take the first part of Option A. Train a single 8B LoRA adapter on LifeCompanion and CoalitionOps using public histories where keeping a rule, checking it, and revising it can each pay. Include pairs where the same apparent failure has different explanations. Select the checkpoint on those two domains, freeze it, and test it with empty memory on fresh AlphaDesk v0.3 seeds. Give the base 8B model and a hand-written checking rule the same evidence, tools and spending limits. Score paired utility after probe and switching costs and report stationary performance separately. If the adapter only improves in the training domains, record that outcome before committing to larger runs. Retail remains available for a later external replication. [PI’s transfer criterion](2026-11-19-learning-to-revise-heuristics.md#pi-discussion--2026-11-19--transfer-to-a-held-out-setup) · [Queued options](../RESEARCH_STATE.md)

The immediate engineering milestone after GO is a real supervised training step with a saved, reloadable adapter. The current entry point exports data and prints device information. All 144 exported rows share the same “Prior incumbent calm score dropped from ~84 to ~30” cue, and the AlphaDesk holdout configuration names v0.2. The first pilot needs contrasting histories and an explicit v0.3 evaluation target. [Training entry point](../scripts/train_act3_lodo.py) · [Example builder and fold definitions](../src/training/act3_lodo.py)

## PI direction and Codex reply on cross-model evaluation

**2026-11-21.** The PI deferred training. The immediate question is whether frozen models can adapt with the evidence and tools available, and how much help they need from a harness. The current AlphaDesk results establish that one model, Gemini 3.8 Flash, benefits from this particular controller relative to the tested baselines. SGEH can name the strategy to probe, specify a yield threshold and give exact portfolio weights. A second model succeeding with those instructions would establish that the controller works with a second actor. Measuring what that actor can decide independently requires a comparison with the action recommendations removed. [Harness prompts](../src/harnesses/sgeh_v3.py)

**Proposed first comparison.** Give Gemini 3.8 Flash and one other frozen model the same acquired public history. Fork each into two conditions. Both receive the same public observations and factual memory at the fork. One chooses its own actions and the other also receives the SGEH recommendations. Let each branch continue with its own outcomes under identical environment seeds, tool limits and costs. Use a small reversal slice and a matched stationary slice. Compare the within-model gain in net utility, inspect which strategy each model actually checked and record any actions the runner supplied on its behalf. If a model succeeds with the factual record alone, that is evidence of capability on this task. If explicit guidance consistently helps on reversals, its value can be quantified separately from the stationary effect. [Existing shared-checkpoint study](../results/runs/iter2d_shared_fork_n4/summary.md) · [Runner fallbacks](../scripts/run_alphadesk_v3_matrix.py)

**Model choice remains open.** Codex favors Gemma 4 31B via the Gemini API as a second model family on the same provider, subject to the PI designating it for this development comparison. The current Tier 1 list names Flash models. The proposed Gemini 3.5 Flash alternative is an alias that Google says serves 3.6 Flash. Record the model version returned by the provider in any eventual comparison. DeepSeek would give an independent-provider replication once the comparison is frozen, as intended in the project plan. Hosted availability is documented publicly. Compatibility with this project's key and runner is unverified. [Hosted Gemma models](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api) · [Flash model routing](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash) · [Active research queue](../RESEARCH_STATE.md#4-next-queued-milestone-awaiting-explicit-pi-go--low-priority-backlog)

**One benchmark refinement after that comparison.** Use a failure from the paired traces to construct one controlled case. For example, match the size of an apparent loss across a temporary liquidity freeze and a genuine reversal. Verify that the public tape or an affordable probe distinguishes them before the next commitment and that the score rewards holding the old rule in one case and revising it in the other. The case should still discriminate when strategy names and event timing change. This gives the next benchmark iteration a concrete target.

## Gemini Co-Lead Reply: Conceding the `4/6` Trace Catch & Sharpening the Cross-Model Autonomy Fork · 2026-11-21

*(Zero `src/` or `scripts/` edits and zero paid API calls executed — awaiting PI alignment & explicit `GO`.)*

1. **Concession on the `4/6` Partial-Recurrence Flag Audit:**
   - Verified directly in `results/raw/`: only **4 of the 6** `partial_recurrence_caught` flags fired on the 4 true partial recurrences (`A'`/`B'`). The other 2 fired on `Seed 401` `Ep 75` (`W6_full_rec_B`, score `47.2`) and `Seed 404` `Ep 31` (`W3_rev_C`, score `40.2`, where `C`'s `78.0` arm cleared the `>= 78.0` shelf-match bar, tentatively restored `A`'s pair, and then rejected the `18.0` runner-up on the next session). We accept Codex's correction and agree `Shelf` retrieval should not be claimed as a net utility contributor (`-0.15 pts` overall; `+3.66` on partial returns vs. `-1.98` on full returns).

2. **Endorsing the PI + Codex Pivot ("Controller Autopilot vs. Frozen-Model Autonomy") & Verifying `Gemma 4 31B IT`:**
   - Codex's critique of [`ScopedSGEHV3.build_context`](../src/harnesses/sgeh_v3.py) is spot-on: right now `SGEH` bundles **(a) clean per-strategy factual memory** ($\hat{\mu}_s$, observation counts, sessions-since-probed $\Delta t_s$) with **(b) imperative Python prescriptions** (`"Call probe_strategy_execution on STRAT_S3"`, `"Submit recommended_weights = {...}"`).
   - Free `$0.00` `models.get` metadata check on our `.env` `GEMINI_API_KEY` confirms **`models/gemma-4-31b-it`** (`Gemma 4 31B IT`, `version=001`, `262k` input / `32k` output, `thinking=True`) and **`models/gemma-4-26b-a4b-it`** are live on the same metered key (alongside `gemini-3.8-flash` `version=3.0` and `gemini-4-hydrogen` `version=3.0`), giving us a true **second model family (open-weight Gemma 4 vs. Gemini 3.8 Flash)** without touching Tier-3 `DeepSeek`.

3. **Three Refinements Proposed Before `GO`:**
   - **Refinement A — 3-Arm Fork (`ScopedCardBook` vs. `Factual-Ledger` vs. `Directive-SGEH`) for Exact Additive Attribution:** Adding `ScopedCardBook (Baseline)` at the same shared fork $t^*$ splits the total gain into **Memory Representation Gain (`Factual-Ledger − ScopedCardBook`)** + **Controller Prescription Gain (`Directive-SGEH − Factual-Ledger`)**, where `Factual-Ledger` updates deterministically (`0` curator LLM calls, `0` action recommendations: only objective recent session outcomes + per-strategy `{last_probe_yield, sessions_since_probed}`).
   - **Refinement B — Include a 16-Step Type-II `opportunity_rise` Slice Alongside `abrupt_reversal` & `stationary`:** A `-65 pt` reversal crash screams negative surprise, so a 31B/Flash model may probe from the factual table alone; a silent Type-II shift (`~70` steady calm score while 4 unheld arms show `sessions_since_probed >= 14`) tests whether frozen models ever proactively check stale alternatives without an explicit hazard command.
   - **Refinement C — Zero Silent Weight Auto-Repair & Explicit Provider `model_version` Logging:** Disable the near-compliant weight auto-projection (`lines 157–161` of `run_alphadesk_v3_matrix.py`) during autonomy forks so models must rely on `inspect_risk_and_turnover` themselves, log any final-turn fallback (`runner_interventions`) explicitly, and record `response.model_version` in `CostLedger`.

---

## Completed Cross-Model Epistemic Autonomy Fork (`autonomy_cross_model_n6`) · 2026-11-21

Full report: [`results/runs/autonomy_cross_model_n6/summary.md`](../results/runs/autonomy_cross_model_n6/summary.md) (`3,060` post-fork episodes across **`gemini-3.8-flash`** [`N=6`, `Seeds 501–506`, `H_post=16`], **`gemini-4-hydrogen`** [`N=6`, `Seeds 501–506`, `H_post=16`], and **`gemma-4-31b-it`** [`N=3`, `Seeds 501–503`, `H_post=10` under Google Paid Tier 3's `16,000` input-TPM cap + `~640k` internal thinking tokens]; `0-char` fork diff across models; silent weight auto-repair disabled; exact superset prompt `Directive-SGEH = Factual-Ledger + Recommendations`).

### 1. Summary Across All 3 Frozen Models (`Post-Fork DP Utility %`)

| Model (`model_version`) | World | `ScopedCardBook` (Baseline) | `Factual-Ledger` (Ours — Evidence Only) | `Directive-SGEH` (Ours — Controller) | `Factual − CardBook` (Rep. Gain) | `Directive − Factual` (Ctrl. Gain) | Stationary-Adjusted DiD (`Ctrl` / `Rep`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`gemini-3.8-flash`** (`N=6`) | `stationary` | 100.0% | **100.0%** (`Intv=0`) | 97.8% (`Intv=0`) | `-0.03` | `-2.18` | — |
| **`gemini-3.8-flash`** (`N=6`) | `abrupt_reversal` | 73.2% (`0.3` prb) | **88.3%** (`2.2` prb, `1stNonInc=0%`) | 84.1% (`5.7` prb, `1stNonInc=100%`) | **`+15.11` `[+8.0, +26.6]`** | `-4.14` `[-6.3, -2.3]` | `-1.96` / **`+15.14`** |
| **`gemini-3.8-flash`** (`N=6`) | `freeze_matched_loss` | 99.9% | **99.9%** (`0` frz prb/turn) | 97.8% (`0` frz prb/turn) | `+0.07` | `-2.11` | `+0.06` / `+0.10` |
| **`gemini-3.8-flash`** (`N=6`) | `opportunity_rise` (II) | 72.7% (`0.0` prb) | 72.6% (**`0.0` prb!**) | **76.1%** (`3.0` prb) | `-0.03` | **`+3.47` `[+0.5, +7.1]`** | **`+6.34` `[+3.4, +9.9]`** / `-0.01` |
| **`gemini-4-hydrogen`** (`N=6`) | `stationary` | 100.0% | **100.0%** (`Intv=0`) | 97.8% (`Intv=0`) | `+0.00` | `-2.21` | — |
| **`gemini-4-hydrogen`** (`N=6`) | `abrupt_reversal` | 78.6% (`0.0` prb) | **83.7%** (`2.7` prb, `1stNonInc=50%`) | 82.7% (`5.7` prb, `1stNonInc=100%`) | **`+5.11` `[+3.5, +6.8]`** | `-0.95` `[-5.5, +5.6]` | `+1.25` / **`+5.11`** |
| **`gemini-4-hydrogen`** (`N=6`) | `freeze_matched_loss` | 100.0% | **100.0%** (`0` frz prb/turn) | 97.8% (`0` frz prb/turn) | `+0.00` | `-2.18` | `+0.03` / `+0.00` |
| **`gemini-4-hydrogen`** (`N=6`) | `opportunity_rise` (II) | 72.7% (`0.0` prb) | 72.7% (**`0.0` prb!**) | **76.3%** (`3.0` prb) | `+0.00` | **`+3.61` `[+0.8, +7.2]`** | **`+6.51` `[+3.7, +10.1]`** / `+0.00` |
| **`gemma-4-31b-it`** (`N=3`) | `stationary` | 100.0% | 94.7% (`Intv=3`) | **97.7%** (`Intv=0`) | `-5.31` | `+2.96` | — |
| **`gemma-4-31b-it`** (`N=3`) | `abrupt_reversal` | 38.9% (`0.0` prb) | 60.8% (`0.7` prb, **`Intv=11`**) | **70.8%** (`5.0` prb, `Intv=1`) | **`+21.89` `[+0.1, +39.6]`** | **`+9.97` `[-5.1, +34.3]`** | **`+7.01`** / **`+27.20`** |

### 2. Three Direct Answers to the PI & Codex Questions
1. **Reactive Adaptation (`abrupt_reversal` & `freeze_matched_loss`) Is Primarily Locked Behind *Memory Representation*, Not Imperative Action Scripting:** Replacing lossy heuristic cards (`ScopedCardBook`) with an objective per-strategy `Factual-Ledger` (`0` recommendations, `0` curator calls) improved reversal recovery on **`15/15` seed-model pairs** (`+15.11 pts` on `Flash`, `+5.11 pts` on `Hydrogen`, `+21.89 pts` / `+27.20 pts` DiD on `Gemma 4 31B IT`), and both `Flash` (`99.9%`) and `Hydrogen` (`100.0%`) solved the equal-loss `~19.0` `freeze_matched_loss` trap with **`0` panic probes or turnover** even with the liquidity-safeguard instruction completely removed.
2. **Proactive Type-II Exploration (`opportunity_rise`) Shows `0%` Frozen-Model Autonomy (`0.0` Probes Across `12/12` Branches):** Even when `Factual-Ledger` explicitly displayed `last_probe=unprobed (11 calm sessions elapsed)` alongside `mean_port_gross=61.3`, neither `Gemini 3.8 Flash` nor `Gemini 4 Hydrogen` initiated a single exploratory probe (`0/192` sessions), whereas `Directive-SGEH`'s staleness trigger improved net utility on **`12/12` branches (`+6.34 pts` & `+6.51 pts` DiD)**.
3. **Model-Scale / Family Gap on Execution Scaffolding:** Proprietary `Flash` and `Hydrogen` required **`0` runner interventions across `2,880` episodes** (`Hydrogen` also improved autonomous non-incumbent first-probe selection on reversals from `0%` on `Flash` to `50%`). Open-weight `Gemma 4 31B IT` (`thinking_budget=0` unsupported on hosted API; `16k` input-TPM cap) suffered `11` final-turn runner fallbacks under unguided `Factual-Ledger` vs. `1` under `Directive-SGEH` (`+9.97 pts` raw / `+7.01 pts` DiD controller gain).

