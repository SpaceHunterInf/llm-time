# Design Decisions, Supervisor Discussion Log & `AlphaDesk-v0.2` Spec

**Date:** 2026-11-19  
**Source:** PI `/grill-me` Alignment Interview + `setup-debates/` Convergence  
**Execution Status:** Specification locked for PI/Codex review. **No code or runs execute until explicit PI `GO`.**

---

## 1. Locked Design Tree Decisions (`D1`–`D7`)

| ID | Design Axis | Agreed Decision | Rationale |
| :--- | :--- | :--- | :--- |
| **D1** | **Primary Paper Contribution (ACL Jan 2027 / ICML 2027)** | **Phenomenon + Learned Meta-Plasticity (`8B`/`27B` LoRA)** | Diagnose the *Stale-Memory Ratchet* causally, use `SGEH` as our hand-built baseline, and make **training an open model (`8B`/`27B` LoRA) to selectively revise its own heuristics & transfer zero-shot across setups** the flagship contribution. |
| **D2** | **Heuristic / Memory Representation** | **Structured Scoped Heuristic Cards** *(provisional v0.2 choice)* | Each card stores `{id, scope_condition, action_rule, predicted_p_high, status: active|shelved}`. Modular enough for clean causal single-card excision and proper Brier forecast scoring, while remaining natural-language so it transfers to `ServeLearnBench` text/DB tasks. *(Logged in §2 for supervisor discussion vs. free-form meta-principles.)* |
| **D3** | **`AlphaDesk-v0.2` Realism & Statefulness** | **Semi-structured natural-language desk briefs + independent sessions in `v0.2`; stateful carryover deferred to `v0.3`** | Replaces v0.1's naked numbers with realistic macro/liquidity desk notes and anonymized strategy IDs (`STRAT_A..D`) so context-vs-surprise is genuinely semantic, while keeping `v0.2` sessions action-independent so the 3-arm causal fork has zero lockup confounds. |
| **D4** | **`AlphaDesk-v0.2` Non-Stationary Regimes** | **All 4 Diagnostic Regimes + Matched Stationary Control** | Includes **(1) Abrupt Reversal**, **(2) Gradual Crowding Decay**, **(3) Type-II Silent Opportunity Rise** (incumbent flat ~70 while alternative rises 40 $\to$ 95), and **(4) Off-Scope Liquidity Shocks** (score drops to ~30 under observable liquidity freeze where rule remains valid), all with **randomized shift onset $t^*$**. |
| **D5** | **Act 3 Training Pipeline (CSD3 & Collaborators)** | **Merged SFT Warmup $\to$ Two-Arm Multi-Turn GRPO (`Arm A/B/C`)** | **Arm A:** LoRA SFT warmup on reference-consistent heuristic revision traces; **Arm B (`SFT -> Outcome-GRPO`):** pure net task utility after probe fees; **Arm C (`SFT -> Epistemic-GRPO`):** net utility + proper Brier calibration score on predicted outcomes. Packaged as turnkey parallel jobs across 3 leave-one-domain-out folds. |
| **D6** | **External Third-Party Benchmark Scope** | **`ServeLearnBench` (`Retail` & `Banking` L1–L3) Primary; `ReScopeBench` Stretch Only** | Pure-Python `ServeLearnBench` provides immediate third-party neutrality + 5 published baselines (`Mem0`, `SkillOpt`, `Continual Harness`, `Prime`, `RAG`) without Docker/arm64 bottlenecks (`CEO-Bench` was a depth analogy only, not a planned eval target). |
| **D7** | **Iteration & Gate Workflow** | **Spec $\to$ Optional Codex Check $\to$ Explicit PI `GO` $\to$ Tier-0 Offline + `<$2` Flash Pilot** | Version every environment/method iteration (`v0.1`, `v0.2`, `v0.3`), enforce mode-free learner interfaces and pre-dispatch concurrent USD reservations (`<$1.95` Tier-1 cap), and refresh the 5-Tab HTML Portal after each approved run. |
| **D8** | **`AlphaDesk-v0.3` Benchmark Parity Upgrade (`2026-11-21`)** | **Two 5-Shift (`6`-Window) Tracks, `H=84`, `6` Strategies, Persistent Holdings $\mathbf{w}_{t-1}$ + `4.0 pt` L1 Turnover Slippage** | `v0.2` (`1` shift, stateless) is frozen strictly as Act 1's causal surgery unit test. `v0.3` matches `ServeLearnBench L3` multi-shift horizons (`5` shifts + `A → B → C → A` recurrence) and `ReScopeBench` cross-episode statefulness (`current_holdings` + turnover cost + multi-turn risk preview tool). |
| **D9** | **Multi-Shift Sequencing & Real-Market Priority (`2026-11-21`)** | **`AlphaDesk-v0.3` First $\to$ Full `6–8` Window `ServeLearnBench L3` Second; Real-World S&P 500 Moved to Low-Priority TODO** | Prioritizes benchmark-grade multi-shift evaluation (`AlphaDesk-v0.3` + `banking_l3` / `retail_l3`) immediately; logs real S&P 500 open-alpha (`Qlib Alpha158/360` + `RD-Agent(Q)` + `FINSABER` anonymization) on the low-priority future backlog. |

---

## 2. Open Questions Logged for Supervisor & Peer Discussion

1. **Structured Scoped Cards vs. Wild Free-Form Meta-Heuristics:**
   - *Current `v0.2` choice:* Structured Scoped Heuristic Cards (`scope_condition`, `action_rule`, `predicted_p_high`).
   - *Alternative to debate with supervisor/peers:* Letting the agent author **open-ended meta-principles / epistemic philosophies** (e.g., *"use $\epsilon$-greedy exploration when variance rises"*, *"reason from first principles when liquidity dries up"*, or free-form self-written operating procedures) without imposing a card schema. We will inspect `v0.2` traces first and revisit whether the card schema helps or over-constrains emergent meta-strategies.
2. **Pure Synthetic Procedural Generators vs. Real / Semi-Real Data Grounding:**
   - *Current `v0.2` choice:* Deterministic procedural factor & shock generator wrapped in realistic analyst/liquidity briefs (guarantees zero pretraining memorization of 2008/2020/2022 market regimes and exact counterfactual likelihoods).
   - *Alternative to debate with supervisor/peers:* Grounding `v0.3+` (or a dedicated real-world evaluation split) in **anonymized, date-shifted post-cutoff empirical time series** (e.g., late-2026 asset/orderbook returns or real drift logs stripped of ticker/date identifiers) so return distributions exhibit authentic fat tails and volatility clustering without enabling memorization shortcuts.

---

## 3. Pre-Registered Specification: `AlphaDesk-v0.2` (Ready for Codex/PI Review Prior to `GO`)

### 3.1 Environment Contract (`src/envs/alpha_desk_v2.py` — Preserving `v0.1` Intact for Regression)
- **Public Task Input:** Anonymized strategies `STRAT_ALPHA`, `STRAT_BETA`, `STRAT_GAMMA`, `STRAT_DELTA`; mode-free `env_name = "alpha_desk_v2"` (zero `:mode` suffix leak).
- **Observable Session Context (`inspect_desk_brief()`, free):** Semi-structured analyst brief reporting regime-relevant macro context (volatility tier, cross-sectional dispersion, and orderbook liquidity state: *normal* vs. *acute market-wide liquidity freeze*).
- **Paid Diagnostic Probe (`probe_strategy_execution(strategy)`, `-12` pts):** Returns noisy live net alpha estimate for one named strategy.
- **Terminal Commit (`submit_allocation(weights)`):** Must sum to `1.0` with each weight $\le 0.65$ (`0.0/100` on compliance breach).
- **5 Seeded Regimes (40-session stream capacity; pre-shift RNG bit-identical up to randomized hidden boundary $t^* \sim \text{Uniform}\{12, \dots, 20\}$; post-boundary scoring window fixed at the first $H_{\text{post}} = 16$ sessions $t^*+1 \dots t^*+16$):**
  1. `stationary`: Initial conditional reward law holds throughout (with stochastic noise).
  2. `abrupt_reversal` (*Primary Iteration-1 Drift Law*): Top two incumbent strategies (`STRAT_ALPHA`, `STRAT_BETA`) and bottom two strategies (`STRAT_GAMMA`, `STRAT_DELTA`) swap expected yields at $t^*$.
  3. `gradual_decay` (*Iteration-2 smooth Type I*): Top incumbent strategies erode linearly over 10 sessions post-$t^*$ while unheld strategies rise.
  4. `opportunity_rise` (*Iteration-2 Type II blindspot test*): Both top-2 incumbent holdings (`0.65` + `0.35`) keep their exact pre-shift reward distributions (`~72` and `~66`), while a **zero-weight unheld strategy** rises from `38` $\to$ `96` at $t^*$—ensuring observable incumbent payoff distributions match `stationary` identically.
  5. `off_scope_shock` (*Iteration-2 Evidence vs. Surprise test*): Underlying reward law does **not** change, but 2 sessions at $t^*$ experience observable **market-wide liquidity freezes** (`spread >= 38 bps`) that depress *all* strategies to `~30`.

### 3.2 Sequenced Execution Plan (One Focused Causal Batch First!)

1. **Iteration 1 (Immediate Target upon PI `GO`) — Paired 3-Branch Causal Ratchet Test:**
   - **Primary Learner & Drift Law:** `ScopedCardBook` (action-attributed counters: a card’s `wins/uses` updates *only* when the agent allocates $\ge 0.30$ weight to that card’s named strategy) on **`abrupt_reversal` vs. paired `stationary` control**.
   - **Shared-Prefix Fork Protocol (per seed $s$):**
     - Run **one** shared pre-boundary acquisition trajectory of length $t^*_s \in \{12, \dots, 20\}$ to obtain learned checkpoint state $M_{t^*_s}$ (1 sham rewrite + 1 utility-excision edit generated once from $M_{t^*_s}$).
     - From $M_{t^*_s}$, evaluate **6 matched 16-session post-boundary branches** (`{abrupt_reversal, stationary} × {Retained, Sham-Edited, Utility-Excised}` = `96` post-boundary + `~16` pre-boundary = `~112` episodes/seed $\approx \$0.42$/seed on `gemini-3.8-flash`).
   - **Budget & Sample Sizing:** Under the strict Tier-1 `--max-usd 1.95` cap (with pre-dispatch concurrent USD reservation), `N = 4` complete paired seeds (`~448` episodes, est. `~$1.68`) run as the exploratory causal pilot; if the PI authorizes a `$4.50` cap, `N = 10` paired seeds run in one batch for tight bootstrap CIs.
   - **Primary Scored Contrasts (paired by seed):**
     1. `Utility-Excised - Sham-Edited` under `abrupt_reversal` (causal ratchet effect; pre-registered meaningful effect $\ge +8.0$ utility pts when $N \ge 8$ yields narrow CIs; treated as exploratory if $N = 4$ intervals remain wide).
     2. `Utility-Excised - Sham-Edited` under `stationary` (cost of discarding valid learned utility rules).
     3. `Retained - Sham-Edited` under both worlds (controls directly for rewrite perturbation).

2. **Iteration 2 (Queued After Iteration-1 Gate) — Online Probe & Calibration Matrix:**
   - Compare `Blind`, `Window-5`, `SRN`, `Recency-CardBook`, `Periodic-Probe`, `SGEH-SurpriseOnly`, `SGEH-NoReset`, and `SGEH+Hazard` across all 5 regimes (`stationary`, `abrupt_reversal`, `gradual_decay`, `opportunity_rise`, `off_scope_shock`), with Brier calibration scored on a fixed evaluator-chosen reference action (`P(score >= 75 | action = STRAT_ALPHA)`).

---

## 4. Pre-Registered Specification: `AlphaDesk-v0.3` Multi-Shift Stateful Benchmark (`2026-11-21` — Awaiting PI `GO`)

Full rationale and literature benchmark parity audit (`ServeLearnBench` vs. `ReScopeBench` vs. `AlphaDesk`): [2026-11-21-alphadesk-v03-multishift-spec.md](file:///home/xz479/llm-time/setup-debates/2026-11-21-alphadesk-v03-multishift-spec.md).

- **Horizon & Multi-Shift Tracks (`H = 84` Sessions, `6` Windows = `5` Shifts, Jittered `13–15` Eps/Window):**
  1. **Track 1 — `multi_abrupt_recurrence`:** `W1: Regime A` $\to$ `W2: Abrupt B + 2-Day Liquidity Freeze` $\to$ `W3: Abrupt C` $\to$ `W4: L3 Recurrence to A` $\to$ `W5: Abrupt D` $\to$ `W6: L3 Recurrence to B`.
  2. **Track 2 — `mixed_decay_opportunity`:** `W1: Regime A` $\to$ `W2: Type-II Silent Opp Rise B` $\to$ `W3: Smooth Crowding Decay B -> C` $\to$ `W4: Type-II Silent Opp Rise D` $\to$ `W5: Smooth Crowding Decay D -> A` $\to$ `W6: Liquidity Shock + Regime E`.
  3. **Matched Control — `stationary` (`H = 84`):** Identical volatility/liquidity tape with stationary calm reward law.
- **6 Anonymized Strategies + Persistent Holdings & Turnover Cost:** `STRAT_S1..STRAT_S6` (`seed % 6` Latin-square rotation), compliance cap `w_i <= 0.55` (`0.55` #1 + `0.45` #2 optimal, leaving 4 unheld `0`-weight strategies), persistent `current_holdings` $\mathbf{w}_{t-1}$, linear L1 turnover slippage `4.0 * 0.5 * ||w_t - w_{t-1}||_1`, and 4 multi-turn tools (`inspect_desk_brief`, `inspect_risk_and_turnover`, `probe_strategy_execution` at `-10 pts`, `submit_allocation`; `max_turns = 6`).
- **`SGEH-v0.3 (Ours)` Upgrades:** (1) `Shelf-Match` 1-probe shelved-regime recall on `W4/W6` recurrence; (2) Post-falsification hazard cooldown (`16` calm steps vs. `9` for unvisited arms, eliminating `iter2d`'s post-reversal re-probe drag).
- **Sequencing:** Validate `AlphaDesk-v0.3` (`Tier 0` $\to$ `<$1.95`/batch `Tier 1` on `gemini-3.8-flash`), then run **full `6–8` window `ServeLearnBench L3`** (`banking_l3`, `retail_l3`).

## 5. Low-Priority Future TODO Backlog
- **Real-World Anonymized S&P 500 Open-Alpha Extension:** Inherit `Microsoft Qlib (Alpha158/Alpha360)` + `RD-Agent(Q)` / `AlphaAgent` formulaic factor DSL under `FINSABER` ticker/date stripping (`ASSET_001..500`, weekly cross-sectional rebalances) as a low-priority camera-ready/v0.4 extension *after* multi-shift `AlphaDesk-v0.3`, full `ServeLearnBench L3`, and Act 3 LoRA GRPO are complete.

