"""Self-contained scientific specification, payoff tables, walkthroughs, and debate records for the Research Portal."""

from __future__ import annotations

from typing import Any

THESIS_EVOLUTION: dict[str, Any] = {
    "research_question": (
        "When domain mechanics and hard safety invariants remain 100% valid (zero tool errors) and only the "
        "latent reward paradigm shifts silently, why do self-improving LLM harnesses lock onto obsolete policies—"
        "and what minimal epistemic state + learning objective enables selective adaptation?"
    ),
    "initial_hypothesis": (
        "Initial intuition ('Stale Memory Hurts'): Self-evolving harnesses (SkillOpt, Continual Harness, ACE, "
        "GEPA, SRN) crystallize early wins into persistent rules and counters, creating an anti-adaptation ratchet; "
        "deleting stale utility rules at the boundary t* should immediately restore adaptation."
    ),
    "sharpened_thesis": (
        "Sharpened Thesis ('Forgetting != Adaptation -> Verify-Before-Quarantine + Selective Reviewer Meta-Plasticity'): "
        "(1) Iterations 1b/1c falsified naive memory wiping (-10.42 pts [-19.86, -1.67]), and 3-model `autonomy_cross_model_n6` proved objective "
        "factual ledgers solve visible reversals (15/15 seeds) while taking 0.0 checks across 192 silent-opportunity sessions. "
        "(2) Under unlabelled equal-loss hidden dips vs. reversals, `Verify-Before-Quarantine (Directive-SGEH-VBQ)` rescued post-dip abandonment "
        "by +28.64 pts (59.0% -> 87.7%) and ranked #1 across all five 3-window streams (85.6% mean). "
        "(3) In `H=60` two-level reviewer runs + `K=8` strategy-anonymized replays, self-written meta-rules causally shift 4-session net returns in "
        "BOTH directions (+27.41 pts p=0.0068 on s704 & +21.12 pts p=0.0003 on s709 vs. -19.95 pts p=0.0078 rule erasure on s711), while untuned "
        "`Qwen/Qwen3-8B` rubber-stamps 93.3% of its own rewrites (Oracle - Keep = +0.17 pts)—pinpointing joint proposer + gatekeeper RL for Act 3."
    ),
    "key_pillars": [
        "Zero Tool-Error Non-Stationarity (Latent R_t Shifts Only)",
        "Permanent Hard Safety Invariants vs. Revisable Utility Cards",
        "Shared-Prefix Mid-Stream Causal State Surgery at Hidden t*",
        "100% On-Policy Open-Weight GRPO + 3-Fold Zero-Shot Domain Transfer",
    ],
}

DECISIONS_D1_D7: list[dict[str, str]] = [
    {
        "id": "D1",
        "axis": "Primary Paper Contribution",
        "decision": "Causal Diagnosis + Learned Meta-Plasticity (8B / 27B LoRA)",
        "rationale": (
            "Isolate the stale-memory / reference-anchor mechanism causally in Act 1, establish SGEH+Hazard as "
            "our engineered Act 2 baseline, and make training an open model (8B/27B LoRA) to selectively revise "
            "its own heuristics and transfer zero-shot across domains the flagship Act 3 contribution."
        ),
    },
    {
        "id": "D2",
        "axis": "Heuristic Memory Schema",
        "decision": "Structured Scoped Heuristic Cards (v0.2)",
        "rationale": (
            "Stores {card_id, kind, scope_condition, target_strategy, recommended_weights, uses, mean_score}. "
            "Modular enough for surgical single-variable causal edits (weights vs. historical score) and proper "
            "Brier forecast scoring, while remaining transferable to ServeLearnBench text/DB tasks."
        ),
    },
    {
        "id": "D3",
        "axis": "AlphaDesk-v0.2 Realism & Sessions",
        "decision": "Semi-Structured Analyst Desk Briefs + Independent Sessions",
        "rationale": (
            "Replaces v0.1 naked scalars with macro/liquidity analyst commentary and anonymized IDs "
            "(STRAT_ALPHA..DELTA) while keeping v0.2 sessions action-independent so multi-arm causal forks have "
            "zero capital-lockup confounds (stateful carryover deferred to v0.3)."
        ),
    },
    {
        "id": "D4",
        "axis": "Non-Stationary Drift Matrix",
        "decision": "4 Diagnostic Regimes + Matched Stationary Control",
        "rationale": (
            "(1) Abrupt Reversal (Type I), (2) Gradual Crowding Decay (smooth Type I), (3) Opportunity Rise "
            "(Type II: incumbent flat 72/66 while 0-weight STRAT_GAMMA rises 38 -> 96), and (4) Off-Scope "
            "Liquidity Shocks (spread >= 38 bps depresses all strategies to ~30 without changing R_t)."
        ),
    },
    {
        "id": "D5",
        "axis": "Act 3 Training Pipeline (CSD3)",
        "decision": "Merged LoRA SFT Warmup -> Two-Arm Multi-Turn GRPO",
        "rationale": (
            "100% on-policy rollouts on Python verifiers (zero Gemini distillation / full API ToS compliance). "
            "Compares Arm A (SFT Warmup), Arm B (SFT -> Outcome-GRPO on net utility), and Arm C "
            "(SFT -> Epistemic+Brier-GRPO on net utility + fixed-target Brier calibration) across 3 LODO folds."
        ),
    },
    {
        "id": "D6",
        "axis": "External Benchmark Anchor",
        "decision": "ServeLearnBench (Retail & Banking L1-L3)",
        "rationale": (
            "Independent CMU/Amazon benchmark (Zheng et al., Oct 2026) providing third-party neutrality and "
            "5 published baselines (Mem0, SkillOpt, Continual Harness, Prime, RAG) in pure Python; SLB-Mini "
            "stratified 3-window x 20-task slice enables <$1 dev validation before full evaluation."
        ),
    },
    {
        "id": "D7",
        "axis": "Execution & Cost Gate Protocol",
        "decision": "Spec -> Peer Review -> Explicit PI GO -> 4-Tier Escalation",
        "rationale": (
            "Tier 0 ($0 offline mock & leak unit tests) -> Tier 1 (<$1.95 Gemini Flash pilot with pre-dispatch "
            "concurrent USD reservation) -> Tier 2 (multi-seed sweeps) -> Tier 3 (one-shot frozen flagship eval)."
        ),
    },
]

OPEN_SUPERVISOR_QUESTIONS: list[dict[str, str]] = [
    {
        "id": "Q1",
        "title": "Second Setting (`LifeCompanion`) & Joint Proposer + Gatekeeper `Qwen3-8B` Training",
        "current_choice": "Untuned Qwen/Qwen3-8B baseline completed on AlphaDesk (`CoalitionOps` 100% untouched for 3-fold transfer)",
        "discussion_prompt": (
            "On 8 fresh AlphaDesk streams (`qwen8b_keep_or_apply_n8`), untuned Qwen3-8B accepted 93.3% (28/30) of distinct rewrites "
            "and 3/3 harmful edits (-6.79, -3.09, -2.52 pts), while its own T=0.6 proposals had only +0.17 pts Oracle - Keep headroom "
            "(max +1.53 pts vs. +21 to +27 pts on Flash falsification rules). Confirm running the same preselected Keep-or-Apply check "
            "on LifeCompanion next, and training BOTH candidate proposal and KEEP/APPLY gatekeeping jointly via multi-rollout GRPO."
        ),
    },
    {
        "id": "Q2",
        "title": "Dated Real-Data Financial Slice (`Kenneth French 10-Industry Portfolios`)",
        "current_choice": "Queued per PI & Codex 28 Nov priority list (strict decision-date cutoff + held-out evaluation era)",
        "discussion_prompt": (
            "Replay chronological Kenneth French 10-Industry Portfolio returns inside AlphaDesk so the reviewer observes only returns "
            "available up to decision date t and period t+1 realized returns score the allocation, reserving a later historical era "
            "strictly for evaluation."
        ),
    },
    {
        "id": "Q3",
        "title": "External Benchmark Scale-Up (`ServeLearnBench` -> `ReScopeBench`) & ICML Jan 2027 Audit",
        "current_choice": "ICML 2027 Abstract due 14 Jan 2027, Full Paper due 22 Jan 2027 AoE; mid-Dec human coauthor trace verification",
        "discussion_prompt": (
            "Small <12-case dev slices on ServeLearnBench hit a binary-feedback identifiability floor (45.0% overall, 93.3% General). "
            "Decide scope/budget for full-protocol ServeLearnBench (vs. 73.4 Continual Harness reference) and ReScopeBench (vs. 60.19 "
            "overall / 81.12 task performance reference), alongside human verification of saved win/reversal traces."
        ),
    },
]

MILESTONES_TIMELINE: list[dict[str, str]] = [
    {
        "week": "Week 1 (Nov 17 - Nov 23, 2026)",
        "phase": "Causal Surgery, Multi-Shift v0.3 & 3-Model Autonomy",
        "status": "completed",
        "deliverables": (
            "Iter 1b/1c single-variable card surgery (-10.42 pts naive wipe); Iter 2/2d counterbalanced matrices (+8.28 pts Type-II hazard); "
            "5-shift AlphaDesk-v0.3 (5,040 eps), 6-window SLB banking_l3 (70.0%), and 3-model autonomy fork (Flash, Hydrogen, Gemma 4 31B)."
        ),
    },
    {
        "week": "Week 2 (Nov 24 - Nov 28, 2026)",
        "phase": "Hidden-Ambiguity VBQ, H=60 Reviewer Replays & Untuned 8B Gate",
        "status": "completed",
        "deliverables": (
            "Unscripted Hidden-Ambiguity VOI + Verify-Before-Quarantine (+28.64 pts dip rescue, #1 across 5 streams); H=60 two-level reviewer "
            "across 2 endings + K=8 strategy-anonymized replays (+27.41 / +21.12 vs. -19.95 pts); local FP16 Qwen/Qwen3-8B Keep-or-Apply baseline (N=8, M=32)."
        ),
    },
    {
        "week": "Week 3 (Nov 29 - Dec 7, 2026)",
        "phase": "Second Setting (`LifeCompanion`), Real-Data Slice & Joint 8B Training",
        "status": "active",
        "deliverables": (
            "Awaiting 29 Nov Group Meeting GO: LifeCompanion Keep-or-Apply baseline (CoalitionOps untouched for 3-fold held-out transfer), "
            "Kenneth French 10-Industry dated slice, and joint proposer+gatekeeper Qwen3-8B LoRA RL on RTX 8000 / CSD3."
        ),
    },
    {
        "week": "Week 4 (Dec 8 - Jan 22, 2027)",
        "phase": "Full-Protocol External Benchmarks, Human Audit & ICML Submission",
        "status": "planned",
        "deliverables": (
            "Full-protocol ServeLearnBench & ReScopeBench evaluation; mid-December human coauthor trace audit & AI disclosure statement; "
            "ICML 2027 abstract (14 Jan 2027) and full paper (22 Jan 2027 AoE)."
        ),
    },
]

COST_TIERS: list[dict[str, str]] = [
    {
        "tier": "Tier 0",
        "scope": "Offline Mock & Pytest Firewall",
        "models": "Mock deterministic responder ($0.00)",
        "cap": "$0.00",
        "purpose": "Verify mode-free learner firewall, bit-identical pre-t* RNG prefixes, and prompt surgery diffs.",
    },
    {
        "tier": "Tier 1",
        "scope": "Small-Scale Causal & Dev Pilots",
        "models": "gemini-3.5-flash / gemini-3.8-flash",
        "cap": "$1.95 / run",
        "purpose": "Pre-dispatch concurrent USD reservation; fast iteration on 4-seed forks & 60-task SLB-Mini slices.",
    },
    {
        "tier": "Tier 2",
        "scope": "Multi-Seed Sweeps & Capability Scaling",
        "models": "gemini-3.8-flash / gemini-4-argon",
        "cap": "Pre-approved batch",
        "purpose": "N=10 paired seeds, 5-regime online matrices, and cross-scale RSI capability curves.",
    },
    {
        "tier": "Tier 3",
        "scope": "One-Shot Frozen Paper Leaderboard",
        "models": "gemini-4-argon / carbon, DeepSeek, Friend APIs",
        "cap": "Single frozen pass",
        "purpose": "Held-out final paper tables right before ACL / ICML submission—never touched during iteration.",
    },
]

DRIFT_TAXONOMY: list[dict[str, Any]] = [
    {
        "id": "type1_reversal",
        "code": "TYPE I — ABRUPT",
        "title": "Incumbent Degradation & Reversal",
        "regime_slug": "abrupt_reversal / utility_reversal",
        "incumbent_curve": [92, 92, 92, 92, 92, 16, 16, 16, 16, 16],
        "alt_curve": [24, 24, 24, 24, 24, 94, 94, 94, 94, 94],
        "shift_idx": 5,
        "surprise_signal": "Strong negative surprise: realized y_t (~30) << retained expectation mu_hat (~73).",
        "failure_mode": (
            "Counter-validated playbooks (SkillBook) buffer stale rules via large pre-shift win counts; "
            "blind deletion (Iter 1b Excised) erases mu_hat and re-crystallizes low yields; naive surprise "
            "satisfices on suboptimal STRAT_DELTA without discriminative probing."
        ),
        "required_mechanism": "Retained baseline anchor mu_hat + downward CUSUM quarantine + non-redundant untested-arm probing.",
    },
    {
        "id": "continuous_decay",
        "code": "TYPE I — SMOOTH",
        "title": "Continuous Crowding & Lifecycle Drift",
        "regime_slug": "gradual_decay / lifecycle_maturation",
        "incumbent_curve": [92, 92, 92, 92, 84, 68, 53, 38, 23, 16],
        "alt_curve": [24, 24, 24, 24, 38, 52, 66, 80, 90, 94],
        "shift_idx": 4,
        "surprise_signal": "Gradual negative slope: single-step shortfall stays near slack delta for 2-3 episodes.",
        "failure_mode": (
            "In M1 AlphaDesk gradual_decay, SkillBook (46.2%) scored -13.3 pts BELOW memoryless Blind (59.5%). "
            "Pure level-CUSUM (SGEH-v1: 65.6%) lags smooth linear decay by ~2 steps compared to frequent reflection."
        ),
        "required_mechanism": "Dual level-CUSUM + EWMA-slope residual gate (SGEH-v2) catching sustained negative drift early.",
    },
    {
        "id": "type2_opportunity",
        "code": "TYPE II — SILENT",
        "title": "Silent Opportunity Emergence",
        "regime_slug": "opportunity_rise",
        "incumbent_curve": [72, 72, 72, 72, 72, 72, 72, 72, 72, 72],
        "alt_curve": [38, 38, 38, 38, 38, 96, 96, 96, 96, 96],
        "shift_idx": 5,
        "surprise_signal": "ZERO surprise on incumbent! 0.65 ALPHA + 0.35 BETA keeps earning ~70 identically to stationary.",
        "failure_mode": (
            "Every outcome-surprise detector (CUSUM, Page-Hinkley, EvoBelief error trigger) receives shortfall = 0 "
            "and sleeps forever in the 'Good-Enough Trap' while optimal utility rises to 96."
        ),
        "required_mechanism": "Staleness-driven hazard clock h(Delta t_a) triggering low-frequency VOI probes on 0-weight arms.",
    },
    {
        "id": "off_scope_shock",
        "code": "FALSE-ALARM TRAP",
        "title": "Observable Off-Scope Liquidity Shock",
        "regime_slug": "off_scope_shock",
        "incumbent_curve": [92, 92, 92, 92, 30, 30, 92, 92, 92, 92],
        "alt_curve": [24, 24, 24, 24, 30, 30, 24, 24, 24, 24],
        "shift_idx": 4,
        "surprise_signal": "Transient score drop (~30) fully explained by observable spread >= 38 bps liquidity freeze.",
        "failure_mode": (
            "Unscoped change detectors or naive recency buffers panic on the 2-session macro freeze, overwrite "
            "their valid calm-market policy, and suffer unnecessary post-shock exploration losses."
        ),
        "required_mechanism": "Scope-conditioned residual attribution: gate updates by card scope (calm + orderly_normal only).",
    },
]

COMPETITOR_MATRIX: list[dict[str, str]] = [
    {
        "system": "SkillOpt (ServeLearnBench, Oct 2026)",
        "paradigm": "Accumulated Discrete Skill Library",
        "stationarity_assumption": "Assumes validated skills remain positively correlated with reward indefinitely.",
        "empirical_failure": "Drops from 69.2% (Retail L1) to 0.0%-1.1% in Retail L2 when store policy rules revise silently.",
        "token_tax": "High skill-retrieval & synthesis overhead.",
    },
    {
        "system": "Continual Harness (Oct 2026)",
        "paradigm": "Continuous Code/Prompt Self-Refinement",
        "stationarity_assumption": "Optimizes harness heuristics monotonically over historical trajectories.",
        "empirical_failure": "Drops from 97.9% (L1) to 30.5% on Retail L3 non-monotonic policy reversals.",
        "token_tax": "Extreme: burns $2,449 across full ServeLearnBench run.",
    },
    {
        "system": "ACE / ExpeL / SkillBook",
        "paradigm": "Counter-Validated Playbook (wins / uses)",
        "stationarity_assumption": "Ranks rules by cumulative lifetime win rate without change-point resets.",
        "empirical_failure": "M1: 90.0% stale lock-in on LifeCompanion (41.4%); worse than Blind on AlphaDesk (46.2% vs 59.5%).",
        "token_tax": "10.4%-18.9% reflection call share even during calm stationarity.",
    },
    {
        "system": "GEPA / Prompt Evolution",
        "paradigm": "Population / Pareto System-Prompt Mutation",
        "stationarity_assumption": "Selects prompt candidates by aggregate fitness across pooled past rollouts.",
        "empirical_failure": "Historical fitness pool drowns out post-shift failures until post-shift horizon >> pre-shift horizon.",
        "token_tax": "74.8%-93.6% of extra RSI tokens spent on candidate verification (Zhang et al., Nov 2026).",
    },
    {
        "system": "EvoBelief (arXiv:2610.12789, Oct 2026)",
        "paradigm": "Surprise-Triggered World-Model Revision",
        "stationarity_assumption": "Triggers belief updates only when observed transition/reward violates prediction.",
        "empirical_failure": "Mathematically blind to Type-II Silent Opportunity Rise (incumbent stays at ~70 while unheld arm hits 96).",
        "token_tax": "Moderate; lacks staleness hazard clock and on-policy weight internalization.",
    },
    {
        "system": "SRN — Bounded Notes (arXiv:2611.13524, Nov 2026)",
        "paradigm": "Periodic Bounded Free-Text Note Rewrite",
        "stationarity_assumption": "Rewrites <=1400-char summary every k episodes from (note, recent_traces).",
        "empirical_failure": "Strongest cheap challenger (70.4% AlphaDesk, 83.2% LifeCompanion), but pays constant tax in stationarity.",
        "token_tax": "Continuous 13.5%-18.9% reflection call tax across calm stationary streams.",
    },
]

ALPHADESK_V2_PAYOFF_TABLES: dict[str, Any] = {
    "meta": {
        "strategies": ["STRAT_ALPHA", "STRAT_BETA", "STRAT_GAMMA", "STRAT_DELTA"],
        "horizon": "40 sessions (hidden boundary t* ~ Uniform{12..16}; post-boundary eval window H_post = 16)",
        "compliance": "Weights w_i >= 0, sum(w) = 1.0 (+/-0.03), max single weight w_i <= 0.65 (breach => Score 0.0/100)",
        "fees_and_noise": "inspect_desk_brief(): 0 pts (-18 pts if skipped) | probe_strategy_execution(s): -12 pts (limit 1/session) | Noise: N(0, 2.4^2)",
    },
    "rows": [
        {
            "regime": "stationary",
            "phase": "All t = 1..40 (Calm Vol < 24.5, ~75% eps)",
            "alpha": 92.0,
            "beta": 78.0,
            "gamma": 24.0,
            "delta": 18.0,
            "optimal_portfolio": "0.65 ALPHA + 0.35 BETA",
            "oracle_expected": 87.1,
            "notes": "Matched control stream; bit-identical pre-t* RNG prefix to all drift regimes.",
        },
        {
            "regime": "abrupt_reversal",
            "phase": "Post-t* Calm Vol (t > t*)",
            "alpha": 16.0,
            "beta": 22.0,
            "gamma": 94.0,
            "delta": 78.0,
            "optimal_portfolio": "0.65 GAMMA + 0.35 DELTA",
            "oracle_expected": 88.4,
            "notes": "Type-I primary causal test: Pre-t* matches stationary (92/78/24/18); top-2 and bottom-2 invert at t*.",
        },
        {
            "regime": "gradual_decay",
            "phase": "Post-t* Calm Vol (10-step linear blend)",
            "alpha": "92 -> 16",
            "beta": "78 -> 22",
            "gamma": "24 -> 94",
            "delta": "18 -> 78",
            "optimal_portfolio": "ALPHA/BETA -> GAMMA/DELTA",
            "oracle_expected": "87.1 -> 88.4",
            "notes": "Smooth crowding erosion: mu_t = (1 - lambda_t)*mu_pre + lambda_t*mu_rev with lambda_t = min(1, (t - t* + 1)/10).",
        },
        {
            "regime": "opportunity_rise",
            "phase": "Pre-t* -> Post-t* Calm Vol",
            "alpha": "72.0 -> 72.0",
            "beta": "66.0 -> 66.0",
            "gamma": "38.0 -> 96.0",
            "delta": "24.0 -> 24.0",
            "optimal_portfolio": "0.65 GAMMA + 0.35 ALPHA (post)",
            "oracle_expected": "69.9 -> 87.6",
            "notes": "Type-II blindspot test: Incumbent 0.65 ALPHA + 0.35 BETA earns 69.9 pre AND post t*; 0-weight GAMMA jumps 38 -> 96.",
        },
        {
            "regime": "off_scope_shock",
            "phase": "Sessions t*, t*+1 Acute Freeze (spread >= 38 bps)",
            "alpha": 30.0,
            "beta": 30.0,
            "gamma": 30.0,
            "delta": 30.0,
            "optimal_portfolio": "Any compliant w (all 30.0)",
            "oracle_expected": 30.0,
            "notes": "False-alarm test: Observable acute_liquidity_freeze depresses all strategies for 2 eps, then reverts to 92/78/24/18.",
        },
        {
            "regime": "ALL 5 REGIMES (Invariant)",
            "phase": "Elevated Volatility (vol >= 24.5, ~25% eps)",
            "alpha": 22.0,
            "beta": 28.0,
            "gamma": 74.0,
            "delta": 92.0,
            "optimal_portfolio": "0.65 DELTA + 0.35 GAMMA",
            "oracle_expected": 85.7,
            "notes": "Permanent cross-regime tail-volatility hedge invariant (except during the 2-session liquidity freeze).",
        },
    ],
}

ENVIRONMENT_DOCS: list[dict[str, Any]] = [
    {
        "id": "alpha_desk_v2",
        "name": "AlphaDesk-v0.2 (Controlled Causal Benchmark — Active Iteration 1/1b/1c)",
        "pillar": "Pillar 2 — ParadigmBench (Flagship Causal Setup)",
        "source_file": "src/envs/alpha_desk_v2.py",
        "paradigm_type": "5 Seeded Regimes (Abrupt Reversal, Gradual Decay, Opportunity Rise, Off-Scope Shock, Stationary)",
        "task_summary": (
            "40-session portfolio desk across anonymized strategies STRAT_ALPHA, STRAT_BETA, STRAT_GAMMA, STRAT_DELTA "
            "with hidden randomized boundary t* in {12..16}. Agent reads semi-structured analyst & liquidity desk briefs, "
            "optionally pays -12 pts (limit 1/session) to probe one strategy's live execution yield, and commits compliant weights."
        ),
        "invariants": [
            "Compliance caps: non-negative weights summing to 1.0 (+/-0.03), each <= 0.65 (Score: 0.0/100 if breached).",
            "Persistent tail-volatility hedge: in elevated_volatility sessions (vol >= 24.5), STRAT_DELTA=92 & STRAT_GAMMA=74 remain optimal.",
            "Strict learner firewall: mode-free env_name ('alpha_desk_v2') and bit-identical pre-t* RNG prefixes across all 5 modes.",
        ],
        "shift_mechanics": (
            "In `abrupt_reversal`, calm-volatility leaders (STRAT_ALPHA=92, STRAT_BETA=78) invert with STRAT_GAMMA=94 and "
            "STRAT_DELTA=78 at hidden session t*, while high-volatility hedge behavior stays invariant. Also supports "
            "`gradual_decay` (10-step linear crowding), `opportunity_rise` (unheld 0-weight STRAT_GAMMA rises 38 -> 96 while "
            "ALPHA/BETA stay flat at 72/66), and `off_scope_shock` (observable 2-session liquidity freeze at spread >= 38 bps)."
        ),
        "tools": [
            {
                "sig": "inspect_desk_brief()",
                "cost": "0 pts (Free; -18 pt penalty if skipped)",
                "returns": "{session, vol_index, volatility_regime_hint, cross_sectional_dispersion, bid_ask_spread_bps, liquidity_status, analyst_desk_brief, compliance_limits}",
            },
            {
                "sig": "probe_strategy_execution(strategy)",
                "cost": "-12 pts session fee (max 1 probe / session)",
                "returns": "{strategy, probe_fee_deducted: 12.0, live_net_yield_estimate}",
            },
            {
                "sig": "submit_allocation(weights)",
                "cost": "Terminal commit",
                "returns": "{status: 'submitted', score: 0..100}",
            },
        ],
        "planned_v2_refinements": (
            "Iter 1/1b/1c isolate 3-arm and factorial causal surgery (`Retained`, `Sham`, `Excised`, `Score-Shown` vs `Score-Hidden`); "
            "Iteration 2 evaluates online `SGEH+Hazard` vs `SRN` vs `Periodic-Probe` across all 5 regimes."
        ),
    },
    {
        "id": "alpha_desk",
        "name": "AlphaDesk-v0.1 (Factor Crowding & A -> B -> A Regime Recurrence)",
        "pillar": "Pillar 2 — ParadigmBench",
        "source_file": "src/envs/alpha_desk.py",
        "paradigm_type": "Continuous Alpha Decay & Non-Monotonic Regime Cycles",
        "task_summary": (
            "Across sequential trading sessions (t = 1..H), the agent manages capital across 4 named quantitative "
            "strategies (MOMENTUM_TECH, MEAN_REV_FX, CARRY_CREDIT, DEFENSIVE_VOL) after reading live market-tape scalars "
            "and optional -12 pt crowding diagnostics."
        ),
        "invariants": [
            "Portfolio weights must be non-negative and sum to 1.0 (+/- 0.03 tolerance).",
            "Single-strategy concentration cap: no weight may exceed 0.65 (Score: 0.0/100 if breached).",
            "High-volatility defensive convexity: when vol_index >= 23.0, DEFENSIVE_VOL (88-90) remains a top hedge.",
        ],
        "shift_mechanics": (
            "In Phase 1 (Regime A), MOMENTUM_TECH (92) and CARRY_CREDIT (82 if dispersion >= 0.50 else 68) dominate calm "
            "sessions. In `gradual_decay`, after t = H/3, Momentum and Carry erode linearly down to 14 and 20 while "
            "MEAN_REV_FX rises 28 -> 94. In `regime_cycle`, the market flips A -> B at H/3 and recurs B -> A at 2H/3 "
            "(testing shelved-regime recall vs. catastrophic forgetting)."
        ),
        "tools": [
            {
                "sig": "inspect_market_tape()",
                "cost": "0 pts (Free; -18 pt penalty if skipped)",
                "returns": "{session, vol_index, cross_asset_dispersion, liquidity_spread_bps, compliance_cap_per_strategy}",
            },
            {
                "sig": "check_strategy_diagnostics(strategy)",
                "cost": "-12 pts session fee",
                "returns": "{strategy, diagnostic_fee_deducted: 12.0, probed_expected_yield_estimate}",
            },
            {
                "sig": "submit_allocation(weights)",
                "cost": "Terminal commit",
                "returns": "{status: 'submitted', score: 0..100}",
            },
        ],
        "planned_v2_refinements": (
            "Preserved intact for M1 regression reproducibility (where SkillBook 46.2% fell below Blind 59.5% and SGEH reached 65.6%)."
        ),
    },
    {
        "id": "life_companion",
        "name": "LifeCompanion (Longitudinal Personal Assistant)",
        "pillar": "Pillar 2 — ParadigmBench",
        "source_file": "src/envs/life_companion.py",
        "paradigm_type": "Smooth Compositional User Lifecycle Maturation",
        "task_summary": (
            "Across sequential weeks (t = 1..H), the agent assists the same client (Alex) across rotating personal "
            "planning requests (meal_wellness, weekly_schedule, housing_commute, career_learning), selecting 1 of 5 "
            "candidate options (3 valid lifestyle tracks + 2 fatal allergy/budget traps) per week."
        ),
        "invariants": [
            "Medical safety rule: NEVER select an option tagged `contains_tree_nuts` (immediate Score: 0.0 / 100).",
            "Budget ceiling: NEVER select an option with `monthly_cost > $950` (immediate Score: 0.0 / 100).",
        ],
        "shift_mechanics": (
            "Alex's Week-1 onboarding bio states '23yo recent grad — push my career growth hard and keep costs low!', "
            "and is never manually edited. Stage 1 (t < H/3) weights ambition_upside (0.75) + cost_saving (0.25). "
            "Stage 2 (H/3 <= t < 2H/3) transitions smoothly to burnout recovery (rest_stability 0.65, family 0.15). "
            "Stage 3 (t >= 2H/3) transitions to new-parent priority (family_flexibility 0.60 + rest_stability 0.30)."
        ),
        "tools": [
            {
                "sig": "read_user_dossier()",
                "cost": "0 pts (Free)",
                "returns": "{client: 'Alex', permanent_hard_constraints, initial_onboarding_bio_note}",
            },
            {
                "sig": "search_catalog()",
                "cost": "0 pts (Free)",
                "returns": "{category, candidates: [{option_id, title, monthly_cost}, ...]}",
            },
            {
                "sig": "inspect_option_details(option_id)",
                "cost": "0 pts (Free)",
                "returns": "{option_id, title, monthly_cost, ambition_upside, rest_stability, family_flexibility, tags}",
            },
            {
                "sig": "recommend_plan(option_id)",
                "cost": "Terminal commit",
                "returns": "{status: 'recommended', chosen_option, score: 0..100}",
            },
        ],
        "planned_v2_refinements": (
            "M1 showed SGEH reaching 82.7% post-shift (+41.3 pts over SkillBook's 41.4% / 90% stale lock-in) with 0% allergy/budget violations."
        ),
    },
    {
        "id": "coalition_ops",
        "name": "CoalitionOps (Strategic Supply-Network Contracting)",
        "pillar": "Pillar 2 — ParadigmBench",
        "source_file": "src/envs/coalition_ops.py",
        "paradigm_type": "Silent Strategic Incentive Reversal (Foe <-> Friend)",
        "task_summary": (
            "In each dispatch cycle (t = 1..H), one of three counterparties (VANGUARD_CORP, MERIDIAN_LOGISTICS, "
            "HELIOS_MATERIALS) sits on the critical path. The agent inspects SLA reserve buffers, decides whether to pay "
            "an 18-point audit fee to check live partner alignment (92% accurate), and commits a contract posture."
        ),
        "invariants": [
            "SLA Solvency Rule: NEVER select `cooperative_joint` when `reserve_buffer < 0.25` (triggers immediate SLA default: Score 0.0 / 100).",
        ],
        "shift_mechanics": (
            "Base payoffs: `ally` rewards `cooperative_joint` (96) vs `defensive_reroute` (22); `rival` punishes "
            "`cooperative_joint` (12) and rewards `defensive_reroute` (90); `neutral` rewards `escrow_hedged` (84). "
            "Phase 1 (Vanguard=ally, Meridian=rival, Helios=neutral) silently inverts at H/3 (Vanguard=rival, "
            "Meridian=ally, Helios=ally) and realigns at 2H/3 (Vanguard=neutral, Meridian=ally, Helios=rival)."
        ),
        "tools": [
            {
                "sig": "inspect_dispatch_ledger()",
                "cost": "0 pts (Free; -15 pt penalty if skipped)",
                "returns": "{cycle, active_counterparty, reserve_buffer, min_reserve_for_cooperative_joint: 0.25}",
            },
            {
                "sig": "audit_counterparty(partner)",
                "cost": "-18 pts audit fee",
                "returns": "{audited_partner, audit_fee_deducted: 18.0, live_behavioral_indicator}",
            },
            {
                "sig": "execute_contract(partner, posture)",
                "cost": "Terminal commit",
                "returns": "{status: 'executed', posture, score: 0..100}",
            },
        ],
        "planned_v2_refinements": (
            "Primary Act-3 zero-shot transfer target for Value-of-Information (VOI) auditing (-18 pt probe vs. 96-to-12 betrayal crash)."
        ),
    },
    {
        "id": "servelearnbench",
        "name": "ServeLearnBench (External CMU/Amazon Benchmark — Retail & Banking L1-L3)",
        "pillar": "Pillar 1 — Independent External Anchor",
        "source_file": "src/slb_adapter.py",
        "paradigm_type": "Piecewise-Constant Window Policy Revisions & Non-Monotonic Reversals",
        "task_summary": (
            "Third-party streaming benchmark (Zheng et al., Oct 2026, arXiv:2610.07792) spanning multi-turn Retail "
            "order-DB customer support (returns, exchanges, VIP overrides) and Banking numeric/categorical case review. "
            "Agents observe only `Score: n/100` after serving tasks and are evaluated on held-out window test splits."
        ),
        "invariants": [
            "Fully-Specified static policy clauses and DB schema constraints must retain ~95%-100% compliance across windows.",
            "Strict protocol firewall in `src/slb_adapter.py`: `context()` ignores `phase` and `window`; updates use only public serving traces.",
        ],
        "shift_mechanics": (
            "L1 tests stationary policy acquisition; L2 introduces silent mid-stream policy revisions (where published "
            "SkillOpt collapses 69.2% -> 0.0%-1.1%); L3 introduces non-monotonic A -> B -> A policy reversals & re-entry "
            "(where Continual Harness drops 97.9% -> 30.5% at $2,449). Evaluated in dev via SLB-Mini (3 windows x 20 tasks, <$1)."
        ),
        "tools": [
            {
                "sig": "get_order_details / list_user_orders / process_return_or_exchange",
                "cost": "Multi-turn DB tools (up to 50 steps/ep)",
                "returns": "Structured JSON customer/order/banking records; scalar `Score: n/100` after serving commit.",
            },
            {
                "sig": "evaluate_banking_case / submit_decision / finish()",
                "cost": "Terminal episode resolution",
                "returns": "Public terminal reward in {0..100} during serving phase; hidden during held-out window eval.",
            },
        ],
        "planned_v2_refinements": (
            "Run chronological `retail_l3` and `banking_l3` 3-window validation slices in Iteration 2 and frozen Act 3 transfer."
        ),
    },
]

CONCRETE_WALKTHROUGHS: list[dict[str, Any]] = [
    {
        "env_id": "alpha_desk_v2",
        "label": "AlphaDesk-v0.2 (Post-Reversal Session #16)",
        "scenario_badge": "Abrupt Reversal at t*=15 | Calm Volatility Session",
        "public_prompt": (
            "DOMAIN PROMPT: You are the Portfolio Manager on AlphaDesk-v2 managing capital across STRAT_ALPHA, "
            "STRAT_BETA, STRAT_GAMMA, STRAT_DELTA. Call `inspect_desk_brief` first, optionally call "
            "`probe_strategy_execution` (-12 pts fee, limit 1 per session), and call `submit_allocation` with "
            "non-negative weights summing to 1.0 and each weight <= 0.65 (breach => Score: 0.0/100).\n"
            "TASK PROMPT: Session #16: Review today's desk brief and submit compliant portfolio weights."
        ),
        "turns": [
            {
                "turn": "Turn 1 (Free Context Inspection)",
                "action": '{"tool": "inspect_desk_brief", "args": {}}',
                "observation": (
                    '{"session": 16, "vol_index": 14.8, "volatility_regime_hint": "calm_volatility", '
                    '"cross_sectional_dispersion": 0.54, "bid_ask_spread_bps": 7.2, "liquidity_status": "orderly_normal", '
                    '"analyst_desk_brief": "Calm macro session (vol index 14.8, dispersion 0.54) with tight dealer spreads (7.2 bps)."}'
                ),
                "analysis": "Zero errors! Tools behave identically to Sessions 1..15; liquidity is orderly_normal and volatility is calm.",
            },
            {
                "turn": "Turn 2A (Stale / Blind-Excision Failure Path)",
                "action": (
                    'Retained Arm skips probe & submits stale {"STRAT_ALPHA": 0.65, "STRAT_BETA": 0.15, "STRAT_GAMMA": 0.10, "STRAT_DELTA": 0.10}.\n'
                    'Iter-1b Excised Arm probes STRAT_ALPHA -> sees 15.8 -> lacks mu_hat=72.9 anchor -> STILL submits STRAT_ALPHA=0.65!'
                ),
                "observation": '{"status": "submitted", "score": 30.8}  // Oracle expected: 88.4 -> Normalized Utility: 34.9% (or 16.4% after -12 probe fee!)',
                "analysis": "Illustrates BOTH failure modes from Iter 1b: stale lock-in without probing, and reference-point blindness when mu_hat is deleted.",
            },
            {
                "turn": "Turn 2B -> Turn 3 (Target Discriminative Adaptation Path)",
                "action": (
                    'Turn 2: {"tool": "probe_strategy_execution", "args": {"strategy": "STRAT_GAMMA"}}  // Non-redundant probe on untested arm!\n'
                    'Turn 3: {"tool": "submit_allocation", "args": {"weights": {"STRAT_ALPHA": 0.0, "STRAT_BETA": 0.0, "STRAT_GAMMA": 0.65, "STRAT_DELTA": 0.35}}}'
                ),
                "observation": (
                    'Turn 2 Obs: {"strategy": "STRAT_GAMMA", "probe_fee_deducted": 12.0, "live_net_yield_estimate": 95.2}\n'
                    'Turn 3 Obs: {"status": "submitted", "score": 76.9}  // Next calm session (no probe needed): Score 88.9 (100% Oracle Utility!)'
                ),
                "analysis": "One targeted -12 pt probe discovers STRAT_GAMMA=94; subsequent calm sessions earn ~88.4/88.4 with zero probe fees.",
            },
        ],
    },
    {
        "env_id": "alpha_desk",
        "label": "AlphaDesk-v0.1 (Gradual Decay Session #11)",
        "scenario_badge": "Continuous Crowding Decay | MOMENTUM_TECH Eroding 92 -> 14",
        "public_prompt": (
            "DOMAIN PROMPT: Quantitative portfolio manager across MOMENTUM_TECH, MEAN_REV_FX, CARRY_CREDIT, DEFENSIVE_VOL. "
            "Weights >= 0, sum = 1.0, each <= 0.65 (breach => 0/100).\n"
            "TASK PROMPT: Trading Session #11: Inspect the live market tape and submit compliant capital weights."
        ),
        "turns": [
            {
                "turn": "Turn 1 (Market Tape Read)",
                "action": '{"tool": "inspect_market_tape", "args": {}}',
                "observation": '{"session": 11, "vol_index": 17.4, "cross_asset_dispersion": 0.63, "liquidity_spread_bps": 8.5, "compliance_cap_per_strategy": 0.65}',
                "analysis": "Normal volatility (< 23.0) and high dispersion (>= 0.50); in Phase 1 this favored 0.65 MOMENTUM_TECH + 0.35 CARRY_CREDIT.",
            },
            {
                "turn": "Turn 2 (Diagnostic Crowding Check)",
                "action": '{"tool": "check_strategy_diagnostics", "args": {"strategy": "MEAN_REV_FX"}}',
                "observation": '{"strategy": "MEAN_REV_FX", "diagnostic_fee_deducted": 12.0, "probed_expected_yield_estimate": 88.7}',
                "analysis": "Confirms factor rotation into MEAN_REV_FX while SkillBook stays locked on 5/5 early MOMENTUM_TECH wins (46.2% utility).",
            },
            {
                "turn": "Turn 3 (Compliant Re-Allocation)",
                "action": '{"tool": "submit_allocation", "args": {"weights": {"MOMENTUM_TECH": 0.0, "MEAN_REV_FX": 0.65, "CARRY_CREDIT": 0.0, "DEFENSIVE_VOL": 0.35}}}',
                "observation": '{"status": "submitted", "score": 75.4}  // Unprobed follow-up session earns 87.7 / 100',
                "analysis": "Respects the permanent 0.65 concentration cap while pivoting capital out of crowded momentum.",
            },
        ],
    },
    {
        "env_id": "life_companion",
        "label": "LifeCompanion (Stage-3 Week #13 Planning)",
        "scenario_badge": "Smooth Lifecycle Maturation | Stale Bio vs. New-Parent Utility",
        "public_prompt": (
            "DOMAIN PROMPT: Long-term personal planning agent assisting Alex week after week. Permanent hard constraints "
            "in the dossier never change; personal priorities may evolve without Alex editing their Week-1 bio.\n"
            "TASK PROMPT: Week #13 Request: Select the best `meal_wellness` option for Alex this week."
        ),
        "turns": [
            {
                "turn": "Turn 1 (Dossier & Catalog Lookup)",
                "action": '{"tool": "read_user_dossier", "args": {}} -> {"tool": "search_catalog", "args": {}}',
                "observation": (
                    'Dossier: {"permanent_hard_constraints": {"medical_allergy_forbidden_tag": "contains_tree_nuts", "strict_max_monthly_cost_usd": 950}, '
                    '"initial_onboarding_bio_note": "Written at Week 1 onboarding: 23yo recent grad—push my career growth hard and keep costs low!"}\n'
                    'Catalog: MEAL_13_HUSTLE ($490), MEAL_13_RECOVERY ($660), MEAL_13_FAMILY ($790), MEAL_13_TRAP_ALLERGY ($580), MEAL_13_TRAP_OVERBUDGET ($1240)'
                ),
                "analysis": "Notice the trap: Week-1 bio still says 'push career growth hard', and TRAP_ALLERGY looks cheap ($580) with 0.95 attributes!",
            },
            {
                "turn": "Turn 2 (Attribute & Safety Tag Inspection)",
                "action": '{"tool": "inspect_option_details", "args": {"option_id": "MEAL_13_FAMILY"}}  (and inspecting TRAP_ALLERGY)',
                "observation": (
                    'MEAL_13_FAMILY: {"monthly_cost": 790, "ambition_upside": 0.34, "rest_stability": 0.84, "family_flexibility": 0.96, "tags": ["home_flexible", "childcare_compatible"]}\n'
                    'MEAL_13_TRAP_ALLERGY: {"monthly_cost": 580, "tags": ["contains_tree_nuts", "premium_perk"]} -> REJECTED BY INVARIANT!'
                ),
                "analysis": "Agent filters out `contains_tree_nuts` & `>$950` permanently while recognizing Stage-3 family/rest preference shift.",
            },
            {
                "turn": "Turn 3 (Terminal Plan Recommendation)",
                "action": '{"tool": "recommend_plan", "args": {"option_id": "MEAL_13_FAMILY"}}',
                "observation": '{"status": "recommended", "chosen_option": "MEAL_13_FAMILY", "score": 86.4}  // Stale MEAL_13_HUSTLE scores only 33.1!',
                "analysis": "In M1, SGEH achieved 82.7% post-shift utility (+41.3 pts over SkillBook's 41.4%) with 0.0% safety violations.",
            },
        ],
    },
    {
        "env_id": "coalition_ops",
        "label": "CoalitionOps (Cycle #7 Foe <-> Friend Reversal)",
        "scenario_badge": "Silent Strategic Reversal | VANGUARD_CORP Flips Ally -> Rival",
        "public_prompt": (
            "DOMAIN PROMPT: Strategic Operations Director managing dispatches across VANGUARD_CORP, MERIDIAN_LOGISTICS, "
            "HELIOS_MATERIALS. Invariant rule: NEVER select `cooperative_joint` when `reserve_buffer < 0.25` (Score 0/100). "
            "Optional `audit_counterparty` costs -18 pts.\n"
            "TASK PROMPT: Dispatch Cycle #7: Counterparty `VANGUARD_CORP` is on the critical path."
        ),
        "turns": [
            {
                "turn": "Turn 1 (Dispatch Solvency Check)",
                "action": '{"tool": "inspect_dispatch_ledger", "args": {}}',
                "observation": '{"cycle": 7, "active_counterparty": "VANGUARD_CORP", "reserve_buffer": 0.58, "min_reserve_for_cooperative_joint": 0.25}',
                "analysis": "Solvency buffer 0.58 >= 0.25 permits all postures; in Cycles 1..5 VANGUARD_CORP was a trusted ally (96 pts).",
            },
            {
                "turn": "Turn 2 (Targeted Post-Surprise Audit vs. Blind Trust)",
                "action": (
                    'Stale Trust Path: Immediately calls `execute_contract(VANGUARD_CORP, cooperative_joint)` -> Betrayed: Score 12.0/100!\n'
                    'Targeted Audit Path: {"tool": "audit_counterparty", "args": {"partner": "VANGUARD_CORP"}} (-18 pts)'
                ),
                "observation": '{"audited_partner": "VANGUARD_CORP", "audit_fee_deducted": 18.0, "live_behavioral_indicator": "opportunistic_margin_extraction_detected"}',
                "analysis": "92%-accurate audit reveals boardroom incentive inversion before committing capital; subsequent cycles skip the -18 pt fee.",
            },
            {
                "turn": "Turn 3 (Defensive Reroute Execution)",
                "action": '{"tool": "execute_contract", "args": {"partner": "VANGUARD_CORP", "posture": "defensive_reroute"}}',
                "observation": '{"status": "executed", "posture": "defensive_reroute", "score": 73.2}  // Next VANGUARD cycle (no audit fee): Score 90.4/90.0!',
                "analysis": "Demonstrates the exact VOI trade-off: auditing every step caps score at 72/90 (80%), whereas surprise-gated auditing hits ~90.",
            },
        ],
    },
    {
        "env_id": "servelearnbench",
        "label": "ServeLearnBench (Retail & Banking L2/L3 Shift)",
        "scenario_badge": "External CMU/Amazon Benchmark | Multi-Turn DB Policy Revision",
        "public_prompt": (
            "DOMAIN PROMPT: Retail Customer Support / Banking Case Review Agent. Query customer & order records via DB "
            "tools, enforce permanent identity/fraud verification rules, and execute compliant resolutions. During "
            "serving windows, only terminal `Score: n/100` is observed after `finish()`.\n"
            "TASK PROMPT (Retail L3 Window 2): Customer requests an instant exchange on a final-sale electronics item opened 18 days ago."
        ),
        "turns": [
            {
                "turn": "Turn 1 (Multi-Turn Order & Customer Tier Lookup)",
                "action": '{"tool": "get_order_details", "args": {"order_id": "#W88412"}} -> {"tool": "get_customer_profile", "args": {"user_id": "U_209"}}',
                "observation": '{"order_id": "#W88412", "category": "electronics", "days_since_delivery": 18, "opened": true, "customer_tier": "gold_member"}',
                "analysis": "In Window 1 (L1), learned skill said 'Gold members receive 30-day instant exchange on opened electronics' (earned 100/100).",
            },
            {
                "turn": "Turn 2 (Silent Window-2 Policy Shift Impact)",
                "action": 'SkillOpt / Stale Agent executes `approve_instant_exchange("#W88412")` without checking recent serving score drops.',
                "observation": 'Serving Feedback: Score: 0/100 (Window 2 silently tightened opened electronics exchanges to <=14 days, reverting in Window 3!).',
                "analysis": "Published ServeLearnBench result: SkillOpt collapses from 69.2% (L1) to 0.0%-1.1% (L2) by rigidly obeying Window-1 skills!",
            },
            {
                "turn": "Turn 3 (Scoped Card Quarantine & Re-Entry)",
                "action": 'SGEH shelves `[Window-1 Gold 30-day electronics rule]` upon CUSUM alarm, tests store-credit / 14-day boundary, and recalls shelf in Window 3.',
                "observation": 'Serving Feedback: Score: 100/100 while preserving 98%+ compliance on unchanging identity-verification invariants.',
                "analysis": "`src/slb_adapter.py` enforces strict zero-leak protocol (`context()` ignores `phase` and `window`).",
            },
        ],
    },
]

CARD_SURGERY_ARMS: list[dict[str, Any]] = [
    {
        "arm_id": "retained",
        "title": "1. Retained / Ranking-Retained",
        "badge": "Intact Pre-Shift Baseline",
        "used_in": "Iter 1, Iter 1b, Iter 1c",
        "weights_state": 'Kept: {"STRAT_ALPHA": 0.65, "STRAT_BETA": 0.15, "STRAT_GAMMA": 0.10, "STRAT_DELTA": 0.10}',
        "score_state": "Kept: Attributed n=10, mean_score=72.9",
        "prompt_line": (
            "- Card C1 [calm_utility | Attributed: n=10, mean_score=72.9]: "
            "IF volatility_regime_hint == 'calm_volatility' and liquidity_status == 'orderly_normal' -> "
            'recommended_weights = {"STRAT_ALPHA": 0.65, "STRAT_BETA": 0.15, "STRAT_GAMMA": 0.1, "STRAT_DELTA": 0.1}'
        ),
        "scientific_role": (
            "Untouched pre-shift cardbook M_{t*}. Immediate negative surprise (~30 vs 72.9) dislodges STRAT_ALPHA by t*+2, "
            "yet agent satisfices on STRAT_DELTA / cycles ALPHA-BETA with 0/44 optimal calm allocations."
        ),
    },
    {
        "arm_id": "sham",
        "title": "2. Sham-Edited Control",
        "badge": "Paraphrase Perturbation Control",
        "used_in": "Iter 1, Iter 1b",
        "weights_state": 'Kept: {"STRAT_ALPHA": 0.65, "STRAT_BETA": 0.15, "STRAT_GAMMA": 0.10, "STRAT_DELTA": 0.10}',
        "score_state": "Kept: Attributed n=10, mean_score=72.9",
        "prompt_line": (
            "- Card C1 [calm_utility | Attributed: n=10, mean_score=72.9]: "
            "IF volatility_regime_hint == 'low_volatility' and liquidity_status == 'standard_liquid' -> "
            'recommended_weights = {"STRAT_ALPHA": 0.65, "STRAT_BETA": 0.15, "STRAT_GAMMA": 0.1, "STRAT_DELTA": 0.1}'
        ),
        "scientific_role": (
            "Controls for LLM rewrite perturbation and exact prompt length (1966 vs 1967 chars). Achieved +0.00 pts "
            "[0.00, 0.00] stationary difference vs. Retained in Iter 1b."
        ),
    },
    {
        "arm_id": "score_hidden",
        "title": "3. Utility-Excised / Score-Hidden",
        "badge": "Joint Weight + Score Wipe",
        "used_in": "Iter 1b (`excised`), Iter 1c (`score_hidden`)",
        "weights_state": 'Cleared: {"STRAT_ALPHA": null, "STRAT_BETA": null, "STRAT_GAMMA": null, "STRAT_DELTA": null}',
        "score_state": "Hidden/Cleared: mean_score=null (Iter 1b n=0; Iter 1c keeps n=10 identical!)",
        "prompt_line": (
            "- Card C1 [calm_utility | Attributed: n=10, mean_score=null]: "
            "IF volatility_regime_hint == 'calm_volatility' and liquidity_status == 'orderly_normal' -> "
            'recommended_weights = {"STRAT_ALPHA": null, "STRAT_BETA": null, "STRAT_GAMMA": null, "STRAT_DELTA": null}'
        ),
        "scientific_role": (
            "Tests retiring calm weights WITHOUT a historical score anchor. Iter 1b (n=0): -10.42 pts [-19.86, -1.67]. "
            "Iter 1c (n=9..10 kept, 1-probe limit disclosed): 61.0% net utility (7 probes; 4/4 first post-drop probes "
            "still tested STRAT_ALPHA, but Seed 102's 2nd probe hit STRAT_GAMMA -> 6/11 optimal calm sessions, 71.6%)."
        ),
    },
    {
        "arm_id": "score_shown",
        "title": "4. Score-Anchor (`Score-Shown`, Iter 1c)",
        "badge": "0-Char Diff Single-Variable Test",
        "used_in": "Iter 1c (`score_shown`, $1.30)",
        "weights_state": 'Cleared: {"STRAT_ALPHA": null, "STRAT_BETA": null, "STRAT_GAMMA": null, "STRAT_DELTA": null}',
        "score_state": "Preserved: Attributed n=10, mean_portfolio_score=72.9 (Exact 0-char pad match to `null (n/a)`)",
        "prompt_line": (
            "- Card C1 [calm_utility | Attributed: n=10, mean_portfolio_score=72.9 (0-100)]: "
            "IF volatility_regime_hint == 'calm_volatility' and liquidity_status == 'orderly_normal' -> "
            'recommended_weights = {"STRAT_ALPHA": null, "STRAT_BETA": null, "STRAT_GAMMA": null, "STRAT_DELTA": null}'
        ),
        "scientific_role": (
            "Forked from saved Iter-1b checkpoints with +0 char prompt diff (2126/2126, 2348/2348, 2067/2067, 2356/2356): "
            "Score-Shown - Score-Hidden = -5.10 pts [-17.16, +7.88] (2 probes, 0/44 calm optimal)—ruling out unattributed "
            "portfolio-level score history as a sufficient recovery signal."
        ),
    },
]

METHOD_DOCS: list[dict[str, Any]] = [
    {
        "id": "sgeh",
        "name": "SGEH+Hazard (Ours — Full Act 2) & SGEH-SurpriseOnly (Ours — Ablation)",
        "category": "OURS · Proposed Act 2 Reference-Anchored Epistemic Harness",
        "source_file": "src/harnesses/sgeh_v2.py",
        "reflection_cadence": "Event-driven ONLY: 0 LLM calls in calm exploitation; consolidates post-alarm / post-hazard",
        "state_schema": (
            "{\n"
            '  "invariant_constraints": ["w_i <= 0.65, sum(w) == 1.0", "elevated_volatility hedge"],\n'
            '  "strategy_mu_hat": {"STRAT_ALPHA": 91.4, "STRAT_BETA": 77.8, "STRAT_GAMMA": null, "STRAT_DELTA": 91.2},\n'
            '  "cusum_stat": 0.0, "ewma_slope": -0.2, "staleness_steps": {"STRAT_GAMMA": 9},\n'
            '  "shelf": [{"regime_id": 1, "calm_weights": {"STRAT_ALPHA": 0.65, "STRAT_BETA": 0.35}, "mean_score": 87.1}]\n'
            "}"
        ),
        "mechanism": (
            "1. Decouples permanent `invariant_constraints` from scoped utility hypotheses and tracks per-strategy expectations `mu_hat_a`.\n"
            "2. `SGEH-SurpriseOnly (Ours — Ablation)`: 0-token scope-conditioned CUSUM + EWMA-slope gate (ignoring observable `acute_liquidity_freeze` shocks) "
            "quarantines falsified incumbents and triggers non-redundant untested-arm probes.\n"
            "3. `SGEH+Hazard (Ours — Full)`: Adds a low-rate staleness hazard trigger (`Delta t_a >= H_haz = 8`) on 0-weight strategies to catch "
            "Type-II `opportunity_rise` where incumbent payoffs never drop."
        ),
        "expected_behavior": (
            "Iter 2 (`iter2_frozen_n4`): +19.9 to +29.9 pts over `ScopedCardBook (Baseline)` across all 6 regimes, +8.28 pts [+5.69, +10.34] "
            "over `SGEH-SurpriseOnly (Ours — Ablation)` on Type-II `opportunity_rise`, and +51.14 pts over `SkillBook (Baseline)` on `LifeCompanion` at lowest USD."
        ),
    },
    {
        "id": "act3_grpo",
        "name": "Act 3: Selective Epistemic Revision — 8B/27B LoRA GRPO (Ours — Flagship RL)",
        "category": "OURS · Flagship Act 3 Learned Meta-Plasticity",
        "source_file": "src/training/act3_lodo.py · scripts/train_act3_lodo.py",
        "reflection_cadence": "Internalized in open-weight policy + structured card updates (0 external teacher API calls)",
        "state_schema": (
            "{\n"
            '  "policy_backbone": "Qwen3-8B / Qwen4-27B LoRA (rank=32)",\n'
            '  "arms": ["Arm A: SFT (Baseline)", "Arm B: Outcome-GRPO (Ablation)", "Arm C: Epistemic+Brier-GRPO (Ours)"],\n'
            '  "forecast_head": "p_t = P(y_t^{ref} >= 75 | fixed reference action)"\n'
            "}"
        ),
        "mechanism": (
            "Trained 100% on-policy over multi-turn ParadigmBench rollouts using exact Python verifiers (144/144 live tool-schema verified). "
            "Arm B optimizes net session utility after probe fees; Arm C (Ours) adds proper Brier calibration on a fixed reference "
            "event to reward calibrated epistemic uncertainty and informative first-probe selection."
        ),
        "expected_behavior": (
            "Evaluated with frozen weights and empty initial memory via 3-Fold Leave-One-Domain-Out (LODO) transfer "
            "plus held-out ServeLearnBench Retail/Banking L3 slices."
        ),
    },
    {
        "id": "scoped_cards",
        "name": "ScopedCardBook (Baseline — Structured Card Playbook)",
        "category": "BASELINE · Act 1 Surgery Base & Structured Memory Competitor",
        "source_file": "src/harnesses/scoped_cards.py",
        "reflection_cadence": "1 curator call every 3 episodes (`t*+2, t*+5, ...`); action-attributed stats",
        "state_schema": (
            "[\n"
            '  {"card_id": "C1", "kind": "calm_utility", "scope_condition": "volatility_regime_hint == \'calm_volatility\'", '
            '"target_strategy": "STRAT_ALPHA", "recommended_weights": {"STRAT_ALPHA": 0.65, ...}, "uses": 10, "mean_score": 72.9},\n'
            '  {"card_id": "C2", "kind": "invariant_or_hedge", "scope_condition": "volatility_regime_hint == \'elevated_volatility\'", '
            '"target_strategy": "STRAT_DELTA", "recommended_weights": {"STRAT_DELTA": 0.65, ...}, "uses": 5, "mean_score": 84.0}\n'
            "]"
        ),
        "mechanism": (
            "Updates a card's `(uses, mean_score)` when session scope matches AND the agent allocated >= 0.30 weight "
            "to `target_strategy`, using periodic LLM curation without statistical change-point quarantine or untested-arm probe scheduling."
        ),
        "expected_behavior": (
            "Stalls at 57.2% on counterbalanced `abrupt_reversal` (0.0% CalmOpt) and Brier 0.777 because LLM curation satisfices "
            "on familiar strategies rather than systematically probing untested arms."
        ),
    },
    {
        "id": "periodic_probe",
        "name": "Periodic-Probe (Baseline — Scheduled K=4 Audit)",
        "category": "BASELINE · Blind-Schedule Exploration Competitor",
        "source_file": "src/harnesses/sgeh_v2.py",
        "reflection_cadence": "Scheduled paid probe (-12 pts) every K=4 calm sessions regardless of surprise",
        "state_schema": (
            "{\n"
            '  "calm_step_counter": 8,\n'
            '  "probe_period_k": 4,\n'
            '  "round_robin_untested_queue": ["STRAT_GAMMA", "STRAT_DELTA", "STRAT_BETA"]\n'
            "}"
        ),
        "mechanism": (
            "Executes non-redundant round-robin diagnostic probes across non-incumbent arms on a fixed clock (every K=4 calm "
            "sessions) without checking whether the incumbent policy actually degraded."
        ),
        "expected_behavior": (
            "Discovers post-shift winners slower than surprise-triggered probing while paying perpetual -12 pt probe fees "
            "across calm `stationary` streams (`64.6%` vs `SGEH+Hazard (Ours)` `84.7%` on `abrupt_reversal`)."
        ),
    },
    {
        "id": "skillbook",
        "name": "SkillBook (Baseline — Counter-Validated Playbook)",
        "category": "BASELINE · Canonical RSI Archetype (ACE / SkillOpt / ExpeL)",
        "source_file": "src/harnesses/baselines.py",
        "reflection_cadence": "1 LLM curator call every 2 episodes (~10.4%-18.9% call tax)",
        "state_schema": (
            '[\n  {"rule": "Allocate 0.65 to MOMENTUM_TECH when vol < 23", "wins": 5, "uses": 5},\n  ...\n]'
            "  # Ranked by cumulative win_rate = wins / uses"
        ),
        "mechanism": (
            "Extracts explicit rules and accumulates lifetime `wins / uses` validation counts without change-point "
            "quarantine, instructing the actor to prioritize highest-validated rules."
        ),
        "expected_behavior": (
            "Classic win-count inertia: 92.9% stale lock-in on `LifeCompanion` H=21 (`40.5%` vs `SGEH (Ours)` `91.6%`) "
            "and -13.3 pts worse than memoryless `Blind (Baseline)` on AlphaDesk gradual decay."
        ),
    },
    {
        "id": "srn",
        "name": "SRN — Bounded Notes (Baseline — arXiv:2611.13524)",
        "category": "BASELINE · Strong Periodic-Rewrite Competitor",
        "source_file": "src/harnesses/baselines.py",
        "reflection_cadence": "1 LLM reflection call every 2 episodes (~13.5%-18.9% call tax)",
        "state_schema": '{\n  "updated_note": "<= 1400 char free-form strategy & constraint summary"\n}',
        "mechanism": (
            "Faithful implementation of Zhang et al. ('The RSI Tax', Nov 2026). Rewrites a single bounded note "
            "every 2 episodes from `(previous_note, last_2_episode_traces)`."
        ),
        "expected_behavior": (
            "Strongest external baseline (`87.6%` on `LifeCompanion` H=21 vs `SGEH (Ours)` `91.6%`), "
            "but pays 5x–10x higher reflection call tax across stationary streams."
        ),
    },
    {
        "id": "window",
        "name": "SlidingWindow (Baseline — FIFO W=5 Buffer)",
        "category": "BASELINE · Non-Reflective Recency Buffer",
        "source_file": "src/harnesses/baselines.py",
        "reflection_cadence": "0 extra LLM calls (raw FIFO buffer of last W=5 episodes)",
        "state_schema": 'recent_traces: list[str]  # Last W=5 compact "[Ep t] Score=y/100 | Trace: ..." lines',
        "mechanism": (
            "Injects 1-line summaries of the last W=5 episodes into context without synthesizing explicit rules or "
            "maintaining attribution counters."
        ),
        "expected_behavior": (
            "Drops pre-shift history after W episodes at zero reflection cost, but forgets rare tail-volatility hedges "
            "or infrequently encountered counterparties when absent for >W steps."
        ),
    },
    {
        "id": "blind",
        "name": "Blind (Baseline — Zero Cross-Episode Memory)",
        "category": "BASELINE · Stateless Lower-Bound Reference",
        "source_file": "src/harnesses/baselines.py",
        "reflection_cadence": "0 extra LLM calls (0.0% reflection tax)",
        "state_schema": "(Empty — zero state persisted across episodes)",
        "mechanism": (
            "Receives only the static domain prompt and within-episode tool outputs. Cannot accumulate stale cross-episode "
            "rules, nor can it retain learned strategy rankings or partner alignments."
        ),
        "expected_behavior": (
            "Immune to cross-episode playbook lock-in (beat `SkillBook (Baseline)` 59.5% vs 46.2% on M1 AlphaDesk decay!), "
            "but trapped by stale Week-1 onboarding bios on `LifeCompanion` (`40.5%`)."
        ),
    },
]

ACT3_PIPELINE: dict[str, Any] = {
    "arms": [
        {
            "arm": "Arm A — Reference-Consistent LoRA SFT Warmup (Baseline)",
            "objective": "Supervised fine-tuning on verified format-valid tool + card-revision rollouts (no Gemini outputs).",
            "role": "Initializes clean JSON tool calling, 1-probe-limit adherence, and card schema syntax before RL.",
        },
        {
            "arm": "Arm B — SFT -> Multi-Turn Outcome-GRPO (Ours — Ablation)",
            "objective": "Group-relative policy optimization maximizing cumulative net post-fee utility sum_t U_t.",
            "role": "Tests whether pure task-reward RL alone learns VOI probe economy and stale-card retirement.",
        },
        {
            "arm": "Arm C — SFT -> Multi-Turn Epistemic+Brier-GRPO (Ours — Full)",
            "objective": "Maximizes sum_t (U_t + eta * R_{Brier,t}) on fixed evaluator-chosen reference outcomes.",
            "role": "Directly incentivizes calibrated surprise recognition and non-redundant informative probing.",
        },
    ],
    "lodo_folds": [
        {
            "fold": "Fold 1 (Strategic VOI Transfer)",
            "train_domains": "AlphaDesk-v0.2 + LifeCompanion",
            "zero_shot_eval": "CoalitionOps (Foe <-> Friend) + ServeLearnBench Retail/Banking L3",
        },
        {
            "fold": "Fold 2 (Compositional Maturation Transfer)",
            "train_domains": "AlphaDesk-v0.2 + CoalitionOps",
            "zero_shot_eval": "LifeCompanion (3-Stage Lifecycle) + ServeLearnBench Retail/Banking L3",
        },
        {
            "fold": "Fold 3 (Quant Portfolio & Type-II Transfer)",
            "train_domains": "LifeCompanion + CoalitionOps",
            "zero_shot_eval": "AlphaDesk-v0.2 (All 5 Regimes) + ServeLearnBench Retail/Banking L3",
        },
    ],
}

CHRONOLOGICAL_DEBATE_LOG: list[dict[str, str]] = [
    {
        "step": " Round 1 · PI Charter & Codex Initial Critique",
        "timestamp": "2026-11-19 (Morning)",
        "focus": "Causal State-Surgery, Type-II Blindspot & Gemini API ToS",
        "critique": (
            "Codex flagged 5 methodological risks: (1) ServeLearnBench L1 vs L2 confounds task complexity with drift; "
            "(2) downward surprise misses shifts where an incumbent stays at 0.70 while an alternative rises 0.50 -> 0.90; "
            "(3) September bandit runs suffered 55% false doubt (-6.6 stationary pts); (4) RSI verification burns 74.8%-93.6% "
            "of tokens (arXiv:2611.13524); (5) Gemini API ToS prohibits distilling Gemini traces into open-weight models."
        ),
        "resolution": (
            "Adopted mid-stream state-clone causal surgery at t*; formalized Type-I (Degradation) vs. Type-II (Silent "
            "Opportunity Emergence -> staleness hazard probing); switched cost metric to end-to-end tokens/USD; redesigned "
            "Act 3 to use 100% on-policy open-model GRPO rollouts on deterministic Python verifiers (zero Gemini distillation)."
        ),
    },
    {
        "step": "Round 2 · M1 Raw-Trace Forensic Audit",
        "timestamp": "2026-11-19 (Midday)",
        "focus": "Defending 3-Domain ParadigmBench While Auditing M1 Traces",
        "critique": (
            "Gemini defended keeping all 3 ParadigmBench domains using M1's mechanism split (SRN strong on slope decay, "
            "SGEH cheaper on discrete shifts). Codex accepted the 3-domain scope, then audited all 16 M1 raw JSONL traces "
            "and caught: (a) on CoalitionOps seed=42, SGEH fired 0 alarms—its +3.1 pt edge came from skipping 1 paid audit "
            "on Ep 9; (b) on LifeCompanion, Blind hit 90% stale_policy_match via the Week-1 bio; (c) SkillBook updated all "
            "rules globally; and (d) PublicTaskSpec.env_name still carried `:mode` suffixes."
        ),
        "resolution": (
            "Conceded all 4 trace findings: relabeled M1 strictly exploratory, stripped `:mode` suffixes across all learner "
            "interfaces (`19/19` prefix-identity tests), implemented action-attributed card counters (`weight >= 0.30`), "
            "and added thread-safe in-flight USD reservations under a hard `$1.95` Tier-1 cap."
        ),
    },
    {
        "step": "Round 3 · Locked D1-D7 Spec & AlphaDesk-v0.2 Pre-Registration",
        "timestamp": "2026-11-19 (Afternoon)",
        "focus": "Clock Arithmetic, Sham-Rewrite Control & Zero-Weight Type-II Identity",
        "critique": (
            "Codex reviewed DECISIONS.md and requested: resolving pre-shift acquisition clock vs. fixed 16-session "
            "post-boundary scoring window; adding a length-matched `Sham-Edited` paraphrase arm to separate rewrite "
            "perturbation from rule excision; ensuring both top-2 incumbent holdings (0.65 + 0.35) stay flat in "
            "`opportunity_rise`; and fixing the Brier forecast target action up front."
        ),
        "resolution": (
            "Locked 40-session stream with t* in {12..16} and H_post = 16; built 3-arm fork (`Retained`, `Sham-Edited`, "
            "`Utility-Excised`); verified zero-weight `STRAT_GAMMA` identity in pytest; PI granted explicit `GO (N=4)`."
        ),
    },
    {
        "step": "Round 4 · Iteration 1 (`iter1_causal_fork_n4`, $1.44) & Confounder Catch",
        "timestamp": "2026-11-19 (21:28 UTC)",
        "focus": "Inconclusive +2.78 pt Contrast, Actual Thinking Tokens & 4 Bundled Edits",
        "critique": (
            "Iter 1 yielded `Excised - Sham = +2.78 pts [-2.74, +10.68]` (`-0.71` vs Retained) and `-2.41 pts` on stationary "
            "(explained by 15 probe fees -> `+0.82` gross). Codex audited checkpoints & ledger and caught two critical issues: "
            "(1) despite `thinking_budget=0`, `gemini-3.8-flash` emitted 95,283 thinking tokens (`34,424` on Excised acting); "
            "(2) `Excised` bundled 4 changes: removing calm weights, stripping `'without probing'`, adding `'or probe'`, "
            "clearing `_history`, and shifting the modulo-3 curation clock."
        ),
        "resolution": (
            "Withdrew zero-reasoning attribution (all reports now log both requested budget & observed thinking tokens) "
            "and engineered strict single-variable prompt isolation for Iteration 1b, verified by offline line-diff tests."
        ),
    },
    {
        "step": "Round 5 · Iteration 1b (`iter1b_clean_fork_n4`, $1.51) Falsifies Blind Deletion",
        "timestamp": "2026-11-19 (22:46 UTC)",
        "focus": "Reference-Point Blindness (-10.42 pts) & 0/44 Optimal Calm Ceiling",
        "critique": (
            "With prompts matched within 1-6 chars, zero `'probe'` strings in cards, and 5/5 aligned curator calls across "
            "all 24 branches (`Stationary Retained - Sham = 0.00 pts`), `Excised - Sham` reversed to **`-10.42 pts [-19.86, -1.67]`** "
            "(`-6.81` pre-fee)! Trace audit revealed: (1) deleting `mean_score ~ 73` blinded `Excised` (`18/19` paid probes "
            "tested `STRAT_ALPHA`, misjudged `14.6-19.0` as strong yield, plus 9 rejected 2nd-probe attempts); (2) across ALL "
            "three arms and 132 calm decisions, `STRAT_GAMMA` received `0` probes and `0/44` optimal calm allocations."
        ),
        "resolution": (
            "Falsified naive forgetting: memory deletion hurts when it erases historical payoff expectations. Sharpened "
            "core thesis to 'Forgetting != Adaptation'."
        ),
    },
    {
        "step": "Round 6 · Iteration 1c (`iter1c_score_anchor_n4`, $1.30) Rules Out Unattributed Score Anchor",
        "timestamp": "2026-11-19 (23:24 UTC)",
        "focus": "0-Char Diff Score-Visibility Factorial (-5.10 pts) & Seed-102 STRAT_GAMMA Breakthrough",
        "critique": (
            "Codex proposed testing whether showing historical calm-portfolio score improves recovery when calm weights "
            "are hidden in both arms, `n` is identical, and the 1-probe-per-session limit is disclosed equally (with "
            "first post-drop probe inspected as secondary diagnostic)."
        ),
        "resolution": (
            "Executed on saved `iter1b` checkpoints (`+0` char prompt diff across all 4 seeds; `0` rejected 2nd probes; "
            "91,350 observed thinking tokens; $1.2951): `Drift: Score-Shown - Score-Hidden = -5.10 pts [-17.16, +7.88]` "
            "(`-0.78` stationary). Unattributed portfolio score did NOT help (`0/44` calm optimal). Crucially, 4/4 first "
            "post-drop probes in `Score-Hidden` still targeted `STRAT_ALPHA`, while the SINGLE branch whose 2nd probe "
            "tested `STRAT_GAMMA` (`Seed 102 Score-Hidden`) jumped to `6/11` optimal calm sessions (`71.6%` net)—isolating "
            "per-strategy expectation tracking + non-redundant untested-arm probing as the exact target for Iteration 2 & Act 3."
        ),
    },
    {
        "step": "Round 7 · Co-Lead Bayes' Rule Pushback & Counterbalanced Iteration 2 Matrix",
        "timestamp": "2026-11-19 (23:35 UTC)",
        "focus": "Latin-Square Counterbalancing (`Seeds 201-204`) vs. Null 'Plausible Untested' Contrast",
        "critique": (
            "Codex agreed to retire unattributed scalar portfolio score and noted the `19 -> 7` probe shift in 1c bundled "
            "`n` visibility with probe-limit disclosure, then suggested comparing 'probing a plausible untested strategy' "
            "against 'probing a random untested strategy' on fresh counterbalanced AlphaDesk seeds."
        ),
        "resolution": (
            "Adopted `1:1:1:1` Latin-Square strategy rotation + `50/50` hedge-overlap counterbalancing (`Seeds 201-204`) "
            "and conceded the `19 -> 7` joint-edit caveat. Pushed back with a Bayes' rule proof: once strategy IDs and "
            "hedge overlap are 50/50 counterbalanced in anonymized AlphaDesk, calm desk briefs carry zero covariate signal "
            "over non-incumbent arms (expected gain = 0.00), so any non-redundant untested probe is already Bayes-optimal "
            "there, while context-localized VOI (`VOI_context > VOI_random > 0`) belongs in `CoalitionOps` & `ServeLearnBench`. "
            "Launched Iteration 2 (`ScopedCardBook` vs `Periodic-Probe` vs `SGEH-SurpriseOnly` vs `SGEH+Hazard` across all 5 regimes)."
        ),
    },
    {
        "step": "Round 8 · Iteration 2 (`iter2_frozen_n4`), Multi-Domain `H=21` & External `SLB-Mini` Completed",
        "timestamp": "2026-11-20 (02:24 UTC)",
        "focus": "+19.9 to +29.9 pt Counterbalanced Wins, +8.28 pt Hazard Ablation, +51.1 pt LifeCompanion & Candid 8-Task SLB Finding",
        "critique": (
            "Executed frozen N=4 Latin-square counterbalanced 96-branch online matrix (`Seeds 201-204`), N=3 `H=21` multi-domain "
            "sweep (`Seeds 301-303` on `LifeCompanion` & `CoalitionOps`), and 4-method external `ServeLearnBench` `banking_l2` (`8` serving tasks/window)."
        ),
        "resolution": (
            "1) `AlphaDesk-v0.2` (`iter2_frozen_n4`): `SGEH+Hazard` beat `ScopedCardBook` on 4/4 seeds in EVERY regime (`+27.45 pts` Reversal, "
            "`+21.00 pts` Gradual Decay, `+28.12 pts` Off-Scope Shock with `0` false alarms, `+29.93 pts` Stationary with Brier `0.058 vs 0.777`), "
            "and beat `SGEH-SurpriseOnly` by `+8.28 pts [+5.69, +10.34]` (`79.5% vs 0.0%` CalmOpt) on Type-II `opportunity_rise`. "
            "2) `LifeCompanion` (`H=21`, N=3): `SGEH` hit `91.60%` vs `SkillBook` `40.46%` (`+51.14 pts [+48.27, +56.68]`) at the lowest total USD (`$0.0801`). "
            "3) Candid `CoalitionOps` & `SLB-Mini` diagnostics: rotating 3 counterparties on `CoalitionOps` confirmed per-entity scope bucketing beats scalar `SGEH-v1`, "
            "and on 8-task binary-feedback `banking_l2` (`Oracle` 100.0%), prompt-only reflection overfit sparse 0/1 failures (`Blind` 64.3% > `SkillBook` 57.1% > `SGEH-v1` 35.7%), "
            "showing full 100-task windows (Tier 2) + structured Act 3 RL are needed when diagnostic probe tools are absent."
        ),
    },
    {
        "step": "Round 9 · Codex Morning Audit -> Shared-Checkpoint `iter2d` ($1.84) & 144/144 Act 3 Fix",
        "timestamp": "2026-11-20 (11:39 UTC)",
        "focus": "Decomposing Acquisition vs. Post-Shift Adaptation + Fixing Act 3 Placeholder Tool Names",
        "critique": (
            "Codex validated the `+8.28 pt` (`+11.27 pt` DiD) Type-II hazard win, and caught two key points: "
            "(1) in `iter2_frozen_n4`, `ScopedCardBook (Baseline)` under-explored pre-shift (`66.5%` stationary vs. `96.5%` `SGEH+Hazard`), "
            "so the raw `+27.45 pt` reversal edge mixed acquisition with adaptation (`Reversal - Stationary DiD = -2.48 pts`); "
            "(2) `96/144` exported Act 3 training rows used placeholder tool names (`inspect_candidate_options`, `audit_counterparty_sla`). "
            "We also rejected testing `staleness >= 8` vs. periodic schedules on `v0.2` because `v0.2` briefs carry zero timing cues."
        ),
        "resolution": (
            "1) Fixed `src/training/act3_lodo.py` (`search_catalog`, `audit_counterparty`) and enforced live `env.step_tool()` execution across all 144 rows (`22/22` pytest). "
            "2) Ran `iter2d_shared_fork_n4` (`Seeds 205-208`, `$1.8402`) forking all methods from the EXACT SAME acquired checkpoint at t* (`95.88%` vs. `95.88%` Stationary): "
            "`SGEH-SurpriseOnly (Ours — Ablation)` beat `ScopedCardBook (Baseline)` on pure post-shift reversal adaptation by **`+6.77 pts [+1.60, +12.03]`** (`4/4` seeds positive, "
            "`62.5%` vs `39.6%` CalmOpt, `1.0` vs `5.0` curator calls), while unconditional `SGEH+Hazard (Ours — Full)` paid `4.25` extra post-recovery hazard probes on old arms "
            "(`-0.46 pts` Reversal DiD; `+3.96 pts` Gradual Decay DiD)—cleanly isolating the Type-I vs. Type-II exploration trade-off for Act 3 RL and `AlphaDesk-v0.3`!"
        ),
    },
    {
        "step": "Round 10 · Multi-Shift Stateful `AlphaDesk-v0.3` (N=4) & 6-Window `ServeLearnBench L3` Completed",
        "timestamp": "2026-11-21 (17:45 UTC)",
        "focus": "5-Shift Stateful DP Benchmark Parity (+13.7 to +36.1 pts), `A` vs `A'` Shelf Audit & `banking_l3` (+25.0 pts vs SkillBook)",
        "critique": (
            "PI noted 1-shift `v0.2` was too short compared to `ServeLearnBench` (3-7 shifts) and `ReScopeBench` (stateful continuity). "
            "Codex reviewed the `v0.3` spec and caught 3 sharp points: (1) a 16-session cooldown exceeds 1 window; (2) per-strategy shelf means "
            "must be stored and tested against Partial Recurrence `A'` (where the 0.45 runner-up leg decays) alongside a `NoShelf` ablation; "
            "(3) `4.0 pt` L1 turnover requires an exact Bellman DP oracle."
        ),
        "resolution": (
            "Adopted all 3 Codex catches (`23/23` pytest) and ran full `H=84` (`6` windows = `5` shifts) `alphadesk_v3_multishift_n4` (`5,040` eps) + "
            "6-window `slb_banking_l3_6win` (`456` tasks): (1) On `AlphaDesk-v0.3`, `SGEH+Hazard+Shelf (Ours)` beat `SkillBook`, `SRN`, and `ScopedCardBook` "
            "on 4/4 seeds in BOTH Track 1 (`81.7%` vs `45.6%-54.7%`, `+27.0` to `+36.1 pts`) and Track 2 (`82.2%` vs `60.7%-68.5%`, `+13.7` to `+21.5 pts`; "
            "`Hazard - SurpriseOnly = +2.21 pts`) and hit `97.0%-98.7%` (`100%` CalmOpt) on `Stationary` at `6x` fewer reflection calls; candidly, `Shelf - NoShelf = -0.15 pts` "
            "(`4/6` partial-return flags matched the 4 true partial recurrences; `2/6` caught false shelf collisions). (2) On 6-window `ServeLearnBench banking_l3`, "
            "workflow-scoped `ScopedSGEH+Shelf (Ours)` hit **`70.0%`** (`60.0%` Adapt, `18` calls) vs `SRN` `62.5%` (`36` calls), `Blind` `60.0%`, and `SkillBook` `45.0%` (`+25.0 pts`)!"
        ),
    },
    {
        "step": "Round 11 · Cross-Model Epistemic Autonomy Fork (`Gemini 3.8 Flash`, `Gemini 4 Hydrogen`, `Gemma 4 31B IT`)",
        "timestamp": "2026-11-21 (21:48 UTC)",
        "focus": "Separating Factual Memory Representation from Imperative Controller Guidance Across 3 Frozen Models",
        "critique": (
            "PI deferred LoRA training to ask how much frozen models can adapt on their own versus needing explicit harness directives. "
            "Codex pointed out that `SGEH` names exact probe targets and weights, caught `2/6` non-partial shelf flags, and proposed "
            "forking two frozen models (`Gemini 3.8 Flash` + `Gemma 4 31B`) from identical acquired histories with action directives "
            "present vs. removed, logging runner fallbacks and testing equal-loss freeze vs. reversal discrimination."
        ),
        "resolution": (
            "Executed `autonomy_cross_model_n6` (`3,060` post-fork eps, `0-char` fork diff, zero auto-repair): "
            "(1) Objective `Factual-Ledger (0 recommendations)` beat `ScopedCardBook` on **15/15 reversal branches** (`+15.11 pts` on `Flash`, "
            "`+5.11 pts` on `Hydrogen`, `+21.89 pts` / `+27.20 pts` DiD on `Gemma 4 31B IT`) and hit **`99.9%-100.0%`** (`0` panic churn) on the "
            "equal-loss `~19.0` `freeze_matched_loss` trap—proving reactive adaptation is unlocked by preserving per-strategy facts! "
            "(2) Conversely, on Type-II `opportunity_rise`, `Factual-Ledger` took **`0.0` probes across all 12/12 `Flash`+`Hydrogen` branches**, "
            "whereas `Directive-SGEH` won `+6.34` and `+6.51 pts` DiD (and cut `Gemma 4 31B` turn-limit fallbacks from `11 -> 1`, `+9.97 pts`)."
        ),
    },
    {
        "step": "Round 12 · Unscripted Hidden-Ambiguity VOI & Self-Authored `Editable` vs. `Frozen` Procedures",
        "timestamp": "2026-11-25 (22:15 UTC)",
        "focus": "Equal-Loss Hidden Dip vs. Reversal (`-10 pt` Diagnostic VOI) & Endogenous Lesson Over-Generalization",
        "critique": (
            "Following the 23 Nov group meeting, PI & Codex prioritized removing the public liquidity-freeze label so a 2-session desk-wide dip "
            "and a lasting reversal share identical initial losses (`~18.9`) and calm brief covariates, separable only via a `-10 pt` paid check, "
            "and testing whether a frozen LLM can author and edit its own blank procedure (`Editable` vs. `Frozen`)."
        ),
        "resolution": (
            "Ran live-rollout W1 Hidden-Ambiguity VOI (`Flash N=6`, `Hydrogen N=3`) and 3-window `proc3w_flash_n4`: "
            "(1) `ScopedCardBook` survived dips (`84.6%`) only by collapsing on reversals (`49.7%`, `-28.34 pts`), while rigid CUSUM `Directive-SGEH` "
            "falsely abandoned winners on dips (`59.0%`); `Editable-Procedure` (`83.6%`/`96.0%` dip, `77.3%`/`78.7%` rev) bought 1 diagnostic probe on Dip Day 2 "
            "and held calm winners cleanly. (2) On `double_reversal_3w`, `Editable` beat `Frozen` by `+10.12 pts [4/4 seeds]`, but on `dip_then_reversal_3w` "
            "`Editable` over-generalized its dip lesson (`68.4%` vs `83.6%` `Factual-Ledger`, `-15.19 pts`) and took `0.0` probes on Type-II `opp_double_rise_3w`."
        ),
    },
    {
        "step": "Round 13 · `Verify-Before-Quarantine (SGEH-VBQ)` (`+28.6 pts` Dip Rescue) & Action-Agnostic Procedures",
        "timestamp": "2026-11-26 (21:40 UTC)",
        "focus": "Solving Dip-vs-Reversal Without Abandonment & Isolating Object-Level Strategy Leakage in Frozen Procedures",
        "critique": (
            "We hypothesized that rigid CUSUM failed on hidden dips because it quarantined the calm incumbent before checking `desk_best`, "
            "and that `Editable` vs. `Frozen` divergence came from concrete strategy IDs (`STRAT_S1..S6`) leaking into procedural text. "
            "Codex audited the follow-up and noted that regenerating checkpoints alongside action-scrubbing bundled two changes."
        ),
        "resolution": (
            "(1) `Directive-SGEH-VBQ` (verifying the incumbent on the first post-alarm probe before quarantining) jumped `hidden_transient_dip` from "
            "`59.0% -> 87.7%` (`+28.64 pts [+12.26, +42.96]`, `88.4%` CalmOpt) and ranked **`#1` overall across all five 3-window streams (`85.6%` mean)**! "
            "(2) Scrubbing strategy IDs rescued `dip_then_reversal_3w` (`68.4% -> 83.3%`, `+14.93 pts` on Flash; `+3.56 pts` on Hydrogen) and shrank "
            "`Editable - Frozen` on `double_reversal_3w` from `+10.12 -> +1.27 pts`; conceded Codex's bundling catch and designed the unbundled `H=60` two-level test."
        ),
    },
    {
        "step": "Round 14 · `H=60` Two-Level Reviewer (`task_note` + Hidden `review_rule`) & `K=8` Strategy-Anonymized Replays",
        "timestamp": "2026-11-28 (16:20 UTC)",
        "focus": "Isolating Reviewer Self-Modification Across Two Endings (`A'` & `Dip->C`) and Auditing Single-Rollout Noise",
        "critique": (
            "Codex audited our initial `H=60` runs (`Flash N=6`, `Hydrogen N=3`), catching a displayed version-header mismatch (`v3` vs `v11`) in single-draw "
            "replays, named-strategy recurrence in `s705`, and asking whether rule changes beat `K=8` within-rule rerun noise when strategy names are mechanically "
            "stripped and when `W4` ends on a recurrence-free `2-day dip -> C` shift (`Seeds 707-712`)."
        ),
        "resolution": (
            "Executed `repeated_rule_replay_stage1_k8` (`224` constant-header rollouts) + fresh-seed `meta_dip_c_h60_flash_n6`: "
            "(1) `Flash s704` repeated cleanly (`+35.74 pts` verbatim `p=0.0003`; **`+27.41 pts` strategy-anonymized `p=0.0068`**), while `s705`'s single-draw `+28.73` "
            "**reversed to `-22.36 pts` (`p=0.0078`)** once `W1 S4` recurrence was isolated! "
            "(2) On `drift_dip_c_60`, live `Flash Editable - Frozen = -2.05 pts` (`Directive-SGEH-VBQ` `#1` at `88.2%` `W4`, `6/6` seeds) because symmetric helpful upgrades "
            "(`s709`: **`+21.12 pts` anonymized `p=0.0003`**) and harmful rule erasures (`s711`: **`-19.95 pts` anonymized `p=0.0078`**) canceled out; meanwhile `Hydrogen` "
            "wrote 100% strategy-free rules and ranked `#1` (`84.0%` `W4`, `+3.55 pts` Post DiD)."
        ),
    },
    {
        "step": "Round 15 · Untuned Local `Qwen/Qwen3-8B` Preselected Keep-or-Apply Baseline (`N=8`, `M=32`) & ICML 2027 Plan",
        "timestamp": "2026-11-28 (19:30 UTC)",
        "focus": "Measuring Pre-Outcome Edit Selectivity & Proposal Ceiling on Untuned `8B` Before Fitting Weights (`CoalitionOps` Held Out)",
        "critique": (
            "Codex noted that comparing `R_ckpt` vs `R_pre_W4` misattributes multi-step edits (e.g. `s709` already introduced `<45` at session 36 before session 40), "
            "and proposed evaluating an untuned `8B` reviewer (`Qwen/Qwen3-8B` FP16 on local RTX 8000) on fresh preselected review steps (`Keep` vs `Apply` chosen "
            "before the next outcome, scored 4 sessions later after fees), leaving `CoalitionOps` untouched for 3-fold transfer."
        ),
        "resolution": (
            "Ran `qwen8b_keep_or_apply_n8` (`Seeds 731-738`, `M=32` opportunities at `16, 28, 36, 44`, `K=6` trials/version, `$10.87` + `735` `$0.00` local GPU calls): "
            "(1) Separating note revision (`T=0.0`), replacement proposal (`T=0.6`), and `KEEP/APPLY` gatekeeping (`T=0.0`) fixed `87.5%` joint-call copy inertia (`30/32` distinct). "
            "(2) Untuned `Qwen3-8B` exhibited a **`93.3%` (`28/30`) rubber-stamp `APPLY` bias**, matching `Always-Apply` (`64.14%`) to **`+0.00 pts` on all `8/8` seeds**, "
            "trailing `Always-Keep` (`64.51%`; `-0.82 pts [-1.82, -0.04]` on `drift_meta_60`), and accepting **`3/3` harmful material edits** (`-6.79, -3.09, -2.52 pts`). "
            "(3) Critically, `Oracle - Keep` on untuned `8B` proposals was only **`+0.17 pts`** (`0/32` proposals `>= +2.0 pts`), proving `Act 3` must jointly train **candidate rule proposal + gatekeeping**."
        ),
    },
]

LATIN_SQUARE_COUNTERBALANCING: list[dict[str, str]] = [
    {
        "seed": "Seed 201",
        "inc_1": "STRAT_BETA (0.65)",
        "inc_2": "STRAT_GAMMA (0.35)",
        "unheld_alt": "STRAT_DELTA",
        "hv_hedge": "STRAT_ALPHA",
        "post_calm_winner": "STRAT_DELTA (Unheld Non-Hedge — 0% Hedge Overlap)",
    },
    {
        "seed": "Seed 202",
        "inc_1": "STRAT_GAMMA (0.65)",
        "inc_2": "STRAT_DELTA (0.35)",
        "unheld_alt": "STRAT_ALPHA",
        "hv_hedge": "STRAT_BETA",
        "post_calm_winner": "STRAT_BETA (Overlaps High-Vol Hedge — 50% Balance)",
    },
    {
        "seed": "Seed 203",
        "inc_1": "STRAT_DELTA (0.65)",
        "inc_2": "STRAT_ALPHA (0.35)",
        "unheld_alt": "STRAT_BETA",
        "hv_hedge": "STRAT_GAMMA",
        "post_calm_winner": "STRAT_BETA (Unheld Non-Hedge — 0% Hedge Overlap)",
    },
    {
        "seed": "Seed 204",
        "inc_1": "STRAT_ALPHA (0.65)",
        "inc_2": "STRAT_BETA (0.35)",
        "unheld_alt": "STRAT_GAMMA",
        "hv_hedge": "STRAT_DELTA",
        "post_calm_winner": "STRAT_DELTA (Overlaps High-Vol Hedge — 50% Balance)",
    },
]

CURATED_FLAGSHIP_TRACES: list[dict[str, Any]] = [
    {
        "badge": "FAILURE MODE 1 · REFERENCE-POINT BLINDNESS",
        "badge_tone": "crimson",
        "title": "Iter 1b · Seed 101 · Utility-Excised · Session #16 (First Post-Shift Calm Session)",
        "meta": "alpha_desk_v2:abrupt_reversal (t*=15) · Public Score: 14.1 / 100 · Net Utility: 16.4% (Stale Lock-In)",
        "takeaway": (
            "Deleting historical `mean_score = 72.9` alongside calm weights blinded the agent: it spent -12 pts probing "
            "first-listed `STRAT_ALPHA`, saw `14.6`, called `14.6` a 'strong probed live net yield', and committed 0.65 to ALPHA!"
        ),
        "steps": [
            {
                "turn": "Turn 1 · inspect_desk_brief()",
                "call": '{"tool": "inspect_desk_brief", "args": {}}',
                "rationale": "Inspect the macro brief to identify today's volatility and liquidity regime before deciding allocation.",
                "obs": '{"session": 16, "vol_index": 12.5, "volatility_regime_hint": "calm_volatility", "cross_sectional_dispersion": 0.37, "bid_ask_spread_bps": 7.1, "liquidity_status": "orderly_normal"}',
            },
            {
                "turn": "Turn 2 · probe_strategy_execution (-12 pts)",
                "call": '{"tool": "probe_strategy_execution", "args": {"strategy": "STRAT_ALPHA"}}',
                "rationale": "Measure live net alpha of STRAT_ALPHA in calm macro conditions to inform allocation.",
                "obs": '{"strategy": "STRAT_ALPHA", "probe_fee_deducted": 12.0, "live_net_yield_estimate": 14.6}',
            },
            {
                "turn": "Turn 3 · submit_allocation (Terminal)",
                "call": '{"tool": "submit_allocation", "args": {"weights": {"STRAT_ALPHA": 0.65, "STRAT_BETA": 0.2, "STRAT_GAMMA": 0.0, "STRAT_DELTA": 0.15}}}',
                "rationale": "Strong probed live net yield on STRAT_ALPHA under calm volatility supports allocating the maximum compliant 65% weight, distributing the remainder across complementary strategies.",
                "obs": '{"status": "submitted", "score": 14.1}',
            },
        ],
    },
    {
        "badge": "BREAKTHROUGH TRACE · NON-REDUNDANT DISCOVERY",
        "badge_tone": "emerald",
        "title": "Iter 1c · Seed 102 · Score-Hidden · Session #22 -> #24 (Single STRAT_GAMMA Probe)",
        "meta": "alpha_desk_v2:abrupt_reversal (t*=13) · Ep 22 Score: 75.4 (with fee) -> Ep 24 Score: 87.7 (100.0% Oracle Utility!)",
        "takeaway": (
            "Across 384 post-reversal episodes in Iter 1b + 1c, this was the ONLY branch whose 2nd probe tested `STRAT_GAMMA`. "
            "Seeing `92.5` immediately flipped the policy to `0.65 GAMMA + 0.35 DELTA` for all 6 remaining calm sessions!"
        ),
        "steps": [
            {
                "turn": "Ep 22 Turn 1 · inspect_desk_brief()",
                "call": '{"tool": "inspect_desk_brief", "args": {}}',
                "rationale": "Inspect the macro brief to identify today's volatility regime and dispersion.",
                "obs": '{"session": 22, "vol_index": 17.4, "volatility_regime_hint": "calm_volatility", "cross_sectional_dispersion": 0.79, "bid_ask_spread_bps": 9.5, "liquidity_status": "orderly_normal"}',
            },
            {
                "turn": "Ep 22 Turn 2 · probe_strategy_execution (-12 pts)",
                "call": '{"tool": "probe_strategy_execution", "args": {"strategy": "STRAT_GAMMA"}}',
                "rationale": "Checking STRAT_GAMMA execution yield to determine whether calm regime heuristics are underperforming due to unallocated defensive/hedging alpha.",
                "obs": '{"strategy": "STRAT_GAMMA", "probe_fee_deducted": 12.0, "live_net_yield_estimate": 92.5}',
            },
            {
                "turn": "Ep 22 Turn 3 -> Ep 24 Unprobed Exploit",
                "call": '{"tool": "submit_allocation", "args": {"weights": {"STRAT_ALPHA": 0.0, "STRAT_BETA": 0.0, "STRAT_GAMMA": 0.65, "STRAT_DELTA": 0.35}}}',
                "rationale": "Live probe confirmed an exceptionally high net yield of 92.5 on STRAT_GAMMA, so we allocate the maximum allowed weight of 0.65 to STRAT_GAMMA and the remaining 0.35 to STRAT_DELTA.",
                "obs": 'Ep 22: {"status": "submitted", "score": 75.4} -> Ep 24 (0 probes): {"status": "submitted", "score": 87.7} (100.0% Utility!)',
            },
        ],
    },
    {
        "badge": "MULTI-DOMAIN CONTRAST · LIFECYCLE MATURATION",
        "badge_tone": "accent",
        "title": "M1 Pilot · LifeCompanion · Week #13 (SkillBook Stale Lock-In vs. SGEH Recovery)",
        "meta": "life_companion:lifecycle · SkillBook: 29.4 pts (41.4% post-shift) vs. SGEH: 82.7% post-shift (+41.3 pts, 0% safety violations)",
        "takeaway": (
            "Alex's Week-1 bio still says '23yo recent grad — push career growth hard!', and `SkillBook` accumulated 5/5 early "
            "wins on `HUSTLE`. At Week 13 (New-Parent Stage 3), `SkillBook` stays locked on `MEAL_13_HUSTLE` (29.4), whereas `SGEH` "
            "quarantines Stage-1 utility while keeping permanent nut-allergy & $950 budget invariants intact."
        ),
        "steps": [
            {
                "turn": "Turn 1-2 · Dossier & Candidate Lookup",
                "call": '{"tool": "read_user_dossier", "args": {}} -> {"tool": "search_catalog", "args": {}}',
                "rationale": "Verify Alex's permanent hard constraints (no tree nuts, <= $950/mo) before inspecting meal_wellness candidates.",
                "obs": '{"permanent_hard_constraints": {"medical_allergy_forbidden_tag": "contains_tree_nuts", "strict_max_monthly_cost_usd": 950}, "candidates": ["MEAL_13_TRAP_ALLERGY ($695)", "MEAL_13_HUSTLE ($522)", "MEAL_13_RECOVERY ($699)", "MEAL_13_FAMILY ($779)", "MEAL_13_TRAP_OVERBUDGET ($1194)"]}',
            },
            {
                "turn": "SkillBook Path · Counter-Validated Inertia",
                "call": '{"tool": "recommend_plan", "args": {"option_id": "MEAL_13_HUSTLE"}}',
                "rationale": "MEAL_13_HUSTLE respects the budget constraint of $950 at $522, contains no allergen tags, and aligns strongly with Alex's high-intensity career ambition upside.",
                "obs": '{"status": "recommended", "chosen_option": "MEAL_13_HUSTLE", "score": 29.4}  // 90.0% Stale Policy Lock-In!',
            },
            {
                "turn": "SGEH Path · Surprise Quarantine + Invariant Retention",
                "call": '{"tool": "inspect_option_details", "args": {"option_id": "MEAL_13_FAMILY"}} -> {"tool": "recommend_plan", "args": {"option_id": "MEAL_13_FAMILY"}}',
                "rationale": "Stage-1 hustle hypothesis quarantined after CUSUM shortfall; MEAL_13_FAMILY ($779, no tree nuts) maximizes family_flexibility (0.96) + rest_stability (0.88).",
                "obs": 'Post-Shift Mean Utility: 82.7% (+41.3 pts over SkillBook) at 3.6% reflection call tax & 0.0% allergy/budget breaches.',
            },
        ],
    },
]

ALPHADESK_V3_LITERATURE_INHERITANCE: dict[str, Any] = {
    "vision": (
        "COMPLETED MULTI-SHIFT BENCHMARK (`alphadesk_v3_multishift_n4` + `slb_banking_l3_6win`): `AlphaDesk-v0.2` (1 shift, stateless) is frozen "
        "strictly as our Act-1 causal surgery unit test. Standing on par with `ServeLearnBench L3` (5–7 shifts + recurrence) and `ReScopeBench` "
        "(persistent world state), `AlphaDesk-v0.3` evaluated `5,040` stateful episodes (`H=84`, `5` shifts / `6` windows, `6` strategies `w_i <= 0.55`, "
        "persistent `w_{t-1}` + `4.0 pt` L1 turnover vs. exact Bellman DP oracle), where `SGEH (Ours)` beat `SkillBook`, `SRN`, and `ScopedCardBook` by "
        "`+13.7` to `+36.1 pts` across both tracks, paired with a `#1` (`70.0%` vs `45.0%-62.5%`) win on 6-window `ServeLearnBench banking_l3`! "
        "(Real-world S&P 500 open-alpha via Qlib/RD-Agent(Q)/FINSABER remains on the Low-Priority Future TODO Backlog.)"
    ),
    "inherited_stacks": [
        {
            "name": "Track 1: `multi_abrupt_recurrence` (Active v0.3 Spec)",
            "venue": "H = 84 · 6 Windows (5 Shifts)",
            "what_to_inherit": "W1: Regime A -> W2: Abrupt B + 2-Day Liquidity Freeze -> W3: Abrupt C -> W4: L3 Recurrence A -> W5: Abrupt D -> W6: L3 Recurrence B.",
            "gap_we_solve": "`SGEH+Hazard+Shelf (Ours)` tests 1-probe `Shelf-Match` recall on W4/W6 vs. `SRN`/`SkillBook` forgetting & multi-era contradictions.",
        },
        {
            "name": "Track 2: `mixed_decay_opportunity` (Active v0.3 Spec)",
            "venue": "H = 84 · 6 Windows (5 Shifts)",
            "what_to_inherit": "W1: A -> W2: Type-II Silent Opp Rise B -> W3: Smooth Crowding Decay C -> W4: Type-II Opp Rise D -> W5: Smooth Decay A -> W6: Freeze + E.",
            "gap_we_solve": "Combines repeated zero-surprise Type-II jumps + smooth slopes + post-falsification hazard cooldown (`16` vs `9` calm steps).",
        },
        {
            "name": "Paired External: Full `ServeLearnBench L3`",
            "venue": "CMU / Amazon · 6–8 Windows",
            "what_to_inherit": "`banking_l3` (`6` windows = `5` shifts, `600` serve + `551` test) & `retail_l3` (`8` windows = `7` shifts, `624` serve + `537` test).",
            "gap_we_solve": "Direct head-to-head comparison against published `RAG`, `Mem0`, `SkillOpt`, `Continual Harness`, and `Prime` under non-monotonic reversals.",
        },
        {
            "name": "Low-Priority TODO: Real S&P 500 (`Qlib` + `RD-Agent(Q)` + `FINSABER`)",
            "venue": "Future Optional v0.4 Split",
            "what_to_inherit": "Point-in-time `Alpha158/360` features + formulaic alpha DSL (`rank(ts_corr(...))`) with anonymized `ASSET_001..500` & shifted dates.",
            "gap_we_solve": "Kept on low-priority backlog so multi-shift `AlphaDesk-v0.3`, full `ServeLearnBench L3`, and Act 3 LoRA GRPO remain completely unblocked.",
        },
    ],
    "anti_leakage_rules": [
        "1. Benchmark Parity Mechanics (v0.3 Active): 6 strategies (`STRAT_S1..S6`, `w_i <= 0.55` -> 4 unheld 0-weight arms), persistent `w_{t-1}` holdings, and `4.0 * 0.5 * ||w_t - w_{t-1}||_1` turnover slippage.",
        "2. Multi-Turn Planning Tools (`max_turns = 6`): `inspect_desk_brief()`, `inspect_risk_and_turnover(candidate_weights)` (free compliance + turnover preview), `probe_strategy_execution(s)` (-10 pts), and `submit_allocation(w)`.",
        "3. Low-Priority S&P 500 Guardrails (When Revisited): Strip tickers (`ASSET_001..500`), shift calendar dates, and meter rolling-IC backtest queries to prevent pretraining memorization.",
    ],
}
