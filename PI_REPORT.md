# PI Project Charter: Technical & Budget Management Report

**Project Code:** `RSI-Paradigm-Shift` (`~/llm-time/`)  
**Date:** November 19, 2026  
**Target Submission Window:** ~December 19, 2026 (4-Week Sprint)  
**Working Title:** *When Self-Improvement Becomes an Anchor: Breaking the Epistemic Ratchet under Non-Stationary Reward Paradigms*

> **Core Executive Thesis:** Non-parametric self-evolving harnesses (`SkillOpt`, `Continual Harness`, `ACE`, `GEPA`, `SRN`) assume a stationary reward function—crystallizing past wins into persistent rules and validation scores. When **domain mechanics stay valid** (no tool crashes) and only the **latent reward/utility paradigm** shifts silently (alpha decay, lifecycle maturation, foe-to-friend reversal), self-improvement acts as an **anti-adaptation ratchet** that amplifies *hypothesis resurrection*. We diagnose this across model scales (Act 1), solve it at low token cost via a **Surprise-Gated Epistemic Harness** (Act 2), and internalize epistemic plasticity directly into open-weight `8B`/`27B` models via SFT + multi-turn GRPO (Act 3).

---

## 1. Technical Architecture & 3-Act Research Design

### 1.1 Evaluation Testbeds (Multi-Turn Tool-Calling Agents)

| Testbed | Source | Drift Dynamics Tested | Why We Need It |
| :--- | :--- | :--- | :--- |
| **Pillar 1: `ServeLearnBench`** | External (`competitors/ServeLearnBench`, Oct 2026) | Piecewise-constant window shifts: `L1` acquisition -> `L2` revision -> `L3` non-monotonic reversals | Instant external credibility + published numbers for `Mem0`, `RAG`, `SkillOpt`, `Continual Harness`, and `Prime` |
| **Pillar 2a: `AlphaDesk`** | Custom (`ParadigmBench` in `src/`) | **Continuous alpha decay / crowding** + macro sign flips (momentum <-> mean-reversion) + recurrence | Captures smooth reward erosion with no discrete boundary cliff (missed by both `ServeLearnBench` & `ReScopeBench`) |
| **Pillar 2b: `LifeCompanion`** | Custom (`ParadigmBench` in `src/`) | **Smooth longitudinal user maturation** (fixed personal constraints + shifting life-stage utility weights) | Punishes both *stale self-written user profiles* and *blind global forgetting* |
| **Pillar 2c: `CoalitionOps`** | Custom (`ParadigmBench` in `src/`) | **Silent strategic utility reversal** (zero-sum foe <-> positive-sum partner) | Forces **active diagnostic probing** to separate stochastic noise from true incentive inversion |
| **Matched Stationary Controls** | Paired across `ParadigmBench` | Identical stochastic observation noise with **zero** underlying paradigm shift | Prevents rewarding "twitchy" change-point priors (fixing the 55% false-alarm failure seen in archived `week2_revision_2026-09.md`) |

*(Note: A 1-panel analytical Multi-Armed Bandit diagram is held in reserve strictly as an optional Figure 1a conceptual teaser.)*

### 1.2 Scientific Questions Across the 3 Acts

1. **Act 1 — Capability Scaling vs. The RSI Ratchet:**
   - Rank performance across the full Gemini capability spectrum (`3.5-flash-lite` -> `3.6-flash` -> `3.8-flash` -> `4-hydrogen` -> `4-neon` -> `4-argon` -> `4-carbon`).
   - **Key Hypothesis:** Scaling base reasoning capability improves *initial rule acquisition (`L1`)*, **but does not eliminate—and can even sharpen—the RSI Ratchet (`L2/L3`)** because stronger instruction-following models adhere *more* faithfully to their own stale self-authored playbooks ("capability-amplified compliance").
2. **Act 2 — Surprise-Gated Epistemic Harness (`SGEH`):**
   - Replaces continuous per-episode LLM reflection (which wastes 75–94% of tokens per *The RSI Tax*, `2611.13524`) with a **scalar reward-residual Likelihood-Ratio / Page-Hinkley gate**:
     - **Calm regime:** `0` verification tokens; execute current structured policy.
     - **Triggered regime:** Shelve (do not delete) the falsified utility assumption, **reset post-changepoint evidence counters to zero** (preventing large-$n$ resurrection), run `1–3` discriminative probes (or check shelved regimes for recurrence), and commit the updated rule.
3. **Act 3 — Internalizing Epistemic Plasticity (`Qwen3-8B` & `Gemma-3-27B` LoRA):**
   - Train open-weight models via **SFT warmup + multi-turn GRPO** on balanced trajectories (**50% silent reward shifts, 50% noisy stationary controls**) so conservative doubt, targeted probing, and committed retraction happen natively in-weights without multi-call harness overhead.

---

## 2. Model Capability Ranking & Pricing Intelligence

All models below were verified live on the workspace `.env` `GEMINI_API_KEY` on Nov 19, 2026:

| Tier & Model ID | Capability Profile | Input / Output (per 1M tokens) | Role in Project |
| :--- | :--- | :---: | :--- |
| **`gemini-3.5-flash-lite`** | Ultra-lightweight baseline | `$0.30` / `$2.50` | Lower-bound capability anchor & unit-smoke testing |
| **`gemini-3.5-flash` (-> `3.6-flash`)** | Standard legacy Flash *(3.5 auto-routes to 3.6)* | `$0.75` / `$3.75` | Continuity anchor with prior repo experiments |
| **`gemini-3.8-flash`** | Strongest Gemini 3 everyday Flash | `$0.75` / `$3.75` *(intro thru Dec 31)* | **Primary fast dev & rapid ablation workhorse** |
| **`gemini-4-hydrogen`** | Fast Gemini 4 coding & execution worker | `$1.50` / `$7.50` | Mid-tier Gemini 4 scaling point & high-throughput worker |
| **`gemini-4-neon`** | High-speed frontier agentic judgment | `$3.00` / `$15.00` | Upper-mid Gemini 4 scaling point |
| **`gemini-4-argon`** | Sustained long-horizon agentic flagship (#1 on `ReScopeBench`) | **`$2.00` / `$10.00` until Nov 30** (`$4/$20` Dec 1) | **Primary frontier agentic benchmark & rollout teacher** |
| **`gemini-4-carbon`** | Peak deep-thinking & research flagship | **`$2.00` / `$10.00` until Nov 30** (`$4/$20` Dec 1) | **Top capability ceiling & hard counterfactual teacher** |
| **Open-Weight (`Qwen3-8B`, `Gemma-3-27B`)** | Trainable open backbones (vLLM / LoRA) | `$0` API (Local / CSD3 GPU) | **Act 3 before/after LoRA internalization targets** |
| **External Tier-3 (`DeepSeek`, Collaborator APIs)** | Cross-family external flagships | Reserved | **Strictly one-shot frozen final paper evaluation** |

**Strategic Pricing Window (Before November 30, 2026):** Both `gemini-4-argon` and `gemini-4-carbon` double in price from `$2/$10` to `$4/$20` per 1M tokens on **December 1** (in 11 days). Completing our Act 1 & Act 2 flagship `Argon`/`Carbon` sweeps before Nov 30—and using Gemini Context Caching + Batch API (50% discount) for bulk Act 3 trajectory generation—effectively doubles to quadruples our flagship compute per dollar.

---

## 3. Budget & Compute Allocation Plan

### 3.1 API Financial Budget (`$70,000+` Gemini Credit Pool Expiring Mid-Jan)

| Phase | Workload Description | Primary Models | Estimated Spend | Hard Gate / Cap |
| :--- | :--- | :--- | :---: | :--- |
| **Phase 0: Infra & Offline Verification** | Verifiers, leak-checks, `Oracle`/`Stale` scripted tests | None (Local CPU) | **`$0`** | 100% unit tests pass |
| **Phase 1: Rapid Dev & Harness Iteration** | `SLB-Mini` (3-window slices) + `ParadigmBench` 50-step dev loops (~40–60 runs) | `3.5-flash`, `3.8-flash`, `4-hydrogen` | **`$60 – $200`** | `--max-usd 5.00` per dev sweep |
| **Phase 2: Act 1 & Act 2 Full Scaling Matrix** | Full 9-scenario `ServeLearnBench` + full `ParadigmBench` x 6 Gemini models x 5 methods x 3 seeds | `3.5-lite` -> `3.8-flash` -> `4-hydrogen/neon/argon/carbon` | **`$2,500 – $5,500`** | Launch after Phase 1 shows >= +15 pt `SGEH` gain |
| **Phase 3: Act 3 Synthetic Rollout Factory** | Generate, verify & filter `50k–100k` multi-turn drift + stationary-control trajectories saved permanently to disk | `4-argon`, `4-carbon`, `3.8-flash` (with Batch/Cache) | **`$6,000 – $14,000`** | Banked to disk well before Jan 15 credit expiry |
| **Phase 4: Paper Stress-Tests & Ablations** | Sensitivity sweeps (drift speed, noise level, horizon length `L in [20, 500]`) | `3.8-flash`, `4-argon` | **`$1,500 – $3,000`** | Reviewer-proofing ablations |
| **Gemini Credit Reserve (Unallocated Buffer)** | Available if we scale teacher data to `250k+` traces or add extra benchmarks | Any Gemini 3/4 | **`~$47,000+` buffer** | Zero risk of credit exhaustion |
| **External Final Eval (Non-Gemini)** | One-shot frozen table (`Blind`, `SRN`, `SkillBook`, `SGEH`) on `DeepSeek` + friend endpoints | External APIs | **`~$150 – $400`** | **Executed ONCE in Week 4** |

### 3.2 GPU & Cluster Compute Allocation

| Compute Node | Hardware Specs | Assigned Workload | Est. Compute Budget |
| :--- | :--- | :--- | :--- |
| **Cloud Dev Server (Current Host)** | **Quadro RTX 8000 (48 GB VRAM)** *(Turing, FP16/FP32—no native BF16)* | Async API orchestrator, CPU simulation rollouts, local `vLLM` evals, `Qwen3-8B` SFT/GRPO smoke tests | Continuous host (already provisioned) |
| **Cambridge CSD3 Cluster** | Multi-GPU `A100-80GB` / `H100` nodes (native BF16) | Full `Qwen3-8B` & `Gemma-3-27B` SFT warmup + multi-turn GRPO LoRA training (3 seeds + ablations) | `~60 – 150` A100 GPU-hours |
| **Collaborator GPUs (GitHub Handoff)** | External A100/H100/RTX nodes | Parallel seed replications or `27B` ablation sweeps via single-command `scripts/run_train.sh` | Opportunistic parallelization |

---

## 4. 4-Week Schedule & Milestones

| Milestone | Target Date | Deliverable & Verification Metric |
| :--- | :---: | :--- |
| **M1: Engine + Dev Proof** | **Nov 22 (Day 3)** | `ServeLearnBench` + `ParadigmBench` (`AlphaDesk`, `LifeCompanion`, `CoalitionOps`) running with live cost ledger; first `$5–$15` Flash (`3.5`/`3.8`) pilot table |
| **M2: Act 1 & Act 2 Flagship Sweep** | **Nov 29 (Day 10)** | Full Gemini capability ranking (`3.5-lite` -> `4-carbon`) completed **before Nov 30 price doubling**; Figures 1–2 locked |
| **M3: Trajectory Bank & 8B Smoke** | **Dec 5 (Day 16)** | `50k+` verified drift + stationary-control trajectories banked to disk; `Qwen3-8B` LoRA verified on Quadro 8000 |
| **M4: CSD3 8B/27B Training & OOD Eval** | **Dec 13 (Day 24)** | Trained `Qwen3-8B` & `Gemma-3-27B` adapters evaluated on held-out `ParadigmBench` laws + `ServeLearnBench` transfer |
| **M5: Final Frozen Eval & Paper Draft** | **Dec 19 (Day 30)** | One-shot external flagship table, ablations, formal inertia proposition, and complete LaTeX submission package |

---

## 5. Risk Register & Mitigations

| Risk | Likelihood | Impact | Pre-Committed Mitigation |
| :--- | :---: | :---: | :--- |
| **`SGEH` triggers false alarms on noisy stationary tasks** | Medium | High | Always pair drift runs 1:1 with **Matched Stationary Controls**; tune Likelihood-Ratio threshold `tau` on stationary false-alarm rate `alpha <= 5%` *before* measuring drift recovery |
| **Reviewer overlap with `EvoBelief` (`2610.12789`) or `SRN` (`2611.13524`)** | High | High | Implement **both** `SRN` and an `EvoBelief`-style world-model baseline directly in our harness and beat them on (a) token cost and (b) silent *utility* reversal/decay + Act 3 weight internalization |
| **`ServeLearnBench` `Pitch` judge model dependency** | Certain | Low | `Pitch` uses `glm-5p3-flash` on Fireworks as judge in their paper; we prioritize deterministic rule-verified **`Retail`** and **`Banking`** first (zero judge variance!), and configure a fixed Gemini judge for internal `Pitch` sweeps |
| **Turing (`sm_75`) vs. Ampere/Hopper (`bf16`) training scripts** | Certain | Medium | Auto-detect GPU capability (`fp16`/`fp32` on Quadro smoke runs; `bf16` + FlashAttention-2 on CSD3) |
| **Act 3 GRPO reward hacking / twitchy exploration** | Medium | Medium | Mix **50% stationary noise-trap trajectories** into SFT/GRPO training and penalize unnecessary diagnostic probes in the RL reward |
