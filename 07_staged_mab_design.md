# Design Doc: Staged Multi-Armed Bandit (Belief Falsification)

**Status:** Approved and implemented  
**Date:** 2026-07-03  
**Authors:** experiment design for LLM + time suite  
**Related:** `02_environment_specs.md`, `04_benchmark_protocol.md`, Reversal / Periodic / Drifting results in `reports/benchmark_50turns_gemini35_v3.md`  
**Code:** `llm_rl_engine/envs/staged_bandit.py`  
**Configs:** `experiment_staged_constant.yaml` (P0), `experiment_staged_mixed.yaml` (P1), `experiment_staged_sinusoidal.yaml` (P2)

---

## 1. Motivation

### 1.1 What we already know

From prior Gemini (`gemini-3.5-flash`) runs (horizon 50, 5 seeds):

| Capability | Env evidence | LLM result (summary) |
|---|---|---|
| Stationary explore/exploit | StationaryBandit | ≈ UCB (34.2 vs 34.8) |
| Soft continuous change | DriftingBandit | Ties UCB; best non-oracle regret |
| Fixed periodic structure | Periodic / PhaseLocked | Mid-pack / ties SW-UCB |
| **One** discrete flip (same family) | ReversalBandit | **Best agent** (32.4 reward, regret 5.0) |
| Full-state timing | Pendulum / Traffic | Ceiling (not diagnostic) |

### 1.2 Gap

We have **not** cleanly tested:

> After the agent has **successfully** learned and exploited a regime, can it **refute** that theory when the **fundamental reward mechanism** changes (possibly to a different family), re-explore, and recover?

Reversal is a special case (one flip, same Bernoulli family). Staged-MAB generalizes this to **multiple scheduled regime changes**, optionally **across mechanism families**, with a clear time-series plot narrative.

### 1.3 Research questions

1. Does the LLM show **explore → exploit → plateau** within a stage?
2. At a hard stage boundary, does reward **drop**, and does the agent **re-explore**?
3. How long is **recovery delay** to near-oracle performance in the new stage?
4. Does recovery depend on whether the new stage is **same family** (new params) vs **cross family** (new mechanism class)?
5. How does LLM compare to **stubborn** (full-history UCB/greedy) vs **forgetful** (sliding-window UCB) baselines?

---

## 2. Environment: `StagedBandit`

### 2.1 High-level definition

A \(K\)-armed Bernoulli bandit whose **reward probability law** is piecewise-constant across discrete **stages**. Within a stage, dynamics follow one **stage family** with fixed parameters. At stage boundaries, the family and/or parameters switch **hard** (no announcement).

```text
t = 0 .. H-1
stage s = s(t) ∈ {0, ..., S-1}
p(t) = law_s(t; θ_s)     # vector of length K
r_t ~ Bernoulli(p_{a_t}(t))
```

Agent observation each step (public only):

```json
{
  "step": 12,
  "chosen_arm": 2,
  "reward": 1.0
}
```

**Never** expose: stage id, family name, change times, \(p(t)\), \(\theta_s\).

### 2.2 Action space

Unchanged from other bandits:

```json
{"arm": <integer in [0, K-1]>}
```

Recommended default: **`K = 4`** (consistent with entire suite).

### 2.3 Stage schedule

Two schedule modes (both supported; pilot uses **fixed equal length**):

| Mode | Definition | Use |
|---|---|---|
| **Equal fixed** | Stage \(s\) occupies \([s \cdot L, (s+1)\cdot L)\) | Primary pilot — clean plots |
| **Jittered fixed** | Stage lengths \(L_s \sim \mathrm{Uniform}\{L_{\min}, L_{\max}\}\) summing ≈ \(H\) | Robustness check later |

Hard cut at boundaries: \(p(t)\) switches immediately when \(t\) enters a new stage. No observation noise on the change (outcomes remain Bernoulli).

### 2.4 Stage families (concrete formulas)

All families output a probability vector \(p \in [0.05, 0.95]^K\) (clip).

#### Family A — `constant` (stationary within stage)

```text
p_i(t) = c_i    for all t in stage
c ~ i.i.d. Beta(α, β) or fixed contrast template
```

**Contrast template (recommended for readability):** one clear best arm:

```text
c_{best} = 0.85
c_{other} = 0.20
best ~ Uniform{0,...,K-1} independent per stage (may collide)
```

#### Family B — `sinusoidal` (periodic within stage)

Same structure as existing `PeriodicBandit`, but parameters **resampled per stage**:

```text
p_i(t) = clip( base_i + amp_i · sin(2π · t_local / P + φ_i), 0.05, 0.95 )
t_local = t - stage_start     # phase relative to stage start (recommended)
# optional: t_local = t       # global time (harder; not for pilot)
```

Defaults aligned with existing PeriodicBandit:

| Param | Range |
|---|---|
| Shared period \(P\) | Uniform integer in `[8, 16]` **per stage** |
| `base_i` | Uniform `[0.25, 0.65]` |
| `amp_i` | Uniform `[0.15, 0.35]` |
| `φ_i` | Uniform `[0, 2π)` |

**Pilot simplification:** shared \(P\) across arms within a stage (as in current PeriodicBandit benchmark).

#### Family C — `linear` (monotonic drift within stage)

One arm’s mean ramps across the stage; others fixed weak:

```text
best = Uniform{0,...,K-1}
p_other = 0.20
p_best(t) = clip( p0 + slope · u, 0.05, 0.95 )
u = (t - stage_start) / max(L-1, 1) ∈ [0, 1]    # fraction through stage
```

**Recommended templates:**

| Variant | \(p_0\) | slope | Story |
|---|---|---|---|
| **ramp_up** | 0.25 | +0.60 | Weak → strong (agent must notice improvement) |
| **ramp_down** | 0.90 | −0.65 | Strong → weak (forces abandon mid-stage) |

Pilot uses **ramp_up** only (simpler narrative). Include **ramp_down** in a secondary condition if needed.

### 2.5 Oracle

Each step (refresh before act — already engine standard):

```python
{
  "stage_index": s,
  "stage_family": "constant" | "sinusoidal" | "linear",
  "stage_start": t0,
  "stage_end": t1,
  "current_probs": p(t),
  "best_arm": argmax p(t),
  "stage_params": {...},   # bases, period, best, etc.
}
```

Never shown to agents; used for regret, recovery metrics, and plots.

---

## 3. Horizon and stage-length recommendations

### 3.1 What we actually ran before (ground truth)

| Setting | Value in this repo |
|---|---|
| Default horizon | **50** turns (all published Gemini runs) |
| ReversalBandit | horizon 50; flip \(\tau \in [15, 35]\) (not a fixed 60) |
| Time-varied / hard | horizon 50 |
| LLM cost (order of magnitude) | ~1 call/step; PhaseLocked 250 steps ≈ $0.06; time-varied ~$0.23 |

User intuition (“60 turns, double for 2 rules…”) is directionally right: **budget per regime ≈ time needed to explore + exploit**. Empirically:

- Stationary / Reversal: meaningful behavior in **~20–50** steps  
- Periodic structure: period up to 16 → need **≥1–2 periods** (~16–32 steps) to even *see* a cycle  
- LLM recovery after Reversal was strong within the remaining post-\(\tau\) horizon (often 15–35 steps left)

### 3.2 Budget model

```text
H = S × L
L = steps per stage (equal schedule)
S = number of stages
```

| Goal per stage | Suggested \(L\) |
|---|---|
| Bare minimum (detect change + crude re-explore) | 15 |
| Explore + short exploit plateau | **20–25** |
| Exploit plateau + 1–2 sinusoid periods | **30–40** |
| Publication-looking plateaus (user’s 100) | 80–100 |

### 3.3 Recommended horizons (decision table)

| Profile | \(S\) | \(L\) | \(H\) | LLM calls (5 seeds, 1 LLM agent) | When to use |
|---|---:|---:|---:|---:|---|
| **Pilot A** | 2 | 30 | **60** | ~300 | Validate drop/recovery metrics + plots |
| **Pilot B (default)** | 3 | 30 | **90** | ~450 | Main design review target |
| **Scale-up** | 3 | 50 | **150** | ~750 | If pilot plateaus too short |
| **Long (user 100×3)** | 3 | 100 | **300** | ~1500 | Final figure only after pilot works |
| Stretch 2-stage long | 2 | 60 | **120** | ~600 | Same-family deep exploit then flip |

**Recommendation for first implementation + first LLM run:**

> **Pilot B: \(H = 90\), \(S = 3\), \(L = 30\)**  
> Rationale: 30 ≈ 2× max period (16) for sinusoid stages; enough room post-change to measure recovery; ~1.8× prior 50-step cost, not 6×.

**Scaling rule of thumb (aligned with your “double” idea):**

```text
L_stage ≈ 25–30   # not 50–60 unless sinusoid needs more
H ≈ S × L_stage
# 2 stages → H≈60
# 3 stages → H≈90
# optional long → L=50 or 100 after pilot
```

Do **not** blindly set \(L = 50 \times S\) unless cost is acceptable; Reversal already showed adaptation can appear in fewer than 50 post-change steps.

### 3.4 Seeds and episodes

Match suite defaults unless pilot is too noisy:

| Param | Pilot | Confirm |
|---|---|---|
| `n_episodes` / seeds | 5 | 10 if recovery delay CI is huge |
| Seeds | `[0,1,2,3,4]` | extend to 0–9 |

---

## 4. Experimental conditions (setup matrix)

Run as **separate named configs** (or one config with multiple env variants). Order: easier scientific isolation → harder narrative.

### 4.1 Condition ladder

| ID | Name | Stages | Family sequence | What it tests |
|---|---|---|---|---|
| **C1** | Two-stage constant flip | 2 | `constant → constant` | Reversal-like but **scheduled** mid-episode (e.g. flip at \(t=L\)); best arm forced different | Controlled bridge from Reversal |
| **C2** | Three-stage constant | 3 | `constant × 3` | Repeated falsification of “arm \(a^\star\) is best forever” | Multi-alpha death, same family |
| **C3** | Three-stage sinusoidal | 3 | `sinusoidal × 3` | Same family; **period/phase/best arm** resampled each stage | Must drop old cycle model |
| **C4** | Three-stage linear | 3 | `linear × 3` (ramp_up) | Same family; new ramp arm each stage | Track changing means without period |
| **C5** | Mixed A (recommended headline) | 3 | `sinusoidal → constant → linear` | **Cross-family** mechanism change | Core research question |
| **C6** | Mixed B (mirror) | 3 | `constant → sinusoidal → constant` | Cross-family with stationary bookends | Robustness of mixed order |
| **C7** (optional) | Mixed full tour | 3 | `sinusoidal → linear → constant` | Alternate order | Order effects |

### 4.2 Pilot priority (what to implement and run first)

| Priority | Condition | \(H,S,L\) | Why |
|---|---|---|---|
| **P0** | **C2** three-stage constant | 90 / 3 / 30 | Simplest multi-falsification; easy oracle; greedy fails clearly |
| **P1** | **C5** mixed sin→const→linear | 90 / 3 / 30 | Headline cross-family story + plot |
| **P2** | **C3** three-stage sinusoidal | 90 / 3 / 30 | Harder within-family; needs \(L \ge 2P\) |
| **P3** | Scale P0+P1 to \(H=150\) (\(L=50\)) | if recovery incomplete at \(L=30\) | |
| **P4** | User-scale \(H=300\) (\(L=100\)) | final figures only | |

**Not first:** C4 alone (linear is less “patterny”); C1 alone (too close to existing Reversal).

### 4.3 Parameter constraints across stages (important)

To force **falsification** rather than lucky same-arm continuity:

| Constraint | Rule |
|---|---|
| **Best-arm diversity (constant stages)** | Sample best arm **without replacement** across stages when \(S \le K\); if \(S > K\), maximize changes |
| **Sinusoid stages** | Resample \(P, \phi, bases\) independently; optionally force \(\arg\max \mathbb{E}[p]\) to change |
| **Linear stages** | Independent `best` arm each stage; prefer ≠ previous stage’s best |
| **Seeded reproducibility** | All \(\theta_s\) drawn from `seed` via env RNG |

### 4.4 Anti-patterns to avoid

| Avoid | Why |
|---|---|
| Soft blended transitions | Blurs recovery delay definition |
| Exposing stage boundaries in obs | Trivializes change detection |
| \(L < 15\) | No time to exploit before next change |
| Sinusoid with \(L < P\) | Incomplete period; unfair vs constant stages |
| All stages same best arm | Agent can “succeed” without falsifying |

---

## 5. Agents / baselines

### 5.1 Required (plot + tables)

| Agent | Role on this task |
|---|---|
| **oracle** | Upper envelope; knows \(p(t)\) each step |
| **llm_raw_history** | Primary subject (`gemini-3.5-flash`, temp 0, raw history) |
| **greedy** | Cumulative means — **expected to overcommit** after change |
| **ucb** | Full-history UCB — stubborn after long first stage |
| **sliding_window_ucb** (window=10) | Forgetful classical — **expected best non-LLM adapter** |
| **random** | Floor |

### 5.2 Optional

| Agent | When |
|---|---|
| **thompson_sampling** | Completeness with prior suite |
| **sliding_window_ucb** window ∈ {5, 15, 20} | Sensitivity if window=10 is poorly matched to \(L\) |
| **epsilon_greedy** | Only if needed for paper narrative |

### 5.3 LLM protocol (unchanged)

- Model: `gemini-3.5-flash`, temperature 0  
- Prompt: raw history (same template as existing runs)  
- Horizon: full \(H\) (no early stop except env done — bandits usually run full horizon)  
- Parallelism: `llm_parallel_workers: 5` across seeds  

### 5.4 Expected qualitative ordering (hypothesis)

```text
oracle  ≥  LLM ≈ sliding_window_ucb  >  thompson ≈ ucb  >  greedy  ≥  random
```

On **cross-family (C5)**, expect larger **post-change drop** and longer recovery for greedy/UCB; LLM interest is whether it **matches or beats SW-UCB** on recovery delay (as on PhaseLocked / Reversal).

---

## 6. Metrics

### 6.1 Episode-level (existing style)

| Metric | Definition |
|---|---|
| `cumulative_reward` | \(\sum_t r_t\) |
| `regret` | \(\sum_t (p^\star_t - p_{a_t,t})\) using step-wise true probs |
| `invalid_action_rate` / parse errors | LLM reliability |

### 6.2 Stage-level (new — core of this design)

For each stage \(s\) with interval \([t_s, t_{s+1})\):

| Metric | Definition |
|---|---|
| `stage_reward[s]` | Sum of rewards in stage |
| `stage_regret[s]` | Regret restricted to stage |
| `mean_reward_post_change[s]` | Mean reward in first \(W\) steps of stage \(s\) (default \(W=5\)) for \(s\ge 1\) |
| `recovery_delay[s]` | Min \(d \ge 1\) s.t. for \(d' \in [d, d+K_{\mathrm{rec}}-1]\) agent arm equals oracle best arm **or** \(r\) rolling mean within \(\varepsilon\) of oracle expected reward; `null` if never |
| `overcommit_rate[s]` | For \(s\ge 1\): fraction of pulls of **previous stage’s modal best arm** (or previous oracle-best at end of stage \(s-1\)) during first \(W_{\mathrm{oc}}\) steps of stage \(s\) |

**Recommended constants:**

```text
W = 5              # immediate post-change window
W_oc = 10          # overcommit window
K_rec = 3          # consecutive “recovered” steps
ε = 0.15           # optional reward-based recovery tolerance
```

### 6.3 Change-point detection (optional diagnostic)

From trajectory only (no oracle): when rolling mean reward drops by \(>\delta\) vs previous stage plateau. Used for analysis of whether LLM’s **behavior** changes when rewards collapse — not for agent input.

---

## 7. Visualization plan

### 7.1 Primary figure: reward vs time (the story plot)

**Type:** 2D line plot, \(x = t\), \(y =\) reward-related series.

**Curves (per seed or mean ± band over seeds):**

1. **Oracle expected reward** \(p^\star_t = \max_i p_i(t)\) (smooth upper envelope)  
2. **Oracle realized** (optional, noisier)  
3. **LLM cumulative mean reward** in a rolling window (window=5) **or** per-step reward smoothed  
4. **Greedy** smoothed reward  
5. **Sliding-window UCB** smoothed reward  

**Annotations:**

- Vertical lines or shaded bands at stage boundaries  
- Stage labels: family name + stage index (e.g. `S1: sinusoidal`, `S2: constant`, `S3: linear`)  
- Optional markers: recovery time estimates for LLM  

**Panels:** one panel per condition (C2, C5, …) or multi-seed spaghetti for LLM only + mean.

### 7.2 Secondary figures

| Figure | Content |
|---|---|
| Per-stage bar chart | Mean stage reward by agent |
| Recovery delay bar | `recovery_delay[s]` by agent, stages 2..S |
| Arm choice heatmap | time × arm for one seed (LLM vs oracle best) |
| Cumulative regret vs time | classic RL curve with stage bands |

### 7.3 Dashboard integration

Extend `generate_run_plots` / dashboard for `StagedBandit`:

- Detect env name  
- Read stage boundaries from trajectory oracle fields (logged per episode end and/or per-step `private_info.stage_index`)  
- Emit `plots/staged_reward_vs_time_{agent}_{seed}.png` and aggregate mean plot  

**Logging requirement:** each step’s `private_info` must include `stage_index`, `step_probs`, `best_arm` for offline plotting without re-sim.

---

## 8. Implementation plan (after design approval)

### 8.1 Code touchpoints

| Component | Work |
|---|---|
| `llm_rl_engine/envs/bandits.py` or `staged_bandit.py` | New `StagedBandit` class |
| `BANDIT_ENV_NAMES` | Add name for regret pipeline |
| `registry.py` | Register + normalize stage config (list of dicts) |
| `classical.py` | No special case if standard arm actions |
| `metrics.py` | Stage metrics + recovery_delay |
| `plots.py` | Reward-vs-time with stage bands |
| `configs/experiment_staged_*.yaml` | P0/P1 configs |
| `tests/test_staged_bandit.py` | Schedule, family formulas, hard cut, oracle |

### 8.2 Config sketch (Pilot B / C5)

```yaml
experiment_name: benchmark_staged_mixed_p1
horizon: 90
n_episodes: 5
seeds: [0, 1, 2, 3, 4]
llm_parallel_workers: 5
generate_plots: true

environments:
  - name: StagedBandit
    n_arms: 4
    stage_length: 30
    stages:
      - family: sinusoidal
        # params sampled at reset unless fixed for debug
      - family: constant
      - family: linear
        linear_variant: ramp_up
    # optional:
    # force_distinct_best_arms: true

agents:
  - name: random
  - name: greedy
  - name: ucb
  - name: thompson_sampling
  - name: sliding_window_ucb
    window: 10
  - name: oracle
  - name: llm_raw_history
    model: gemini-3.5-flash
    temperature: 0
    max_retries: 1
```

### 8.3 Config sketch (Pilot B / C2 — simplest)

```yaml
experiment_name: benchmark_staged_constant_p0
horizon: 90
# ...
environments:
  - name: StagedBandit
    n_arms: 4
    stage_length: 30
    stages:
      - family: constant
      - family: constant
      - family: constant
    force_distinct_best_arms: true
```

### 8.4 Cost estimates (order of magnitude)

Assume ~$0.0002–0.0003 per LLM step (from prior ~$0.06 / 250 steps):

| Run | LLM steps (5 seeds × H) | Rough cost |
|---|---:|---:|
| P0 C2, H=90 | 450 | ~$0.10–0.15 |
| P0+P1, H=90 each | 900 | ~$0.20–0.30 |
| Scale H=150 both | 1500 | ~$0.35–0.50 |
| Long H=300 one condition | 1500 | ~$0.35–0.50 |

Classical agents free.

### 8.5 Implementation order

1. `StagedBandit` + unit tests (schedule, families, oracle)  
2. Metrics: stage regret, recovery_delay, overcommit  
3. Plots: reward vs time + stage bands  
4. Config P0 (C2) classical-only smoke  
5. Config P0 + LLM  
6. Config P1 (C5) classical + LLM  
7. Report section + figures  
8. Optional scale \(H=150\) or \(300\)

---

## 9. Success criteria (how we judge the experiment)

### 9.1 Design success (before caring about LLM wins)

- [ ] Oracle curve shows **3 clear plateaus / shapes** matching families  
- [ ] Greedy **drops and fails to recover** (or recovers slowly) after best-arm change  
- [ ] SW-UCB recovers faster than full UCB/greedy on average  
- [ ] Recovery delay metric is finite for oracle (≈1) and well-defined for others  

### 9.2 Scientific outcomes (any of these is a “result”)

| Outcome | Interpretation |
|---|---|
| LLM recovers like SW-UCB, beats greedy | Continues “test-time adaptation” story |
| LLM plateaus then **never** recovers | Fails at falsification / re-exploration |
| LLM recovers on C2 but fails on C5 | Same-family OK; cross-family hard |
| LLM recovers but with huge delay | Partial belief revision |

### 9.3 Failure modes of the *design* (watch for)

| Symptom | Likely cause | Fix |
|---|---|---|
| No one exploits stage 1 | \(L\) too short or SNR too low | Increase \(L\) or \(p\) contrast |
| No visible drop at boundary | Best arm same across stages | `force_distinct_best_arms` |
| Everyone “recovers” instantly | Change too weak | Increase contrast / family difference |
| Only noise, no plateaus | Bernoulli variance dominates | Report expected-reward curves; more seeds |

---

## 10. Relation to prior proposals (summary)

| Prior proposal | Role relative to StagedBandit |
|---|---|
| ReversalBandit | Special case ≈ C1 with random \(\tau\); staged uses **fixed calendar** for clean plots |
| FiniteLifeAlpha | Same spirit (alpha dies); staged makes **schedule + family** explicit |
| ExploitDecayBandit | Orthogonal (use-driven decay); can be a **stage family** later |
| PhaseLocked / Periodic | Fixed law; staged **breaks** that permanence on purpose |
| HypothesisCard | Cognitive framing; staged is the bandit operationalization |

---

## 11. Open decisions for reviewer

Please mark preferences before implementation:

| # | Decision | Options | Design default |
|---|---|---|---|
| D1 | Pilot horizon | 60 / **90** / 150 | **90 (3×30)** |
| D2 | First LLM condition | C2 only / **C2+C5** / C5 only | **C2+C5** |
| D3 | Linear definition | ramp_up only / include ramp_down | **ramp_up only** |
| D4 | Sinusoid time base | stage-local \(t\) / global \(t\) | **stage-local** |
| D5 | Window for SW-UCB | 5 / **10** / 15 | **10** (suite default) |
| D6 | Long run | skip / after pilot | **after pilot** |
| D7 | Env name | `StagedBandit` / `RegimeBandit` / `AlphaStageBandit` | **`StagedBandit`** |

---

## 12. Appendix: worked timeline examples

### 12.1 Pilot B / C5 (headline)

```text
H=90, L=30, K=4

t∈[0,30):  sinusoidal  (P~U{8..16}, stage-local phase)
t∈[30,60): constant    (one best arm 0.85, others 0.20; ≠ modal sin best if possible)
t∈[60,90): linear      (ramp_up on new best arm 0.25→0.85)
```

### 12.2 Pilot B / C2 (simplest multi-falsify)

```text
H=90, L=30, K=4

t∈[0,30):  constant, best=arm A
t∈[30,60): constant, best=arm B ≠ A
t∈[60,90): constant, best=arm C ≠ B
```

### 12.3 User-scale long figure (post-pilot only)

```text
H=300, L=100, S=3
Same families as C5 or C2
Cost ≈ 3× Pilot B per condition
```

### 12.4 Mapping user “60 / 120 / 180” idea

| User sketch | This design’s analogue |
|---|---|
| 60 turns, 1 rule | Single-stage baseline (skip; covered by Stationary) |
| 120 turns, 2 rules | \(H=60\)–\(120\), \(S=2\), \(L=H/2\) → **Pilot A** |
| 180 turns, 3 rules | \(H=90\)–\(180\), \(S=3\) → **Pilot B (90)** or mid (150) or long (180–300) |

We compress 180→**90** for the first LLM run to control cost while preserving three visible phases; expand if plateaus are visually insufficient.

---

## 13. Approval checklist

- [x] Horizon / stage length (D1) accepted — **H=90, L=30, S=3**  
- [x] Condition order (D2) accepted — **C2 + C5**  
- [x] Family formulas accepted  
- [x] Baseline set accepted  
- [x] Metrics + primary plot accepted  
- [x] Implemented `StagedBandit`  

### Run commands

```powershell
# P0: three-stage constant
uv run llm-rl-benchmark --config llm_rl_engine/configs/experiment_staged_constant.yaml

# P1: sin → constant → linear
uv run llm-rl-benchmark --config llm_rl_engine/configs/experiment_staged_mixed.yaml

# P2: 3x sinusoidal (same family, new params each stage)
uv run llm-rl-benchmark --config llm_rl_engine/configs/experiment_staged_sinusoidal.yaml
```

Primary plot after a run: `runs/<date>_benchmark_staged_*/plots/staged_reward_vs_time.png`

**End of design doc.**
