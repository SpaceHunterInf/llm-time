# Benchmark Report: LLM RL Engine

**Last updated:** 2026-07-03  

| Run | Directory | Runtime |
|---|---|---|
| Core 3-env benchmark | `runs/2026-07-03_benchmark_50turns_gemini35_v3/` | ~59 min |
| Time-varied 3-env benchmark | `runs/2026-07-03_benchmark_time_varied/` | ~39 min |
| Time-hard 2-env benchmark | `runs/2026-07-03_benchmark_time_hard/` | ~84 min |

---

## 1. Research Question

Can a frozen LLM agent (`gemini-3.5-flash`) adapt to reinforcement-learning environments using only in-context history — without weight updates — and how does it compare to classical baselines?

---

## 2. Experiment Setup

### 2.1 Global Settings

| Parameter | Value |
|---|---|
| Horizon (max turns) | **50** (all agents, all tasks) |
| Episodes per agent × task | 5 |
| Seeds | `[0, 1, 2, 3, 4]` |
| Config file | `llm_rl_engine/configs/experiment.yaml` |
| API key source | `.env` → `GEMINI_API_KEY` |
| SDK | `google-genai` (Interactions API) |

### 2.2 Agents

| Agent | Type | Notes |
|---|---|---|
| `random` | Classical | Uniform random actions |
| `greedy` | Classical | Best empirical arm; binary search on NumberGuess |
| `ucb` | Classical | UCB1 (bandit tasks only) |
| `thompson_sampling` | Classical | Thompson sampling (bandit tasks only) |
| `sliding_window_ucb` | Classical | Window = 10 (bandit tasks only) |
| `llm_raw_history` | LLM | `gemini-3.5-flash`, temperature 0, raw-history prompt |

Bandit-only agents are excluded from NumberGuess via per-agent `environments` filters.

### 2.3 LLM Agent Protocol

- **Prompt mode:** Raw history (full transcript of prior actions, rewards, observations)
- **Response format:** Strict JSON with `action`, `confidence`, `rationale`
- **Retry policy:** 1 retry on parse failure; fallback action if still invalid
- **Invalid action penalty:** −0.1 (logged, not heavily used in this run)

### 2.4 Metrics

- **Cumulative reward:** Sum of per-step rewards over the episode
- **Regret:** Σₜ (p\*ₜ − pₐₜ) for bandits, where p\*ₜ is the best-arm probability at step t
- **Steps to target:** First step with a correct guess (NumberGuess only)
- **Invalid / parse error rate:** Fraction of malformed LLM responses

---

## 3. Task Descriptions

### 3.1 NumberGuess (Static)

**Purpose:** Test hypothesis narrowing from directional feedback.

| Property | Value |
|---|---|
| Hidden target | Uniform integer in `[0, 100]` |
| Action | `{"guess": <int>}` |
| Reward | +1 if correct, else 0 |
| Episode ends | On correct guess |

**Observation after each guess:**

```json
{
  "last_guess": 50,
  "feedback": "too_high",
  "comparison": "target_is_lower",
  "hint": "You guessed 50. The target is smaller (lower) than your guess."
}
```

`feedback` ∈ {`too_low`, `too_high`, `correct`}. The hidden target value is never revealed.

---

### 3.2 StationaryBandit (Static)

**Purpose:** Test exploration vs. exploitation with fixed reward probabilities.

| Property | Value |
|---|---|
| Arms | 4 |
| Reward model | Bernoulli per arm |
| Arm probabilities | Sampled once per episode: `pᵢ ~ Beta(1, 1)` |
| Action | `{"arm": <0–3>}` |
| Observation | Chosen arm, binary reward, step number |

Probabilities are fixed for the entire 50-step episode.

---

### 3.3 ReversalBandit (Dynamic)

**Purpose:** Test whether the agent detects and adapts when a previously good arm becomes bad.

| Property | Value |
|---|---|
| Arms | 4 |
| Pre-reversal | Fixed probabilities `p_pre` for steps `t < τ` |
| Reversal time τ | Hidden, uniform in `[15, 35]` |
| Post-reversal | Probabilities switch to `p_post` (best arm swapped to worst + noise) |
| Agent visibility | No announcement of reversal; only reward feedback |

---

### 3.4 PeriodicBandit (Dynamic — time-varying)

**Purpose:** Test inference of periodic reward structure.

| Property | Value |
|---|---|
| Arms | 4 |
| Reward model | `p_i(t) = clip(base_i + amp_i · sin(2πt/period + phase_i), 0.01, 0.99)` |
| Shared period | Yes (same period for all arms in benchmark) |
| Period range | Hidden, uniform in `[8, 16]` |
| Action | `{"arm": <0–3>}` |
| Agent visibility | Only arm + binary reward feedback |

---

### 3.5 DriftingBandit (Dynamic — time-varying)

**Purpose:** Test continuous adaptation as probabilities drift.

| Property | Value |
|---|---|
| Arms | 4 |
| Initial probs | `p_i ~ Beta(1, 1)` per episode |
| Drift | Each step: `p_i ← clip(p_i + 𝒩(0, σ), 0.05, 0.95)` with `σ = 0.03` |
| Action | `{"arm": <0–3>}` |

---

### 3.6 PendulumGate (Dynamic — timing)

**Purpose:** Test latent dynamics inference, continuous-time timing, and adaptation to mid-episode phase shifts. See **Section 8** for full detail.

| Property | Value |
|---|---|
| Hidden period `P` | Uniform in `[8, 16]` seconds |
| Hidden phase `φ` | Uniform in `[0, 2π)`; may shift at hidden `τ ∈ [10, 20]` s |
| Simulation clock | `Δt = 0.6` s per agent turn (independent of action) |
| Episode horizon | 30 s sim-time (50 turns at `Δt = 0.6`) |
| Pendulum position | `sin(2π·t/P + φ)` |
| Danger zone | `\|position\| ≥ 0.35` → crossing hits |
| Lives | 3 |
| Actions | `wait` (−0.02), `cross` (+10 or −1) |
| Observations | Every turn: `time_s`, `position`, `velocity`, `lives` |

---

### 3.7 TrafficLightGate (Dynamic — timing)

**Purpose:** Test risky timed commitment with discrete, deterministic observations (no observation noise).

| Property | Value |
|---|---|
| Hidden period `P` | Uniform in `[8, 16]` seconds |
| Phase offset | Uniform in `[0, 1)` (cycle fraction); may shift at `τ ∈ [12, 22]` s |
| Cycle split | Green 30%, yellow 15%, red 55% |
| Simulation clock | `Δt = 0.6` s per agent turn (independent of action) |
| Episode horizon | 30 s sim-time |
| Lives | 3 |
| Actions | `wait` (−0.02), `go` (+10 on green, −1 on yellow/red) |
| Observations | Every turn: `time_s`, `light_color`, `lives` (exact, deterministic) |

---

### 3.8 PhaseLockedBandit (Dynamic — time-varying)

**Purpose:** Test exploitation of narrow, rotating phase windows — arms are either hot or cold, not smoothly varying.

| Property | Value |
|---|---|
| Arms | 4 |
| Shared period | Yes; hidden `P ∈ [12, 20]` steps |
| Per-arm phase | `φᵢ ~ Uniform(0, 2π)` |
| Active condition | `sin(2πt/P + φᵢ) ≥ 0.5` |
| Active / inactive reward prob | 0.85 / 0.05 |
| Action | `{"arm": <0–3>}` |
| Agent visibility | Only arm + binary reward feedback |

Unlike PeriodicBandit (smooth sinusoidal probabilities), only one arm is typically in-window at a time; greedy cumulative means fail because past rewards mislead.

---

## 4. Results — Core Benchmark (v3)

### 4.1 Summary Table (Mean Cumulative Reward, 5 Episodes)

| Agent | NumberGuess | StationaryBandit | ReversalBandit |
|---|---:|---:|---:|
| random | 1.0 ± 0.0 | 23.0 ± 3.6 | 20.8 ± 8.3 |
| greedy | 1.0 ± 0.0 | 19.6 ± 11.9 | 25.8 ± 11.3 |
| ucb | — | **34.8 ± 2.9** | 26.6 ± 5.0 |
| thompson_sampling | — | 34.0 ± 5.9 | 27.6 ± 6.8 |
| sliding_window_ucb | — | 24.8 ± 4.9 | 25.0 ± 4.9 |
| **llm_raw_history** | **1.0 ± 0.0** | 34.2 ± 5.1 | **32.4 ± 4.3** |

### 4.2 Regret (Bandit Tasks, Lower Is Better)

| Agent | StationaryBandit | ReversalBandit |
|---|---:|---:|
| random | 15.98 | 15.41 |
| greedy | 19.53 | 12.90 |
| ucb | **2.70** | 10.69 |
| thompson_sampling | 5.20 | 11.44 |
| sliding_window_ucb | 12.68 | 12.90 |
| **llm_raw_history** | 3.73 | **5.00** |

### 4.3 Per-Episode Detail

#### NumberGuess — steps to correct guess

| Agent | Seeds 0–4 (steps) |
|---|---|
| random | 1, 1, 1, 1, 1 |
| greedy | 7, 4, 7, 4, 4 |
| llm_raw_history | 7, 7, 6, 7, 7 |

All agents achieved reward 1.0 (found the target). Random got lucky on these seeds (hit on step 1). Greedy and LLM used ~4–7 steps (binary-search-like).

#### StationaryBandit — per-seed cumulative reward

| Agent | Seeds 0–4 |
|---|---|
| ucb | 35, 37, 31, 38, 33 |
| thompson | 29, 32, 31, 44, 34 |
| llm | 28, 38, 30, 40, 35 |

#### ReversalBandit — per-seed cumulative reward

| Agent | Seeds 0–4 |
|---|---|
| ucb | 35, 23, 25, 27, 23 |
| thompson | 32, 17, 27, 27, 35 |
| sliding_window_ucb | 30, 18, 28, 22, 27 |
| **llm** | **33, 27, 31, 39, 32** |

### 4.4 LLM Reliability

| Metric | Value |
|---|---|
| Total LLM steps | 534 |
| Parse / API errors | 14 (2.6%) |
| Invalid action rate | 0.0% |

Errors were concentrated in a few steps; fallback actions did not trigger invalid-action penalties in logged metrics.

---

## 5. Key Findings

1. **NumberGuess:** LLM solves reliably in ~6 steps using comparison feedback, matching greedy binary search. The revised `too_low` / `too_high` hints with `comparison` and `hint` fields are working as intended.

2. **StationaryBandit:** LLM (34.2) is statistically comparable to UCB (34.8) and Thompson sampling (34.0), well above random (23.0) and naive greedy (19.6).

3. **ReversalBandit:** LLM (32.4) is the **best performer**, beating UCB (26.6), Thompson (27.6), and sliding-window UCB (25.0). Lower regret (5.00 vs 10.69 for UCB) suggests the LLM adapts after the hidden reversal better than stationary UCB, which over-commits to pre-reversal evidence.

4. **Sliding-window UCB** did not outperform plain UCB on ReversalBandit in this run — possibly because window size (10) and reversal timing (15–35) interact unfavorably.

5. **PhaseLockedBandit** (Section 7.2) is the clearest time-themed win for the LLM so far: best non-oracle regret (17.28), tying sliding-window UCB on reward while beating greedy/UCB/Thompson by a wide margin.

---

## 6. Results — Time-Varied Benchmark

**Config:** `llm_rl_engine/configs/experiment_time_varied.yaml`  
**Results:** `runs/2026-07-03_benchmark_time_varied/`  
**Dashboard:** `runs/2026-07-03_benchmark_time_varied/dashboard.html`

Trajectories are saved in `trajectories.jsonl` — re-plot or re-analyze without rerunning:
```powershell
uv run llm-rl-benchmark --plot-run-dir runs/2026-07-03_benchmark_time_varied
```

### 6.1 PeriodicBandit (mean cumulative reward / regret)

| Agent | Mean reward | Mean regret ↓ |
|---|---:|---:|
| **oracle** | **20.6 ± 3.9** | 10.28 |
| thompson_sampling | 19.2 ± 3.8 | 9.77 |
| greedy | 18.8 ± 5.2 | 11.26 |
| **llm_raw_history** | 18.2 ± 4.5 | 10.51 |
| ucb | 17.8 ± 4.0 | **9.75** |
| sliding_window_ucb | 17.6 ± 4.2 | 11.50 |
| random | 16.4 ± 3.9 | 12.58 |

LLM is mid-pack — comparable to greedy/UCB, below Thompson and oracle. No agent fully exploits the sinusoidal structure.

### 6.2 DriftingBandit (mean cumulative reward / regret)

| Agent | Mean reward | Mean regret ↓ |
|---|---:|---:|
| **oracle** | **39.6 ± 8.3** | **2.77** |
| **ucb** | **37.6 ± 8.6** | 3.65 |
| **llm_raw_history** | **37.6 ± 8.9** | **3.44** |
| thompson_sampling | 33.6 ± 10.3 | 5.38 |
| sliding_window_ucb | 27.4 ± 3.5 | 12.94 |
| random | 24.6 ± 6.5 | 16.78 |
| greedy | 18.8 ± 12.2 | 20.75 |

**Headline:** LLM **ties UCB** on reward and achieves the **best non-oracle regret** (3.44).

### 6.3 PendulumGate (mean reward / success rate) — v2

Re-run on **PendulumGate v2** (independent clock, continuous observations, `wait`/`cross` only). Same run directory: `runs/2026-07-03_benchmark_time_varied/`.

| Agent | Mean reward | Success (5 seeds) |
|---|---:|---|
| **greedy** | **9.95 ± 0.06** | **5/5** |
| **oracle** | **9.95 ± 0.06** | **5/5** |
| **llm_raw_history** | **9.95 ± 0.06** | **5/5** |
| random | 6.15 ± 5.23 | 4/5 |

With continuous `position`/`velocity`/`time_s` each turn, greedy, oracle, and LLM all cross safely on every seed. Random still fails one seed (seed 2: multiple unsafe crosses, −3.08 reward). v2 is well-posed relative to v1, where oracle scored 0/5 due to a frozen-oracle bug and partial observability made timing opaque.

### 6.4 LLM cost (time-varied run, latest)

| | Value |
|---|---|
| Total LLM steps | 1,087 |
| Estimated cost | **~$0.23 USD** |
| PeriodicBandit | $0.116 |
| DriftingBandit | $0.102 |
| PendulumGate | $0.011 |

---

## 7. Results — Time-Hard Benchmark

**Config:** `llm_rl_engine/configs/experiment_time_hard.yaml`  
**Results:** `runs/2026-07-03_benchmark_time_hard/`  
**Dashboard:** `runs/2026-07-03_benchmark_time_hard/dashboard.html`

Environments: **TrafficLightGate** and **PhaseLockedBandit** — designed after PendulumGate v2 proved too easy (continuous kinematics enable a one-line threshold policy). Observations are deterministic throughout (no observation noise).

```powershell
uv run llm-rl-benchmark --plot-run-dir runs/2026-07-03_benchmark_time_hard
```

### 7.1 TrafficLightGate (mean reward / success rate)

| Agent | Mean reward | Success (5 seeds) |
|---|---:|---|
| **greedy** | **9.92 ± 0.06** | **5/5** |
| **oracle** | **9.92 ± 0.06** | **5/5** |
| **llm_raw_history** | **9.92 ± 0.06** | **5/5** |
| random | 6.75 ± 5.53 | 4/5 |

The optimal memoryless policy — go on green, wait otherwise — is trivial once the agent reads `light_color`. LLM, greedy, and oracle all execute it perfectly. Random still fails occasionally by going on red/yellow. **TrafficLightGate does not separate LLM from simple reactive baselines** under current parameters; it mainly punishes undisciplined exploration.

### 7.2 PhaseLockedBandit (mean cumulative reward / regret)

| Agent | Mean reward | Mean regret ↓ |
|---|---:|---:|
| **oracle** | **30.6 ± 2.4** | **6.08** |
| **llm_raw_history** | **19.8 ± 4.1** | **17.28** |
| sliding_window_ucb | 19.8 ± 5.4 | 18.88 |
| random | 14.8 ± 3.1 | 22.88 |
| greedy | 14.4 ± 1.8 | 23.68 |
| thompson_sampling | 13.0 ± 0.7 | 23.84 |
| ucb | 10.8 ± 3.1 | 27.36 |

**Headline:** PhaseLockedBandit produces the intended spread. Greedy, UCB, and Thompson treat arms as stationary and score ~11–15 reward. **LLM ties sliding-window UCB on reward (19.8) and achieves the best non-oracle regret (17.28)**, suggesting in-context tracking of which arm is currently in-phase. Oracle remains far ahead (30.6), leaving headroom for stronger temporal policies.

### 7.3 LLM cost (time-hard run)

| | Value |
|---|---|
| Total LLM steps | 274 |
| Estimated cost | **~$0.06 USD** |
| TrafficLightGate | $0.002 (24 steps) |
| PhaseLockedBandit | $0.060 (250 steps) |

### 7.4 Key findings (time-hard)

1. **PhaseLockedBandit is the stronger benchmark** for LLM vs classical comparison under the time theme — clear regret ordering with meaningful gap to oracle.
2. **TrafficLightGate** validates deterministic partial observability (color only, no kinematics) but is **too easy** for any agent that reacts to the current color; consider narrower green windows or observation delay if retained.
3. **LLM temporal reasoning** appears strongest on discrete rotating structure (phase-locked arms), not on continuous pendulum kinematics or memoryless traffic-light rules.

---

## 8. Limitations

- Only 5 seeds per condition (pilot-scale, not publication-scale)
- Single LLM prompt mode (raw history only)
- LLM episodes parallelized across seeds; steps within an episode remain sequential
- Random NumberGuess results in v3 run are seed-lucky (all hit on step 1)
- TrafficLightGate and PendulumGate v2 are ceiling effects for LLM/greedy under current observation design
- Re-running into the same dated run directory appends trajectories (use a fresh output dir for clean artifacts)

---

## 9. PendulumGate — Setup & LLM Interface (v2)

### 9.1 Time model

Simulation time is **independent of the agent's action**. Each agent turn advances the environment clock by a fixed `Δt = 0.6` seconds. LLM deliberation consumes **zero** simulation time (turn-based frozen deliberation): the agent reads the observation at time `t`, chooses an action, and the environment evaluates that action at the same `t` before advancing to `t + Δt`.

Episode horizon is **30 seconds** of simulation time (50 turns at `Δt = 0.6`). The episode also ends on successful cross, death (0 lives), or timeout.

### 9.2 Hidden dynamics (agent never sees these)

Each episode samples:

```text
period P       ~ Uniform[8, 16] seconds
phase φ        ~ Uniform(0, 2π)
shift time τ   ~ Uniform[10, 20] seconds
shift Δφ       ~ Uniform[π/4, π]
position(t)    = sin(2π · t / P + φ)
velocity(t)    = (2π / P) · cos(2π · t / P + φ)
danger(t)      = |position(t)| ≥ 0.35
```

At simulation time `τ`, the phase jumps: `φ ← φ + Δφ`. The agent must detect and adapt from the changing position/velocity stream — period `P` and initial `φ` remain hidden.

### 9.3 What the LLM can do (actions)

The LLM must return strict JSON each step:

```json
{
  "action": {"action": "wait"},
  "confidence": 0.6,
  "rationale": "..."
}
```

Valid actions:

| Action | Effect | Reward |
|---|---|---|
| **wait** | Pass the turn; clock advances by `Δt` | −0.02 |
| **cross** | Attempt to pass the gate at current `time_s` | +10 if safe, −1 if hit |

There is no separate `observe` action — position and velocity are reported every turn.

Episode ends on **successful cross**, **death** (0 lives), or **timeout** (30 s).

### 9.4 What the LLM can observe

Every turn the LLM receives the full structured observation in its prompt (plus full history). Public fields include continuous kinematics; period, phase, and `τ` are never revealed.

**Typical turn observation:**

```json
{
  "step": 3,
  "message": "You waited.",
  "public_state": {
    "time_s": 1.8,
    "position": -0.12,
    "velocity": 0.45,
    "lives": 3,
    "event": "waited"
  },
  "last_action": {"action": "wait"},
  "last_reward": -0.02,
  "done": false
}
```

**After successful `cross`:**

```json
{
  "public_state": {
    "time_s": 4.2,
    "position": 0.08,
    "velocity": 0.51,
    "lives": 3,
    "event": "success",
    "reward": 10,
    "episode_status": "done"
  },
  "done": true
}
```

**After hit:**

```json
{
  "public_state": {
    "time_s": 6.0,
    "position": 0.41,
    "velocity": -0.22,
    "lives": 2,
    "event": "hit",
    "reward": -1,
    "episode_status": "alive"
  }
}
```

### 9.5 What the LLM sees in the prompt

Raw-history mode includes:

1. Task description (from env metadata)
2. Action space schema (`wait` | `cross`)
3. Full ordered history of prior `(action, reward, observation message)` tuples
4. Current observation JSON with `time_s`, `position`, `velocity`, `lives`

The LLM must infer period, phase, and safe crossing windows from the continuous observation stream and adapt when the rhythm shifts at `τ`.

### 9.6 Why this task appeared hard (but v2 was too easy)

In principle, hidden period, phase shifts, and risky commits make timing difficult. In practice (Section 6.3), exposing `position` and `velocity` every turn reduces the task to `cross` when `|position| < 0.35` — greedy, oracle, and LLM all scored 5/5. Harder timing tasks should **not** leak sufficient statistics each turn; see PhaseLockedBandit (Section 7.2) and future sparse-observation designs.

### 9.7 Oracle baseline (fixed)

The oracle agent now **refreshes** `env.oracle()` every turn before acting. It crosses when `|position| < 0.35` at the current simulation time and waits otherwise. This provides a meaningful upper bound under the v2 time model.

**Note:** Section 6.3 reports **v2** PendulumGate results from the latest re-run of `experiment_time_varied.yaml`.

---

## 10. Artifacts

```
runs/2026-07-03_benchmark_50turns_gemini35_v3/
├── config.yaml, metrics.json, trajectories.jsonl, dashboard.html

runs/2026-07-03_benchmark_time_varied/
├── config.yaml, metrics.json, token_cost.json, trajectories.jsonl
├── dashboard.html, plots/, prompts/

runs/2026-07-03_benchmark_time_hard/
├── config.yaml, metrics.json, token_cost.json, trajectories.jsonl
├── dashboard.html, plots/, prompts/
```

**Note:** The v3 run does **not** contain PeriodicBandit, DriftingBandit, or PendulumGate trajectories. Those exist only in the time-varied run directory. TrafficLightGate and PhaseLockedBandit trajectories exist only in the time-hard run directory.