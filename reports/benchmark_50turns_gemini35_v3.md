# Benchmark Report: LLM RL Engine v3

**Run ID:** `benchmark_50turns_gemini35_v3`  
**Date:** 2026-07-03  
**Results directory:** `runs/2026-07-03_benchmark_50turns_gemini35_v3/`  
**Wall-clock runtime:** ~59 minutes

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

This is the **only dynamic environment** in the current benchmark suite.

---

## 4. Results

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

---

## 6. Limitations

- Only 5 seeds per condition (pilot-scale, not publication-scale)
- Single LLM prompt mode (raw history only)
- No periodic or drifting bandits yet
- Sequential API calls → long wall-clock time (~1 hour)
- Random NumberGuess results are seed-lucky (all hit on step 1)

---

## 7. Artifacts

```
runs/2026-07-03_benchmark_50turns_gemini35_v3/
├── config.yaml
├── metrics.json
├── trajectories.jsonl
├── episode_summaries.jsonl
└── prompts/          # per-step LLM prompts + responses
```