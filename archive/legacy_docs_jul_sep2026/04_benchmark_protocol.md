# Benchmark Protocol

## Goal

Evaluate whether frozen LLM agents can adapt online in static and dynamic RL environments, and compare them against classical baselines.

## Main Claims to Test

1. LLMs can or cannot explore effectively in long-horizon static environments.
2. LLMs can or cannot recover after reward structure changes.
3. LLMs can or cannot infer temporal patterns such as periodicity.
4. Explicit summaries or heuristics may improve performance.
5. Raw long histories may degrade adaptation relative to compact state representations.

## Episode Settings

Initial horizons:

| Phase | Horizon |
|---|---:|
| Smoke tests | 5-10 |
| Main static benchmark | 20-40 |
| Main dynamic benchmark | 40-100 |
| Stress tests | 100-500 |

Use at least 20 seeds per environment for smoke-level model testing, and 100+ seeds for final reported results if budget allows.

## Agents

### Classical Baselines

| Agent | Purpose |
|---|---|
| Random | Lower bound. |
| Greedy | Tests naive exploitation. |
| Epsilon-greedy | Simple exploration baseline. |
| UCB | Strong stationary-bandit baseline. |
| Thompson sampling | Strong stationary-bandit baseline. |
| Sliding-window UCB | Non-stationary baseline. |
| EXP3 | Adversarial/non-stationary baseline. |
| Oracle | Upper bound with hidden parameters. |

### LLM Agents

| Agent | Prompt Condition |
|---|---|
| LLM-RawHistory | Full history only. |
| LLM-Summary | External sufficient statistics. |
| LLM-SelfSummary | Agent maintains its own summary. |
| LLM-Heuristic | Agent maintains a compact heuristic. |
| LLM-Prediction | Agent forecasts candidate outcomes before acting. |
| LLM-Heuristic+Prediction | Combined condition. |

## Metrics

### General Metrics

| Metric | Definition |
|---|---|
| Cumulative reward | Sum of rewards over episode. |
| Regret | Oracle reward minus agent reward. |
| Normalized reward | Reward scaled by random and oracle policies. |
| Invalid action rate | Fraction of unparsable or invalid actions. |
| Token cost | Input/output tokens per episode. |
| Latency | Wall-clock time per action. |

### Bandit Metrics

| Metric | Definition |
|---|---|
| Exploration rate | Fraction of actions not currently believed best. |
| Best-arm identification | Whether final choice matches true best static arm. |
| Dynamic regret | Regret against best arm at each time step. |
| Switch delay | Steps needed to move away from old best arm after reversal. |
| Perseveration | Repeatedly choosing a now-bad arm after evidence changes. |

### NumberGuess Metrics

| Metric | Definition |
|---|---|
| Steps to target | First step where exact target is guessed. |
| Final error | Absolute distance from target at end. |
| Search efficiency | Compared against binary search. |
| Repeat rate | Fraction of repeated guesses. |

### Pendulum Metrics

| Metric | Definition |
|---|---|
| Success rate | Fraction of episodes with successful crossing. |
| Attempts to success | Number of crossing attempts before success. |
| Unsafe action rate | Fraction of crossings attempted in danger window. |
| Wait cost | Reward lost by excessive waiting. |
| Adaptation after phase shift | Recovery after hidden dynamics change. |

## Statistical Reporting

For each environment and agent:

- mean cumulative reward
- standard error or bootstrap confidence interval
- median and interquartile range
- regret distribution
- invalid action rate
- token cost

Recommended plots:

- cumulative reward over time
- action distribution over time
- arm choice by phase for periodic bandit
- recovery curve after reversal
- regret vs horizon
- performance vs context length

## Ablations

### History Ablation

| Condition | Purpose |
|---|---|
| Full ordered history | Main condition. |
| Shuffled history | Tests whether temporal order matters. |
| Last N steps only | Tests recency sensitivity. |
| Summary only | Tests compression benefit. |
| No history | Tests prior-only behavior. |

### Heuristic Ablation

| Condition | Purpose |
|---|---|
| No heuristic | Main baseline. |
| Self-generated heuristic | Tests heuristic learning. |
| Provided generic heuristic | Tests whether simple advice helps. |
| Provided environment-specific heuristic | Tests upper bound from correct rule. |
| Misleading heuristic | Tests susceptibility to stale/wrong rules. |

### Dynamic Difficulty Ablation

| Variable | Values |
|---|---|
| reversal frequency | rare, medium, frequent |
| period length | short, medium, long |
| reward noise | low, medium, high |
| number of arms | 2, 4, 8, 16 |
| horizon | 20, 40, 100, 500 |

## Core Tables for Paper 1

Table 1: Environment suite and hidden parameters.

Table 2: Agent performance across static environments.

Table 3: Agent performance across dynamic environments.

Table 4: Ablation of raw history vs summary vs self-summary.

Table 5: Classical baselines vs LLM agents by regret.

## Core Tables for Paper 2

Table 1: Heuristic conditions.

Table 2: Performance gains from heuristic memory.

Table 3: Heuristic quality metrics.

Table 4: Robustness to misleading or stale heuristics.

## Reproducibility Requirements

- Store exact environment config.
- Store exact prompt templates.
- Store exact model name and sampling parameters.
- Store all seeds.
- Store raw LLM responses.
- Store parsed actions and parse errors.
- Store hidden environment parameters for post-hoc analysis only.

