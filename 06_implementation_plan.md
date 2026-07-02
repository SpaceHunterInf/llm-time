# Implementation Plan

## Milestone 1: Minimal Engine

Build:

- `Env` base class
- `Agent` base class
- `run_episode`
- JSONL logger
- config loader
- metric calculator

Acceptance checks:

- a random agent can run 10 episodes on NumberGuess
- every step is logged
- seeds reproduce identical hidden targets and rewards

## Milestone 2: Static Environments

Build:

- NumberGuess
- StationaryBandit

Baselines:

- random
- greedy
- epsilon-greedy
- UCB
- Thompson sampling

Acceptance checks:

- UCB and Thompson outperform random on StationaryBandit
- binary-search-like behavior is possible on NumberGuess
- oracle metrics are computed but hidden from agents

## Milestone 3: Dynamic Environments

Build:

- ReversalBandit
- DriftingBandit
- PeriodicBandit

Baselines:

- sliding-window UCB
- discounted UCB or discounted Thompson sampling
- oracle dynamic policy

Acceptance checks:

- static UCB performs poorly after reversal compared with sliding-window UCB
- oracle achieves high dynamic reward
- periodic reward probabilities are reproducible by seed

## Milestone 4: LLM Agent Wrapper

Build:

- prompt renderer
- model client abstraction
- JSON action parser
- retry-on-invalid-response
- prompt/response logging

Prompt modes:

- raw history
- summary history
- self-summary
- heuristic before action
- prediction before action

Acceptance checks:

- LLM agent can run one full episode without manual intervention
- invalid JSON is caught and logged
- all prompts and responses are stored

## Milestone 5: Metrics and Plots

Build metric scripts for:

- cumulative reward
- regret
- dynamic regret
- invalid action rate
- exploration rate
- switch delay
- phase-dependent action distribution
- heuristic update frequency

Build plots:

- reward over time
- regret over time
- arm choice over time
- arm choice by phase
- post-reversal recovery curve

Acceptance checks:

- one command produces metrics from JSONL logs
- plots are saved with experiment config

## Milestone 6: Heuristic Experiments

Add:

- preset generic heuristic condition
- preset task-specific heuristic condition
- self-generated initial heuristic condition
- self-updated heuristic condition
- wrong heuristic condition

Acceptance checks:

- heuristic text is logged at every step
- heuristic can be replayed without raw history
- wrong heuristic condition is implemented without leaking hidden parameters

## Suggested Repository Structure

```text
llm_rl_engine/
  envs/
    base.py
    number_guess.py
    bandits.py
    pendulum_gate.py
  agents/
    base.py
    classical.py
    llm.py
  prompts/
    raw_history.txt
    summary_history.txt
    heuristic_before_action.txt
    prediction_before_action.txt
  runners/
    run_episode.py
    run_benchmark.py
  analysis/
    metrics.py
    plots.py
  configs/
    smoke.yaml
    static_bandit.yaml
    dynamic_bandit.yaml
    heuristic_ablation.yaml
  tests/
    test_envs.py
    test_baselines.py
    test_logging.py
```

## First Smoke Config

```yaml
experiment_name: smoke_static
horizon: 20
n_episodes: 5
seeds: [0, 1, 2, 3, 4]

environments:
  - name: NumberGuess
    target_min: 0
    target_max: 100
    feedback: exact_comparison
  - name: StationaryBandit
    n_arms: 4
    reward_distribution: bernoulli
    alpha: 1
    beta: 1

agents:
  - name: random
  - name: greedy
  - name: ucb
  - name: thompson_sampling
```

## First Dynamic Config

```yaml
experiment_name: smoke_dynamic
horizon: 40
n_episodes: 10
seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

environments:
  - name: ReversalBandit
    n_arms: 4
    reversal_schedule: fixed_hidden_time
    reversal_time_range: [15, 25]
  - name: PeriodicBandit
    n_arms: 4
    shared_period: true
    period_range: [8, 20]
    amplitude_range: [0.15, 0.35]

agents:
  - name: random
  - name: ucb
  - name: sliding_window_ucb
    window: 10
  - name: oracle
```

## First LLM Config

```yaml
experiment_name: smoke_llm_heuristics
horizon: 40
n_episodes: 5
seeds: [0, 1, 2, 3, 4]

environments:
  - name: PeriodicBandit
    n_arms: 4
    shared_period: true
    period_range: [8, 16]

agents:
  - name: llm_raw_history
    model: MODEL_NAME
    temperature: 0
  - name: llm_summary
    model: MODEL_NAME
    temperature: 0
  - name: llm_heuristic_before_action
    model: MODEL_NAME
    temperature: 0
```

## Build Notes for the Agent

- Keep the first version small and testable.
- Do not implement all environments before the runner is stable.
- Prefer structured configs and JSONL logs from the beginning.
- Implement classical baselines before LLM agents, so environment behavior can be validated cheaply.
- Keep prompt templates versioned, because prompt changes are experimental changes.
- Add unit tests for hidden-parameter leakage.

