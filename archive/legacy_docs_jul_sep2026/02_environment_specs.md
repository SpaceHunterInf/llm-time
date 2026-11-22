# Environment Specs

This file defines the initial task suite. The suite should move from simple static inference to dynamic hidden-time inference.

## Common Interface

Each environment should define:

- `name`
- `description`
- `action_space`
- `observation_format`
- `reward_range`
- `hidden_parameters`
- `difficulty_parameters`
- `oracle_policy`, if easy to define

Agents should only see the public task description and observations.

## Environment 0: NumberGuess

### Purpose

Test basic hypothesis narrowing from feedback.

### Setup

The environment samples a hidden integer target:

```text
target ~ Uniform({0, 1, ..., 100})
```

At each step the agent guesses an integer.

### Action

```json
{"guess": 42}
```

### Observation

After each guess:

```json
{
  "feedback": "too_low",
  "distance_bucket": "far"
}
```

Recommended feedback variants:

| Variant | Feedback |
|---|---|
| Exact comparison | `too_low`, `too_high`, `correct` |
| Distance bucket | `very_far`, `far`, `near`, `very_near`, `correct` |
| Noisy comparison | comparison is wrong with small probability |
| Delayed feedback | feedback from previous step only |

### Reward

Simple:

```text
reward = 1 if guess == target else 0
```

Dense:

```text
reward = -abs(guess - target) / 100
```

### Metrics

- steps to exact hit
- final absolute error
- binary-search efficiency
- repeated invalid guesses

## Environment 1: StationaryBandit

### Purpose

Test basic exploration and exploitation.

### Setup

There are `K` arms. Each arm has a fixed Bernoulli reward probability.

```text
p_i ~ Beta(alpha, beta)
r_t ~ Bernoulli(p_{a_t})
```

### Action

```json
{"arm": 2}
```

### Observation

```json
{
  "chosen_arm": 2,
  "reward": 1,
  "step": 13
}
```

### Difficulty Parameters

- `n_arms`: 2, 4, 8, 16
- gap between best and second-best arm
- reward noise
- horizon

### Metrics

- cumulative reward
- regret against oracle best arm
- exploration rate
- probability of identifying best arm by final step

## Environment 2: ReversalBandit

### Purpose

Test whether the agent detects that a previously good action has become bad.

### Setup

There are `K` arms. Reward probabilities are fixed until a hidden reversal time.

```text
for t < tau:
    p = p_pre
for t >= tau:
    p = p_post
```

Usually, the best arm before reversal becomes poor after reversal.

### Reversal Schedules

| Schedule | Description |
|---|---|
| Fixed hidden time | reversal at hidden `tau` |
| Performance-triggered | reversal after agent gets N rewards |
| Random hazard | reversal occurs with probability `h` each step |
| Multiple reversals | environment alternates several times |

### Metrics

- pre-reversal reward
- post-reversal recovery
- change detection delay
- perseveration length after reversal
- number of exploratory probes after losses

## Environment 3: DriftingBandit

### Purpose

Test adaptation when reward distributions change gradually.

### Setup

Reward probabilities evolve through a random walk or smooth drift.

```text
p_i(t + 1) = clip(p_i(t) + epsilon_i(t), 0.05, 0.95)
epsilon_i(t) ~ Normal(0, sigma)
```

Alternative:

```text
p_i(t) = sigmoid(b_i + slope_i * t)
```

### Metrics

- dynamic regret
- sensitivity to recent evidence
- overcommitment to stale evidence
- performance under different drift rates

## Environment 4: PeriodicBandit

### Purpose

Test whether the agent can infer temporal structure such as phase and period.

### Setup

Each arm has a reward probability that varies periodically:

```text
p_i(t) = clip(base_i + amp_i * sin(2*pi*t/period_i + phase_i), 0.01, 0.99)
```

Simpler variant:

```text
One arm is best during phase A, another during phase B.
```

### Action

```json
{"arm": 1}
```

### Observation

```json
{
  "chosen_arm": 1,
  "reward": 0,
  "step": 24
}
```

### Difficulty Parameters

- known vs unknown total horizon
- shared period across arms vs separate periods
- deterministic periodic reward vs probabilistic reward
- visible clock vs hidden clock
- short period vs long period
- phase shift mid-episode

### Metrics

- dynamic regret against best current arm
- phase-switch detection
- entropy of arm choices over phase
- ability to exploit periodic windows

## Environment 5: PendulumGate

### Purpose

Test timing, latent dynamics inference, and action planning under risk.

### Setup

The agent must cross a gate blocked by a swinging pendulum. The pendulum has hidden dynamics:

```text
period P
phase phi
danger_window width W
agent_crossing_time C
```

At each step, the agent may:

- wait
- probe/observe
- attempt crossing

### Action

```json
{"action": "wait"}
```

or

```json
{"action": "cross"}
```

Optional:

```json
{"action": "observe"}
```

### Observation

Minimal:

```json
{
  "event": "you waited",
  "pendulum_position_hint": "left"
}
```

After crossing:

```json
{
  "event": "hit",
  "reward": -1,
  "episode_status": "alive"
}
```

or:

```json
{
  "event": "success",
  "reward": 10,
  "episode_status": "done"
}
```

### Reward

```text
success: +10
hit: -1 or death penalty
wait: small negative cost
observe: small negative cost
```

### Difficulty Parameters

- deterministic vs noisy pendulum
- exact vs coarse position hints
- agent speed known vs unknown
- period static vs changes after several attempts
- one-shot death vs multiple lives

### Metrics

- attempts to success
- number of unsafe crossings
- waiting efficiency
- inferred period accuracy, if the agent reports it
- success rate under changed phase/period

## Environment Families to Add Later

| Environment | Idea |
|---|---|
| HiddenRuleCards | Wisconsin-card-sorting-like hidden rule changes. |
| MazeWithChangingDoors | Door colors alternate between safe/unsafe by hidden pattern. |
| ResourceForaging | Rewards regrow over time after harvesting. |
| PredatorTiming | Agent must infer predator patrol period. |
| DelayedRewardBandit | Action rewards appear several steps later, testing credit assignment. |

