# Agent Protocol

This file defines how LLM agents interact with the engine.

## Main Principle

The environment should communicate with the agent through structured observations, but the LLM should receive a rendered text prompt and return strict JSON.

Do not rely on free-form text parsing for actions.

## Action Response Schema

Every LLM response must contain one JSON object.

For bandits:

```json
{
  "action": {"arm": 2},
  "confidence": 0.68,
  "rationale": "Arm 2 has the best recent reward rate, but I will keep testing alternatives occasionally."
}
```

For number guessing:

```json
{
  "action": {"guess": 50},
  "confidence": 0.5,
  "rationale": "Start near the midpoint to use comparison feedback."
}
```

For pendulum:

```json
{
  "action": {"action": "wait"},
  "confidence": 0.55,
  "rationale": "I need more timing information before crossing."
}
```

The runner should parse only the `action` field for environment execution.

## Prompt Modes

### Mode A: Raw History

The agent sees the full transcript of previous actions and rewards.

Use this to test whether the LLM can infer useful patterns without external summaries.

Prompt skeleton:

```text
You are an agent in a reinforcement-learning environment.
Your goal is to maximize cumulative reward over this episode.

Task:
{task_description}

Action space:
{action_space}

History:
{raw_history}

Current observation:
{observation}

Return exactly one JSON object with this schema:
{
  "action": ...,
  "confidence": number from 0 to 1,
  "rationale": string
}
```

### Mode B: Summary History

The agent sees a compact external summary.

This condition tests whether failure is caused by poor exploration or by inability to compress long histories.

Summary may include:

- counts per arm
- empirical reward per arm
- recent reward per arm
- last switch candidates
- uncertainty estimates

Important: this is an assisted condition. It should not replace raw-history evaluation.

### Mode C: Self-Summary

The agent must maintain its own summary.

At each step, it outputs:

```json
{
  "memory_update": "Arm 1 has 3 rewards out of 5; Arm 2 has 0 out of 2.",
  "action": {"arm": 1},
  "confidence": 0.7,
  "rationale": "Arm 1 currently has the best empirical rate."
}
```

The runner stores `memory_update` and shows it to the agent at the next step.

### Mode D: Heuristic Before Action

The agent must write or revise a compact heuristic before choosing.

```json
{
  "heuristic": "Prefer arms with high recent reward, but after two consecutive failures test another arm because the environment may change.",
  "action": {"arm": 3},
  "confidence": 0.62,
  "rationale": "Arm 3 has strong recent evidence and fits the current heuristic."
}
```

### Mode E: Prediction Before Action

The agent must forecast outcomes for candidate actions before choosing.

```json
{
  "predictions": [
    {"action": {"arm": 0}, "expected_reward": 0.35},
    {"action": {"arm": 1}, "expected_reward": 0.55}
  ],
  "action": {"arm": 1},
  "confidence": 0.6,
  "rationale": "Arm 1 has the highest expected reward."
}
```

This enables calibration analysis.

## Invalid Responses

If parsing fails:

1. Log the raw response.
2. Optionally retry once with a strict repair prompt.
3. If still invalid, execute a default invalid action penalty.

Recommended invalid action penalty:

```text
reward = -0.1
```

For tasks with death, invalid action should not automatically kill the agent unless that is part of the environment design.

## Anti-Leakage Rules

Never include:

- hidden reward probabilities
- target numbers
- reversal time
- period or phase
- oracle action
- private diagnostic info

Do include:

- public step number, if the environment allows a visible clock
- action space
- rewards already observed by the agent
- public feedback

## Prompt Randomization

To avoid semantic shortcuts:

- randomize arm labels: `A/B/C`, `red/blue/green`, `0/1/2`, abstract symbols
- vary task wording across seeds
- include purely abstract labels in at least one condition

## Context Management

For horizons above the context limit:

| Strategy | Purpose |
|---|---|
| truncate oldest raw history | minimal baseline |
| external sufficient statistics | assisted upper bound |
| self-summary memory | tests agent-generated compression |
| heuristic memory | tests policy-like compression |

Log which strategy is used.

