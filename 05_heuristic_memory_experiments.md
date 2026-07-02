# Heuristic Memory Experiments

## Goal

Test whether explicit heuristics help frozen LLM agents adapt in long-horizon static and dynamic environments.

The key idea is to separate:

- planning from current observation
- remembering raw history
- summarizing statistics
- inducing reusable heuristics
- revising heuristics after feedback

## Definition

A heuristic is a compact natural-language rule or policy sketch that guides action selection.

Examples:

```text
If an arm has recently failed several times after previously succeeding, test alternatives because the environment may have reversed.
```

```text
In periodic environments, do not treat old rewards as permanently valid; track which arm works at which step range.
```

```text
For number guessing, use interval narrowing rather than random guessing.
```

## Experimental Conditions

### Condition 1: No Heuristic

The agent gets only task description, history, and current observation.

Purpose:

- baseline for raw in-context adaptation

### Condition 2: Generic Preset Heuristic

The prompt includes a generic rule:

```text
You should explore uncertain actions early, exploit actions with strong evidence, and revise your belief when recent evidence contradicts old evidence.
```

Purpose:

- tests whether simple prompting improves behavior

### Condition 3: Task-Specific Preset Heuristic

The prompt includes environment-specific advice without revealing hidden parameters.

For reversal bandit:

```text
The best arm may change. If a previously good arm starts failing repeatedly, probe other arms.
```

For periodic bandit:

```text
Rewards may depend on time. Track whether arms work better at different steps instead of estimating one fixed value per arm.
```

Purpose:

- tests whether knowing the right hypothesis class helps

### Condition 4: Self-Generated Initial Heuristic

Before the first action, the agent must propose a heuristic based only on the task description.

Response schema:

```json
{
  "heuristic": "I will initially sample every arm, then compare recent rewards and revisit arms if outcomes change.",
  "action": {"arm": 0},
  "confidence": 0.5
}
```

Purpose:

- tests whether LLM priors produce useful exploration strategies

### Condition 5: Self-Updated Heuristic

After every step, the agent must revise or retain its heuristic.

Response schema:

```json
{
  "heuristic_update": {
    "previous": "Arm 1 seemed best.",
    "new": "Arm 1 was best early, but recent failures suggest a reversal or periodic change. Probe Arm 3 next.",
    "changed": true
  },
  "action": {"arm": 3},
  "confidence": 0.61
}
```

Purpose:

- tests whether the agent can turn feedback into procedural memory

### Condition 6: Episode-Level Reflection

The agent updates heuristic only after an episode ends.

Purpose:

- closer to Reflexion-style verbal reinforcement learning
- tests cross-episode improvement

### Condition 7: Wrong or Stale Heuristic

The prompt includes misleading advice.

Example:

```text
The best arm is usually stable. Once you find a good arm, keep exploiting it.
```

Purpose:

- tests whether the agent can override bad heuristics with evidence
- reveals prompt overcompliance

## Heuristic Quality Metrics

### Predictive Utility

Does the heuristic help predict the agent's next action?

Measure:

- ask another evaluator model or rule-based parser to infer recommended action from heuristic
- compare with actual action

### Environmental Accuracy

Does the heuristic describe the true hidden structure?

Examples:

- says environment is static when it is periodic: inaccurate
- says arm quality changes after reversal: accurate
- estimates period close to true period: partially accurate

### Compression Quality

Does the heuristic preserve useful information from the trajectory?

Compare:

- raw history agent
- heuristic-only agent
- heuristic plus last N steps

### Adaptivity

Does the heuristic change when evidence changes?

Metrics:

- update frequency
- update delay after reversal
- whether old false claims are removed
- whether new claims are added after evidence

### Staleness

Does the agent keep using an obsolete heuristic after environment changes?

Metrics:

- number of steps heuristic remains wrong after reversal
- reward loss while heuristic is stale

## Prompt Template: Heuristic Before Action

```text
You are a frozen LLM agent in a reinforcement-learning environment.
Your goal is to maximize cumulative reward over a long episode.

You cannot update your weights. Your only way to adapt is to use the interaction history and maintain a useful heuristic.

Task:
{task_description}

Action space:
{action_space}

Previous heuristic:
{previous_heuristic}

Recent history:
{recent_history}

Current observation:
{observation}

Before choosing, revise your heuristic if the evidence suggests it is stale or incomplete.

Return exactly one JSON object:
{
  "heuristic": "compact rule for future actions",
  "action": ...,
  "confidence": number from 0 to 1,
  "rationale": "brief reason"
}
```

## Prompt Template: Episode Reflection

```text
You completed an episode in an RL environment.

Task:
{task_description}

Trajectory:
{trajectory}

Final score:
{score}

Write a compact heuristic that would improve performance in a future episode from the same environment family.

Return exactly one JSON object:
{
  "heuristic": "...",
  "evidence": "...",
  "failure_modes": ["..."],
  "recommended_strategy": "..."
}
```

## Expected Failure Modes

| Failure Mode | Description |
|---|---|
| Diary instead of heuristic | Agent restates history but gives no reusable rule. |
| Overcommitment | Agent finds one good action and stops exploring. |
| Stale rule | Agent keeps an old rule after reversal. |
| False periodicity | Agent hallucinates cycles in noise. |
| Under-exploration | Agent refuses to test uncertain actions. |
| Prompt obedience to wrong heuristic | Agent follows bad advice despite contradictory evidence. |
| Verbose memory bloat | Heuristic grows too long and becomes unusable. |

## Paper 2 Hypotheses

H1:

> Self-generated heuristics improve performance over raw history in long-horizon dynamic environments.

H2:

> Task-specific preset heuristics improve early performance but may reduce discovery if they are too strong.

H3:

> Heuristics help most when the environment has reusable structure, such as reversal or periodicity.

H4:

> Wrong heuristics harm LLM agents more than classical baselines, revealing overreliance on linguistic priors.

H5:

> Heuristic quality predicts reward better than rationale quality.

