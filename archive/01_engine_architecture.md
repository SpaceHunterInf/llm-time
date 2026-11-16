# Engine Architecture

## Goal

Build a small, extensible RL engine where both classical policies and LLM agents can interact with the same environments through a shared API.

The engine should support:

- static tasks such as number guessing and stationary bandits
- non-stationary tasks such as reversal and drifting bandits
- time-dependent tasks such as periodic bandits and pendulum timing gates
- long-horizon interaction
- detailed trajectory logging
- prompt-condition comparisons
- deterministic reproduction via seeds

## Core Abstractions

### Environment

Each environment exposes a Gym-like interface.

```python
class Env:
    def reset(self, seed: int | None = None) -> Observation:
        ...

    def step(self, action: Action) -> StepResult:
        ...

    def oracle(self) -> dict:
        """Return hidden parameters for analysis only, never for the agent."""
        ...

    @property
    def action_space(self) -> ActionSpace:
        ...

    @property
    def metadata(self) -> dict:
        ...
```

### Observation

Use a structured object. Avoid passing raw strings inside the engine.

```python
Observation = dict
```

Minimum fields:

```json
{
  "step": 0,
  "message": "Episode started.",
  "public_state": {},
  "last_action": null,
  "last_reward": null,
  "done": false
}
```

For LLM agents, convert this structured observation into text at the prompt boundary.

### StepResult

```python
@dataclass
class StepResult:
    observation: dict
    reward: float
    done: bool
    info: dict
```

`info` may include hidden diagnostic information, but this must not be shown to the agent.

### Agent

```python
class Agent:
    def reset(self, task_spec: dict, seed: int | None = None) -> None:
        ...

    def act(self, observation: dict, history: list[dict]) -> Action:
        ...
```

For LLM agents, `act` should:

1. build prompt from task spec, observation, and history
2. call model
3. parse action JSON
4. validate against action space
5. return the parsed action

### Runner

```python
def run_episode(env: Env, agent: Agent, horizon: int, seed: int) -> Trajectory:
    obs = env.reset(seed=seed)
    agent.reset(task_spec=env.metadata, seed=seed)
    history = []

    for t in range(horizon):
        action = agent.act(obs, history)
        result = env.step(action)
        record = make_record(obs, action, result)
        history.append(record)
        obs = result.observation
        if result.done:
            break

    return Trajectory(records=history, oracle=env.oracle(), metadata=env.metadata)
```

## Logging

Every step should be stored as JSONL.

Required fields:

```json
{
  "run_id": "uuid",
  "env_name": "PeriodicBandit",
  "seed": 123,
  "episode_id": 0,
  "step": 7,
  "observation": {},
  "action": {},
  "reward": 1.0,
  "done": false,
  "public_info": {},
  "private_info": {},
  "agent_name": "gpt_raw_history",
  "prompt_mode": "raw_history",
  "prompt": "...",
  "raw_response": "...",
  "parsed_response": {},
  "parse_error": null
}
```

For classical baselines, `prompt` and `raw_response` can be null.

## Result Directory Layout

```text
runs/
  2026-xx-xx_experiment_name/
    config.yaml
    trajectories.jsonl
    metrics.json
    prompts/
      run_id_step_000.txt
```

## Config Example

```yaml
experiment_name: smoke_periodic_bandit
horizon: 40
n_episodes: 20
seeds: [0, 1, 2, 3, 4]

environments:
  - name: PeriodicBandit
    n_arms: 4
    period_range: [8, 20]
    amplitude_range: [0.15, 0.35]
    base_range: [0.25, 0.65]

agents:
  - name: random
  - name: ucb
  - name: sliding_window_ucb
    window: 10
  - name: llm_raw_history
    model: MODEL_NAME
  - name: llm_heuristic_before_action
    model: MODEL_NAME
```

## Important Implementation Constraints

- The environment must never leak hidden parameters into observations.
- The agent must output a machine-parseable action.
- Invalid actions should be logged and penalized with a configurable penalty.
- Keep task rendering separate from task mechanics.
- Use deterministic seeds for hidden parameters and stochastic rewards.

