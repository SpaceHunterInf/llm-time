from __future__ import annotations

import uuid
from typing import Any

from llm_rl_engine.agents.base import Agent
from llm_rl_engine.agents.classical import OracleAgent
from llm_rl_engine.envs.base import Env
from llm_rl_engine.types import Trajectory, TrajectoryRecord


def make_record(
    step: int,
    observation: dict[str, Any],
    action: dict[str, Any],
    reward: float,
    done: bool,
    public_info: dict[str, Any],
    private_info: dict[str, Any],
    agent_info: dict[str, Any] | None = None,
) -> TrajectoryRecord:
    agent_info = agent_info or {}
    return TrajectoryRecord(
        step=step,
        observation=observation,
        action=action,
        reward=reward,
        done=done,
        public_info=public_info,
        private_info=private_info,
        prompt=agent_info.get("prompt"),
        raw_response=agent_info.get("raw_response"),
        parsed_response=agent_info.get("parsed_response"),
        parse_error=agent_info.get("parse_error"),
        token_usage=agent_info.get("token_usage"),
    )


def run_episode(
    env: Env,
    agent: Agent,
    horizon: int,
    seed: int,
    episode_id: int = 0,
    run_id: str | None = None,
) -> Trajectory:
    run_id = run_id or str(uuid.uuid4())
    obs = env.reset(seed=seed)
    agent.reset(task_spec=env.metadata, seed=seed)

    if isinstance(agent, OracleAgent):
        agent.set_oracle(env.oracle())

    history: list[dict[str, Any]] = []
    records: list[TrajectoryRecord] = []

    for _t in range(horizon):
        if isinstance(agent, OracleAgent):
            agent.set_oracle(env.oracle())

        step_info: dict[str, Any] = {}
        if hasattr(agent, "act_with_info"):
            action, info = agent.act_with_info(obs, history)
            step_info = {
                "prompt": info.prompt,
                "raw_response": info.raw_response,
                "parsed_response": info.parsed_response,
                "parse_error": info.parse_error,
                "token_usage": info.token_usage,
            }
        else:
            action = agent.act(obs, history)

        result = env.step(action)
        public_info = {
            k: v for k, v in result.info.items() if not k.startswith("_")
            and k not in ("invalid_action",)
        }
        private_info = dict(result.info)

        record = make_record(
            step=obs.get("step", len(records)),
            observation=obs,
            action=action,
            reward=result.reward,
            done=result.done,
            public_info=public_info,
            private_info=private_info,
            agent_info=step_info,
        )
        records.append(record)

        history.append(
            {
                "step": record.step,
                "observation": obs,
                "action": action,
                "reward": result.reward,
                "done": result.done,
                "public_info": public_info,
                "private_info": private_info,
                "parsed_response": step_info.get("parsed_response"),
            }
        )
        obs = result.observation
        if result.done:
            break

    agent_name = getattr(agent, "name", agent.__class__.__name__)
    return Trajectory(
        records=records,
        oracle=env.oracle(),
        metadata=env.metadata,
        agent_name=agent_name,
        run_id=run_id,
        seed=seed,
        episode_id=episode_id,
    )