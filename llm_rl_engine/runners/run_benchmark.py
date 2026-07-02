from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from llm_rl_engine.agents.classical import make_classical_agent
from llm_rl_engine.agents.llm import LLM_AGENT_NAMES, make_llm_agent
from llm_rl_engine.analysis.metrics import compute_episode_metrics, compute_run_metrics
from llm_rl_engine.config.loader import load_config
from llm_rl_engine.envs.registry import make_env
from llm_rl_engine.logging.jsonl_logger import JSONLLogger
from llm_rl_engine.runners.run_episode import run_episode


def make_agent(config: dict[str, Any]):
    name = config.get("name", "")
    if name.startswith("llm") or name in LLM_AGENT_NAMES:
        return make_llm_agent(config)
    return make_classical_agent(config)


def run_benchmark(
    config_path: str | Path,
    output_dir: str | Path = "runs",
) -> Path:
    config = load_config(config_path)
    experiment_name = config.get("experiment_name", "experiment")
    horizon = config.get("horizon", 20)
    n_episodes = config.get("n_episodes", 5)
    seeds = config.get("seeds", list(range(n_episodes)))

    logger = JSONLLogger(Path(output_dir), experiment_name, config)
    all_summaries: list[dict[str, Any]] = []

    for env_cfg in config.get("environments", []):
        env_name = env_cfg["name"]
        env_kwargs = {k: v for k, v in env_cfg.items() if k != "name"}
        env = make_env(env_name, **env_kwargs)

        for agent_cfg in config.get("agents", []):
            agent = make_agent(agent_cfg)
            prompt_mode = agent_cfg.get("prompt_mode") or LLM_AGENT_NAMES.get(
                agent_cfg.get("name", ""), None
            )

            for episode_id, seed in enumerate(seeds[:n_episodes]):
                run_id = str(uuid.uuid4())
                trajectory = run_episode(
                    env=env,
                    agent=agent,
                    horizon=horizon,
                    seed=seed,
                    episode_id=episode_id,
                    run_id=run_id,
                )
                metrics = compute_episode_metrics(trajectory)
                for record in trajectory.records:
                    logger.log_step(trajectory, record, env_name, prompt_mode=prompt_mode)
                logger.log_episode_summary(trajectory, env_name, metrics)
                all_summaries.append(
                    {
                        "agent_name": trajectory.agent_name,
                        "env_name": env_name,
                        "seed": seed,
                        "episode_id": episode_id,
                        "cumulative_reward": metrics["cumulative_reward"],
                        "metrics": metrics,
                    }
                )

    run_metrics = _aggregate_summaries(all_summaries)
    metrics_path = logger.directory / "metrics.json"
    metrics_path.write_text(json.dumps(run_metrics, indent=2, default=str), encoding="utf-8")
    return logger.directory


def _aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in summaries:
        key = (s["agent_name"], s["env_name"])
        by_key.setdefault(key, []).append(s)
    return {
        f"{agent}__{env}": compute_run_metrics(rows) for (agent, env), rows in by_key.items()
    }