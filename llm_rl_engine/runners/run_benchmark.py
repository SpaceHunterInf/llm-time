from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_rl_engine.agents.classical import make_classical_agent
from llm_rl_engine.agents.llm import LLM_AGENT_NAMES, is_llm_agent_config, make_llm_agent
from llm_rl_engine.analysis.metrics import compute_episode_metrics, compute_run_metrics
from llm_rl_engine.analysis.plots import generate_run_plots
from llm_rl_engine.analysis.token_cost import aggregate_run_token_report
from llm_rl_engine.config.loader import load_config
from llm_rl_engine.envs.registry import make_env
from llm_rl_engine.logging.jsonl_logger import JSONLLogger
from llm_rl_engine.runners.run_episode import run_episode


@dataclass
class BenchmarkJob:
    env_cfg: dict[str, Any]
    agent_cfg: dict[str, Any]
    episode_id: int
    seed: int


@dataclass
class BenchmarkJobResult:
    trajectory: Any
    metrics: dict[str, Any]
    env_name: str
    prompt_mode: str | None


def make_agent(config: dict[str, Any]):
    name = config.get("name", "")
    if is_llm_agent_config(config):
        return make_llm_agent(config)
    return make_classical_agent(config)


def _agent_runs_on_env(agent_cfg: dict[str, Any], env_name: str) -> bool:
    allowed = agent_cfg.get("environments")
    if allowed is None:
        return True
    return env_name in allowed


def _run_job(job: BenchmarkJob, horizon: int) -> BenchmarkJobResult:
    env_name = job.env_cfg["name"]
    env_kwargs = {k: v for k, v in job.env_cfg.items() if k != "name"}
    env = make_env(env_name, **env_kwargs)
    agent = make_agent(job.agent_cfg)
    prompt_mode = job.agent_cfg.get("prompt_mode") or LLM_AGENT_NAMES.get(
        job.agent_cfg.get("name", ""), None
    )
    trajectory = run_episode(
        env=env,
        agent=agent,
        horizon=horizon,
        seed=job.seed,
        episode_id=job.episode_id,
        run_id=str(uuid.uuid4()),
    )
    metrics = compute_episode_metrics(trajectory)
    return BenchmarkJobResult(
        trajectory=trajectory,
        metrics=metrics,
        env_name=env_name,
        prompt_mode=prompt_mode,
    )


def _build_jobs(config: dict[str, Any]) -> list[BenchmarkJob]:
    n_episodes = config.get("n_episodes", 5)
    seeds = config.get("seeds", list(range(n_episodes)))
    jobs: list[BenchmarkJob] = []
    for env_cfg in config.get("environments", []):
        for agent_cfg in config.get("agents", []):
            if not _agent_runs_on_env(agent_cfg, env_cfg["name"]):
                continue
            for episode_id, seed in enumerate(seeds[:n_episodes]):
                jobs.append(
                    BenchmarkJob(
                        env_cfg=env_cfg,
                        agent_cfg=agent_cfg,
                        episode_id=episode_id,
                        seed=seed,
                    )
                )
    return jobs


def _is_llm_job(job: BenchmarkJob) -> bool:
    return is_llm_agent_config(job.agent_cfg)


def run_benchmark(
    config_path: str | Path,
    output_dir: str | Path = "runs",
) -> Path:
    config = load_config(config_path)
    experiment_name = config.get("experiment_name", "experiment")
    horizon = config.get("horizon", 20)
    llm_workers = int(config.get("llm_parallel_workers", 5))
    generate_plots = config.get("generate_plots", True)

    logger = JSONLLogger(Path(output_dir), experiment_name, config)
    jobs = _build_jobs(config)
    llm_jobs = [j for j in jobs if _is_llm_job(j)]
    classical_jobs = [j for j in jobs if not _is_llm_job(j)]

    results: list[BenchmarkJobResult] = []

    for job in classical_jobs:
        results.append(_run_job(job, horizon))

    if llm_jobs:
        with ThreadPoolExecutor(max_workers=max(1, llm_workers)) as pool:
            futures = {pool.submit(_run_job, job, horizon): job for job in llm_jobs}
            for future in as_completed(futures):
                results.append(future.result())

    all_summaries: list[dict[str, Any]] = []
    for result in results:
        for record in result.trajectory.records:
            logger.log_step(
                result.trajectory, record, result.env_name, prompt_mode=result.prompt_mode
            )
        logger.log_episode_summary(result.trajectory, result.env_name, result.metrics)
        all_summaries.append(
            {
                "agent_name": result.trajectory.agent_name,
                "env_name": result.env_name,
                "seed": result.trajectory.seed,
                "episode_id": result.trajectory.episode_id,
                "cumulative_reward": result.metrics["cumulative_reward"],
                "metrics": result.metrics,
            }
        )

    run_metrics = _aggregate_summaries(all_summaries)

    llm_model = _infer_llm_model(config)
    token_report = aggregate_run_token_report(
        str(logger.trajectories_path),
        model=llm_model,
        pricing=config.get("pricing"),
    )
    run_metrics["token_usage"] = token_report

    metrics_path = logger.directory / "metrics.json"
    metrics_path.write_text(json.dumps(run_metrics, indent=2, default=str), encoding="utf-8")

    token_path = logger.directory / "token_cost.json"
    token_path.write_text(json.dumps(token_report, indent=2, default=str), encoding="utf-8")

    if generate_plots:
        generate_run_plots(logger.directory, run_metrics, token_report)

    return logger.directory


def _infer_llm_model(config: dict[str, Any]) -> str:
    for agent_cfg in config.get("agents", []):
        if is_llm_agent_config(agent_cfg):
            return agent_cfg.get("model", "gemini-3.5-flash")
    return "gemini-3.5-flash"


def _aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in summaries:
        key = (s["agent_name"], s["env_name"])
        by_key.setdefault(key, []).append(s)
    return {
        f"{agent}__{env}": compute_run_metrics(rows) for (agent, env), rows in by_key.items()
    }