from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from llm_rl_engine.types import Trajectory, TrajectoryRecord


class JSONLLogger:
    def __init__(self, output_dir: Path, experiment_name: str, config: dict[str, Any]) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.run_dir = output_dir / f"{timestamp}_{experiment_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir = self.run_dir / "prompts"
        self.prompts_dir.mkdir(exist_ok=True)
        self.trajectories_path = self.run_dir / "trajectories.jsonl"
        self.config_path = self.run_dir / "config.yaml"
        self._write_config(config)

    @property
    def directory(self) -> Path:
        return self.run_dir

    def _write_config(self, config: dict[str, Any]) -> None:
        with self.config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def log_step(
        self,
        trajectory: Trajectory,
        record: TrajectoryRecord,
        env_name: str,
        prompt_mode: str | None = None,
    ) -> None:
        entry = {
            "run_id": trajectory.run_id,
            "env_name": env_name,
            "seed": trajectory.seed,
            "episode_id": trajectory.episode_id,
            "step": record.step,
            "observation": record.observation,
            "action": record.action,
            "reward": record.reward,
            "done": record.done,
            "public_info": record.public_info,
            "private_info": record.private_info,
            "agent_name": trajectory.agent_name,
            "prompt_mode": prompt_mode,
            "prompt": record.prompt,
            "raw_response": record.raw_response,
            "parsed_response": record.parsed_response,
            "parse_error": record.parse_error,
        }
        with self.trajectories_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        if record.prompt:
            prompt_path = (
                self.prompts_dir
                / f"{trajectory.run_id}_ep{trajectory.episode_id:03d}_step_{record.step:03d}.txt"
            )
            content = record.prompt
            if record.raw_response:
                content += "\n\n--- RESPONSE ---\n" + record.raw_response
            prompt_path.write_text(content, encoding="utf-8")

    def log_episode_summary(
        self,
        trajectory: Trajectory,
        env_name: str,
        metrics: dict[str, Any],
    ) -> None:
        summary = {
            "run_id": trajectory.run_id,
            "env_name": env_name,
            "agent_name": trajectory.agent_name,
            "seed": trajectory.seed,
            "episode_id": trajectory.episode_id,
            "cumulative_reward": sum(r.reward for r in trajectory.records),
            "n_steps": len(trajectory.records),
            "oracle": trajectory.oracle,
            "metrics": metrics,
        }
        summary_path = self.run_dir / "episode_summaries.jsonl"
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, default=str) + "\n")