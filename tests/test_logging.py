from __future__ import annotations

import json
from pathlib import Path

from llm_rl_engine.agents.classical import make_classical_agent
from llm_rl_engine.envs.number_guess import NumberGuess
from llm_rl_engine.logging.jsonl_logger import JSONLLogger
from llm_rl_engine.runners.run_episode import run_episode


def test_jsonl_logging(tmp_path: Path):
    env = NumberGuess()
    agent = make_classical_agent({"name": "random"})
    trajectory = run_episode(env, agent, horizon=5, seed=0)

    logger = JSONLLogger(tmp_path, "test_exp", {"horizon": 5})
    for record in trajectory.records:
        logger.log_step(trajectory, record, "NumberGuess")

    lines = logger.trajectories_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(trajectory.records)
    first = json.loads(lines[0])
    assert first["env_name"] == "NumberGuess"
    assert "observation" in first
    assert "action" in first