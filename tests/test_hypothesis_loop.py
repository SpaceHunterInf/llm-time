from __future__ import annotations

from llm_rl_engine.agents.history_views import render_bandit_stats, render_interface_blocks
from llm_rl_engine.agents.llm import make_llm_agent
from llm_rl_engine.analysis.metrics import _hypothesis_loop_metrics, compute_episode_metrics
from llm_rl_engine.types import TokenUsage, Trajectory, TrajectoryRecord


def test_stats_tally_from_history():
    history = [
        {"step": 0, "action": {"arm": 1}, "reward": 1.0, "observation": {"message": "x"}},
        {"step": 1, "action": {"arm": 1}, "reward": 0.0, "observation": {"message": "x"}},
        {"step": 2, "action": {"arm": 0}, "reward": 1.0, "observation": {"message": "x"}},
    ]
    text = render_bandit_stats(history, n_arms=4, last_k=5, recency_window=10)
    assert "arm" in text
    assert "mean_all" in text
    assert "mean_w10" in text
    assert "1.000" in text or "0.500" in text
    assert "regime" not in text.lower()
    assert "0.85" not in text


def test_interface_modes():
    history = [
        {"step": i, "action": {"arm": 0}, "reward": 1.0, "observation": {"message": ""}}
        for i in range(20)
    ]
    stats_r, hist_r = render_interface_blocks(history, 4, "raw", history_tail=10)
    assert stats_r == "(not provided)"
    assert "Step 0:" in hist_r
    stats_s, hist_s = render_interface_blocks(history, 4, "stats", history_tail=10)
    assert "Per-arm" in stats_s
    assert "omitted" in hist_s.lower() or "omitted" in hist_s or "raw step" in hist_s
    _stats, hist_sr = render_interface_blocks(history, 4, "stats_plus_recent", history_tail=10)
    assert "omitted 10" in hist_sr
    assert "Step 19:" in hist_sr
    assert "Step 0:" not in hist_sr


def test_hypothesis_prompt_has_stats_placeholder():
    agent = make_llm_agent(
        {
            "name": "llm_hypothesis",
            "prompt_mode": "hypothesis_loop",
            "interface": "stats_plus_recent",
            "n_arms": 4,
        }
    )
    agent.reset({"name": "StagedBandit", "n_arms": 4, "description": "bandit", "action_space": "arm"})
    prompt = agent._build_prompt({"step": 0, "message": "start"}, [])
    assert "hypothesis" in prompt.lower()
    assert "Sufficient statistics" in prompt
    assert "regime" not in prompt.lower()
    assert "mean_w" in agent._build_prompt(
        {"step": 1, "message": "x"},
        [{"step": 0, "action": {"arm": 1}, "reward": 1.0, "observation": {}}],
    )


def test_taught_surprise_prompt_is_meta_not_detect():
    agent = make_llm_agent(
        {
            "name": "llm_hyp_taught_surprise",
            "prompt_mode": "hypothesis_taught_surprise",
            "interface": "stats_plus_recent",
            "n_arms": 4,
        }
    )
    agent.reset({"name": "StagedBandit", "n_arms": 4, "description": "bandit", "action_space": "arm"})
    prompt = agent._build_prompt({"step": 0, "message": "start"}, [])
    lower = prompt.lower()
    template = agent._template.lower()
    assert "belief-revision" in lower
    assert "surprising" in lower
    # Stats tables may list mean_w; the *instruction* must not name it.
    assert "mean_w" not in template
    assert "collaps" not in template
    assert "regime" not in template
    assert "detection rule" not in template


def test_taught_detect_prompt_instructs_recency():
    agent = make_llm_agent(
        {
            "name": "llm_hyp_taught_detect",
            "prompt_mode": "hypothesis_taught_detect",
            "interface": "stats_plus_recent",
            "n_arms": 4,
        }
    )
    agent.reset({"name": "StagedBandit", "n_arms": 4, "description": "bandit", "action_space": "arm"})
    prompt = agent._build_prompt({"step": 0, "message": "start"}, [])
    assert "Detection rule" in prompt
    assert "mean_w" in prompt
    assert "lifetime mean" in prompt.lower() or "mean_all" in prompt
    assert "regime" not in prompt.lower()


def _rec(step: int, hyp_arm: int, true_probs: list[float], stable: bool = True, conf: float = 0.9):
    return TrajectoryRecord(
        step=step,
        observation={"step": step},
        action={"arm": hyp_arm},
        reward=1.0,
        done=False,
        private_info={"step_probs": true_probs},
        parsed_response={
            "hypothesis": {"type": "best_arm", "best_arm": hyp_arm, "stable": stable},
            "confidence": conf,
            "action": {"arm": hyp_arm},
        },
        token_usage=TokenUsage(),
    )


def test_lock_falsify_relock_metrics():
    # Stage 0: true best 3; lock immediately. Stage 1: true best 1; keep 3 then switch.
    recs = []
    p0 = [0.2, 0.2, 0.2, 0.85]
    p1 = [0.2, 0.85, 0.2, 0.2]
    for t in range(10):
        recs.append(_rec(t, 3, p0))
    for t in range(10, 13):
        recs.append(_rec(t, 3, p1, stable=True))  # still old H
    recs.append(_rec(13, 1, p1, stable=False, conf=0.4))
    for t in range(14, 20):
        recs.append(_rec(t, 1, p1, stable=True))

    traj = Trajectory(
        records=recs,
        oracle={
            "stage_length": 10,
            "n_stages": 2,
            "stage_params": [{"best": 3}, {"best": 1}],
        },
        metadata={"name": "StagedBandit"},
        agent_name="llm_hypothesis",
    )
    m = _hypothesis_loop_metrics(traj, traj.oracle, k_lock=3)
    assert m["lock_time"][0] == 1
    assert m["falsification_delay"][1] == 4  # steps 10,11,12 still old; 13 switches (1-based 4)
    assert m["relock_time"][1] == 4  # consecutive correct from index 13 → start 4 in stage
    assert m["mean_confidence"] is not None
    assert compute_episode_metrics(traj)["lock_time"][0] == 1
