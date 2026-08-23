from __future__ import annotations

from llm_rl_engine.agents.classical import make_classical_agent
from llm_rl_engine.analysis.metrics import compute_episode_metrics
from llm_rl_engine.envs.registry import make_env
from llm_rl_engine.envs.staged_bandit import StagedBandit
from llm_rl_engine.runners.run_episode import run_episode


def test_staged_bandit_hard_cut_constant():
    env = StagedBandit(
        n_arms=4,
        stage_length=10,
        force_distinct_best_arms=True,
        stages=[{"family": "constant"}, {"family": "constant"}, {"family": "constant"}],
    )
    env.reset(seed=0)
    params = env.oracle()["stage_params"]
    bests = [p["best"] for p in params]
    assert len(set(bests)) == 3

    # Within stage 0, best arm should match stage 0
    for _ in range(10):
        best = env.oracle()["best_arm"]
        assert best == bests[0]
        env.step({"arm": best})

    # Stage 1 begins
    assert env._stage_index_at(env._step) == 1
    assert env.oracle()["stage_params"][1]["best"] == bests[1]


def test_staged_bandit_mixed_families():
    env = make_env(
        "StagedBandit",
        stage_length=5,
        stages=[
            {"family": "sinusoidal"},
            {"family": "constant"},
            {"family": "linear", "linear_variant": "ramp_up"},
        ],
    )
    env.reset(seed=1)
    families = [p["family"] for p in env.oracle()["stage_params"]]
    assert families == ["sinusoidal", "constant", "linear"]

    # Advance into linear stage and check ramp increases
    for _ in range(10):
        env.step({"arm": 0})
    p_mid = list(env.oracle()["current_probs"])
    for _ in range(4):
        env.step({"arm": 0})
    p_late = list(env.oracle()["current_probs"])
    best = env.oracle()["stage_params"][2]["best"]
    assert p_late[best] >= p_mid[best] - 1e-9


def test_staged_bandit_no_stage_leak_in_obs():
    env = StagedBandit(stage_length=5, stages=[{"family": "constant"}] * 2)
    obs = env.reset(seed=0)
    assert "stage" not in str(obs.get("public_state", {})).lower() or "stage_index" not in obs
    result = env.step({"arm": 0})
    pub = result.observation.get("public_state", {})
    assert "stage_index" not in pub
    assert "stage_family" not in pub
    assert "step_probs" not in pub


def test_staged_bandit_metrics_recovery_fields():
    env = StagedBandit(
        stage_length=15,
        stages=[{"family": "constant"}, {"family": "constant"}, {"family": "constant"}],
        force_distinct_best_arms=True,
    )
    agent = make_classical_agent({"name": "oracle"})
    traj = run_episode(env, agent, horizon=45, seed=0)
    metrics = compute_episode_metrics(traj)
    assert metrics["n_stages"] == 3
    assert len(metrics["stage_rewards"]) == 3
    assert metrics["recovery_delay"][0] is None
    # Oracle should recover immediately (delay 1) on later stages
    assert metrics["recovery_delay"][1] == 1
    assert metrics["regret"] is not None


def test_registry_staged_bandit():
    env = make_env(
        "StagedBandit",
        stages=["constant", "sinusoidal", "linear"],
        stage_length=10,
    )
    assert env.metadata["name"] == "StagedBandit"
    env.reset(seed=0)
    assert len(env.oracle()["stage_params"]) == 3


def test_staged_description_modes_no_leak_in_blind():
    from llm_rl_engine.envs.staged_bandit import staged_bandit_description

    blind = staged_bandit_description(4, "blind").lower()
    aware = staged_bandit_description(4, "aware").lower()
    minimal = staged_bandit_description(4, "minimal").lower()

    for leak in ("regime", "stage", "abandon", "non-station", "fixed"):
        assert leak not in blind, f"blind description leaks {leak!r}: {blind}"
    # "change" must not appear as coaching (allow none)
    assert "change" not in blind
    assert "regime" in aware and "change" in aware
    assert "arm" in minimal

    env_blind = StagedBandit(description_mode="blind")
    env_aware = StagedBandit(description_mode="aware")
    assert env_blind.metadata["description_mode"] == "blind"
    assert "regime" not in env_blind.metadata["description"].lower()
    assert "regime" in env_aware.metadata["description"].lower()


def test_llm_agent_description_override():
    from llm_rl_engine.agents.llm import make_llm_agent

    agent = make_llm_agent(
        {
            "name": "llm_aware",
            "description_mode": "aware",
            "n_arms": 4,
            "model": "gemini-3.5-flash",
        }
    )
    agent.reset({"name": "StagedBandit", "description": "SHOULD BE OVERRIDDEN", "n_arms": 4})
    assert "regime" in agent._task_spec["description"].lower()
    assert "SHOULD BE OVERRIDDEN" not in agent._task_spec["description"]
