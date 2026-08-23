from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_rl_engine.envs.bandits import BANDIT_ENV_NAMES
from llm_rl_engine.types import Trajectory


def compute_episode_metrics(trajectory: Trajectory) -> dict[str, Any]:
    rewards = [r.reward for r in trajectory.records]
    cumulative = sum(rewards)
    invalid_count = sum(
        1 for r in trajectory.records if r.private_info.get("invalid_action")
    )
    parse_errors = sum(1 for r in trajectory.records if r.parse_error)

    input_tokens = sum(
        (r.token_usage.input_tokens if r.token_usage else 0) for r in trajectory.records
    )
    output_tokens = sum(
        (r.token_usage.output_tokens if r.token_usage else 0) for r in trajectory.records
    )

    metrics: dict[str, Any] = {
        "cumulative_reward": cumulative,
        "n_steps": len(trajectory.records),
        "invalid_action_rate": invalid_count / max(len(trajectory.records), 1),
        "parse_error_rate": parse_errors / max(len(trajectory.records), 1),
        "reward_per_step": rewards,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    env_name = trajectory.metadata.get("name", "")
    oracle = trajectory.oracle

    if env_name == "NumberGuess":
        metrics.update(_number_guess_metrics(trajectory, oracle))
    elif env_name in BANDIT_ENV_NAMES:
        metrics.update(_bandit_metrics(trajectory, oracle, env_name))
        if env_name == "StagedBandit":
            metrics.update(_staged_bandit_metrics(trajectory, oracle))
        if env_name == "ExploitDecayBandit":
            metrics.update(_exploit_decay_metrics(trajectory, oracle))
        if _has_hypothesis_fields(trajectory):
            metrics.update(_hypothesis_loop_metrics(trajectory, oracle))
    elif env_name == "PendulumGate":
        metrics.update(_pendulum_metrics(trajectory, oracle))
    elif env_name == "TrafficLightGate":
        metrics.update(_traffic_light_metrics(trajectory, oracle))

    metrics["regret"] = _compute_regret(trajectory, oracle, env_name)
    return metrics


def _number_guess_metrics(trajectory: Trajectory, oracle: dict[str, Any]) -> dict[str, Any]:
    target = oracle.get("target")
    steps_to_target = None
    for i, rec in enumerate(trajectory.records):
        guess = rec.action.get("guess")
        if guess == target:
            steps_to_target = i + 1
            break
    final_guess = trajectory.records[-1].action.get("guess") if trajectory.records else None
    final_error = (
        abs(final_guess - target) if final_guess is not None and target is not None else None
    )
    return {
        "steps_to_target": steps_to_target,
        "final_error": final_error,
    }


def _bandit_metrics(
    trajectory: Trajectory, oracle: dict[str, Any], env_name: str
) -> dict[str, Any]:
    probs = oracle.get("current_probs") or oracle.get("probs", [])
    best_arm = oracle.get("best_arm")
    if best_arm is None and probs:
        best_arm = max(range(len(probs)), key=lambda i: probs[i])

    chosen_arms = [r.action.get("arm") for r in trajectory.records]
    exploration = sum(1 for a in chosen_arms if a != best_arm) / max(len(chosen_arms), 1)

    metrics: dict[str, Any] = {
        "exploration_rate": exploration,
        "best_arm": best_arm,
        "final_arm": chosen_arms[-1] if chosen_arms else None,
    }

    if env_name == "ReversalBandit":
        tau = oracle.get("tau")
        if tau is not None:
            post_records = [r for r in trajectory.records if r.step >= tau]
            post_best = oracle.get("post_probs", [])
            if post_records and post_best:
                new_best = max(range(len(post_best)), key=lambda i: post_best[i])
                switch_delay = None
                for i, rec in enumerate(post_records):
                    if rec.action.get("arm") == new_best:
                        switch_delay = i + 1
                        break
                metrics["switch_delay"] = switch_delay
    return metrics


def _staged_bandit_metrics(
    trajectory: Trajectory, oracle: dict[str, Any]
) -> dict[str, Any]:
    """Stage-level reward, recovery delay, and overcommit after regime changes."""
    stage_length = int(oracle.get("stage_length") or 30)
    n_stages = int(oracle.get("n_stages") or 1)
    records = trajectory.records
    W = 5
    W_oc = 10
    K_rec = 3

    stage_rewards: list[float] = []
    stage_regrets: list[float] = []
    recovery_delays: list[int | None] = []
    overcommit_rates: list[float | None] = []
    post_change_mean_rewards: list[float | None] = []

    prev_end_best: int | None = None
    for s in range(n_stages):
        stage_recs = [
            r
            for r in records
            if r.private_info.get("stage_index", r.step // stage_length) == s
        ]
        if not stage_recs:
            # Fallback: partition by step index
            stage_recs = [
                r
                for r in records
                if s * stage_length <= r.step < (s + 1) * stage_length
            ]

        stage_rewards.append(sum(r.reward for r in stage_recs))
        stage_regret = 0.0
        for r in stage_recs:
            arm = r.action.get("arm")
            probs = r.private_info.get("step_probs") or []
            if arm is None or not probs:
                continue
            stage_regret += max(probs) - (probs[arm] if arm < len(probs) else 0.0)
        stage_regrets.append(stage_regret)

        if s == 0:
            recovery_delays.append(None)
            overcommit_rates.append(None)
            post_change_mean_rewards.append(None)
        else:
            post = stage_recs[:W]
            post_change_mean_rewards.append(
                sum(r.reward for r in post) / max(len(post), 1) if post else None
            )
            recovery_delays.append(_recovery_delay(stage_recs, K_rec))
            if prev_end_best is not None and stage_recs:
                window = stage_recs[:W_oc]
                over = sum(
                    1 for r in window if r.action.get("arm") == prev_end_best
                )
                overcommit_rates.append(over / max(len(window), 1))
            else:
                overcommit_rates.append(None)

        if stage_recs:
            last_probs = stage_recs[-1].private_info.get("step_probs") or []
            if last_probs:
                prev_end_best = max(range(len(last_probs)), key=lambda i: last_probs[i])
            else:
                prev_end_best = stage_recs[-1].private_info.get("best_arm")
        else:
            prev_end_best = None

    families = [
        (p or {}).get("family")
        for p in (oracle.get("stage_params") or [])
    ]
    return {
        "n_stages": n_stages,
        "stage_length": stage_length,
        "stage_families": families,
        "stage_rewards": stage_rewards,
        "stage_regrets": stage_regrets,
        "recovery_delay": recovery_delays,
        "overcommit_rate": overcommit_rates,
        "post_change_mean_reward": post_change_mean_rewards,
        "mean_recovery_delay": _safe_mean(
            [d for d in recovery_delays if d is not None]
        ),
        "mean_overcommit_rate": _safe_mean(
            [r for r in overcommit_rates if r is not None]
        ),
    }


def _exploit_decay_metrics(
    trajectory: Trajectory, oracle: dict[str, Any]
) -> dict[str, Any]:
    """Overmine after alpha death and recovery onto the new alpha."""
    records = trajectory.records
    W = 10
    K_rec = 3

    death_indices = [
        i for i, r in enumerate(records) if r.private_info.get("alpha_died")
    ]
    death_steps = [records[i].step for i in death_indices]

    overmine_rates: list[float | None] = []
    recovery_delays: list[int | None] = []
    post_death_rewards: list[float | None] = []

    for di in death_indices:
        death_rec = records[di]
        old_alpha = death_rec.private_info.get("alpha_arm")
        if old_alpha is None:
            old_alpha = death_rec.action.get("arm")
        new_alpha = death_rec.private_info.get("alpha_arm_after")
        post = records[di + 1 :]

        window = post[:W]
        if old_alpha is not None and window:
            over = sum(1 for r in window if r.action.get("arm") == old_alpha)
            overmine_rates.append(over / len(window))
        else:
            overmine_rates.append(None)

        delay: int | None = None
        streak = 0
        for i, r in enumerate(post):
            target = new_alpha
            if target is None:
                probs = r.private_info.get("step_probs") or []
                if probs:
                    target = max(range(len(probs)), key=lambda j: probs[j])
            if target is not None and r.action.get("arm") == target:
                streak += 1
                if streak >= K_rec:
                    delay = i - K_rec + 2  # 1-based index of streak start within post
                    break
            else:
                streak = 0
        recovery_delays.append(delay)

        if window:
            post_death_rewards.append(sum(r.reward for r in window) / len(window))
        else:
            post_death_rewards.append(None)

    valid_om = [x for x in overmine_rates if x is not None]
    valid_rec = [x for x in recovery_delays if x is not None]
    valid_pr = [x for x in post_death_rewards if x is not None]
    return {
        "n_alpha_deaths": int(oracle.get("n_alpha_deaths") or len(death_steps)),
        "alpha_death_steps": death_steps,
        "overmine_rate_after_death": overmine_rates,
        "recovery_delay_after_death": recovery_delays,
        "post_death_mean_reward": post_death_rewards,
        "mean_overmine_rate": sum(valid_om) / len(valid_om) if valid_om else None,
        "mean_recovery_delay": sum(valid_rec) / len(valid_rec) if valid_rec else None,
        "mean_post_death_reward": sum(valid_pr) / len(valid_pr) if valid_pr else None,
    }


def _recovery_delay(stage_recs: list[Any], k_rec: int = 3) -> int | None:
    """First offset (1-based) where agent matches oracle best for k_rec consecutive steps."""
    if not stage_recs:
        return None
    matches: list[bool] = []
    for r in stage_recs:
        arm = r.action.get("arm")
        best = r.private_info.get("best_arm")
        probs = r.private_info.get("step_probs") or []
        if best is None and probs:
            best = max(range(len(probs)), key=lambda i: probs[i])
        matches.append(arm is not None and best is not None and arm == best)

    for d in range(len(matches)):
        window = matches[d : d + k_rec]
        if len(window) == k_rec and all(window):
            return d + 1
    return None


def _traffic_light_metrics(trajectory: Trajectory, oracle: dict[str, Any]) -> dict[str, Any]:
    success = any(r.private_info.get("event") == "success" for r in trajectory.records)
    go_attempts = sum(1 for r in trajectory.records if r.action.get("action") == "go")
    unsafe = sum(1 for r in trajectory.records if r.private_info.get("unsafe"))
    shift_time = oracle.get("phase_shift_time")
    return {
        "success": success,
        "go_attempts": go_attempts,
        "unsafe_goes": unsafe,
        "final_lives": oracle.get("lives"),
        "phase_shift_time": shift_time,
        "phase_shifted": oracle.get("phase_shifted"),
    }


def _pendulum_metrics(trajectory: Trajectory, oracle: dict[str, Any]) -> dict[str, Any]:
    success = any(r.private_info.get("event") == "success" for r in trajectory.records)
    cross_attempts = sum(
        1 for r in trajectory.records if r.action.get("action") == "cross"
    )
    unsafe = sum(1 for r in trajectory.records if r.private_info.get("unsafe"))
    shift_time = oracle.get("phase_shift_time")
    recovery_delay = None
    if shift_time is not None:
        post_shift = [
            r
            for r in trajectory.records
            if (r.observation.get("public_state") or {}).get("time_s", 0) >= shift_time
        ]
        for i, rec in enumerate(post_shift):
            if rec.action.get("action") == "cross":
                recovery_delay = i + 1
                break
    return {
        "success": success,
        "cross_attempts": cross_attempts,
        "unsafe_crosses": unsafe,
        "final_lives": oracle.get("lives"),
        "phase_shift_time": shift_time,
        "phase_shifted": oracle.get("phase_shifted"),
        "recovery_delay_after_shift": recovery_delay,
    }


def _compute_regret(
    trajectory: Trajectory, oracle: dict[str, Any], env_name: str
) -> float | None:
    if env_name not in BANDIT_ENV_NAMES:
        return None

    regret = 0.0
    for rec in trajectory.records:
        arm = rec.action.get("arm")
        if arm is None:
            continue
        probs = rec.private_info.get("step_probs")
        if not probs:
            probs = _fallback_probs(rec, oracle, env_name)
        if not probs:
            continue
        best_prob = max(probs)
        arm_prob = probs[arm] if arm < len(probs) else 0.0
        regret += best_prob - arm_prob
    return regret


def _fallback_probs(
    rec: Any, oracle: dict[str, Any], env_name: str
) -> list[float]:
    if env_name == "ReversalBandit":
        tau = oracle.get("tau", 0)
        return oracle.get("pre_probs", []) if rec.step < tau else oracle.get("post_probs", [])
    if env_name == "DriftingBandit":
        history = oracle.get("prob_history", [])
        idx = max(0, min(rec.step - 1, len(history) - 1))
        return history[idx] if history else oracle.get("probs", [])
    if env_name in ("PhaseLockedBandit", "StagedBandit", "ExploitDecayBandit"):
        history = oracle.get("prob_history", [])
        idx = max(0, min(rec.step - 1, len(history) - 1))
        return history[idx] if history else oracle.get("current_probs", [])
    return oracle.get("current_probs") or oracle.get("probs", [])


def _has_hypothesis_fields(trajectory: Trajectory) -> bool:
    for rec in trajectory.records:
        parsed = rec.parsed_response or {}
        if isinstance(parsed.get("hypothesis"), dict):
            return True
    return False


def _parse_hypothesis(rec: Any) -> tuple[int | None, bool | None, float | None]:
    parsed = rec.parsed_response or {}
    raw_h = parsed.get("hypothesis")
    h = raw_h if isinstance(raw_h, dict) else {}
    best = h.get("best_arm")
    try:
        best_i = int(best) if best is not None else None
    except (TypeError, ValueError):
        best_i = None
    stable = h.get("stable")
    if isinstance(stable, bool):
        stable_b: bool | None = stable
    else:
        stable_b = None
    conf = parsed.get("confidence")
    try:
        conf_f = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf_f = None
    return best_i, stable_b, conf_f


def _true_best_from_record(rec: Any, oracle: dict[str, Any]) -> int | None:
    probs = rec.private_info.get("step_probs") if rec.private_info else None
    if not probs:
        history = oracle.get("prob_history") or []
        idx = max(0, min(int(rec.step), len(history) - 1)) if history else 0
        probs = history[idx] if history else oracle.get("current_probs")
    if probs:
        return max(range(len(probs)), key=lambda i: probs[i])
    params = oracle.get("stage_params") or []
    L = int(oracle.get("stage_length") or 30)
    s = min(int(rec.step) // max(L, 1), max(len(params) - 1, 0))
    if params and "best" in params[s]:
        return int(params[s]["best"])
    return oracle.get("best_arm")


def _first_consecutive_match(flags: list[bool], k: int) -> int | None:
    """1-based start index of first k consecutive True, or None."""
    if k <= 0:
        return None
    run = 0
    for i, ok in enumerate(flags):
        if ok:
            run += 1
            if run >= k:
                return i - k + 2
        else:
            run = 0
    return None


def _hypothesis_loop_metrics(
    trajectory: Trajectory, oracle: dict[str, Any], k_lock: int = 3
) -> dict[str, Any]:
    """Lock / falsify / re-lock and confidence calibration vs oracle best arm."""
    records = trajectory.records
    L = int(oracle.get("stage_length") or 0)
    params = oracle.get("stage_params") or []
    n_stages = int(oracle.get("n_stages") or (len(params) if params else 1))
    if L <= 0:
        L = max(len(records), 1)
        n_stages = 1

    stage_true: list[int | None] = []
    if params:
        stage_true = [p.get("best") for p in params]
    else:
        stage_true = [None] * n_stages

    lock_times: list[int | None] = []
    falsify_delays: list[int | None] = []
    relock_times: list[int | None] = []
    hyp_correct: list[int] = []
    conf_pairs: list[tuple[float, bool]] = []

    for s in range(n_stages):
        start = s * L
        end = min((s + 1) * L, len(records))
        recs = records[start:end]
        true_best = stage_true[s] if s < len(stage_true) else None
        flags: list[bool] = []
        for rec in recs:
            h_best, _stable, conf = _parse_hypothesis(rec)
            tb = true_best
            if tb is None:
                tb = _true_best_from_record(rec, oracle)
            ok = h_best is not None and tb is not None and h_best == tb
            flags.append(ok)
            hyp_correct.append(1 if ok else 0)
            if conf is not None:
                conf_pairs.append((max(0.0, min(1.0, conf)), ok))

        lock = _first_consecutive_match(flags, k_lock)
        lock_times.append(lock)
        if s == 0:
            falsify_delays.append(None)
            relock_times.append(None)
            continue

        old_best = stage_true[s - 1] if s - 1 < len(stage_true) else None
        delay: int | None = None
        for i, rec in enumerate(recs):
            h_best, stable, _conf = _parse_hypothesis(rec)
            left = h_best is not None and old_best is not None and h_best != old_best
            unstable = stable is False
            if left or unstable:
                delay = i + 1
                break
        falsify_delays.append(delay)
        relock_times.append(lock)

    n_hyp_steps = max(len(hyp_correct), 1)
    bins = [0.2, 0.4, 0.6, 0.8, 1.01]
    cal: list[dict[str, float]] = []
    for i, hi in enumerate(bins):
        lo = 0.0 if i == 0 else bins[i - 1]
        group = [(c, ok) for c, ok in conf_pairs if lo <= c < hi]
        if not group:
            continue
        cal.append(
            {
                "lo": lo,
                "hi": min(hi, 1.0),
                "n": float(len(group)),
                "mean_confidence": sum(c for c, _ in group) / len(group),
                "accuracy": sum(1 for _, ok in group if ok) / len(group),
            }
        )

    valid_lock = [x for x in lock_times if x is not None]
    valid_f = [x for x in falsify_delays if x is not None]
    valid_r = [x for x in relock_times if x is not None]
    return {
        "hypothesis_correct_rate": sum(hyp_correct) / n_hyp_steps,
        "lock_time": lock_times,
        "falsification_delay": falsify_delays,
        "relock_time": relock_times,
        "mean_lock_time": sum(valid_lock) / len(valid_lock) if valid_lock else None,
        "mean_falsification_delay": sum(valid_f) / len(valid_f) if valid_f else None,
        "mean_relock_time": sum(valid_r) / len(valid_r) if valid_r else None,
        "mean_confidence": (
            sum(c for c, _ in conf_pairs) / len(conf_pairs) if conf_pairs else None
        ),
        "confidence_calibration": cal,
        "k_lock": k_lock,
    }


def compute_run_metrics(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {}
    rewards = [s["cumulative_reward"] for s in summaries]
    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / max(n - 1, 1)
    return {
        "mean_cumulative_reward": mean,
        "std_cumulative_reward": variance**0.5,
        "n_episodes": n,
        "mean_regret": _safe_mean([s.get("metrics", {}).get("regret") for s in summaries]),
        "mean_invalid_action_rate": _safe_mean(
            [s.get("metrics", {}).get("invalid_action_rate") for s in summaries]
        ),
        "total_input_tokens": sum(s.get("metrics", {}).get("input_tokens", 0) for s in summaries),
        "total_output_tokens": sum(s.get("metrics", {}).get("output_tokens", 0) for s in summaries),
    }


def _safe_mean(values: list[Any]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def load_and_compute_metrics(trajectories_path: Path) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}

    with trajectories_path.open("r", encoding="utf-8") as f:
        episodes: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
        for line in f:
            row = json.loads(line)
            key = (
                row["agent_name"],
                row["env_name"],
                row["seed"],
                row["episode_id"],
            )
            episodes.setdefault(key, []).append(row)

    for key, steps in episodes.items():
        agent_name, env_name, seed, episode_id = key
        cumulative = sum(s["reward"] for s in steps)
        invalid = sum(1 for s in steps if s.get("private_info", {}).get("invalid_action"))
        summary = {
            "agent_name": agent_name,
            "env_name": env_name,
            "seed": seed,
            "episode_id": episode_id,
            "cumulative_reward": cumulative,
            "invalid_action_rate": invalid / max(len(steps), 1),
        }
        summaries.append(summary)
        group_key = (agent_name, env_name)
        by_key.setdefault(group_key, []).append(summary)

    aggregated = {
        f"{agent}:{env}": compute_run_metrics(rows) for (agent, env), rows in by_key.items()
    }
    return {"summaries": summaries, "aggregated": aggregated}