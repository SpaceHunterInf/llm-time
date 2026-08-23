from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ENV_DESCRIPTIONS = {
    "NumberGuess": "Guess a hidden integer with higher/lower feedback.",
    "StationaryBandit": "4-arm bandit with fixed reward probabilities.",
    "ReversalBandit": "Bandit where the best arm flips at a hidden time.",
    "PeriodicBandit": "Bandit with sinusoidal reward probabilities over time.",
    "DriftingBandit": "Bandit where probabilities drift via a random walk.",
    "PendulumGate": "Time a safe crossing using continuous pendulum position/velocity observations.",
    "TrafficLightGate": "Cross on green while a hidden cyclic traffic light runs.",
    "PhaseLockedBandit": "Bandit arms profitable only during narrow rotating phase windows.",
    "StagedBandit": "Bandit with hard-cut stage changes across reward families.",
    "ExploitDecayBandit": "Exploiting the current best arm degrades it until a new alpha spawns.",
}


def generate_run_plots(
    run_dir: Path,
    run_metrics: dict[str, Any] | None = None,
    token_report: dict[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    trajectories_path = run_dir / "trajectories.jsonl"
    if not trajectories_path.exists():
        return plots_dir

    if run_metrics is None:
        run_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if token_report is None and (run_dir / "token_cost.json").exists():
        token_report = json.loads((run_dir / "token_cost.json").read_text(encoding="utf-8"))

    episodes = _load_episodes(trajectories_path)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        _plot_reward_bars(episodes, run_metrics, plots_dir, plt, np)
        _plot_cumulative_reward_curves(episodes, plots_dir, plt, np)
        _plot_multi_agent_reward_vs_time(episodes, plots_dir, plt, np)
        _plot_arm_choices(episodes, plots_dir, plt, np)
        _plot_staged_reward_vs_time(episodes, plots_dir, plt, np)
        _plot_hypothesis_loop(episodes, plots_dir, plt, np)
        if token_report:
            _plot_token_usage(token_report, plots_dir, plt)
    except ImportError:
        pass

    _write_dashboard_html(run_dir, plots_dir, run_metrics, token_report, episodes)
    return plots_dir


def _load_episodes(path: Path) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    episodes: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["agent_name"], row["env_name"], row["seed"], row["episode_id"])
            episodes.setdefault(key, []).append(row)
    for key in episodes:
        episodes[key].sort(key=lambda r: r["step"])
    return episodes


def _plot_reward_bars(episodes, run_metrics, plots_dir, plt, np) -> None:
    envs = sorted({k[1] for k in episodes})
    for env_name in envs:
        agents = sorted({k[0] for k in episodes if k[1] == env_name})
        means = []
        stds = []
        for agent in agents:
            key = f"{agent}__{env_name}"
            entry = run_metrics.get(key, {})
            means.append(entry.get("mean_cumulative_reward", 0))
            stds.append(entry.get("std_cumulative_reward", 0))

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(agents))
        ax.bar(x, means, yerr=stds, capsize=4, color="#4C78A8", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=25, ha="right")
        ax.set_title(f"Mean cumulative reward — {env_name}")
        ax.set_ylabel("Reward")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"reward_bar_{env_name}.png", dpi=140)
        plt.close(fig)


def _plot_cumulative_reward_curves(episodes, plots_dir, plt, np) -> None:
    grouped: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for (agent, env, _seed, _ep), steps in episodes.items():
        rewards = [s["reward"] for s in steps]
        cumulative = list(np.cumsum(rewards))
        grouped[(agent, env)].append(cumulative)

    for (agent, env), curves in grouped.items():
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]
        mean_curve = np.mean(padded, axis=0)

        fig, ax = plt.subplots(figsize=(8, 4))
        for curve in curves:
            ax.plot(curve, alpha=0.2, color="#9ECAE1")
        ax.plot(mean_curve, color="#E45756", linewidth=2, label="mean across seeds")
        ax.set_title(f"Cumulative reward — {agent} on {env}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative reward")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        safe_name = f"curve_{agent}_{env}".replace(" ", "_")
        fig.savefig(plots_dir / f"{safe_name}.png", dpi=140)
        plt.close(fig)


_AGENT_COLORS = {
    "oracle": "#2ca02c",
    "llm_raw_history": "#d62728",
    "llm_blind": "#d62728",
    "llm_aware": "#e377c2",
    "llm_minimal": "#bcbd22",
    "greedy": "#1f77b4",
    "sliding_window_ucb": "#ff7f0e",
    "ucb": "#9467bd",
    "thompson_sampling": "#8c564b",
    "random": "#7f7f7f",
}


def _plot_multi_agent_reward_vs_time(episodes, plots_dir, plt, np) -> None:
    """Overlay agents: smoothed instantaneous reward + cumulative reward vs time.

    Used for non-staged bandits (ExploitDecay, PhaseLocked, etc.) so LLM vs
    classical comparison is visible in one figure — not only per-agent curves.
    """
    envs = sorted({k[1] for k in episodes})
    # StagedBandit has its own dedicated figure.
    envs = [e for e in envs if e != "StagedBandit"]
    if not envs:
        return

    window = 5
    for env_name in envs:
        by_agent: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
        for (agent, env, _seed, _ep), steps in episodes.items():
            if env == env_name:
                by_agent[agent].append(steps)
        if not by_agent:
            continue

        # Prefer a canonical agent order for legend readability
        preferred = [
            "oracle",
            "llm_raw_history",
            "llm_blind",
            "llm_aware",
            "llm_minimal",
            "sliding_window_ucb",
            "ucb",
            "thompson_sampling",
            "greedy",
            "random",
        ]
        agents = [a for a in preferred if a in by_agent] + sorted(
            a for a in by_agent if a not in preferred
        )

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        ax_inst, ax_cum = axes

        # Oracle expected reward envelope from step_probs (any agent logs them)
        sample_eps = next(iter(by_agent.values()))
        max_len = max(len(ep) for ep in sample_eps)
        orc_mat = []
        for ep in sample_eps:
            orc = [
                max((r.get("private_info") or {}).get("step_probs") or [float("nan")])
                for r in ep
            ]
            if len(orc) < max_len:
                orc = orc + [orc[-1]] * (max_len - len(orc)) if orc else [float("nan")] * max_len
            orc_mat.append(orc)
        mean_orc = np.nanmean(np.array(orc_mat, dtype=float), axis=0)
        ax_inst.plot(
            mean_orc,
            label="oracle E[r*]",
            color="#2ca02c",
            linestyle="--",
            linewidth=1.6,
            alpha=0.9,
        )

        kernel = np.ones(window) / window
        for agent in agents:
            ep_list = by_agent[agent]
            max_len_a = max(len(ep) for ep in ep_list)
            rew_mat = []
            cum_mat = []
            for ep in ep_list:
                rews = [float(r["reward"]) for r in ep]
                if len(rews) < max_len_a:
                    rews = rews + [0.0] * (max_len_a - len(rews))
                rew_mat.append(rews)
                cum_mat.append(list(np.cumsum(rews)))
            mean_rew = np.mean(np.array(rew_mat, dtype=float), axis=0)
            mean_cum = np.mean(np.array(cum_mat, dtype=float), axis=0)
            smooth = np.convolve(mean_rew, kernel, mode="same")
            color = _AGENT_COLORS.get(agent)
            lw = 2.4 if agent.startswith("llm") or agent == "oracle" else 1.8
            ax_inst.plot(smooth, label=f"{agent} (roll-{window})", color=color, linewidth=lw)
            ax_cum.plot(mean_cum, label=agent, color=color, linewidth=lw)

        # ExploitDecay: mark mean alpha-death times if present
        if env_name == "ExploitDecayBandit":
            death_positions: list[int] = []
            for ep in sample_eps:
                for r in ep:
                    if (r.get("private_info") or {}).get("alpha_died"):
                        death_positions.append(int(r.get("step", 0)))
            if death_positions:
                # histogram of death steps → vertical lines at unique common bins
                for t in sorted(set(death_positions)):
                    ax_inst.axvline(t, color="#aaa", linestyle=":", linewidth=0.8, alpha=0.35)
                    ax_cum.axvline(t, color="#aaa", linestyle=":", linewidth=0.8, alpha=0.35)
                ax_inst.plot([], [], color="#aaa", linestyle=":", label="α death (any seed)")

        ax_inst.set_ylabel(f"Reward (rolling mean, w={window})")
        ax_inst.set_title(f"{env_name} — reward vs time (mean over seeds)")
        ax_inst.legend(loc="upper right", fontsize=8, ncol=2)
        ax_inst.grid(alpha=0.3)

        ax_cum.set_xlabel("Step")
        ax_cum.set_ylabel("Cumulative reward")
        ax_cum.set_title(f"{env_name} — cumulative reward vs time")
        ax_cum.legend(loc="upper left", fontsize=8, ncol=2)
        ax_cum.grid(alpha=0.3)

        fig.tight_layout()
        safe = env_name.replace(" ", "_")
        fig.savefig(plots_dir / f"reward_vs_time_{safe}.png", dpi=140)
        plt.close(fig)


def _plot_staged_reward_vs_time(episodes, plots_dir, plt, np) -> None:
    """Primary StagedBandit figure: smoothed reward vs time with stage bands."""
    staged = {
        k: steps
        for k, steps in episodes.items()
        if k[1] == "StagedBandit"
    }
    if not staged:
        return

    # Group by agent
    by_agent: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for (agent, _env, _seed, _ep), steps in staged.items():
        by_agent[agent].append(steps)

    # Collect stage boundaries / families from any episode
    sample = next(iter(staged.values()))
    boundaries = []
    families: list[str] = []
    stage_length = 30
    for row in sample:
        info = row.get("private_info") or {}
        if "stage_end" in info and info["stage_end"] not in boundaries:
            if info.get("stage_index", 0) > 0:
                boundaries.append(info.get("stage_start", 0))
        if info.get("stage_family") and info["stage_family"] not in families:
            # rebuild ordered later
            pass
        if info.get("stage_end"):
            stage_length = max(stage_length, info["stage_end"] // max(info.get("stage_index", 0) + 1, 1))

    # Ordered stage labels from first seed timeline
    stage_labels: list[tuple[int, int, str]] = []
    seen_stages: set[int] = set()
    for row in sample:
        info = row.get("private_info") or {}
        s_idx = info.get("stage_index")
        if s_idx is None or s_idx in seen_stages:
            continue
        seen_stages.add(s_idx)
        start = info.get("stage_start", s_idx * stage_length)
        end = info.get("stage_end", (s_idx + 1) * stage_length)
        fam = info.get("stage_family", f"stage{s_idx}")
        stage_labels.append((start, end, f"S{s_idx}:{fam}"))

    if not stage_labels and sample:
        max_step = max(r["step"] for r in sample) + 1
        n_est = max(1, (max_step + stage_length - 1) // stage_length)
        stage_labels = [
            (s * stage_length, (s + 1) * stage_length, f"S{s}")
            for s in range(n_est)
        ]

    colors = _AGENT_COLORS

    fig, ax = plt.subplots(figsize=(11, 5))
    window = 5

    # Oracle expected-reward envelope from any agent's logged step_probs
    envelope_src = next(iter(by_agent.values()))
    max_len_env = max(len(ep) for ep in envelope_src)
    oracle_mat = []
    for ep in envelope_src:
        orc = [
            max((r.get("private_info") or {}).get("step_probs") or [float("nan")])
            for r in ep
        ]
        if len(orc) < max_len_env:
            orc = orc + [orc[-1]] * (max_len_env - len(orc)) if orc else [float("nan")] * max_len_env
        oracle_mat.append(orc)
    mean_orc = np.nanmean(np.array(oracle_mat, dtype=float), axis=0)
    ax.plot(
        mean_orc,
        label="oracle E[r*]",
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        alpha=0.95,
    )

    for agent, ep_list in sorted(by_agent.items()):
        max_len = max(len(ep) for ep in ep_list)
        reward_mat = []
        for ep in ep_list:
            rews = [r["reward"] for r in ep]
            if len(rews) < max_len:
                rews = rews + [rews[-1]] * (max_len - len(rews)) if rews else [0.0] * max_len
            reward_mat.append(rews)

        mean_rew = np.nanmean(np.array(reward_mat, dtype=float), axis=0)
        kernel = np.ones(window) / window
        smooth = np.convolve(mean_rew, kernel, mode="same")
        color = colors.get(agent, None)
        ax.plot(smooth, label=f"{agent} (roll-{window})", color=color, linewidth=2)

    # Stage bands
    band_colors = ["#e8f4ea", "#e8eef8", "#f8efe8", "#f3e8f8"]
    for i, (start, end, label) in enumerate(stage_labels):
        ax.axvspan(start, end, color=band_colors[i % len(band_colors)], alpha=0.45, zorder=0)
        mid = (start + end) / 2
        ymin, ymax = ax.get_ylim()
        ax.text(mid, ymax * 0.02 + ymin * 0.98, label, ha="center", va="top", fontsize=9)

    for start, _end, _label in stage_labels[1:]:
        ax.axvline(start, color="#333", linestyle=":", linewidth=1, alpha=0.7)

    ax.set_title("StagedBandit — smoothed reward vs time (mean over seeds)")
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Reward (rolling mean, w={window})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "staged_reward_vs_time.png", dpi=140)
    plt.close(fig)

    # Per-agent single-seed detail for LLM if present
    llm_eps = [(k, v) for k, v in staged.items() if k[0] == "llm_raw_history"]
    for (agent, env, seed, ep_id), steps in llm_eps[:3]:
        fig, ax = plt.subplots(figsize=(10, 4))
        rewards = [s["reward"] for s in steps]
        expected = [
            max((s.get("private_info") or {}).get("step_probs") or [0.0]) for s in steps
        ]
        arms = [s["action"].get("arm") for s in steps]
        t = np.arange(len(steps))
        ax.plot(t, expected, "g--", label="oracle E[r*]", alpha=0.8)
        if len(rewards) >= window:
            smooth = np.convolve(rewards, np.ones(window) / window, mode="same")
            ax.plot(t, smooth, "r-", label=f"LLM roll-{window}")
        ax.scatter(t, rewards, c="#d62728", s=12, alpha=0.35, label="LLM reward")
        for i, (start, end, label) in enumerate(stage_labels):
            ax.axvspan(start, min(end, len(steps)), color=band_colors[i % len(band_colors)], alpha=0.35, zorder=0)
            ax.axvline(start, color="#333", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(f"LLM seed {seed} — reward vs time ({env})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"staged_llm_seed{seed}_e{ep_id}.png", dpi=130)
        plt.close(fig)


def _plot_hypothesis_loop(episodes, plots_dir, plt, np) -> None:
    """Hypothesis correctness over time + confidence reliability (LLM only)."""
    hyp_eps: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for (agent, _env, _seed, _ep), steps in episodes.items():
        if any(isinstance((s.get("parsed_response") or {}).get("hypothesis"), dict) for s in steps):
            hyp_eps[agent].append(steps)
    if not hyp_eps:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    ax_h, ax_c = axes
    window = 5
    kernel = np.ones(window) / window

    for agent, ep_list in sorted(hyp_eps.items()):
        max_len = max(len(ep) for ep in ep_list)
        mat = []
        for ep in ep_list:
            flags = []
            for s in ep:
                h = (s.get("parsed_response") or {}).get("hypothesis") or {}
                h_best = h.get("best_arm") if isinstance(h, dict) else None
                probs = (s.get("private_info") or {}).get("step_probs") or []
                true_best = max(range(len(probs)), key=lambda i: probs[i]) if probs else None
                flags.append(1.0 if h_best is not None and true_best is not None and int(h_best) == true_best else 0.0)
            if len(flags) < max_len:
                flags = flags + [flags[-1] if flags else 0.0] * (max_len - len(flags))
            mat.append(flags)
        mean = np.mean(np.array(mat, dtype=float), axis=0)
        smooth = np.convolve(mean, kernel, mode="same")
        color = _AGENT_COLORS.get(agent)
        ax_h.plot(smooth, label=agent, color=color, linewidth=2)

    ax_h.set_ylim(-0.05, 1.05)
    ax_h.set_xlabel("Step")
    ax_h.set_ylabel("P(hypothesis = true best)")
    ax_h.set_title("Hypothesis lock vs oracle best arm (mean over seeds)")
    ax_h.legend(fontsize=8)
    ax_h.grid(alpha=0.3)

    # Reliability: pool all hypothesis steps
    pairs: list[tuple[float, float]] = []
    for ep_list in hyp_eps.values():
        for ep in ep_list:
            for s in ep:
                parsed = s.get("parsed_response") or {}
                h = parsed.get("hypothesis") or {}
                if not isinstance(h, dict):
                    continue
                try:
                    conf = float(parsed.get("confidence"))
                    h_best = int(h.get("best_arm"))
                except (TypeError, ValueError):
                    continue
                probs = (s.get("private_info") or {}).get("step_probs") or []
                if not probs:
                    continue
                true_best = max(range(len(probs)), key=lambda i: probs[i])
                pairs.append((conf, 1.0 if h_best == true_best else 0.0))
    if pairs:
        edges = np.linspace(0, 1, 6)
        xs, ys, ns = [], [], []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            grp = [p for p in pairs if lo <= p[0] < hi or (i == len(edges) - 2 and p[0] == 1.0)]
            if not grp:
                continue
            xs.append(sum(p[0] for p in grp) / len(grp))
            ys.append(sum(p[1] for p in grp) / len(grp))
            ns.append(len(grp))
        ax_c.plot([0, 1], [0, 1], linestyle="--", color="#888", label="perfect calibration")
        ax_c.plot(xs, ys, marker="o", color="#d62728", label="empirical")
        for x, y, n in zip(xs, ys, ns):
            ax_c.annotate(str(n), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax_c.set_xlim(0, 1)
        ax_c.set_ylim(0, 1)
        ax_c.set_xlabel("Reported confidence")
        ax_c.set_ylabel("Hypothesis accuracy")
        ax_c.set_title("Confidence calibration")
        ax_c.legend(fontsize=8)
        ax_c.grid(alpha=0.3)
    else:
        ax_c.set_visible(False)

    fig.tight_layout()
    fig.savefig(plots_dir / "hypothesis_loop.png", dpi=140)
    plt.close(fig)


def _plot_arm_choices(episodes, plots_dir, plt, np) -> None:
    bandit_envs = {
        "StationaryBandit",
        "ReversalBandit",
        "PeriodicBandit",
        "DriftingBandit",
        "PhaseLockedBandit",
        "StagedBandit",
        "ExploitDecayBandit",
    }
    for (agent, env, seed, ep_id), steps in episodes.items():
        if env not in bandit_envs:
            continue
        arms = [s["action"].get("arm") for s in steps if s["action"].get("arm") is not None]
        if not arms:
            continue
        n_arms = max(arms) + 1
        counts = [arms.count(a) for a in range(n_arms)]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(n_arms), counts, color="#72B7B2")
        ax.set_title(f"Arm choices — {agent} / {env} / seed {seed}")
        ax.set_xlabel("Arm")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fname = f"arms_{agent}_{env}_s{seed}_e{ep_id}.png".replace(" ", "_")
        fig.savefig(plots_dir / fname, dpi=120)
        plt.close(fig)


def _plot_token_usage(token_report: dict[str, Any], plots_dir, plt) -> None:
    total = token_report.get("total", {})
    if not total.get("n_llm_steps"):
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    labels = ["Input tokens", "Output tokens"]
    values = [total.get("input_tokens", 0), total.get("output_tokens", 0)]
    axes[0].bar(labels, values, color=["#F58518", "#54A24B"])
    axes[0].set_title("LLM token usage (total run)")
    axes[0].set_ylabel("Tokens")

    cost_labels = ["Input $", "Output $", "Total $"]
    cost_values = [
        total.get("input_cost_usd", 0),
        total.get("output_cost_usd", 0),
        total.get("total_cost_usd", 0),
    ]
    axes[1].bar(cost_labels, cost_values, color=["#B279A2", "#FF9DA6", "#EECA3B"])
    axes[1].set_title(f"Estimated cost ({total.get('model', 'model')})")
    axes[1].set_ylabel("USD")

    fig.tight_layout()
    fig.savefig(plots_dir / "token_usage.png", dpi=140)
    plt.close(fig)


def _write_dashboard_html(
    run_dir: Path,
    plots_dir: Path,
    run_metrics: dict[str, Any],
    token_report: dict[str, Any] | None,
    episodes: dict,
) -> None:
    envs = sorted({k[1] for k in episodes})
    plot_files = sorted(p.name for p in plots_dir.glob("*.png"))

    rows = []
    for key, stats in sorted(run_metrics.items()):
        if key == "token_usage":
            continue
        agent, env = key.split("__", 1)
        rows.append(
            f"<tr><td>{agent}</td><td>{env}</td>"
            f"<td>{stats.get('mean_cumulative_reward', 0):.2f}</td>"
            f"<td>{stats.get('std_cumulative_reward', 0):.2f}</td>"
            f"<td>{stats.get('mean_regret', '—')}</td></tr>"
        )

    token_section = ""
    if token_report and token_report.get("total"):
        t = token_report["total"]
        token_section = f"""
        <h2>Token &amp; cost estimate</h2>
        <ul>
          <li>Model: {t.get('model')}</li>
          <li>LLM steps: {t.get('n_llm_steps')}</li>
          <li>Input tokens: {t.get('input_tokens'):,}</li>
          <li>Output tokens: {t.get('output_tokens'):,}</li>
          <li>Estimated cost: ${t.get('total_cost_usd', 0):.4f} USD</li>
        </ul>
        """

    env_cards = "".join(
        f"<div class='card'><h3>{e}</h3><p>{ENV_DESCRIPTIONS.get(e, '')}</p></div>"
        for e in envs
    )
    plot_imgs = "".join(
        f"<div class='plot'><img src='plots/{p}' alt='{p}'/><p>{p}</p></div>"
        for p in plot_files
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>LLM RL Benchmark Dashboard</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f7f7fb; color: #222; }}
h1 {{ margin-bottom: 0.2rem; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }}
.card {{ background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 4px #0001; max-width: 280px; }}
table {{ border-collapse: collapse; background: white; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
th {{ background: #eef2ff; }}
.plots {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
.plot {{ background: white; padding: 0.5rem; border-radius: 8px; max-width: 420px; }}
.plot img {{ max-width: 100%; height: auto; }}
</style></head><body>
<h1>Benchmark Dashboard</h1>
<p>Run directory: <code>{run_dir.name}</code></p>
<h2>Environments in this run</h2>
<div class="cards">{env_cards}</div>
{token_section}
<h2>Results summary</h2>
<table>
<tr><th>Agent</th><th>Environment</th><th>Mean reward</th><th>Std</th><th>Mean regret</th></tr>
{''.join(rows)}
</table>
<h2>Plots</h2>
<div class="plots">{plot_imgs or '<p>No plots generated (install matplotlib).</p>'}</div>
</body></html>"""

    (run_dir / "dashboard.html").write_text(html, encoding="utf-8")