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
        _plot_arm_choices(episodes, plots_dir, plt, np)
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


def _plot_arm_choices(episodes, plots_dir, plt, np) -> None:
    bandit_envs = {
        "StationaryBandit", "ReversalBandit", "PeriodicBandit", "DriftingBandit"
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