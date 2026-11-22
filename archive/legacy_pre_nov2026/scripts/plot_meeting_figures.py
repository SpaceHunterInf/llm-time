"""Combined n=20 meeting figures. Writes reports/figures/meeting_2026-08-24/."""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "meeting_2026-08-24"

LABELS = {
    "oracle": "Oracle",
    "llm_hyp_taught_detect": "Taught detect",
    "llm_hyp_surprise_stat": "Surprise + p-value",
    "llm_hyp_taught_surprise": "Taught surprise",
    "llm_hyp_raw": "Vanilla (raw)",
    "ucb": "UCB (full history)",
    "sliding_window_ucb": "SW-UCB",
    "greedy": "Greedy",
    "random": "Random",
}
COLORS = {
    "oracle": "#2ca02c",
    "llm_hyp_taught_detect": "#1f77b4",
    "llm_hyp_surprise_stat": "#ff7f0e",
    "llm_hyp_taught_surprise": "#9467bd",
    "llm_hyp_raw": "#d62728",
    "ucb": "#7f7f7f",
    "sliding_window_ucb": "#c49c94",
    "greedy": "#17becf",
    "random": "#bbbbbb",
}
ORDER = [
    "oracle",
    "llm_hyp_taught_detect",
    "llm_hyp_surprise_stat",
    "llm_hyp_taught_surprise",
    "llm_hyp_raw",
    "ucb",
    "sliding_window_ucb",
    "greedy",
    "random",
]


def load_summaries():
    rows = []
    specs = [
        ("runs/*p0_hypothesis_n20_[ab]", None),
        ("runs/*p0_hypothesis_surprise_[ab]", {"llm_hyp_taught_surprise"}),
        ("runs/*p0_hypothesis_surprise_stat*", {"llm_hyp_surprise_stat"}),
    ]
    for pattern, keep in specs:
        for run in sorted(ROOT.glob(pattern)):
            path = run / "episode_summaries.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if keep is None or row["agent_name"] in keep:
                    rows.append(row)
    by = defaultdict(list)
    for r in rows:
        by[r["agent_name"]].append(r)
    return by


def load_hyp_episodes():
    """agent -> list of episode step lists (hypothesis flags, conf, reward)."""
    specs = {
        "llm_hyp_raw": "runs/*p0_hypothesis_n20_[ab]",
        "llm_hyp_taught_detect": "runs/*p0_hypothesis_n20_[ab]",
        "llm_hyp_taught_surprise": "runs/*p0_hypothesis_surprise_[ab]",
        "llm_hyp_surprise_stat": "runs/*p0_hypothesis_surprise_stat*",
    }
    out: dict[str, list[list[dict]]] = defaultdict(list)
    for agent, pattern in specs.items():
        for run in sorted(ROOT.glob(pattern)):
            path = run / "trajectories.jsonl"
            if not path.exists():
                continue
            buckets: dict[int, list] = defaultdict(list)
            with path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("agent_name") != agent:
                        continue
                    buckets[int(row["seed"])].append(row)
            for seed, steps in buckets.items():
                steps.sort(key=lambda r: r["step"])
                out[agent].append(steps)
    return out


def style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def stage_bands(ax, ymax=None):
    for start, color, label in (
        (0, "#e8f5e9", "Stage 0"),
        (30, "#e3f2fd", "Stage 1"),
        (60, "#fff8e1", "Stage 2"),
    ):
        ax.axvspan(start, start + 30, color=color, alpha=0.9, zorder=0)
        ax.text(start + 15, ax.get_ylim()[1] if ymax is None else ymax, label, ha="center", va="top", fontsize=9, color="#555")
    ax.axvline(30, color="#333", linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(60, color="#333", linestyle=":", linewidth=1, alpha=0.5)


def plot_reward_bars(by):
    agents = [a for a in ORDER if a in by]
    means, stds, colors, labels = [], [], [], []
    for a in agents:
        rew = [e["cumulative_reward"] for e in by[a]]
        means.append(sum(rew) / len(rew))
        stds.append(float(np.std(rew, ddof=1)) if len(rew) > 1 else 0.0)
        colors.append(COLORS[a])
        labels.append(LABELS[a])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(agents))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.9, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Cumulative reward (H = 90)")
    ax.set_title("Blind P0, n = 20 — mean ± 1 sd")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig01_reward_bars.png", dpi=160)
    plt.close(fig)


def plot_hypothesis_time(eps):
    fig, ax = plt.subplots(figsize=(11, 5))
    window = 5
    kernel = np.ones(window) / window
    ax.set_ylim(-0.02, 1.12)
    stage_bands(ax, ymax=1.10)
    for agent in [
        "llm_hyp_taught_detect",
        "llm_hyp_surprise_stat",
        "llm_hyp_taught_surprise",
        "llm_hyp_raw",
    ]:
        ep_list = eps[agent]
        mat = []
        for steps in ep_list:
            flags = []
            for s in steps:
                h = (s.get("parsed_response") or {}).get("hypothesis") or {}
                h_best = h.get("best_arm") if isinstance(h, dict) else None
                probs = (s.get("private_info") or {}).get("step_probs") or []
                true_best = max(range(len(probs)), key=lambda i: probs[i]) if probs else None
                ok = h_best is not None and true_best is not None and int(h_best) == true_best
                flags.append(1.0 if ok else 0.0)
            mat.append(flags[:90] + [flags[-1]] * max(0, 90 - len(flags)))
        mean = np.mean(np.array(mat, dtype=float)[:, :90], axis=0)
        smooth = np.convolve(mean, kernel, mode="same")
        ax.plot(smooth, color=COLORS[agent], linewidth=2.2, label=LABELS[agent])
    ax.set_xlabel("Step")
    ax.set_ylabel("P(declared best arm = oracle best)")
    ax.set_title("Does the model’s hypothesis track the true best arm? (n = 20, 5-step smooth)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig02_hypothesis_vs_time.png", dpi=160)
    plt.close(fig)


def plot_falsify(by):
    agents = [
        "llm_hyp_raw",
        "llm_hyp_taught_surprise",
        "llm_hyp_surprise_stat",
        "llm_hyp_taught_detect",
    ]
    data, labels, colors = [], [], []
    for a in agents:
        vals = []
        for e in by[a]:
            fd = e["metrics"].get("falsification_delay") or []
            vals.extend(x for x in fd[1:] if x is not None)
        data.append(vals)
        labels.append(LABELS[a])
        colors.append(COLORS[a])
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Steps after a stage cut until H is dropped")
    ax.set_title("Falsification delay (both cuts, n = 20 × 2 = 40 events)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig03_falsify_delay.png", dpi=160)
    plt.close(fig)


def plot_reward_time(eps, by):
    fig, ax = plt.subplots(figsize=(11, 5))
    window = 5
    kernel = np.ones(window) / window
    ax.set_ylim(0.0, 1.05)
    stage_bands(ax, ymax=1.02)
    # oracle envelope from any LLM
    src = next(iter(eps.values()))
    orc = []
    for steps in src:
        row = [max((s.get("private_info") or {}).get("step_probs") or [np.nan]) for s in steps]
        orc.append(row[:90] + [row[-1]] * max(0, 90 - len(row)))
    ax.plot(np.nanmean(np.array(orc), axis=0)[:90], color=COLORS["oracle"], linestyle="--", linewidth=1.6, label="Oracle E[r*]")
    for agent in [
        "llm_hyp_taught_detect",
        "llm_hyp_surprise_stat",
        "llm_hyp_taught_surprise",
        "llm_hyp_raw",
        "ucb",
        "sliding_window_ucb",
    ]:
        if agent in eps:
            ep_list = eps[agent]
        else:
            continue
        mat = []
        for steps in ep_list:
            r = [s["reward"] for s in steps]
            mat.append(r[:90] + [r[-1]] * max(0, 90 - len(r)))
        mean = np.mean(np.array(mat, dtype=float)[:, :90], axis=0)
        smooth = np.convolve(mean, kernel, mode="same")
        ax.plot(smooth, color=COLORS[agent], linewidth=2.0, label=LABELS[agent])
    # UCB from summaries-only n20 trajectories
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (5-step rolling mean)")
    ax.set_title("Instantaneous reward on blind P0 (mean over seeds)")
    ax.legend(loc="lower left", ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig04_reward_vs_time.png", dpi=160)
    plt.close(fig)


def load_classical_episodes():
    extra: dict[str, list] = defaultdict(list)
    for run in sorted(ROOT.glob("runs/*p0_hypothesis_n20_[ab]")):
        path = run / "trajectories.jsonl"
        if not path.exists():
            continue
        buckets: dict[tuple[str, int], list] = defaultdict(list)
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                a = row.get("agent_name")
                if a not in ("ucb", "sliding_window_ucb", "oracle"):
                    continue
                buckets[(a, int(row["seed"]))].append(row)
        for (a, _seed), steps in buckets.items():
            steps.sort(key=lambda r: r["step"])
            extra[a].append(steps)
    return extra


def plot_calibration(eps):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888", label="Perfect calibration")
    edges = np.linspace(0, 1, 6)
    for agent in [
        "llm_hyp_raw",
        "llm_hyp_taught_detect",
        "llm_hyp_taught_surprise",
        "llm_hyp_surprise_stat",
    ]:
        pairs = []
        for steps in eps[agent]:
            for s in steps:
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
        xs, ys = [], []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            grp = [p for p in pairs if lo <= p[0] < hi or (i == len(edges) - 2 and p[0] == 1.0)]
            if len(grp) < 20:
                continue
            xs.append(sum(p[0] for p in grp) / len(grp))
            ys.append(sum(p[1] for p in grp) / len(grp))
        ax.plot(xs, ys, marker="o", color=COLORS[agent], linewidth=2, label=LABELS[agent])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Reported confidence")
    ax.set_ylabel("Hypothesis accuracy")
    ax.set_title("Confidence calibration (n = 20)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "fig05_calibration.png", dpi=160)
    plt.close(fig)


def copy_existing():
    copies = [
        (
            ROOT / "runs/2026-07-31_benchmark_staged_constant_p0/plots/staged_reward_vs_time.png",
            OUT / "fig06_p0_action_level.png",
        ),
        (
            ROOT / "runs/2026-08-01_benchmark_exploit_decay/plots/reward_vs_time_ExploitDecayBandit.png",
            OUT / "fig07_exploit_decay.png",
        ),
        (
            ROOT / "runs/2026-07-31_benchmark_staged_sinusoidal_p2/plots/staged_reward_vs_time.png",
            OUT / "fig08_p2_sinusoids.png",
        ),
        (
            ROOT / "runs/2026-07-31_benchmark_staged_p0_long/plots/staged_reward_vs_time.png",
            OUT / "fig09_p0_long.png",
        ),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)
            print("copied", dst.name)
        else:
            print("MISSING", src)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    print("loading summaries…")
    by = load_summaries()
    print("loading trajectories…")
    eps = load_hyp_episodes()
    extra = load_classical_episodes()
    for k, v in extra.items():
        eps[k] = v
    plot_reward_bars(by)
    print("fig01")
    plot_hypothesis_time(eps)
    print("fig02")
    plot_falsify(by)
    print("fig03")
    plot_reward_time(eps, by)
    print("fig04")
    plot_calibration(eps)
    print("fig05")
    copy_existing()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
