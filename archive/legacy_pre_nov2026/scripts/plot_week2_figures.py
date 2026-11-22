"""Figures for the Sept 2026 revision / inertia / ExploitDecay note."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "week2_2026-09"
LABELS = ["Vanilla", "Surprise", "Surprise+p", "Detect"]
COLORS = ["#d62728", "#9467bd", "#ff7f0e", "#1f77b4"]


def style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
        }
    )


def grouped_bars(path, title, ylabel, series, ylim=None, legend_ncol=2):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(LABELS))
    n = len(series)
    width = 0.8 / n
    for i, (name, vals, color) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=name, color=color, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(ncol=legend_ncol)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()

    grouped_bars(
        OUT / "fig01_change_vs_control.png",
        "Reward on changing P0 vs matched no-change (n = 10)",
        "Cumulative reward",
        [
            ("World flips", [56.6, 59.2, 60.4, 60.9], "#4C78A8"),
            ("No change (control)", [72.3, 71.1, 70.4, 65.7], "#F58518"),
        ],
        ylim=(0, 85),
    )

    grouped_bars(
        OUT / "fig02_doubt_retract_n20.png",
        "P0 n = 20: doubt is fast; retract is slow (40 cuts)",
        "Steps after the cut",
        [
            ("Doubt", [6.7, 3.6, 4.5, 3.2], "#54A24B"),
            ("Retract", [11.3, 12.1, 11.0, 9.5], "#E45756"),
        ],
        ylim=(0, 16),
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    rates = [0, 20, 30, 55]
    ax.bar(LABELS, rates, color=COLORS, alpha=0.9)
    ax.set_ylabel("False doubt at ghost cuts (%)")
    ax.set_title("No-change control: teaching buys false alarms (n = 10, 20 ghost cuts)")
    ax.set_ylim(0, 70)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig03_false_doubt.png", dpi=160)
    plt.close(fig)

    L = np.array([10, 30, 60])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.plot(L, [7.7, 10.5, 10.6], "o-", color="#d62728", label="Vanilla doubt")
    ax.plot(L, [9.1, 17.1, 17.9], "s-", color="#d62728", alpha=0.6, label="Vanilla retract")
    ax.plot(L, [2.6, 4.2, 2.9], "o-", color="#1f77b4", label="Detect doubt")
    ax.plot(L, [7.8, 14.8, 10.6], "s-", color="#1f77b4", alpha=0.6, label="Detect retract")
    ax.set_xlabel("Pre-change length L")
    ax.set_ylabel("Steps after the flip")
    ax.set_title("Inertia: delays vs how long H was true")
    ax.set_xticks(L)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(L, [10, 30, 50], "o-", color="#d62728", label="Vanilla")
    ax.plot(L, [40, 50, 40], "o-", color="#9467bd", label="Surprise")
    ax.plot(L, [40, 90, 50], "o-", color="#ff7f0e", label="Surprise+p")
    ax.plot(L, [60, 90, 50], "o-", color="#1f77b4", label="Detect")
    ax.set_xlabel("Pre-change length L")
    ax.set_ylabel("Resurrection rate (%)")
    ax.set_title("Inertia: return to the old H")
    ax.set_xticks(L)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig04_inertia.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    names = ["Oracle", "Detect", "Surprise", "Vanilla", "Surprise+p", "SW-UCB"]
    rew = [53.4, 34.2, 32.6, 32.4, 31.4, 29.7]
    om = [10, 31, 50, 44, 60, 18]
    cols = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728", "#ff7f0e", "#c49c94"]
    axes[0].bar(names, rew, color=cols, alpha=0.9)
    axes[0].set_ylabel("Reward (H = 90)")
    axes[0].set_title("ExploitDecay: reward")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(names, om, color=cols, alpha=0.9)
    axes[1].set_ylabel("Overmine after alpha death (%)")
    axes[1].set_title("ExploitDecay: still pulling the dead arm")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig05_exploit_decay.png", dpi=160)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
