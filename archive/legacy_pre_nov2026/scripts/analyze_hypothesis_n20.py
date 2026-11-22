"""Summarize P0 vanilla-raw vs taught-detect at n=20."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(x, digits=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a"
    return f"{x:.{digits}f}"


def flatten(lists):
    out = []
    for xs in lists:
        if xs is None:
            continue
        out.extend(xs)
    return out


def main() -> None:
    runs = sorted(
        p
        for p in Path(__file__).resolve().parents[1].glob("runs/*p0_hypothesis_n20*")
        if (p / "episode_summaries.jsonl").exists()
    )
    if not runs:
        print("no run")
        return
    print("runs:")
    for run in runs:
        print(" ", run)
    rows = []
    for run in runs:
        rows.extend(
            json.loads(l)
            for l in (run / "episode_summaries.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        )
    by = defaultdict(list)
    for r in rows:
        by[r["agent_name"]].append(r)

    agents = [
        "oracle",
        "llm_hyp_taught_detect",
        "llm_hyp_raw",
        "ucb",
        "sliding_window_ucb",
        "greedy",
        "random",
    ]
    print(
        f"{'agent':24} {'n':>3} {'rew':>10} {'reg':>10} {'h%':>5} "
        f"{'lock':>6} {'fals':>6} {'relo':>6} {'conf':>5} {'relock%':>8}"
    )
    for a in agents:
        if a not in by:
            continue
        eps = by[a]
        rew = [e["cumulative_reward"] for e in eps]
        reg = [e["metrics"].get("regret") for e in eps]
        h = [e["metrics"].get("hypothesis_correct_rate") for e in eps]
        locks = flatten(e["metrics"].get("lock_time") for e in eps)
        fals = flatten(e["metrics"].get("falsification_delay") for e in eps)
        relock = flatten(e["metrics"].get("relock_time") for e in eps)
        n_relock_slots = sum(
            1 for e in eps for x in (e["metrics"].get("relock_time") or [])
        )
        n_relocked = sum(
            1
            for e in eps
            for x in (e["metrics"].get("relock_time") or [])
            if x is not None
        )
        relock_pct = 100.0 * n_relocked / n_relock_slots if n_relock_slots else float("nan")
        h_mean = mean(h)
        print(
            f"{a:24} {len(eps):3d} "
            f"{fmt(mean(rew)):>5}±{fmt(std(rew)):<4} "
            f"{fmt(mean(reg)):>5}±{fmt(std(reg)):<4} "
            f"{(h_mean * 100 if h_mean == h_mean else float('nan')):4.0f}% "
            f"{fmt(mean(locks)):6} "
            f"{fmt(mean(fals)):6} "
            f"{fmt(mean(relock)):6} "
            f"{fmt(mean([e['metrics'].get('mean_confidence') for e in eps]), 2):5} "
            f"{fmt(relock_pct, 0):>7}%"
        )

    print("\nPer-seed lock / falsify / relock (LLM):")
    for a in ["llm_hyp_raw", "llm_hyp_taught_detect"]:
        if a not in by:
            continue
        print(f"  {a}")
        for e in sorted(by[a], key=lambda x: x["seed"]):
            m = e["metrics"]
            h = m.get("hypothesis_correct_rate")
            h_s = f"{h:.2f}" if h is not None else "n/a"
            print(
                f"    s{e['seed']:02d}: R={e['cumulative_reward']:.0f} "
                f"h={h_s} "
                f"lock={m.get('lock_time')} fals={m.get('falsification_delay')} "
                f"relock={m.get('relock_time')}"
            )


if __name__ == "__main__":
    main()
