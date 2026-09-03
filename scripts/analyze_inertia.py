"""Belief inertia: doubt / retract / resurrect vs pre-change length L."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from llm_rl_engine.analysis.metrics import revision_sequence, stage_starts_from_oracle

ROOT = Path(__file__).resolve().parents[1]
AGENTS = [
    "llm_hyp_raw",
    "llm_hyp_taught_surprise",
    "llm_hyp_surprise_stat",
    "llm_hyp_taught_detect",
]
LABELS = {
    "llm_hyp_raw": "Vanilla raw",
    "llm_hyp_taught_surprise": "Taught surprise",
    "llm_hyp_surprise_stat": "Surprise + p-value",
    "llm_hyp_taught_detect": "Taught detect",
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x, d=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return " n/a"
    return f"{x:.{d}f}"


def parse_h(step):
    pr = step.get("parsed_response") or {}
    h = pr.get("hypothesis") if isinstance(pr.get("hypothesis"), dict) else {}
    try:
        best = int(h["best_arm"]) if h.get("best_arm") is not None else None
    except (TypeError, ValueError):
        best = None
    st = h.get("stable") if isinstance(h.get("stable"), bool) else None
    try:
        act = int(step.get("action", {}).get("arm"))
    except (TypeError, ValueError):
        act = None
    return best, st, act


def first_policy_leave(actions, old_best):
    if old_best is None:
        return None
    for i, a in enumerate(actions):
        if a is not None and a != old_best:
            return i + 1
    return None


def load_L(L: int):
    runs = sorted(ROOT.glob(f"runs/*p0_inertia_L{L}_*"))
    by = defaultdict(list)
    rew = defaultdict(list)
    seen: dict[tuple[str, int], dict] = {}
    for run in runs:
        summ_path = run / "episode_summaries.jsonl"
        traj_path = run / "trajectories.jsonl"
        if not summ_path.exists() or not traj_path.exists():
            continue
        oracles = {}
        rewards = {}
        for line in summ_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["agent_name"], int(row["seed"]))
            oracles[key] = row.get("oracle") or {}
            rewards[key] = row.get("cumulative_reward")
        buckets = defaultdict(list)
        with traj_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                buckets[(row["agent_name"], int(row["seed"]))].append(row)
        for (agent, seed), steps in buckets.items():
            if agent not in AGENTS:
                continue
            steps.sort(key=lambda r: r["step"])
            oracle = oracles.get((agent, seed), {})
            params = oracle.get("stage_params") or []
            starts = stage_starts_from_oracle(oracle, len(steps))
            if len(starts) < 2:
                continue
            start, end = starts[1], len(steps)
            window = [r for r in steps if start <= int(r["step"]) < end]
            parsed = [parse_h(r) for r in window]
            seq = revision_sequence(
                [p[0] for p in parsed],
                [p[1] for p in parsed],
                [p[2] for p in parsed],
                params[0].get("best") if params else None,
                params[1].get("best") if len(params) > 1 else None,
            )
            seq["policy_leave"] = first_policy_leave(
                [p[2] for p in parsed],
                params[0].get("best") if params else None,
            )
            seen[(agent, seed)] = {"seq": seq, "rew": rewards.get((agent, seed))}
    for (agent, _seed), item in seen.items():
        by[agent].append(item["seq"])
        rew[agent].append(item["rew"])
    return by, rew


def main() -> None:
    print(
        f"{'L':>4} {'agent':22} {'n':>3} {'doubt':>6} {'retract':>8} "
        f"{'policy':>7} {'res%':>6} {'rew':>6}"
    )
    for L in (10, 30, 60):
        by, rew = load_L(L)
        if not by:
            print(f"{L:4d}  (no run)")
            continue
        for agent in AGENTS:
            seqs = by.get(agent, [])
            if not seqs:
                continue
            print(
                f"{L:4d} {LABELS[agent]:22} {len(seqs):3d} "
                f"{fmt(mean([s['doubt_latency'] for s in seqs])):>6} "
                f"{fmt(mean([s['retraction_latency'] for s in seqs])):>8} "
                f"{fmt(mean([s['policy_leave'] for s in seqs])):>7} "
                f"{fmt(100 * mean([1.0 if s['resurrected'] else 0.0 for s in seqs]), 0):>5}% "
                f"{fmt(mean(rew[agent]), 1):>6}"
            )


if __name__ == "__main__":
    main()
