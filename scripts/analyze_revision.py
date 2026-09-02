"""Doubt / retract / resurrect / belief–action on existing P0 n=20 traces."""
from __future__ import annotations

import json
import math
import statistics
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


def std(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else float("nan")


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a"
    return f"{x:.{d}f}"


def load_episodes():
    specs = {
        "llm_hyp_raw": "runs/*p0_hypothesis_n20_[ab]",
        "llm_hyp_taught_detect": "runs/*p0_hypothesis_n20_[ab]",
        "llm_hyp_taught_surprise": "runs/*p0_hypothesis_surprise_[ab]",
        "llm_hyp_surprise_stat": "runs/*p0_hypothesis_surprise_stat*",
    }
    out: dict[str, dict[int, dict]] = {a: {} for a in AGENTS}
    for agent, pattern in specs.items():
        for run in sorted(ROOT.glob(pattern)):
            summ = run / "episode_summaries.jsonl"
            traj = run / "trajectories.jsonl"
            if not summ.exists() or not traj.exists():
                continue
            oracles: dict[int, dict] = {}
            for line in summ.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("agent_name") != agent:
                    continue
                oracles[int(row["seed"])] = row.get("oracle") or {}
            buckets: dict[int, list] = defaultdict(list)
            with traj.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("agent_name") != agent:
                        continue
                    buckets[int(row["seed"])].append(row)
            for seed, steps in buckets.items():
                steps.sort(key=lambda r: r["step"])
                out[agent][seed] = {"oracle": oracles.get(seed, {}), "steps": steps}
    return out


def parse_h(step):
    pr = step.get("parsed_response") or {}
    h = pr.get("hypothesis") if isinstance(pr.get("hypothesis"), dict) else {}
    best = h.get("best_arm")
    try:
        best_i = int(best) if best is not None else None
    except (TypeError, ValueError):
        best_i = None
    st = h.get("stable")
    st_b = st if isinstance(st, bool) else None
    try:
        act = int(step.get("action", {}).get("arm"))
    except (TypeError, ValueError):
        act = None
    return best_i, st_b, act


def episode_cuts(ep):
    steps = ep["steps"]
    oracle = ep["oracle"]
    params = oracle.get("stage_params") or []
    starts = stage_starts_from_oracle(oracle, len(steps))
    bests = [p.get("best") for p in params] if params else []
    seqs = []
    for s in range(1, len(starts)):
        start = starts[s]
        end = starts[s + 1] if s + 1 < len(starts) else len(steps)
        window = [r for r in steps if start <= int(r["step"]) < end]
        h, st, act = [], [], []
        for r in window:
            hb, sb, a = parse_h(r)
            h.append(hb)
            st.append(sb)
            act.append(a)
        old = bests[s - 1] if s - 1 < len(bests) else None
        new = bests[s] if s < len(bests) else None
        seqs.append(revision_sequence(h, st, act, old, new, k_persist=3))
    return seqs


def main() -> None:
    print("loading traces…")
    data = load_episodes()
    print(
        f"{'agent':22} {'n':>3} {'doubt':>7} {'retract':>8} {'search':>7} "
        f"{'res%':>6} {'n_res':>6} {'consist':>8} {'pull_old|doubt':>14} "
        f"{'Hnew pull old':>13}"
    )
    by_agent_seqs: dict[str, list] = {}
    for agent in AGENTS:
        seqs = []
        for seed, ep in sorted(data[agent].items()):
            seqs.extend(episode_cuts(ep))
        by_agent_seqs[agent] = seqs
        n_ep = len(data[agent])
        print(
            f"{LABELS[agent]:22} {n_ep:3d} "
            f"{fmt(mean([s['doubt_latency'] for s in seqs]), 1):>7} "
            f"{fmt(mean([s['retraction_latency'] for s in seqs]), 1):>8} "
            f"{fmt(mean([s['search_duration'] for s in seqs]), 1):>7} "
            f"{fmt(100 * mean([1.0 if s['resurrected'] else 0.0 for s in seqs]), 0):>5}% "
            f"{fmt(mean([s['n_resurrections'] for s in seqs]), 2):>6} "
            f"{fmt(mean([s['belief_action_consistency'] for s in seqs]), 2):>8} "
            f"{fmt(mean([s['frac_post_doubt_pull_old'] for s in seqs]), 2):>14} "
            f"{fmt(mean([s['frac_declare_new_pull_old'] for s in seqs]), 2):>13}"
        )

    print("\nPer-cut resurrection / retract missing:")
    for agent in AGENTS:
        seqs = by_agent_seqs[agent]
        n = len(seqs)
        n_res = sum(1 for s in seqs if s["resurrected"])
        n_no_ret = sum(1 for s in seqs if s["retraction_latency"] is None)
        n_doubt = sum(1 for s in seqs if s["doubt_latency"] is not None)
        print(
            f"  {LABELS[agent]:22} cuts={n} doubted={n_doubt} "
            f"never_retract={n_no_ret} resurrected={n_res} "
            f"({100 * n_res / n:.0f}%)"
        )

    print("\nDoubt vs retract (mean over cuts that have both):")
    for agent in AGENTS:
        gaps = [
            s["retraction_latency"] - s["doubt_latency"]
            for s in by_agent_seqs[agent]
            if s["doubt_latency"] is not None and s["retraction_latency"] is not None
        ]
        print(
            f"  {LABELS[agent]:22} retract-doubt={fmt(mean(gaps), 1)} "
            f"n={len(gaps)}"
        )


if __name__ == "__main__":
    main()
