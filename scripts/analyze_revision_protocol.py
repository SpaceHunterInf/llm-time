"""Real cuts vs matched no-change (ghost cuts) for the four P0 prompts."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from llm_rl_engine.analysis.metrics import revision_sequence, stage_starts_from_oracle

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "change": [
        ROOT / "runs/2026-09-02_benchmark_p0_revision_change_ad",
        ROOT / "runs/2026-09-02_benchmark_p0_revision_change_ss",
    ],
    "control": [
        ROOT / "runs/2026-09-02_benchmark_p0_revision_control_ad",
        ROOT / "runs/2026-09-02_benchmark_p0_revision_control_ss",
    ],
}
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


def load_run(run: Path):
    if not (run / "episode_summaries.jsonl").exists():
        return {}
    summ = defaultdict(dict)
    for line in (run / "episode_summaries.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        summ[row["agent_name"]][int(row["seed"])] = row
    buckets = defaultdict(lambda: defaultdict(list))
    with (run / "trajectories.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            buckets[row["agent_name"]][int(row["seed"])].append(row)
    out = {}
    for agent, seeds in buckets.items():
        out[agent] = {}
        for seed, steps in seeds.items():
            steps.sort(key=lambda r: r["step"])
            out[agent][seed] = {
                "summary": summ.get(agent, {}).get(seed, {}),
                "steps": steps,
            }
    return out


def episode_cuts(ep):
    steps = ep["steps"]
    oracle = (ep.get("summary") or {}).get("oracle") or {}
    params = oracle.get("stage_params") or []
    starts = stage_starts_from_oracle(oracle, len(steps))
    bests = [p.get("best") for p in params]
    seqs = []
    for s in range(1, len(starts)):
        start = starts[s]
        end = starts[s + 1] if s + 1 < len(starts) else len(steps)
        window = [r for r in steps if start <= int(r["step"]) < end]
        parsed = [parse_h(r) for r in window]
        h = [p[0] for p in parsed]
        st = [p[1] for p in parsed]
        act = [p[2] for p in parsed]
        old = bests[s - 1] if s - 1 < len(bests) else None
        new = bests[s] if s < len(bests) else None
        seqs.append(revision_sequence(h, st, act, old, new))
    return seqs


def collect(kind: str):
    by_agent = defaultdict(list)
    rewards = defaultdict(list)
    for run in RUNS[kind]:
        data = load_run(run)
        for agent in AGENTS:
            for seed, ep in sorted(data.get(agent, {}).items()):
                by_agent[agent].extend(episode_cuts(ep))
                rewards[agent].append(ep["summary"].get("cumulative_reward"))
    return by_agent, rewards


def row(label, seqs, rew):
    n = len(seqs)
    n_d = sum(1 for s in seqs if s["doubt_latency"] is not None)
    n_r = sum(1 for s in seqs if s["retraction_latency"] is not None)
    n_z = sum(1 for s in seqs if s["resurrected"])
    return (
        f"{label:22} {fmt(mean(rew), 1):>6} "
        f"{n_d:2d}/{n} ({100 * n_d / n:3.0f}%) {fmt(mean([s['doubt_latency'] for s in seqs])):>6} "
        f"{n_r:2d}/{n} ({100 * n_r / n:3.0f}%) {fmt(mean([s['retraction_latency'] for s in seqs])):>7} "
        f"{n_z:2d}/{n} ({100 * n_z / n:3.0f}%)"
    )


def main() -> None:
    change, ch_rew = collect("change")
    control, co_rew = collect("control")
    hdr = (
        f"{'agent':22} {'rew':>6} {'doubt':>14} {'lat':>6} "
        f"{'retract':>14} {'lat':>7} {'resurrect':>14}"
    )
    print("CHANGING (true flips, n=10 seeds × 2 cuts)")
    print(hdr)
    for a in AGENTS:
        print(row(LABELS[a], change[a], ch_rew[a]))
    print()
    print("NO-CHANGE CONTROL (ghost cuts; doubt = false revision)")
    print(hdr)
    for a in AGENTS:
        print(row(LABELS[a], control[a], co_rew[a]))


if __name__ == "__main__":
    main()
