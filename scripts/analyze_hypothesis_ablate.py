"""Summarize P0 hypothesis interface + taught-detect ablation."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    runs = sorted(Path(__file__).resolve().parents[1].glob("runs/*p0_hypothesis_ablate*"))
    if not runs:
        print("no run")
        return
    run = runs[-1]
    print("run:", run)
    rows = [
        json.loads(l)
        for l in (run / "episode_summaries.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    by = defaultdict(list)
    for r in rows:
        by[r["agent_name"]].append(r)

    agents = [
        "oracle",
        "llm_hyp_raw",
        "llm_hyp_stats",
        "llm_hyp_sr",
        "llm_hyp_taught_detect",
        "ucb",
        "sliding_window_ucb",
        "greedy",
        "random",
    ]
    print(
        f"{'agent':24} {'rew':>6} {'reg':>6} {'h%':>5} {'lock':>6} {'fals':>6} {'relo':>6} {'conf':>5}"
    )
    for a in agents:
        if a not in by:
            continue
        eps = by[a]
        rew = [e["cumulative_reward"] for e in eps]
        reg = [e["metrics"].get("regret") for e in eps]
        h = [e["metrics"].get("hypothesis_correct_rate") for e in eps]
        print(
            f"{a:24} {mean(rew):6.1f} {mean(reg):6.1f} "
            f"{(mean(h)*100 if any(x is not None for x in h) else float('nan')):4.0f}% "
            f"{mean([e['metrics'].get('mean_lock_time') for e in eps]):6.1f} "
            f"{mean([e['metrics'].get('mean_falsification_delay') for e in eps]):6.1f} "
            f"{mean([e['metrics'].get('mean_relock_time') for e in eps]):6.1f} "
            f"{mean([e['metrics'].get('mean_confidence') for e in eps]):5.2f}"
        )

    print("\nPer-seed lock / falsify / relock (LLM):")
    for a in ["llm_hyp_raw", "llm_hyp_stats", "llm_hyp_sr", "llm_hyp_taught_detect"]:
        print(f"  {a}")
        for e in sorted(by[a], key=lambda x: x["seed"]):
            m = e["metrics"]
            print(
                f"    s{e['seed']}: R={e['cumulative_reward']:.0f} "
                f"h={m.get('hypothesis_correct_rate', 0):.2f} "
                f"lock={m.get('lock_time')} fals={m.get('falsification_delay')} "
                f"relock={m.get('relock_time')}"
            )


if __name__ == "__main__":
    main()
