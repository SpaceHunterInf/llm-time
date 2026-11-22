"""Summarize Phase 0 hypothesis-loop P0 smoke."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    runs = sorted(Path(__file__).resolve().parents[1].glob("runs/*p0_hypothesis*"))
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

    print(f"{'agent':22} {'rew':>6} {'reg':>6} {'h_acc':>6} {'lock':>12} {'fals':>12} {'relo':>12}")
    for a in ["oracle", "llm_hypothesis", "ucb", "sliding_window_ucb", "greedy", "random"]:
        if a not in by:
            continue
        eps = by[a]
        rew = [e["cumulative_reward"] for e in eps]
        reg = [e["metrics"].get("regret") or 0 for e in eps]
        h = [e["metrics"].get("hypothesis_correct_rate") for e in eps]
        h = [x for x in h if x is not None]
        locks = [e["metrics"].get("mean_lock_time") for e in eps]
        fals = [e["metrics"].get("mean_falsification_delay") for e in eps]
        relo = [e["metrics"].get("mean_relock_time") for e in eps]
        def fmt(xs):
            v = [x for x in xs if x is not None]
            return f"{mean(v):.1f}" if v else "n/a"
        print(
            f"{a:22} {mean(rew):6.1f} {mean(reg):6.1f} "
            f"{(mean(h)*100 if h else float('nan')):5.0f}% "
            f"{fmt(locks):>12} {fmt(fals):>12} {fmt(relo):>12}"
        )

    print("\nLLM per seed:")
    for e in sorted(by.get("llm_hypothesis", []), key=lambda x: x["seed"]):
        m = e["metrics"]
        print(
            f"  s{e['seed']}: R={e['cumulative_reward']:.0f} "
            f"h_acc={m.get('hypothesis_correct_rate')} "
            f"lock={m.get('lock_time')} fals={m.get('falsification_delay')} "
            f"relock={m.get('relock_time')} conf={m.get('mean_confidence')}"
        )
    print("plots:", list((run / "plots").glob("hypothesis*")))


if __name__ == "__main__":
    main()
