"""Compare vanilla-raw, taught-detect, and taught-surprise on P0 (n=20)."""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLM_AGENTS = ["llm_hyp_raw", "llm_hyp_taught_detect", "llm_hyp_taught_surprise"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return float("nan")
    return statistics.stdev(xs)


def fmt(x, digits=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a"
    return f"{x:.{digits}f}"


def _load(pattern: str, keep_agents: set[str] | None = None) -> list[dict]:
    rows = []
    for run in sorted(ROOT.glob(pattern)):
        path = run / "episode_summaries.jsonl"
        if not path.exists():
            continue
        print(" ", run)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if keep_agents is None or row["agent_name"] in keep_agents:
                rows.append(row)
    return rows


def main() -> None:
    print("runs:")
    rows = []
    rows.extend(
        _load(
            "runs/*p0_hypothesis_n20_[ab]",
            {
                "llm_hyp_raw",
                "llm_hyp_taught_detect",
                "oracle",
                "ucb",
                "sliding_window_ucb",
                "greedy",
                "random",
            },
        )
    )
    rows.extend(_load("runs/*p0_hypothesis_surprise*", {"llm_hyp_taught_surprise"}))
    if not rows:
        print("no run")
        return

    by = defaultdict(list)
    for r in rows:
        by[r["agent_name"]].append(r)

    agents = [
        "oracle",
        "llm_hyp_taught_surprise",
        "llm_hyp_taught_detect",
        "llm_hyp_raw",
        "ucb",
        "sliding_window_ucb",
        "greedy",
        "random",
    ]
    print()
    print(
        f"{'agent':24} {'n':>3} {'rew':>10} {'reg':>10} {'h%':>5} "
        f"{'s0L':>5} {'fals':>5} {'s1L':>5} {'s2L':>5} {'conf':>5}"
    )
    for a in agents:
        if a not in by:
            continue
        eps = by[a]
        rew = [e["cumulative_reward"] for e in eps]
        reg = [e["metrics"].get("regret") for e in eps]
        h = [e["metrics"].get("hypothesis_correct_rate") for e in eps]
        s0l, s1l, s2l, fals = [], [], [], []
        for e in eps:
            lt = e["metrics"].get("lock_time") or []
            fd = e["metrics"].get("falsification_delay") or []
            if lt:
                s0l.append(lt[0] if len(lt) > 0 else None)
                s1l.append(lt[1] if len(lt) > 1 else None)
                s2l.append(lt[2] if len(lt) > 2 else None)
            fals.extend(x for x in fd[1:] if x is not None)
        h_mean = mean(h)
        print(
            f"{a:24} {len(eps):3d} "
            f"{fmt(mean(rew)):>5}±{fmt(std(rew)):<4} "
            f"{fmt(mean(reg)):>5}±{fmt(std(reg)):<4} "
            f"{(h_mean * 100 if h_mean == h_mean else float('nan')):4.0f}% "
            f"{fmt(mean(s0l)):5} "
            f"{fmt(mean(fals)):5} "
            f"{fmt(mean(s1l)):5} "
            f"{fmt(mean(s2l)):5} "
            f"{fmt(mean([e['metrics'].get('mean_confidence') for e in eps]), 2):5}"
        )

    print("\nFalsify / re-lock coverage (stages 1–2):")
    for a in LLM_AGENTS:
        if a not in by:
            continue
        n_f = n_f_ok = n_l1 = n_l1_ok = n_l2 = n_l2_ok = 0
        for e in by[a]:
            fd = e["metrics"].get("falsification_delay") or []
            lt = e["metrics"].get("lock_time") or []
            for x in fd[1:]:
                n_f += 1
                if x is not None:
                    n_f_ok += 1
            if len(lt) > 1:
                n_l1 += 1
                if lt[1] is not None:
                    n_l1_ok += 1
            if len(lt) > 2:
                n_l2 += 1
                if lt[2] is not None:
                    n_l2_ok += 1
        print(f"  {a:24} fals {n_f_ok}/{n_f}  lock1 {n_l1_ok}/{n_l1}  lock2 {n_l2_ok}/{n_l2}")

    if "llm_hyp_raw" in by and "llm_hyp_taught_surprise" in by:
        raw = {e["seed"]: e["cumulative_reward"] for e in by["llm_hyp_raw"]}
        for other in ["llm_hyp_taught_detect", "llm_hyp_taught_surprise"]:
            if other not in by:
                continue
            o = {e["seed"]: e["cumulative_reward"] for e in by[other]}
            common = sorted(set(raw) & set(o))
            d = [o[s] - raw[s] for s in common]
            print(
                f"\npaired {other} - raw: mean={mean(d):.2f} sd={std(d):.2f} "
                f"se={std(d) / math.sqrt(len(d)):.2f} wins={sum(x > 0 for x in d)}/{len(d)}"
            )

    print("\nPer-seed (LLM):")
    for a in LLM_AGENTS:
        if a not in by:
            continue
        print(f"  {a}")
        for e in sorted(by[a], key=lambda x: x["seed"]):
            m = e["metrics"]
            h = m.get("hypothesis_correct_rate")
            h_s = f"{h:.2f}" if h is not None else "n/a"
            print(
                f"    s{e['seed']:02d}: R={e['cumulative_reward']:.0f} "
                f"h={h_s} lock={m.get('lock_time')} "
                f"fals={m.get('falsification_delay')} relock={m.get('relock_time')}"
            )


if __name__ == "__main__":
    main()
