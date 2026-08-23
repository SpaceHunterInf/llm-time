"""Stage-wise lock/falsify for combined n=20 shards."""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else float("nan")


def main() -> None:
    rows = []
    root = Path(__file__).resolve().parents[1]
    for run in sorted(root.glob("runs/*p0_hypothesis_n20_[ab]")):
        path = run / "episode_summaries.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    by = defaultdict(list)
    for r in rows:
        by[r["agent_name"]].append(r)

    header = (
        f"{'agent':22} {'s0_lock':>8} {'s1_fals':>8} {'s1_lock':>8} "
        f"{'s2_fals':>8} {'s2_lock':>8}  lock1  lock2  fals1  fals2"
    )
    print(header)
    for a in ["llm_hyp_raw", "llm_hyp_taught_detect"]:
        s0l, s1f, s1l, s2f, s2l = [], [], [], [], []
        for e in by[a]:
            m = e["metrics"]
            lt = m.get("lock_time") or [None, None, None]
            fd = m.get("falsification_delay") or [None, None, None]
            s0l.append(lt[0])
            s1l.append(lt[1])
            s2l.append(lt[2])
            s1f.append(fd[1])
            s2f.append(fd[2])
        print(
            f"{a:22} {mean(s0l):8.1f} {mean(s1f):8.1f} {mean(s1l):8.1f} "
            f"{mean(s2f):8.1f} {mean(s2l):8.1f}  "
            f"{sum(x is not None for x in s1l):2d}/20  "
            f"{sum(x is not None for x in s2l):2d}/20  "
            f"{sum(x is not None for x in s1f):2d}/20  "
            f"{sum(x is not None for x in s2f):2d}/20"
        )
        print(
            f"{'sd':22} {sd(s0l):8.1f} {sd(s1f):8.1f} {sd(s1l):8.1f} "
            f"{sd(s2f):8.1f} {sd(s2l):8.1f}"
        )

    raw_by = {e["seed"]: e["cumulative_reward"] for e in by["llm_hyp_raw"]}
    td_by = {e["seed"]: e["cumulative_reward"] for e in by["llm_hyp_taught_detect"]}
    d = [td_by[s] - raw_by[s] for s in sorted(raw_by)]
    print()
    print(
        f"paired taught-raw reward: mean={mean(d):.2f} sd={sd(d):.2f} "
        f"se={sd(d) / math.sqrt(len(d)):.2f} taught_wins={sum(x > 0 for x in d)}/{len(d)}"
    )

    raw_f, td_f = [], []
    for e in by["llm_hyp_raw"]:
        fd = e["metrics"].get("falsification_delay") or []
        raw_f.extend(x for x in fd[1:] if x is not None)
    for e in by["llm_hyp_taught_detect"]:
        fd = e["metrics"].get("falsification_delay") or []
        td_f.extend(x for x in fd[1:] if x is not None)
    print(
        f"falsify delay stages 1-2: raw n={len(raw_f)} mean={mean(raw_f):.2f} sd={sd(raw_f):.2f}; "
        f"taught n={len(td_f)} mean={mean(td_f):.2f} sd={sd(td_f):.2f}"
    )

    print()
    for label, name in [("A", "n20_a"), ("B", "n20_b")]:
        matches = list(root.glob(f"runs/*p0_hypothesis_{name}/token_cost.json"))
        if not matches:
            continue
        tot = json.loads(matches[-1].read_text(encoding="utf-8"))["total"]
        print(f"shard {label}: steps={tot['n_llm_steps']} cost=${tot['total_cost_usd']:.2f}")


if __name__ == "__main__":
    main()
