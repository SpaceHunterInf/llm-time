"""Analyze P0-long (H=180, 6 constant stages, blind)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    run = Path(__file__).resolve().parents[1] / "runs/2026-07-31_benchmark_staged_p0_long"
    rows = [
        json.loads(line)
        for line in (run / "episode_summaries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["agent_name"]].append(r)

    agents = [
        "oracle",
        "llm_raw_history",
        "ucb",
        "sliding_window_ucb",
        "thompson_sampling",
        "greedy",
        "random",
    ]
    print("P0-long H=180 6x constant blind")
    header = f"{'agent':22} {'rew':>7} {'reg':>7} " + " ".join(f"R{s}" for s in range(6))
    print(header)
    for a in agents:
        eps = by[a]
        rew = [e["cumulative_reward"] for e in eps]
        reg = [e["metrics"]["regret"] for e in eps]
        n_st = len(eps[0]["metrics"]["stage_rewards"])
        sr = [[e["metrics"]["stage_rewards"][s] for e in eps] for s in range(n_st)]
        rs = " ".join(f"{mean(sr[s]):5.1f}" for s in range(n_st))
        print(f"{a:22} {mean(rew):7.1f} {mean(reg):7.1f} {rs}")

    print()
    print("LLM per seed:")
    for e in sorted(by["llm_raw_history"], key=lambda x: x["seed"]):
        m = e["metrics"]
        print(f"  s{e['seed']}: R={e['cumulative_reward']:.0f} stages={m['stage_rewards']}")
        print(f"       rec={m['recovery_delay']}")
        print(f"       oc={m['overcommit_rate']}")

    print()
    for a in ["llm_raw_history", "ucb", "sliding_window_ucb", "oracle"]:
        eps = by[a]
        early = mean([sum(e["metrics"]["stage_rewards"][:3]) for e in eps])
        late = mean([sum(e["metrics"]["stage_rewards"][3:]) for e in eps])
        print(
            f"{a:22} early(s0-2)={early:.1f} late(s3-5)={late:.1f} "
            f"late/early={late / early:.2f}"
        )

    o = mean([e["cumulative_reward"] for e in by["oracle"]])
    l = mean([e["cumulative_reward"] for e in by["llm_raw_history"]])
    u = mean([e["cumulative_reward"] for e in by["ucb"]])
    sw = mean([e["cumulative_reward"] for e in by["sliding_window_ucb"]])
    print()
    print(f"LLM/oracle={l / o * 100:.1f}%  LLM-ucb={l - u:+.1f}  LLM-sw={l - sw:+.1f}")
    # vs short P0 blind (from ablation)
    print("Compare to short P0 blind (H=90, 3 stages): llm=52.0 ucb=37.4 oracle=73.0")
    print(f"  long LLM efficiency {l/o*100:.1f}% vs short blind {52/73*100:.1f}%")
    print(f"  long LLM/oracle residual={o - l:.1f}; per-step reward LLM={l/180:.3f} short={52/90:.3f}")


if __name__ == "__main__":
    main()
