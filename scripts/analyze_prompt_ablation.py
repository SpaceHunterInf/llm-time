"""Analyze StagedBandit P0 prompt ablation (blind / aware / minimal)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/2026-07-31_benchmark_staged_p0_prompt_ablation"
    rows = load_jsonl(run / "episode_summaries.jsonl")
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["agent_name"]].append(r)

    agents = [
        "oracle",
        "llm_aware",
        "llm_blind",
        "llm_minimal",
        "ucb",
        "sliding_window_ucb",
        "thompson_sampling",
        "greedy",
        "random",
    ]
    print("P0 prompt ablation (3x constant, H=90, seeds=0..4)")
    print("Task text only differs for llm_*; dynamics identical.\n")
    print(
        f"{'agent':22} {'reward':>7} {'regret':>7} "
        f"{'R0':>5} {'R1':>5} {'R2':>5} "
        f"{'oc1':>5} {'oc2':>5} {'rec1':>8} {'rec2':>8}"
    )
    for a in agents:
        if a not in by:
            continue
        eps = by[a]
        rew = [e["cumulative_reward"] for e in eps]
        reg = [e["metrics"]["regret"] for e in eps]
        sr = [[e["metrics"]["stage_rewards"][s] for e in eps] for s in range(3)]

        def oc(s: int) -> str:
            vals = [
                e["metrics"]["overcommit_rate"][s]
                for e in eps
                if e["metrics"]["overcommit_rate"][s] is not None
            ]
            return f"{mean(vals)*100:4.0f}%" if vals else "  n/a"

        def rec(s: int) -> str:
            vals = [e["metrics"]["recovery_delay"][s] for e in eps]
            got = [v for v in vals if v is not None]
            if not got:
                return f"0/{len(eps)}"
            return f"{len(got)}/{len(eps)}@{mean(got):.0f}"

        print(
            f"{a:22} {mean(rew):7.1f} {mean(reg):7.1f} "
            f"{mean(sr[0]):5.1f} {mean(sr[1]):5.1f} {mean(sr[2]):5.1f} "
            f"{oc(1)} {oc(2)} {rec(1):>8} {rec(2):>8}"
        )

    print("\nPer-seed LLM rewards:")
    for a in ("llm_aware", "llm_blind", "llm_minimal"):
        eps = sorted(by[a], key=lambda e: e["seed"])
        rewards = [e["cumulative_reward"] for e in eps]
        print(f"  {a:14} {rewards}  mean={mean(rewards):.1f}")
        for e in eps:
            m = e["metrics"]
            print(
                f"    seed {e['seed']}: stages={m['stage_rewards']} "
                f"rec={m['recovery_delay']} oc={m['overcommit_rate']}"
            )

    print("\nDeltas (vs llm_aware):")
    aware = mean([e["cumulative_reward"] for e in by["llm_aware"]])
    for a in ("llm_blind", "llm_minimal", "ucb"):
        r = mean([e["cumulative_reward"] for e in by[a]])
        print(f"  {a:14} {r:.1f}  ({r - aware:+.1f} vs aware)")

    # Pairwise seed diffs blind - aware
    print("\nSeed-wise blind - aware:")
    ba = {e["seed"]: e["cumulative_reward"] for e in by["llm_blind"]}
    aa = {e["seed"]: e["cumulative_reward"] for e in by["llm_aware"]}
    diffs = []
    for s in sorted(ba):
        d = ba[s] - aa[s]
        diffs.append(d)
        print(f"  seed {s}: blind={ba[s]:.0f} aware={aa[s]:.0f}  delta={d:+.0f}")
    print(f"  mean delta blind-aware: {mean(diffs):+.1f}")


if __name__ == "__main__":
    main()
