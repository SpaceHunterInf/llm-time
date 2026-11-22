"""Summarize prompt ablations across P0, P1, P2."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(run: Path, label: str) -> dict:
    rows = load_jsonl(run / "episode_summaries.jsonl")
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["agent_name"]].append(r)

    out = {"label": label}
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
    print("=" * 72)
    print(label)
    print("=" * 72)
    fam = rows[0]["metrics"].get("stage_families", [])
    print(f"stages={fam}")
    print(
        f"{'agent':20} {'rew':>6} {'reg':>6} {'R0':>5} {'R1':>5} {'R2':>5} "
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
            f"{a:20} {mean(rew):6.1f} {mean(reg):6.1f} "
            f"{mean(sr[0]):5.1f} {mean(sr[1]):5.1f} {mean(sr[2]):5.1f} "
            f"{oc(1)} {oc(2)} {rec(1):>8} {rec(2):>8}"
        )
        out[a] = {
            "reward": mean(rew),
            "regret": mean(reg),
            "stage": [mean(sr[s]) for s in range(3)],
        }
    print()
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = [
        ("runs/2026-07-31_benchmark_staged_p0_prompt_ablation", "P0: 3x constant"),
        ("runs/2026-07-31_benchmark_staged_p1_prompt_ablation", "P1: sin→const→linear"),
        ("runs/2026-07-31_benchmark_staged_p2_prompt_ablation", "P2: 3x sinusoidal"),
    ]
    all_sum = []
    for rel, label in runs:
        path = root / rel
        if not path.exists():
            print(f"MISSING {path}")
            continue
        all_sum.append(summarize(path, label))

    print("=" * 72)
    print("SYSTEMATIC MATRIX: mean reward by (task x prompt)")
    print("=" * 72)
    print(f"{'task':22} {'aware':>8} {'blind':>8} {'minimal':>8} {'ucb':>8} {'sw-ucb':>8} {'oracle':>8}")
    for s in all_sum:
        print(
            f"{s['label']:22} "
            f"{s.get('llm_aware', {}).get('reward', float('nan')):8.1f} "
            f"{s.get('llm_blind', {}).get('reward', float('nan')):8.1f} "
            f"{s.get('llm_minimal', {}).get('reward', float('nan')):8.1f} "
            f"{s.get('ucb', {}).get('reward', float('nan')):8.1f} "
            f"{s.get('sliding_window_ucb', {}).get('reward', float('nan')):8.1f} "
            f"{s.get('oracle', {}).get('reward', float('nan')):8.1f}"
        )

    print()
    print("Delta (blind - aware) and (minimal - aware):")
    for s in all_sum:
        a = s.get("llm_aware", {}).get("reward")
        b = s.get("llm_blind", {}).get("reward")
        m = s.get("llm_minimal", {}).get("reward")
        u = s.get("ucb", {}).get("reward")
        sw = s.get("sliding_window_ucb", {}).get("reward")
        if a is None:
            continue
        best_c = max(x for x in (u, sw) if x is not None)
        print(
            f"  {s['label']:22} blind-aware={b-a:+.1f}  min-aware={m-a:+.1f}  "
            f"bestLLM-bestClass={max(a,b,m)-best_c:+.1f}"
        )


if __name__ == "__main__":
    main()
