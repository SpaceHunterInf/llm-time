"""One-off analysis of StagedBandit P0/P1 runs."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def analyze(run: Path, label: str) -> None:
    rows = load_jsonl(run / "episode_summaries.jsonl")
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["agent_name"]].append(r)

    fam = rows[0]["metrics"]["stage_families"]
    L = rows[0]["metrics"]["stage_length"]
    print("=" * 72)
    print(label)
    print("=" * 72)
    print(f"Stages: {fam} | L={L} | H=90 | seeds=5")
    print()

    agents = [
        "oracle",
        "llm_raw_history",
        "ucb",
        "sliding_window_ucb",
        "thompson_sampling",
        "greedy",
        "random",
    ]

    header = (
        f"{'agent':22} {'reward':>7} {'regret':>7} "
        f"{'R0':>5} {'R1':>5} {'R2':>5} "
        f"{'oc1':>5} {'oc2':>5} "
        f"{'rec1':>8} {'rec2':>8} "
        f"{'pc1':>5} {'pc2':>5}"
    )
    print(header)

    summary: dict[str, dict] = {}
    for a in agents:
        eps = by[a]
        rew = [e["cumulative_reward"] for e in eps]
        reg = [e["metrics"]["regret"] for e in eps]
        sr = [[e["metrics"]["stage_rewards"][s] for e in eps] for s in range(3)]

        def oc_mean(stage: int) -> str:
            vals = [e["metrics"]["overcommit_rate"][stage] for e in eps]
            vals = [v for v in vals if v is not None]
            return f"{mean(vals)*100:4.0f}%" if vals else "  n/a"

        def rec_str(stage: int) -> str:
            vals = [e["metrics"]["recovery_delay"][stage] for e in eps]
            recovered = [v for v in vals if v is not None]
            n = len(eps)
            if not recovered:
                return f"0/{n}"
            return f"{len(recovered)}/{n}@{mean(recovered):.0f}"

        def pc_mean(stage: int) -> str:
            vals = [e["metrics"]["post_change_mean_reward"][stage] for e in eps]
            vals = [v for v in vals if v is not None]
            return f"{mean(vals):5.2f}" if vals else "  n/a"

        print(
            f"{a:22} {mean(rew):7.1f} {mean(reg):7.1f} "
            f"{mean(sr[0]):5.1f} {mean(sr[1]):5.1f} {mean(sr[2]):5.1f} "
            f"{oc_mean(1)} {oc_mean(2)} "
            f"{rec_str(1):>8} {rec_str(2):>8} "
            f"{pc_mean(1)} {pc_mean(2)}"
        )
        summary[a] = {
            "reward": mean(rew),
            "regret": mean(reg),
            "stage": [mean(sr[s]) for s in range(3)],
            "rewards": rew,
        }

    print()
    print("LLM per-seed detail:")
    for e in sorted(by["llm_raw_history"], key=lambda x: x["seed"]):
        m = e["metrics"]
        print(
            f"  seed {e['seed']}: reward={e['cumulative_reward']:.0f} "
            f"stageR={m['stage_rewards']} "
            f"overcommit={m['overcommit_rate']} "
            f"recovery={m['recovery_delay']} "
            f"post_chg={m['post_change_mean_reward']}"
        )

    print()
    print("Stage-wise mean reward (all agents):")
    for s in range(3):
        scores = {a: mean([e["metrics"]["stage_rewards"][s] for e in by[a]]) for a in agents}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_a, best_v = ranked[0]
        llm_v = scores["llm_raw_history"]
        print(f"  stage {s} ({fam[s]}): best={best_a} ({best_v:.1f}), llm={llm_v:.1f}")
        print(f"    ranking: " + ", ".join(f"{a}={v:.1f}" for a, v in ranked))

    o = summary["oracle"]["reward"]
    l = summary["llm_raw_history"]["reward"]
    classical = [a for a in agents if a not in ("oracle", "llm_raw_history")]
    best_c = max(summary[a]["reward"] for a in classical)
    best_c_name = max(classical, key=lambda a: summary[a]["reward"])
    print()
    print(f"LLM / oracle efficiency: {l/o*100:.1f}%")
    print(f"Best classical ({best_c_name}) / oracle: {best_c/o*100:.1f}%")
    print(f"LLM - best classical: {l-best_c:+.1f} reward ({(l-best_c)/best_c*100:+.1f}%)")
    print(f"Oracle - LLM residual: {o-l:.1f}")
    print(f"LLM reward std across seeds: {statistics.pstdev(summary['llm_raw_history']['rewards']):.2f}")

    # Drop at boundaries: stage0 vs stage1 for agents that did well in stage0
    print()
    print("Boundary drop (stage0_mean - stage1_mean):")
    for a in agents:
        d = summary[a]["stage"][0] - summary[a]["stage"][1]
        print(f"  {a:22} {d:+.1f}  (s0={summary[a]['stage'][0]:.1f} -> s1={summary[a]['stage'][1]:.1f})")

    print()
    print("Second boundary (stage1_mean - stage2_mean):")
    for a in agents:
        d = summary[a]["stage"][1] - summary[a]["stage"][2]
        print(f"  {a:22} {d:+.1f}  (s1={summary[a]['stage'][1]:.1f} -> s2={summary[a]['stage'][2]:.1f})")


def arm_tracking(run: Path, label: str) -> None:
    """Fraction of steps where agent pulls current best arm, by stage."""
    trajs = load_jsonl(run / "trajectories.jsonl")
    # trajectories.jsonl is per-step or per-episode?
    if not trajs:
        print("no trajectories")
        return
    sample = trajs[0]
    if "steps" in sample or "trajectory" in sample:
        episodes = trajs
    else:
        # per-step records; group by (agent, seed, episode)
        groups: dict[tuple, list] = defaultdict(list)
        for t in trajs:
            key = (t.get("agent_name") or t.get("agent"), t.get("seed"), t.get("episode_id", 0))
            groups[key].append(t)
        episodes = []
        for (agent, seed, ep), steps in groups.items():
            steps = sorted(steps, key=lambda x: x.get("step", x.get("t", 0)))
            episodes.append({"agent_name": agent, "seed": seed, "steps": steps})

    print()
    print("-" * 72)
    print(f"{label}: fraction of pulls on true best arm by stage")
    by_agent: dict[str, list] = defaultdict(list)
    for ep in episodes:
        by_agent[ep["agent_name"]].append(ep)

    # Need oracle probs — from episode_summaries
    summaries = load_jsonl(run / "episode_summaries.jsonl")
    # seed -> prob_history from any agent (same seed same env)
    seed_probs = {}
    for s in summaries:
        if s["agent_name"] == "oracle":
            seed_probs[s["seed"]] = s["oracle"]["prob_history"]
        elif s["seed"] not in seed_probs and "prob_history" in s.get("oracle", {}):
            seed_probs[s["seed"]] = s["oracle"]["prob_history"]

    agents = ["oracle", "llm_raw_history", "ucb", "sliding_window_ucb", "greedy", "random"]
    L = 30
    for a in agents:
        if a not in by_agent:
            continue
        stage_hits = [[] for _ in range(3)]
        for ep in by_agent[a]:
            seed = ep["seed"]
            probs = seed_probs.get(seed)
            if probs is None:
                continue
            steps = ep.get("steps") or ep.get("trajectory") or []
            # if steps empty, try per-step already expanded
            if not steps and "action" in ep:
                continue
            for i, step in enumerate(steps):
                if i >= len(probs):
                    break
                best = max(range(len(probs[i])), key=lambda j: probs[i][j])
                action = step.get("action") or step.get("a")
                if isinstance(action, dict):
                    arm = action.get("arm", action.get("action"))
                else:
                    arm = action
                if arm is None:
                    arm = step.get("arm")
                try:
                    arm = int(arm)
                except (TypeError, ValueError):
                    continue
                stage = min(i // L, 2)
                stage_hits[stage].append(1.0 if arm == best else 0.0)
        means = []
        for s in range(3):
            means.append(mean(stage_hits[s]) if stage_hits[s] else float("nan"))
        print(
            f"  {a:22} best-arm rate: "
            f"s0={means[0]*100:5.1f}%  s1={means[1]*100:5.1f}%  s2={means[2]*100:5.1f}%  "
            f"all={mean([x for xs in stage_hits for x in xs])*100:5.1f}%"
        )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = [
        ("runs/2026-07-31_benchmark_staged_constant_p0", "P0: 3x constant (distinct best arms)"),
        ("runs/2026-07-31_benchmark_staged_mixed_p1", "P1: sin -> constant -> linear"),
        ("runs/2026-07-31_benchmark_staged_sinusoidal_p2", "P2: 3x sinusoidal (new params each stage)"),
    ]
    for i, (rel, label) in enumerate(runs):
        path = root / rel
        if not path.exists():
            print(f"SKIP missing: {path}")
            continue
        if i:
            print()
        analyze(path, label)
        arm_tracking(path, label.split(":")[0])


if __name__ == "__main__":
    main()
