import json
from collections import defaultdict
from pathlib import Path


def true_best(r):
    p = (r.get("private_info") or {}).get("step_probs") or []
    if not p:
        return None
    return max(range(len(p)), key=lambda i: p[i])


def main():
    run = Path(__file__).resolve().parents[1] / "runs/2026-08-21_benchmark_p0_hypothesis_phase0"
    by_seed = defaultdict(list)
    for line in (run / "trajectories.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("agent_name") == "llm_hypothesis":
            by_seed[r["seed"]].append(r)

    times = list(range(0, 6)) + [25, 28, 29, 30, 31, 32, 35, 40, 50, 55, 58, 59, 60, 61, 65, 75, 85, 89]
    for seed in range(5):
        steps = sorted(by_seed[seed], key=lambda x: x["step"])
        print("=" * 64, "seed", seed)
        for t in times:
            r = next((x for x in steps if x["step"] == t), None)
            if not r:
                continue
            pr = r.get("parsed_response") or {}
            h = pr.get("hypothesis") or {}
            print(
                f"  t={t:2d} true={true_best(r)} act={r['action'].get('arm')} "
                f"H={h.get('best_arm')} stable={h.get('stable')} "
                f"conf={pr.get('confidence')} heur={pr.get('heuristic')} R={r['reward']}"
            )
            note = pr.get("heuristic_text") or pr.get("rationale") or ""
            if t in (5, 30, 31, 60, 61) and note:
                print(f"       {str(note)[:180]}")


if __name__ == "__main__":
    main()
