"""CPU feasibility of the paired revision objective (task 007). No APIs, no training."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
import sys

from llm_rl_engine.analysis.paired_revision import (
    algebra_numeric_checks,
    evaluate_probes,
    fixture_coverage,
    generate_pairs,
    showcase_pair,
    solver_input,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "reports" / "paired_revision_feasibility"


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rss0 = rss_mb()
    pairs = generate_pairs()
    algebra = algebra_numeric_checks()
    coverage = fixture_coverage(pairs)
    probes_train = evaluate_probes(pairs, split="train")
    probes_test = evaluate_probes(pairs, split="test")
    show = showcase_pair(pairs)
    elapsed = time.perf_counter() - t0
    rss1 = rss_mb()

    solver_rows = []
    evaluator_rows = []
    for pair in pairs:
        solver_rows.append(
            {
                "pair_id": pair["pair_id"],
                "split": pair["split"],
                "family_id": pair["family_id"],
                "enumeration": pair["enumeration"],
                "pair_hash": pair["pair_hash"],
                "previous_states_equal": pair["previous_states_equal"],
                "a": solver_input(pair["a"]),
                "b": solver_input(pair["b"]),
                "a_identity": pair["a"]["identity"],
                "b_identity": pair["b"]["identity"],
            }
        )
        evaluator_rows.append(
            {
                "pair_id": pair["pair_id"],
                "split": pair["split"],
                "d_diff": pair["d_diff"],
                "a": pair["a"]["evaluator"],
                "b": pair["b"]["evaluator"],
            }
        )

    manifest = {
        "kind": "paired_revision_feasibility_v1",
        "n_pairs": len(pairs),
        "split": coverage["by_split"],
        "enumerated_evidence": True,
        "runtime_s": elapsed,
        "rss_mb_start": rss0,
        "rss_mb_peak": rss1,
        "algebra": algebra,
        "coverage": coverage,
        "probes_train": probes_train,
        "probes_test": probes_test,
        "showcase": show,
        "note": "Synthetic prediction-error probes, not a training result. No APIs.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "solver_inputs.json").write_text(json.dumps(solver_rows, indent=2, sort_keys=True) + "\n")
    (out / "evaluator.json").write_text(json.dumps(evaluator_rows, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {out} pairs={len(pairs)} runtime_s={elapsed:.3f} rss_mb={rss1:.1f} "
        f"related_gap={coverage['max_unobserved_related_gap']:.6g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
