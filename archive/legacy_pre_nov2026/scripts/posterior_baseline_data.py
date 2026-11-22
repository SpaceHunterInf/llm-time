"""Generate the task-008 posterior-distillation dataset. CPU only. No model load."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from llm_rl_engine.analysis.posterior_baseline import (
    DATASET_SEED,
    generate_dataset,
    write_dataset,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "reports" / "posterior_baseline_preflight" / "data"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DATASET_SEED)
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    dataset = generate_dataset(seed=args.seed)
    hashes = write_dataset(dataset, args.output_dir)
    elapsed = time.perf_counter() - t0
    summary = {
        "n": dataset["n"],
        "coverage": dataset["coverage"],
        "file_hashes": hashes,
        "runtime_s": elapsed,
        "seed": args.seed,
        "attempts": dataset["attempts"],
    }
    (args.output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"n": dataset["n"], "runtime_s": round(elapsed, 3), "attempts": dataset["attempts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
