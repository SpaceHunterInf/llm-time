"""Command-line entry point for running benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_rl_engine.runners.run_benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM RL benchmark experiments.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Directory for run logs and metrics.",
    )
    args = parser.parse_args()
    out = run_benchmark(args.config, output_dir=args.output_dir)
    print(f"Benchmark complete. Results saved to: {out}")


if __name__ == "__main__":
    main()