"""Command-line entry point for running benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_rl_engine.analysis.plots import generate_run_plots
from llm_rl_engine.config.env import load_dotenv
from llm_rl_engine.runners.run_benchmark import run_benchmark


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run LLM RL benchmark experiments.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment YAML config (runs benchmark).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Directory for run logs and metrics.",
    )
    parser.add_argument(
        "--plot-run-dir",
        type=str,
        help="Generate plots/dashboard for an existing run directory.",
    )
    args = parser.parse_args()

    if args.plot_run_dir:
        plots = generate_run_plots(Path(args.plot_run_dir))
        print(f"Plots saved to: {plots}")
        return

    if not args.config:
        parser.error("Provide --config to run a benchmark, or --plot-run-dir to visualize.")

    out = run_benchmark(args.config, output_dir=args.output_dir)
    print(f"Benchmark complete. Results saved to: {out}")
    print(f"Dashboard: {out / 'dashboard.html'}")


if __name__ == "__main__":
    main()