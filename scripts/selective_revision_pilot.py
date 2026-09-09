"""Selective-revision fixed-history pilot. Dry-run is the default.

Does not read .env, construct an API client, or use the network unless --live
is passed with an explicit budget. Do not run live until the research lead
supplies execution settings.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from llm_rl_engine.analysis.selective_revision_harness import (
    BoundedGeminiClient,
    LiveConfigError,
    PilotRun,
    live_settings_from_config,
    load_pilot_config,
    run_jobs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "llm_rl_engine" / "configs" / "pilot_selective_revision_v1.yaml"
DEFAULT_DRYRUN_DIR = REPO_ROOT / "reports" / "selective_revision_pilot_v1_dryrun"
DEFAULT_REPORT = REPO_ROOT / "reports" / "selective_revision_pilot_dryrun.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selective-revision pilot (dry-run default).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--live", action="store_true", help="Make provider calls. Requires budget flags.")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--smoke", action="store_true", help="At most 3 live jobs.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--max-usd", type=float, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--thinking-budget", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config, spec = load_pilot_config(args.config)
    if args.live:
        output_dir = args.output_dir or (REPO_ROOT / "runs" / "selective_revision_pilot_v1")
        run = PilotRun(output_dir, config, spec, repo_root=REPO_ROOT)
        try:
            settings = live_settings_from_config(
                config,
                {
                    "model": args.model,
                    "max_calls": args.max_calls,
                    "max_usd": args.max_usd,
                    "max_input_tokens": args.max_input_tokens,
                    "max_output_tokens": args.max_output_tokens,
                    "thinking_budget": args.thinking_budget,
                },
            )
        except LiveConfigError as exc:
            print(f"Rejected live config before any calls: {exc}", file=sys.stderr)
            return 2
        from llm_rl_engine.config.env import load_dotenv

        load_dotenv()
        client = BoundedGeminiClient(
            settings.model,
            max_output_tokens=settings.max_output_tokens,
            thinking_budget=settings.thinking_budget,
            temperature=settings.temperature,
        )
        smoke = 3 if args.smoke else None
        result = run_jobs(
            run,
            settings,
            client,
            smoke_limit=smoke,
            resume=args.resume,
            progress=True,
        )
        print(f"Live run artifacts: {output_dir}")
        print(canonical_summary(result))
        return 0

    output_dir = args.output_dir or DEFAULT_DRYRUN_DIR
    run = PilotRun(output_dir, config, spec, repo_root=REPO_ROOT)
    report = run.write_dry_run()
    markdown = output_dir / "dryrun.md"
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(markdown, DEFAULT_REPORT)
    json_copy = REPO_ROOT / "reports" / "selective_revision_pilot_dryrun.json"
    shutil.copyfile(output_dir / "dryrun.json", json_copy)
    print(f"Dry-run wrote {output_dir}")
    print(f"Planned calls: {report['manifest']['n_planned_calls']}")
    print(
        "Reserved USD for grid plus retries: "
        f"{report['manifest']['reserved_usd_full_grid']:.6f}"
    )
    print(f"Copied report to {DEFAULT_REPORT}")
    return 0


def canonical_summary(result: dict) -> str:
    summary = result.get("summary") or {}
    ledger = result.get("ledger") or {}
    return (
        f"attempted={summary.get('n_attempted')} valid={summary.get('n_valid')} "
        f"api_fail={summary.get('n_api_failure')} parse_fail={summary.get('n_parse_failure')} "
        f"decision_err={summary.get('n_decision_error')} spent_usd={ledger.get('spent_usd')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
