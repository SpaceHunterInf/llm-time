#!/usr/bin/env python3
"""Run cost-capped ParadigmBench experiments (with parallel cell workers) and export dashboard."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_dashboard import build_static_snapshot  # noqa: E402
from src.core.ledger import BudgetExceededError, CostLedger, send_monitor_email  # noqa: E402
from src.runner import LifetimeSummary, run_lifetime  # noqa: E402


def format_markdown_table(rows: list[LifetimeSummary]) -> str:
    header = (
        "| Environment | Harness | Model | Mean Score | Pre-Shift | Post-Shift | Stale Lock-In | Inv. Viol | RSI Tax | USD |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    ordered = sorted(rows, key=lambda r: (r.env_spec, -r.post_shift_utility))
    for r in ordered:
        lines.append(
            f"| `{r.env_spec}` | **{r.harness_name}** | `{r.model}` | "
            f"{r.mean_score:.1f} | {r.pre_shift_utility:.1f}% | **{r.post_shift_utility:.1f}%** | "
            f"{r.stale_lock_in_rate:.1f}% | {r.invariant_violation_rate:.1f}% | "
            f"{r.rsi_tax_call_share:.1f}% ({r.total_reflection_calls}) | `${r.total_usd:.4f}` |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ParadigmBench experiment sweep")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--phase", default="dev", help="Budget phase tag (dev, act1_act2, act3_datagen)")
    parser.add_argument(
        "--envs",
        default="life_companion:lifecycle,coalition_ops:reversal,alpha_desk:gradual_decay,coalition_ops:stationary",
        help="Comma-separated environment specs",
    )
    parser.add_argument(
        "--harnesses",
        default="blind,srn,skillbook,sgeh",
        help="Comma-separated harness names",
    )
    parser.add_argument("--model", default="gemini-3.8-flash", help="Model ID (or 'mock')")
    parser.add_argument("--horizon", type=int, default=15, help="Episodes per lifetime")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--thinking-budget", type=int, default=256, help="Thinking token cap per call")
    parser.add_argument("--workers", type=int, default=4, help="Parallel lifetime workers")
    parser.add_argument("--max-usd", type=float, default=3.5, help="Hard USD guardrail for this sweep")
    parser.add_argument("--email-on-finish", action="store_true", help="Email summary to gs06monitor@gmail.com")
    args = parser.parse_args()

    envs = [x.strip() for x in args.envs.split(",") if x.strip()]
    harnesses = [x.strip() for x in args.harnesses.split(",") if x.strip()]
    jobs = [(e, h) for e in envs for h in harnesses]
    total_units = len(jobs) * args.horizon

    ledger = CostLedger(
        results_root=ROOT / "results",
        run_id=args.run_id,
        phase=args.phase,
        max_usd=args.max_usd,
    )
    ledger.update_progress(completed_units=0, total_units=total_units, status="running", current_label="launching")

    summaries: list[LifetimeSummary] = []
    done_lock = threading.Lock()
    done_cells = 0
    status = "completed"

    def _run_one(env_spec: str, h_name: str) -> LifetimeSummary:
        nonlocal done_cells
        summary, _ = run_lifetime(
            env_spec=env_spec,
            harness_name=h_name,
            model=args.model,
            seed=args.seed,
            horizon=args.horizon,
            ledger=ledger,
            dotenv_path=ROOT / ".env",
            thinking_budget=args.thinking_budget,
            progress_offset=done_cells * args.horizon,
            progress_total=total_units,
        )
        with done_lock:
            done_cells += 1
            ledger.update_progress(
                completed_units=done_cells * args.horizon,
                total_units=total_units,
                status="running",
                current_label=f"completed {env_spec} | {h_name}",
            )
            print(
                f"[DONE {done_cells}/{len(jobs)}] {env_spec} | {h_name:9s} | "
                f"PostShift={summary.post_shift_utility:5.1f}% | LockIn={summary.stale_lock_in_rate:5.1f}% | "
                f"RSITax={summary.rsi_tax_call_share:4.1f}% | Cell=${summary.total_usd:.4f} | Run=${ledger.run_usd:.4f}",
                flush=True,
            )
        return summary

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = {pool.submit(_run_one, e, h): (e, h) for e, h in jobs}
            for fut in as_completed(futs):
                summaries.append(fut.result())
    except BudgetExceededError as exc:
        status = "halted_budget"
        print(f"[BUDGET HALT] {exc}", flush=True)
    except Exception as exc:
        status = "failed"
        ledger.update_progress(
            completed_units=done_cells * args.horizon,
            total_units=total_units,
            status=status,
            current_label=type(exc).__name__,
        )
        raise

    ledger.update_progress(
        completed_units=done_cells * args.horizon,
        total_units=total_units,
        status=status,
        current_label="finished",
    )

    md_table = format_markdown_table(summaries)
    report_path = ledger.run_dir / "summary.md"
    report_path.write_text(
        f"# Run Summary: `{args.run_id}`\n\n"
        f"- **Status:** `{status}`\n"
        f"- **Total Run Spend:** `${ledger.run_usd:.4f}` (Cap: `${args.max_usd:.2f}`)\n\n"
        f"{md_table}\n",
        encoding="utf-8",
    )
    snap_path = build_static_snapshot(ROOT, ROOT / "results" / "supervisor_dashboard.html")
    print(f"\n{md_table}\n\nSaved summary: {report_path}\nUpdated dashboard snapshot: {snap_path}", flush=True)

    if args.email_on_finish:
        send_monitor_email(
            subject=f"Run {args.run_id} ({status}) - ${ledger.run_usd:.4f}",
            body=f"Run {args.run_id} finished with status={status}, spend=${ledger.run_usd:.4f}.\n\n{md_table}\n",
        )


if __name__ == "__main__":
    main()
