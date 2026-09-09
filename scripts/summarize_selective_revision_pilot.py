"""Write a descriptive markdown summary from a live/fake pilot run directory."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from llm_rl_engine.analysis.selective_revision_harness import load_jsonl
from llm_rl_engine.analysis.selective_revision_pilot import summarize_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    results = load_jsonl(args.run_dir / "results.jsonl")
    attempts = load_jsonl(args.run_dir / "attempts.jsonl")
    ledger = {}
    ledger_path = args.run_dir / "ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    summary = summarize_scores(results)
    by_treatment: dict[str, list] = defaultdict(list)
    for row in results:
        by_treatment[row.get("treatment", "?")].append(row)
    lines = [
        f"# Selective revision pilot run: `{args.run_dir.name}`",
        "",
        "Descriptive grid summary, not a population estimate. No confidence intervals.",
        "",
        "## Totals",
        "",
        f"- Attempted jobs: {summary['n_attempted']}",
        f"- Valid JSON: {summary['n_valid']}",
        f"- API failures: {summary['n_api_failure']}",
        f"- Parse failures: {summary['n_parse_failure']}",
        f"- Decision errors among valid: {summary['n_decision_error']}",
        f"- ITT mean decision loss: {summary['itt_mean_decision_loss']}",
        f"- Valid mean Brier (P high): {summary['valid_mean_brier']}",
        f"- Valid mean C drift from prior predictive: {summary['valid_mean_c_drift']}",
        f"- Overpriced diagnostics (valid): {summary['overpriced_diagnostic_count']}",
        f"- Ledger spent_usd: {ledger.get('spent_usd')}",
        f"- Ledger attempts: {ledger.get('attempts')}",
        "",
        "## By treatment",
        "",
        "| Treatment | n | valid | decision errors | mean ITT loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("raw", "counts", "reference"):
        rows = by_treatment.get(name, [])
        sub = summarize_scores(rows)
        lines.append(
            f"| {name} | {sub['n_attempted']} | {sub['n_valid']} | "
            f"{sub['n_decision_error']} | {sub['itt_mean_decision_loss']} |"
        )
    attempted = [a for a in attempts if a.get("status") == "attempted"]
    if attempted:
        thoughts = [int(a.get("thoughts_tokens") or 0) for a in attempted]
        outputs = [int(a.get("output_tokens") or 0) for a in attempted]
        inputs = [int(a.get("input_tokens") or 0) for a in attempted]
        lines += [
            "",
            "## Token usage (completed attempts)",
            "",
            f"- n attempts logged complete: {len(attempted)}",
            f"- input tokens min/max: {min(inputs)} / {max(inputs)}",
            f"- billed output min/max: {min(outputs)} / {max(outputs)}",
            f"- thoughts min/max: {min(thoughts)} / {max(thoughts)}",
        ]
    lines += ["", "## Per job", "", "| case | treatment | valid | action | optimal | loss | failure |", "|---|---|---|---|---|---:|---|"]
    for row in results:
        lines.append(
            f"| {row.get('case_id')} | {row.get('treatment')} | {row.get('valid')} | "
            f"{row.get('chosen_action')} | {row.get('optimal_action')} | "
            f"{row.get('decision_loss')} | {row.get('failure_kind')} |"
        )
    text = "\n".join(lines) + "\n"
    out = args.out or (args.run_dir / "summary.md")
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
