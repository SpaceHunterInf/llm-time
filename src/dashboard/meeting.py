"""Render the dated group meeting from its saved AlphaDesk and banking reports."""

from __future__ import annotations

from html import escape
from pathlib import Path
from string import Template
from typing import Any


MODEL_LABELS = {
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "gemini-4-hydrogen": "Gemini 4 Hydrogen",
    "gemma-4-31b-it": "Gemma 4 31B IT",
}


def _table(caption: str, headings: tuple[str, ...], rows: list[list[str]]) -> str:
    if not rows:
        return '<p class="figure-note">The saved report is unavailable in this checkout.</p>'
    head = "".join(f'<th scope="col">{escape(label)}</th>' for label in headings)
    body = "".join(
        f'<tr><th scope="row">{escape(row[0])}</th>'
        + "".join(f"<td>{escape(cell)}</td>" for cell in row[1:]) + "</tr>"
        for row in rows
    )
    return (f'<table class="brief-table"><caption>{escape(caption)}</caption>'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")


def render_meeting_brief(telemetry: dict[str, Any]) -> str:
    report = next((r for r in telemetry.get("matrix_reports", [])
                   if r.get("run_id") == "autonomy_cross_model_n6" and "by_model_mode" in r), {})
    reversal, opportunity, gains = [], [], []
    for model, label in MODEL_LABELS.items():
        modes = report.get("by_model_mode", {}).get(model, {})
        paired = report.get("paired_contrasts", {}).get(model, {})
        if modes.get("abrupt_reversal") and paired.get("abrupt_reversal"):
            arms = modes["abrupt_reversal"]
            reversal.append([label] + [f'{arms[arm]["post_dp_utility_pct"]:.1f}%'
                            for arm in ("scoped_cards_v3", "factual_ledger_v3", "sgeh_directive_v3")])
            gains.extend(paired["abrupt_reversal"]["factual_minus_cardbook_rep_gain"]["per_seed"])
        if paired.get("opportunity_rise") and paired.get("opportunity_stationary"):
            opportunity.append([label] + [
                f'{paired[world]["directive_minus_factual_ctrl_gain"]["mean"]:+.2f}'.replace("-", "−")
                for world in ("opportunity_rise", "opportunity_stationary")
            ])

    ledger_branches = [r for r in report.get("records", [])
                       if r["arm"] == "factual_ledger_v3" and r["fork_mode"] == "opportunity_rise"]
    bank_rows = []
    bank = (telemetry.get("slb_l3_report") or {}).get("methods", {})
    for key, label in (("sgeh_slb", "Our banking harness"), ("srn", "Strategy note (SRN)"),
                       ("blind", "No memory"), ("skillbook", "SkillBook rule library")):
        if key in bank:
            method = bank[key]
            total = sum(w["n_test"] for w in method["windows"] if w["window"] > 0)
            passes = sum(score == 100 for score in method["curve_test_scores"][-total:]) if total else 0
            bank_rows.append([label, f"{passes}/{total}", str(method["reflection_calls"])])

    template = Template(Path(__file__).with_name("meeting.html").read_text(encoding="utf-8"))
    return template.substitute(
        reversal_tally=f"{sum(g > 0 for g in gains)}/{len(gains)}" if gains else "—",
        reversal_table=_table("Net utility after an abrupt reversal", ("Model", "Rule cards", "Ledger", "Ledger + directions"), reversal),
        opportunity_checks=sum(r["probes_total"] for r in ledger_branches) if ledger_branches else "—",
        opportunity_sessions=sum(r["episodes_post"] for r in ledger_branches) if ledger_branches else "—",
        opportunity_branches=len(ledger_branches) if ledger_branches else "—",
        opportunity_table=_table("Effect of adding directions to the ledger", ("Model", "Alternative improved", "Stable control"), opportunity),
        banking_table=_table("Held-out banking cases", ("Memory method", "Cases passed", "Reflection calls"), bank_rows),
    )
