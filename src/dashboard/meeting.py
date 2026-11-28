"""Render the dated group meeting deck from saved AlphaDesk, banking, H=60 reviewer, and Qwen3-8B reports."""

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
    return (
        f'<table class="brief-table"><caption>{escape(caption)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def render_meeting_brief(telemetry: dict[str, Any]) -> str:
    m_reports = telemetry.get("matrix_reports", [])
    report = next(
        (r for r in m_reports if r.get("run_id") == "autonomy_cross_model_n6" and "by_model_mode" in r),
        {},
    )
    reversal, opportunity, gains = [], [], []
    for model, label in MODEL_LABELS.items():
        modes = report.get("by_model_mode", {}).get(model, {})
        paired = report.get("paired_contrasts", {}).get(model, {})
        if modes.get("abrupt_reversal") and paired.get("abrupt_reversal"):
            arms = modes["abrupt_reversal"]
            reversal.append(
                [label]
                + [
                    f'{arms[arm]["post_dp_utility_pct"]:.1f}%'
                    for arm in ("scoped_cards_v3", "factual_ledger_v3", "sgeh_directive_v3")
                ]
            )
            gains.extend(paired["abrupt_reversal"]["factual_minus_cardbook_rep_gain"]["per_seed"])
        if paired.get("opportunity_rise") and paired.get("opportunity_stationary"):
            opportunity.append(
                [label]
                + [
                    f'{paired[world]["directive_minus_factual_ctrl_gain"]["mean"]:+.2f}'.replace("-", "−")
                    for world in ("opportunity_rise", "opportunity_stationary")
                ]
            )

    ledger_branches = [
        r
        for r in report.get("records", [])
        if r["arm"] == "factual_ledger_v3" and r["fork_mode"] == "opportunity_rise"
    ]
    bank_rows = []
    bank = (telemetry.get("slb_l3_report") or {}).get("methods", {})
    for key, label in (
        ("sgeh_slb", "Our banking harness"),
        ("srn", "Strategy note (SRN)"),
        ("blind", "No memory"),
        ("skillbook", "SkillBook rule library"),
    ):
        if key in bank:
            method = bank[key]
            total = sum(w["n_test"] for w in method["windows"] if w["window"] > 0)
            passes = sum(score == 100 for score in method["curve_test_scores"][-total:]) if total else 0
            bank_rows.append([label, f"{passes}/{total}", str(method["reflection_calls"])])

    # Nov 25-26 Hidden-Ambiguity VOI + Verify-Before-Quarantine (VBQ)
    voi_rep = next((r for r in m_reports if r.get("run_id") == "voi_hidden_ambiguity_complete_n6"), {})
    voi_rows: list[list[str]] = []
    voi_flash = voi_rep.get("by_model_mode", {}).get("gemini-3.8-flash", {})
    for arm_k, arm_lbl in (
        ("scoped_cards_v3", "Rule cards (baseline)"),
        ("factual_ledger_v3", "Factual ledger"),
        ("editable_proc_v3", "Self-authored procedure"),
        ("sgeh_directive_v3", "Rigid CUSUM quarantine"),
        ("sgeh_vbq_v3", "Verify-Before-Quarantine (VBQ)"),
    ):
        dip_d = voi_flash.get("hidden_transient_dip", {}).get(arm_k)
        rev_d = voi_flash.get("abrupt_reversal", {}).get(arm_k)
        if dip_d and rev_d:
            voi_rows.append(
                [
                    arm_lbl,
                    f'{float(dip_d.get("post_dp_utility_pct", 0.0)):.1f}%',
                    f'{float(rev_d.get("post_dp_utility_pct", 0.0)):.1f}%',
                    f'{float(dip_d.get("mean_probes", 0.0)):.1f}',
                ]
            )

    # Nov 27-28 H=60 Two-Level Reviewer & K=8 Repeated-Trial Replay
    meta_orig = next((r for r in m_reports if r.get("run_id") == "meta_reviewer_h60_flash_n6"), {})
    meta_dipc = next((r for r in m_reports if r.get("run_id") == "meta_dip_c_h60_flash_n6"), {})
    meta_hyd = next((r for r in m_reports if r.get("run_id") == "meta_reviewer_h60_hydrogen_n3"), {})
    h60_rows: list[list[str]] = []
    for run_lbl, rep_obj, mdl_k, mode_k in (
        ("Flash · Partial return A' (N=6)", meta_orig, "gemini-3.8-flash", "drift_meta_60"),
        ("Flash · Dip→C ending (N=6)", meta_dipc, "gemini-3.8-flash", "drift_dip_c_60"),
        ("Hydrogen · Partial return A' (N=3)", meta_hyd, "gemini-4-hydrogen", "drift_meta_60"),
    ):
        mm = rep_obj.get("by_model_mode", {}).get(mdl_k, {}).get(mode_k, {})
        if mm:
            h60_rows.append(
                [
                    run_lbl,
                    f'{float(mm.get("factual_ledger_v3", {}).get("w4_dp_utility_pct", 0.0)):.1f}%',
                    f'{float(mm.get("frozen_reviewer_v3", {}).get("w4_dp_utility_pct", 0.0)):.1f}%',
                    f'{float(mm.get("editable_reviewer_v3", {}).get("w4_dp_utility_pct", 0.0)):.1f}%',
                    f'{float(mm.get("sgeh_vbq_v3", {}).get("w4_dp_utility_pct", 0.0)):.1f}%',
                ]
            )

    rep_k8 = next((r for r in m_reports if r.get("run_id") == "repeated_rule_replay_stage1_k8"), {})
    k8_rows: list[list[str]] = []
    for ck in rep_k8.get("checkpoints", []):
        pkey = str(ck.get("preset_key", ""))
        if pkey in ("flash_s704_w4_ep48", "flash_s705_w4_ep44", "flash_s709_dipc_w4_ep44", "flash_s711_dipc_w4_ep40"):
            cv = ck.get("contrasts", {}).get("revised_minus_initial_verbatim", {})
            ca = ck.get("contrasts", {}).get("revised_minus_initial_anonymized", {})
            k8_rows.append(
                [
                    pkey.replace("flash_", "").replace("_", " "),
                    f'{float(cv.get("diff_mean", 0.0)):+.2f} pts (p={float(cv.get("perm_p_two_sided", 1.0)):.4f})'.replace("-", "−"),
                    f'{float(ca.get("diff_mean", 0.0)):+.2f} pts (p={float(ca.get("perm_p_two_sided", 1.0)):.4f})'.replace("-", "−"),
                ]
            )

    # Nov 28 Untuned Qwen/Qwen3-8B Keep-or-Apply Gatekeeper
    qwen_rep = next((r for r in m_reports if r.get("run_id") == "qwen8b_keep_or_apply_n8"), {})
    qwen_rows: list[list[str]] = []
    for sl_lbl, sl_obj in (
        ("All preselected (M=32)", qwen_rep.get("all_preselected_summary", {})),
        ("Distinct proposals (M=30)", qwen_rep.get("distinct_proposal_summary", {})),
        ("Original ending A' (M=16)", qwen_rep.get("by_meta_mode", {}).get("drift_meta_60", {})),
        ("Dip→C ending (M=16)", qwen_rep.get("by_meta_mode", {}).get("drift_dip_c_60", {})),
    ):
        if sl_obj and int(sl_obj.get("n_opportunities", 0)) > 0:
            qwen_rows.append(
                [
                    sl_lbl,
                    f'{float(sl_obj.get("apply_rate_pct", 0.0)):.1f}%',
                    f'{float(sl_obj.get("always_keep_dp_pct", 0.0)):.2f}%',
                    f'{float(sl_obj.get("always_apply_dp_pct", 0.0)):.2f}%',
                    f'{float(sl_obj.get("qwen8b_gate_dp_pct", 0.0)):.2f}%',
                    f'{float(sl_obj.get("oracle_gate_dp_pct", 0.0)):.2f}%',
                ]
            )

    nt_apply = qwen_rep.get("distinct_proposal_summary", {}).get("apply_rate_pct")
    qwen_apply_stat = f"{float(nt_apply):.1f}%" if nt_apply is not None else "—"

    template = Template(Path(__file__).with_name("meeting.html").read_text(encoding="utf-8"))
    return template.substitute(
        reversal_tally=f"{sum(g > 0 for g in gains)}/{len(gains)}" if gains else "—",
        reversal_table=_table(
            "Net utility after an abrupt reversal",
            ("Model", "Rule cards", "Ledger", "Ledger + directions"),
            reversal,
        ),
        opportunity_checks=sum(r["probes_total"] for r in ledger_branches) if ledger_branches else "—",
        opportunity_sessions=sum(r["episodes_post"] for r in ledger_branches) if ledger_branches else "—",
        opportunity_branches=len(ledger_branches) if ledger_branches else "—",
        opportunity_table=_table(
            "Effect of adding directions to the ledger",
            ("Model", "Alternative improved", "Stable control"),
            opportunity,
        ),
        banking_table=_table(
            "Held-out banking cases",
            ("Memory method", "Cases passed", "Reflection calls"),
            bank_rows,
        ),
        voi_table=_table(
            "Matching initial loss & free signals (Gemini 3.8 Flash, N=6)",
            ("Method", "2-day hidden dip", "Lasting reversal", "Dip checks"),
            voi_rows,
        ),
        h60_table=_table(
            "Final-window (W4) score in 60-session two-level reviewer runs",
            ("Setting", "Facts only", "Frozen rule", "Editable rule", "VBQ controller"),
            h60_rows,
        ),
        k8_table=_table(
            "Eight-trial counterfactual rule replay at identical saved states",
            ("Checkpoint", "Revised − Initial (verbatim)", "Strategy names stripped"),
            k8_rows,
        ),
        qwen_apply_rate=qwen_apply_stat,
        qwen_table=_table(
            "Untuned Qwen3-8B pre-outcome Keep-or-Apply vs. baselines (Seeds 731–738, K=6)",
            ("Slice", "8B Apply%", "Always-Keep", "Always-Apply", "Untuned 8B", "Oracle ceiling"),
            qwen_rows,
        ),
    )
