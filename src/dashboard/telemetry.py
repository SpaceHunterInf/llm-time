"""Comprehensive read-only data & documentation collector for the Project Portal & Dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from src.core.ledger import TOTAL_GEMINI_CREDIT_POOL_USD, mask_email_for_display
from src.core.pricing import get_model_rate
from src.dashboard.handbook_content import (
    ACT3_PIPELINE,
    ALPHADESK_V2_PAYOFF_TABLES,
    ALPHADESK_V3_LITERATURE_INHERITANCE,
    CARD_SURGERY_ARMS,
    CHRONOLOGICAL_DEBATE_LOG,
    COMPETITOR_MATRIX,
    CONCRETE_WALKTHROUGHS,
    COST_TIERS,
    CURATED_FLAGSHIP_TRACES,
    DECISIONS_D1_D7,
    DRIFT_TAXONOMY,
    ENVIRONMENT_DOCS,
    LATIN_SQUARE_COUNTERBALANCING,
    METHOD_DOCS,
    MILESTONES_TIMELINE,
    OPEN_SUPERVISOR_QUESTIONS,
    THESIS_EVOLUTION,
)

SEED_SUFFIX_RE = re.compile(r"_s(\d+)$")


def _extract_seed_label(model_seed: str) -> str:
    m = SEED_SUFFIX_RE.search(model_seed)
    return m.group(1) if m else "all"


def _has_probe_step(ep_obj: dict[str, Any]) -> bool:
    for st in ep_obj.get("steps", []):
        if isinstance(st, dict) and st.get("tool") in (
            "probe_strategy_execution",
            "check_strategy_diagnostics",
            "audit_counterparty",
        ):
            return True
    return False


def _collect_trace_samples(results_dir: Path) -> list[dict[str, Any]]:
    """Load representative pre-shift, probe, and post-shift episodes across real runs in `results/raw/`."""
    raw_root = results_dir / "raw"
    samples: list[dict[str, Any]] = []
    if not raw_root.is_dir():
        return samples

    all_dirs = [d for d in raw_root.iterdir() if d.is_dir() and list(d.glob("*.jsonl"))]
    non_smoke = [d for d in all_dirs if not d.name.startswith("smoke")]
    target_dirs = sorted(non_smoke or all_dirs, key=lambda d: d.stat().st_mtime, reverse=True)[:6]

    for run_dir in target_dirs:
        run_id = run_dir.name
        for trace_file in sorted(run_dir.glob("*.jsonl")):
            parts = trace_file.stem.split("__")
            if len(parts) < 3:
                continue
            env_slug, harness_name, model_seed = parts[0], parts[1], parts[2]
            seed_str = _extract_seed_label(model_seed)
            rows: list[dict[str, Any]] = []
            try:
                with open(trace_file, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            rows.append(json.loads(line))
            except (OSError, json.JSONDecodeError):
                continue
            if not rows:
                continue

            probe_indices = [i for i, r in enumerate(rows) if i >= len(rows) // 3 and _has_probe_step(r)]
            chosen_set = {
                0,
                min(len(rows) - 1, max(1, len(rows) // 2)),
                max(0, len(rows) - 2),
            }
            if probe_indices:
                chosen_set.add(probe_indices[0])

            for idx in sorted(chosen_set)[:4]:
                ep_obj = rows[idx]
                samples.append(
                    {
                        "run_id": run_id,
                        "env_slug": env_slug,
                        "harness": harness_name,
                        "seed": seed_str,
                        "model_seed": model_seed,
                        "episode": int(ep_obj.get("episode", idx + 1)),
                        "t_star": ep_obj.get("t_star"),
                        "regime": str(ep_obj.get("regime", "")),
                        "public_score": float(ep_obj.get("public_score", 0.0)),
                        "normalized_utility_pct": round(100.0 * float(ep_obj.get("normalized_utility", 0.0)), 1),
                        "stale_policy_match": bool(ep_obj.get("stale_policy_match", False)),
                        "optimal_policy_match": bool(ep_obj.get("optimal_policy_match", False)),
                        "invariant_violation": bool(ep_obj.get("invariant_violation", False)),
                        "probes_used": int(ep_obj.get("probes_used", 1 if _has_probe_step(ep_obj) else 0)),
                        "harness_meta": ep_obj.get("harness_meta", {}),
                        "steps": ep_obj.get("steps", []),
                    }
                )
    return samples


def _collect_checkpoint_samples(runs_dir: Path) -> dict[str, Any]:
    """Collect saved `checkpoint_seed_101.json` metadata across causal runs for surgery verification."""
    out: dict[str, Any] = {}
    if not runs_dir.is_dir():
        return out
    for run_subdir in sorted(runs_dir.iterdir()):
        if not run_subdir.is_dir():
            continue
        ck_file = run_subdir / "checkpoint_seed_101.json"
        if ck_file.is_file():
            try:
                raw_ck = json.loads(ck_file.read_text(encoding="utf-8"))
                out[run_subdir.name] = {
                    "seed": raw_ck.get("seed", 101),
                    "t_star": raw_ck.get("t_star"),
                    "fork_offset": raw_ck.get("fork_offset"),
                    "fork_source": raw_ck.get("fork_source", "shared_prefix"),
                    "context_char_lengths": raw_ck.get("context_char_lengths", {}),
                    "history_lengths_at_fork": raw_ck.get("history_lengths_at_fork", {}),
                }
            except (OSError, json.JSONDecodeError):
                continue
    return out


def collect_dashboard_telemetry(workspace_root: Path) -> dict[str, Any]:
    """Assemble full Project Handbook documentation, math context, debates, budget ledger, and live results.

    Security invariant: Never reads `.env` or exposes unmasked PII/secrets.
    """
    root = workspace_root.resolve()
    results_dir = root / "results"
    summary_path = results_dir / "ledger" / "summary.json"
    runs_dir = results_dir / "runs"

    if summary_path.is_file():
        try:
            ledger_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            ledger_summary = {}
    else:
        ledger_summary = {}

    spent_usd = round(float(ledger_summary.get("total_usd", 0.0)), 4)
    budget_block = {
        "credit_pool_usd": TOTAL_GEMINI_CREDIT_POOL_USD,
        "spent_usd": spent_usd,
        "remaining_usd": round(TOTAL_GEMINI_CREDIT_POOL_USD - spent_usd, 4),
        "total_calls": int(ledger_summary.get("total_calls", 0)),
        "total_input_tokens": int(ledger_summary.get("total_input_tokens", 0)),
        "total_output_tokens": int(ledger_summary.get("total_output_tokens", 0)),
        "total_thinking_tokens": int(ledger_summary.get("total_thinking_tokens", 0)),
        "monitor_contact": mask_email_for_display("gs06monitor@gmail.com"),
        "by_phase": ledger_summary.get("by_phase", {}),
        "by_model": ledger_summary.get("by_model", {}),
        "planned_allocations": [
            {"phase": "Phase 0-1: Tier-1 Causal & Fast Dev Slices (<$1.95/run)", "planned_usd": "$60 - $200", "key": "dev"},
            {"phase": "Phase 2: Act 1 & Act 2 Multi-Seed Capability Matrix", "planned_usd": "$2,500 - $5,500", "key": "act1_act2"},
            {"phase": "Phase 3: Stress Sweeps & Scaling Ablations", "planned_usd": "$4,000 - $10,000", "key": "act3_datagen"},
            {"phase": "Phase 4: Frozen Cross-Lab Final Leaderboard", "planned_usd": "$150 - $400 (ext) + $1,500 Gemini", "key": "final_eval"},
        ],
    }

    run_statuses: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    causal_report_entries: list[tuple[float, dict[str, Any]]] = []
    matrix_report_entries: list[tuple[float, dict[str, Any]]] = []
    run_summaries: list[tuple[float, dict[str, str]]] = []
    slb_mini_report: dict[str, Any] | None = None
    slb_l3_report: dict[str, Any] | None = None

    has_iter2_agg = (runs_dir / "iter2_frozen_n4").is_dir()
    has_iter2d_agg = (runs_dir / "iter2d_shared_fork_n4").is_dir()
    has_slb_complete = (runs_dir / "slb_banking_card_complete").is_dir()
    has_slb_l3_complete = (runs_dir / "slb_banking_l3_6win").is_dir()
    has_v3_agg = any(
        (runs_dir / d_name).is_dir()
        for d_name in ("alphadesk_v3_multishift_n4", "alphadesk_v3_multishift_n2", "alphadesk_v3_complete")
    )
    has_autonomy_agg = any(
        (runs_dir / d_name).is_dir()
        for d_name in ("autonomy_cross_model_n6", "autonomy_cross_model_n4")
    )

    superseded_shards = {"iter2_cb_s201"}
    if has_iter2_agg:
        superseded_shards.update({"iter2_frozen_s201", "iter2_frozen_s202", "iter2_frozen_s203", "iter2_frozen_s204"})
    if has_iter2d_agg:
        superseded_shards.update({"iter2d_shared_s205_206", "iter2d_shared_s207_208"})
    if has_slb_complete:
        superseded_shards.update({"slb_banking_card_l2", "slb_mini_banking_l2", "slb_retail_exchange_l2"})
    if has_slb_l3_complete and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith("slb_b_l3_"):
                superseded_shards.add(d.name)
    if has_v3_agg and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith("v3_"):
                superseded_shards.add(d.name)
    if has_autonomy_agg and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and (d.name.startswith("autonomy_s") or d.name.startswith("autonomy_gemma")):
                superseded_shards.add(d.name)
    if (runs_dir / "voi_hidden_ambiguity_complete_n6").is_dir() and runs_dir.is_dir():
        superseded_shards.add("voi_hidden_ambiguity_flash_n6")
        for d in runs_dir.iterdir():
            if d.is_dir() and (d.name.startswith("voi_ha_flash_") or d.name.startswith("voi_vbq_dec_")):
                superseded_shards.add(d.name)
    elif (runs_dir / "voi_hidden_ambiguity_flash_n6").is_dir() and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith("voi_ha_flash_"):
                superseded_shards.add(d.name)
    if (runs_dir / "voi_hidden_ambiguity_hydrogen_n3").is_dir() and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith("voi_ha_hydro_"):
                superseded_shards.add(d.name)
    if (runs_dir / "proc3w_complete_n4").is_dir() and runs_dir.is_dir():
        superseded_shards.add("proc3w_flash_n4")
        for d in runs_dir.iterdir():
            if d.is_dir() and (
                d.name.startswith("proc3w_high_")
                or d.name.startswith("proc3w_opp_")
                or d.name.startswith("proc3w_dv_")
                or d.name == "proc3w_w1_ckpt"
            ):
                superseded_shards.add(d.name)
    elif (runs_dir / "proc3w_flash_n4").is_dir() and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and (d.name.startswith("proc3w_high_") or d.name.startswith("proc3w_opp_") or d.name == "proc3w_w1_ckpt"):
                superseded_shards.add(d.name)
    if (runs_dir / "proc3w_hydrogen_n3").is_dir() and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith("proc3w_hy_"):
                superseded_shards.add(d.name)
    if (runs_dir / "slb_banking_l3_transfer_6win").is_dir() and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith("slb_xtransfer_"):
                superseded_shards.add(d.name)
    if (runs_dir / "slb_banking_l3_evenstride_6win").is_dir() and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and (d.name.startswith("slb_es_") or d.name.startswith("slb_es9_")):
                superseded_shards.add(d.name)
    if (runs_dir / "meta_reviewer_h60_flash_n6").is_dir() and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and (
                d.name.startswith("meta60_drift_s")
                or d.name.startswith("meta60_stat_s")
                or d.name == "meta60_w1_ckpt_flash"
            ):
                superseded_shards.add(d.name)
    if (runs_dir / "meta_reviewer_h60_hydrogen_n3").is_dir() and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and (d.name.startswith("meta60_hy_") or d.name == "meta60_w1_ckpt_hydro"):
                superseded_shards.add(d.name)
    if (runs_dir / "repeated_rule_replay_stage1_k8").is_dir() and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith("rep_stage1_"):
                superseded_shards.add(d.name)
    if (runs_dir / "meta_dip_c_h60_flash_n6").is_dir() and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and (
                d.name.startswith("meta60_dipc_")
                or d.name.startswith("meta60_s2_stat_")
                or d.name == "meta60_s2_w1_ckpt_flash"
            ):
                superseded_shards.add(d.name)
    if (runs_dir / "qwen8b_keep_or_apply_n8").is_dir() and runs_dir.is_dir():
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith("gate8b_s"):
                superseded_shards.add(d.name)

    if runs_dir.is_dir():
        for run_subdir in sorted(runs_dir.iterdir()):
            if not run_subdir.is_dir():
                continue
            run_id = run_subdir.name
            prog_file = run_subdir / "progress.json"
            if prog_file.is_file():
                try:
                    prog_obj = json.loads(prog_file.read_text(encoding="utf-8"))
                    if "run_id" not in prog_obj:
                        prog_obj["run_id"] = run_id
                    if prog_obj.get("status") == "running":
                        upd_raw = str(prog_obj.get("updated_at_utc", ""))
                        try:
                            age_s = (datetime.now(timezone.utc) - datetime.fromisoformat(upd_raw)).total_seconds()
                            if age_s > 2700:
                                prog_obj["status"] = "halted"
                        except ValueError:
                            pass
                    if run_id not in superseded_shards or prog_obj.get("status") == "running":
                        run_statuses.append(prog_obj)
                except json.JSONDecodeError:
                    pass

            cells_subdir = run_subdir / "cells"
            if cells_subdir.is_dir():
                for cfile in sorted(cells_subdir.glob("*.json")):
                    try:
                        cell_obj = json.loads(cfile.read_text(encoding="utf-8"))
                        if isinstance(cell_obj, dict):
                            summary_dict = cell_obj.get("summary")
                            if isinstance(summary_dict, dict):
                                summary_dict.setdefault("run_id", run_id)
                            cells.append(cell_obj)
                    except json.JSONDecodeError:
                        pass

            if run_id in superseded_shards:
                continue

            c_rep = run_subdir / "causal_fork_report.json"
            has_matrix_json = (run_subdir / "iter2_matrix_report.json").is_file() or (run_subdir / "online_matrix_report.json").is_file()
            if c_rep.is_file() and not has_matrix_json:
                try:
                    rep_obj = json.loads(c_rep.read_text(encoding="utf-8"))
                    rep_obj.setdefault("run_id", run_id)
                    causal_report_entries.append((c_rep.stat().st_mtime, rep_obj))
                except (OSError, json.JSONDecodeError):
                    pass

            # Pick up online matrix / SLB JSON reports without duplicating shared_fork_report.json
            for jfile in sorted(run_subdir.glob("*.json")):
                jname = jfile.name
                if (
                    jname
                    in (
                        "progress.json",
                        "causal_fork_report.json",
                        "shared_fork_report.json",
                        "w1_live_checkpoints.json",
                        "procedure_versions_audit.json",
                        "reviewer_versions_audit.json",
                        "counterfactual_w4_replay.json",
                        "repeated_replay_trials_full.json",
                        "keep_or_apply_streams_full.json",
                    )
                    or jname.startswith("checkpoint_")
                ):
                    continue
                try:
                    m_obj = json.loads(jfile.read_text(encoding="utf-8"))
                    if isinstance(m_obj, dict):
                        m_obj.setdefault("run_id", run_id)
                        m_obj.setdefault("report_file", jname)
                        if jname == "slb_mini_report.json" and run_id == "slb_banking_card_complete":
                            slb_mini_report = m_obj
                        elif jname == "slb_mini_report.json" and run_id == "slb_banking_l3_6win":
                            slb_l3_report = m_obj
                        else:
                            matrix_report_entries.append((jfile.stat().st_mtime, m_obj))
                except (OSError, json.JSONDecodeError):
                    pass

            if run_id.startswith("multidomain_h21_") or run_id in ("slb_banking_card_complete", "slb_banking_l3_6win"):
                continue

            sum_md = run_subdir / "summary.md"
            if sum_md.is_file():
                try:
                    md_text = sum_md.read_text(encoding="utf-8")[:12000]
                    run_summaries.append((sum_md.stat().st_mtime, {"run_id": run_id, "markdown": md_text}))
                except OSError:
                    pass

    # Aggregate Multi-Domain H=21 N=3 cells if present
    md_cells = [c.get("summary", {}) for c in cells if str(c.get("summary", {}).get("run_id", "")).startswith("multidomain_h21_")]
    multidomain_h21_summary: list[dict[str, Any]] = []
    if md_cells:
        envs_md = sorted({str(s.get("env_spec", "")) for s in md_cells if s.get("env_spec")})
        for espec in envs_md:
            row_methods: dict[str, dict[str, float]] = {}
            for h_key in ("blind", "skillbook", "srn", "sgeh"):
                sub = [s for s in md_cells if s.get("env_spec") == espec and s.get("harness_name") == h_key]
                if sub:
                    n_s = len(sub)
                    row_methods[h_key] = {
                        "n": n_s,
                        "post_shift_utility": round(sum(float(x.get("post_shift_utility", 0.0)) for x in sub) / n_s, 2),
                        "stale_lock_pct": round(100.0 * sum(float(x.get("stale_lock_in_rate", 0.0)) for x in sub) / n_s, 1),
                        "ref_calls": round(sum(float(x.get("total_reflection_calls", 0.0)) for x in sub) / n_s, 1),
                        "usd": round(sum(float(x.get("total_usd", 0.0)) for x in sub) / n_s, 4),
                    }
            if row_methods:
                multidomain_h21_summary.append({"env_spec": espec, "methods": row_methods})

    # Newest runs first so latest scientific iteration is prominent
    causal_reports = [entry[1] for entry in sorted(causal_report_entries, key=lambda x: x[0], reverse=True)]
    matrix_reports = [entry[1] for entry in sorted(matrix_report_entries, key=lambda x: x[0], reverse=True)]
    ordered_summaries = [entry[1] for entry in sorted(run_summaries, key=lambda x: x[0], reverse=True)]

    model_catalog = []
    for m_id in (
        "Qwen/Qwen3-8B",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.8-flash",
        "gemini-4-hydrogen",
        "gemma-4-31b-it",
        "gemini-4-neon",
        "gemini-4-argon",
        "gemini-4-carbon",
    ):
        rate = get_model_rate(m_id)
        model_catalog.append(
            {
                "model": m_id,
                "tier": rate.tier,
                "rank": rate.capability_rank,
                "input_per_1m": rate.input_per_1m,
                "output_per_1m": rate.output_per_1m,
            }
        )

    roadmap = [
        {
            "act": "Act 1: Causal State-Surgery & Mechanism Isolation",
            "status": "in_progress" if (cells or causal_reports) else "ready",
            "tag": "Empirical Proof",
            "goal": (
                "Isolate stale action-ranking inertia vs. historical expectation anchors (mu_hat) via shared-prefix "
                "forks on AlphaDesk-v0.2, anchored externally on ServeLearnBench Retail & Banking L1-L3."
            ),
        },
        {
            "act": "Act 2: Surprise-Gated Epistemic Harness (SGEH + Hazard)",
            "status": "in_progress"
            if any("sgeh" in str(c.get("summary", {}).get("harness_name", "")).lower() for c in cells)
            else "ready",
            "tag": "Engineered Baseline",
            "goal": (
                "Decouple permanent invariants from revisable scoped hypotheses; pair 0-token CUSUM + EWMA-slope "
                "quarantine with non-redundant untested-arm probing and Type-II staleness hazard clocks h(Delta t_a)."
            ),
        },
        {
            "act": "Act 3: On-Policy Weight Internalization (8B / 27B LoRA GRPO)",
            "status": "planned",
            "tag": "Flagship Contribution",
            "goal": (
                "Train open backbones (Qwen3-8B / Qwen4-27B LoRA) via 100% on-policy SFT -> Outcome-GRPO vs. "
                "Epistemic+Brier-GRPO on CSD3, proving zero-shot transfer across 3 Leave-One-Domain-Out folds."
            ),
        },
    ]

    legacy_debates = [
        {
            "topic": item["focus"],
            "codex_point": item["critique"],
            "resolution": item["resolution"],
        }
        for item in CHRONOLOGICAL_DEBATE_LOG
    ]

    return {
        "project": {
            "title": "Epistemic Plasticity under Non-Stationary Reward Paradigms",
            "subtitle": "Breaking the Self-Evolution Ratchet via Reference-Anchored Probing & Learned Meta-Plasticity",
            "target_venues": "ACL (January 2027) / ICML 2027",
            "host": "Quadro RTX 8000 (48 GB) · Cambridge CSD3 · Collaborator GPUs",
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        },
        "thesis_evolution": THESIS_EVOLUTION,
        "decisions_d1_d7": DECISIONS_D1_D7,
        "open_questions": OPEN_SUPERVISOR_QUESTIONS,
        "milestones": MILESTONES_TIMELINE,
        "cost_tiers": COST_TIERS,
        "roadmap": roadmap,
        "drift_taxonomy": DRIFT_TAXONOMY,
        "competitor_matrix": COMPETITOR_MATRIX,
        "alphadesk_v2_payoffs": ALPHADESK_V2_PAYOFF_TABLES,
        "alphadesk_v3_inheritance": ALPHADESK_V3_LITERATURE_INHERITANCE,
        "latin_square_counterbalancing": LATIN_SQUARE_COUNTERBALANCING,
        "environments": ENVIRONMENT_DOCS,
        "concrete_walkthroughs": CONCRETE_WALKTHROUGHS,
        "curated_flagship_traces": CURATED_FLAGSHIP_TRACES,
        "methods": METHOD_DOCS,
        "card_surgery_arms": CARD_SURGERY_ARMS,
        "act3_pipeline": ACT3_PIPELINE,
        "chronological_debates": CHRONOLOGICAL_DEBATE_LOG,
        "debates": legacy_debates,
        "checkpoint_samples": _collect_checkpoint_samples(runs_dir),
        "trace_samples": _collect_trace_samples(results_dir),
        "budget": budget_block,
        "model_catalog": model_catalog,
        "causal_reports": causal_reports,
        "matrix_reports": matrix_reports,
        "multidomain_h21_summary": multidomain_h21_summary,
        "slb_mini_report": slb_mini_report,
        "slb_l3_report": slb_l3_report,
        "run_summaries": ordered_summaries,
        "runs": sorted(run_statuses, key=lambda r: str(r.get("updated_at_utc", "")), reverse=True),
        "cells": cells,
    }
