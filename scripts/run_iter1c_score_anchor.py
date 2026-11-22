#!/usr/bin/env python3
"""Iteration 1c: Saved-Checkpoint Factorial Test of Historical Score Visibility vs. Action Ranking."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_causal_fork_v2 import _run_episode_range, bootstrap_mean_ci  # noqa: E402
from src.core.ledger import BudgetExceededError, CostLedger, atomic_write_json, send_monitor_email  # noqa: E402
from src.core.llm_client import MeteredLLMClient  # noqa: E402
from src.envs.alpha_desk_v2 import AlphaDeskV2Env  # noqa: E402
from src.harnesses.base import PublicEpisodeRecord  # noqa: E402
from src.harnesses.scoped_cards import ScopedCardBookHarness  # noqa: E402
from src.runner import _build_acting_system_prompt  # noqa: E402


def analyze_post_fork_probes(logs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute secondary probe-selection & calm-recovery diagnostics requested in peer review."""
    by_strat: Counter[str] = Counter()
    rejected_second_probes = 0
    first_any_probe: dict[str, Any] | None = None
    first_after_poor_calm: dict[str, Any] | None = None
    saw_poor_calm: dict[str, Any] | None = None
    calm_total = 0
    calm_optimal = 0
    calm_gamma_dominant = 0

    for row in logs:
        ep = int(row["episode"])
        is_hv = bool(row["is_high_vol"])
        steps = row.get("steps", [])

        for st in steps:
            if st.get("tool") == "probe_strategy_execution":
                obs = st.get("obs", {})
                if isinstance(obs, dict) and "live_net_yield_estimate" in obs:
                    strat = str(st.get("args", {}).get("strategy", "")).upper()
                    est = obs.get("live_net_yield_estimate")
                    by_strat[strat] += 1
                    p_info = {"episode": ep, "strategy": strat, "estimate": est, "is_high_vol": is_hv}
                    if first_any_probe is None:
                        first_any_probe = p_info
                    if saw_poor_calm is not None and first_after_poor_calm is None:
                        first_after_poor_calm = {
                            "trigger_calm_episode": saw_poor_calm["episode"],
                            "trigger_calm_score": saw_poor_calm["score"],
                            "probe_episode": ep,
                            "strategy": strat,
                            "estimate": est,
                            "non_alpha_probe": strat != "STRAT_ALPHA",
                        }
                elif isinstance(obs, dict) and "Limit 1 execution probe" in str(obs.get("error", "")):
                    rejected_second_probes += 1

        if not is_hv:
            calm_total += 1
            if row.get("optimal_policy_match"):
                calm_optimal += 1
            subs = [st.get("args", {}).get("weights", {}) for st in steps if st.get("tool") == "submit_allocation"]
            if subs and isinstance(subs[-1], dict) and subs[-1].get("STRAT_GAMMA", 0.0) >= 0.50:
                calm_gamma_dominant += 1
            if saw_poor_calm is None and float(row.get("public_score", 100.0)) < 50.0:
                saw_poor_calm = {"episode": ep, "score": float(row["public_score"])}

    return {
        "paid_probes_by_strategy": dict(by_strat),
        "rejected_extra_probe_attempts": rejected_second_probes,
        "first_post_fork_probe": first_any_probe,
        "first_probe_after_poor_calm": first_after_poor_calm,
        "calm_sessions": calm_total,
        "calm_optimal_matches": calm_optimal,
        "calm_gamma_dominant_sessions": calm_gamma_dominant,
    }


def load_saved_iter1b_checkpoint(
    *, results_root: Path, source_run_id: str, seed: int
) -> tuple[int, ScopedCardBookHarness, dict[str, Any]]:
    """Reconstruct exact pre-fork `ScopedCardBookHarness` and prefix logs from saved `iter1b` checkpoint."""
    ck_path = results_root / "runs" / source_run_id / f"checkpoint_seed_{seed}.json"
    ck = json.loads(ck_path.read_text(encoding="utf-8"))
    t_star = int(ck["t_star"])

    raw_candidates = sorted(
        (results_root / "raw" / source_run_id).glob(f"alpha_desk_v2_stationary__retained__*s{seed}.jsonl")
    )
    if not raw_candidates:
        raise FileNotFoundError(f"Missing saved stationary retained raw trace for seed {seed} in {source_run_id}")
    raw_lines = [json.loads(line) for line in raw_candidates[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    prefix_logs = raw_lines[:t_star]

    h = ScopedCardBookHarness(reflect_every=3)
    for idx, r in enumerate(prefix_logs):
        h._history.append(
            PublicEpisodeRecord(
                env_name="alpha_desk_v2",
                episode_idx=idx,
                timestamp=int(r.get("episode", idx + 1)),
                task_prompt=f"Session #{idx + 1}: Review today's desk brief and submit compliant portfolio weights to maximize realized session score (0-100).",
                steps=r.get("steps", []),
                public_score=float(r.get("public_score", 0.0)),
            )
        )
    h.cards = json.loads(json.dumps(ck["retained_cards"]))
    h.set_post_fork_schedule()

    pre_norm_u = [round(100.0 * float(r.get("normalized_utility", 0.0)), 1) for r in prefix_logs]
    pre_summary = {
        "logs": prefix_logs,
        "curve_utility": pre_norm_u,
        "mean_utility_pct": round(sum(pre_norm_u) / max(1, len(pre_norm_u)), 2),
        "acting_calls": sum(len(r.get("steps", [])) for r in prefix_logs),
        "reflection_calls": sum(int(r.get("harness_meta", {}).get("reflection_calls", 0)) for r in prefix_logs),
        "usd": 0.0,  # Reused from saved checkpoint at $0.00!
    }
    return t_star, h, pre_summary


def run_iter1c_experiment(
    *,
    run_id: str,
    seeds: list[int],
    model: str,
    post_window: int,
    max_usd: float,
    workers: int,
    thinking_budget: int,
    results_root: Path,
    dotenv_path: Path | None,
    source_run_id: str | None = "iter1b_clean_fork_n4",
) -> dict[str, Any]:
    ledger = CostLedger(
        results_root=results_root,
        run_id=run_id,
        phase="dev",
        max_usd=max_usd,
    )
    est_total_eps = len(seeds) * (6 * post_window if source_run_id else (14 + 6 * post_window))
    done_eps = 0
    prog_lock = threading.Lock()

    def _tick(label: str) -> None:
        nonlocal done_eps
        with prog_lock:
            done_eps += 1
            ledger.update_progress(
                completed_units=min(done_eps, est_total_eps),
                total_units=est_total_eps,
                status="running",
                current_label=label,
            )

    client = MeteredLLMClient(
        model=model,
        ledger=ledger,
        dotenv_path=dotenv_path,
        thinking_budget=thinking_budget,
        temperature=0.0,
    )

    traces_dir = ledger.results_root / "raw" / ledger.run_id
    traces_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = ledger.run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    per_seed_results: list[dict[str, Any]] = []
    arm_names = ("score_shown", "score_hidden", "ranking_retained")

    for seed in seeds:
        if source_run_id and (results_root / "runs" / source_run_id / f"checkpoint_seed_{seed}.json").is_file():
            t_star, base_harness, pre_res = load_saved_iter1b_checkpoint(
                results_root=results_root, source_run_id=source_run_id, seed=seed
            )
            fork_source = f"saved_checkpoint:{source_run_id}"
            print(f"\n[SEED {seed}] Loaded saved checkpoint ({source_run_id}) at t*={t_star} ($0.00 pre-shift)...", flush=True)
        else:
            pre_env = AlphaDeskV2Env(mode="stationary", horizon=40, seed=seed)
            t_star = pre_env.t_star
            base_harness = ScopedCardBookHarness(reflect_every=3)
            pre_res = _run_episode_range(
                env=pre_env,
                harness=base_harness,
                client=client,
                ep_indices=range(0, t_star),
                method_tag="scoped_cards_preshift",
                force_final_curate=True,
                on_ep_done=lambda s=seed: _tick(f"s{s} pre"),
            )
            base_harness.set_post_fork_schedule()
            fork_source = "fresh_acquisition"

        h_ranking_retained = base_harness.clone()
        h_score_shown = base_harness.make_weights_excised_score_shown()
        h_score_hidden = base_harness.make_weights_excised_score_hidden()

        fork_env = AlphaDeskV2Env(mode="stationary", horizon=40, seed=seed, t_star_override=t_star)
        fork_spec = fork_env.start_episode(t_star)
        rendered_prompts = {
            "score_shown": _build_acting_system_prompt(
                fork_spec.domain_prompt, fork_spec.tool_schemas, h_score_shown.build_context(fork_spec)
            ),
            "score_hidden": _build_acting_system_prompt(
                fork_spec.domain_prompt, fork_spec.tool_schemas, h_score_hidden.build_context(fork_spec)
            ),
            "ranking_retained": _build_acting_system_prompt(
                fork_spec.domain_prompt, fork_spec.tool_schemas, h_ranking_retained.build_context(fork_spec)
            ),
        }
        snapshot_record = {
            "seed": seed,
            "t_star": t_star,
            "fork_source": fork_source,
            "fork_offset": base_harness.fork_offset,
            "history_lengths_at_fork": {
                "score_shown": len(h_score_shown._history),
                "score_hidden": len(h_score_hidden._history),
                "ranking_retained": len(h_ranking_retained._history),
            },
            "score_shown_cards": h_score_shown.cards,
            "score_hidden_cards": h_score_hidden.cards,
            "ranking_retained_cards": h_ranking_retained.cards,
            "rendered_actor_system_prompts_at_fork": rendered_prompts,
            "context_char_lengths": {k: len(v) for k, v in rendered_prompts.items()},
        }
        atomic_write_json(ledger.run_dir / f"checkpoint_seed_{seed}.json", snapshot_record)
        print(
            f"  [FORK t*={t_star}] Prompt chars={snapshot_record['context_char_lengths']} "
            f"(score_shown - score_hidden = {len(rendered_prompts['score_shown']) - len(rendered_prompts['score_hidden']):+d} chars)",
            flush=True,
        )

        branch_defs = [
            ("abrupt_reversal", "score_shown", h_score_shown.clone()),
            ("abrupt_reversal", "score_hidden", h_score_hidden.clone()),
            ("abrupt_reversal", "ranking_retained", h_ranking_retained.clone()),
            ("stationary", "score_shown", h_score_shown.clone()),
            ("stationary", "score_hidden", h_score_hidden.clone()),
            ("stationary", "ranking_retained", h_ranking_retained.clone()),
        ]
        branch_outputs: dict[tuple[str, str], dict[str, Any]] = {}

        def _exec_branch(mode_name: str, arm_name: str, h_copy: ScopedCardBookHarness) -> tuple[str, str, dict[str, Any]]:
            b_env = AlphaDeskV2Env(mode=mode_name, horizon=40, seed=seed, t_star_override=t_star)
            res = _run_episode_range(
                env=b_env,
                harness=h_copy,
                client=client,
                ep_indices=range(t_star, t_star + post_window),
                method_tag=f"iter1c_{arm_name}",
                force_final_curate=False,
                on_ep_done=lambda s=seed, m=mode_name, a=arm_name: _tick(f"s{s} {m}:{a}"),
            )
            res["probe_diagnostics"] = analyze_post_fork_probes(res["logs"])
            return mode_name, arm_name, res

        with ThreadPoolExecutor(max_workers=max(1, min(6, workers))) as pool:
            futs = [pool.submit(_exec_branch, m, a, hc) for m, a, hc in branch_defs]
            for fut in as_completed(futs):
                m_name, a_name, b_res = fut.result()
                branch_outputs[(m_name, a_name)] = b_res
                pd = b_res["probe_diagnostics"]
                print(
                    f"    -> [{m_name:15s} | {a_name:16s}] Net={b_res['mean_utility_pct']:5.1f}% (Gross={b_res['gross_utility_pct']:5.1f}%) | "
                    f"Calm={b_res['calm_utility_pct']:5.1f}% | Probes={b_res['probes_total']:2d} ({pd['paid_probes_by_strategy']}) | "
                    f"Rej2nd={pd['rejected_extra_probe_attempts']} | CalmOpt={pd['calm_optimal_matches']}/{pd['calm_sessions']} | Run=${ledger.run_usd:.4f}",
                    flush=True,
                )

                cell_slug = f"alpha_desk_v2_{m_name}__{a_name}__{model.replace('/', '_')}_s{seed}"
                stitched_logs = pre_res["logs"] + b_res["logs"]
                with open(traces_dir / f"{cell_slug}.jsonl", "w", encoding="utf-8") as handle:
                    for row in stitched_logs:
                        handle.write(json.dumps(row) + "\n")

                total_calls = pre_res["acting_calls"] + b_res["acting_calls"] + pre_res["reflection_calls"] + b_res["reflection_calls"]
                ref_calls = pre_res["reflection_calls"] + b_res["reflection_calls"]
                cell_summary = {
                    "run_id": run_id,
                    "env_spec": f"alpha_desk_v2:{m_name}",
                    "harness_name": f"cards_{a_name}",
                    "model": model,
                    "seed": seed,
                    "episodes": t_star + post_window,
                    "mean_score": b_res["mean_score"],
                    "mean_normalized_utility": b_res["mean_utility_pct"],
                    "pre_shift_utility": pre_res["mean_utility_pct"],
                    "post_shift_utility": b_res["mean_utility_pct"],
                    "stale_lock_in_rate": b_res["stale_lock_in_pct"],
                    "invariant_violation_rate": b_res["invariant_violation_pct"],
                    "total_acting_calls": pre_res["acting_calls"] + b_res["acting_calls"],
                    "total_reflection_calls": ref_calls,
                    "rsi_tax_call_share": round(100.0 * ref_calls / max(1, total_calls), 2),
                    "total_usd": round(b_res["usd"], 6),
                    "reflection_usd": 0.0,
                }
                atomic_write_json(
                    cells_dir / f"{cell_slug}.json",
                    {
                        "summary": cell_summary,
                        "curve_utility": pre_res["curve_utility"] + b_res["curve_utility"],
                        "curve_score": [r["public_score"] for r in stitched_logs],
                        "alarms": [],
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )

        dr_shown_minus_hid = round(
            branch_outputs[("abrupt_reversal", "score_shown")]["mean_utility_pct"]
            - branch_outputs[("abrupt_reversal", "score_hidden")]["mean_utility_pct"],
            2,
        )
        dr_shown_minus_hid_gross = round(
            branch_outputs[("abrupt_reversal", "score_shown")]["gross_utility_pct"]
            - branch_outputs[("abrupt_reversal", "score_hidden")]["gross_utility_pct"],
            2,
        )
        st_shown_minus_hid = round(
            branch_outputs[("stationary", "score_shown")]["mean_utility_pct"]
            - branch_outputs[("stationary", "score_hidden")]["mean_utility_pct"],
            2,
        )
        dr_shown_minus_ret = round(
            branch_outputs[("abrupt_reversal", "score_shown")]["mean_utility_pct"]
            - branch_outputs[("abrupt_reversal", "ranking_retained")]["mean_utility_pct"],
            2,
        )
        st_shown_minus_ret = round(
            branch_outputs[("stationary", "score_shown")]["mean_utility_pct"]
            - branch_outputs[("stationary", "ranking_retained")]["mean_utility_pct"],
            2,
        )
        dr_hid_minus_ret = round(
            branch_outputs[("abrupt_reversal", "score_hidden")]["mean_utility_pct"]
            - branch_outputs[("abrupt_reversal", "ranking_retained")]["mean_utility_pct"],
            2,
        )

        per_seed_results.append(
            {
                "seed": seed,
                "t_star": t_star,
                "fork_source": fork_source,
                "pre_shift_utility_pct": pre_res["mean_utility_pct"],
                "abrupt_reversal": {
                    a: {k: v for k, v in branch_outputs[("abrupt_reversal", a)].items() if k != "logs"}
                    for a in arm_names
                },
                "stationary": {
                    a: {k: v for k, v in branch_outputs[("stationary", a)].items() if k != "logs"}
                    for a in arm_names
                },
                "paired_contrasts": {
                    "drift_score_shown_minus_score_hidden": dr_shown_minus_hid,
                    "drift_score_shown_minus_score_hidden_gross": dr_shown_minus_hid_gross,
                    "stationary_score_shown_minus_score_hidden": st_shown_minus_hid,
                    "did_score_shown_minus_score_hidden": round(dr_shown_minus_hid - st_shown_minus_hid, 2),
                    "drift_score_shown_minus_ranking_retained": dr_shown_minus_ret,
                    "stationary_score_shown_minus_ranking_retained": st_shown_minus_ret,
                    "drift_score_hidden_minus_ranking_retained": dr_hid_minus_ret,
                },
            }
        )

    def _ci(key: str) -> dict[str, Any]:
        m, lo, hi = bootstrap_mean_ci([r["paired_contrasts"][key] for r in per_seed_results])
        return {"mean": m, "ci95": [lo, hi]}

    c_primary_score = _ci("drift_score_shown_minus_score_hidden")
    c_primary_score_gross = _ci("drift_score_shown_minus_score_hidden_gross")
    c_stat_score = _ci("stationary_score_shown_minus_score_hidden")
    c_did_score = _ci("did_score_shown_minus_score_hidden")
    c_drift_rank_exc = _ci("drift_score_shown_minus_ranking_retained")
    c_stat_rank_exc = _ci("stationary_score_shown_minus_ranking_retained")
    c_drift_joint_exc = _ci("drift_score_hidden_minus_ranking_retained")

    aggregate = {
        "run_id": run_id,
        "experiment_type": "iter1c_score_visibility_factorial",
        "source_checkpoint_run": source_run_id,
        "model": model,
        "requested_thinking_budget": thinking_budget,
        "observed_run_input_tokens": ledger.run_input_tokens,
        "observed_run_output_tokens": ledger.run_output_tokens,
        "observed_run_thinking_tokens": ledger.run_thinking_tokens,
        "n_seeds": len(per_seed_results),
        "seeds": seeds,
        "post_window": post_window,
        "total_run_usd": round(ledger.run_usd, 6),
        "max_usd_cap": max_usd,
        "iter1c_contrasts_mean_and_95ci": {
            "drift_score_shown_minus_score_hidden": c_primary_score,
            "drift_score_shown_minus_score_hidden_gross": c_primary_score_gross,
            "stationary_score_shown_minus_score_hidden": c_stat_score,
            "did_score_shown_minus_score_hidden": c_did_score,
            "drift_score_shown_minus_ranking_retained": c_drift_rank_exc,
            "stationary_score_shown_minus_ranking_retained": c_stat_rank_exc,
            "drift_score_hidden_minus_ranking_retained": c_drift_joint_exc,
        },
        # Backward-compatible keys for dashboard causal contrast renderer:
        "contrasts_mean_and_95ci": {
            "drift_excised_minus_sham": c_primary_score,
            "drift_excised_minus_retained": c_drift_rank_exc,
            "stationary_excised_minus_sham": c_stat_score,
            "diff_in_diffs_drift_minus_stationary": c_did_score,
            "drift_retained_minus_sham": c_drift_joint_exc,
            "stationary_retained_minus_sham": c_stat_rank_exc,
        },
        "contrast_labels": {
            "drift_excised_minus_sham": "Drift: Score-Shown - Score-Hidden (Both Weights=null, Primary 1c Test)",
            "drift_excised_minus_retained": "Drift: Score-Shown - Ranking-Retained (Retiring Weights While Keeping Score)",
            "stationary_excised_minus_sham": "Stationary: Score-Shown - Score-Hidden (Both Weights=null)",
            "diff_in_diffs_drift_minus_stationary": "Diff-in-Diffs (Drift - Stationary): Score-Shown - Score-Hidden",
            "drift_retained_minus_sham": "Drift: Score-Hidden - Ranking-Retained (Joint Excision vs. Untouched)",
            "stationary_retained_minus_sham": "Stationary: Score-Shown - Ranking-Retained (Cost of Retiring Valid Weights)",
        },
        "per_seed": per_seed_results,
    }
    atomic_write_json(ledger.run_dir / "causal_fork_report.json", aggregate)
    ledger.update_progress(
        completed_units=est_total_eps,
        total_units=est_total_eps,
        status="completed",
        current_label="finished iter1c score-anchor fork",
    )
    return aggregate


def render_iter1c_markdown(agg: dict[str, Any]) -> str:
    c = agg["iter1c_contrasts_mean_and_95ci"]
    lines = [
        f"# Iteration 1c Saved-Checkpoint Score-Anchor Factorial Report: `{agg['run_id']}`",
        "",
        f"- **Forked From Saved Checkpoints:** `{agg.get('source_checkpoint_run')}` (`$0.00` pre-shift variance)",
        f"- **Model:** `{agg['model']}` | **Paired Seeds ($N$):** `{agg['n_seeds']}` (`{agg['seeds']}`) | **Post-Boundary Window:** `{agg['post_window']}` sessions",
        f"- **Requested `thinking_budget`:** `{agg['requested_thinking_budget']}` | **Observed Tokens:** `in={agg['observed_run_input_tokens']:,}`, `out={agg['observed_run_output_tokens']:,}`, `thinking={agg['observed_run_thinking_tokens']:,}`",
        f"- **Total Run Spend:** `${agg['total_run_usd']:.4f}` (Hard Cap: `${agg['max_usd_cap']:.2f}`)",
        "",
        "## 1. Seed-Paired Causal Contrasts (16-Session Post-Boundary Utility %)",
        "",
        "| Contrast | Mean Diff (pts) | 95% Bootstrap CI | Question Tested |",
        "|---|---:|---:|---|",
        f"| **Drift (`abrupt_reversal`): `Score-Shown − Score-Hidden`** | **`{c['drift_score_shown_minus_score_hidden']['mean']:+.2f}`** | `[{c['drift_score_shown_minus_score_hidden']['ci95'][0]:+.2f}, {c['drift_score_shown_minus_score_hidden']['ci95'][1]:+.2f}]` | **Primary 1-Variable Test:** With weights hidden & `n` identical, does historical calm score improve recovery? |",
        f"| **Drift (Pre-Fee Gross): `Score-Shown − Score-Hidden`** | `{c['drift_score_shown_minus_score_hidden_gross']['mean']:+.2f}` | `[{c['drift_score_shown_minus_score_hidden_gross']['ci95'][0]:+.2f}, {c['drift_score_shown_minus_score_hidden_gross']['ci95'][1]:+.2f}]` | Pre-probe-fee allocation quality difference (`Score-Shown − Score-Hidden`) |",
        f"| **Stationary Control: `Score-Shown − Score-Hidden`** | `{c['stationary_score_shown_minus_score_hidden']['mean']:+.2f}` | `[{c['stationary_score_shown_minus_score_hidden']['ci95'][0]:+.2f}, {c['stationary_score_shown_minus_score_hidden']['ci95'][1]:+.2f}]` | Effect of showing historical calm score in stationary world |",
        f"| **Diff-in-Diffs (`Drift − Stationary`): `Score-Shown − Score-Hidden`** | **`{c['did_score_shown_minus_score_hidden']['mean']:+.2f}`** | `[{c['did_score_shown_minus_score_hidden']['ci95'][0]:+.2f}, {c['did_score_shown_minus_score_hidden']['ci95'][1]:+.2f}]` | Regime-specific interaction of historical score visibility |",
        f"| **Drift (`abrupt_reversal`): `Score-Shown − Ranking-Retained`** | `{c['drift_score_shown_minus_ranking_retained']['mean']:+.2f}` | `[{c['drift_score_shown_minus_ranking_retained']['ci95'][0]:+.2f}, {c['drift_score_shown_minus_ranking_retained']['ci95'][1]:+.2f}]` | Effect of hiding stale action ranking while **preserving** historical score & `n` |",
        f"| **Stationary Control: `Score-Shown − Ranking-Retained`** | `{c['stationary_score_shown_minus_ranking_retained']['mean']:+.2f}` | `[{c['stationary_score_shown_minus_ranking_retained']['ci95'][0]:+.2f}, {c['stationary_score_shown_minus_ranking_retained']['ci95'][1]:+.2f}]` | Cost of hiding valid action ranking when historical score is preserved |",
        f"| **Drift (`abrupt_reversal`): `Score-Hidden − Ranking-Retained`** | `{c['drift_score_hidden_minus_ranking_retained']['mean']:+.2f}` | `[{c['drift_score_hidden_minus_ranking_retained']['ci95'][0]:+.2f}, {c['drift_score_hidden_minus_ranking_retained']['ci95'][1]:+.2f}]` | Joint hiding of weights + score vs. untouched memory |",
        "",
        "## 2. Per-Seed Breakdown & Secondary Probe-Selection Diagnostics (Reversal)",
        "",
        "| Seed | $t^*$ | `Score-Shown` Net (Probes / CalmOpt) | `Score-Hidden` Net (Probes / CalmOpt) | `Ranking-Retained` Net (Probes / CalmOpt) | `Shown − Hidden` | First Probe After Poor Calm (`Shown` vs `Hidden`) |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in agg["per_seed"]:
        dr = row["abrupt_reversal"]
        pc = row["paired_contrasts"]
        sh_p = dr["score_shown"]["probe_diagnostics"]
        hi_p = dr["score_hidden"]["probe_diagnostics"]
        re_p = dr["ranking_retained"]["probe_diagnostics"]
        fp_sh = sh_p["first_probe_after_poor_calm"]
        fp_hi = hi_p["first_probe_after_poor_calm"]
        fp_sh_txt = f"Ep{fp_sh['probe_episode']}:{fp_sh['strategy']}({fp_sh['estimate']})" if fp_sh else "none"
        fp_hi_txt = f"Ep{fp_hi['probe_episode']}:{fp_hi['strategy']}({fp_hi['estimate']})" if fp_hi else "none"
        lines.append(
            f"| `{row['seed']}` | `{row['t_star']}` | "
            f"**{dr['score_shown']['mean_utility_pct']:.1f}%** ({dr['score_shown']['probes_total']} / {sh_p['calm_optimal_matches']}/{sh_p['calm_sessions']}) | "
            f"{dr['score_hidden']['mean_utility_pct']:.1f}% ({dr['score_hidden']['probes_total']} / {hi_p['calm_optimal_matches']}/{hi_p['calm_sessions']}) | "
            f"{dr['ranking_retained']['mean_utility_pct']:.1f}% ({dr['ranking_retained']['probes_total']} / {re_p['calm_optimal_matches']}/{re_p['calm_sessions']}) | "
            f"**`{pc['drift_score_shown_minus_score_hidden']:+.1f}`** | "
            f"`{fp_sh_txt}` vs `{fp_hi_txt}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Iteration 1c Historical Score Visibility Experiment")
    parser.add_argument("--run-id", default="iter1c_score_anchor_n4", help="Run ID")
    parser.add_argument("--source-run-id", default="iter1b_clean_fork_n4", help="Saved checkpoint run ID")
    parser.add_argument("--seeds", default="101,102,103,104", help="Comma-separated seeds")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Model ID or 'mock'")
    parser.add_argument("--post-window", type=int, default=16, help="Post-boundary sessions scored")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Requested thinking budget")
    parser.add_argument("--workers", type=int, default=6, help="Parallel branch workers")
    parser.add_argument("--max-usd", type=float, default=1.95, help="Strict Tier-1 USD cap")
    parser.add_argument("--email-on-finish", action="store_true", help="Email report to gs06monitor@gmail.com")
    args = parser.parse_args()

    seed_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    try:
        agg = run_iter1c_experiment(
            run_id=args.run_id,
            seeds=seed_list,
            model=args.model,
            post_window=args.post_window,
            max_usd=args.max_usd,
            workers=args.workers,
            thinking_budget=args.thinking_budget,
            results_root=ROOT / "results",
            dotenv_path=ROOT / ".env",
            source_run_id=args.source_run_id,
        )
    except BudgetExceededError as exc:
        print(f"[BUDGET GUARDRAIL HALT] {exc}", flush=True)
        sys.exit(2)

    md = render_iter1c_markdown(agg)
    out_md = ROOT / "results" / "runs" / args.run_id / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\n{md}\nSaved report: {out_md}", flush=True)

    if args.email_on_finish:
        send_monitor_email(
            subject=f"Iter1c Score Anchor ({args.run_id}) - ${agg['total_run_usd']:.4f}",
            body=md,
        )


if __name__ == "__main__":
    main()
