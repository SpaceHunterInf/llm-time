#!/usr/bin/env python3
"""Iteration 2: Counterbalanced 4-Method x 6-Regime Online Adaptation & Calibration Benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import threading
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_causal_fork_v2 import bootstrap_mean_ci  # noqa: E402
from src.core.ledger import BudgetExceededError, CostLedger, atomic_write_json, send_monitor_email  # noqa: E402
from src.core.llm_client import MeteredLLMClient  # noqa: E402
from src.envs.alpha_desk_v2 import AlphaDeskV2Env, STRATEGIES_V2  # noqa: E402
from src.harnesses import make_harness  # noqa: E402
from src.harnesses.base import PublicEpisodeRecord  # noqa: E402
from src.runner import _build_acting_system_prompt  # noqa: E402

METHODS_ITER2 = ("scoped_cards", "periodic_probe", "sgeh_surprise", "sgeh_hazard")
FAMILY_A_MODES = ("stationary", "abrupt_reversal", "gradual_decay", "off_scope_shock")
FAMILY_B_MODES = ("opportunity_stationary", "opportunity_rise")
ALL_MODES_ITER2 = FAMILY_A_MODES + FAMILY_B_MODES


def _mock_iter2_actor(steps_so_far: list[dict[str, Any]], sys_prompt: str) -> dict[str, Any]:
    if not steps_so_far:
        return {"tool": "inspect_desk_brief", "args": {}, "rationale": "Inspect session brief."}
    first_obs = steps_so_far[0].get("obs", {})
    is_hv = first_obs.get("volatility_regime_hint") == "elevated_volatility"
    is_freeze = first_obs.get("liquidity_status") == "acute_liquidity_freeze"
    if not is_hv and not is_freeze and len(steps_so_far) == 1:
        for s in STRATEGIES_V2:
            if f"on untested `{s}`" in sys_prompt or f"on long-untested `{s}`" in sys_prompt or f"on `{s}`" in sys_prompt:
                return {"tool": "probe_strategy_execution", "args": {"strategy": s}, "rationale": "Scheduled/diagnostic probe."}
    for st in steps_so_far:
        obs = st.get("obs", {})
        if isinstance(obs, dict) and float(obs.get("live_net_yield_estimate", 0.0)) >= 80.0:
            win_s = str(obs.get("strategy", "STRAT_ALPHA"))
            other_s = next(x for x in STRATEGIES_V2 if x != win_s)
            w = {x: (0.65 if x == win_s else (0.35 if x == other_s else 0.0)) for x in STRATEGIES_V2}
            return {"tool": "submit_allocation", "args": {"weights": w}, "rationale": "Allocate to high probed yield."}
    w = {"STRAT_ALPHA": 0.65, "STRAT_BETA": 0.35, "STRAT_GAMMA": 0.0, "STRAT_DELTA": 0.0}
    return {"tool": "submit_allocation", "args": {"weights": w}, "rationale": "Default compliant weights."}


def _calm_ref_event_ge_75(env: AlphaDeskV2Env, ep: int, w_ref: dict[str, float]) -> int | None:
    """Return fixed-target ground-truth binary indicator `I[E[R_calm(w_ref)] >= 75.0]` on normal calm sessions."""
    ctx = env._episodes[ep]
    if ctx["is_high_vol"] or env._is_liquidity_freeze(ep):
        return None
    active_p, _, _ = env._payoffs(ep)
    exp_score = sum(float(w_ref.get(s, 0.0)) * float(active_p[s]) for s in STRATEGIES_V2)
    return 1 if exp_score >= 75.0 else 0


def run_episode_slice_iter2(
    *,
    env: AlphaDeskV2Env,
    harness: Any,
    client: MeteredLLMClient,
    ep_indices: range,
    method_tag: str,
    w_ref: dict[str, float],
    force_final_curate: bool = False,
    on_ep_done: Any = None,
) -> dict[str, Any]:
    logs: list[dict[str, Any]] = []
    acting_calls = 0
    reflection_calls = 0
    acting_usd = 0.0
    reflection_usd = 0.0
    acting_thinking_tokens = 0
    alarms_fired = 0
    brier_losses: list[float] = []
    ep_list = list(ep_indices)

    for idx_in_slice, ep in enumerate(ep_list):
        # Fixed-target Brier forecast BEFORE observing episode outcome:
        gt_event = _calm_ref_event_ge_75(env, ep, w_ref)
        pred_p = (
            float(harness.predict_calm_portfolio_prob_ge_75(w_ref))
            if hasattr(harness, "predict_calm_portfolio_prob_ge_75")
            else 0.5
        )
        if gt_event is not None:
            brier_losses.append((pred_p - float(gt_event)) ** 2)

        spec = env.start_episode(ep)
        h_ctx = harness.build_context(spec)
        sys_prompt = _build_acting_system_prompt(spec.domain_prompt, spec.tool_schemas, h_ctx)

        steps: list[dict[str, Any]] = []
        done = False
        final_score = 0.0
        priv: dict[str, Any] = {}

        for turn in range(env.max_turns_per_episode):
            hist_str = (
                "\n".join(
                    f"Turn {idx + 1}: called `{s['tool']}`({json.dumps(s['args'])}) -> {json.dumps(s['obs'])}"
                    for idx, s in enumerate(steps)
                )
                or "(No tools called yet in this session.)"
            )
            rem = env.max_turns_per_episode - turn
            must_commit = " FINAL TURN: Call `submit_allocation` now!" if rem == 1 else ""
            usr_prompt = (
                f"{spec.task_prompt}\n\n"
                f"Session Tool History:\n{hist_str}\n\n"
                f"Turns remaining: {rem}.{must_commit}"
            )
            resp = client.generate_json(
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                environment=spec.env_name,
                method=method_tag,
                mock_responder=lambda _sp, _up, _st=list(steps), _sys=sys_prompt: _mock_iter2_actor(_st, _sys),
            )
            acting_calls += 1
            acting_usd += resp.usd
            acting_thinking_tokens += int(resp.thinking_tokens)

            parsed = resp.parsed_json or {}
            t_name = str(parsed.get("tool", "")).strip()
            t_args = parsed.get("args", {})
            if not isinstance(t_args, dict):
                t_args = {}

            outcome = env.step_tool(t_name, t_args)
            steps.append(
                {
                    "turn": turn + 1,
                    "tool": t_name,
                    "args": t_args,
                    "rationale": str(parsed.get("rationale", ""))[:220],
                    "obs": outcome.observation,
                }
            )
            if outcome.done:
                done = True
                final_score = float(outcome.public_score or 0.0)
                priv = outcome.private_eval
                break

        if not done:
            final_score = 0.0
            priv = {
                "regime": "timeout",
                "is_high_vol": False,
                "expected_score": 0.0,
                "oracle_expected_score": 90.0,
                "normalized_utility": 0.0,
                "stale_policy_match": False,
                "optimal_policy_match": False,
                "invariant_violation": True,
                "probes_used": 0,
            }

        pub_rec = PublicEpisodeRecord(
            env_name=spec.env_name,
            episode_idx=ep,
            timestamp=ep + 1,
            task_prompt=spec.task_prompt,
            steps=steps,
            public_score=final_score,
        )
        is_last = force_final_curate and (idx_in_slice == len(ep_list) - 1)
        h_meta = harness.on_episode_complete(pub_rec, client, force_curate=is_last)
        reflection_calls += int(h_meta.get("reflection_calls", 0))
        reflection_usd += float(h_meta.get("reflection_usd", 0.0))
        if h_meta.get("alarm_fired"):
            alarms_fired += 1

        exp_sc = float(priv.get("expected_score", 0.0))
        orc_sc = float(priv.get("oracle_expected_score", 90.0))
        prb_used = int(priv.get("probes_used", 0))
        inv_viol = bool(priv.get("invariant_violation", False))
        gross_exp = 0.0 if inv_viol else min(100.0, exp_sc + 12.0 * prb_used)
        gross_norm_u = round(gross_exp / max(1.0, orc_sc), 4)

        logs.append(
            {
                "episode": ep + 1,
                "public_score": final_score,
                "normalized_utility": float(priv.get("normalized_utility", 0.0)),
                "gross_normalized_utility": gross_norm_u,
                "stale_policy_match": bool(priv.get("stale_policy_match", False)),
                "optimal_policy_match": bool(priv.get("optimal_policy_match", False)),
                "invariant_violation": inv_viol,
                "is_high_vol": bool(priv.get("is_high_vol", False)),
                "regime": str(priv.get("regime", "")),
                "probes_used": prb_used,
                "brier_pred_p": pred_p,
                "brier_gt_event": gt_event,
                "harness_meta": h_meta,
                "steps": steps,
            }
        )
        if on_ep_done is not None:
            on_ep_done()

    norm_u = [100.0 * r["normalized_utility"] for r in logs]
    gross_u = [100.0 * r["gross_normalized_utility"] for r in logs]
    calm_rows = [r for r in logs if not r["is_high_vol"] and r["regime"] != "off_scope_liquidity_freeze"]
    hv_rows = [r for r in logs if r["is_high_vol"]]
    return {
        "episodes": len(logs),
        "mean_score": round(statistics.mean(r["public_score"] for r in logs), 2) if logs else 0.0,
        "mean_utility_pct": round(statistics.mean(norm_u), 2) if norm_u else 0.0,
        "gross_utility_pct": round(statistics.mean(gross_u), 2) if gross_u else 0.0,
        "calm_utility_pct": round(100.0 * statistics.mean(r["normalized_utility"] for r in calm_rows), 2) if calm_rows else 0.0,
        "calm_optimal_pct": round(100.0 * sum(1 for r in calm_rows if r["optimal_policy_match"]) / max(1, len(calm_rows)), 2),
        "high_vol_hedge_utility_pct": round(100.0 * statistics.mean(r["normalized_utility"] for r in hv_rows), 2) if hv_rows else 0.0,
        "stale_lock_in_pct": round(100.0 * sum(1 for r in logs if r["stale_policy_match"]) / max(1, len(logs)), 2),
        "invariant_violation_pct": round(100.0 * sum(1 for r in logs if r["invariant_violation"]) / max(1, len(logs)), 2),
        "fixed_target_brier": round(statistics.mean(brier_losses), 4) if brier_losses else 0.0,
        "alarms_fired": alarms_fired,
        "probes_total": sum(r["probes_used"] for r in logs),
        "acting_calls": acting_calls,
        "reflection_calls": reflection_calls,
        "acting_thinking_tokens": acting_thinking_tokens,
        "usd": round(acting_usd + reflection_usd, 6),
        "curve_utility": [round(x, 1) for x in norm_u],
        "logs": logs,
    }


def run_iter2_batch(
    *,
    run_id: str,
    seeds: list[int],
    methods: tuple[str, ...] = METHODS_ITER2,
    model: str = "gemini-3.8-flash",
    post_window: int = 16,
    max_usd: float = 1.95,
    workers: int = 6,
    thinking_budget: int = 0,
    results_root: Path,
    dotenv_path: Path | None,
) -> dict[str, Any]:
    ledger = CostLedger(
        results_root=results_root,
        run_id=run_id,
        phase="dev",
        max_usd=max_usd,
    )
    est_total_eps = len(seeds) * len(methods) * (2 * 14 + len(ALL_MODES_ITER2) * post_window)
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

    per_seed_data: list[dict[str, Any]] = []

    for seed in seeds:
        probe_env = AlphaDeskV2Env(mode="stationary", horizon=40, seed=seed, counterbalance=True)
        t_star = probe_env.t_star
        role_meta = probe_env.private_role_map()
        w_ref = role_meta["reference_incumbent_weights"]
        print(
            f"\n[SEED {seed} | t*={t_star}] Counterbalanced roles: inc_1={role_meta['inc_1']}, "
            f"hv_hedge={role_meta['hv_hedge']}, unheld={role_meta['unheld_alt']}, "
            f"hedge_overlaps_winner={role_meta['hedge_overlaps_calm_winner']}",
            flush=True,
        )

        # Step 1: Run shared pre-shift prefixes per (method, family) in parallel
        prefix_jobs = []
        for m_name in methods:
            prefix_jobs.append(("family_a", "stationary", m_name))
            prefix_jobs.append(("family_b", "opportunity_stationary", m_name))

        prefixes: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}

        def _run_pre(fam_tag: str, pre_mode: str, meth: str) -> tuple[str, str, Any, dict[str, Any]]:
            p_env = AlphaDeskV2Env(mode=pre_mode, horizon=40, seed=seed, t_star_override=t_star, counterbalance=True)
            h_inst = make_harness(meth)
            res = run_episode_slice_iter2(
                env=p_env,
                harness=h_inst,
                client=client,
                ep_indices=range(0, t_star),
                method_tag=f"{meth}_pre",
                w_ref=w_ref,
                force_final_curate=True,
                on_ep_done=lambda s=seed, m=meth: _tick(f"s{s} pre:{m}"),
            )
            if hasattr(h_inst, "set_post_fork_schedule"):
                h_inst.set_post_fork_schedule()
            return fam_tag, meth, h_inst, res

        with ThreadPoolExecutor(max_workers=max(1, min(6, workers))) as pool:
            for fut in as_completed([pool.submit(_run_pre, ft, pm, mn) for ft, pm, mn in prefix_jobs]):
                ft, mn, h_done, p_res = fut.result()
                prefixes[(ft, mn)] = (h_done, p_res)
                print(
                    f"  [PRE {ft} | {mn:14s}] Util={p_res['mean_utility_pct']:5.1f}% | "
                    f"Probes={p_res['probes_total']} | Ref={p_res['reflection_calls']} | Run=${ledger.run_usd:.4f}",
                    flush=True,
                )

        # Step 2: Fork all 6 post-boundary regimes per method in parallel
        branch_tasks = []
        for m_name in methods:
            h_a, pre_a = prefixes[("family_a", m_name)]
            for mode_a in FAMILY_A_MODES:
                branch_tasks.append((mode_a, m_name, h_a.clone(), pre_a, len(FAMILY_A_MODES)))
            h_b, pre_b = prefixes[("family_b", m_name)]
            for mode_b in FAMILY_B_MODES:
                branch_tasks.append((mode_b, m_name, h_b.clone(), pre_b, len(FAMILY_B_MODES)))

        seed_matrix: dict[str, dict[str, Any]] = {mode: {} for mode in ALL_MODES_ITER2}

        def _run_post(
            mode_name: str, meth: str, h_clone: Any, pre_r: dict[str, Any], fam_divisor: int
        ) -> tuple[str, str, dict[str, Any], dict[str, Any], int]:
            b_env = AlphaDeskV2Env(mode=mode_name, horizon=40, seed=seed, t_star_override=t_star, counterbalance=True)
            b_res = run_episode_slice_iter2(
                env=b_env,
                harness=h_clone,
                client=client,
                ep_indices=range(t_star, t_star + post_window),
                method_tag=meth,
                w_ref=w_ref,
                force_final_curate=False,
                on_ep_done=lambda s=seed, md=mode_name, mt=meth: _tick(f"s{s} {md}:{mt}"),
            )
            return mode_name, meth, b_res, pre_r, fam_divisor

        with ThreadPoolExecutor(max_workers=max(1, min(6, workers))) as pool:
            futs = [pool.submit(_run_post, md, mt, hc, pr, dv) for md, mt, hc, pr, dv in branch_tasks]
            for fut in as_completed(futs):
                md_name, mt_name, b_res, pre_r, fam_div = fut.result()
                seed_matrix[md_name][mt_name] = {k: v for k, v in b_res.items() if k != "logs"}
                print(
                    f"    -> [{md_name:22s} | {mt_name:14s}] Net={b_res['mean_utility_pct']:5.1f}% "
                    f"(CalmOpt={b_res['calm_optimal_pct']:5.1f}%) | Brier={b_res['fixed_target_brier']:.3f} | "
                    f"Alarms={b_res['alarms_fired']} | Probes={b_res['probes_total']:2d} | Ref={b_res['reflection_calls']} | "
                    f"Run=${ledger.run_usd:.4f}",
                    flush=True,
                )

                cell_slug = f"alpha_desk_v2_cb_{md_name}__{mt_name}__{model.replace('/', '_')}_s{seed}"
                stitched_logs = pre_r["logs"] + b_res["logs"]
                with open(traces_dir / f"{cell_slug}.jsonl", "w", encoding="utf-8") as handle:
                    for row in stitched_logs:
                        handle.write(json.dumps(row) + "\n")

                tot_calls = pre_r["acting_calls"] + b_res["acting_calls"] + pre_r["reflection_calls"] + b_res["reflection_calls"]
                ref_calls = pre_r["reflection_calls"] + b_res["reflection_calls"]
                cell_summary = {
                    "run_id": run_id,
                    "env_spec": f"alpha_desk_v2:{md_name}",
                    "harness_name": mt_name,
                    "model": model,
                    "seed": seed,
                    "episodes": t_star + post_window,
                    "mean_score": b_res["mean_score"],
                    "mean_normalized_utility": b_res["mean_utility_pct"],
                    "pre_shift_utility": pre_r["mean_utility_pct"],
                    "post_shift_utility": b_res["mean_utility_pct"],
                    "stale_lock_in_rate": b_res["stale_lock_in_pct"],
                    "invariant_violation_rate": b_res["invariant_violation_pct"],
                    "fixed_target_brier": b_res["fixed_target_brier"],
                    "alarms_fired": b_res["alarms_fired"],
                    "total_acting_calls": pre_r["acting_calls"] + b_res["acting_calls"],
                    "total_reflection_calls": ref_calls,
                    "rsi_tax_call_share": round(100.0 * ref_calls / max(1, tot_calls), 2),
                    "total_usd": round(pre_r["usd"] / fam_div + b_res["usd"], 6),
                    "reflection_usd": 0.0,
                }
                atomic_write_json(
                    cells_dir / f"{cell_slug}.json",
                    {
                        "summary": cell_summary,
                        "curve_utility": pre_r["curve_utility"] + b_res["curve_utility"],
                        "curve_score": [r["public_score"] for r in stitched_logs],
                        "alarms": [
                            r["episode"] for r in stitched_logs if r.get("harness_meta", {}).get("alarm_fired")
                        ],
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )

        per_seed_data.append(
            {
                "seed": seed,
                "t_star": t_star,
                "role_map": role_meta,
                "modes": seed_matrix,
            }
        )

    rep = summarize_iter2_seeds(
        run_id=run_id,
        model=model,
        thinking_budget=thinking_budget,
        post_window=post_window,
        per_seed_data=per_seed_data,
        total_usd=round(ledger.run_usd, 6),
        max_usd=max_usd,
        in_tok=ledger.run_input_tokens,
        out_tok=ledger.run_output_tokens,
        think_tok=ledger.run_thinking_tokens,
    )
    atomic_write_json(ledger.run_dir / "iter2_matrix_report.json", rep)
    ledger.update_progress(
        completed_units=est_total_eps,
        total_units=est_total_eps,
        status="completed",
        current_label="finished iter2 online matrix",
    )
    return rep


def summarize_iter2_seeds(
    *,
    run_id: str,
    model: str,
    thinking_budget: int,
    post_window: int,
    per_seed_data: list[dict[str, Any]],
    total_usd: float,
    max_usd: float,
    in_tok: int,
    out_tok: int,
    think_tok: int,
) -> dict[str, Any]:
    by_mode_summary: dict[str, Any] = {}
    paired_contrasts: dict[str, Any] = {}

    for mode in ALL_MODES_ITER2:
        by_mode_summary[mode] = {}
        for meth in METHODS_ITER2:
            rows = [s["modes"][mode][meth] for s in per_seed_data if meth in s["modes"].get(mode, {})]
            if not rows:
                continue
            u_mean, u_lo, u_hi = bootstrap_mean_ci([r["mean_utility_pct"] for r in rows])
            by_mode_summary[mode][meth] = {
                "mean_utility_pct": u_mean,
                "ci95": [u_lo, u_hi],
                "calm_optimal_pct": round(statistics.mean(r["calm_optimal_pct"] for r in rows), 2),
                "fixed_target_brier": round(statistics.mean(r["fixed_target_brier"] for r in rows), 4),
                "mean_probes": round(statistics.mean(r["probes_total"] for r in rows), 2),
                "mean_alarms": round(statistics.mean(r["alarms_fired"] for r in rows), 2),
                "mean_reflection_calls": round(statistics.mean(r["reflection_calls"] for r in rows), 2),
            }

        # Key paired differences on this regime
        paired_contrasts[mode] = {}
        for a, b, cname in (
            ("sgeh_hazard", "scoped_cards", "sgeh_hazard_minus_scoped_cards"),
            ("sgeh_hazard", "sgeh_surprise", "sgeh_hazard_minus_sgeh_surprise"),
            ("sgeh_hazard", "periodic_probe", "sgeh_hazard_minus_periodic_probe"),
            ("sgeh_surprise", "scoped_cards", "sgeh_surprise_minus_scoped_cards"),
        ):
            diffs = [
                round(s["modes"][mode][a]["mean_utility_pct"] - s["modes"][mode][b]["mean_utility_pct"], 2)
                for s in per_seed_data
                if a in s["modes"].get(mode, {}) and b in s["modes"].get(mode, {})
            ]
            if diffs:
                m, lo, hi = bootstrap_mean_ci(diffs)
                paired_contrasts[mode][cname] = {"mean": m, "ci95": [lo, hi], "per_seed": diffs}

    return {
        "run_id": run_id,
        "experiment_type": "iter2_counterbalanced_online_matrix",
        "model": model,
        "requested_thinking_budget": thinking_budget,
        "observed_run_input_tokens": in_tok,
        "observed_run_output_tokens": out_tok,
        "observed_run_thinking_tokens": think_tok,
        "n_seeds": len(per_seed_data),
        "seeds": [s["seed"] for s in per_seed_data],
        "post_window": post_window,
        "total_run_usd": total_usd,
        "max_usd_cap": max_usd,
        "by_mode_summary": by_mode_summary,
        "paired_contrasts": paired_contrasts,
        "per_seed": per_seed_data,
    }


def render_iter2_markdown(rep: dict[str, Any]) -> str:
    lines = [
        f"# Iteration 2 Counterbalanced Online Matrix Report: `{rep['run_id']}`",
        "",
        f"- **Model:** `{rep['model']}` | **Counterbalanced Seeds ($N$):** `{rep['n_seeds']}` (`{rep['seeds']}`) | **Post-Boundary Window:** `{rep['post_window']}` sessions",
        f"- **Requested `thinking_budget`:** `{rep['requested_thinking_budget']}` | **Observed Tokens:** `in={rep['observed_run_input_tokens']:,}`, `out={rep['observed_run_output_tokens']:,}`, `thinking={rep['observed_run_thinking_tokens']:,}`",
        f"- **Total Spend:** `${rep['total_run_usd']:.4f}`",
        "",
        "## 1. Post-Boundary Net Normalized Utility (%) & Fixed-Target Brier Calibration",
        "",
        "| Regime | `ScopedCardBook` (Brier / Ref) | `Periodic-Probe` (Brier / Probes) | `SGEH-SurpriseOnly` (Brier / Alarms) | `SGEH+Hazard` (Brier / Probes) | `SGEH+Hazard − ScopedCards` |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ALL_MODES_ITER2:
        ms = rep["by_mode_summary"].get(mode, {})
        if not ms:
            continue
        sc = ms.get("scoped_cards", {})
        pp = ms.get("periodic_probe", {})
        ss = ms.get("sgeh_surprise", {})
        sh = ms.get("sgeh_hazard", {})
        diff = rep["paired_contrasts"].get(mode, {}).get("sgeh_hazard_minus_scoped_cards", {"mean": 0.0, "ci95": [0.0, 0.0]})
        lines.append(
            f"| `{mode}` | "
            f"{sc.get('mean_utility_pct', 0):.1f}% ({sc.get('fixed_target_brier', 0):.3f} / {sc.get('mean_reflection_calls', 0):.1f}) | "
            f"{pp.get('mean_utility_pct', 0):.1f}% ({pp.get('fixed_target_brier', 0):.3f} / {pp.get('mean_probes', 0):.1f}) | "
            f"{ss.get('mean_utility_pct', 0):.1f}% ({ss.get('fixed_target_brier', 0):.3f} / {ss.get('mean_alarms', 0):.1f}) | "
            f"**{sh.get('mean_utility_pct', 0):.1f}%** ({sh.get('fixed_target_brier', 0):.3f} / {sh.get('mean_probes', 0):.1f}) | "
            f"**`{diff['mean']:+.2f}`** `[{diff['ci95'][0]:+.2f}, {diff['ci95'][1]:+.2f}]` |"
        )
    return "\n".join(lines) + "\n"


def aggregate_existing_iter2_runs(results_root: Path, run_ids: list[str], out_run_id: str) -> Path:
    merged_seeds: list[dict[str, Any]] = []
    tot_usd = 0.0
    in_t = out_t = th_t = 0
    model = "gemini-3.8-flash"
    tb = 0
    pw = 16
    for rid in run_ids:
        p = results_root / "runs" / rid / "iter2_matrix_report.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        merged_seeds.extend(d["per_seed"])
        tot_usd += float(d["total_run_usd"])
        in_t += int(d["observed_run_input_tokens"])
        out_t += int(d["observed_run_output_tokens"])
        th_t += int(d["observed_run_thinking_tokens"])
        model = d["model"]
        tb = int(d["requested_thinking_budget"])
        pw = int(d["post_window"])

    agg = summarize_iter2_seeds(
        run_id=out_run_id,
        model=model,
        thinking_budget=tb,
        post_window=pw,
        per_seed_data=merged_seeds,
        total_usd=round(tot_usd, 6),
        max_usd=round(1.95 * len(run_ids), 2),
        in_tok=in_t,
        out_tok=out_t,
        think_tok=th_t,
    )
    out_dir = results_root / "runs" / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "iter2_matrix_report.json", agg)

    # Also emit a `causal_fork_report.json` view so the dashboard Causal Contrasts panel displays Iteration 2 automatically!
    pc = agg["paired_contrasts"]
    rev_c = pc["abrupt_reversal"]["sgeh_hazard_minus_scoped_cards"]
    opp_c = pc["opportunity_rise"]["sgeh_hazard_minus_sgeh_surprise"]
    stat_c = pc["stationary"]["sgeh_hazard_minus_scoped_cards"]
    did_c = bootstrap_mean_ci(
        [
            round(a - b, 2)
            for a, b in zip(rev_c["per_seed"], stat_c["per_seed"])
        ]
    )
    grad_c = pc["gradual_decay"]["sgeh_hazard_minus_scoped_cards"]
    shock_c = pc["off_scope_shock"]["sgeh_hazard_minus_scoped_cards"]
    compat = {
        "run_id": out_run_id,
        "experiment_type": "iter2_counterbalanced_online_matrix",
        "model": model,
        "requested_thinking_budget": tb,
        "observed_run_input_tokens": in_t,
        "observed_run_output_tokens": out_t,
        "observed_run_thinking_tokens": th_t,
        "n_seeds": agg["n_seeds"],
        "seeds": agg["seeds"],
        "post_window": pw,
        "total_run_usd": round(tot_usd, 6),
        "max_usd_cap": round(1.95 * len(run_ids), 2),
        "contrasts_mean_and_95ci": {
            "drift_excised_minus_sham": rev_c,
            "drift_excised_minus_retained": opp_c,
            "stationary_excised_minus_sham": stat_c,
            "diff_in_diffs_drift_minus_stationary": {"mean": did_c[0], "ci95": [did_c[1], did_c[2]]},
            "drift_retained_minus_sham": grad_c,
            "stationary_retained_minus_sham": shock_c,
        },
        "contrast_labels": {
            "drift_excised_minus_sham": "Abrupt Reversal: SGEH+Hazard - ScopedCardBook (Counterbalanced N=4)",
            "drift_excised_minus_retained": "Type-II Opportunity Rise: SGEH+Hazard - SGEH-SurpriseOnly (Hazard Ablation)",
            "stationary_excised_minus_sham": "Stationary Control: SGEH+Hazard - ScopedCardBook",
            "diff_in_diffs_drift_minus_stationary": "Diff-in-Diffs (Reversal - Stationary): SGEH+Hazard - ScopedCardBook",
            "drift_retained_minus_sham": "Gradual Decay: SGEH+Hazard - ScopedCardBook",
            "stationary_retained_minus_sham": "Off-Scope Liquidity Freeze: SGEH+Hazard - ScopedCardBook",
        },
        "per_seed": merged_seeds,
    }
    atomic_write_json(out_dir / "causal_fork_report.json", compat)
    md = render_iter2_markdown(agg)
    out_md = out_dir / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\n{md}\nSaved merged Iteration 2 report to {out_md}", flush=True)
    return out_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Iteration 2 Counterbalanced Online Adaptation Matrix")
    parser.add_argument("--run-id", default="iter2_cb_s201", help="Run ID")
    parser.add_argument("--seeds", default="201", help="Comma-separated counterbalanced seeds")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Model ID or 'mock'")
    parser.add_argument("--post-window", type=int, default=16, help="Post-boundary sessions scored")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Requested thinking budget")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers")
    parser.add_argument("--max-usd", type=float, default=1.95, help="Strict Tier-1 USD cap per run")
    parser.add_argument("--aggregate-runs", default="", help="Optional comma-separated run IDs to aggregate")
    parser.add_argument("--email-on-finish", action="store_true", help="Send milestone email")
    args = parser.parse_args()

    if args.aggregate_runs:
        rids = [x.strip() for x in args.aggregate_runs.split(",") if x.strip()]
        out_md = aggregate_existing_iter2_runs(ROOT / "results", rids, args.run_id)
        if args.email_on_finish:
            send_monitor_email(
                subject=f"[RSI-Sprint] Iter 2 N={len(rids)} Counterbalanced Matrix Complete ({args.run_id})",
                body=out_md.read_text(encoding="utf-8"),
            )
        return

    seed_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    try:
        rep = run_iter2_batch(
            run_id=args.run_id,
            seeds=seed_list,
            model=args.model,
            post_window=args.post_window,
            max_usd=args.max_usd,
            workers=args.workers,
            thinking_budget=args.thinking_budget,
            results_root=ROOT / "results",
            dotenv_path=ROOT / ".env",
        )
    except BudgetExceededError as exc:
        print(f"[BUDGET GUARDRAIL HALT] {exc}", flush=True)
        sys.exit(2)

    md = render_iter2_markdown(rep)
    out_md = ROOT / "results" / "runs" / args.run_id / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\n{md}\nSaved report: {out_md}", flush=True)


if __name__ == "__main__":
    main()
