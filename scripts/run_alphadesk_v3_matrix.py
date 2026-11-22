#!/usr/bin/env python3
"""AlphaDesk-v0.3 Multi-Shift (`5` Shifts / `6` Windows, `H=84`) Stateful Evaluation Runner.

Evaluates external baselines (`SkillBook`, `SRN`, `ScopedCardBook`) vs. `SGEH-v0.3` ablations
(`sgeh_surprise_shelf`, `sgeh_hazard_noshelf`, `sgeh_hazard_shelf`) across:
  - Track 1 (`multi_abrupt_recurrence`): Discrete reversals + 2-day liquidity shock + Full (`A`) & Partial (`A'`) recurrence
  - Track 2 (`mixed_decay_opportunity`): Type-II silent opportunity rises + smooth crowding decays + recurrence + shock
  - Matched `stationary` Control (`H = 84`)
"""

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
from src.core.ledger import BudgetExceededError, CostLedger, atomic_write_json  # noqa: E402
from src.core.llm_client import MeteredLLMClient  # noqa: E402
from src.envs.alpha_desk_v3 import AlphaDeskV3Env, MAX_WEIGHT_V3, STRATEGIES_V3  # noqa: E402
from src.harnesses import make_harness  # noqa: E402
from src.harnesses.base import PublicEpisodeRecord  # noqa: E402
from src.harnesses.sgeh_v3 import normalize_weights_v3  # noqa: E402
from src.runner import _build_acting_system_prompt  # noqa: E402

DEFAULT_METHODS_V3 = (
    "skillbook",
    "srn",
    "scoped_cards_v3",
    "sgeh_surprise_shelf",
    "sgeh_hazard_noshelf",
    "sgeh_hazard_shelf",
)

METHOD_LABELS_V3 = {
    "skillbook": "SkillBook (Baseline)",
    "srn": "SRN (Baseline)",
    "scoped_cards_v3": "ScopedCardBook (Baseline)",
    "sgeh_surprise_shelf": "SGEH-Surprise+Shelf (Ours — No-Hazard Abl)",
    "sgeh_hazard_noshelf": "SGEH+Hazard-NoShelf (Ours — No-Shelf Abl)",
    "sgeh_hazard_shelf": "SGEH+Hazard+Shelf (Ours — Full)",
}


def _mock_v3_actor(steps_so_far: list[dict[str, Any]], sys_prompt: str) -> dict[str, Any]:
    if not steps_so_far:
        return {"tool": "inspect_desk_brief", "args": {}, "rationale": "Read session brief & current holdings."}
    first_obs = steps_so_far[0].get("obs", {})
    is_freeze = first_obs.get("liquidity_status") == "acute_liquidity_freeze"
    holdings = first_obs.get("current_holdings", {})
    if is_freeze and isinstance(holdings, dict) and holdings:
        return {"tool": "submit_allocation", "args": {"weights": holdings}, "rationale": "Hold steady during liquidity freeze."}

    if len(steps_so_far) == 1:
        for s in STRATEGIES_V3:
            if f"on shelved primary `{s}`" in sys_prompt or f"on long-untested `{s}`" in sys_prompt or f"on `{s}`" in sys_prompt:
                return {"tool": "probe_strategy_execution", "args": {"strategy": s}, "rationale": "Execute targeted probe."}

    for st in steps_so_far:
        obs = st.get("obs", {})
        if isinstance(obs, dict) and float(obs.get("live_net_yield_estimate", 0.0)) >= 78.0:
            win_s = str(obs.get("strategy", "STRAT_S1"))
            other_s = next(x for x in STRATEGIES_V3 if x != win_s)
            return {
                "tool": "submit_allocation",
                "args": {"weights": normalize_weights_v3({}, win_s, other_s)},
                "rationale": "Allocate 0.55/0.45 to high probed strategy.",
            }
    return {
        "tool": "submit_allocation",
        "args": {"weights": normalize_weights_v3({}, "STRAT_S1", "STRAT_S2")},
        "rationale": "Default compliant 6-strategy allocation.",
    }


def run_v3_lifetime(
    *,
    mode: str,
    seed: int,
    horizon: int,
    method_name: str,
    client: MeteredLLMClient,
    on_ep_done: Any = None,
) -> dict[str, Any]:
    env = AlphaDeskV3Env(mode=mode, horizon=horizon, seed=seed)
    harness = make_harness(method_name)

    logs: list[dict[str, Any]] = []
    acting_calls = 0
    reflection_calls = 0
    acting_usd = 0.0
    reflection_usd = 0.0
    thinking_tokens = 0

    for ep in range(horizon):
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
                method=f"v3_{method_name}",
                mock_responder=lambda s_p, _u_p, st=steps: _mock_v3_actor(st, s_p),
            )
            acting_calls += 1
            acting_usd += resp.usd
            thinking_tokens += resp.thinking_tokens

            parsed = resp.parsed_json or {}
            t_name = str(parsed.get("tool", ""))
            t_args = parsed.get("args", {})
            if not isinstance(t_args, dict):
                t_args = {}

            if not t_name or rem == 1 and t_name != "submit_allocation":
                if not steps:
                    t_name, t_args = "inspect_desk_brief", {}
                elif rem == 1:
                    # Fallback compliant hold/allocation if model forgot terminal tool on final turn
                    cur_h = steps[0].get("obs", {}).get("current_holdings", {})
                    w_fb = normalize_weights_v3(cur_h, "STRAT_S1", "STRAT_S2")
                    t_name, t_args = "submit_allocation", {"weights": w_fb}

            # Normalize near-compliant submit_allocation weights if LLM drifted by e.g. 0.56 instead of 0.55
            if t_name == "submit_allocation" and isinstance(t_args.get("weights"), dict):
                raw_w = t_args["weights"]
                vals = [float(v) for v in raw_w.values() if isinstance(v, (int, float))]
                if vals and sum(vals) > 0.85 and max(vals) <= MAX_WEIGHT_V3 + 0.06 and min(vals) >= -1e-4:
                    t_args = {"weights": normalize_weights_v3(raw_w, "STRAT_S1", "STRAT_S2")}

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
            outcome = env.step_tool(
                "submit_allocation",
                {"weights": normalize_weights_v3({}, "STRAT_S1", "STRAT_S2")},
            )
            final_score = float(outcome.public_score or 0.0)
            priv = outcome.private_eval

        pub_rec = PublicEpisodeRecord(
            env_name=spec.env_name,
            episode_idx=ep,
            timestamp=spec.timestamp,
            task_prompt=spec.task_prompt,
            steps=steps,
            public_score=final_score,
        )
        h_meta = harness.on_episode_complete(pub_rec, client)
        reflection_calls += int(h_meta.get("reflection_calls", 0))
        reflection_usd += float(h_meta.get("reflection_usd", 0.0))

        logs.append(
            {
                "episode": ep + 1,
                "window": int(priv.get("window", env.window_index(ep))),
                "regime": priv.get("regime", ""),
                "is_high_vol": bool(priv.get("is_high_vol", False)),
                "is_liquidity_freeze": bool(priv.get("is_liquidity_freeze", False)),
                "public_score": final_score,
                "expected_net_no_noise": float(priv.get("expected_net_no_noise", final_score)),
                "dp_oracle_step_net": float(priv.get("dp_oracle_step_net", 85.0)),
                "normalized_utility": float(priv.get("normalized_utility", 0.0)),
                "optimal_policy_match": bool(priv.get("optimal_policy_match", False)),
                "stale_policy_match": bool(priv.get("stale_policy_match", False)),
                "invariant_violation": bool(priv.get("invariant_violation", False)),
                "turnover_fee": float(priv.get("turnover_fee", 0.0)),
                "probes_used": int(priv.get("probes_used", 0)),
                "harness_meta": h_meta,
                "steps": steps,
            }
        )
        if on_ep_done is not None:
            on_ep_done()

    # Compute overall, post-W1 (`W2..W6` = shifted multi-window horizon), and per-window DP-normalized utilities!
    dp_cum_oracle = sum(r["dp_oracle_step_net"] for r in logs)
    agent_cum_expected_net = sum(r["expected_net_no_noise"] for r in logs)
    full_dp_utility_pct = round(100.0 * agent_cum_expected_net / max(1.0, dp_cum_oracle), 2)

    post_logs = [r for r in logs if r["window"] >= 1] or logs
    post_dp_oracle = sum(r["dp_oracle_step_net"] for r in post_logs)
    post_agent_net = sum(r["expected_net_no_noise"] for r in post_logs)
    post_shift_dp_utility_pct = round(100.0 * post_agent_net / max(1.0, post_dp_oracle), 2)

    pre_logs = [r for r in logs if r["window"] == 0] or logs[:1]
    pre_dp_oracle = sum(r["dp_oracle_step_net"] for r in pre_logs)
    pre_shift_dp_utility_pct = round(
        100.0 * sum(r["expected_net_no_noise"] for r in pre_logs) / max(1.0, pre_dp_oracle), 2
    )

    per_window_pct: list[float] = []
    per_window_tags: list[str] = []
    for w_i in range(6):
        w_rows = [r for r in logs if r["window"] == w_i]
        if w_rows:
            w_pct = round(
                100.0 * sum(r["expected_net_no_noise"] for r in w_rows) / max(1.0, sum(r["dp_oracle_step_net"] for r in w_rows)),
                2,
            )
            calm_tags = [r["regime"] for r in w_rows if not r["is_high_vol"] and not r["is_liquidity_freeze"]]
            w_tag = calm_tags[-1] if calm_tags else w_rows[0]["regime"]
        else:
            w_pct = 0.0
            w_tag = f"W{w_i + 1}"
        per_window_pct.append(w_pct)
        per_window_tags.append(w_tag)

    calm_post = [r for r in post_logs if not r["is_high_vol"] and not r["is_liquidity_freeze"]]
    calm_opt_pct = round(100.0 * sum(1 for r in calm_post if r["optimal_policy_match"]) / max(1, len(calm_post)), 2)
    stale_lock_pct = round(100.0 * sum(1 for r in calm_post if r["stale_policy_match"]) / max(1, len(calm_post)), 2)
    viol_pct = round(100.0 * sum(1 for r in logs if r["invariant_violation"]) / max(1, len(logs)), 2)

    true_restorations = 0
    partial_caught = 0
    hazard_probes = 0
    alarms_total = sum(1 for r in logs if r["harness_meta"].get("alarm_fired"))
    if hasattr(harness, "buckets"):
        cb = harness.buckets.get("calm_volatility")
        if cb is not None:
            true_restorations = int(getattr(cb, "true_restorations", 0))
            partial_caught = int(getattr(cb, "partial_recurrence_caught", 0))
            hazard_probes = int(getattr(cb, "hazard_probes_triggered", 0))

    return {
        "mode": mode,
        "seed": seed,
        "method": method_name,
        "episodes": horizon,
        "full_dp_utility_pct": full_dp_utility_pct,
        "pre_shift_utility_pct": pre_shift_dp_utility_pct,
        "post_shift_utility_pct": post_shift_dp_utility_pct,
        "calm_optimal_pct": calm_opt_pct,
        "stale_lock_in_pct": stale_lock_pct,
        "invariant_violation_pct": viol_pct,
        "per_window_utility_pct": per_window_pct,
        "per_window_tags": per_window_tags,
        "probes_total": sum(r["probes_used"] for r in logs),
        "turnover_pts_total": round(sum(r["turnover_fee"] for r in logs), 2),
        "alarms_fired": alarms_total,
        "hazard_probes": hazard_probes,
        "true_restorations": true_restorations,
        "partial_recurrence_caught": partial_caught,
        "acting_calls": acting_calls,
        "reflection_calls": reflection_calls,
        "thinking_tokens": thinking_tokens,
        "usd": round(acting_usd + reflection_usd, 6),
        "reflection_usd": round(reflection_usd, 6),
        "curve_utility": [round(100.0 * r["normalized_utility"], 2) for r in logs],
        "logs": logs,
    }


def run_v3_batch(
    *,
    run_id: str,
    modes: list[str],
    seeds: list[int],
    methods: list[str],
    horizon: int,
    model: str,
    max_usd: float,
    workers: int,
    thinking_budget: int,
    results_root: Path,
    dotenv_path: Path | None,
) -> dict[str, Any]:
    ledger = CostLedger(results_root=results_root, run_id=run_id, phase="dev", max_usd=max_usd)
    total_eps = len(modes) * len(seeds) * len(methods) * horizon
    done_eps = 0
    lock = threading.Lock()

    def _tick(lbl: str) -> None:
        nonlocal done_eps
        with lock:
            done_eps += 1
            if done_eps % 4 == 0 or done_eps == total_eps:
                ledger.update_progress(
                    completed_units=min(done_eps, total_eps),
                    total_units=total_eps,
                    status="running",
                    current_label=lbl,
                    log_every=28,
                )

    client = MeteredLLMClient(
        model=model,
        ledger=ledger,
        dotenv_path=dotenv_path,
        thinking_budget=thinking_budget,
        temperature=0.0,
    )
    traces_dir = ledger.results_root / "raw" / run_id
    traces_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = ledger.run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(md, sd, mth) for md in modes for sd in seeds for mth in methods]
    raw_results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), workers))) as pool:
        futs = [
            pool.submit(
                run_v3_lifetime,
                mode=md,
                seed=sd,
                horizon=horizon,
                method_name=mth,
                client=client,
                on_ep_done=lambda m=md, s=sd, h=mth: _tick(f"{m} s{s} {h}"),
            )
            for md, sd, mth in jobs
        ]
        for fut in as_completed(futs):
            res = fut.result()
            md, sd, mth = res["mode"], res["seed"], res["method"]
            print(
                f"  -> [{md:23s} | s{sd} | {mth:19s}] PostW2-6={res['post_shift_utility_pct']:5.1f}% "
                f"(Full={res['full_dp_utility_pct']:5.1f}%, CalmOpt={res['calm_optimal_pct']:5.1f}%) | "
                f"W1..6={[round(x, 0) for x in res['per_window_utility_pct']]} | "
                f"Prb={res['probes_total']:2d} Turn={res['turnover_pts_total']:4.1f} "
                f"Ref={res['reflection_calls']:2d} TrueRec={res['true_restorations']} PartCatch={res['partial_recurrence_caught']} | "
                f"Cell=${res['usd']:.4f} Run=${ledger.run_usd:.4f}",
                flush=True,
            )
            cell_slug = f"alpha_desk_v3_{md}__{mth}__{model.replace('/', '_')}_s{sd}"
            with open(traces_dir / f"{cell_slug}.jsonl", "w", encoding="utf-8") as fh:
                for row in res["logs"]:
                    fh.write(json.dumps(row) + "\n")

            tot_calls = res["acting_calls"] + res["reflection_calls"]
            atomic_write_json(
                cells_dir / f"{cell_slug}.json",
                {
                    "summary": {
                        "run_id": run_id,
                        "env_spec": f"alpha_desk_v3:{md}",
                        "harness_name": mth,
                        "model": model,
                        "seed": sd,
                        "episodes": horizon,
                        "mean_score": round(statistics.mean(r["public_score"] for r in res["logs"]), 2),
                        "mean_normalized_utility": res["full_dp_utility_pct"],
                        "pre_shift_utility": res["pre_shift_utility_pct"],
                        "post_shift_utility": res["post_shift_utility_pct"],
                        "stale_lock_in_rate": round(res["stale_lock_in_pct"] / 100.0, 4),
                        "invariant_violation_rate": round(res["invariant_violation_pct"] / 100.0, 4),
                        "total_acting_calls": res["acting_calls"],
                        "total_reflection_calls": res["reflection_calls"],
                        "rsi_tax_call_share": round(res["reflection_calls"] / max(1, tot_calls), 4),
                        "total_usd": res["usd"],
                        "reflection_usd": res["reflection_usd"],
                        "per_window_utility_pct": res["per_window_utility_pct"],
                        "per_window_tags": res["per_window_tags"],
                        "probes_total": res["probes_total"],
                        "turnover_pts_total": res["turnover_pts_total"],
                        "true_restorations": res["true_restorations"],
                        "partial_recurrence_caught": res["partial_recurrence_caught"],
                    },
                    "curve_utility": res["curve_utility"],
                    "curve_score": [r["public_score"] for r in res["logs"]],
                    "alarms": [r["episode"] for r in res["logs"] if r["harness_meta"].get("alarm_fired")],
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            raw_results.append({k: v for k, v in res.items() if k != "logs"})

    rep = summarize_v3_report(
        run_id=run_id,
        model=model,
        horizon=horizon,
        thinking_budget=thinking_budget,
        records=raw_results,
        total_usd=round(ledger.run_usd, 6),
        in_tok=ledger.run_input_tokens,
        out_tok=ledger.run_output_tokens,
        think_tok=ledger.run_thinking_tokens,
    )
    atomic_write_json(ledger.run_dir / "alphadesk_v3_report.json", rep)
    ledger.update_progress(
        completed_units=total_eps,
        total_units=total_eps,
        status="completed",
        current_label="finished alphadesk v0.3 batch",
    )
    return rep


def summarize_v3_report(
    *,
    run_id: str,
    model: str,
    horizon: int,
    thinking_budget: int,
    records: list[dict[str, Any]],
    total_usd: float,
    in_tok: int,
    out_tok: int,
    think_tok: int,
) -> dict[str, Any]:
    modes = sorted({r["mode"] for r in records})
    methods = [m for m in DEFAULT_METHODS_V3 if any(r["method"] == m for r in records)]
    seeds = sorted({int(r["seed"]) for r in records})

    by_mode: dict[str, dict[str, Any]] = {}
    contrasts: dict[str, dict[str, Any]] = {}

    for md in modes:
        by_mode[md] = {}
        contrasts[md] = {}
        for mth in methods:
            sub = sorted([r for r in records if r["mode"] == md and r["method"] == mth], key=lambda x: x["seed"])
            if not sub:
                continue
            p_m, p_lo, p_hi = bootstrap_mean_ci([r["post_shift_utility_pct"] for r in sub])
            f_m, f_lo, f_hi = bootstrap_mean_ci([r["full_dp_utility_pct"] for r in sub])
            win_means = [
                round(statistics.mean(r["per_window_utility_pct"][w_i] for r in sub), 2)
                for w_i in range(6)
            ]
            by_mode[md][mth] = {
                "n_seeds": len(sub),
                "post_shift_utility_pct": p_m,
                "post_shift_ci95": [p_lo, p_hi],
                "full_dp_utility_pct": f_m,
                "full_dp_ci95": [f_lo, f_hi],
                "calm_optimal_pct": round(statistics.mean(r["calm_optimal_pct"] for r in sub), 2),
                "stale_lock_in_pct": round(statistics.mean(r["stale_lock_in_pct"] for r in sub), 2),
                "per_window_utility_pct": win_means,
                "mean_probes": round(statistics.mean(r["probes_total"] for r in sub), 2),
                "mean_turnover_pts": round(statistics.mean(r["turnover_pts_total"] for r in sub), 2),
                "mean_reflection_calls": round(statistics.mean(r["reflection_calls"] for r in sub), 2),
                "mean_usd": round(statistics.mean(r["usd"] for r in sub), 4),
                "true_restorations_sum": sum(int(r["true_restorations"]) for r in sub),
                "partial_caught_sum": sum(int(r["partial_recurrence_caught"]) for r in sub),
            }

        # Paired per-seed contrasts for `sgeh_hazard_shelf` (Ours — Full) vs baselines & ablations
        full_by_seed = {r["seed"]: r["post_shift_utility_pct"] for r in records if r["mode"] == md and r["method"] == "sgeh_hazard_shelf"}
        for comp in ("skillbook", "srn", "scoped_cards_v3", "sgeh_surprise_shelf", "sgeh_hazard_noshelf"):
            comp_by_seed = {r["seed"]: r["post_shift_utility_pct"] for r in records if r["mode"] == md and r["method"] == comp}
            common = sorted(set(full_by_seed) & set(comp_by_seed))
            if common:
                diffs = [round(full_by_seed[s] - comp_by_seed[s], 2) for s in common]
                dm, dlo, dhi = bootstrap_mean_ci(diffs)
                contrasts[md][f"sgeh_hazard_shelf_minus_{comp}"] = {
                    "mean": dm,
                    "ci95": [dlo, dhi],
                    "per_seed": diffs,
                }

    return {
        "run_id": run_id,
        "experiment_type": "alphadesk_v3_multishift_stateful",
        "model": model,
        "horizon": horizon,
        "requested_thinking_budget": thinking_budget,
        "observed_run_input_tokens": in_tok,
        "observed_run_output_tokens": out_tok,
        "observed_run_thinking_tokens": think_tok,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "modes": modes,
        "total_run_usd": total_usd,
        "by_mode_summary": by_mode,
        "paired_contrasts": contrasts,
        "records": records,
    }


def render_v3_markdown(rep: dict[str, Any]) -> str:
    lines = [
        f"# AlphaDesk-v0.3 Multi-Shift (`5` Shifts / `6` Windows, `H={rep['horizon']}`) Report: `{rep['run_id']}`",
        "",
        f"- **Model:** `{rep['model']}` | **Seeds:** `{rep['seeds']}` | **Total Spend:** `${rep['total_run_usd']:.4f}`",
        f"- **Observed Tokens:** `in={rep['observed_run_input_tokens']:,}`, `out={rep['observed_run_output_tokens']:,}`, `thinking={rep['observed_run_thinking_tokens']:,}`",
        "",
    ]
    for md in rep["modes"]:
        bm = rep["by_mode_summary"].get(md, {})
        con = rep["paired_contrasts"].get(md, {})
        lines.extend(
            [
                f"## Track / Mode: `{md}` (`6` Windows, Stateful `w_{{t-1}}` + `4.0 pt` L1 Turnover DP Oracle)",
                "",
                "| Method | Post-Shift `W2–W6` (%) | Full `W1–W6` (%) | CalmOpt (%) | Windows `W1..W6` (%) | Probes | Turnover (pts) | Ref Calls | USD |",
                "|---|---:|---:|---:|---|---:|---:|---:|---:|",
            ]
        )
        for mth in DEFAULT_METHODS_V3:
            if mth not in bm:
                continue
            row = bm[mth]
            lbl = METHOD_LABELS_V3.get(mth, mth)
            if mth == "sgeh_hazard_shelf":
                lbl = f"**{lbl}**"
            w_str = " / ".join(f"{w:.0f}" for w in row["per_window_utility_pct"])
            lines.append(
                f"| {lbl} | **{row['post_shift_utility_pct']:.1f}%** | {row['full_dp_utility_pct']:.1f}% | "
                f"{row['calm_optimal_pct']:.1f}% | `{w_str}` | {row['mean_probes']:.1f} | "
                f"{row['mean_turnover_pts']:.1f} | {row['mean_reflection_calls']:.1f} | `${row['mean_usd']:.4f}` |"
            )
        lines.append("")
        for k, v in con.items():
            lines.append(
                f"- **`{k}` (`W2–W6`):** **`{v['mean']:+.2f} pts`** `[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]` (per-seed: `{v['per_seed']}`)"
            )
        sh_full = bm.get("sgeh_hazard_shelf", {})
        if sh_full:
            lines.append(
                f"- **Recurrence Shelf Diagnostics (`SGEH+Hazard+Shelf [Ours]`):** True Full Restorations = `{sh_full.get('true_restorations_sum', 0)}`, Partial Recurrences (`A'`) Caught Immediately = `{sh_full.get('partial_caught_sum', 0)}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def aggregate_v3_runs(results_root: Path, run_ids: list[str], out_run_id: str) -> Path:
    all_records: list[dict[str, Any]] = []
    tot_usd = 0.0
    in_t = out_t = th_t = 0
    model = "gemini-3.8-flash"
    horizon = 84
    tb = 0
    for rid in run_ids:
        d = json.loads((results_root / "runs" / rid / "alphadesk_v3_report.json").read_text(encoding="utf-8"))
        all_records.extend(d.get("records", []))
        tot_usd += float(d.get("total_run_usd", 0.0))
        in_t += int(d.get("observed_run_input_tokens", 0))
        out_t += int(d.get("observed_run_output_tokens", 0))
        th_t += int(d.get("observed_run_thinking_tokens", 0))
        model = str(d.get("model", model))
        horizon = int(d.get("horizon", horizon))
        tb = int(d.get("requested_thinking_budget", tb))

    rep = summarize_v3_report(
        run_id=out_run_id,
        model=model,
        horizon=horizon,
        thinking_budget=tb,
        records=all_records,
        total_usd=round(tot_usd, 6),
        in_tok=in_t,
        out_tok=out_t,
        think_tok=th_t,
    )
    out_dir = results_root / "runs" / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "alphadesk_v3_report.json", rep)
    md = render_v3_markdown(rep)
    out_md = out_dir / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\n{md}\nSaved merged AlphaDesk-v0.3 report: {out_md}", flush=True)
    return out_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AlphaDesk-v0.3 Multi-Shift Stateful Matrix")
    parser.add_argument("--run-id", default="v3_pilot_t1_s401", help="Run ID")
    parser.add_argument("--modes", default="multi_abrupt_recurrence", help="Comma-separated AlphaDesk-v0.3 modes")
    parser.add_argument("--seeds", default="401", help="Comma-separated seeds")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS_V3), help="Comma-separated methods")
    parser.add_argument("--horizon", type=int, default=84, help="Lifetime episodes (default 84 = 6 windows)")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Model ID or 'mock'")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Requested thinking budget")
    parser.add_argument("--workers", type=int, default=6, help="Parallel threads")
    parser.add_argument("--max-usd", type=float, default=1.95, help="Strict Tier-1 per-run USD cap")
    parser.add_argument("--aggregate-runs", default="", help="Comma-separated run IDs to merge")
    args = parser.parse_args()

    if args.aggregate_runs:
        rids = [x.strip() for x in args.aggregate_runs.split(",") if x.strip()]
        aggregate_v3_runs(ROOT / "results", rids, args.run_id)
        return

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]

    try:
        rep = run_v3_batch(
            run_id=args.run_id,
            modes=modes,
            seeds=seeds,
            methods=methods,
            horizon=args.horizon,
            model=args.model,
            max_usd=args.max_usd,
            workers=args.workers,
            thinking_budget=args.thinking_budget,
            results_root=ROOT / "results",
            dotenv_path=ROOT / ".env",
        )
    except BudgetExceededError as exc:
        print(f"[BUDGET GUARDRAIL HALT] {exc}", flush=True)
        sys.exit(2)

    md = render_v3_markdown(rep)
    out_md = ROOT / "results" / "runs" / args.run_id / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\n{md}\nSaved report: {out_md}", flush=True)


if __name__ == "__main__":
    main()
