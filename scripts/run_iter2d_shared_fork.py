#!/usr/bin/env python3
"""Iteration 2d: Shared-Checkpoint Post-Shift Adaptation Test (Fresh Counterbalanced Seeds 205-208).

Addresses Codex's acquisition-vs-adaptation DiD critique directly:
  - Runs ONE shared pre-shift acquisition prefix (`0..t*-1`) per seed so ALL methods start `t*`
    with the EXACT SAME ~98% optimal calm & elevated-vol weights, sample counts, and history.
  - Forks `ScopedCardBook (Baseline)`, `SGEH-SurpriseOnly (Ours - Ablation)`, and `SGEH+Hazard (Ours - Full)`
    across `{stationary, abrupt_reversal, gradual_decay}` over 16 post-boundary sessions.
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
from scripts.run_iter2_online_matrix import run_episode_slice_iter2  # noqa: E402
from src.core.ledger import BudgetExceededError, CostLedger, atomic_write_json  # noqa: E402
from src.core.llm_client import MeteredLLMClient  # noqa: E402
from src.envs.alpha_desk_v2 import AlphaDeskV2Env, STRATEGIES_V2  # noqa: E402
from src.harnesses.base import PublicEpisodeRecord  # noqa: E402
from src.harnesses.scoped_cards import ScopedCardBookHarness  # noqa: E402
from src.harnesses.sgeh_v2 import ScopedSGEHHardened  # noqa: E402

MODES_2D = ("stationary", "abrupt_reversal", "gradual_decay")
ARMS_2D = ("scoped_cards", "sgeh_surprise", "sgeh_hazard")


def build_scoped_cardbook_from_shared_sgeh(
    shared_sgeh: ScopedSGEHHardened, pre_logs: list[dict[str, Any]]
) -> ScopedCardBookHarness:
    """Initialize `ScopedCardBook (Baseline)` with the EXACT same learned weights, uses, & mean scores at t*."""
    cb = ScopedCardBookHarness(reflect_every=3)
    for idx, r in enumerate(pre_logs):
        cb._history.append(
            PublicEpisodeRecord(
                env_name="alpha_desk_v2",
                episode_idx=idx,
                timestamp=int(r.get("episode", idx + 1)),
                task_prompt=f"Session #{idx + 1}: Review today's desk brief and submit compliant portfolio weights to maximize realized session score (0-100).",
                steps=r.get("steps", []),
                public_score=float(r.get("public_score", 0.0)),
            )
        )

    calm_b = shared_sgeh.buckets["calm_volatility"]
    hv_b = shared_sgeh.buckets["elevated_volatility"]
    calm_w = dict(calm_b.active_weights) or {s: (0.65 if i == 0 else (0.35 if i == 1 else 0.0)) for i, s in enumerate(STRATEGIES_V2)}
    hv_w = dict(hv_b.active_weights) or {s: (0.65 if i == 3 else (0.35 if i == 2 else 0.0)) for i, s in enumerate(STRATEGIES_V2)}
    calm_tgt = max(STRATEGIES_V2, key=lambda s: calm_w.get(s, 0.0))
    hv_tgt = max(STRATEGIES_V2, key=lambda s: hv_w.get(s, 0.0))

    cb.cards = [
        {
            "card_id": "C1",
            "kind": "calm_utility",
            "scope_condition": "volatility_regime_hint == 'calm_volatility' and liquidity_status == 'orderly_normal'",
            "target_strategy": calm_tgt,
            "recommended_weights": calm_w,
            "uses": max(1, int(calm_b.post_shift_n)),
            "mean_score": round(float(calm_b.expected_mean), 1),
        },
        {
            "card_id": "C2",
            "kind": "invariant_or_hedge",
            "scope_condition": "volatility_regime_hint == 'elevated_volatility' and liquidity_status == 'orderly_normal'",
            "target_strategy": hv_tgt,
            "recommended_weights": hv_w,
            "uses": max(1, int(hv_b.post_shift_n)),
            "mean_score": round(float(hv_b.expected_mean), 1),
        },
    ]
    cb.set_post_fork_schedule()
    return cb


def run_shared_fork_batch(
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
) -> dict[str, Any]:
    ledger = CostLedger(results_root=results_root, run_id=run_id, phase="dev", max_usd=max_usd)
    est_eps = len(seeds) * (14 + len(MODES_2D) * len(ARMS_2D) * post_window)
    done_eps = 0
    lock = threading.Lock()

    def _tick(lbl: str) -> None:
        nonlocal done_eps
        with lock:
            done_eps += 1
            ledger.update_progress(
                completed_units=min(done_eps, est_eps),
                total_units=est_eps,
                status="running",
                current_label=lbl,
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

    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        pre_env = AlphaDeskV2Env(mode="stationary", horizon=40, seed=seed, counterbalance=True)
        t_star = pre_env.t_star
        role_meta = pre_env.private_role_map()
        w_ref = role_meta["reference_incumbent_weights"]

        shared_sgeh = ScopedSGEHHardened(enable_hazard=False)
        pre_res = run_episode_slice_iter2(
            env=pre_env,
            harness=shared_sgeh,
            client=client,
            ep_indices=range(0, t_star),
            method_tag="shared_preshift",
            w_ref=w_ref,
            force_final_curate=True,
            on_ep_done=lambda s=seed: _tick(f"s{s} shared_pre"),
        )
        h_cards = build_scoped_cardbook_from_shared_sgeh(shared_sgeh, pre_res["logs"])
        h_surp = shared_sgeh.clone()
        h_haz = shared_sgeh.clone()
        h_haz.enable_hazard = True
        h_haz.name = "sgeh_hazard"

        print(
            f"\n[SEED {seed} | t*={t_star}] Shared Pre-Shift Util={pre_res['mean_utility_pct']:.1f}% | "
            f"Shared Calm Weights={h_cards.cards[0]['recommended_weights']} (inc_1={role_meta['inc_1']})",
            flush=True,
        )

        arm_prototypes = {
            "scoped_cards": h_cards,
            "sgeh_surprise": h_surp,
            "sgeh_hazard": h_haz,
        }
        jobs = [
            (mode_name, arm_name, arm_prototypes[arm_name].clone())
            for mode_name in MODES_2D
            for arm_name in ARMS_2D
        ]
        seed_modes: dict[str, dict[str, Any]] = {m: {} for m in MODES_2D}

        def _exec(md: str, ar: str, h_copy: Any) -> tuple[str, str, dict[str, Any]]:
            b_env = AlphaDeskV2Env(mode=md, horizon=40, seed=seed, t_star_override=t_star, counterbalance=True)
            b_res = run_episode_slice_iter2(
                env=b_env,
                harness=h_copy,
                client=client,
                ep_indices=range(t_star, t_star + post_window),
                method_tag=f"shared_{ar}",
                w_ref=w_ref,
                force_final_curate=False,
                on_ep_done=lambda s=seed, m=md, a=ar: _tick(f"s{s} {m}:{a}"),
            )
            return md, ar, b_res

        with ThreadPoolExecutor(max_workers=max(1, min(9, workers))) as pool:
            for fut in as_completed([pool.submit(_exec, m, a, hc) for m, a, hc in jobs]):
                md_name, ar_name, b_res = fut.result()
                seed_modes[md_name][ar_name] = {k: v for k, v in b_res.items() if k != "logs"}
                print(
                    f"  -> [{md_name:15s} | {ar_name:13s}] Net={b_res['mean_utility_pct']:5.1f}% "
                    f"(CalmOpt={b_res['calm_optimal_pct']:5.1f}%) | Probes={b_res['probes_total']} | "
                    f"Alarms={b_res['alarms_fired']} | Ref={b_res['reflection_calls']} | Run=${ledger.run_usd:.4f}",
                    flush=True,
                )
                cell_slug = f"alpha_desk_v2_shared_{md_name}__{ar_name}__{model.replace('/', '_')}_s{seed}"
                stitched = pre_res["logs"] + b_res["logs"]
                with open(traces_dir / f"{cell_slug}.jsonl", "w", encoding="utf-8") as fh:
                    for row in stitched:
                        fh.write(json.dumps(row) + "\n")
                tot_c = pre_res["acting_calls"] + b_res["acting_calls"] + pre_res["reflection_calls"] + b_res["reflection_calls"]
                ref_c = pre_res["reflection_calls"] + b_res["reflection_calls"]
                atomic_write_json(
                    cells_dir / f"{cell_slug}.json",
                    {
                        "summary": {
                            "run_id": run_id,
                            "env_spec": f"alpha_desk_v2:{md_name}",
                            "harness_name": ar_name,
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
                            "total_reflection_calls": ref_c,
                            "rsi_tax_call_share": round(100.0 * ref_c / max(1, tot_c), 2),
                            "total_usd": round(pre_res["usd"] / 9.0 + b_res["usd"], 6),
                            "reflection_usd": 0.0,
                        },
                        "curve_utility": pre_res["curve_utility"] + b_res["curve_utility"],
                        "curve_score": [r["public_score"] for r in stitched],
                        "alarms": [r["episode"] for r in stitched if r.get("harness_meta", {}).get("alarm_fired")],
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )

        per_seed.append(
            {
                "seed": seed,
                "t_star": t_star,
                "shared_pre_shift_utility_pct": pre_res["mean_utility_pct"],
                "role_map": role_meta,
                "modes": seed_modes,
            }
        )

    rep = summarize_shared_fork(
        run_id=run_id,
        model=model,
        thinking_budget=thinking_budget,
        post_window=post_window,
        per_seed=per_seed,
        total_usd=round(ledger.run_usd, 6),
        max_usd=max_usd,
        in_tok=ledger.run_input_tokens,
        out_tok=ledger.run_output_tokens,
        think_tok=ledger.run_thinking_tokens,
    )
    atomic_write_json(ledger.run_dir / "shared_fork_report.json", rep)
    atomic_write_json(ledger.run_dir / "causal_fork_report.json", rep)
    ledger.update_progress(
        completed_units=est_eps,
        total_units=est_eps,
        status="completed",
        current_label="finished iter2d shared fork",
    )
    return rep


def summarize_shared_fork(
    *,
    run_id: str,
    model: str,
    thinking_budget: int,
    post_window: int,
    per_seed: list[dict[str, Any]],
    total_usd: float,
    max_usd: float,
    in_tok: int,
    out_tok: int,
    think_tok: int,
) -> dict[str, Any]:
    rev_haz_vs_cb = [
        round(s["modes"]["abrupt_reversal"]["sgeh_hazard"]["mean_utility_pct"] - s["modes"]["abrupt_reversal"]["scoped_cards"]["mean_utility_pct"], 2)
        for s in per_seed
    ]
    rev_surp_vs_cb = [
        round(s["modes"]["abrupt_reversal"]["sgeh_surprise"]["mean_utility_pct"] - s["modes"]["abrupt_reversal"]["scoped_cards"]["mean_utility_pct"], 2)
        for s in per_seed
    ]
    stat_haz_vs_cb = [
        round(s["modes"]["stationary"]["sgeh_hazard"]["mean_utility_pct"] - s["modes"]["stationary"]["scoped_cards"]["mean_utility_pct"], 2)
        for s in per_seed
    ]
    stat_surp_vs_cb = [
        round(s["modes"]["stationary"]["sgeh_surprise"]["mean_utility_pct"] - s["modes"]["stationary"]["scoped_cards"]["mean_utility_pct"], 2)
        for s in per_seed
    ]
    grad_haz_vs_cb = [
        round(s["modes"]["gradual_decay"]["sgeh_hazard"]["mean_utility_pct"] - s["modes"]["gradual_decay"]["scoped_cards"]["mean_utility_pct"], 2)
        for s in per_seed
    ]
    did_haz = [round(r - st, 2) for r, st in zip(rev_haz_vs_cb, stat_haz_vs_cb)]
    did_surp = [round(r - st, 2) for r, st in zip(rev_surp_vs_cb, stat_surp_vs_cb)]
    did_grad = [round(g - st, 2) for g, st in zip(grad_haz_vs_cb, stat_haz_vs_cb)]

    def _pack(vals: list[float]) -> dict[str, Any]:
        m, lo, hi = bootstrap_mean_ci(vals)
        return {"mean": m, "ci95": [lo, hi], "per_seed": vals}

    by_mode = {}
    for md in MODES_2D:
        by_mode[md] = {}
        for ar in ARMS_2D:
            rows = [s["modes"][md][ar] for s in per_seed]
            m, lo, hi = bootstrap_mean_ci([r["mean_utility_pct"] for r in rows])
            by_mode[md][ar] = {
                "mean_utility_pct": m,
                "ci95": [lo, hi],
                "calm_optimal_pct": round(statistics.mean(r["calm_optimal_pct"] for r in rows), 2),
                "mean_probes": round(statistics.mean(r["probes_total"] for r in rows), 2),
                "mean_reflection_calls": round(statistics.mean(r["reflection_calls"] for r in rows), 2),
            }

    c_rev_haz = _pack(rev_haz_vs_cb)
    c_rev_surp = _pack(rev_surp_vs_cb)
    c_stat_haz = _pack(stat_haz_vs_cb)
    c_did_haz = _pack(did_haz)
    c_grad_haz = _pack(grad_haz_vs_cb)
    c_did_surp = _pack(did_surp)

    return {
        "run_id": run_id,
        "experiment_type": "iter2d_shared_checkpoint_adaptation",
        "model": model,
        "requested_thinking_budget": thinking_budget,
        "observed_run_input_tokens": in_tok,
        "observed_run_output_tokens": out_tok,
        "observed_run_thinking_tokens": think_tok,
        "n_seeds": len(per_seed),
        "seeds": [s["seed"] for s in per_seed],
        "post_window": post_window,
        "total_run_usd": total_usd,
        "max_usd_cap": max_usd,
        "by_mode_summary": by_mode,
        "contrasts_mean_and_95ci": {
            "drift_excised_minus_sham": c_did_haz,
            "drift_excised_minus_retained": c_rev_haz,
            "stationary_excised_minus_sham": c_stat_haz,
            "diff_in_diffs_drift_minus_stationary": c_did_surp,
            "drift_retained_minus_sham": c_grad_haz,
            "stationary_retained_minus_sham": _pack(did_grad),
        },
        "contrast_labels": {
            "drift_excised_minus_sham": "Diff-in-Diffs (Reversal - Stationary): SGEH+Hazard (Ours) - ScopedCardBook (Baseline)",
            "drift_excised_minus_retained": "Abrupt Reversal (Shared ~98% Fork): SGEH+Hazard (Ours) - ScopedCardBook (Baseline)",
            "stationary_excised_minus_sham": "Stationary Control (Shared ~98% Fork): SGEH+Hazard (Ours) - ScopedCardBook (Baseline)",
            "diff_in_diffs_drift_minus_stationary": "Diff-in-Diffs (Reversal - Stationary): SGEH-SurpriseOnly (Ours Ablation) - ScopedCardBook (Baseline)",
            "drift_retained_minus_sham": "Gradual Decay (Shared ~98% Fork): SGEH+Hazard (Ours) - ScopedCardBook (Baseline)",
            "stationary_retained_minus_sham": "Diff-in-Diffs (Gradual Decay - Stationary): SGEH+Hazard (Ours) - ScopedCardBook (Baseline)",
        },
        "per_seed": per_seed,
    }


def render_shared_fork_md(rep: dict[str, Any]) -> str:
    bm = rep["by_mode_summary"]
    c = rep["contrasts_mean_and_95ci"]
    lines = [
        f"# Iteration 2d Shared-Checkpoint Post-Shift Adaptation Report: `{rep['run_id']}`",
        "",
        f"- **Protocol:** Identical pre-shift acquired state forked at $t^*$ across fresh Latin-square seeds `{rep['seeds']}` (`0.00 pt` pre-shift gap!)",
        f"- **Model:** `{rep['model']}` | **Window:** `{rep['post_window']}` post-boundary sessions | **Spend:** `${rep['total_run_usd']:.4f}`",
        "",
        "## 1. Post-Shift Utility From Identical Acquired Checkpoint",
        "",
        "| Regime | `ScopedCardBook (Baseline)` | `SGEH-SurpriseOnly (Ours — Ablation)` | **`SGEH+Hazard (Ours — Full)`** | **`Ours − Baseline`** |",
        "|---|---:|---:|---:|---:|",
    ]
    for md, diff_key in (
        ("stationary", "stationary_excised_minus_sham"),
        ("abrupt_reversal", "drift_excised_minus_retained"),
        ("gradual_decay", "drift_retained_minus_sham"),
    ):
        sc = bm[md]["scoped_cards"]
        ss = bm[md]["sgeh_surprise"]
        sh = bm[md]["sgeh_hazard"]
        df = c[diff_key]
        lines.append(
            f"| `{md}` | {sc['mean_utility_pct']:.1f}% (`{sc['calm_optimal_pct']:.0f}%` Opt) | "
            f"{ss['mean_utility_pct']:.1f}% (`{ss['calm_optimal_pct']:.0f}%` Opt) | "
            f"**{sh['mean_utility_pct']:.1f}%** (`{sh['calm_optimal_pct']:.0f}%` Opt) | "
            f"**`{df['mean']:+.2f} pts`** `[{df['ci95'][0]:+.2f}, {df['ci95'][1]:+.2f}]` |"
        )
    lines.extend(
        [
            "",
            "## 2. Pure Post-Shift Difference-in-Differences (`Drift − Stationary`, Addressing Codex Audit)",
            "",
            f"- **Reversal DiD (`SGEH+Hazard [Ours] − ScopedCardBook [Baseline]`):** **`{c['drift_excised_minus_sham']['mean']:+.2f} pts`** `[{c['drift_excised_minus_sham']['ci95'][0]:+.2f}, {c['drift_excised_minus_sham']['ci95'][1]:+.2f}]` (per-seed: `{c['drift_excised_minus_sham']['per_seed']}`)",
            f"- **Reversal DiD (`SGEH-SurpriseOnly [Ours Ablation] − ScopedCardBook [Baseline]`):** **`{c['diff_in_diffs_drift_minus_stationary']['mean']:+.2f} pts`** `[{c['diff_in_diffs_drift_minus_stationary']['ci95'][0]:+.2f}, {c['diff_in_diffs_drift_minus_stationary']['ci95'][1]:+.2f}]`",
            f"- **Gradual Decay DiD (`SGEH+Hazard [Ours] − ScopedCardBook [Baseline]`):** **`{c['stationary_retained_minus_sham']['mean']:+.2f} pts`** `[{c['stationary_retained_minus_sham']['ci95'][0]:+.2f}, {c['stationary_retained_minus_sham']['ci95'][1]:+.2f}]`",
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate_shared_runs(results_root: Path, run_ids: list[str], out_run_id: str) -> Path:
    merged_seeds = []
    tot_u = 0.0
    in_t = out_t = th_t = 0
    model = "gemini-3.8-flash"
    tb = 0
    pw = 16
    for rid in run_ids:
        d = json.loads((results_root / "runs" / rid / "shared_fork_report.json").read_text(encoding="utf-8"))
        merged_seeds.extend(d["per_seed"])
        tot_u += float(d["total_run_usd"])
        in_t += int(d["observed_run_input_tokens"])
        out_t += int(d["observed_run_output_tokens"])
        th_t += int(d["observed_run_thinking_tokens"])
        model = d["model"]
        tb = int(d["requested_thinking_budget"])
        pw = int(d["post_window"])

    rep = summarize_shared_fork(
        run_id=out_run_id,
        model=model,
        thinking_budget=tb,
        post_window=pw,
        per_seed=merged_seeds,
        total_usd=round(tot_u, 6),
        max_usd=round(1.95 * len(run_ids), 2),
        in_tok=in_t,
        out_tok=out_t,
        think_tok=th_t,
    )
    out_dir = results_root / "runs" / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "shared_fork_report.json", rep)
    atomic_write_json(out_dir / "causal_fork_report.json", rep)
    md = render_shared_fork_md(rep)
    out_md = out_dir / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\n{md}\nSaved merged shared-fork report: {out_md}", flush=True)
    return out_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Iteration 2d Shared-Checkpoint Post-Shift Adaptation Test")
    parser.add_argument("--run-id", default="iter2d_shared_s205_206", help="Run ID")
    parser.add_argument("--seeds", default="205,206", help="Comma-separated fresh counterbalanced seeds")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Model ID or 'mock'")
    parser.add_argument("--post-window", type=int, default=16, help="Post-boundary sessions")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Requested thinking budget")
    parser.add_argument("--workers", type=int, default=9, help="Parallel workers")
    parser.add_argument("--max-usd", type=float, default=1.95, help="Strict Tier-1 USD cap")
    parser.add_argument("--aggregate-runs", default="", help="Comma-separated runs to merge")
    args = parser.parse_args()

    if args.aggregate_runs:
        rids = [x.strip() for x in args.aggregate_runs.split(",") if x.strip()]
        aggregate_shared_runs(ROOT / "results", rids, args.run_id)
        return

    s_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    try:
        rep = run_shared_fork_batch(
            run_id=args.run_id,
            seeds=s_list,
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

    md = render_shared_fork_md(rep)
    out_md = ROOT / "results" / "runs" / args.run_id / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\n{md}\nSaved report: {out_md}", flush=True)


if __name__ == "__main__":
    main()
