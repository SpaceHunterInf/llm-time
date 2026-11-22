#!/usr/bin/env python3
"""Iteration 1b: Single-Variable Paired 3-Branch Causal Ratchet Runner on AlphaDesk-v0.2."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import statistics
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_dashboard import build_static_snapshot  # noqa: E402
from src.core.ledger import BudgetExceededError, CostLedger, atomic_write_json, send_monitor_email  # noqa: E402
from src.core.llm_client import MeteredLLMClient  # noqa: E402
from src.envs.alpha_desk_v2 import AlphaDeskV2Env, STRATEGIES_V2  # noqa: E402
from src.harnesses.base import PublicEpisodeRecord  # noqa: E402
from src.harnesses.scoped_cards import ScopedCardBookHarness  # noqa: E402
from src.runner import _build_acting_system_prompt  # noqa: E402


def _mock_v2_actor(steps_so_far: list[dict[str, Any]], sys_prompt: str) -> dict[str, Any]:
    if not steps_so_far:
        return {"tool": "inspect_desk_brief", "args": {}, "rationale": "Read desk brief."}
    obs = steps_so_far[-1].get("obs", {})
    if obs.get("volatility_regime_hint") == "elevated_volatility":
        w = {"STRAT_ALPHA": 0.05, "STRAT_BETA": 0.05, "STRAT_GAMMA": 0.25, "STRAT_DELTA": 0.65}
    elif '"STRAT_ALPHA": null' in sys_prompt and len(steps_so_far) == 1:
        return {"tool": "probe_strategy_execution", "args": {"strategy": "STRAT_GAMMA"}, "rationale": "Probe unranked."}
    elif any(st.get("obs", {}).get("live_net_yield_estimate", 0) >= 80 for st in steps_so_far):
        w = {"STRAT_ALPHA": 0.05, "STRAT_BETA": 0.05, "STRAT_GAMMA": 0.65, "STRAT_DELTA": 0.25}
    else:
        w = {"STRAT_ALPHA": 0.65, "STRAT_BETA": 0.25, "STRAT_GAMMA": 0.05, "STRAT_DELTA": 0.05}
    return {"tool": "submit_allocation", "args": {"weights": w}, "rationale": "Commit weights."}


def _run_episode_range(
    *,
    env: AlphaDeskV2Env,
    harness: ScopedCardBookHarness,
    client: MeteredLLMClient,
    ep_indices: range,
    method_tag: str,
    force_final_curate: bool = False,
    on_ep_done: Any = None,
) -> dict[str, Any]:
    logs: list[dict[str, Any]] = []
    acting_calls = 0
    reflection_calls = 0
    acting_usd = 0.0
    reflection_usd = 0.0
    acting_thinking_tokens = 0
    ep_list = list(ep_indices)

    for idx_in_slice, ep in enumerate(ep_list):
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
                mock_responder=lambda _sp, _up, _st=list(steps), _sys=sys_prompt: _mock_v2_actor(_st, _sys),
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
                "submitted_weights": {s: 0.0 for s in STRATEGIES_V2},
            }

        pub_rec = PublicEpisodeRecord(
            env_name=spec.env_name,
            episode_idx=ep,
            timestamp=ep + 1,
            task_prompt=spec.task_prompt,
            steps=steps,
            public_score=final_score,
        )
        is_last_pre = force_final_curate and (idx_in_slice == len(ep_list) - 1)
        h_meta = harness.on_episode_complete(pub_rec, client, force_curate=is_last_pre)
        reflection_calls += int(h_meta.get("reflection_calls", 0))
        reflection_usd += float(h_meta.get("reflection_usd", 0.0))

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
                "harness_meta": h_meta,
                "steps": steps,
            }
        )
        if on_ep_done is not None:
            on_ep_done()

    norm_u = [100.0 * r["normalized_utility"] for r in logs]
    gross_u = [100.0 * r["gross_normalized_utility"] for r in logs]
    calm_rows = [r for r in logs if not r["is_high_vol"]]
    hv_rows = [r for r in logs if r["is_high_vol"]]
    return {
        "episodes": len(logs),
        "mean_score": round(statistics.mean(r["public_score"] for r in logs), 2) if logs else 0.0,
        "mean_utility_pct": round(statistics.mean(norm_u), 2) if norm_u else 0.0,
        "gross_utility_pct": round(statistics.mean(gross_u), 2) if gross_u else 0.0,
        "calm_utility_pct": round(100.0 * statistics.mean(r["normalized_utility"] for r in calm_rows), 2) if calm_rows else 0.0,
        "high_vol_hedge_utility_pct": round(100.0 * statistics.mean(r["normalized_utility"] for r in hv_rows), 2) if hv_rows else 0.0,
        "stale_lock_in_pct": round(100.0 * sum(1 for r in logs if r["stale_policy_match"]) / max(1, len(logs)), 2),
        "invariant_violation_pct": round(100.0 * sum(1 for r in logs if r["invariant_violation"]) / max(1, len(logs)), 2),
        "probes_total": sum(r["probes_used"] for r in logs),
        "acting_calls": acting_calls,
        "reflection_calls": reflection_calls,
        "acting_thinking_tokens": acting_thinking_tokens,
        "usd": round(acting_usd + reflection_usd, 6),
        "curve_utility": [round(x, 1) for x in norm_u],
        "logs": logs,
    }


def bootstrap_mean_ci(diffs: list[float], n_boot: int = 2000, seed: int = 909) -> tuple[float, float, float]:
    if not diffs:
        return 0.0, 0.0, 0.0
    mean_v = statistics.mean(diffs)
    if len(diffs) == 1:
        return round(mean_v, 2), round(mean_v, 2), round(mean_v, 2)
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choice(diffs) for _ in range(len(diffs)))
        for _ in range(n_boot)
    )
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return round(mean_v, 2), round(lo, 2), round(hi, 2)


def run_causal_experiment(
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
    ledger = CostLedger(
        results_root=results_root,
        run_id=run_id,
        phase="dev",
        max_usd=max_usd,
    )
    est_total_eps = len(seeds) * (15 + 6 * post_window)
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

    per_seed_results: list[dict[str, Any]] = []
    traces_dir = ledger.results_root / "raw" / ledger.run_id
    traces_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = ledger.run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        pre_env = AlphaDeskV2Env(mode="stationary", horizon=40, seed=seed)
        t_star = pre_env.t_star
        print(f"\n[SEED {seed}] Hidden boundary t*={t_star} | Running shared pre-shift acquisition (0..{t_star - 1})...", flush=True)

        base_harness = ScopedCardBookHarness(reflect_every=3)
        pre_res = _run_episode_range(
            env=pre_env,
            harness=base_harness,
            client=client,
            ep_indices=range(0, t_star),
            method_tag="scoped_cards_preshift",
            force_final_curate=True,  # Consolidate cleanly at t*-1 without duplicating history!
            on_ep_done=lambda s=seed: _tick(f"seed {s} pre-shift"),
        )
        # Align post-boundary curation clock (`t*+2, t*+5, ...`) identically across all 3 arms!
        base_harness.set_post_fork_schedule()

        h_retained = base_harness.clone()
        h_sham, sham_usd = base_harness.make_sham_edited_copy(client, env_name="alpha_desk_v2")
        h_excised = h_sham.make_utility_excised_from_sham()

        fork_spec = pre_env.start_episode(t_star)
        rendered_prompts = {
            arm_k: _build_acting_system_prompt(fork_spec.domain_prompt, fork_spec.tool_schemas, arm_h.build_context(fork_spec))
            for arm_k, arm_h in (("retained", h_retained), ("sham", h_sham), ("excised", h_excised))
        }
        snapshot_record = {
            "seed": seed,
            "t_star": t_star,
            "fork_offset": h_retained.fork_offset,
            "history_lengths_at_fork": {
                "retained": len(h_retained._history),
                "sham": len(h_sham._history),
                "excised": len(h_excised._history),
            },
            "retained_cards": h_retained.cards,
            "sham_cards": h_sham.cards,
            "excised_cards": h_excised.cards,
            "rendered_actor_system_prompts_at_fork": rendered_prompts,
            "context_char_lengths": {k: len(v) for k, v in rendered_prompts.items()},
            "sham_edit_usd": round(sham_usd, 6),
        }
        atomic_write_json(ledger.run_dir / f"checkpoint_seed_{seed}.json", snapshot_record)
        print(
            f"  [FORK t*={t_star}] Actor prompt chars={snapshot_record['context_char_lengths']} | "
            f"History len={snapshot_record['history_lengths_at_fork']} -> 6 post-boundary branches ({post_window} eps)...",
            flush=True,
        )

        branch_defs = [
            ("abrupt_reversal", "retained", h_retained.clone()),
            ("abrupt_reversal", "sham", h_sham.clone()),
            ("abrupt_reversal", "excised", h_excised.clone()),
            ("stationary", "retained", h_retained.clone()),
            ("stationary", "sham", h_sham.clone()),
            ("stationary", "excised", h_excised.clone()),
        ]

        branch_outputs: dict[tuple[str, str], dict[str, Any]] = {}

        def _exec_branch(mode_name: str, arm_name: str, h_copy: ScopedCardBookHarness) -> tuple[str, str, dict[str, Any]]:
            b_env = AlphaDeskV2Env(mode=mode_name, horizon=40, seed=seed, t_star_override=t_star)
            res = _run_episode_range(
                env=b_env,
                harness=h_copy,
                client=client,
                ep_indices=range(t_star, t_star + post_window),
                method_tag=f"scoped_{arm_name}",
                force_final_curate=False,
                on_ep_done=lambda s=seed, m=mode_name, a=arm_name: _tick(f"s{s} {m}:{a}"),
            )
            return mode_name, arm_name, res

        with ThreadPoolExecutor(max_workers=max(1, min(6, workers))) as pool:
            futs = [pool.submit(_exec_branch, m, a, hc) for m, a, hc in branch_defs]
            for fut in as_completed(futs):
                m_name, a_name, b_res = fut.result()
                branch_outputs[(m_name, a_name)] = b_res
                print(
                    f"    -> [{m_name:15s} | {a_name:8s}] PostUtil={b_res['mean_utility_pct']:5.1f}% | "
                    f"CalmUtil={b_res['calm_utility_pct']:5.1f}% | HVHedge={b_res['high_vol_hedge_utility_pct']:5.1f}% | "
                    f"Probes={b_res['probes_total']:2d} | RefCalls={b_res['reflection_calls']} | Run=${ledger.run_usd:.4f}",
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
                    "total_usd": round(pre_res["usd"] / 6.0 + b_res["usd"], 6),
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

        drift_exc_minus_sham = round(
            branch_outputs[("abrupt_reversal", "excised")]["mean_utility_pct"]
            - branch_outputs[("abrupt_reversal", "sham")]["mean_utility_pct"],
            2,
        )
        drift_exc_minus_ret = round(
            branch_outputs[("abrupt_reversal", "excised")]["mean_utility_pct"]
            - branch_outputs[("abrupt_reversal", "retained")]["mean_utility_pct"],
            2,
        )
        drift_ret_minus_sham = round(
            branch_outputs[("abrupt_reversal", "retained")]["mean_utility_pct"]
            - branch_outputs[("abrupt_reversal", "sham")]["mean_utility_pct"],
            2,
        )
        stat_exc_minus_sham = round(
            branch_outputs[("stationary", "excised")]["mean_utility_pct"]
            - branch_outputs[("stationary", "sham")]["mean_utility_pct"],
            2,
        )
        stat_ret_minus_sham = round(
            branch_outputs[("stationary", "retained")]["mean_utility_pct"]
            - branch_outputs[("stationary", "sham")]["mean_utility_pct"],
            2,
        )

        per_seed_results.append(
            {
                "seed": seed,
                "t_star": t_star,
                "pre_shift_utility_pct": pre_res["mean_utility_pct"],
                "abrupt_reversal": {
                    a: {k: v for k, v in branch_outputs[("abrupt_reversal", a)].items() if k != "logs"}
                    for a in ("retained", "sham", "excised")
                },
                "stationary": {
                    a: {k: v for k, v in branch_outputs[("stationary", a)].items() if k != "logs"}
                    for a in ("retained", "sham", "excised")
                },
                "paired_contrasts": {
                    "drift_excised_minus_sham": drift_exc_minus_sham,
                    "drift_excised_minus_retained": drift_exc_minus_ret,
                    "drift_retained_minus_sham": drift_ret_minus_sham,
                    "stationary_excised_minus_sham": stat_exc_minus_sham,
                    "stationary_retained_minus_sham": stat_ret_minus_sham,
                    "difference_in_differences": round(drift_exc_minus_sham - stat_exc_minus_sham, 2),
                },
            }
        )

    c_drift_ratchet = bootstrap_mean_ci([r["paired_contrasts"]["drift_excised_minus_sham"] for r in per_seed_results])
    c_drift_vs_ret = bootstrap_mean_ci([r["paired_contrasts"]["drift_excised_minus_retained"] for r in per_seed_results])
    c_drift_rewrite = bootstrap_mean_ci([r["paired_contrasts"]["drift_retained_minus_sham"] for r in per_seed_results])
    c_stat_excise = bootstrap_mean_ci([r["paired_contrasts"]["stationary_excised_minus_sham"] for r in per_seed_results])
    c_stat_rewrite = bootstrap_mean_ci([r["paired_contrasts"]["stationary_retained_minus_sham"] for r in per_seed_results])
    c_did = bootstrap_mean_ci([r["paired_contrasts"]["difference_in_differences"] for r in per_seed_results])

    aggregate = {
        "run_id": run_id,
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
        "contrasts_mean_and_95ci": {
            "drift_excised_minus_sham": {"mean": c_drift_ratchet[0], "ci95": [c_drift_ratchet[1], c_drift_ratchet[2]]},
            "drift_excised_minus_retained": {"mean": c_drift_vs_ret[0], "ci95": [c_drift_vs_ret[1], c_drift_vs_ret[2]]},
            "drift_retained_minus_sham": {"mean": c_drift_rewrite[0], "ci95": [c_drift_rewrite[1], c_drift_rewrite[2]]},
            "stationary_excised_minus_sham": {"mean": c_stat_excise[0], "ci95": [c_stat_excise[1], c_stat_excise[2]]},
            "stationary_retained_minus_sham": {"mean": c_stat_rewrite[0], "ci95": [c_stat_rewrite[1], c_stat_rewrite[2]]},
            "diff_in_diffs_drift_minus_stationary": {"mean": c_did[0], "ci95": [c_did[1], c_did[2]]},
        },
        "per_seed": per_seed_results,
    }
    atomic_write_json(ledger.run_dir / "causal_fork_report.json", aggregate)
    ledger.update_progress(
        completed_units=est_total_eps,
        total_units=est_total_eps,
        status="completed",
        current_label="finished causal fork",
    )
    return aggregate


def render_markdown_report(agg: dict[str, Any]) -> str:
    c = agg["contrasts_mean_and_95ci"]
    lines = [
        f"# Iteration 1b Single-Variable Causal Fork Report: `{agg['run_id']}`",
        "",
        f"- **Model:** `{agg['model']}` | **Paired Seeds ($N$):** `{agg['n_seeds']}` (`{agg['seeds']}`) | **Post-Boundary Window:** `{agg['post_window']}` sessions",
        f"- **Requested `thinking_budget`:** `{agg['requested_thinking_budget']}` | **Observed Tokens:** `in={agg['observed_run_input_tokens']:,}`, `out={agg['observed_run_output_tokens']:,}`, `thinking={agg['observed_run_thinking_tokens']:,}`",
        f"- **Total Run Spend:** `${agg['total_run_usd']:.4f}` (Hard Cap: `${agg['max_usd_cap']:.2f}`)",
        "",
        "## 1. Seed-Paired Causal Contrasts (Normalized Post-Boundary Utility %)",
        "",
        "| Contrast | Mean Diff (pts) | 95% Bootstrap CI | Interpretation |",
        "|---|---:|---:|---|",
        f"| **Drift (`abrupt_reversal`): `Utility-Excised − Sham-Edited`** | **`{c['drift_excised_minus_sham']['mean']:+.2f}`** | `[{c['drift_excised_minus_sham']['ci95'][0]:+.2f}, {c['drift_excised_minus_sham']['ci95'][1]:+.2f}]` | **Primary Single-Variable Ratchet Contrast** (probing text, history & schedule held constant) |",
        f"| **Drift (`abrupt_reversal`): `Utility-Excised − Retained`** | `{c['drift_excised_minus_retained']['mean']:+.2f}` | `[{c['drift_excised_minus_retained']['ci95'][0]:+.2f}, {c['drift_excised_minus_retained']['ci95'][1]:+.2f}]` | Comparison against untouched pre-shift cards |",
        f"| **Stationary Control: `Utility-Excised − Sham-Edited`** | `{c['stationary_excised_minus_sham']['mean']:+.2f}` | `[{c['stationary_excised_minus_sham']['ci95'][0]:+.2f}, {c['stationary_excised_minus_sham']['ci95'][1]:+.2f}]` | Cost of removing valid calm ranking when world did NOT change |",
        f"| **Difference-in-Differences (`Drift − Stationary`)** | **`{c['diff_in_diffs_drift_minus_stationary']['mean']:+.2f}`** | `[{c['diff_in_diffs_drift_minus_stationary']['ci95'][0]:+.2f}, {c['diff_in_diffs_drift_minus_stationary']['ci95'][1]:+.2f}]` | Net regime-specific causal interaction |",
        f"| **Rewrite Control (Drift): `Retained − Sham-Edited`** | `{c['drift_retained_minus_sham']['mean']:+.2f}` | `[{c['drift_retained_minus_sham']['ci95'][0]:+.2f}, {c['drift_retained_minus_sham']['ci95'][1]:+.2f}]` | Scope-condition paraphrase check |",
        f"| **Rewrite Control (Stationary): `Retained − Sham-Edited`** | `{c['stationary_retained_minus_sham']['mean']:+.2f}` | `[{c['stationary_retained_minus_sham']['ci95'][0]:+.2f}, {c['stationary_retained_minus_sham']['ci95'][1]:+.2f}]` | Scope-condition paraphrase check |",
        "",
        "## 2. Per-Seed Breakdown (Including Probe Counts & Reflection Calls)",
        "",
        "| Seed | $t^*$ | Drift: Ret (Probes/Ref) | Drift: Sham (Probes/Ref) | Drift: Exc (Probes/Ref) | Drift `Exc−Sham` | Stat: Ret (Probes) | Stat: Sham (Probes) | Stat: Exc (Probes) | Stat `Exc−Sham` |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agg["per_seed"]:
        dr = row["abrupt_reversal"]
        st = row["stationary"]
        pc = row["paired_contrasts"]
        lines.append(
            f"| `{row['seed']}` | `{row['t_star']}` | "
            f"{dr['retained']['mean_utility_pct']:.1f}% ({dr['retained']['probes_total']}/{dr['retained']['reflection_calls']}) | "
            f"{dr['sham']['mean_utility_pct']:.1f}% ({dr['sham']['probes_total']}/{dr['sham']['reflection_calls']}) | "
            f"**{dr['excised']['mean_utility_pct']:.1f}%** ({dr['excised']['probes_total']}/{dr['excised']['reflection_calls']}) | "
            f"**`{pc['drift_excised_minus_sham']:+.1f}`** | "
            f"{st['retained']['mean_utility_pct']:.1f}% ({st['retained']['probes_total']}) | "
            f"{st['sham']['mean_utility_pct']:.1f}% ({st['sham']['probes_total']}) | "
            f"{st['excised']['mean_utility_pct']:.1f}% ({st['excised']['probes_total']}) | "
            f"`{pc['stationary_excised_minus_sham']:+.1f}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Iteration-1b Single-Variable Paired Causal Ratchet Experiment")
    parser.add_argument("--run-id", default="iter1b_clean_fork_n4", help="Run ID")
    parser.add_argument("--seeds", default="101,102,103,104", help="Comma-separated seeds")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Model ID or 'mock'")
    parser.add_argument("--post-window", type=int, default=16, help="Post-boundary sessions scored")
    parser.add_argument("--thinking-budget", type=int, default=0, help="Requested thinking budget per call")
    parser.add_argument("--workers", type=int, default=6, help="Parallel branch workers")
    parser.add_argument("--max-usd", type=float, default=1.95, help="Strict Tier-1 USD cap")
    parser.add_argument("--email-on-finish", action="store_true", help="Email report to gs06monitor@gmail.com")
    args = parser.parse_args()

    seed_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    try:
        agg = run_causal_experiment(
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

    md = render_markdown_report(agg)
    out_md = ROOT / "results" / "runs" / args.run_id / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    snap = build_static_snapshot(ROOT, ROOT / "results" / "supervisor_dashboard.html")
    print(f"\n{md}\nSaved report: {out_md}\nUpdated portal snapshot: {snap}", flush=True)

    if args.email_on_finish:
        send_monitor_email(
            subject=f"Iter1b Clean Fork ({args.run_id}) - ${agg['total_run_usd']:.4f}",
            body=md,
        )


if __name__ == "__main__":
    main()
