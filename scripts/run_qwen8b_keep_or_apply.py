#!/usr/bin/env python3
"""Untuned `Qwen3-8B` Keep-or-Apply Reviewer Gatekeeper Evaluation (`One Step Toward Option A`).

Implements Astra/Codex's 28 Nov pre-training gatekeeper protocol
(`setup-debates/2026-11-27-h60-reviewer-self-improvement.md`, §"One step toward Option A"):
  - Evaluates **fresh, unseen `AlphaDesk` streams** (`Seeds 721–724` on `drift_meta_60` and `Seeds 725–728` on
    `drift_dip_c_60`), leaving the third setup (`CoalitionOps`) 100% untouched for the eventual transfer test.
  - Uses local untuned **`Qwen/Qwen3-8B`** (`FP16` on Quadro RTX 8000) as the Reviewer & Keep/Apply Gatekeeper, and
    **`gemini-3.8-flash`** as the fixed Actor (`same actor, tools, and 4-session budget`).
  - Preselects review opportunities **`PRESELECTED_EDIT_EPS = (16, 28, 36, 40)`** *before* any stream runs:
    1. At `t_edit`, `Qwen3-8B` sees its `current_rule`, its `proposed_rule`, and ONLY the public observations up to
       `t_edit`, and records its `KEEP` vs `APPLY` choice **strictly before** episode `t_edit + 1` runs.
    2. Because `review_rule` is invisible to the actor, episodes `t_edit + 1 .. t_use` (`t_use = t_edit + 4`) share
       the **exact same history** for both `Keep` and `Apply`.
    3. At `t_use` (`20, 32, 40, 44`), compares `current_rule` (`Keep`) vs `proposed_rule` (`Apply`) on that identical
       history across `K = 6` repeated `(1 Qwen3-8B review + 4 Flash actor sessions)` trials after probe & turnover
       fees, comparing **`Untuned Qwen3-8B Gatekeeper`** vs **`Always-Keep`** vs **`Always-Apply`** (plus the
       **`Counterfactual Oracle`** ceiling) and exporting `reviewer_gate_examples.jsonl` for later `8B` weight training.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys
import threading
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_meta_reviewer_h60 import _execute_single_episode  # noqa: E402
from src.core.ledger import BudgetExceededError, CostLedger, atomic_write_json  # noqa: E402
from src.core.llm_client import MeteredLLMClient  # noqa: E402
from src.core.local_hf_client import LocalHFMeteredClient  # noqa: E402
from src.envs.alpha_desk_v3 import AlphaDeskV3MetaHorizonEnv  # noqa: E402
from src.envs.base import PublicTaskSpec  # noqa: E402
from src.harnesses.procedure_harness import TwoLevelReviewerHarness  # noqa: E402

PRESELECTED_EDIT_EPS = (16, 28, 36, 44)


def _eval_rule_version_at_t_use(
    *,
    env_at_t_use: AlphaDeskV3MetaHorizonEnv,
    harness_at_t_use: TwoLevelReviewerHarness,
    rule_text: str,
    version_tag: str,
    t_use_1idx: int,
    k_trials: int,
    reviewer_model: LocalHFMeteredClient,
    actor_client: MeteredLLMClient,
    workers: int,
    seed_base: int,
    on_ep_done: Any = None,
) -> dict[str, Any]:
    """Run `K` repeated `(1 Qwen3-8B review + 4 Flash actor sessions)` trials for `rule_text` at `t_use`."""
    rollout_sessions = list(range(t_use_1idx + 1, min(env_at_t_use.horizon + 1, t_use_1idx + 1 + harness_at_t_use.review_cadence)))
    disp_ver = int(harness_at_t_use.rule_version)
    in_note_v = int(harness_at_t_use.note_version)

    def _one_trial(t_idx: int) -> dict[str, Any]:
        snap_h = copy.deepcopy(harness_at_t_use)
        env_c = copy.deepcopy(env_at_t_use)
        # Seeded light sampling (T=0.25) across K trials so both reviewer phrasing and actor execution reliability are measured
        trial_rev_client = reviewer_model.with_sampling(
            temperature=0.25 if k_trials > 1 else 0.0,
            sample_seed=seed_base + t_idx * 997,
        )
        rev_out = snap_h.revise_task_note_from_rule(
            env_name="alpha_desk_v3",
            episode_idx=t_use_1idx - 1,
            active_review_rule=rule_text,
            active_rule_version=disp_ver,
            client=trial_rev_client,
            method_suffix=f"tuse_{version_tag}_t{t_idx}",
        )
        prop_note = rev_out["proposed_task_note"] or snap_h.task_note
        snap_h.task_note = prop_note
        snap_h.note_version = in_note_v + 1

        net_sum = 0.0
        dp_sum = 0.0
        probes_sum = 0
        turn_sum = 0.0
        usd_sum = 0.0
        for s_1idx in rollout_sessions:
            ep_out = _execute_single_episode(
                env=env_c,
                ep=s_1idx - 1,
                harness=snap_h,
                client=actor_client,
                method_tag=f"gate8b_eval_{version_tag}_t{t_idx}",
            )
            snap_h.factual.on_episode_complete(ep_out["pub_rec"], actor_client)
            priv = ep_out["priv"]
            net_sum += float(priv.get("expected_net_no_noise", 0.0))
            dp_sum += float(priv.get("dp_oracle_step_net", 85.0))
            probes_sum += int(priv.get("probes_used", 0))
            turn_sum += float(priv.get("turnover_fee", 0.0))
            usd_sum += float(ep_out["acting_usd"])
            if on_ep_done is not None:
                on_ep_done()

        return {
            "trial": t_idx,
            "proposed_task_note": prop_note,
            "block_expected_net": round(net_sum, 3),
            "block_dp_net": round(dp_sum, 3),
            "block_dp_pct": round(100.0 * net_sum / max(1.0, dp_sum), 2),
            "block_probes": probes_sum,
            "block_turnover_pts": round(turn_sum, 2),
            "actor_usd": round(usd_sum, 6),
        }

    trials: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(k_trials, workers))) as pool:
        futs = [pool.submit(_one_trial, ti) for ti in range(1, k_trials + 1)]
        for f in as_completed(futs):
            trials.append(f.result())
    trials.sort(key=lambda x: int(x["trial"]))

    dp_vals = [float(t["block_dp_pct"]) for t in trials]
    net_vals = [float(t["block_expected_net"]) for t in trials]
    return {
        "rule_text": rule_text,
        "k_trials": k_trials,
        "rollout_sessions": rollout_sessions,
        "mean_dp_pct": round(statistics.mean(dp_vals), 2),
        "sd_dp_pct": round(statistics.stdev(dp_vals), 2) if len(dp_vals) > 1 else 0.0,
        "per_trial_dp_pct": [round(v, 2) for v in dp_vals],
        "mean_net_pts": round(statistics.mean(net_vals), 2),
        "mean_probes": round(statistics.mean(float(t["block_probes"]) for t in trials), 2),
        "mean_turnover_pts": round(statistics.mean(float(t["block_turnover_pts"]) for t in trials), 2),
        "unique_notes": len({t["proposed_task_note"].strip() for t in trials}),
        "total_actor_usd": round(sum(float(t["actor_usd"]) for t in trials), 6),
        "sample_task_note": trials[0]["proposed_task_note"] if trials else "",
    }


def run_seed_keep_or_apply_stream(
    *,
    seed: int,
    meta_mode: str,
    horizon: int,
    preselected_eps: tuple[int, ...],
    k_trials: int,
    workers: int,
    reviewer_client: LocalHFMeteredClient,
    actor_client: MeteredLLMClient,
    on_ep_done: Any = None,
) -> dict[str, Any]:
    """Run one fresh stream up to `max(preselected_eps) + 4`, evaluating preselected Keep-or-Apply opportunities."""
    env = AlphaDeskV3MetaHorizonEnv(meta_mode=meta_mode, seed=seed, horizon=horizon)
    assert env.env_name == "alpha_desk_v3"  # Guarantee held-out CoalitionOps is never touched!

    harness = TwoLevelReviewerHarness(mode="editable_rule", review_cadence=4, recent_k=8)
    max_backbone_ep = min(horizon, max(preselected_eps) + harness.review_cadence)

    pending_opportunities: dict[int, dict[str, Any]] = {}
    evaluated_opportunities: list[dict[str, Any]] = []
    backbone_logs: list[dict[str, Any]] = []

    for ep_0idx in range(0, max_backbone_ep):
        ep_1idx = ep_0idx + 1
        ep_out = _execute_single_episode(
            env=env,
            ep=ep_0idx,
            harness=harness,
            client=actor_client,
            method_tag="gate8b_backbone_actor",
        )
        ep_completed_utc = datetime.now(timezone.utc).isoformat()
        pub_rec = ep_out["pub_rec"]
        harness.factual.on_episode_complete(pub_rec, actor_client)
        if on_ep_done is not None:
            on_ep_done()

        priv = ep_out["priv"]
        backbone_logs.append(
            {
                "episode": ep_1idx,
                "completed_at_utc": ep_completed_utc,
                "window": int(priv.get("window", env.window_index(ep_0idx))),
                "regime": str(priv.get("regime", "")),
                "public_score": float(ep_out["final_score"]),
                "expected_net_no_noise": float(priv.get("expected_net_no_noise", 0.0)),
                "dp_oracle_step_net": float(priv.get("dp_oracle_step_net", 85.0)),
                "probes_used": int(priv.get("probes_used", 0)),
            }
        )

        # 1. If `ep_1idx` is `t_use` (`t_edit + 4`) for a preselected opportunity recorded 4 sessions ago,
        #    evaluate BOTH `current_rule` (`Keep`) and `proposed_rule` (`Apply`) on this EXACT SAME history!
        if ep_1idx in pending_opportunities:
            opp = pending_opportunities.pop(ep_1idx)
            r_keep = str(opp["current_rule"])
            r_apply = str(opp["proposed_rule"])
            identical_text = r_keep.strip() == r_apply.strip()

            keep_eval = _eval_rule_version_at_t_use(
                env_at_t_use=env,
                harness_at_t_use=harness,
                rule_text=r_keep,
                version_tag="keep",
                t_use_1idx=ep_1idx,
                k_trials=k_trials,
                reviewer_model=reviewer_client,
                actor_client=actor_client,
                workers=workers,
                seed_base=seed * 10_000 + ep_1idx * 100 + 1,
                on_ep_done=on_ep_done,
            )
            if identical_text:
                apply_eval = copy.deepcopy(keep_eval)
            else:
                apply_eval = _eval_rule_version_at_t_use(
                    env_at_t_use=env,
                    harness_at_t_use=harness,
                    rule_text=r_apply,
                    version_tag="apply",
                    t_use_1idx=ep_1idx,
                    k_trials=k_trials,
                    reviewer_model=reviewer_client,
                    actor_client=actor_client,
                    workers=workers,
                    seed_base=seed * 10_000 + ep_1idx * 100 + 51,
                    on_ep_done=on_ep_done,
                )

            v_keep = float(keep_eval["mean_dp_pct"])
            v_apply = float(apply_eval["mean_dp_pct"])
            choice = str(opp["qwen8b_decision"])
            v_qwen8b = v_apply if choice == "APPLY" else v_keep
            v_oracle = max(v_keep, v_apply)

            evaluated_opportunities.append(
                {
                    **opp,
                    "t_use_episode": ep_1idx,
                    "first_post_choice_outcome_utc": backbone_logs[int(opp["t_edit_episode"])]["completed_at_utc"],
                    "identical_rule_text": identical_text,
                    "keep_eval": keep_eval,
                    "apply_eval": apply_eval,
                    "v_keep_dp_pct": round(v_keep, 2),
                    "v_apply_dp_pct": round(v_apply, 2),
                    "v_qwen8b_dp_pct": round(v_qwen8b, 2),
                    "v_oracle_dp_pct": round(v_oracle, 2),
                    "apply_minus_keep_dp_pct": round(v_apply - v_keep, 2),
                    "qwen8b_minus_keep_dp_pct": round(v_qwen8b - v_keep, 2),
                    "qwen8b_minus_apply_dp_pct": round(v_qwen8b - v_apply, 2),
                    "apply_minus_keep_net_pts": round(float(apply_eval["mean_net_pts"]) - float(keep_eval["mean_net_pts"]), 2),
                }
            )

        # 2. Periodic 4-session review by untuned `Qwen3-8B` (cleanly separated: note revision -> candidate rule proposal -> keep/apply gate)
        if len(harness.factual.history) % harness.review_cadence == 0:
            note_out = harness.revise_task_note_from_rule(
                env_name="alpha_desk_v3",
                episode_idx=ep_0idx,
                active_review_rule=harness.review_rule,
                active_rule_version=harness.rule_version,
                client=reviewer_client,
                method_suffix="backbone_revise_note",
            )
            prop_note = note_out["proposed_task_note"]
            if prop_note:
                if prop_note != harness.task_note or harness.note_version == 0:
                    harness.task_note = prop_note
                    harness.note_version += 1

            # During W1 acquisition (`ep_1idx <= env.t_w2`) when initializing from blank, acquire initial rule cleanly
            is_init_phase = ep_1idx <= env.t_w2 or not harness.review_rule.strip()
            prop_client = (
                reviewer_client
                if is_init_phase and not harness.review_rule.strip()
                else reviewer_client.with_sampling(temperature=0.6, sample_seed=seed * 1000 + ep_1idx)
            )
            prop_out = harness.propose_replacement_review_rule(
                env_name="alpha_desk_v3",
                episode_idx=ep_0idx,
                current_rule=harness.review_rule,
                active_task_note=harness.task_note,
                client=prop_client,
            )
            prop_rule = prop_out["proposed_review_rule"]

            if is_init_phase:
                if prop_rule and prop_rule != harness.review_rule:
                    harness.review_rule = prop_rule
                    harness.rule_version += 1
                continue

            cur_rule = harness.review_rule
            cand_rule = prop_rule or cur_rule

            # Ask untuned `Qwen3-8B` (`T=0.0`) to choose `KEEP` vs `APPLY` strictly BEFORE episode `ep_1idx + 1`!
            gate_out = harness.choose_keep_or_apply_rule(
                env_name="alpha_desk_v3",
                episode_idx=ep_0idx,
                current_rule=cur_rule,
                proposed_rule=cand_rule,
                active_task_note=harness.task_note,
                client=reviewer_client,
            )
            decision = str(gate_out["decision"])

            if ep_1idx in preselected_eps and (ep_1idx + harness.review_cadence) <= horizon:
                fact_snapshot = harness.factual.build_context(
                    PublicTaskSpec(
                        env_name="alpha_desk_v3",
                        episode_idx=ep_0idx,
                        timestamp=ep_1idx,
                        domain_prompt="",
                        task_prompt="",
                        tool_schemas=[],
                    )
                )
                pending_opportunities[ep_1idx + harness.review_cadence] = {
                    "seed": seed,
                    "meta_mode": meta_mode,
                    "window_at_edit": int(env.window_index(ep_0idx)) + 1,
                    "window_ranges": list(env.window_ranges),
                    "t_edit_episode": ep_1idx,
                    "decided_before_next_ep_utc": gate_out["decided_at_utc"],
                    "current_rule": cur_rule,
                    "proposed_rule": cand_rule,
                    "active_task_note_at_edit": harness.task_note,
                    "factual_ledger_at_edit": fact_snapshot,
                    "qwen8b_decision": decision,
                    "qwen8b_rationale": gate_out["rationale"],
                }

            # Apply `Qwen3-8B`'s choice to advance the backbone's active `review_rule`
            if decision == "APPLY" and cand_rule != harness.review_rule:
                harness.review_rule = cand_rule
                harness.rule_version += 1

    # Verify causal timestamp ordering (`decided_before_next_ep_utc <= first_post_choice_outcome_utc`)
    for opp in evaluated_opportunities:
        assert opp["decided_before_next_ep_utc"] <= opp["first_post_choice_outcome_utc"]

    return {
        "seed": seed,
        "meta_mode": meta_mode,
        "window_ranges": list(env.window_ranges),
        "opportunities": evaluated_opportunities,
        "backbone_logs": backbone_logs,
    }


def _bootstrap_ci(vals: list[float], seed: int = 20261128) -> dict[str, Any]:
    if not vals:
        return {"mean": 0.0, "ci95": [0.0, 0.0], "n": 0}
    m = statistics.mean(vals)
    if len(vals) == 1:
        return {"mean": round(m, 2), "ci95": [round(m, 2), round(m, 2)], "n": 1}
    rng = random.Random(seed)
    n = len(vals)
    boots = sorted(statistics.mean(vals[rng.randrange(n)] for _ in range(n)) for _ in range(5000))
    return {
        "mean": round(m, 2),
        "ci95": [round(boots[int(0.025 * 5000)], 2), round(boots[int(0.975 * 5000)], 2)],
        "n": n,
    }


def summarize_keep_or_apply_report(
    *,
    run_id: str,
    reviewer_model_id: str,
    actor_model_id: str,
    k_trials: int,
    preselected_eps: list[int],
    streams: list[dict[str, Any]],
    total_usd: float,
    in_tok: int,
    out_tok: int,
    think_tok: int,
) -> dict[str, Any]:
    all_opps = [
        opp
        for st in sorted(streams, key=lambda s: (s["meta_mode"], int(s["seed"])))
        for opp in st["opportunities"]
    ]
    nontrivial = [o for o in all_opps if not o["identical_rule_text"]]
    material = [o for o in nontrivial if abs(float(o["apply_minus_keep_dp_pct"])) >= 2.0]

    def _agg_slice(sub: list[dict[str, Any]]) -> dict[str, Any]:
        if not sub:
            return {
                "n_opportunities": 0,
                "apply_rate_pct": 0.0,
                "always_keep_dp_pct": 0.0,
                "always_apply_dp_pct": 0.0,
                "qwen8b_gate_dp_pct": 0.0,
                "oracle_gate_dp_pct": 0.0,
                "qwen8b_minus_keep": {"mean": 0.0, "ci95": [0.0, 0.0], "n": 0},
                "qwen8b_minus_apply": {"mean": 0.0, "ci95": [0.0, 0.0], "n": 0},
                "apply_minus_keep": {"mean": 0.0, "ci95": [0.0, 0.0], "n": 0},
                "oracle_headroom_over_qwen8b": {"mean": 0.0, "ci95": [0.0, 0.0], "n": 0},
            }
        return {
            "n_opportunities": len(sub),
            "apply_rate_pct": round(100.0 * sum(1 for o in sub if o["qwen8b_decision"] == "APPLY") / len(sub), 1),
            "always_keep_dp_pct": round(statistics.mean(float(o["v_keep_dp_pct"]) for o in sub), 2),
            "always_apply_dp_pct": round(statistics.mean(float(o["v_apply_dp_pct"]) for o in sub), 2),
            "qwen8b_gate_dp_pct": round(statistics.mean(float(o["v_qwen8b_dp_pct"]) for o in sub), 2),
            "oracle_gate_dp_pct": round(statistics.mean(float(o["v_oracle_dp_pct"]) for o in sub), 2),
            "qwen8b_minus_keep": _bootstrap_ci([float(o["qwen8b_minus_keep_dp_pct"]) for o in sub]),
            "qwen8b_minus_apply": _bootstrap_ci([float(o["qwen8b_minus_apply_dp_pct"]) for o in sub]),
            "apply_minus_keep": _bootstrap_ci([float(o["apply_minus_keep_dp_pct"]) for o in sub]),
            "oracle_headroom_over_qwen8b": _bootstrap_ci(
                [float(o["v_oracle_dp_pct"]) - float(o["v_qwen8b_dp_pct"]) for o in sub]
            ),
        }

    # Classification accuracy on material edits (`|Apply - Keep| >= 2.0 pts`)
    tp = sum(1 for o in material if o["qwen8b_decision"] == "APPLY" and float(o["apply_minus_keep_dp_pct"]) >= 2.0)
    tn = sum(1 for o in material if o["qwen8b_decision"] == "KEEP" and float(o["apply_minus_keep_dp_pct"]) <= -2.0)
    fp = sum(1 for o in material if o["qwen8b_decision"] == "APPLY" and float(o["apply_minus_keep_dp_pct"]) <= -2.0)
    fn = sum(1 for o in material if o["qwen8b_decision"] == "KEEP" and float(o["apply_minus_keep_dp_pct"]) >= 2.0)

    by_mode: dict[str, Any] = {}
    for md in sorted({str(o["meta_mode"]) for o in all_opps}):
        by_mode[md] = _agg_slice([o for o in all_opps if o["meta_mode"] == md])

    by_edit_ep: dict[str, Any] = {}
    for ep_k in preselected_eps:
        sub_e = [o for o in all_opps if int(o["t_edit_episode"]) == int(ep_k)]
        if sub_e:
            by_edit_ep[f"ep_{ep_k}_to_{ep_k + 4}"] = _agg_slice(sub_e)

    clean_opps = [
        {
            k: v
            for k, v in o.items()
            if k not in ("factual_ledger_at_edit", "keep_eval", "apply_eval")
        }
        | {
            "keep_probes": o["keep_eval"]["mean_probes"],
            "apply_probes": o["apply_eval"]["mean_probes"],
            "keep_sd": o["keep_eval"]["sd_dp_pct"],
            "apply_sd": o["apply_eval"]["sd_dp_pct"],
        }
        for o in all_opps
    ]

    return {
        "experiment_type": "qwen8b_keep_or_apply_gate",
        "run_id": run_id,
        "reviewer_model": reviewer_model_id,
        "actor_model": actor_model_id,
        "held_out_transfer_setup_untouched": "coalition_ops",
        "k_trials_per_version": k_trials,
        "preselected_edit_episodes": preselected_eps,
        "n_streams": len(streams),
        "seeds": sorted({int(s["seed"]) for s in streams}),
        "total_run_usd": round(total_usd, 6),
        "observed_run_input_tokens": in_tok,
        "observed_run_output_tokens": out_tok,
        "observed_run_thinking_tokens": think_tok,
        "all_preselected_summary": _agg_slice(all_opps),
        "distinct_proposal_summary": _agg_slice(nontrivial),
        "material_edit_confusion": {
            "threshold_pts": 2.0,
            "n_material": len(material),
            "helpful_edits_count": tp + fn,
            "harmful_edits_count": tn + fp,
            "true_positive_apply_helpful": tp,
            "true_negative_keep_harmful": tn,
            "false_positive_apply_harmful": fp,
            "false_negative_keep_helpful": fn,
            "accuracy_pct": round(100.0 * (tp + tn) / len(material), 1) if material else 0.0,
        },
        "by_meta_mode": by_mode,
        "by_edit_step": by_edit_ep,
        "opportunities": clean_opps,
    }


def render_keep_or_apply_markdown(rep: dict[str, Any]) -> str:
    all_s = rep["all_preselected_summary"]
    nt_s = rep["distinct_proposal_summary"]
    cm = rep["material_edit_confusion"]

    lines = [
        f"# Untuned `{rep['reviewer_model']}` Keep-or-Apply Reviewer Gatekeeper: `{rep['run_id']}`",
        "",
        f"- **Reviewer & Gatekeeper:** `{rep['reviewer_model']}` (`$0.00` local RTX 8000 FP16) | **Fixed Actor:** `{rep['actor_model']}` | **Held-Out Third Setup Untouched:** `{rep['held_out_transfer_setup_untouched']}`",
        f"- **Fresh Streams:** `Seeds {rep['seeds']}` (`N = {rep['n_streams']}`) | **Preselected Reviews:** `{rep['preselected_edit_episodes']}` (`M = {all_s['n_opportunities']}` opportunities) | **Eval Trials / Version:** `K = {rep['k_trials_per_version']}` | **Total Spend:** `${rep['total_run_usd']:.4f}`",
        "",
        "## 1. Gatekeeper Policy Comparison (Identical History, Actor, & 4-Session After-Fee Budget)",
        "",
        "| Slice | Opportunities (`M`) | `8B` Apply Rate | `Always-Keep` DP (%) | `Always-Apply` DP (%) | **Untuned `8B` Policy DP (%)** | `Oracle` Ceiling (%) | `8B − Keep` (95% CI) | `8B − Apply` (95% CI) | `Oracle − 8B` Headroom |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lbl, sl in [
        ("All Preselected Opportunities", all_s),
        ("Non-Identical Rule Proposals (`R_prop != R_cur`)", nt_s),
        *[(f"Mode: `{m}`", d) for m, d in rep["by_meta_mode"].items()],
        *[(f"Step: `{st}`", d) for st, d in rep["by_edit_step"].items()],
    ]:
        if sl["n_opportunities"] == 0:
            continue
        qk = sl["qwen8b_minus_keep"]
        qa = sl["qwen8b_minus_apply"]
        oh = sl["oracle_headroom_over_qwen8b"]
        lines.append(
            f"| {lbl} | `{sl['n_opportunities']}` | `{sl['apply_rate_pct']:.1f}%` | "
            f"{sl['always_keep_dp_pct']:.2f}% | {sl['always_apply_dp_pct']:.2f}% | "
            f"**{sl['qwen8b_gate_dp_pct']:.2f}%** | {sl['oracle_gate_dp_pct']:.2f}% | "
            f"**`{qk['mean']:+.2f}`** `[{qk['ci95'][0]:+.2f}, {qk['ci95'][1]:+.2f}]` | "
            f"**`{qa['mean']:+.2f}`** `[{qa['ci95'][0]:+.2f}, {qa['ci95'][1]:+.2f}]` | "
            f"`+{oh['mean']:.2f} pts` |"
        )

    lines.extend(
        [
            "",
            "## 2. Classification of Material Edits (`|V_apply − V_keep| >= 2.0 pts`)",
            "",
            f"- **Material Proposals:** `{cm['n_material']}` (`{cm['helpful_edits_count']}` helpful `>= +2.0 pts`, `{cm['harmful_edits_count']}` harmful `<= -2.0 pts`)",
            f"- **Untuned `8B` Gatekeeper Accuracy:** **`{cm['accuracy_pct']:.1f}%`** (`TP_apply_helpful = {cm['true_positive_apply_helpful']}`, `TN_keep_harmful = {cm['true_negative_keep_harmful']}`, `FP_apply_harmful = {cm['false_positive_apply_harmful']}`, `FN_keep_helpful = {cm['false_negative_keep_helpful']}`)",
            "",
            "## 3. Per-Opportunity Pre-Outcome Choices & Next-Review (`t_use`) Scores",
            "",
            "| Seed | Mode | `t_edit -> t_use` | `8B` Choice | `Keep` DP (`σ`) | `Apply` DP (`σ`) | `Apply − Keep` | `8B` Achieved | Distinct Rule? |",
            "|---:|---|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for o in rep["opportunities"]:
        lines.append(
            f"| `{o['seed']}` | `{o['meta_mode']}` | `{o['t_edit_episode']} -> {o['t_use_episode']}` | "
            f"**`{o['qwen8b_decision']}`** | {o['v_keep_dp_pct']:.1f}% (`{o['keep_sd']:.1f}`) | "
            f"{o['v_apply_dp_pct']:.1f}% (`{o['apply_sd']:.1f}`) | **`{o['apply_minus_keep_dp_pct']:+.2f}`** | "
            f"{o['v_qwen8b_dp_pct']:.1f}% | `{not o['identical_rule_text']}` |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _write_training_examples_jsonl(path: Path, streams: list[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for st in sorted(streams, key=lambda s: (s["meta_mode"], int(s["seed"]))):
            for o in st["opportunities"]:
                row = {
                    "seed": o["seed"],
                    "meta_mode": o["meta_mode"],
                    "t_edit_episode": o["t_edit_episode"],
                    "t_use_episode": o["t_use_episode"],
                    "decided_before_next_ep_utc": o["decided_before_next_ep_utc"],
                    "current_rule": o["current_rule"],
                    "proposed_rule": o["proposed_rule"],
                    "identical_rule_text": o["identical_rule_text"],
                    "active_task_note_at_edit": o["active_task_note_at_edit"],
                    "factual_ledger_at_edit": o["factual_ledger_at_edit"],
                    "untuned_qwen8b_decision": o["qwen8b_decision"],
                    "untuned_qwen8b_rationale": o["qwen8b_rationale"],
                    "v_keep_dp_pct": o["v_keep_dp_pct"],
                    "v_apply_dp_pct": o["v_apply_dp_pct"],
                    "apply_minus_keep_dp_pct": o["apply_minus_keep_dp_pct"],
                    "apply_minus_keep_net_pts": o["apply_minus_keep_net_pts"],
                    "optimal_action": "APPLY" if float(o["apply_minus_keep_dp_pct"]) > 0.0 else "KEEP",
                }
                fh.write(json.dumps(row) + "\n")
                count += 1
    return count


def run_keep_or_apply_batch(
    *,
    run_id: str,
    reviewer_model_id: str,
    actor_model_id: str,
    seeds: list[int],
    meta_mode: str,
    horizon: int,
    preselected_eps: tuple[int, ...],
    k_trials: int,
    workers: int,
    max_usd: float,
    results_root: Path,
    dotenv_path: Path | None,
) -> dict[str, Any]:
    ledger = CostLedger(results_root=results_root, run_id=run_id, phase="dev", max_usd=max_usd)
    reviewer_client = LocalHFMeteredClient(model=reviewer_model_id, ledger=ledger, temperature=0.0)
    actor_client = MeteredLLMClient(
        model=actor_model_id,
        ledger=ledger,
        dotenv_path=None if actor_model_id == "mock" else dotenv_path,
        thinking_budget=0,
        temperature=0.0,
    )

    backbone_len = min(horizon, max(preselected_eps) + 4)
    est_eps_per_seed = backbone_len + len(preselected_eps) * 2 * k_trials * 4
    total_units = max(1, len(seeds) * est_eps_per_seed)
    done_units = 0
    lock = threading.Lock()

    def _tick(lbl: str) -> None:
        nonlocal done_units
        with lock:
            done_units += 1
            if done_units % 6 == 0 or done_units == total_units:
                ledger.update_progress(
                    completed_units=min(done_units, total_units),
                    total_units=total_units,
                    status="running",
                    current_label=lbl,
                    log_every=30,
                )

    streams: list[dict[str, Any]] = []
    for sd in seeds:
        st = run_seed_keep_or_apply_stream(
            seed=sd,
            meta_mode=meta_mode,
            horizon=horizon,
            preselected_eps=preselected_eps,
            k_trials=k_trials,
            workers=workers,
            reviewer_client=reviewer_client,
            actor_client=actor_client,
            on_ep_done=lambda s=sd: _tick(f"gate8b {meta_mode} s{s}"),
        )
        streams.append(st)
        for o in st["opportunities"]:
            print(
                f"  [s{sd} {meta_mode} | ep {o['t_edit_episode']}->{o['t_use_episode']}] "
                f"8B={o['qwen8b_decision']:5s} | Keep={o['v_keep_dp_pct']:5.1f}% Apply={o['v_apply_dp_pct']:5.1f}% "
                f"(Δ={o['apply_minus_keep_dp_pct']:+6.2f}) | Run=${ledger.run_usd:.4f}",
                flush=True,
            )

    atomic_write_json(ledger.run_dir / "keep_or_apply_streams_full.json", {"streams": streams})
    _write_training_examples_jsonl(ledger.run_dir / "reviewer_gate_examples.jsonl", streams)

    rep = summarize_keep_or_apply_report(
        run_id=run_id,
        reviewer_model_id=reviewer_model_id,
        actor_model_id=actor_model_id,
        k_trials=k_trials,
        preselected_eps=list(preselected_eps),
        streams=streams,
        total_usd=ledger.run_usd,
        in_tok=ledger.run_input_tokens,
        out_tok=ledger.run_output_tokens,
        think_tok=ledger.run_thinking_tokens,
    )
    atomic_write_json(ledger.run_dir / "keep_or_apply_report.json", rep)
    md = render_keep_or_apply_markdown(rep)
    (ledger.run_dir / "summary.md").write_text(md, encoding="utf-8")
    ledger.update_progress(
        completed_units=total_units,
        total_units=total_units,
        status="completed",
        current_label=f"completed {run_id}",
    )
    return rep


def aggregate_keep_or_apply_runs(run_ids: list[str], out_run_id: str, results_root: Path) -> Path:
    all_streams: list[dict[str, Any]] = []
    tot_usd = 0.0
    in_t = out_t = th_t = 0
    rev_mdl = "Qwen/Qwen3-8B"
    act_mdl = "gemini-3.8-flash"
    k_trials = 6
    pre_eps: list[int] = list(PRESELECTED_EDIT_EPS)

    for rid in run_ids:
        rdir = results_root / "runs" / rid
        rep_d = json.loads((rdir / "keep_or_apply_report.json").read_text(encoding="utf-8"))
        rev_mdl = str(rep_d.get("reviewer_model", rev_mdl))
        act_mdl = str(rep_d.get("actor_model", act_mdl))
        k_trials = int(rep_d.get("k_trials_per_version", k_trials))
        pre_eps = list(rep_d.get("preselected_edit_episodes", pre_eps))
        tot_usd += float(rep_d.get("total_run_usd", 0.0))
        in_t += int(rep_d.get("observed_run_input_tokens", 0))
        out_t += int(rep_d.get("observed_run_output_tokens", 0))
        th_t += int(rep_d.get("observed_run_thinking_tokens", 0))
        full_d = json.loads((rdir / "keep_or_apply_streams_full.json").read_text(encoding="utf-8"))
        all_streams.extend(full_d.get("streams", []))

    out_dir = results_root / "runs" / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "keep_or_apply_streams_full.json", {"streams": all_streams})
    _write_training_examples_jsonl(out_dir / "reviewer_gate_examples.jsonl", all_streams)

    rep = summarize_keep_or_apply_report(
        run_id=out_run_id,
        reviewer_model_id=rev_mdl,
        actor_model_id=act_mdl,
        k_trials=k_trials,
        preselected_eps=pre_eps,
        streams=all_streams,
        total_usd=tot_usd,
        in_tok=in_t,
        out_tok=out_t,
        think_tok=th_t,
    )
    atomic_write_json(out_dir / "keep_or_apply_report.json", rep)
    md = render_keep_or_apply_markdown(rep)
    out_md = out_dir / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    return out_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Untuned Qwen3-8B Keep-or-Apply Reviewer Gatekeeper Runner")
    parser.add_argument("--run-id", default="qwen8b_gate_dev", help="Run ID")
    parser.add_argument("--reviewer-model", default="Qwen/Qwen3-8B", help="Local HF reviewer/gatekeeper model")
    parser.add_argument("--actor-model", default="gemini-3.8-flash", help="Fixed actor model")
    parser.add_argument("--seeds", default="721", help="Comma-separated fresh seeds")
    parser.add_argument("--meta-mode", choices=("drift_meta_60", "drift_dip_c_60", "stationary_meta_60"), default="drift_meta_60")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--preselected-eps", default="16,28,36,44", help="Comma-separated preselected review episodes")
    parser.add_argument("--k-trials", type=int, default=6, help="Repeated trials per rule version at t_use")
    parser.add_argument("--workers", type=int, default=6, help="Parallel actor threads")
    parser.add_argument("--max-usd", type=float, default=1.95, help="Hard USD cap per shard")
    parser.add_argument("--results-root", default=str(ROOT / "results"))
    parser.add_argument("--aggregate-runs", default="", help="Comma-separated shard run IDs to merge")
    args = parser.parse_args()

    res_root = Path(args.results_root)
    if args.aggregate_runs:
        rids = [x.strip() for x in args.aggregate_runs.split(",") if x.strip()]
        out_md = aggregate_keep_or_apply_runs(rids, args.run_id, res_root)
        print(out_md.read_text(encoding="utf-8"))
        return

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    pre_eps = tuple(int(x.strip()) for x in args.preselected_eps.split(",") if x.strip())

    try:
        rep = run_keep_or_apply_batch(
            run_id=args.run_id,
            reviewer_model_id=args.reviewer_model,
            actor_model_id=args.actor_model,
            seeds=seeds,
            meta_mode=args.meta_mode,
            horizon=args.horizon,
            preselected_eps=pre_eps,
            k_trials=args.k_trials,
            workers=args.workers,
            max_usd=args.max_usd,
            results_root=res_root,
            dotenv_path=ROOT / ".env",
        )
    except BudgetExceededError as exc:
        print(f"[BUDGET CIRCUIT BREAKER] {exc}", flush=True)
        sys.exit(2)

    print("\n" + render_keep_or_apply_markdown(rep), flush=True)


if __name__ == "__main__":
    main()
