#!/usr/bin/env python3
"""ServeLearnBench-Mini (`SLB-Mini`): External Multi-Turn Evolving Policy Evaluation (<$1.95/run)."""

from __future__ import annotations

import argparse
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
SLB_ROOT = ROOT / "competitors" / "ServeLearnBench"
if SLB_ROOT.is_dir() and str(SLB_ROOT) not in sys.path:
    sys.path.insert(0, str(SLB_ROOT))

from servelearnbench.engine.env import Env  # type: ignore[import-not-found]  # noqa: E402
from servelearnbench.engine.types import Action, FINISH_ACTION_NAME  # type: ignore[import-not-found]  # noqa: E402
from servelearnbench.scenarios import get_bundle  # type: ignore[import-not-found]  # noqa: E402

from src.core.ledger import BudgetExceededError, CostLedger, atomic_write_json, send_monitor_email  # noqa: E402
from src.core.llm_client import MeteredLLMClient  # noqa: E402
from src.envs.base import PublicTaskSpec  # noqa: E402
from src.harnesses import make_harness  # noqa: E402
from src.harnesses.base import PublicEpisodeRecord  # noqa: E402


def _slb_finish_schema(domain: str) -> dict[str, Any]:
    if domain == "banking":
        return {
            "name": FINISH_ACTION_NAME,
            "description": "End the episode after your decision tool call (`approve_case`, `deny_case`, `escalate_case`, or `decide_batch`). Takes `{}`.",
            "args": {},
        }
    return {
        "name": FINISH_ACTION_NAME,
        "description": "End episode: `{}` after completed order action, `{\"answer\": ...}` for a question, or `{\"decision\": \"refuse\", \"reason\": \"<code>\"}` to decline.",
        "args": {"answer": "optional", "decision": "optional 'refuse'", "reason": "refusal code if refusing"},
    }


def _convert_slb_tools(tools_info: list[dict[str, Any]], domain: str) -> list[dict[str, Any]]:
    out = []
    for t in tools_info:
        fn = t.get("function", {})
        out.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "args": fn.get("parameters", {}).get("properties", {}),
            }
        )
    out.append(_slb_finish_schema(domain))
    return out


def stratified_window_tasks(
    tasks: list[Any], n_adapt: int, n_general: int, allowed_families: set[str] | None = None
) -> list[Any]:
    """Select deterministic chronological slice balancing `adapt` (shifted rules) and `general` (unchanged rules)."""
    pool = [t for t in tasks if (not allowed_families or str(t.z.get("family", "")) in allowed_families)]
    adapt = [t for t in pool if t.z.get("slice") == "adapt"][:n_adapt]
    general = [t for t in pool if t.z.get("slice") == "general"][:n_general]
    picked = adapt + general
    if len(picked) < (n_adapt + n_general):
        seen_ids = {t.task_id for t in picked}
        extra = [t for t in pool if t.task_id not in seen_ids][: (n_adapt + n_general - len(picked))]
        picked.extend(extra)
    picked.sort(key=lambda t: (int(t.z.get("t", 0)), str(t.task_id)))
    return picked


def run_single_slb_task(
    *,
    bundle: dict[str, Any],
    task: Any,
    extra_context: str,
    client: MeteredLLMClient,
    method_tag: str,
    max_turns: int = 8,
) -> dict[str, Any]:
    meta = bundle["meta"]
    domain = str(meta["domain"])
    env = Env(
        load_data=bundle["load_data"],
        tools=bundle["tools"],
        policy=bundle["policy_md"],
        workflow=bundle["workflow_md"],
        docs_toc=bundle["docs_toc"],
        verifier=bundle.get("verifier"),
    )
    env.reset(task)
    tool_list = _convert_slb_tools(env.tools_info, domain)
    tools_txt = "\n".join(
        f"- `{t['name']}`: {t['description']} | args={json.dumps(t['args'])}" for t in tool_list
    )
    sys_prompt = (
        f"{meta['system_prompt']}\n"
        + (f"\n{extra_context}\n" if extra_context.strip() else "")
        + "\nAVAILABLE TOOLS (call exactly ONE tool per turn as strict JSON):\n"
        + tools_txt
        + '\n\nOUTPUT FORMAT (strict JSON object ONLY): {"tool": "<tool_name>", "args": {...}, "rationale": "<brief reason>"}'
    )

    steps: list[dict[str, Any]] = []
    calls = 0
    usd = 0.0
    for turn in range(max_turns):
        hist_txt = (
            "\n".join(
                f"Turn {i + 1}: `{s['tool']}`({json.dumps(s['args'])}) -> {str(s['obs'])[:700]}"
                for i, s in enumerate(steps)
            )
            or "(No tool calls yet.)"
        )
        rem = max_turns - turn
        must_fin = f" FINAL TURN: Call `{FINISH_ACTION_NAME}` now!" if rem == 1 else ""
        usr_prompt = (
            f"Customer / Case Task:\n{task.instruction}\n\n"
            f"Tool Execution History:\n{hist_txt}\n\n"
            f"Turns remaining: {rem}.{must_fin}"
        )
        resp = client.generate_json(
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            environment=f"slb_{meta['scenario']}",
            method=method_tag,
            mock_responder=lambda _sp, _up, _t=turn: (
                {"tool": "read_docs", "args": {}, "rationale": "Read docs."}
                if _t == 0
                else {"tool": FINISH_ACTION_NAME, "args": {}, "rationale": "Finish mock."}
            ),
        )
        calls += 1
        usd += resp.usd
        parsed = resp.parsed_json or {}
        t_name = str(parsed.get("tool", "")).strip()
        t_args = parsed.get("args", {})
        if not isinstance(t_args, dict):
            t_args = {}

        if t_name == FINISH_ACTION_NAME:
            env.step(Action(name=FINISH_ACTION_NAME, kwargs=t_args))
            steps.append({"turn": turn + 1, "tool": t_name, "args": t_args, "obs": {"accepted": True}})
            break
        if t_name in env.tools_map:
            try:
                obs_str = env.step(Action(name=t_name, kwargs=t_args)).observation
            except Exception as exc:  # noqa: BLE001
                obs_str = f"tool_error: {type(exc).__name__}: {str(exc)[:200]}"
            steps.append({"turn": turn + 1, "tool": t_name, "args": t_args, "obs": obs_str})
        else:
            steps.append(
                {
                    "turn": turn + 1,
                    "tool": t_name,
                    "args": t_args,
                    "obs": f"unknown tool '{t_name}'; valid tools: {list(env.tools_map) + [FINISH_ACTION_NAME]}",
                }
            )

    rw = env.compute_reward()
    score_100 = round(100.0 * float(rw.reward), 1)
    return {
        "task_id": str(task.task_id),
        "slice": str(task.z.get("slice", "")),
        "family": str(task.z.get("family", "")),
        "reward": float(rw.reward),
        "public_score": score_100,
        "calls": calls,
        "usd": usd,
        "instruction": str(task.instruction)[:350],
        "steps": steps,
    }


def evaluate_slb_method(
    *,
    scenario: str,
    method_name: str,
    windows: list[int],
    serve_adapt: int,
    serve_general: int,
    test_adapt: int,
    test_general: int,
    client: MeteredLLMClient,
    workers: int,
    allowed_families: set[str] | None = None,
    on_task_done: Any = None,
) -> dict[str, Any]:
    bundle = get_bundle(scenario, fresh=True)
    ws_starts = bundle["rules"].WINDOW_STARTS
    all_serving = list(bundle["serving"])
    learns = method_name not in ("blind", "oracle")
    harness = make_harness(method_name) if learns else None
    if method_name == "sgeh" and harness is not None:
        # Calibrate SGEH for 0/100 discrete SLB cases: acquire across initial W0 serving stream, alarm on >=2 misses
        harness.state.probing_steps_remaining = serve_adapt + serve_general
        harness.post_alarm_probe_steps = 6
        harness.consolidate_window = 12
        harness.drift_slack = 15.0
        harness.alarm_threshold = 110.0

    per_window: list[dict[str, Any]] = []
    curve_test_scores: list[float] = []
    stitched_test_logs: list[dict[str, Any]] = []
    total_acting_calls = 0
    total_ref_calls = 0
    total_usd = 0.0
    alarms_windows: list[int] = []
    ep_global = 0

    for w in windows:
        # 1. Serving stream (strictly learner-public feedback ONLY; zero `window` or `z` leakage!)
        if learns and harness is not None:
            w_serv_all = [t for t in all_serving if t.z.get("window") == w]
            w_serv = stratified_window_tasks(w_serv_all, serve_adapt, serve_general, allowed_families)
            block_sz = 3
            for i in range(0, len(w_serv), block_sz):
                blk = w_serv[i : i + block_sz]
                dummy_spec = PublicTaskSpec(
                    env_name=f"slb_{scenario.split('_')[0]}",
                    episode_idx=ep_global,
                    timestamp=ep_global + 1,
                    domain_prompt="",
                    task_prompt=str(blk[0].instruction) if blk else "",
                    tool_schemas=[],
                )
                frozen_ctx = harness.build_context(dummy_spec)
                with ThreadPoolExecutor(max_workers=max(1, min(workers, len(blk)))) as pool:
                    blk_rows = list(
                        pool.map(
                            lambda t: run_single_slb_task(
                                bundle=bundle,
                                task=t,
                                extra_context=frozen_ctx,
                                client=client,
                                method_tag=f"slb_{method_name}_serve",
                            ),
                            blk,
                        )
                    )
                for r in blk_rows:
                    ep_global += 1
                    total_acting_calls += int(r["calls"])
                    total_usd += float(r["usd"])
                    pub = PublicEpisodeRecord(
                        env_name=f"slb_{scenario.split('_')[0]}",
                        episode_idx=ep_global,
                        timestamp=ep_global,
                        task_prompt=r["instruction"],
                        steps=r["steps"],
                        public_score=float(r["public_score"]),
                    )
                    h_meta = harness.on_episode_complete(pub, client)
                    total_ref_calls += int(h_meta.get("reflection_calls", 0))
                    total_usd += float(h_meta.get("reflection_usd", 0.0))
                    if h_meta.get("alarm_fired"):
                        alarms_windows.append(w)
                    if on_task_done:
                        on_task_done()

        # 2. Held-out test evaluation at end of window `w` (frozen memory; outcomes NEVER fed to `harness`!)
        w_test_all = list(bundle["test_set"](w))
        w_test = stratified_window_tasks(w_test_all, test_adapt, test_general, allowed_families)

        def _ctx_for_test_task(t_obj: Any) -> str:
            if method_name == "oracle":
                return str(bundle["rules"].oracle_note(ws_starts[w]))
            if harness is not None:
                d_sp = PublicTaskSpec(
                    env_name=f"slb_{scenario.split('_')[0]}",
                    episode_idx=ep_global,
                    timestamp=ep_global + 1,
                    domain_prompt="",
                    task_prompt=str(t_obj.instruction),
                    tool_schemas=[],
                )
                return harness.build_context(d_sp)
            return ""

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(w_test)))) as pool:
            test_rows = list(
                pool.map(
                    lambda t: run_single_slb_task(
                        bundle=bundle,
                        task=t,
                        extra_context=_ctx_for_test_task(t),
                        client=client,
                        method_tag=f"slb_{method_name}_test",
                    ),
                    w_test,
                )
            )
        for tr in test_rows:
            total_acting_calls += int(tr["calls"])
            total_usd += float(tr["usd"])
            curve_test_scores.append(float(tr["public_score"]))
            stitched_test_logs.append(
                {
                    "episode": len(stitched_test_logs) + 1,
                    "window": w,
                    "public_score": tr["public_score"],
                    "normalized_utility": tr["reward"],
                    "stale_policy_match": (w > 0 and tr["slice"] == "adapt" and tr["reward"] == 0.0),
                    "optimal_policy_match": tr["reward"] >= 1.0,
                    "invariant_violation": (tr["slice"] == "general" and tr["reward"] == 0.0),
                    "regime": f"W{w}_{tr['slice']}",
                    "harness_meta": {},
                    "steps": tr["steps"],
                }
            )
            if on_task_done:
                on_task_done()

        adapt_rows = [r for r in test_rows if r["slice"] == "adapt"]
        gen_rows = [r for r in test_rows if r["slice"] == "general"]
        w_all = round(sum(r["public_score"] for r in test_rows) / max(1, len(test_rows)), 2)
        w_adp = round(sum(r["public_score"] for r in adapt_rows) / max(1, len(adapt_rows)), 2)
        w_gen = round(sum(r["public_score"] for r in gen_rows) / max(1, len(gen_rows)), 2)
        per_window.append(
            {
                "window": w,
                "overall_pct": w_all,
                "adapt_pct": w_adp,
                "general_pct": w_gen,
                "n_test": len(test_rows),
            }
        )
        print(
            f"  [{scenario} | {method_name:10s} | W{w}] Test Overall={w_all:5.1f}% "
            f"(Adapt={w_adp:5.1f}%, General={w_gen:5.1f}%) | USD=${total_usd:.4f}",
            flush=True,
        )

    w0_pct = per_window[0]["overall_pct"] if per_window else 0.0
    post_wins = per_window[1:] if len(per_window) > 1 else per_window
    post_overall = round(sum(x["overall_pct"] for x in post_wins) / max(1, len(post_wins)), 2)
    post_adapt = round(sum(x["adapt_pct"] for x in post_wins) / max(1, len(post_wins)), 2)
    post_general = round(sum(x["general_pct"] for x in post_wins) / max(1, len(post_wins)), 2)

    return {
        "scenario": scenario,
        "method": method_name,
        "windows": per_window,
        "w0_acquisition_pct": w0_pct,
        "post_shift_overall_pct": post_overall,
        "post_shift_adapt_pct": post_adapt,
        "post_shift_general_pct": post_general,
        "alarms_windows": sorted(set(alarms_windows)),
        "acting_calls": total_acting_calls,
        "reflection_calls": total_ref_calls,
        "usd": round(total_usd, 6),
        "curve_test_scores": curve_test_scores,
        "logs": stitched_test_logs,
    }


def run_slb_mini_suite(
    *,
    run_id: str,
    scenario: str,
    methods: list[str],
    model: str,
    max_usd: float,
    workers: int,
    results_root: Path,
    dotenv_path: Path | None,
    serve_adapt: int = 8,
    serve_general: int = 4,
    test_adapt: int = 6,
    test_general: int = 4,
    num_windows: int = 3,
    allowed_families: set[str] | None = None,
) -> dict[str, Any]:
    ledger = CostLedger(results_root=results_root, run_id=run_id, phase="dev", max_usd=max_usd)
    total_avail_w = len(get_bundle(scenario, fresh=True)["rules"].WINDOW_STARTS)
    windows = list(range(total_avail_w if num_windows <= 0 else min(total_avail_w, num_windows)))
    est_tasks = sum(
        len(windows) * ((serve_adapt + serve_general + test_adapt + test_general) if m not in ("blind", "oracle") else (test_adapt + test_general))
        for m in methods
    )
    done_t = 0
    lock = threading.Lock()

    def _tick(lbl: str) -> None:
        nonlocal done_t
        with lock:
            done_t += 1
            ledger.update_progress(
                completed_units=min(done_t, est_tasks),
                total_units=est_tasks,
                status="running",
                current_label=lbl,
                log_every=12,
            )

    client = MeteredLLMClient(
        model=model,
        ledger=ledger,
        dotenv_path=dotenv_path,
        thinking_budget=0,
        temperature=0.0,
    )
    traces_dir = ledger.results_root / "raw" / run_id
    traces_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = ledger.run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    results_by_method: dict[str, Any] = {}
    for m_name in methods:
        print(f"\n[SLB-MINI {scenario}] Evaluating method `{m_name}` across W0 -> W1 -> W2...", flush=True)
        res = evaluate_slb_method(
            scenario=scenario,
            method_name=m_name,
            windows=windows,
            serve_adapt=serve_adapt,
            serve_general=serve_general,
            test_adapt=test_adapt,
            test_general=test_general,
            client=client,
            workers=workers,
            allowed_families=allowed_families,
            on_task_done=lambda mn=m_name: _tick(f"{scenario}:{mn}"),
        )
        results_by_method[m_name] = {k: v for k, v in res.items() if k != "logs"}

        cell_slug = f"slb_{scenario}__{m_name}__{model.replace('/', '_')}_s1"
        with open(traces_dir / f"{cell_slug}.jsonl", "w", encoding="utf-8") as fh:
            for row in res["logs"]:
                fh.write(json.dumps(row) + "\n")

        tot_c = res["acting_calls"] + res["reflection_calls"]
        cell_summary = {
            "run_id": run_id,
            "env_spec": f"slb:{scenario}",
            "harness_name": m_name,
            "model": model,
            "seed": 1,
            "episodes": len(res["logs"]),
            "mean_score": res["post_shift_overall_pct"],
            "mean_normalized_utility": res["post_shift_overall_pct"],
            "pre_shift_utility": res["w0_acquisition_pct"],
            "post_shift_utility": res["post_shift_overall_pct"],
            "post_shift_adapt_utility": res["post_shift_adapt_pct"],
            "post_shift_general_utility": res["post_shift_general_pct"],
            "stale_lock_in_rate": round(100.0 - res["post_shift_adapt_pct"], 2),
            "invariant_violation_rate": round(100.0 - res["post_shift_general_pct"], 2),
            "total_acting_calls": res["acting_calls"],
            "total_reflection_calls": res["reflection_calls"],
            "rsi_tax_call_share": round(100.0 * res["reflection_calls"] / max(1, tot_c), 2),
            "total_usd": res["usd"],
            "reflection_usd": 0.0,
        }
        atomic_write_json(
            cells_dir / f"{cell_slug}.json",
            {
                "summary": cell_summary,
                "curve_utility": res["curve_test_scores"],
                "curve_score": res["curve_test_scores"],
                "alarms": res["alarms_windows"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    rep = {
        "run_id": run_id,
        "scenario": scenario,
        "model": model,
        "total_run_usd": round(ledger.run_usd, 6),
        "max_usd_cap": max_usd,
        "observed_run_input_tokens": ledger.run_input_tokens,
        "observed_run_output_tokens": ledger.run_output_tokens,
        "observed_run_thinking_tokens": ledger.run_thinking_tokens,
        "methods": results_by_method,
    }
    atomic_write_json(ledger.run_dir / "slb_mini_report.json", rep)
    ledger.update_progress(
        completed_units=est_tasks,
        total_units=est_tasks,
        status="completed",
        current_label=f"finished SLB-Mini {scenario}",
    )
    return rep


def render_slb_markdown(rep: dict[str, Any]) -> str:
    any_m = next(iter(rep.get("methods", {}).values()), {})
    n_w = len(any_m.get("windows", [])) if isinstance(any_m, dict) else 3
    w_lbl = f"W1–W{max(1, n_w - 1)}" if n_w >= 2 else "Post-Shift"
    lines = [
        f"# External Benchmark Report (`ServeLearnBench`): `{rep['run_id']}` (`{rep['scenario']}`)",
        "",
        f"- **Model:** `{rep['model']}` | **Total Spend:** `${rep['total_run_usd']:.4f}` (Cap: `${rep['max_usd_cap']:.2f}`)",
        f"- **Observed Tokens:** `in={rep['observed_run_input_tokens']:,}`, `out={rep['observed_run_output_tokens']:,}`, `thinking={rep['observed_run_thinking_tokens']:,}`",
        "",
        f"| Method | W0 Acq (%) | Post-Shift Overall ({w_lbl} %) | Post-Shift `Adapt` (Shifted Rules %) | Post-Shift `General` (Unchanged Rules %) | Ref Calls | USD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m, d in rep["methods"].items():
        lines.append(
            f"| `{m}` | {d['w0_acquisition_pct']:.1f}% | **{d['post_shift_overall_pct']:.1f}%** | "
            f"**{d['post_shift_adapt_pct']:.1f}%** | {d['post_shift_general_pct']:.1f}% | "
            f"{d['reflection_calls']} | `${d['usd']:.4f}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ServeLearnBench-Mini external 3-window shift benchmark")
    parser.add_argument("--run-id", default="slb_mini_banking_l2", help="Run ID")
    parser.add_argument("--scenario", default="banking_l2", help="SLB scenario (e.g. banking_l2, retail_l2)")
    parser.add_argument("--methods", default="blind,skillbook,sgeh,oracle", help="Comma-separated methods")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Model ID or 'mock'")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers")
    parser.add_argument("--max-usd", type=float, default=1.95, help="Strict Tier-1 USD cap")
    parser.add_argument("--families", default="", help="Optional comma-separated rule families for focused SLB-Mini")
    parser.add_argument("--serve-adapt", type=int, default=6, help="Serving adapt tasks per window")
    parser.add_argument("--serve-general", type=int, default=3, help="Serving general tasks per window")
    parser.add_argument("--test-adapt", type=int, default=5, help="Test adapt tasks per window")
    parser.add_argument("--test-general", type=int, default=3, help="Test general tasks per window")
    parser.add_argument("--num-windows", type=int, default=3, help="Number of chronological windows (0 = all scenario windows)")
    parser.add_argument("--aggregate-runs", default="", help="Comma-separated SLB run IDs to merge into one report")
    parser.add_argument("--email-on-finish", action="store_true", help="Email milestone report")
    args = parser.parse_args()

    if args.aggregate_runs:
        rids = [x.strip() for x in args.aggregate_runs.split(",") if x.strip()]
        merged_methods: dict[str, Any] = {}
        tot_u = 0.0
        in_t = out_t = th_t = 0
        scen = args.scenario
        mdl = args.model
        for rid in rids:
            d = json.loads((ROOT / "results" / "runs" / rid / "slb_mini_report.json").read_text(encoding="utf-8"))
            merged_methods.update(d.get("methods", {}))
            tot_u += float(d.get("total_run_usd", 0.0))
            in_t += int(d.get("observed_run_input_tokens", 0))
            out_t += int(d.get("observed_run_output_tokens", 0))
            th_t += int(d.get("observed_run_thinking_tokens", 0))
            scen = str(d.get("scenario", scen))
            mdl = str(d.get("model", mdl))
        out_dir = ROOT / "results" / "runs" / args.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        rep = {
            "run_id": args.run_id,
            "scenario": scen,
            "model": mdl,
            "total_run_usd": round(tot_u, 6),
            "max_usd_cap": round(1.95 * len(rids), 2),
            "observed_run_input_tokens": in_t,
            "observed_run_output_tokens": out_t,
            "observed_run_thinking_tokens": th_t,
            "methods": merged_methods,
        }
        atomic_write_json(out_dir / "slb_mini_report.json", rep)
        md = render_slb_markdown(rep)
        (out_dir / "summary.md").write_text(md, encoding="utf-8")
        print(f"\n{md}\nSaved merged SLB report: {out_dir / 'summary.md'}", flush=True)
        return

    meths = [x.strip() for x in args.methods.split(",") if x.strip()]
    fams = {x.strip() for x in args.families.split(",") if x.strip()} or None
    try:
        rep = run_slb_mini_suite(
            run_id=args.run_id,
            scenario=args.scenario,
            methods=meths,
            model=args.model,
            max_usd=args.max_usd,
            workers=args.workers,
            results_root=ROOT / "results",
            dotenv_path=ROOT / ".env",
            serve_adapt=args.serve_adapt,
            serve_general=args.serve_general,
            test_adapt=args.test_adapt,
            test_general=args.test_general,
            num_windows=args.num_windows,
            allowed_families=fams,
        )
    except BudgetExceededError as exc:
        print(f"[BUDGET GUARDRAIL HALT] {exc}", flush=True)
        sys.exit(2)

    md = render_slb_markdown(rep)
    out_md = ROOT / "results" / "runs" / args.run_id / "summary.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\n{md}\nSaved SLB-Mini report: {out_md}", flush=True)
    if args.email_on_finish:
        send_monitor_email(
            subject=f"[RSI-Sprint] ServeLearnBench-Mini ({args.scenario}) Complete - ${rep['total_run_usd']:.4f}",
            body=md,
        )


if __name__ == "__main__":
    main()
