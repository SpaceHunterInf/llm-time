"""Task 006 live driver: offline references, two-call smoke, eight lifetimes.

Checkpoint/resume is disabled. A nonempty output directory is refused.
Run from single_life/:

  UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python -m single_life.procedure_pilot
"""

from __future__ import annotations

import argparse
import json
import traceback
from decimal import Decimal
from pathlib import Path
from typing import Any

from single_life.contracts import sha256_bytes, sha256_text
from single_life.gemini_adapter import TransportError, load_api_key, redact
from single_life.identity import dependency_identity, source_identity
from single_life.procedure_client import (
    INPUT_TOKEN_CAP,
    MAX_GENERATION_ATTEMPTS,
    OUTPUT_TOKEN_CAP,
    ProcedureGeminiClient,
    attempt_cap_usd,
    completion_cap_usd,
    confirm_model,
    pricing_record,
)
from single_life.procedure_dsl import (
    ACTION_ORDER,
    FROZEN_CALLS,
    HORIZON,
    INSTRUCTION,
    LOCAL_ALGORITHM,
    PARSER_POLICY,
    RANDOM_ALGORITHM,
    RANDOM_SEED,
    REVISION_CALLS,
    SMOKE_CALLS,
    SMOKE_HORIZON,
    WHOLE_ALGORITHM,
)
from single_life.procedure_progress import ProcedureLedger
from single_life.procedure_runner import atomic_write_text, run_lifetime, run_reference
from single_life.procedure_world import WORLD_IDS, config_document, evaluator_cycle_means, load_config

EXPERIMENT = "006-procedure-acquisition"
WORLDS = WORLD_IDS
CONDITIONS = ("revision", "frozen")
REFERENCES = (RANDOM_ALGORITHM, LOCAL_ALGORITHM, WHOLE_ALGORITHM)
CHECKPOINTS = (25, 37, 49, 61)
WINDOWS = (("actions_1_96", 1, 96), ("actions_1_24", 1, 24), ("actions_25_42", 25, 42), ("actions_43_96", 43, 96))
BANNED_PROMPT_MARKERS = ("SENTINEL_PRIVATE", "oracle_best", "change_at", "table_phase", "PROGRESS", "cap_usd")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def experiment_dir(root: Path) -> Path:
    return root / "experiments" / EXPERIMENT


def ensure_fresh(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing nonempty directory {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def window_sums(events: list[dict[str, Any]]) -> dict[str, int]:
    sums = {}
    for name, start, end in WINDOWS:
        sums[name] = sum(int(event["utility_tenths"]) for event in events if start <= int(event["index"]) <= end)
    return sums


def active_artifact(artifacts: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    chosen = [item for item in artifacts if int(item["deployment_index"]) <= index]
    return chosen[-1] if chosen else None


def describe_saved_run(run_dir: Path) -> dict[str, Any]:
    events = read_jsonl(run_dir / "public_log.jsonl")
    artifacts = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    versions = {int(item["version"]): item for item in artifacts}
    probes = 0
    for event in events:
        version = event.get("procedure_version")
        if version is None:
            continue
        artifact = versions[int(version)]
        if len(artifact["procedure"][event["type"]]) > 1:
            probes += 1
    checkpoints = {}
    for index in CHECKPOINTS:
        artifact = active_artifact(artifacts, index)
        checkpoints[str(index)] = None if artifact is None else {
            "content_hash": artifact["content_hash"],
            "deployment_index": artifact["deployment_index"],
            "procedure": artifact["procedure"],
            "semantic_change": artifact["semantic_change"],
        }
    type_rules = {}
    previous = None
    for index in CHECKPOINTS:
        current = checkpoints[str(index)]
        if current is None or previous is None:
            previous = current or previous
            continue
        stayed = [
            typ
            for typ in ("T1", "T2", "T3")
            if current["procedure"][typ] == previous["procedure"][typ]
        ]
        type_rules[f"{previous['deployment_index']}_to_{current['deployment_index']}"] = stayed
        previous = current
    acquired = active_artifact(artifacts, 25)
    cycle = None if acquired is None else evaluator_cycle_means(acquired["procedure"])
    return {
        "actions": [event["action"] for event in events],
        "checkpoints": checkpoints,
        "committed_index": manifest["committed_index"],
        "cycle_means_at_25_evaluator_only": cycle,
        "probes_from_lists_longer_than_one": probes,
        "semantic_deployments": [
            item["deployment_index"] for item in artifacts if item["semantic_change"]
        ],
        "status": manifest["status"],
        "type_rules_unchanged_between_checkpoints": type_rules,
        "usage_estimate_usd": manifest["usage_estimate_usd"],
        "windows_utility_tenths": window_sums(events),
    }


def preflight_text() -> str:
    per_attempt = attempt_cap_usd()
    cap = completion_cap_usd()
    return "\n".join(
        [
            "Task 006 preflight. Usage estimate from published rates, not a provider invoice.",
            "Model gemini-3.5-flash, temperature 0, thinkingLevel low, maxOutputTokens 2048 including thinking.",
            f"Input cap {INPUT_TOKEN_CAP} enforced by countTokens before generateContent.",
            f"Scheduled calls {SMOKE_CALLS.__len__()} smoke + 32 revision + 12 frozen = 46.",
            f"Maximum generation attempts {MAX_GENERATION_ATTEMPTS} = 46 calls times 2.",
            f"Per-attempt cap ${format(per_attempt, 'f')} = {INPUT_TOKEN_CAP} * 1.50/1e6 + {OUTPUT_TOKEN_CAP} * 9/1e6.",
            f"Completion cap ${format(cap, 'f')}.",
            "Token-count requests are separate and are not added to this generation cap.",
            "Unknown generation usage keeps the per-attempt reserve.",
            "Settings stay fixed after the smoke. There is no second prompt and no retune.",
            "",
        ]
    )


def registration_document(root: Path, model_check: dict[str, Any]) -> dict[str, Any]:
    config_path = root / "configs" / "006_worlds.json"
    return {
        "action_order": list(ACTION_ORDER),
        "call_schedule": {
            "frozen": list(FROZEN_CALLS),
            "revision": list(REVISION_CALLS),
            "smoke": list(SMOKE_CALLS),
        },
        "config_sha256": sha256_bytes(config_path.read_bytes()),
        "dependency_identity": dependency_identity(root),
        "failure_policy": PARSER_POLICY,
        "frozen_after_action": 25,
        "horizon": HORIZON,
        "instruction": INSTRUCTION,
        "instruction_sha256": sha256_text(INSTRUCTION),
        "interpreter": PARSER_POLICY["interpreter"],
        "model_check": model_check,
        "no_boundary_call": "The schedule has no generation at action 43.",
        "pricing": pricing_record(),
        "random_seed": RANDOM_SEED,
        "reference_algorithms": {
            LOCAL_ALGORITHM: "Per type, sample ACTION_ORDER once, then exploit the highest cached utility, ties by ACTION_ORDER. A mismatch clears that type's other cells and keeps the new cell.",
            RANDOM_ALGORITHM: f"uniform over ACTION_ORDER with random.Random({RANDOM_SEED}), a fresh generator in each world",
            WHOLE_ALGORITHM: "Same as local relearning, except a mismatch clears every type and keeps only the triggering cell.",
        },
        "same_settings_every_call": True,
        "smoke_horizon": SMOKE_HORIZON,
        "smoke_transfers_to_pilot": False,
        "source": source_identity(root),
        "task": "006",
        "utility_unit": "tenths of a utility point",
        "worlds": config_document(),
    }


def scan_prompts(root: Path) -> dict[str, Any]:
    hits: list[str] = []
    files = list((root / "runs").glob("006-procedure-*/**/responses/*.json"))
    files.extend((root / "runs" / "006-procedure-smoke" / "responses").glob("*.json"))
    seen = set()
    for path in files:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        prompt = data.get("prompt")
        if not isinstance(prompt, str):
            continue
        for marker in BANNED_PROMPT_MARKERS:
            if marker in prompt:
                hits.append(f"{path.name}:{marker}")
    return {"files_with_prompt": len(seen), "banned_hits": hits}


def write_svg(path: Path, runs: dict[str, dict[str, dict[str, Any]]]) -> None:
    colors = {
        "revision": "#1f4e79",
        "frozen": "#c45911",
        LOCAL_ALGORITHM: "#548235",
        WHOLE_ALGORITHM: "#833c0c",
        RANDOM_ALGORITHM: "#7f7f7f",
    }
    width = 980
    panel_h = 150
    height = 70 + panel_h * len(WORLDS)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="16" y="24" font-family="sans-serif" font-size="14">Task 006 cumulative utility in tenths. Evaluator boundary is researcher-only.</text>',
    ]
    legend_x = 16
    for name, color in colors.items():
        parts.append(f'<line x1="{legend_x}" y1="40" x2="{legend_x + 24}" y2="40" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{legend_x + 28}" y="44" font-family="sans-serif" font-size="11">{name}</text>')
        legend_x += 180
    for row, world in enumerate(WORLDS):
        top = 60 + row * panel_h
        series = runs[world]
        prepared: dict[str, list[int]] = {}
        low = 0
        high = 1
        for name, item in series.items():
            total = 0
            cumulative = []
            for utility in item["utilities"]:
                total += utility
                cumulative.append(total)
            if cumulative:
                low = min(low, min(cumulative))
                high = max(high, max(cumulative))
            prepared[name] = cumulative
        span = max(high - low, 1)
        parts.append(f'<text x="16" y="{top + 16}" font-family="sans-serif" font-size="13">{world}</text>')
        origin_y = top + panel_h - 24
        plot_h = panel_h - 48
        plot_w = 900

        def x_of(index: int) -> float:
            return 50 + (index - 1) * (plot_w / 95)

        def y_of(value: int) -> float:
            return origin_y - ((value - low) / span) * plot_h

        boundary_x = x_of(43)
        parts.append(
            f'<line x1="{boundary_x:.1f}" y1="{top + 20}" x2="{boundary_x:.1f}" y2="{origin_y}" stroke="#999999" stroke-dasharray="3 3"/>'
        )
        parts.append(
            f'<text x="{boundary_x + 4:.1f}" y="{top + 32}" font-family="sans-serif" font-size="10" fill="#666666">evaluator boundary</text>'
        )
        for name, cumulative in prepared.items():
            if not cumulative:
                continue
            points = [f"{x_of(index):.1f},{y_of(value):.1f}" for index, value in enumerate(cumulative, start=1)]
            parts.append(
                f'<polyline fill="none" stroke="{colors[name]}" stroke-width="1.4" points="{" ".join(points)}"/>'
            )
            for deployment in series[name].get("deployments", []):
                if 1 <= deployment <= len(cumulative):
                    parts.append(
                        f'<circle cx="{x_of(deployment):.1f}" cy="{y_of(cumulative[deployment - 1]):.1f}" r="2.5" fill="{colors[name]}"/>'
                    )
    parts.append("</svg>")
    atomic_write_text(path, "\n".join(parts) + "\n")


def _utilities(run_dir: Path) -> list[int]:
    return [int(event["utility_tenths"]) for event in read_jsonl(run_dir / "public_log.jsonl")]


def _deployments(run_dir: Path) -> list[int]:
    events = read_jsonl(run_dir / "public_log.jsonl")
    return [int(event["index"]) for event in events if event.get("semantic_deployment")]


def write_summary(root: Path, *, status: str, blocker: str | None, ledger: ProcedureLedger) -> None:
    experiment = experiment_dir(root)
    lifetimes: dict[str, dict[str, Any]] = {}
    plot_runs: dict[str, dict[str, dict[str, Any]]] = {}
    for world in WORLDS:
        lifetimes[world] = {}
        plot_runs[world] = {}
        for condition in CONDITIONS:
            run_dir = root / "runs" / "006-procedure-pilot" / world / condition
            if (run_dir / "manifest.json").is_file():
                lifetimes[world][condition] = describe_saved_run(run_dir)
                plot_runs[world][condition] = {"utilities": _utilities(run_dir), "deployments": _deployments(run_dir)}
        for algorithm in REFERENCES:
            run_dir = root / "runs" / "006-procedure-references" / world / algorithm
            if (run_dir / "manifest.json").is_file():
                lifetimes[world][algorithm] = describe_saved_run(run_dir)
                plot_runs[world][algorithm] = {"utilities": _utilities(run_dir), "deployments": []}
    if any(plot_runs.values()):
        write_svg(experiment / "timeline.svg", plot_runs)
    snapshot = ledger.snapshot()
    payload = {
        "blocker": blocker,
        "cap_usd": format(completion_cap_usd(), "f"),
        "lifetimes": lifetimes,
        "progress": snapshot,
        "prompt_scan": scan_prompts(root),
        "status": status,
        "task": "006",
        "unit": "utility_tenths; 8 means 0.8 utility points",
        "interpretation_limit": (
            "One lifetime per cell. Acquisition and revision are descriptive. "
            "Local relearning is a competent alternative on this deterministic toy. "
            "Not RSI, not a population effect, and not adoption of a paper benchmark."
        ),
    }
    atomic_write_text(experiment / "summary.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Task 006 summary", "", f"Status: {status}", ""]
    if blocker:
        lines.append(f"Blocker: {blocker}")
        lines.append("")
    lines.append("| World | Run | 1-96 | 1-24 | 25-42 | 43-96 | USD |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for world, runs in lifetimes.items():
        for name, item in runs.items():
            windows = item["windows_utility_tenths"]
            lines.append(
                f"| {world} | {name} | {windows['actions_1_96']} | {windows['actions_1_24']} | "
                f"{windows['actions_25_42']} | {windows['actions_43_96']} | {item['usage_estimate_usd']} |"
            )
    lines.append("")
    lines.append("Checkpoint procedures and evaluator-only cycle means are in summary.json.")
    atomic_write_text(experiment / "summary.md", "\n".join(lines) + "\n")


def execute(root: Path) -> int:
    config_path = root / "configs" / "006_worlds.json"
    try:
        load_config(config_path)
    except (OSError, ValueError) as exc:
        print(f"BLOCKED {exc}", flush=True)
        return 2
    key = load_api_key()
    if not key:
        print("BLOCKED GEMINI_API_KEY is not set", flush=True)
        return 2
    try:
        model_check = confirm_model(key)
    except TransportError as exc:
        print(f"BLOCKED model check failed: {exc}", flush=True)
        return 2
    experiment = experiment_dir(root)
    targets = [
        experiment,
        root / "runs" / "006-procedure-smoke",
        root / "runs" / "006-procedure-pilot",
        root / "runs" / "006-procedure-references",
    ]
    try:
        for path in targets:
            ensure_fresh(path)
    except FileExistsError as exc:
        print(f"BLOCKED {exc}", flush=True)
        return 2
    experiment.mkdir(parents=True, exist_ok=True)
    record = registration_document(root, model_check)
    atomic_write_text(experiment / "registration.json", json.dumps(record, indent=2, sort_keys=True) + "\n")
    atomic_write_text(experiment / "model_check.json", json.dumps(model_check, indent=2, sort_keys=True) + "\n")
    for world in WORLDS:
        for algorithm in REFERENCES:
            run_reference(
                world=world,
                algorithm=algorithm,
                run_dir=root / "runs" / "006-procedure-references" / world / algorithm,
            )
    preflight = preflight_text()
    print(preflight, flush=True)
    atomic_write_text(experiment / "preflight.txt", preflight)
    ledger = ProcedureLedger(
        experiment / "progress.json",
        planned_smoke_calls=len(SMOKE_CALLS),
        planned_revision_calls=len(WORLDS) * len(REVISION_CALLS),
        planned_frozen_calls=len(WORLDS) * len(FROZEN_CALLS),
        planned_smoke_ticks=SMOKE_HORIZON,
        planned_revision_ticks=len(WORLDS) * HORIZON,
        planned_frozen_ticks=len(WORLDS) * HORIZON,
        planned_references=len(WORLDS) * len(REFERENCES),
        attempt_cap=attempt_cap_usd(),
        completion_cap=completion_cap_usd(),
    )
    ledger.mark_references(len(WORLDS) * len(REFERENCES))
    ledger.start_heartbeat()
    client = ProcedureGeminiClient(key)
    status = "complete"
    blocker = None
    try:
        ledger.set_phase("smoke")
        smoke = run_lifetime(
            world="stable",
            condition="smoke",
            run_dir=root / "runs" / "006-procedure-smoke",
            client=client,
            ledger=ledger,
            horizon=SMOKE_HORIZON,
            lifetime_id="life-006-smoke",
        )
        if smoke.status != "completed" or smoke.committed_index != SMOKE_HORIZON:
            status = "blocked"
            blocker = f"smoke {smoke.status}: {smoke.reason}"
        else:
            problem = _smoke_problem(root / "runs" / "006-procedure-smoke")
            if problem:
                status = "blocked"
                blocker = problem
        if status == "complete":
            frozen = {
                "instruction_sha256": sha256_text(INSTRUCTION),
                "max_output_tokens": OUTPUT_TOKEN_CAP,
                "model": "gemini-3.5-flash",
                "not_retuned_after_smoke": True,
                "temperature": 0,
                "thinking_level": "low",
            }
            atomic_write_text(experiment / "frozen.json", json.dumps(frozen, indent=2, sort_keys=True) + "\n")
            ledger.set_phase("pilot")
            for world in WORLDS:
                for condition in CONDITIONS:
                    result = run_lifetime(
                        world=world,
                        condition=condition,
                        run_dir=root / "runs" / "006-procedure-pilot" / world / condition,
                        client=client,
                        ledger=ledger,
                        lifetime_id=f"life-006-{world}-{condition}",
                    )
                    if result.status != "completed":
                        status = "incomplete"
                        blocker = f"{world}/{condition} {result.status}: {result.reason}"
                        if result.stop_task:
                            break
                if status != "complete":
                    break
        ledger.finish("complete" if status == "complete" else status)
        write_summary(root, status=status, blocker=blocker, ledger=ledger)
        print(f"PILOT {status}", flush=True)
        return 0 if status == "complete" else 2
    except Exception as exc:
        message = redact(f"{type(exc).__name__}: {exc}", key)
        ledger.finish("blocked")
        write_summary(root, status="blocked", blocker=message, ledger=ledger)
        traceback.print_exc()
        print(f"BLOCKED {message}", flush=True)
        return 1
    finally:
        ledger.stop_heartbeat()


def _smoke_problem(run_dir: Path) -> str | None:
    paths = sorted((run_dir / "responses").glob("c*-a*.json"))
    if len(paths) < SMOKE_HORIZON:
        return "smoke wrote fewer generation responses than calls"
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("parser_status") != "ok":
            return f"{path.name} parser_status={data.get('parser_status')}"
        if data.get("finish_reason") != "STOP":
            return f"{path.name} finish_reason={data.get('finish_reason')}"
        counted = data.get("counted_input_tokens")
        if type(counted) is not int or counted > INPUT_TOKEN_CAP:
            return f"{path.name} counted_input_tokens={counted}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    args = parser.parse_args(argv)
    return execute(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
