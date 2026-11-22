"""State-handoff localizer (task 006). Dry-run default; --live requires budget flags."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from llm_rl_engine.analysis.selective_revision_pilot import estimate_tokens, sha256_text
from llm_rl_engine.analysis.selective_revision_sequential import (
    HANDOFF_INTERFACES,
    HANDOFF_N,
    handoff_jobs,
    handoff_tapes,
    map_action_from_v1,
    reference_at,
)
from llm_rl_engine.analysis.token_cost import estimate_cost_usd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "reports" / "selective_revision_state_handoff_v1_dryrun"


def smoke_first(jobs: list[dict]) -> list[dict]:
    wanted = [("shared_s3", "R", 0), ("independent_s3", "S", 0), ("stable_s12", "C", 0)]
    front = []
    rest = []
    seen = set()
    for job in jobs:
        key = (job["tape_id"], job["interface"], job["repeat"])
        if key in wanted and key not in seen:
            front.append(job)
            seen.add(key)
        else:
            rest.append(job)
    ordered = []
    for w in wanted:
        ordered.append(next(j for j in front if (j["tape_id"], j["interface"], j["repeat"]) == w))
    return ordered + rest


def write_dry_run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = smoke_first(handoff_jobs())
    (output_dir / "prompts").mkdir(exist_ok=True)
    unique = {}
    actions = {}
    for job in jobs:
        name = f"{job['tape_id']}_{job['interface']}.txt"
        if name not in unique:
            (output_dir / "prompts" / name).write_text(job["prompt"], encoding="utf-8")
            unique[name] = job["prompt"]
        act = map_action_from_v1(job["reference"]["decision"]["optimal_action"])
        actions[act] = actions.get(act, 0) + 1
    max_est = max(estimate_tokens(p) for p in unique.values())
    n_jobs = len(jobs)
    reserved = n_jobs * 2 * estimate_cost_usd("gemini-3.5-flash", 4000, 2048 + 8192)["total_cost_usd"]
    tapes = {j["tape_id"]: j["tape"] for j in jobs}
    refs = {
        tid: {
            "n32": map_action_from_v1(reference_at(t, HANDOFF_N)["decision"]["optimal_action"]),
            "structure": t.structure,
        }
        for tid, t in tapes.items()
    }
    manifest = {
        "kind": "state_handoff_v1_dryrun",
        "n_jobs": n_jobs,
        "n_unique_prompts": len(unique),
        "interfaces": list(HANDOFF_INTERFACES),
        "checkpoint": 32,
        "previous_state": 20,
        "repeats": 2,
        "token_estimate_max": max_est,
        "illustrative_reserved_usd": reserved,
        "optimal_action_counts": actions,
        "per_tape_optimal": refs,
        "limitation": "No safe-optimal cell in this six-tape set. Development diagnostic, not a holdout.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    index = [
        {
            "tape_id": j["tape_id"],
            "interface": j["interface"],
            "repeat": j["repeat"],
            "prompt_hash": sha256_text(j["prompt"]),
        }
        for j in jobs
    ]
    (output_dir / "prompt_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    md = [
        "# State-handoff localizer dry-run",
        "",
        f"- Jobs: {n_jobs} (6 tapes × 4 interfaces × 2 repeats)",
        f"- Unique prompts: {len(unique)}",
        f"- Heuristic max tokens: {max_est}",
        f"- Illustrative reserve (96 attempts): ${reserved:.5f}",
        f"- Optimal actions over 48 jobs: {actions}",
        "",
        "Interfaces R/F/S/C. Smoke-first: shared R, independent S, stable C.",
        "",
    ]
    (output_dir / "dryrun.md").write_text("\n".join(md), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--thinking-budget", type=int, default=8192)
    parser.add_argument("--max-input-tokens", type=int, default=4000)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--max-usd", type=float, default=None)
    args = parser.parse_args(argv)
    if not args.live:
        manifest = write_dry_run(args.output_dir)
        print(f"Wrote {args.output_dir} jobs={manifest['n_jobs']}")
        return 0
    from llm_rl_engine.analysis.selective_revision_harness import (
        BoundedGeminiClient,
        IdentityError,
        LiveSettings,
        check_or_init_identity,
    )
    from llm_rl_engine.analysis.selective_revision_sequential_run import run_sequential, sequential_identity
    from llm_rl_engine.analysis.token_cost import DEFAULT_MODEL_PRICING
    from llm_rl_engine.config.env import load_dotenv

    jobs = smoke_first(handoff_jobs())
    n_jobs = 3 if args.smoke else 48
    max_calls = args.max_calls if args.max_calls is not None else 96
    max_usd = args.max_usd if args.max_usd is not None else 20.0
    rates = DEFAULT_MODEL_PRICING["gemini-3.5-flash"]
    settings = LiveSettings(
        model=args.model,
        max_calls=int(max_calls),
        max_usd=float(max_usd),
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        thinking_budget=args.thinking_budget,
        max_retries=1,
        temperature=0.0,
        pricing=dict(rates),
    )
    output_dir = args.output_dir
    if output_dir == DEFAULT_OUT:
        output_dir = REPO / "runs" / "selective_revision_state_handoff_v1"
    tapes = handoff_tapes()
    identity = sequential_identity(settings, REPO, tapes)
    try:
        check_or_init_identity(output_dir, identity, settings)
    except IdentityError as exc:
        print(f"Rejected resume before constructing a client: {exc}", file=sys.stderr)
        return 2
    load_dotenv()
    client = BoundedGeminiClient(
        settings.model,
        max_output_tokens=settings.max_output_tokens,
        thinking_budget=settings.thinking_budget,
        temperature=settings.temperature,
    )
    # run_sequential uses sequential_jobs(tapes) by default — pass custom via monkeypatch
    selected = jobs[:n_jobs] if args.smoke else jobs
    result = run_sequential(
        output_dir,
        settings,
        client,
        repo_root=REPO,
        smoke_limit=None,
        resume=True,
        progress=True,
        tapes=tapes,
        jobs=selected,
    )
    print(f"Live artifacts: {output_dir}")
    print(result.get("summary"))
    print(result.get("ledger"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
