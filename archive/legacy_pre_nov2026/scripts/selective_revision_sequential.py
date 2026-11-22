"""Sequential selective-revision dry-run. No live calls unless --live is supplied later."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from llm_rl_engine.analysis.selective_revision_pilot import estimate_tokens, sha256_text, canonical_json
from llm_rl_engine.analysis.selective_revision_sequential import (
    CONDITIONS,
    DEFAULT_H,
    PROBE_AFTER,
    development_tapes,
    initial_state_payload,
    previous_checkpoint,
    prompt_for_job,
    reference_at,
    sequential_jobs,
    serialize_state,
)
from llm_rl_engine.analysis.token_cost import estimate_cost_usd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "reports" / "selective_revision_sequential_v2_dryrun"


def tape_public(tape) -> dict:
    return {
        "tape_id": tape.tape_id,
        "seed": tape.seed,
        "structure": tape.structure,
        "hazard": tape.hazard,
        "labels": tape.labels,
        "schedule": list(tape.schedule),
        "public_observations": list(tape.public_observations),
        "probes": {
            str(n): {
                "probability_high": reference_at(tape, n)["probability_high"],
                "optimal_action": reference_at(tape, n)["decision"]["optimal_action"],
                "net_diagnostic_advantage": reference_at(tape, n)["decision"]["net_diagnostic_advantage"],
            }
            for n in PROBE_AFTER
        },
    }


def write_plot(tapes, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from llm_rl_engine.analysis.selective_revision_sequential import filter_prefix, marginals

    tape = next(t for t in tapes if t.tape_id == "shared_s0")
    xs = list(range(1, 45))
    ys = []
    for n in xs:
        ys.append(marginals(filter_prefix(tape.prefix(n), structure=tape.structure, hazard=tape.hazard))["linked"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(xs, ys, color="black")
    for p in PROBE_AFTER:
        ax.axvline(p, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("observations")
    ax.set_ylabel("P(linked high)")
    ax.set_title("shared_s0 reference P(linked high)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_dry_run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tapes = development_tapes()
    jobs = sequential_jobs(tapes)
    (output_dir / "prompts").mkdir(exist_ok=True)
    public_tapes = [tape_public(t) for t in tapes]
    (output_dir / "tapes_public.json").write_text(
        json.dumps(public_tapes, indent=2, sort_keys=True) + "\n"
    )
    private = [
        {
            "tape_id": t.tape_id,
            "private_latents": t.private_latents,
            "private_events": t.private_events,
        }
        for t in tapes
    ]
    (output_dir / "tapes_private.json").write_text(
        json.dumps(private, indent=2, sort_keys=True) + "\n"
    )
    prompt_index = []
    max_est = 0
    for job in jobs:
        prev_n = previous_checkpoint(job["n"])
        prev = None
        if job["condition"] == "self_state":
            # Live uses the condition's last parsed joint. Dry-run uses a size-matched
            # reference placeholder and labels it in the prompt hash note.
            if prev_n is None:
                prev = initial_state_payload()
            else:
                prev = serialize_state(
                    reference_at(job["tape"], prev_n)["joint"],
                    f"after_obs_{prev_n}",
                )
        prompt = prompt_for_job(job, prev)
        name = f"{job['tape_id']}_n{job['n']}_{job['condition']}.txt"
        (output_dir / "prompts" / name).write_text(prompt, encoding="utf-8")
        est = estimate_tokens(prompt)
        max_est = max(max_est, est)
        prompt_index.append(
            {
                "tape_id": job["tape_id"],
                "n": job["n"],
                "condition": job["condition"],
                "prompt_hash": sha256_text(prompt),
                "token_estimate_heuristic": est,
                "request_identity": sha256_text(
                    canonical_json(
                        {
                            "tape_id": job["tape_id"],
                            "n": job["n"],
                            "condition": job["condition"],
                            "prompt_hash": sha256_text(prompt),
                        }
                    )
                ),
            }
        )
    write_plot(tapes, output_dir / "figures" / "shared_s0_linked.png")
    n_jobs = len(jobs)
    attempts = n_jobs * 2
    # illustrative caps from the brief
    reserved = attempts * estimate_cost_usd(
        "gemini-3.5-flash", 4000, 2048 + 8192
    )["total_cost_usd"]
    actions = {}
    for tape in tapes:
        for n in PROBE_AFTER:
            act = reference_at(tape, n)["decision"]["optimal_action"]
            actions[act] = actions.get(act, 0) + 1
    manifest = {
        "schema_version": 1,
        "kind": "sequential_selective_revision_dryrun",
        "n_tapes": len(tapes),
        "n_jobs": n_jobs,
        "n_conditions": len(CONDITIONS),
        "probes": list(PROBE_AFTER),
        "hazard_nonzero": DEFAULT_H,
        "planned_attempts_with_retry": attempts,
        "token_estimate_kind": "offline_heuristic",
        "token_estimate_max": max_est,
        "illustrative_reserved_usd_4000_in_10240_out": reserved,
        "optimal_action_counts_over_tape_probes": actions,
        "limitation": (
            "Illustrative reserve uses 4000 input and 10240 provider output tokens "
            "(2048+8192 thinking). Recheck with provider counts before live. "
            "Decision probes do not change the tape."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "prompt_index.json").write_text(
        json.dumps(prompt_index, indent=2, sort_keys=True) + "\n"
    )
    samples = {
        "reconstruct_n12": (output_dir / "prompts" / f"{tapes[0].tape_id}_n12_reconstruct.txt").read_text(),
    }
    (output_dir / "prompt_samples.json").write_text(
        json.dumps({k: sha256_text(v) for k, v in samples.items()}, indent=2) + "\n"
    )
    md = [
        "# Sequential selective revision: dry-run",
        "",
        "CPU-only frozen tapes. No Gemini calls.",
        "",
        f"- Tapes: {len(tapes)} (3 seeds × shared/independent + 2 stationary controls)",
        f"- Jobs: {n_jobs} (4 checkpoints × 3 conditions)",
        f"- Retry reserve: {attempts} attempts",
        f"- Max heuristic prompt tokens: {max_est}",
        f"- Illustrative reserve at 4000 in / 10240 out: ${reserved:.4f}",
        "",
        "Optimal v1-mapped actions across 32 tape×probe cells: "
        + ", ".join(f"{k}={v}" for k, v in sorted(actions.items())),
        "",
        "Live execution is not authorized by this dry-run. Commander reviews fixtures first.",
        "",
        f"Figure: `{output_dir / 'figures' / 'shared_s0_linked.png'}`",
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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking-budget", type=int, default=8192)
    parser.add_argument("--max-input-tokens", type=int, default=4000)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--max-usd", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args(argv)
    if not args.live:
        manifest = write_dry_run(args.output_dir)
        print(f"Wrote {args.output_dir} jobs={manifest['n_jobs']}")
        return 0
    from llm_rl_engine.analysis.selective_revision_harness import (
        BoundedGeminiClient,
        IdentityError,
        LiveConfigError,
        LiveSettings,
    )
    from llm_rl_engine.analysis.selective_revision_sequential_run import run_sequential
    from llm_rl_engine.analysis.token_cost import DEFAULT_MODEL_PRICING

    n_jobs = 3 if args.smoke else 96
    max_calls = args.max_calls if args.max_calls is not None else n_jobs * (1 + args.max_retries)
    max_usd = args.max_usd if args.max_usd is not None else 20.0
    rates = DEFAULT_MODEL_PRICING.get(args.model) or DEFAULT_MODEL_PRICING["gemini-3.5-flash"]
    try:
        settings = LiveSettings(
            model=args.model,
            max_calls=int(max_calls),
            max_usd=float(max_usd),
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            thinking_budget=args.thinking_budget,
            max_retries=args.max_retries,
            temperature=args.temperature,
            pricing=dict(rates),
        )
    except (TypeError, ValueError) as exc:
        print(f"Rejected live config: {exc}", file=sys.stderr)
        return 2
    output_dir = args.output_dir
    if output_dir == DEFAULT_OUT:
        output_dir = REPO / "runs" / "selective_revision_sequential_v2"
    from llm_rl_engine.analysis.selective_revision_sequential_run import sequential_identity
    from llm_rl_engine.analysis.selective_revision_sequential import development_tapes
    from llm_rl_engine.analysis.selective_revision_harness import check_or_init_identity

    tapes = development_tapes()
    identity = sequential_identity(settings, REPO, tapes)
    try:
        check_or_init_identity(output_dir, identity, settings)
    except IdentityError as exc:
        print(f"Rejected resume before constructing a client: {exc}", file=sys.stderr)
        return 2
    from llm_rl_engine.config.env import load_dotenv

    load_dotenv()
    client = BoundedGeminiClient(
        settings.model,
        max_output_tokens=settings.max_output_tokens,
        thinking_budget=settings.thinking_budget,
        temperature=settings.temperature,
    )
    result = run_sequential(
        output_dir,
        settings,
        client,
        repo_root=REPO,
        smoke_limit=3 if args.smoke else None,
        resume=True,
        progress=True,
        tapes=tapes,
    )
    print(f"Live artifacts: {output_dir}")
    print(result.get("summary"))
    print(result.get("ledger"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
