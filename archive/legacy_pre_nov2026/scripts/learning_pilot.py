"""Learning-pilot CLI (task 009). YAML is the executable config.

Commands (CPU, this repo):
  UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --group train python scripts/learning_pilot.py render
  UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --group train python scripts/learning_pilot.py backend-check --device cpu
  UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --group train python scripts/learning_pilot.py verify-path
  UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --group train python scripts/learning_pilot.py probe --mode head --device cpu

External CUDA (isolated project, will not restore CPU torch into the default env):
  cd learning_pilot && UV_CACHE_DIR=/tmp/llm-time-uv-cache uv sync --extra cu124
  cd learning_pilot && uv run --extra cu124 python ../scripts/learning_pilot.py backend-check --device cuda
  cd learning_pilot && uv run --extra cu124 python ../scripts/learning_pilot.py probe --mode head --device cuda
  cd learning_pilot && uv run --extra cu124 python ../scripts/learning_pilot.py fit --mode head --seed 0 --device cuda
  cd learning_pilot && uv run --extra cu124 python ../scripts/learning_pilot.py fit --mode head --seed 0 --device cuda --resume
  cd learning_pilot && uv run --extra cu124 python ../scripts/learning_pilot.py eval --run-dir runs/learning_pilot/head_s0 --split structural --device cuda
  cd learning_pilot && uv run --extra cu124 python ../scripts/learning_pilot.py aggregate --runs-root runs/learning_pilot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

REPO = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO / "llm_rl_engine" / "configs" / "learning_pilot.yaml"


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cmd_render(args: argparse.Namespace) -> int:
    from llm_rl_engine.analysis.learning_pilot import V1_DATA, V2_DATA, rerender_v2

    out = Path(args.output_dir) if args.output_dir else V2_DATA
    result = rerender_v2(V1_DATA, out)
    print(json.dumps({"n": result["n"], "renderer": "v2"}, indent=2))
    return 0


def cmd_backend_check(args: argparse.Namespace) -> int:
    import torch
    from llm_rl_engine.analysis.learning_pilot_train import require_device

    device = require_device(args.device)
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    if args.device == "cuda" and device.type != "cuda":
        raise SystemExit("backend-check failed: cuda not in use")
    print(json.dumps(info, indent=2))
    return 0


def _extra_code(config_path: Path) -> tuple[str, ...]:
    try:
        rel = str(config_path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        rel = str(config_path)
    if rel.endswith("learning_pilot.yaml"):
        return ()
    return (rel,)


def cmd_fit(args: argparse.Namespace) -> int:
    from llm_rl_engine.analysis.learning_pilot import load_config
    from llm_rl_engine.analysis.learning_pilot_train import fit_run, require_device, set_cpu_threads

    cfg = load_config(args.config)
    set_cpu_threads(cfg.cpu_threads)
    device = require_device(args.device)
    out = Path(args.out_dir) if args.out_dir else REPO / "runs" / "learning_pilot" / f"{args.mode}_s{args.seed}"
    result = fit_run(
        cfg,
        mode=args.mode,
        seed=args.seed,
        data_dir=Path(cfg.data_dir) if Path(cfg.data_dir).is_absolute() else REPO / cfg.data_dir,
        out_dir=out,
        device=device,
        resume=args.resume,
        max_steps=args.max_steps,
        extra_code=_extra_code(Path(args.config)),
    )
    print(json.dumps({"step": result["state"]["step"], "updates": result["state"]["optimizer_updates"], "out": result["out_dir"]}))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    import torch
    from llm_rl_engine.analysis.learning_pilot import join_records, load_config, load_jsonl
    from llm_rl_engine.analysis.learning_pilot_train import (
        NumericMLP,
        build_reader_matched,
        eval_dev_kl,
        fit_run,
        require_device,
        save_run_checkpoint,
    )
    from llm_rl_engine.analysis.posterior_baseline_model import set_cpu_threads

    cfg = load_config(args.config)
    set_cpu_threads(cfg.cpu_threads)
    device = require_device(args.device)
    data_dir = Path(cfg.data_dir) if Path(cfg.data_dir).is_absolute() else REPO / cfg.data_dir
    train = join_records(load_jsonl(data_dir / "train_model.jsonl"), load_jsonl(data_dir / "train_audit.jsonl"))[: cfg.effective_batch]
    dev = join_records(load_jsonl(data_dir / "dev_model.jsonl"), load_jsonl(data_dir / "dev_audit.jsonl"))[: cfg.effective_batch]
    scratch = REPO / "reports" / "learning_pilot" / "scratch" / f"probe_{args.mode}"
    if scratch.exists():
        import shutil
        shutil.rmtree(scratch)
    t0 = time.perf_counter()
    result = fit_run(
        cfg,
        mode=args.mode,
        seed=0,
        data_dir=data_dir,
        out_dir=scratch,
        device=device,
        max_steps=1,
        train_rows=train,
        dev_rows=dev,
    )
    probe = {
        "mode": args.mode,
        "device": str(device),
        "torch": torch.__version__,
        "load_and_one_step_s": time.perf_counter() - t0,
        "optimizer_updates": result["state"]["optimizer_updates"],
        "rss_mb": rss_mb(),
        "note": "Disposable probe. Not a scientific run. CPU timings are not GPU-hour estimates.",
    }
    if device.type == "cuda":
        probe["peak_alloc_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(json.dumps(probe, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import torch
    from llm_rl_engine.analysis.learning_pilot import (
        evaluate_split,
        join_records,
        load_config,
        load_jsonl,
        summarize_eval,
        write_jsonl,
    )
    from llm_rl_engine.analysis.learning_pilot_train import (
        NumericMLP,
        build_reader_matched,
        load_run_checkpoint,
        predict_lex,
        require_device,
    )
    from llm_rl_engine.analysis.posterior_baseline_model import set_cpu_threads

    cfg = load_config(args.config)
    if args.split in ("same_topology", "structural"):
        if cfg.heldout_eval == "never" or not args.allow_holdout:
            raise SystemExit("refusing holdout eval (010 never / 009 needs --allow-holdout after selection)")
    set_cpu_threads(cfg.cpu_threads)
    device = require_device(args.device)
    data_dir = Path(cfg.data_dir) if Path(cfg.data_dir).is_absolute() else REPO / cfg.data_dir
    run_dir = Path(args.run_dir)
    from llm_rl_engine.analysis.learning_pilot import (
        experiment_identity,
        split_file_hashes,
        training_fingerprint,
        write_eval_artifact,
    )
    train = join_records(load_jsonl(data_dir / "train_model.jsonl"), load_jsonl(data_dir / "train_audit.jsonl"))
    dev = join_records(load_jsonl(data_dir / "dev_model.jsonl"), load_jsonl(data_dir / "dev_audit.jsonl"))
    ckpt_id = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["identity"]
    mode = ckpt_id["mode"]
    live = experiment_identity(
        cfg,
        mode=mode,
        seed=ckpt_id["seed"],
        data_dir=data_dir,
        train_rows=train,
        dev_rows=dev,
        extra_code=_extra_code(Path(args.config)),
    )
    live["device"] = str(device)
    live["torch"] = torch.__version__
    live["dependencies"] = {"torch": torch.__version__}
    if mode == "mlp":
        module = NumericMLP(cfg.mlp_hidden).to(device)
        tok = None
    else:
        cache = Path(cfg.cache_dir)
        if not cache.is_absolute():
            cache = REPO / cache
        module, tok = build_reader_matched(
            mode, cfg, cache_dir=cache, device=device, head_seed=ckpt_id["seed"],
            adapter_seed=ckpt_id.get("adapter_seed"),
        )
    load_run_checkpoint(run_dir / "best.pt", module, live, strict_identity=True)
    if args.split == "train":
        records = train
    elif args.split == "dev":
        records = dev
    else:
        records = join_records(load_jsonl(data_dir / f"{args.split}_model.jsonl"), load_jsonl(data_dir / f"{args.split}_audit.jsonl"))
    preds = {}
    for rec in records:
        preds[rec["audit"]["example_id"]] = predict_lex(module, rec, tok, device, cfg, mode)
    scored = evaluate_split(records, preds)
    summary = summarize_eval(scored["rows"], analysis_seed=cfg.analysis_seed)
    man = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    write_eval_artifact(
        run_dir,
        args.split,
        preds=preds,
        summary=summary,
        n=scored["n"],
        checkpoint_identity=ckpt_id,
        data_hashes=split_file_hashes(data_dir, args.split) if (data_dir / f"{args.split}_model.jsonl").exists() else live["data_hashes"],
        live_fingerprint=training_fingerprint(live),
        selected=(man.get("train_state") or {}).get("best"),
    )
    write_jsonl(run_dir / f"eval_{args.split}_rows.jsonl", scored["rows"])
    print(json.dumps({"split": args.split, "n": scored["n"], "joint_kl": summary["overall"]["joint_kl"]}))
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    from llm_rl_engine.analysis.learning_pilot import load_config

    cfg = load_config(args.config)
    root = Path(args.runs_root)
    expected = [f"head_s{s}" for s in cfg.seeds] + [f"lora_s{s}" for s in cfg.seeds] + [f"mlp_s{s}" for s in cfg.seeds]
    missing = [name for name in expected if not (root / name / "eval_structural.json").exists()]
    if missing:
        raise SystemExit(f"incomplete aggregation, missing {missing}")
    blob = {}
    for name in expected:
        blob[name] = json.loads((root / name / "eval_structural.json").read_text())
    out = root / "aggregate.json"
    out.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_runs": len(expected), "out": str(out)}))
    return 0


def cmd_verify_path(args: argparse.Namespace) -> int:
    """Bounded actual-path check: 3 optimizer updates per control (2 unique + replay)."""
    import torch
    from llm_rl_engine.analysis.learning_pilot import join_records, load_config, load_jsonl
    from llm_rl_engine.analysis.learning_pilot_train import (
        NumericMLP,
        build_reader_matched,
        fit_run,
        require_device,
        snapshot_params,
    )
    from llm_rl_engine.analysis.posterior_baseline_model import max_abs_diff, set_cpu_threads

    cfg = load_config(
        args.config,
        overrides={"effective_batch": 2, "microbatch": 1, "optimizer_steps_per_run": 2, "epochs": 1},
    )
    set_cpu_threads(cfg.cpu_threads)
    device = require_device(args.device)
    data_dir = Path(cfg.data_dir) if Path(cfg.data_dir).is_absolute() else REPO / cfg.data_dir
    train = join_records(load_jsonl(data_dir / "train_model.jsonl"), load_jsonl(data_dir / "train_audit.jsonl"))[:8]
    dev = join_records(load_jsonl(data_dir / "dev_model.jsonl"), load_jsonl(data_dir / "dev_audit.jsonl"))[:4]
    counts = {}
    scratch = REPO / "reports" / "learning_pilot" / "scratch" / "verify_path"
    scratch.mkdir(parents=True, exist_ok=True)

    # matched heads
    cache = Path(cfg.cache_dir)
    if not cache.is_absolute():
        cache = REPO / cache
    head_a, _ = build_reader_matched("head", cfg, cache_dir=cache, device=device, head_seed=0)
    head_b, _ = build_reader_matched("lora", cfg, cache_dir=cache, device=device, head_seed=0, adapter_seed=10_007)
    wdiff = float((head_a.head.proj.weight - head_b.head.proj.weight).abs().max())
    bdiff = float((head_a.head.proj.bias - head_b.head.proj.bias).abs().max())
    head_a.eval()
    head_b.eval()
    with torch.no_grad():
        from llm_rl_engine.analysis.posterior_baseline_model import encode_batch, load_tokenizer
        tok = load_tokenizer(cache)
        enc = encode_batch(tok, [train[0]["model_input"]["text"], train[1]["model_input"]["text"]], device=device)
        pa = head_a(**enc)["log_probs"]
        pb = head_b(**enc)["log_probs"]
        pdiff = float((pa - pb).abs().max())
    matched = {"weight_max_abs": wdiff, "bias_max_abs": bdiff, "logprob_max_abs": pdiff, "tolerance": 1e-6}

    for mode in ("mlp", "head", "lora"):
        cont = scratch / f"{mode}_cont"
        step1 = scratch / f"{mode}_step1"
        resume = scratch / f"{mode}_resume"
        for p in (cont, step1, resume):
            if p.exists():
                import shutil
                shutil.rmtree(p)
        extra = {}
        if mode == "mlp":
            torch.manual_seed(0)
            extra["module"] = NumericMLP(cfg.mlp_hidden).to(device)
        fit_run(cfg, mode=mode, seed=0, data_dir=data_dir, out_dir=cont, device=device, max_steps=2, train_rows=train, dev_rows=dev, **extra)
        extra = {}
        if mode == "mlp":
            torch.manual_seed(0)
            extra["module"] = NumericMLP(cfg.mlp_hidden).to(device)
        fit_run(cfg, mode=mode, seed=0, data_dir=data_dir, out_dir=step1, device=device, max_steps=1, train_rows=train, dev_rows=dev, **extra)
        import shutil
        shutil.copytree(step1, resume)
        extra = {}
        if mode == "mlp":
            extra["module"] = NumericMLP(cfg.mlp_hidden).to(device)
        out = fit_run(cfg, mode=mode, seed=0, data_dir=data_dir, out_dir=resume, device=device, resume=True, max_steps=2, train_rows=train, dev_rows=dev, **extra)
        counts[mode] = {
            "unique_steps": 2,
            "executed_updates_this_command": 3,
            "final_step": out["state"]["step"],
        }
        # LoRA vs head gradient separation: after one lora step, frozen backbone delta 0
    report = {
        "matched_heads": matched,
        "optimizer_counts": counts,
        "device": str(device),
        "rss_mb": rss_mb(),
        "note": "Actual fit/resume path. Not a learning result. Holdouts not scored.",
    }
    outp = REPO / "reports" / "learning_pilot" / "verify_path.json"
    outp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if matched["weight_max_abs"] > 1e-6 or matched["logprob_max_abs"] > 1e-5:
        raise SystemExit("matched-head check failed")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_YAML)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--output-dir", type=Path, default=None)
    b = sub.add_parser("backend-check")
    b.add_argument("--device", default="cpu")
    f = sub.add_parser("fit")
    f.add_argument("--mode", choices=("head", "lora", "mlp"), required=True)
    f.add_argument("--seed", type=int, required=True)
    f.add_argument("--device", default="cpu")
    f.add_argument("--out-dir", type=Path, default=None)
    f.add_argument("--resume", action="store_true")
    f.add_argument("--max-steps", type=int, default=None)
    p = sub.add_parser("probe")
    p.add_argument("--mode", choices=("head", "lora", "mlp"), required=True)
    p.add_argument("--device", default="cpu")
    e = sub.add_parser("eval")
    e.add_argument("--run-dir", type=Path, required=True)
    e.add_argument("--split", required=True)
    e.add_argument("--device", default="cpu")
    e.add_argument("--allow-holdout", action="store_true")
    a = sub.add_parser("aggregate")
    a.add_argument("--runs-root", type=Path, required=True)
    vp = sub.add_parser("verify-path")
    vp.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.cmd == "render":
        return cmd_render(args)
    if args.cmd == "backend-check":
        return cmd_backend_check(args)
    if args.cmd == "fit":
        return cmd_fit(args)
    if args.cmd == "probe":
        return cmd_probe(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "aggregate":
        return cmd_aggregate(args)
    if args.cmd == "verify-path":
        return cmd_verify_path(args)
    raise SystemExit(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
