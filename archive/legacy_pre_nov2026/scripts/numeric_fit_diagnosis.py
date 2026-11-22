"""Task 010: bounded CPU numeric fitting diagnosis. No Qwen, no GPU, no holdouts."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO / "llm_rl_engine" / "configs" / "numeric_fit_diagnosis.yaml"
MLP009 = REPO / "runs" / "learning_pilot"
OUT_ROOT = REPO / "runs" / "numeric_fit_diagnosis"
REPORT_DIR = REPO / "reports" / "numeric_fit_diagnosis"


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _data_dir(cfg) -> Path:
    p = Path(cfg.data_dir)
    return p if p.is_absolute() else REPO / p


def cmd_fit(args: argparse.Namespace) -> int:
    from llm_rl_engine.analysis.learning_pilot import load_config
    from llm_rl_engine.analysis.learning_pilot_train import fit_run, require_device, set_cpu_threads

    cfg = load_config(args.config)
    if args.device != "cpu":
        raise SystemExit("010 is CPU only")
    set_cpu_threads(cfg.cpu_threads)
    device = require_device("cpu")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    wall = float(cfg.wall_time_s or 600)
    deadline = time.monotonic() + wall
    t0 = time.monotonic()
    extra = ("llm_rl_engine/configs/numeric_fit_diagnosis.yaml",)
    results = []
    for seed in cfg.seeds:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            results.append({"seed": seed, "stop_reason": "wall_time", "skipped": True})
            break
        out = out_root / f"mlp_s{seed}"
        resume = (out / "last.pt").exists()
        result = fit_run(
            cfg,
            mode="mlp",
            seed=int(seed),
            data_dir=_data_dir(cfg),
            out_dir=out,
            device=device,
            resume=resume,
            extra_code=extra,
            deadline_monotonic=deadline,
        )
        results.append(
            {
                "seed": seed,
                "step": result["state"]["step"],
                "updates": result["state"]["optimizer_updates"],
                "stop_reason": result.get("stop_reason"),
                "incomplete": bool(result["state"].get("incomplete")),
                "best": result["state"]["best"],
                "out": result["out_dir"],
            }
        )
        if result.get("stop_reason") == "wall_time":
            break
    blob = {
        "wall_s": time.monotonic() - t0,
        "wall_budget_s": wall,
        "rss_mb": rss_mb(),
        "device": "cpu",
        "threads": cfg.cpu_threads,
        "runs": results,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "fit_status.json").write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    print(json.dumps(blob, indent=2))
    return 0


def _eval_split(cfg, run_dir: Path, split: str, device, extra) -> dict:
    import torch
    from llm_rl_engine.analysis.learning_pilot import (
        evaluate_split,
        experiment_identity,
        join_records,
        load_jsonl,
        split_file_hashes,
        summarize_eval,
        training_fingerprint,
        write_eval_artifact,
        write_jsonl,
    )
    from llm_rl_engine.analysis.learning_pilot_train import NumericMLP, load_run_checkpoint, predict_lex

    if split not in ("train", "dev"):
        raise SystemExit(f"010 eval is train/dev only, got {split}")
    data_dir = _data_dir(cfg)
    train = join_records(load_jsonl(data_dir / "train_model.jsonl"), load_jsonl(data_dir / "train_audit.jsonl"))
    dev = join_records(load_jsonl(data_dir / "dev_model.jsonl"), load_jsonl(data_dir / "dev_audit.jsonl"))
    records = train if split == "train" else dev
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    live = experiment_identity(
        cfg,
        mode="mlp",
        seed=ckpt["identity"]["seed"],
        data_dir=data_dir,
        train_rows=train,
        dev_rows=dev,
        extra_code=extra,
    )
    live["device"] = str(device)
    live["torch"] = torch.__version__
    live["dependencies"] = {"torch": torch.__version__}
    module = NumericMLP(cfg.mlp_hidden).to(device)
    load_run_checkpoint(run_dir / "best.pt", module, live, strict_identity=True)
    preds = {rec["audit"]["example_id"]: predict_lex(module, rec, None, device, cfg, "mlp") for rec in records}
    scored = evaluate_split(records, preds)
    summary = summarize_eval(scored["rows"], analysis_seed=cfg.analysis_seed)
    man = json.loads((run_dir / "manifest.json").read_text())
    write_eval_artifact(
        run_dir,
        split,
        preds=preds,
        summary=summary,
        n=scored["n"],
        checkpoint_identity=ckpt["identity"],
        data_hashes=split_file_hashes(data_dir, split),
        live_fingerprint=training_fingerprint(live),
        selected=(man.get("train_state") or {}).get("best"),
    )
    write_jsonl(run_dir / f"eval_{split}_rows.jsonl", scored["rows"])
    recomputed = evaluate_split(records, preds)
    re_sum = summarize_eval(recomputed["rows"], analysis_seed=cfg.analysis_seed)
    if abs(re_sum["overall"]["joint_kl"] - summary["overall"]["joint_kl"]) > 1e-12:
        raise RuntimeError("eval recompute mismatch")
    return {"split": split, "n": scored["n"], "joint_kl": summary["overall"]["joint_kl"]}


def cmd_eval(args: argparse.Namespace) -> int:
    import torch
    from llm_rl_engine.analysis.learning_pilot import load_config
    from llm_rl_engine.analysis.posterior_baseline_model import set_cpu_threads

    cfg = load_config(args.config)
    try:
        set_cpu_threads(cfg.cpu_threads)
    except RuntimeError:
        pass
    device = torch.device("cpu")
    extra = ("llm_rl_engine/configs/numeric_fit_diagnosis.yaml",)
    out = {}
    for seed in cfg.seeds:
        run_dir = Path(args.out_root) / f"mlp_s{seed}"
        if not (run_dir / "best.pt").exists():
            out[f"mlp_s{seed}"] = {"error": "no best.pt"}
            continue
        out[f"mlp_s{seed}"] = {
            "train": _eval_split(cfg, run_dir, "train", device, extra),
            "dev": _eval_split(cfg, run_dir, "dev", device, extra),
        }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "eval_status.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2))
    return 0


def _score_named(records, pred_fn):
    from llm_rl_engine.analysis.learning_pilot import evaluate_split, summarize_eval

    preds = {rec["audit"]["example_id"]: pred_fn(rec) for rec in records}
    scored = evaluate_split(records, preds)
    return scored, preds


def cmd_analyze(args: argparse.Namespace) -> int:
    import torch
    from llm_rl_engine.analysis.learning_pilot import (
        evaluate_split,
        join_records,
        load_config,
        load_jsonl,
        load_pred_lex,
        reference_pred_lex,
        summarize_eval,
    )
    from llm_rl_engine.analysis.learning_pilot_train import NumericMLP, load_run_checkpoint, predict_lex
    from llm_rl_engine.analysis.posterior_baseline_model import set_cpu_threads

    cfg = load_config(args.config)
    try:
        set_cpu_threads(cfg.cpu_threads)
    except RuntimeError:
        pass
    device = torch.device("cpu")
    data_dir = _data_dir(cfg)
    splits = {
        "train": join_records(load_jsonl(data_dir / "train_model.jsonl"), load_jsonl(data_dir / "train_audit.jsonl")),
        "dev": join_records(load_jsonl(data_dir / "dev_model.jsonl"), load_jsonl(data_dir / "dev_audit.jsonl")),
    }
    ref_names = ("exact", "stale", "transition_only", "independent_sameinfo")
    refs = {}
    for split, recs in splits.items():
        refs[split] = {}
        for name in ref_names:
            scored, _ = _score_named(recs, lambda rec, n=name: reference_pred_lex(rec, n))
            refs[split][name] = summarize_eval(scored["rows"], analysis_seed=cfg.analysis_seed)

    mlp009 = {}
    for seed in cfg.seeds:
        ckpt = MLP009 / f"mlp_s{seed}" / "best.pt"
        mlp009[str(seed)] = {}
        if not ckpt.exists():
            mlp009[str(seed)] = {"error": "missing 009 checkpoint"}
            continue
        module = NumericMLP(cfg.mlp_hidden).to(device)
        load_run_checkpoint(ckpt, module, None, strict_identity=False)
        for split, recs in splits.items():
            preds = {rec["audit"]["example_id"]: predict_lex(module, rec, None, device, cfg, "mlp") for rec in recs}
            scored = evaluate_split(recs, preds)
            mlp009[str(seed)][split] = summarize_eval(scored["rows"], analysis_seed=cfg.analysis_seed)
            dest = REPORT_DIR / "mlp009_replay" / f"s{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / f"eval_{split}.json").write_text(
                json.dumps({"n": scored["n"], "summary": mlp009[str(seed)][split]}, indent=2, sort_keys=True) + "\n"
            )

    learned = {}
    paired = {}
    for seed in cfg.seeds:
        run_dir = Path(args.out_root) / f"mlp_s{seed}"
        learned[str(seed)] = {}
        paired[str(seed)] = {}
        man = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
        metrics = load_jsonl(run_dir / "metrics.jsonl") if (run_dir / "metrics.jsonl").exists() else []
        learned[str(seed)]["manifest"] = {
            "step": (man.get("train_state") or {}).get("step"),
            "updates": (man.get("train_state") or {}).get("optimizer_updates"),
            "best": (man.get("train_state") or {}).get("best"),
            "incomplete": man.get("incomplete"),
            "stop_reason": man.get("stop_reason"),
        }
        learned[str(seed)]["curve"] = [
            {"split": r.get("split"), "step": r.get("step"), "epoch": r.get("epoch"), "kl": r.get("kl"), "lr": r.get("lr")}
            for r in metrics
        ]
        for split, recs in splits.items():
            pred_path = run_dir / f"eval_{split}_preds.jsonl"
            if not pred_path.exists():
                learned[str(seed)][split] = None
                continue
            preds = load_pred_lex(pred_path)
            scored = evaluate_split(recs, preds)
            learned[str(seed)][split] = summarize_eval(scored["rows"], analysis_seed=cfg.analysis_seed)
            by_id = {row["example_id"]: row for row in scored["rows"]}
            diffs = {}
            for name in ("stale", "transition_only", "independent_sameinfo"):
                ref_scored, _ = _score_named(recs, lambda rec, n=name: reference_pred_lex(rec, n))
                delta = []
                for r in ref_scored["rows"]:
                    delta.append(by_id[r["example_id"]]["joint_kl"] - r["joint_kl"])
                diffs[name] = {
                    "mean_kl_minus_ref": sum(delta) / len(delta),
                    "n_better": sum(d < 0 for d in delta),
                    "n": len(delta),
                }
            if mlp009.get(str(seed), {}).get(split):
                # paired vs 009 uses identical examples; 009 summary only — recompute from replay file if present
                pass
            paired[str(seed)][split] = diffs

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    blob = {"refs": refs, "mlp009": mlp009, "learned": learned, "paired_vs_refs": paired}
    (REPORT_DIR / "analysis.json").write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"wrote": str(REPORT_DIR / "analysis.json")}))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    rc = cmd_fit(args)
    if rc != 0:
        return rc
    rc = cmd_eval(args)
    if rc != 0:
        return rc
    return cmd_analyze(args)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_YAML)
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    sub.add_parser("eval")
    sub.add_parser("analyze")
    sub.add_parser("run")
    args = p.parse_args(argv)
    args.device = "cpu"
    if args.cmd == "fit":
        return cmd_fit(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "analyze":
        return cmd_analyze(args)
    if args.cmd == "run":
        return cmd_run(args)
    raise SystemExit(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
