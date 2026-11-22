"""External ordinary-KL pilot. Do not launch the six runs from task 008.

Commands:

  UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --group train python scripts/posterior_baseline_train.py probe \\
    --cache-dir models/hf --device cuda

  UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --group train python scripts/posterior_baseline_train.py fit \\
    --mode head --seed 0 --data-dir reports/posterior_baseline_preflight/data --out-dir runs/posterior_baseline/head_s0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def probe(args: argparse.Namespace) -> int:
    import torch
    from llm_rl_engine.analysis.posterior_baseline_model import (
        TrainConfig,
        build_reader,
        encode_batch,
        kl_loss,
        param_groups,
        select_device,
        set_cpu_threads,
    )

    set_cpu_threads(args.cpu_threads)
    device = torch.device(args.device) if args.device else select_device()
    data = load_jsonl(Path(args.data_dir) / "train_model.jsonl")[: args.batch_size]
    audit = load_jsonl(Path(args.data_dir) / "train_audit.jsonl")[: args.batch_size]
    cfg = TrainConfig(name="probe", mode=args.mode, seed=0, batch_size=args.batch_size)
    t0 = time.perf_counter()
    reader, tok = build_reader(args.mode, args.cache_dir, device=device)
    load_s = time.perf_counter() - t0
    texts = [r["text"] for r in data]
    teacher = torch.tensor([a["q_current_lex"] for a in audit], dtype=torch.float32, device=device)
    enc = encode_batch(tok, texts, device=device)
    opt = torch.optim.AdamW(param_groups(reader, cfg))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    reader.train()
    opt.zero_grad(set_to_none=True)
    out = reader(**enc)
    loss = kl_loss(out["log_probs"], teacher)
    loss.backward()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    step_s = time.perf_counter() - t1
    tokens = int(enc["attention_mask"].sum())
    peak = None
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    result = {
        "device": str(device),
        "mode": args.mode,
        "batch_size": args.batch_size,
        "tokens": tokens,
        "load_s": load_s,
        "step_s": step_s,
        "examples_per_s": args.batch_size / step_s,
        "tokens_per_s": tokens / step_s,
        "peak_alloc_mb": peak,
        "note": "Hardware probe only. Not a training result. CPU timings are not GPU estimates.",
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


def fit(args: argparse.Namespace) -> int:
    """Ordinary-KL pilot. Not invoked by task 008 preflight."""
    import math
    import random
    import torch
    from torch.optim.lr_scheduler import LambdaLR
    from llm_rl_engine.analysis.posterior_baseline import sha256_text
    from llm_rl_engine.analysis.posterior_baseline_model import (
        MODEL_REVISION,
        TrainConfig,
        build_reader,
        encode_batch,
        kl_loss,
        param_groups,
        save_checkpoint,
        select_device,
        set_cpu_threads,
        rng_capture,
    )

    set_cpu_threads(args.cpu_threads)
    device = torch.device(args.device) if args.device else select_device()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = TrainConfig(name=f"{args.mode}_s{args.seed}", mode=args.mode, seed=args.seed)
    train = load_jsonl(Path(args.data_dir) / "train_model.jsonl")
    train_y = load_jsonl(Path(args.data_dir) / "train_audit.jsonl")
    dev = load_jsonl(Path(args.data_dir) / "dev_model.jsonl")
    dev_y = load_jsonl(Path(args.data_dir) / "dev_audit.jsonl")
    order = list(range(len(train)))
    rng = random.Random(args.seed)
    reader, tok = build_reader(args.mode, args.cache_dir, device=device)
    opt = torch.optim.AdamW(param_groups(reader, cfg), betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(cfg.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 1.0 - progress)

    sched = LambdaLR(opt, lr_lambda)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"dev_kl": float("inf"), "step": -1}
    step = 0
    log = []
    for epoch in range(cfg.epochs):
        rng.shuffle(order)
        reader.train()
        for start in range(0, len(train), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            texts = [train[i]["text"] for i in idx]
            teacher = torch.tensor([train_y[i]["q_current_lex"] for i in idx], dtype=torch.float32, device=device)
            enc = encode_batch(tok, texts, device=device, max_length=cfg.max_length)
            opt.zero_grad(set_to_none=True)
            out = reader(**enc)
            loss = kl_loss(out["log_probs"], teacher)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in reader.parameters() if p.requires_grad], cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            log.append({"split": "train", "step": step, "epoch": epoch, "kl": float(loss.detach().cpu())})
        reader.eval()
        dev_losses = []
        with torch.no_grad():
            for start in range(0, len(dev), cfg.batch_size):
                texts = [r["text"] for r in dev[start : start + cfg.batch_size]]
                teacher = torch.tensor(
                    [a["q_current_lex"] for a in dev_y[start : start + cfg.batch_size]],
                    dtype=torch.float32,
                    device=device,
                )
                enc = encode_batch(tok, texts, device=device, max_length=cfg.max_length)
                out = reader(**enc)
                dev_losses.append(float(kl_loss(out["log_probs"], teacher).cpu()) * len(texts))
        dev_kl = sum(dev_losses) / len(dev)
        log.append({"split": "dev", "step": step, "epoch": epoch, "kl": dev_kl})
        if dev_kl < best["dev_kl"]:
            best = {"dev_kl": dev_kl, "step": step, "epoch": epoch}
            save_checkpoint(
                out_dir / "best.pt",
                reader=reader,
                optimizer=opt,
                cfg=cfg,
                step=step,
                rng_state=rng_capture(),
            )
    (out_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    man = {
        "config": {
            "mode": args.mode,
            "seed": args.seed,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "head_lr": cfg.head_lr,
            "lora_lr": cfg.lora_lr,
            "model_revision": MODEL_REVISION,
        },
        "n_train": len(train),
        "n_dev": len(dev),
        "optimizer_steps": step,
        "best": best,
        "heldout_evaluated": False,
        "data_train_hash": sha256_text((Path(args.data_dir) / "train_model.jsonl").read_text()),
        "data_dev_hash": sha256_text((Path(args.data_dir) / "dev_model.jsonl").read_text()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"best": best, "steps": step, "out": str(out_dir)}))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--mode", choices=("head", "lora"), default="head")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cache-dir", type=Path, default=REPO / "models" / "hf")
    p.add_argument("--data-dir", type=Path, default=REPO / "reports" / "posterior_baseline_preflight" / "data")
    p.add_argument("--device", default="")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=None)
    f = sub.add_parser("fit")
    f.add_argument("--mode", choices=("head", "lora"), required=True)
    f.add_argument("--seed", type=int, required=True)
    f.add_argument("--data-dir", type=Path, required=True)
    f.add_argument("--out-dir", type=Path, required=True)
    f.add_argument("--cache-dir", type=Path, default=REPO / "models" / "hf")
    f.add_argument("--device", default="")
    f.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.cmd == "probe":
        return probe(args)
    return fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
