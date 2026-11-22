"""Bounded local verification for ordinary posterior-KL (task 008).

At most four optimizer steps per configuration. Not a training result.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import time

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "reports" / "posterior_baseline_preflight" / "data"
SCRATCH = REPO / "reports" / "posterior_baseline_preflight" / "scratch"
HF_CACHE = REPO / "models" / "hf"


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_jsonl(path: Path, n: int | None = None) -> list[dict]:
    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            rows.append(json.loads(line))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=HF_CACHE)
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)

    os.environ.setdefault("HF_HOME", str(args.cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(args.cache_dir))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from llm_rl_engine.analysis.posterior_baseline_model import (
        MODEL_ID,
        MODEL_REVISION,
        MAX_VERIFY_STEPS,
        TrainConfig,
        build_reader,
        encode_batch,
        kl_loss,
        max_abs_diff,
        param_groups,
        rng_capture,
        rng_restore,
        save_checkpoint,
        load_head_and_adapter,
        select_device,
        set_cpu_threads,
        snapshot_params,
        trainable_name_set,
    )
    import torch

    set_cpu_threads(args.cpu_threads)
    device = select_device()
    dtype_note = "float32 (Turing GPUs are not assumed to have native BF16)"
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    train_model = load_jsonl(DATA / "train_model.jsonl", 8)
    train_audit = load_jsonl(DATA / "train_audit.jsonl", 8)
    texts = [r["text"] for r in train_model[:4]]
    teachers = torch.tensor([train_audit[i]["q_current_lex"] for i in range(4)], dtype=torch.float32)

    report: dict = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cpu_threads": args.cpu_threads,
        "dtype": "float32",
        "dtype_note": dtype_note,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_verify_steps_allowed": MAX_VERIFY_STEPS,
        "rss_mb_start": rss_mb(),
    }

    t_load = time.perf_counter()
    reader, tok = build_reader("head", args.cache_dir, device=device)
    report["load_s"] = time.perf_counter() - t_load
    report["hidden_size"] = int(reader.backbone.config.hidden_size)
    report["pad_token"] = tok.pad_token
    report["chat_template_used"] = False

    encoded = encode_batch(tok, texts, device=device)
    report["batch_token_lengths"] = encoded["attention_mask"].sum(dim=1).tolist()
    report["max_batch_tokens"] = int(encoded["attention_mask"].sum(dim=1).max())
    if report["max_batch_tokens"] > 1024:
        raise SystemExit("prompt exceeded 1024 tokens")

    reader.eval()
    with torch.no_grad():
        out = reader(**encoded)
    probs = out["probs"].cpu()
    report["prob_row_sums"] = probs.sum(dim=-1).tolist()
    loss0 = float(kl_loss(out["log_probs"], teachers.to(device)[: len(texts)]).cpu())
    report["forward_kl"] = loss0
    report["forward_finite"] = bool(torch.isfinite(out["log_probs"]).all().item())

    # mixed-length last-token: compare batched vs single
    solo = []
    for text in texts:
        enc = encode_batch(tok, [text], device=device)
        with torch.no_grad():
            solo.append(reader(**enc)["log_probs"][0].cpu())
    batched = out["log_probs"].cpu()
    report["mixed_length_max_abs"] = max(
        float((batched[i] - solo[i]).abs().max()) for i in range(len(texts))
    )

    del reader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    batch_texts = texts[:2]
    batch_teacher = teachers[:2].to(device)
    verify_modes = []
    for mode in ("head", "lora"):
        cfg = TrainConfig(name=f"preflight_{mode}", mode=mode, seed=0)
        reader, tok = build_reader(mode, args.cache_dir, device=device)
        enc = encode_batch(tok, batch_texts, device=device)
        opt = torch.optim.AdamW(param_groups(reader, cfg), betas=(0.9, 0.999))
        names_train = trainable_name_set(reader)
        names_frozen = {n for n, p in reader.named_parameters() if not p.requires_grad}
        before = snapshot_params(reader)
        reader.train()
        opt.zero_grad(set_to_none=True)
        out1 = reader(**enc)
        loss1 = kl_loss(out1["log_probs"], batch_teacher)
        loss1.backward()
        torch.nn.utils.clip_grad_norm_([p for p in reader.parameters() if p.requires_grad], cfg.grad_clip)
        opt.step()
        after1 = snapshot_params(reader)
        frozen_delta = max_abs_diff(before, after1, sorted(names_frozen)) if names_frozen else 0.0
        train_delta = max_abs_diff(before, after1, sorted(names_train))
        rng_after1 = rng_capture()
        ckpt = args.scratch_dir / f"{mode}_step1.pt"
        save_checkpoint(ckpt, reader=reader, optimizer=opt, cfg=cfg, step=1, rng_state=rng_after1)

        opt.zero_grad(set_to_none=True)
        out2 = reader(**enc)
        loss2 = kl_loss(out2["log_probs"], batch_teacher)
        loss2.backward()
        opt.step()
        after2_cont = snapshot_params(reader)
        log_cont = out2["log_probs"].detach().cpu()

        reader2, tok2 = build_reader(mode, args.cache_dir, device=device)
        payload = load_head_and_adapter(ckpt, reader2, cfg)
        opt2 = torch.optim.AdamW(param_groups(reader2, cfg), betas=(0.9, 0.999))
        opt2.load_state_dict(payload["optimizer"])
        rng_restore(payload["rng"])
        enc2 = encode_batch(tok2, batch_texts, device=device)
        reader2.train()
        opt2.zero_grad(set_to_none=True)
        out2b = reader2(**enc2)
        loss2b = kl_loss(out2b["log_probs"], batch_teacher)
        loss2b.backward()
        opt2.step()
        after2_resume = snapshot_params(reader2)
        resume_delta = max_abs_diff(after2_cont, after2_resume, sorted(names_train))
        verify_modes.append(
            {
                "mode": mode,
                "n_trainable": len(names_train),
                "n_frozen": len(names_frozen),
                "step1_loss": float(loss1.detach().cpu()),
                "step2_loss": float(loss2.detach().cpu()),
                "step2_resume_loss": float(loss2b.detach().cpu()),
                "frozen_max_abs_change": frozen_delta,
                "trainable_max_abs_change": train_delta,
                "resume_trainable_max_abs": resume_delta,
                "resume_logprob_max_abs": float((log_cont - out2b["log_probs"].detach().cpu()).abs().max()),
                "optimizer_steps_executed": 3,
                "unique_optimizer_steps": 2,
                "checkpoint_bytes": ckpt.stat().st_size,
            }
        )
        del reader, reader2, opt, opt2
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # length histogram on a 64-example training sample (no holdout model eval)
    sample = load_jsonl(DATA / "train_model.jsonl", 64)
    tok = None
    from llm_rl_engine.analysis.posterior_baseline_model import load_tokenizer

    tok = load_tokenizer(args.cache_dir)
    lens = [len(tok(r["text"], add_special_tokens=True)["input_ids"]) for r in sample]
    report["token_length_sample64"] = {
        "n": len(lens),
        "min": min(lens),
        "max": max(lens),
        "mean": sum(lens) / len(lens),
    }
    report["modes"] = verify_modes
    report["rss_mb_peak"] = rss_mb()
    if torch.cuda.is_available():
        report["cuda_peak_alloc_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        report["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        report["cuda_peak_alloc_mb"] = None
        report["gpu_name"] = None
    report["packages"] = {
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
    }
    out_path = REPO / "reports" / "posterior_baseline_preflight" / "verification.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"device": report["device"], "forward_kl": report["forward_kl"], "modes": [m["mode"] for m in verify_modes]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
