"""Fit, resume, probe, and numeric MLP for the ordinary-KL pilot."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from llm_rl_engine.analysis.learning_pilot import (
    REPO,
    PilotConfig,
    append_jsonl,
    atomic_save,
    experiment_identity,
    file_sha256,
    identity_mismatch_fields,
    join_records,
    load_jsonl,
    reconcile_metrics,
    resolve_warmup_steps,
    runtime_provenance,
)
from llm_rl_engine.analysis.posterior_baseline import (
    LEX_KEYS,
    numeric_encode,
    score_prediction,
)
from llm_rl_engine.analysis.posterior_baseline_model import (
    JointHead,
    PosteriorReader,
    apply_lora,
    encode_batch,
    freeze_backbone,
    kl_loss,
    load_backbone,
    load_tokenizer,
    rng_capture,
    rng_restore,
    set_cpu_threads,
    snapshot_params,
)

MLP_IN = 47


class NumericMLP(nn.Module):
    def __init__(self, hidden: int = 64, n_states: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MLP_IN, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_states),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.net(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return {"logits": logits, "log_probs": log_probs, "probs": log_probs.exp()}


def require_device(requested: str) -> torch.device:
    requested = requested or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        if "+cpu" in torch.__version__ and "+cu" not in torch.__version__:
            raise RuntimeError(f"CUDA requested but torch is a CPU build: {torch.__version__}")
        probe = torch.zeros(1, device="cuda")
        if probe.device.type != "cuda":
            raise RuntimeError("CUDA requested but tensor placement is not cuda")
        return torch.device("cuda")
    if requested != "cpu":
        raise ValueError(requested)
    return torch.device("cpu")


def make_head(hidden_size: int, seed: int) -> JointHead:
    g = torch.Generator()
    g.manual_seed(int(seed))
    head = JointHead(hidden_size)
    bound = 1.0 / math.sqrt(hidden_size)
    with torch.no_grad():
        head.proj.weight.uniform_(-bound, bound, generator=g)
        head.proj.bias.uniform_(-bound, bound, generator=g)
    return head


def build_reader_matched(
    mode: str,
    cfg: PilotConfig,
    *,
    cache_dir: Path,
    device: torch.device,
    head_seed: int,
    adapter_seed: int | None = None,
) -> tuple[Any, Any]:
    tok = load_tokenizer(cache_dir, revision=cfg.model_revision)
    backbone = load_backbone(cache_dir, revision=cfg.model_revision, device=device)
    freeze_backbone(backbone)
    if mode == "lora":
        torch.manual_seed(int(adapter_seed if adapter_seed is not None else head_seed + 10_007))
        backbone = apply_lora(
            backbone,
            r=cfg.lora_r,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
            targets=cfg.lora_targets,
        )
        for name, p in backbone.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
    elif mode != "head":
        raise ValueError(mode)
    hidden = int(backbone.config.hidden_size)
    head = make_head(hidden, head_seed).to(device)
    reader = PosteriorReader(backbone, head).to(device)
    if next(reader.parameters()).device.type != device.type:
        raise RuntimeError("model not on requested device")
    return reader, tok


def param_groups_decay(module: nn.Module, lr: float, cfg: PilotConfig, *, extra: list[nn.Parameter] | None = None) -> list[dict[str, Any]]:
    decay, nodecay = [], []
    seen = set()
    params = list(module.named_parameters())
    if extra:
        params += [(f"extra.{i}", p) for i, p in enumerate(extra)]
    for name, p in params:
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if name.endswith("bias") or ".bias" in name:
            nodecay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": cfg.weight_decay})
    if nodecay:
        groups.append({"params": nodecay, "lr": lr, "weight_decay": cfg.weight_decay_bias})
    return groups


def reader_param_groups(reader: PosteriorReader, cfg: PilotConfig) -> list[dict[str, Any]]:
    groups = param_groups_decay(reader.head, cfg.head_lr, cfg)
    lora = [p for n, p in reader.backbone.named_parameters() if p.requires_grad]
    if lora:
        decay, nodecay = [], []
        for n, p in reader.backbone.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith("bias"):
                nodecay.append(p)
            else:
                decay.append(p)
        if decay:
            groups.append({"params": decay, "lr": cfg.lora_lr, "weight_decay": cfg.weight_decay})
        if nodecay:
            groups.append({"params": nodecay, "lr": cfg.lora_lr, "weight_decay": cfg.weight_decay_bias})
    return groups


def make_scheduler(opt: torch.optim.Optimizer, cfg: PilotConfig) -> LambdaLR:
    total = int(cfg.optimizer_steps_per_run)
    warmup = resolve_warmup_steps(cfg)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 1.0 - progress)

    return LambdaLR(opt, lr_lambda)


def _strict_load_trainable(module: nn.Module, payload: Mapping[str, torch.Tensor]) -> None:
    current = {n: p for n, p in module.named_parameters() if p.requires_grad}
    extra = set(payload) - set(current)
    missing = set(current) - set(payload)
    if extra or missing:
        raise RuntimeError(f"checkpoint tensor mismatch extra={sorted(extra)[:8]} missing={sorted(missing)[:8]}")
    for n, p in current.items():
        p.data.copy_(payload[n].to(p.device))


def save_run_checkpoint(
    path: Path,
    *,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    identity: Mapping[str, Any],
    train_state: Mapping[str, Any],
) -> None:
    payload = {
        "identity": dict(identity),
        "train_state": dict(train_state),
        "trainable": {n: p.detach().cpu().clone() for n, p in module.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": rng_capture(),
        "lex_keys": list(LEX_KEYS),
    }
    atomic_save(path, payload)


def load_run_checkpoint(
    path: Path,
    module: nn.Module,
    identity: Mapping[str, Any] | None = None,
    *,
    strict_identity: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved_id = payload.get("identity") or {}
    extra = set(payload.get("trainable") or {}) - {n for n, p in module.named_parameters() if p.requires_grad}
    missing = {n for n, p in module.named_parameters() if p.requires_grad} - set(payload.get("trainable") or {})
    if extra or missing:
        raise RuntimeError(f"checkpoint tensor mismatch extra={sorted(extra)[:8]} missing={sorted(missing)[:8]}")
    if strict_identity:
        if identity is None:
            raise RuntimeError("strict identity check requires a live identity")
        bad = identity_mismatch_fields(saved_id, identity)
        if bad:
            raise RuntimeError(f"identity mismatch on {bad}")
    _strict_load_trainable(module, payload["trainable"])
    return payload


def row_numeric_features(row: Mapping[str, Any]) -> list[float]:
    m = row["model_input"]
    return numeric_encode(
        q_previous_lex=m["q_previous_lex"],
        partition=m["partition"],
        hazard=m["hazard"],
        suffix=m["suffix"],
    )


def precompute_mlp_features(rows: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    feats = [row_numeric_features(r) for r in rows]
    if rows:
        for i in {0, len(rows) // 2, len(rows) - 1}:
            live = row_numeric_features(rows[i])
            if live != feats[i]:
                raise RuntimeError(f"numeric encoder mismatch at row {i}")
    return feats


def encode_mlp_batch(
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    cache: Sequence[Sequence[float]] | None = None,
    indices: Sequence[int] | None = None,
) -> torch.Tensor:
    if cache is not None and indices is not None:
        feats = [cache[i] for i in indices]
    else:
        feats = [row_numeric_features(r) for r in rows]
    return torch.tensor(feats, dtype=torch.float32, device=device)


def _batch_kl(module, rows, tokenizer, device, cfg, mode, *, feature_cache=None, row_indices=None):
    teacher = torch.tensor([r["audit"]["q_current_lex"] for r in rows], dtype=torch.float32, device=device)
    if mode == "mlp":
        x = encode_mlp_batch(rows, device, cache=feature_cache, indices=row_indices)
        out = module(x)
    else:
        texts = [r["model_input"]["text"] for r in rows]
        enc = encode_batch(tokenizer, texts, device=device, max_length=cfg.max_length)
        out = module(**enc)
    return kl_loss(out["log_probs"], teacher), out


def eval_dev_kl(module, rows, tokenizer, device, cfg, mode, batch_size: int, *, feature_cache=None) -> float:
    module.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            idxs = list(range(i, i + len(chunk))) if feature_cache is not None else None
            loss, _ = _batch_kl(
                module, chunk, tokenizer, device, cfg, mode,
                feature_cache=feature_cache, row_indices=idxs,
            )
            total += float(loss.cpu()) * len(chunk)
            n += len(chunk)
    return total / max(n, 1)


def fit_run(
    cfg: PilotConfig,
    *,
    mode: str,
    seed: int,
    data_dir: Path,
    out_dir: Path,
    device: torch.device,
    resume: bool = False,
    max_steps: int | None = None,
    module: nn.Module | None = None,
    tokenizer=None,
    train_rows: list[dict] | None = None,
    dev_rows: list[dict] | None = None,
    holdout_guard: bool = True,
    deadline_monotonic: float | None = None,
    extra_code: Sequence[str] = (),
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    loaded_from_files = train_rows is None
    if holdout_guard:
        for name in ("same_topology", "structural"):
            if loaded_from_files:
                continue
    out_dir.mkdir(parents=True, exist_ok=True)
    identity_path = out_dir / "identity.json"
    last_path = out_dir / "last.pt"
    if not resume and (identity_path.exists() or last_path.exists()):
        raise RuntimeError(f"refusing non-resume start in existing run directory {out_dir}")
    if train_rows is None:
        train_rows = join_records(load_jsonl(data_dir / "train_model.jsonl"), load_jsonl(data_dir / "train_audit.jsonl"))
        dev_rows = join_records(load_jsonl(data_dir / "dev_model.jsonl"), load_jsonl(data_dir / "dev_audit.jsonl"))
    live_identity = experiment_identity(
        cfg,
        mode=mode,
        seed=seed,
        data_dir=data_dir,
        train_rows=train_rows,
        dev_rows=dev_rows,
        extra_code=extra_code,
    )
    live_identity["device"] = str(device)
    live_identity["torch"] = torch.__version__
    live_identity["dependencies"] = {"torch": torch.__version__}
    live_identity["trainable_parameter_count"] = None
    if module is None:
        if mode == "mlp":
            torch.manual_seed(seed)
            module = NumericMLP(cfg.mlp_hidden).to(device)
            tokenizer = None
            opt = torch.optim.AdamW(param_groups_decay(module, cfg.mlp_lr, cfg), betas=cfg.betas, eps=cfg.eps)
        else:
            cache = Path(cfg.cache_dir)
            if not cache.is_absolute():
                cache = REPO / cache
            module, tokenizer = build_reader_matched(
                mode, cfg, cache_dir=cache, device=device, head_seed=seed,
                adapter_seed=seed + 10_007,
            )
            opt = torch.optim.AdamW(reader_param_groups(module, cfg), betas=cfg.betas, eps=cfg.eps)
    else:
        lr = cfg.mlp_lr if mode == "mlp" else cfg.head_lr
        opt = torch.optim.AdamW(param_groups_decay(module, lr, cfg), betas=cfg.betas, eps=cfg.eps)
    live_identity["trainable_parameter_count"] = sum(p.numel() for p in module.parameters() if p.requires_grad)
    sched = make_scheduler(opt, cfg)
    n = len(train_rows)
    eb = cfg.effective_batch
    mb = cfg.microbatch
    shuffle_rng = random.Random(seed)
    state = {
        "epoch": 0,
        "step": 0,
        "cursor": 0,
        "permutation": list(range(n)),
        "last_dev_epoch": -1,
        "best": {"dev_kl": float("inf"), "step": -1, "epoch": -1},
        "optimizer_updates": 0,
        "last_train": None,
        "last_dev": None,
        "incomplete": False,
        "stop_reason": None,
    }
    metrics_path = out_dir / "metrics.jsonl"
    best_path = out_dir / "best.pt"
    feature_cache = precompute_mlp_features(train_rows) if mode == "mlp" else None
    dev_feature_cache = precompute_mlp_features(dev_rows) if mode == "mlp" else None
    identity = live_identity
    if resume:
        payload = load_run_checkpoint(last_path, module, live_identity, strict_identity=True)
        identity = dict(payload["identity"])
        opt.load_state_dict(payload["optimizer"])
        sched.load_state_dict(payload["scheduler"])
        rng_restore(payload["rng"])
        state = dict(payload["train_state"])
        shuffle_rng.setstate(state["shuffle_rng"])
        mig = runtime_provenance(identity, live_identity)
        append_jsonl(
            out_dir / "continuations.jsonl",
            {
                "event": "resume",
                "step": state["step"],
                "epoch": state["epoch"],
                "device_migration": mig,
            },
        )
        reconcile_metrics(metrics_path, state)
    else:
        identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
        if metrics_path.exists():
            metrics_path.unlink()
    limit = max_steps if max_steps is not None else cfg.optimizer_steps_per_run
    stop_reason = "completed"

    def persist(kind: str) -> None:
        packed = dict(state)
        packed["shuffle_rng"] = shuffle_rng.getstate()
        path = last_path if kind == "last" else best_path
        save_run_checkpoint(path, module=module, optimizer=opt, scheduler=sched, identity=identity, train_state=packed)

    def finish_epoch_if_needed() -> None:
        if state["cursor"] < n:
            return
        did_dev = False
        if state["last_dev_epoch"] != state["epoch"]:
            dev_kl = eval_dev_kl(
                module, dev_rows, tokenizer, device, cfg, mode, eb,
                feature_cache=dev_feature_cache,
            )
            state["last_dev"] = {
                "split": "dev",
                "step": state["step"],
                "epoch": state["epoch"],
                "kl": dev_kl,
            }
            state["last_dev_epoch"] = state["epoch"]
            if dev_kl < state["best"]["dev_kl"]:
                state["best"] = {"dev_kl": dev_kl, "step": state["step"], "epoch": state["epoch"]}
                persist("best")
            did_dev = True
        state["epoch"] += 1
        state["cursor"] = 0
        persist("last")
        if did_dev:
            append_jsonl(metrics_path, dict(state["last_dev"]))

    def timed_out() -> bool:
        return deadline_monotonic is not None and time.monotonic() >= deadline_monotonic

    finish_epoch_if_needed()
    while state["step"] < limit:
        if timed_out():
            stop_reason = "wall_time"
            state["incomplete"] = True
            state["stop_reason"] = stop_reason
            persist("last")
            break
        finish_epoch_if_needed()
        if state["cursor"] == 0:
            state["permutation"] = list(range(n))
            shuffle_rng.shuffle(state["permutation"])
        idx = state["permutation"][state["cursor"] : state["cursor"] + eb]
        if not idx:
            raise RuntimeError("empty batch; pending epoch work should have been completed first")
        module.train()
        opt.zero_grad(set_to_none=True)
        batch_ids = [train_rows[i]["audit"]["example_id"] for i in idx]
        n_micro = 0
        loss_acc = 0.0
        n_ex = 0
        for start in range(0, len(idx), mb):
            micro_i = idx[start : start + mb]
            rows = [train_rows[i] for i in micro_i]
            loss, _ = _batch_kl(
                module, rows, tokenizer, device, cfg, mode,
                feature_cache=feature_cache, row_indices=list(micro_i),
            )
            (loss * (len(micro_i) / len(idx))).backward()
            loss_acc += float(loss.detach().cpu()) * len(micro_i)
            n_ex += len(micro_i)
            n_micro += 1
        torch.nn.utils.clip_grad_norm_([p for p in module.parameters() if p.requires_grad], cfg.grad_clip)
        opt.step()
        sched.step()
        state["step"] += 1
        state["optimizer_updates"] += 1
        state["cursor"] += eb
        lr_now = sched.get_last_lr()[0]
        train_kl = loss_acc / max(n_ex, 1)
        state["last_train"] = {
            "split": "train",
            "step": state["step"],
            "epoch": state["epoch"],
            "kl": train_kl,
            "lr": lr_now,
            "batch_ids": batch_ids,
            "n_microbatches": n_micro,
        }
        persist("last")
        append_jsonl(metrics_path, dict(state["last_train"]))
        print(
            f"{mode} seed={seed} step={state['step']}/{limit} epoch={state['epoch']} "
            f"train_kl={train_kl:.4f} lr={lr_now:.2e}",
            flush=True,
        )
        finish_epoch_if_needed()
        if timed_out():
            stop_reason = "wall_time"
            state["incomplete"] = True
            state["stop_reason"] = stop_reason
            persist("last")
            break
    else:
        finish_epoch_if_needed()
        if state["step"] >= limit and not state.get("incomplete"):
            stop_reason = "completed"
            state["stop_reason"] = stop_reason
            persist("last")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "identity": identity,
                "train_state": {k: v for k, v in state.items() if k not in ("permutation", "shuffle_rng")},
                "n_train": n,
                "heldout_evaluated": False,
                "incomplete": bool(state.get("incomplete")),
                "stop_reason": state.get("stop_reason") or stop_reason,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    return {"state": state, "identity": identity, "out_dir": str(out_dir), "stop_reason": state.get("stop_reason") or stop_reason}


def predict_lex(module, row, tokenizer, device, cfg, mode) -> list[float]:
    module.eval()
    with torch.no_grad():
        _, out = _batch_kl(module, [row], tokenizer, device, cfg, mode)
    return out["probs"][0].cpu().tolist()
