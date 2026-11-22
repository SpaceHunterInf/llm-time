"""Qwen3-0.6B-Base 16-state readout for ordinary posterior KL.

Imported only from the optional `train` dependency group. Not a new method.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

from llm_rl_engine.analysis.posterior_baseline import (
    LEX_KEYS,
    MAX_INPUT_TOKENS,
    lex_joint_to_legacy,
    marginals,
)
from llm_rl_engine.analysis.selective_revision import decision_values
from llm_rl_engine.analysis.posterior_baseline import ACTION_MAP

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGETS = ("q_proj", "v_proj")
HEAD_LR = 1e-3
LORA_LR = 1e-4
MAX_VERIFY_STEPS = 4


@dataclass(frozen=True)
class TrainConfig:
    name: str
    mode: str
    seed: int
    epochs: int = 3
    batch_size: int = 32
    head_lr: float = HEAD_LR
    lora_lr: float = LORA_LR
    max_length: int = MAX_INPUT_TOKENS
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION


def set_cpu_threads(n: int = 4) -> None:
    torch.set_num_threads(n)
    torch.set_num_interop_threads(min(n, 2))


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class JointHead(nn.Module):
    def __init__(self, hidden_size: int, n_states: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, n_states)

    def forward(self, last_hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(last_hidden)


class PosteriorReader(nn.Module):
    def __init__(self, backbone: nn.Module, head: JointHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        lengths = attention_mask.sum(dim=1) - 1
        b = torch.arange(hidden.size(0), device=hidden.device)
        last = hidden[b, lengths]
        logits = self.head(last)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        return {"logits": logits, "log_probs": log_probs, "probs": probs, "lengths": lengths}


def kl_loss(log_probs: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    teacher = teacher.float()
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return F.kl_div(log_probs, teacher, reduction="batchmean", log_target=False)


def load_tokenizer(cache_dir: str | Path, revision: str = MODEL_REVISION) -> Any:
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=revision,
        cache_dir=str(cache_dir),
        trust_remote_code=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_backbone(cache_dir: str | Path, revision: str = MODEL_REVISION, device: torch.device | None = None) -> nn.Module:
    device = device or select_device()
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=revision,
        cache_dir=str(cache_dir),
        dtype=torch.float32,
        trust_remote_code=False,
    )
    return model.to(device)


def freeze_backbone(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def apply_lora(
    backbone: nn.Module,
    *,
    r: int = LORA_R,
    alpha: int = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
    targets: Sequence[str] = LORA_TARGETS,
) -> nn.Module:
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(targets),
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(backbone, cfg)


def build_reader(mode: str, cache_dir: str | Path, device: torch.device | None = None) -> tuple[PosteriorReader, Any]:
    device = device or select_device()
    tok = load_tokenizer(cache_dir)
    backbone = load_backbone(cache_dir, device=device)
    hidden = int(backbone.config.hidden_size)
    if mode == "head":
        freeze_backbone(backbone)
    elif mode == "lora":
        freeze_backbone(backbone)
        backbone = apply_lora(backbone)
        for p in backbone.parameters():
            if p.requires_grad is False:
                continue
        for name, p in backbone.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
    else:
        raise ValueError(mode)
    head = JointHead(hidden).to(device)
    reader = PosteriorReader(backbone, head).to(device)
    return reader, tok


def encode_batch(
    tokenizer,
    texts: Sequence[str],
    *,
    device: torch.device,
    max_length: int = MAX_INPUT_TOKENS,
) -> dict[str, torch.Tensor]:
    raw = tokenizer(
        list(texts),
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=True,
    )
    lengths = raw["attention_mask"].sum(dim=1)
    if int(lengths.max()) > max_length:
        raise ValueError(f"sequence length {int(lengths.max())} exceeds {max_length}; refuse silent truncation")
    return {k: v.to(device) for k, v in raw.items()}


def param_groups(reader: PosteriorReader, cfg: TrainConfig) -> list[dict[str, Any]]:
    head_params = list(reader.head.parameters())
    lora_params = [p for n, p in reader.backbone.named_parameters() if p.requires_grad and not n.startswith("ignore")]
    lora_params = [p for n, p in reader.backbone.named_parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": cfg.head_lr, "weight_decay": 0.0}]
    if lora_params:
        groups.append({"params": lora_params, "lr": cfg.lora_lr, "weight_decay": 0.0})
    return groups


def trainable_name_set(reader: PosteriorReader) -> set[str]:
    return {n for n, p in reader.named_parameters() if p.requires_grad}


def snapshot_params(reader: PosteriorReader) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in reader.named_parameters()}


def max_abs_diff(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor], names: Sequence[str] | None = None) -> float:
    keys = names if names is not None else a.keys()
    best = 0.0
    for n in keys:
        best = max(best, float((a[n] - b[n]).abs().max()))
    return best


def predicted_action(probs_lex: Sequence[float], query_component: int) -> str:
    legacy = lex_joint_to_legacy(probs_lex)
    p = marginals(legacy)[query_component]
    raw = decision_values(p)
    return ACTION_MAP[raw["optimal_action"]]


def save_checkpoint(
    path: Path,
    *,
    reader: PosteriorReader,
    optimizer: torch.optim.Optimizer | None,
    cfg: TrainConfig,
    step: int,
    rng_state: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable = {n: p.detach().cpu().clone() for n, p in reader.named_parameters() if p.requires_grad}
    payload = {
        "step": step,
        "config": asdict(cfg),
        "model_revision": MODEL_REVISION,
        "head": {k: v.detach().cpu().clone() for k, v in reader.head.state_dict().items()},
        "trainable": trainable,
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "rng": rng_state,
        "lex_keys": list(LEX_KEYS),
    }
    torch.save(payload, path)


def load_head_and_adapter(
    path: Path,
    reader: PosteriorReader,
    cfg: TrainConfig,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    reader.head.load_state_dict(payload["head"])
    current = dict(reader.named_parameters())
    for name, tensor in payload["trainable"].items():
        if name in current:
            current[name].data.copy_(tensor.to(current[name].device))
    return payload


def rng_capture() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def rng_restore(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
