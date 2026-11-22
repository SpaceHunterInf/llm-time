"""Ordinary-KL learning-pilot package (task 009). No paired loss. No holdout training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import yaml

from llm_rl_engine.analysis.posterior_baseline import (
    FORBIDDEN_MODEL_MARKERS,
    LEX_KEYS,
    MAX_INPUT_TOKENS,
    NUMERIC_ENCODER_VERSION,
    RENDERER_VERSION,
    assert_clean_text,
    canonicalize_partition,
    canonical_json,
    family_identity,
    filter_observations,
    individual_input_hash,
    kl_div,
    legacy_joint_to_lex,
    lex_joint_to_legacy,
    numeric_encode,
    public_model_row,
    render_model_text,
    same_information_independent,
    score_prediction,
    sha256_text,
)

REPO = Path(__file__).resolve().parents[2]
V1_DATA = REPO / "reports" / "posterior_baseline_preflight" / "data"
V2_DATA = REPO / "reports" / "learning_pilot" / "data_v2"
DEFAULT_CONFIG = REPO / "llm_rl_engine" / "configs" / "learning_pilot.yaml"
CODE_HASH_PATHS = (
    "llm_rl_engine/analysis/posterior_baseline.py",
    "llm_rl_engine/analysis/posterior_baseline_model.py",
    "llm_rl_engine/analysis/learning_pilot.py",
    "llm_rl_engine/analysis/learning_pilot_train.py",
    "scripts/learning_pilot.py",
    "llm_rl_engine/configs/learning_pilot.yaml",
)

MIGRATION_IDENTITY_KEYS = ("device", "torch", "dependencies")
TRAINING_FINGERPRINT_KEYS = (
    "kind",
    "version",
    "mode",
    "seed",
    "head_seed",
    "adapter_seed",
    "mlp_seed",
    "model_id",
    "model_revision",
    "renderer_version",
    "numeric_encoder_version",
    "lex_keys",
    "dtype",
    "effective_batch",
    "microbatch",
    "epochs",
    "optimizer",
    "lora",
    "data_hashes",
    "code_hashes",
    "mlp_hidden",
    "loss",
    "optimizer_steps_per_run",
    "warmup_steps",
    "checkpoint_selection",
)


@dataclass(frozen=True)
class PilotConfig:
    kind: str
    version: str
    renderer_version: str
    numeric_encoder_version: str
    model_id: str
    model_revision: str
    loss: str
    chat_template: bool
    dtype: str
    max_length: int
    epochs: int
    effective_batch: int
    microbatch: int
    optimizer: str
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    weight_decay_bias: float
    head_lr: float
    lora_lr: float
    mlp_lr: float
    grad_clip: float
    warmup_ratio: float
    warmup_steps: int
    seeds: tuple[int, ...]
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    mlp_hidden: int
    checkpoint_selection: str
    heldout_eval: str
    optimizer_steps_per_run: int
    n_language_runs: int
    n_numeric_runs: int
    data_dir: str
    cache_dir: str
    cpu_threads: int
    analysis_seed: int
    wall_time_s: int | None

    def __post_init__(self) -> None:
        if self.kind != "ordinary_posterior_kl_pilot":
            raise ValueError(self.kind)
        if self.loss != "joint_kl_teacher_to_student":
            raise ValueError(self.loss)
        if self.chat_template:
            raise ValueError("chat_template must be false")
        if self.dtype != "float32":
            raise ValueError("dtype is frozen at float32")
        if self.effective_batch % self.microbatch != 0:
            raise ValueError("effective_batch must divide by microbatch")
        if self.checkpoint_selection != "development_kl_only":
            raise ValueError(self.checkpoint_selection)
        if int(self.warmup_steps) < 1:
            raise ValueError("warmup_steps")


ALLOWED_CONFIG_KEYS = {f.name for f in fields(PilotConfig)}


def load_config(path: Path | str, overrides: Mapping[str, Any] | None = None) -> PilotConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    extra = set(raw) - ALLOWED_CONFIG_KEYS
    if extra:
        raise ValueError(f"unknown config fields: {sorted(extra)}")
    merged = dict(raw)
    if overrides:
        unknown = set(overrides) - ALLOWED_CONFIG_KEYS
        if unknown:
            raise ValueError(f"unknown overrides: {sorted(unknown)}")
        merged.update(overrides)
    merged["seeds"] = tuple(merged["seeds"])
    merged["lora_targets"] = tuple(merged["lora_targets"])
    merged["betas"] = tuple(merged["betas"])
    if merged.get("warmup_steps") is None:
        merged["warmup_steps"] = max(1, int(merged["warmup_ratio"] * merged["optimizer_steps_per_run"]))
    merged["wall_time_s"] = merged.get("wall_time_s")
    return PilotConfig(**merged)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes(root: Path | None = None, extra: Sequence[str] = ()) -> dict[str, str]:
    root = root or REPO
    out = {}
    for rel in list(CODE_HASH_PATHS) + [str(x) for x in extra]:
        p = root / rel if not Path(rel).is_absolute() else Path(rel)
        key = rel if not Path(rel).is_absolute() else str(Path(rel).resolve().relative_to(root) if str(p).startswith(str(root)) else p)
        try:
            key = str(p.resolve().relative_to(root.resolve()))
        except ValueError:
            key = str(p)
        out[key] = file_sha256(p) if p.exists() else "missing"
    return out


def rows_fingerprint(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ids = [r["audit"]["example_id"] for r in rows]
    blob = json.dumps(
        [{"id": r["audit"]["example_id"], "q": r["audit"]["q_current_lex"]} for r in rows],
        sort_keys=True,
    )
    return {
        "n": len(ids),
        "example_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "targets_sha256": hashlib.sha256(blob.encode()).hexdigest(),
    }


def resolve_warmup_steps(cfg: PilotConfig) -> int:
    return int(cfg.warmup_steps)


def training_fingerprint(identity: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key in TRAINING_FINGERPRINT_KEYS:
        if key in identity:
            out[key] = identity[key]
    resolved = identity.get("resolved_config") or {}
    out.setdefault("mlp_hidden", resolved.get("mlp_hidden"))
    out.setdefault("loss", resolved.get("loss"))
    out.setdefault("optimizer_steps_per_run", resolved.get("optimizer_steps_per_run"))
    out.setdefault("warmup_steps", (identity.get("optimizer") or {}).get("warmup_steps"))
    out.setdefault("checkpoint_selection", resolved.get("checkpoint_selection"))
    return out


def identity_mismatch_fields(saved: Mapping[str, Any], live: Mapping[str, Any]) -> list[str]:
    a, b = training_fingerprint(saved), training_fingerprint(live)
    bad = []
    keys = sorted(set(a) | set(b))
    for key in keys:
        if a.get(key) != b.get(key):
            bad.append(key)
    return bad


def runtime_provenance(saved: Mapping[str, Any], live: Mapping[str, Any]) -> dict[str, Any] | None:
    diff = {}
    for key in MIGRATION_IDENTITY_KEYS:
        if saved.get(key) != live.get(key):
            diff[key] = {"from": saved.get(key), "to": live.get(key)}
    return diff or None


def reconcile_metrics(path: Path, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Drop log rows ahead of the committed checkpoint; fill last train/dev if missing."""
    step = int(state.get("step", 0))
    last_dev_epoch = int(state.get("last_dev_epoch", -1))
    kept: list[dict[str, Any]] = []
    if path.exists():
        for row in load_jsonl(path):
            if row.get("split") == "train":
                if int(row.get("step", -1)) <= step:
                    kept.append(row)
            elif row.get("split") == "dev":
                if int(row.get("step", -1)) <= step and int(row.get("epoch", -1)) <= last_dev_epoch:
                    kept.append(row)
            else:
                kept.append(row)
    seen_train = {int(r["step"]) for r in kept if r.get("split") == "train"}
    last_train = state.get("last_train")
    if last_train and int(last_train.get("step", -1)) == step and step not in seen_train:
        kept.append(dict(last_train, split="train"))
    seen_dev = {(int(r["step"]), int(r["epoch"])) for r in kept if r.get("split") == "dev"}
    last_dev = state.get("last_dev")
    if last_dev and int(last_dev.get("epoch", -99)) <= last_dev_epoch:
        key = (int(last_dev["step"]), int(last_dev["epoch"]))
        if key not in seen_dev:
            kept.append(dict(last_dev, split="dev"))
    kept.sort(key=lambda r: (0 if r.get("split") == "train" else 1, int(r.get("step", 0)), int(r.get("epoch", 0))))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in kept))
    return kept


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def join_records(model_rows: Sequence[Mapping[str, Any]], audit_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(model_rows) != len(audit_rows):
        raise ValueError("model/audit length mismatch")
    out = []
    seen = set()
    for model, audit in zip(model_rows, audit_rows):
        eid = audit["example_id"]
        if eid in seen:
            raise ValueError(f"duplicate example_id {eid}")
        seen.add(eid)
        if "example_id" in model and model["example_id"] != eid:
            raise ValueError(f"example_id mismatch {model['example_id']} vs {eid}")
        out.append({"model_input": dict(model), "audit": dict(audit)})
    return out


def rerender_v2(v1_dir: Path, v2_dir: Path) -> dict[str, Any]:
    v1_dir = Path(v1_dir)
    v2_dir = Path(v2_dir)
    v2_dir.mkdir(parents=True, exist_ok=True)
    parent = {name: file_sha256(v1_dir / name) for name in (
        "train_model.jsonl", "train_audit.jsonl", "dev_model.jsonl", "dev_audit.jsonl",
        "same_topology_model.jsonl", "same_topology_audit.jsonl",
        "structural_model.jsonl", "structural_audit.jsonl", "manifest.json",
    )}
    hashes = {}
    n = {}
    for split in ("train", "dev", "same_topology", "structural"):
        model_rows = load_jsonl(v1_dir / f"{split}_model.jsonl")
        audit_rows = load_jsonl(v1_dir / f"{split}_audit.jsonl")
        joined = join_records(model_rows, audit_rows)
        new_model = []
        new_audit = []
        for rec in joined:
            m, a = rec["model_input"], rec["audit"]
            suffix = [tuple(x) for x in m["suffix"]]
            prefix = [tuple(x) for x in a["prefix"]]
            part = canonicalize_partition(m["partition"])
            q_prev = a["q_previous_legacy"]
            q_cur = filter_observations(list(prefix) + list(suffix), part, m["hazard"])
            if any(abs(x - y) > 1e-12 for x, y in zip(q_cur, a["q_current_legacy"])):
                raise ValueError(f"target changed for {a['example_id']}")
            text = render_model_text(
                names=m["names"],
                partition=part,
                hazard=m["hazard"],
                q_previous=q_prev,
                suffix=suffix,
                n_prefix=int(m["n_prefix"]),
            )
            assert_clean_text(text)
            same = same_information_independent(q_prev, suffix, m["hazard"])
            model_out = {
                "example_id": a["example_id"],
                "text": text,
                "names": m["names"],
                "partition": m["partition"],
                "hazard": m["hazard"],
                "q_init": m["q_init"],
                "low": m["low"],
                "high": m["high"],
                "redraw": m["redraw"],
                "n_prefix": m["n_prefix"],
                "suffix": m["suffix"],
                "q_previous_lex": m["q_previous_lex"],
                "query_component": m["query_component"],
                "query_name": m["query_name"],
                "renderer_version": RENDERER_VERSION,
            }
            for key in ("family_id", "split", "topology_type", "prefix", "q_current_lex"):
                if key in model_out:
                    raise ValueError(key)
            audit_out = dict(a)
            audit_out["independent_fullprefix_lex"] = a["independent_lex"]
            audit_out["independent_sameinfo_lex"] = [float(x) for x in legacy_joint_to_lex(same)]
            audit_out["text_hash"] = sha256_text(text)
            audit_out["renderer_version"] = RENDERER_VERSION
            audit_out["parent_input_hash"] = a["input_hash"]
            new_model.append(model_out)
            new_audit.append(audit_out)
        write_jsonl(v2_dir / f"{split}_model.jsonl", new_model)
        write_jsonl(v2_dir / f"{split}_audit.jsonl", new_audit)
        hashes[f"{split}_model"] = sha256_text((v2_dir / f"{split}_model.jsonl").read_text())
        hashes[f"{split}_audit"] = sha256_text((v2_dir / f"{split}_audit.jsonl").read_text())
        n[split] = len(new_model)
    manifest = {
        "version": "learning_pilot_v2_render",
        "renderer_version": RENDERER_VERSION,
        "parent_v1_hashes": parent,
        "file_hashes": hashes,
        "n": n,
        "lex_keys": list(LEX_KEYS),
        "note": "Same sampled trajectories as 008. Text-only renderer correction.",
    }
    (v2_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    hashes["manifest"] = sha256_text((v2_dir / "manifest.json").read_text())
    return {"n": n, "file_hashes": hashes, "parent_v1_hashes": parent}


def experiment_identity(
    cfg: PilotConfig,
    *,
    mode: str,
    seed: int,
    data_dir: Path,
    train_rows: Sequence[Mapping[str, Any]] | None = None,
    dev_rows: Sequence[Mapping[str, Any]] | None = None,
    extra_code: Sequence[str] = (),
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    data_hashes: dict[str, Any] = {}
    files_present = all((data_dir / f"{split}_{kind}.jsonl").exists() for split in ("train", "dev") for kind in ("model", "audit"))
    if files_present and train_rows is None:
        data_hashes = {
            "source": "files",
            "train_model": file_sha256(data_dir / "train_model.jsonl"),
            "train_audit": file_sha256(data_dir / "train_audit.jsonl"),
            "dev_model": file_sha256(data_dir / "dev_model.jsonl"),
            "dev_audit": file_sha256(data_dir / "dev_audit.jsonl"),
        }
    if train_rows is not None:
        data_hashes["source"] = "in_memory_rows" if not files_present else data_hashes.get("source", "files")
        data_hashes["train_rows"] = rows_fingerprint(train_rows)
        data_hashes["dev_rows"] = rows_fingerprint(dev_rows or [])
        if files_present:
            data_hashes["train_model"] = file_sha256(data_dir / "train_model.jsonl")
            data_hashes["train_audit"] = file_sha256(data_dir / "train_audit.jsonl")
            data_hashes["dev_model"] = file_sha256(data_dir / "dev_model.jsonl")
            data_hashes["dev_audit"] = file_sha256(data_dir / "dev_audit.jsonl")
    elif not files_present:
        data_hashes = {"source": "fixture_missing_files"}
    warmup = resolve_warmup_steps(cfg)
    resolved = asdict(cfg)
    resolved["seeds"] = list(cfg.seeds)
    resolved["lora_targets"] = list(cfg.lora_targets)
    resolved["betas"] = list(cfg.betas)
    return {
        "kind": cfg.kind,
        "version": cfg.version,
        "mode": mode,
        "seed": seed,
        "head_seed": seed,
        "adapter_seed": seed + 10_007 if mode == "lora" else None,
        "mlp_seed": seed if mode == "mlp" else None,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "renderer_version": cfg.renderer_version,
        "numeric_encoder_version": cfg.numeric_encoder_version,
        "lex_keys": list(LEX_KEYS),
        "dtype": cfg.dtype,
        "effective_batch": cfg.effective_batch,
        "microbatch": cfg.microbatch,
        "epochs": cfg.epochs,
        "mlp_hidden": cfg.mlp_hidden,
        "loss": cfg.loss,
        "optimizer_steps_per_run": cfg.optimizer_steps_per_run,
        "warmup_steps": warmup,
        "checkpoint_selection": cfg.checkpoint_selection,
        "optimizer": {
            "name": cfg.optimizer,
            "betas": list(cfg.betas),
            "eps": cfg.eps,
            "weight_decay": cfg.weight_decay,
            "weight_decay_bias": cfg.weight_decay_bias,
            "head_lr": cfg.head_lr,
            "lora_lr": cfg.lora_lr,
            "mlp_lr": cfg.mlp_lr,
            "grad_clip": cfg.grad_clip,
            "warmup_ratio": cfg.warmup_ratio,
            "warmup_steps": warmup,
        },
        "lora": {
            "r": cfg.lora_r,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "targets": list(cfg.lora_targets),
        },
        "resolved_config": resolved,
        "data_hashes": data_hashes,
        "code_hashes": code_hashes(extra=extra_code),
        "heldout_read": False,
    }


def atomic_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    import torch
    torch.save(dict(payload), tmp)
    tmp.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def evaluate_split(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Sequence[float]],
    *,
    missing_policy: str = "error",
) -> dict[str, Any]:
    """Evaluate keyed predictions. Incomplete runs must not be dropped silently."""
    rows = []
    missing = []
    for rec in records:
        eid = rec["audit"]["example_id"]
        if eid not in predictions:
            missing.append(eid)
            continue
        scored = score_prediction(
            predictions[eid],
            rec["audit"]["q_current_legacy"],
            partition=rec["model_input"]["partition"],
            suffix=rec["model_input"]["suffix"],
            r_legacy=rec["audit"]["r_joint_legacy"],
        )
        scored["example_id"] = eid
        scored["family_id"] = rec["audit"]["family_id"]
        scored["hazard"] = rec["model_input"]["hazard"]
        scored["topology_type"] = rec["audit"]["topology_type"]
        rows.append(scored)
    if missing and missing_policy == "error":
        raise RuntimeError(f"missing predictions for {len(missing)} examples; incomplete run")
    return {"n": len(rows), "n_missing": len(missing), "rows": rows}


def summarize_eval(eval_rows: Sequence[Mapping[str, Any]], *, analysis_seed: int) -> dict[str, Any]:
    def mean_or_na(vals: list[float]) -> float | None:
        if not vals:
            return None
        return sum(vals) / len(vals)

    def subset(pred) -> list[Mapping[str, Any]]:
        return [r for r in eval_rows if pred(r)]

    def pack(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"n": 0, "joint_kl": None, "mean_abs_error": None, "mean_regret": None}
        return {
            "n": len(items),
            "joint_kl": mean_or_na([r["joint_kl"] for r in items]),
            "mean_abs_error": mean_or_na([r["mean_abs_error"] for r in items]),
            "mean_regret": mean_or_na([r["mean_regret"] for r in items]),
            "role_observed_abs": mean_or_na(
                [r["role_means"]["observed"]["abs_error"] for r in items if r["role_means"]["observed"] is not None]
            ),
            "role_related_abs": mean_or_na(
                [r["role_means"]["related_unobserved"]["abs_error"] for r in items if r["role_means"]["related_unobserved"] is not None]
            ),
            "role_unrelated_abs": mean_or_na(
                [r["role_means"]["unrelated"]["abs_error"] for r in items if r["role_means"]["unrelated"] is not None]
            ),
            "n_with_related": sum(r["role_means"]["related_unobserved"] is not None for r in items),
            "n_with_unrelated": sum(r["role_means"]["unrelated"] is not None for r in items),
        }

    actions = {}
    for r in eval_rows:
        for c in r["per_component"]:
            actions[c["optimal_action"]] = actions.get(c["optimal_action"], 0) + 1
    worst = sorted(eval_rows, key=lambda r: r["joint_kl"], reverse=True)[:5]
    fams = [r["family_id"] for r in eval_rows]
    boot = bootstrap_family_interval([r["joint_kl"] for r in eval_rows], fams, seed=analysis_seed) if eval_rows else None
    return {
        "overall": pack(eval_rows),
        "h0": pack(subset(lambda r: r["hazard"] == 0.0)),
        "h03": pack(subset(lambda r: r["hazard"] != 0.0)),
        "actions": actions,
        "worst_example_ids": [w["example_id"] for w in worst],
        "family_bootstrap_kl": boot,
    }


def bootstrap_family_interval(
    values: Sequence[float],
    families: Sequence[str],
    *,
    seed: int,
    n_boot: int = 1000,
) -> dict[str, float]:
    rng = random.Random(seed)
    by: dict[str, list[float]] = {}
    for fam, val in zip(families, values):
        by.setdefault(fam, []).append(val)
    keys = list(by)
    means = []
    for _ in range(n_boot):
        draw = [rng.choice(keys) for _ in keys]
        acc = []
        for k in draw:
            acc.extend(by[k])
        means.append(sum(acc) / len(acc))
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return {"mean": sum(values) / len(values), "lo": lo, "hi": hi, "n_families": len(keys)}


def split_file_hashes(data_dir: Path, split: str) -> dict[str, str]:
    data_dir = Path(data_dir)
    return {
        f"{split}_model": file_sha256(data_dir / f"{split}_model.jsonl"),
        f"{split}_audit": file_sha256(data_dir / f"{split}_audit.jsonl"),
    }


def write_eval_artifact(
    run_dir: Path,
    split: str,
    *,
    preds: Mapping[str, Sequence[float]],
    summary: Mapping[str, Any],
    n: int,
    checkpoint_identity: Mapping[str, Any],
    data_hashes: Mapping[str, Any],
    live_fingerprint: Mapping[str, Any],
    selected: Mapping[str, Any] | None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    pred_rows = [{"example_id": eid, "pred_lex": [float(x) for x in joint]} for eid, joint in sorted(preds.items())]
    write_jsonl(run_dir / f"eval_{split}_preds.jsonl", pred_rows)
    blob = {
        "n": n,
        "split": split,
        "summary": summary,
        "selected_checkpoint": selected,
        "checkpoint_identity_fingerprint": training_fingerprint(checkpoint_identity),
        "live_fingerprint": live_fingerprint,
        "data_hashes": dict(data_hashes),
    }
    (run_dir / f"eval_{split}.json").write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    return {
        "summary": run_dir / f"eval_{split}.json",
        "preds": run_dir / f"eval_{split}_preds.jsonl",
    }


def load_pred_lex(path: Path) -> dict[str, list[float]]:
    out = {}
    for row in load_jsonl(path):
        out[row["example_id"]] = [float(x) for x in row["pred_lex"]]
    return out


def reference_pred_lex(rec: Mapping[str, Any], name: str) -> list[float]:
    audit = rec["audit"]
    if name == "exact":
        return list(audit["q_current_lex"])
    if name == "stale":
        return list(audit["stale_lex"])
    if name == "transition_only":
        return list(audit["r_lex"])
    if name == "independent_sameinfo":
        return list(audit["independent_sameinfo_lex"])
    raise KeyError(name)
