from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from llm_rl_engine.analysis.learning_pilot import (
    evaluate_split,
    experiment_identity,
    load_config,
    load_pred_lex,
    rows_fingerprint,
    summarize_eval,
)
from llm_rl_engine.analysis.posterior_baseline import build_example
from llm_rl_engine.analysis.paired_revision import all_partitions_of_type

REPO = Path(__file__).resolve().parents[1]
CFG010 = REPO / "llm_rl_engine/configs/numeric_fit_diagnosis.yaml"
CFG009 = REPO / "llm_rl_engine/configs/learning_pilot.yaml"


def _rows(n: int = 4):
    recs = []
    part = all_partitions_of_type("four_singletons")[0]
    for i in range(n):
        rec = build_example(
            example_id=f"fix{i}",
            split="train",
            names=("A", "B", "C", "D"),
            partition=part,
            hazard=0.0,
            prefix=((i % 4, 1), (0, 1), (1, 0), (2, 1)),
            suffix=((3, i % 2),),
        )
        recs.append({"model_input": rec["model_input"], "audit": rec["audit"]})
    return recs


def _tiny_cfg(**over):
    merged = {"effective_batch": 2, "microbatch": 1, "optimizer_steps_per_run": 2, "epochs": 1, "warmup_steps": 1}
    merged.update(over)
    return load_config(CFG009, overrides=merged)


def test_010_config_warmup_is_19_not_192():
    cfg = load_config(CFG010)
    assert cfg.warmup_steps == 19
    assert cfg.optimizer_steps_per_run == 1920
    assert cfg.epochs == 30
    assert cfg.heldout_eval == "never"
    assert int(cfg.warmup_ratio * cfg.optimizer_steps_per_run) == 192
    torch = pytest.importorskip("torch")
    from llm_rl_engine.analysis.learning_pilot_train import make_scheduler

    p = torch.nn.Parameter(torch.ones(1), requires_grad=True)
    opt = torch.optim.AdamW([p], lr=cfg.mlp_lr)
    sched = make_scheduler(opt, cfg)
    lrs = []
    for _ in range(192):
        opt.zero_grad()
        p.sum().backward()
        opt.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    # LambdaLR steps once in __init__, matching the 009 logged first LR of base*2/19.
    assert lrs[0] == pytest.approx(cfg.mlp_lr * 2 / 19)
    assert lrs[17] == pytest.approx(cfg.mlp_lr)
    progress = (192 - 19) / (1920 - 19)
    assert lrs[191] == pytest.approx(cfg.mlp_lr * max(0.0, 1.0 - progress), rel=1e-6)
    assert lrs[191] > 0.8 * cfg.mlp_lr


def test_fixture_identity_is_not_full_dataset_hash(tmp_path):
    cfg = _tiny_cfg()
    rows = _rows(4)
    ident = experiment_identity(cfg, mode="mlp", seed=0, data_dir=tmp_path, train_rows=rows, dev_rows=rows[:2])
    assert ident["data_hashes"]["source"] == "in_memory_rows"
    assert ident["data_hashes"]["train_rows"] == rows_fingerprint(rows)
    v2 = REPO / "reports/learning_pilot/data_v2/train_model.jsonl"
    if v2.exists():
        from llm_rl_engine.analysis.learning_pilot import file_sha256
        assert ident["data_hashes"]["train_rows"]["targets_sha256"] != file_sha256(v2)
    assert any(k.endswith("learning_pilot_train.py") for k in ident["code_hashes"])


def test_refuse_non_resume_existing_dir(tmp_path):
    torch = pytest.importorskip("torch")
    from llm_rl_engine.analysis.learning_pilot_train import NumericMLP, fit_run

    cfg = _tiny_cfg()
    rows = _rows(4)
    device = torch.device("cpu")
    out = tmp_path / "run"
    torch.manual_seed(0)
    fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=out, device=device,
        max_steps=1, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
    )
    with pytest.raises(RuntimeError, match="non-resume"):
        fit_run(
            cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=out, device=device,
            max_steps=1, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
        )


def test_reject_changed_lr_batch_and_code(tmp_path):
    torch = pytest.importorskip("torch")
    from llm_rl_engine.analysis.learning_pilot_train import NumericMLP, fit_run

    cfg = _tiny_cfg()
    rows = _rows(4)
    device = torch.device("cpu")
    out = tmp_path / "run"
    torch.manual_seed(0)
    fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=out, device=device,
        max_steps=1, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
    )
    cfg_lr = _tiny_cfg(mlp_lr=1e-4)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        fit_run(
            cfg_lr, mode="mlp", seed=0, data_dir=tmp_path, out_dir=out, device=device,
            resume=True, max_steps=2, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
        )
    cfg_bs = _tiny_cfg(effective_batch=1)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        fit_run(
            cfg_bs, mode="mlp", seed=0, data_dir=tmp_path, out_dir=out, device=device,
            resume=True, max_steps=2, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
        )
    payload = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    payload["identity"]["code_hashes"] = dict(payload["identity"]["code_hashes"], fake="0" * 64)
    torch.save(payload, out / "last.pt")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        fit_run(
            cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=out, device=device,
            resume=True, max_steps=2, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
        )


def test_uninterrupted_vs_resume_params_scheduler_batches(tmp_path):
    torch = pytest.importorskip("torch")
    from llm_rl_engine.analysis.learning_pilot_train import NumericMLP, fit_run

    cfg = _tiny_cfg()
    rows = _rows(4)
    device = torch.device("cpu")
    torch.manual_seed(0)
    cont = fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=tmp_path / "cont", device=device,
        max_steps=2, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
    )
    torch.manual_seed(0)
    fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=tmp_path / "step1", device=device,
        max_steps=1, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
    )
    shutil.copytree(tmp_path / "step1", tmp_path / "resume")
    resumed = fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=tmp_path / "resume", device=device,
        resume=True, max_steps=2, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
    )
    assert resumed["state"]["step"] == 2
    assert resumed["state"]["optimizer_updates"] == 2
    a = torch.load(tmp_path / "cont" / "last.pt", map_location="cpu", weights_only=False)
    b = torch.load(tmp_path / "resume" / "last.pt", map_location="cpu", weights_only=False)
    for n in a["trainable"]:
        assert torch.equal(a["trainable"][n], b["trainable"][n])
    assert a["scheduler"]["last_epoch"] == b["scheduler"]["last_epoch"]
    mt = [json.loads(l) for l in (tmp_path / "cont" / "metrics.jsonl").read_text().splitlines() if '"split": "train"' in l]
    mr = [json.loads(l) for l in (tmp_path / "resume" / "metrics.jsonl").read_text().splitlines() if '"split": "train"' in l]
    assert mt[0]["batch_ids"] != mt[1]["batch_ids"]
    assert mt[1]["batch_ids"] == mr[1]["batch_ids"]
    assert mt[1]["lr"] == pytest.approx(mr[1]["lr"])
    assert cont["state"]["best"]["step"] == resumed["state"]["best"]["step"]


def test_epoch_boundary_before_after_and_final_step(tmp_path):
    torch = pytest.importorskip("torch")
    from llm_rl_engine.analysis.learning_pilot_train import NumericMLP, fit_run

    cfg = _tiny_cfg()
    rows = _rows(4)
    device = torch.device("cpu")
    torch.manual_seed(0)
    fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=tmp_path / "full", device=device,
        max_steps=2, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
    )
    full_dev = [json.loads(l) for l in (tmp_path / "full" / "metrics.jsonl").read_text().splitlines() if '"split": "dev"' in l]
    assert len(full_dev) == 1

    shutil.copytree(tmp_path / "full", tmp_path / "after")
    after = fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=tmp_path / "after", device=device,
        resume=True, max_steps=2, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
    )
    assert after["state"]["optimizer_updates"] == 2
    after_dev = [json.loads(l) for l in (tmp_path / "after" / "metrics.jsonl").read_text().splitlines() if '"split": "dev"' in l]
    assert len(after_dev) == 1

    shutil.copytree(tmp_path / "full", tmp_path / "before")
    payload = torch.load(tmp_path / "before" / "last.pt", map_location="cpu", weights_only=False)
    st = payload["train_state"]
    st["epoch"] = 0
    st["cursor"] = 4
    st["last_dev_epoch"] = -1
    st["last_dev"] = None
    st["best"] = {"dev_kl": float("inf"), "step": -1, "epoch": -1}
    payload["train_state"] = st
    torch.save(payload, tmp_path / "before" / "last.pt")
    metrics = [json.loads(l) for l in (tmp_path / "before" / "metrics.jsonl").read_text().splitlines() if l.strip()]
    (tmp_path / "before" / "metrics.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in metrics if r.get("split") == "train")
    )
    before = fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=tmp_path / "before", device=device,
        resume=True, max_steps=2, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
    )
    assert before["state"]["optimizer_updates"] == 2
    assert before["state"]["step"] == 2
    before_dev = [json.loads(l) for l in (tmp_path / "before" / "metrics.jsonl").read_text().splitlines() if '"split": "dev"' in l]
    assert len(before_dev) == 1
    assert before["state"]["last_dev_epoch"] == 0
    assert before["state"]["epoch"] == 1


def test_reconcile_drops_log_ahead_of_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from llm_rl_engine.analysis.learning_pilot_train import NumericMLP, fit_run

    cfg = _tiny_cfg()
    rows = _rows(4)
    device = torch.device("cpu")
    torch.manual_seed(0)
    fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=tmp_path / "run", device=device,
        max_steps=1, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
    )
    with (tmp_path / "run" / "metrics.jsonl").open("a") as f:
        f.write(json.dumps({"split": "train", "step": 99, "epoch": 0, "kl": 0.0, "lr": 0.0, "batch_ids": ["x"]}) + "\n")
    fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=tmp_path / "run", device=device,
        resume=True, max_steps=2, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
    )
    steps = [json.loads(l)["step"] for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines() if '"split": "train"' in l]
    assert 99 not in steps
    assert steps == [1, 2]


def test_eval_joints_recompute_summary(tmp_path):
    torch = pytest.importorskip("torch")
    from llm_rl_engine.analysis.learning_pilot import write_eval_artifact, write_jsonl
    from llm_rl_engine.analysis.learning_pilot_train import NumericMLP, fit_run, predict_lex

    cfg = _tiny_cfg()
    rows = _rows(4)
    device = torch.device("cpu")
    torch.manual_seed(0)
    out = fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=tmp_path / "run", device=device,
        max_steps=2, module=NumericMLP().to(device), train_rows=rows, dev_rows=rows[:2],
    )
    module = NumericMLP().to(device)
    from llm_rl_engine.analysis.learning_pilot_train import load_run_checkpoint
    load_run_checkpoint(tmp_path / "run" / "best.pt", module, out["identity"], strict_identity=True)
    preds = {r["audit"]["example_id"]: predict_lex(module, r, None, device, cfg, "mlp") for r in rows[:2]}
    scored = evaluate_split(rows[:2], preds)
    summary = summarize_eval(scored["rows"], analysis_seed=1)
    write_eval_artifact(
        tmp_path / "run",
        "dev",
        preds=preds,
        summary=summary,
        n=scored["n"],
        checkpoint_identity=out["identity"],
        data_hashes=out["identity"]["data_hashes"],
        live_fingerprint=out["identity"],
        selected=out["state"]["best"],
    )
    write_jsonl(tmp_path / "run" / "eval_dev_rows.jsonl", scored["rows"])
    loaded = load_pred_lex(tmp_path / "run" / "eval_dev_preds.jsonl")
    again = evaluate_split(rows[:2], loaded)
    again_sum = summarize_eval(again["rows"], analysis_seed=1)
    assert again_sum["overall"]["joint_kl"] == pytest.approx(summary["overall"]["joint_kl"], abs=1e-12)
    assert again_sum["overall"]["mean_regret"] == pytest.approx(summary["overall"]["mean_regret"], abs=1e-12)


def test_precompute_matches_encoder():
    torch = pytest.importorskip("torch")
    from llm_rl_engine.analysis.learning_pilot_train import precompute_mlp_features, row_numeric_features

    rows = _rows(3)
    feats = precompute_mlp_features(rows)
    assert feats[1] == row_numeric_features(rows[1])
    assert len(feats[0]) == 47
