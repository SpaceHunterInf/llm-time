from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from llm_rl_engine.analysis.learning_pilot import (
    ALLOWED_CONFIG_KEYS,
    evaluate_split,
    load_config,
    summarize_eval,
)
from llm_rl_engine.analysis.paired_revision import (
    DEFAULT_H,
    all_partitions_of_type,
    canonicalize_partition,
    filter_observations,
    permute_joint,
    permute_observations,
    permute_partition,
)
from llm_rl_engine.analysis.posterior_baseline import (
    RENDERER_VERSION,
    TIMING_SENTENCE,
    build_example,
    kl_div,
    legacy_joint_to_lex,
    numeric_encode,
    permute_numeric_features,
    render_model_text,
    same_information_independent,
    score_prediction,
)

REPO = Path(__file__).resolve().parents[1]


def _example(related: bool = True, h: float = DEFAULT_H):
    part = canonicalize_partition([[0, 1], [2], [3]]) if related else all_partitions_of_type("four_singletons")[0]
    prefix = ((0, 1), (0, 1), (0, 0), (0, 1))
    suffix = ((0, 1), (0, 0))
    return build_example(
        example_id="unit-rel" if related else "unit-ind",
        split="train",
        names=("A", "B", "C", "D"),
        partition=part,
        hazard=h,
        prefix=prefix,
        suffix=suffix,
    )


def test_config_rejects_unknown_and_loads_yaml():
    cfg = load_config(REPO / "llm_rl_engine/configs/learning_pilot.yaml")
    assert cfg.effective_batch == 32
    assert cfg.microbatch == 8
    assert cfg.dtype == "float32"
    with pytest.raises(ValueError, match="unknown"):
        load_config(REPO / "llm_rl_engine/configs/learning_pilot.yaml", overrides={"not_a_field": 1})
    assert "microbatch" in ALLOWED_CONFIG_KEYS


def test_renderer_v2_states_reset_before_observation():
    rec = _example()
    text = rec["model_input"]["text"]
    assert TIMING_SENTENCE in text
    assert "No extra transition" in text
    assert "independently of other groups" in text
    assert RENDERER_VERSION == "v2"


def test_same_information_independent_does_not_replay_prefix():
    rec = _example()
    suffix = [tuple(x) for x in rec["model_input"]["suffix"]]
    same = same_information_independent(rec["audit"]["q_previous_legacy"], suffix, rec["model_input"]["hazard"])
    full = rec["audit"]["independent_lex"]
    # different information interface: not required to match the 008 prefix replay
    assert len(same) == 16
    assert math.fsum(same) == pytest.approx(1.0)
    assert same != pytest.approx(full, abs=1e-8) or rec["audit"]["topology_type"] == "four_singletons"


def test_numeric_encode_is_47d_and_permutation_equivariant():
    rec = _example()
    feat = numeric_encode(
        q_previous_lex=rec["model_input"]["q_previous_lex"],
        partition=rec["model_input"]["partition"],
        hazard=rec["model_input"]["hazard"],
        suffix=rec["model_input"]["suffix"],
    )
    assert len(feat) == 47
    perm = (2, 0, 3, 1)
    ppart = permute_partition(rec["model_input"]["partition"], perm)
    pprev = permute_joint(rec["audit"]["q_previous_legacy"], perm)
    psuf = permute_observations([tuple(x) for x in rec["model_input"]["suffix"]], perm)
    feat2 = numeric_encode(
        q_previous_lex=legacy_joint_to_lex(pprev),
        partition=ppart,
        hazard=rec["model_input"]["hazard"],
        suffix=psuf,
    )
    assert feat2 == pytest.approx(permute_numeric_features(feat, perm), abs=1e-12)
    dumped = json.dumps(feat)
    assert "train" not in dumped
    assert "prefix" not in dumped


def test_exact_prediction_zero_error_and_all_four_components():
    rec = _example()
    scored = score_prediction(
        rec["audit"]["q_current_lex"],
        rec["audit"]["q_current_legacy"],
        partition=rec["model_input"]["partition"],
        suffix=rec["model_input"]["suffix"],
        r_legacy=rec["audit"]["r_joint_legacy"],
    )
    assert scored["joint_kl"] == pytest.approx(0.0, abs=1e-12)
    assert scored["mean_regret"] == pytest.approx(0.0, abs=1e-12)
    assert len(scored["per_component"]) == 4
    assert scored["roles"]["related_unobserved"] == [1]
    stale = score_prediction(
        rec["audit"]["stale_lex"],
        rec["audit"]["q_current_legacy"],
        partition=rec["model_input"]["partition"],
        suffix=rec["model_input"]["suffix"],
        r_legacy=rec["audit"]["r_joint_legacy"],
    )
    expected = kl_div(rec["audit"]["q_current_legacy"], rec["audit"]["q_previous_legacy"])
    assert stale["joint_kl"] == pytest.approx(expected, abs=1e-12)
    assert stale["joint_kl"] > 0


def test_h0_unrelated_present_and_empty_role_is_none():
    rec = _example(related=True, h=0.0)
    scored = score_prediction(
        rec["audit"]["q_current_lex"],
        rec["audit"]["q_current_legacy"],
        partition=rec["model_input"]["partition"],
        suffix=rec["model_input"]["suffix"],
        r_legacy=rec["audit"]["r_joint_legacy"],
    )
    assert scored["role_means"]["unrelated"] is not None
    rec2 = _example(related=False, h=0.0)
    scored2 = score_prediction(
        rec2["audit"]["q_current_lex"],
        rec2["audit"]["q_current_legacy"],
        partition=rec2["model_input"]["partition"],
        suffix=rec2["model_input"]["suffix"],
        r_legacy=rec2["audit"]["r_joint_legacy"],
    )
    assert scored2["role_means"]["related_unobserved"] is None


def test_evaluate_split_errors_on_incomplete_run():
    rec = _example()
    packed = {"model_input": rec["model_input"], "audit": rec["audit"]}
    with pytest.raises(RuntimeError, match="incomplete"):
        evaluate_split([packed], {})
    out = evaluate_split([packed], {rec["audit"]["example_id"]: rec["audit"]["q_current_lex"]})
    summ = summarize_eval(out["rows"], analysis_seed=1)
    assert summ["overall"]["n"] == 1
    assert summ["h0"]["n"] == 0
    assert summ["h0"]["joint_kl"] is None


def test_stub_fit_resume_distinct_batches():
    torch = pytest.importorskip("torch")
    from llm_rl_engine.analysis.learning_pilot import load_config
    from llm_rl_engine.analysis.learning_pilot_train import NumericMLP, fit_run, load_run_checkpoint

    cfg = load_config(
        REPO / "llm_rl_engine/configs/learning_pilot.yaml",
        overrides={"effective_batch": 2, "microbatch": 1, "optimizer_steps_per_run": 2, "epochs": 1},
    )
    recs = []
    for i in range(4):
        rec = build_example(
            example_id=f"fix{i}",
            split="train",
            names=("A", "B", "C", "D"),
            partition=all_partitions_of_type("four_singletons")[0],
            hazard=0.0,
            prefix=((i % 4, 1), (0, 1), (1, 0), (2, 1)),
            suffix=((3, i % 2),),
        )
        recs.append({"model_input": rec["model_input"], "audit": rec["audit"]})
    dev = recs[:2]
    tmp = REPO / "reports" / "learning_pilot" / "scratch" / "stub_resume"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    device = torch.device("cpu")
    torch.manual_seed(0)
    r1 = fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp, out_dir=tmp / "cont", device=device,
        max_steps=2, module=NumericMLP(cfg.mlp_hidden).to(device), train_rows=recs, dev_rows=dev,
        holdout_guard=True,
    )
    metrics = [json.loads(l) for l in (tmp / "cont" / "metrics.jsonl").read_text().splitlines() if l.strip()]
    train_rows = [m for m in metrics if m.get("split") == "train"]
    assert train_rows[0]["batch_ids"] != train_rows[1]["batch_ids"]
    r2dir = tmp / "resume"
    import shutil
    shutil.copytree(tmp / "cont", r2dir)
    # rewind to step 1 checkpoint by running 1 step then resume 2
    shutil.rmtree(r2dir)
    torch.manual_seed(0)
    fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp, out_dir=tmp / "step1", device=device,
        max_steps=1, module=NumericMLP(cfg.mlp_hidden).to(device), train_rows=recs, dev_rows=dev,
    )
    shutil.copytree(tmp / "step1", tmp / "resume")
    torch.manual_seed(99)
    resumed = fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp, out_dir=tmp / "resume", device=device,
        resume=True, max_steps=2, module=NumericMLP(cfg.mlp_hidden).to(device), train_rows=recs, dev_rows=dev,
    )
    assert resumed["state"]["step"] == 2
    assert resumed["state"]["optimizer_updates"] == 2
    cont_metrics = [json.loads(l) for l in (tmp / "cont" / "metrics.jsonl").read_text().splitlines() if '{"split": "train"' in l or '"split": "train"' in l]
    res_metrics = [json.loads(l) for l in (tmp / "resume" / "metrics.jsonl").read_text().splitlines() if '"split": "train"' in l]
    assert cont_metrics[1]["batch_ids"] == res_metrics[1]["batch_ids"]
    assert cont_metrics[1]["lr"] == pytest.approx(res_metrics[1]["lr"])


def test_v2_render_preserves_targets_and_ids():
    data = REPO / "reports" / "learning_pilot" / "data_v2"
    if not (data / "manifest.json").exists():
        pytest.skip("v2 render not generated")
    from llm_rl_engine.analysis.learning_pilot import join_records, load_jsonl
    from llm_rl_engine.analysis.posterior_baseline import TIMING_SENTENCE, sha256_text

    man = json.loads((data / "manifest.json").read_text())
    assert man["n"]["train"] == 2048
    v1 = REPO / "reports" / "posterior_baseline_preflight" / "data"
    for split in ("train", "dev", "same_topology", "structural"):
        v2 = join_records(load_jsonl(data / f"{split}_model.jsonl"), load_jsonl(data / f"{split}_audit.jsonl"))
        a1 = [json.loads(l) for l in (v1 / f"{split}_audit.jsonl").read_text().splitlines() if l.strip()]
        assert len(v2) == len(a1)
        for rec, old in zip(v2, a1):
            assert rec["audit"]["example_id"] == old["example_id"]
            assert rec["audit"]["q_current_legacy"] == pytest.approx(old["q_current_legacy"], abs=1e-12)
            assert rec["model_input"]["example_id"] == rec["audit"]["example_id"]
            assert TIMING_SENTENCE in rec["model_input"]["text"]
            assert "prefix" not in rec["model_input"]
            assert rec["audit"]["family_id"] == old["family_id"]
        assert sha256_text((data / f"{split}_model.jsonl").read_text()) == man["file_hashes"][f"{split}_model"]


def test_identity_mismatch_refuses_to_update(tmp_path):
    torch = pytest.importorskip("torch")
    from llm_rl_engine.analysis.learning_pilot import load_config
    from llm_rl_engine.analysis.learning_pilot_train import NumericMLP, fit_run

    cfg = load_config(
        REPO / "llm_rl_engine/configs/learning_pilot.yaml",
        overrides={"effective_batch": 2, "microbatch": 1, "optimizer_steps_per_run": 1},
    )
    rec = _example()
    rows = [{"model_input": rec["model_input"], "audit": rec["audit"]}] * 2
    rows[1]["audit"] = dict(rows[1]["audit"], example_id="other")
    out = tmp_path / "run"
    fit_run(
        cfg, mode="mlp", seed=0, data_dir=tmp_path, out_dir=out, device=torch.device("cpu"),
        max_steps=1, module=NumericMLP().to("cpu"), train_rows=rows, dev_rows=rows,
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        fit_run(
            cfg, mode="mlp", seed=1, data_dir=tmp_path, out_dir=out, device=torch.device("cpu"),
            resume=True, max_steps=1, module=NumericMLP().to("cpu"), train_rows=rows, dev_rows=rows,
        )
