from __future__ import annotations

import json
import math
from itertools import permutations
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from llm_rl_engine.analysis.paired_revision import (
    DEFAULT_H,
    all_partitions_of_type,
    canonicalize_partition,
    filter_observations,
    permute_joint,
    permute_observations,
    permute_partition,
    transition_matrix,
)
from llm_rl_engine.analysis.posterior_baseline import (
    FORBIDDEN_MODEL_MARKERS,
    LEX_KEYS,
    assert_clean_text,
    build_example,
    family_identity,
    generate_dataset,
    legacy_joint_to_lex,
    lex_joint_to_legacy,
    lex_key_to_legacy_index,
    product_group_kernel_matrix,
    public_model_row,
    render_model_text,
)


def test_lex_roundtrip_and_order():
    assert LEX_KEYS[0] == "0000"
    assert LEX_KEYS[1] == "0001"
    assert LEX_KEYS[-1] == "1111"
    assert lex_key_to_legacy_index("1000") == 1
    assert lex_key_to_legacy_index("0001") == 8
    joint = [float(i) for i in range(16)]
    lex = legacy_joint_to_lex(joint)
    assert lex_joint_to_legacy(lex) == pytest.approx(joint)
    assert lex[0] == joint[0]
    assert lex[1] == joint[8]


def test_product_kernel_independent_of_redraw_helper():
    for kind in ("four_singletons", "one_pair", "two_pairs", "triple"):
        for part in all_partitions_of_type(kind):
            for h in (0.0, DEFAULT_H, 0.4, 1.0):
                prod = product_group_kernel_matrix(part, h)
                built = transition_matrix(part, h)
                for i in range(16):
                    assert prod[i] == pytest.approx(built[i], abs=1e-12)
                    assert math.fsum(prod[i]) == pytest.approx(1.0)


def test_family_identity_renaming_invariant_and_order_sensitive():
    prefix = ((0, 1), (1, 0), (0, 1))
    suffix = ((2, 1),)
    a = family_identity(prefix, suffix)
    b = family_identity(((3, 1), (0, 0), (3, 1)), ((1, 1),))
    assert a == b
    c = family_identity(((1, 0), (0, 1), (0, 1)), ((2, 1),))
    assert c != a
    d = family_identity(prefix, suffix)
    assert d == a


def test_renderer_omits_prefix_and_targets():
    part = canonicalize_partition([[0, 1], [2], [3]])
    prefix = ((0, 1), (0, 1), (1, 0), (2, 1))
    suffix = ((0, 0), (3, 1))
    rec = build_example(
        example_id="unit",
        split="train",
        names=("A", "B", "C", "D"),
        partition=part,
        hazard=DEFAULT_H,
        prefix=prefix,
        suffix=suffix,
    )
    text = rec["model_input"]["text"]
    assert_clean_text(text)
    assert "Subsequent observations only" in text
    assert "t=5:" in text
    assert "t=1:" not in text
    assert "q_current" not in text
    assert "topology" not in text.lower()
    assert "train" not in text
    pub = public_model_row(rec)
    assert "prefix" not in pub
    assert "q_current_lex" not in pub
    assert "split" not in pub
    other = dict(rec)
    other["audit"] = dict(rec["audit"], split="dev", family_id="other", topology_type="triple")
    assert public_model_row(other)["text"] == text
    replay = rec["audit"]["replay_legacy"]
    assert replay == pytest.approx(rec["audit"]["q_current_legacy"], abs=1e-12)


def test_metadata_does_not_change_model_text():
    part = all_partitions_of_type("four_singletons")[0]
    prefix = ((0, 1),) * 4
    suffix = ((1, 0),)
    a = build_example(
        example_id="a", split="train", names=("A", "B", "C", "D"),
        partition=part, hazard=0.0, prefix=prefix, suffix=suffix,
    )
    b = build_example(
        example_id="b", split="structural", names=("A", "B", "C", "D"),
        partition=part, hazard=0.0, prefix=prefix, suffix=suffix,
    )
    assert a["model_input"]["text"] == b["model_input"]["text"]
    assert a["audit"]["family_id"] == b["audit"]["family_id"]


def test_semantic_permutation_matches_lex_target():
    part = canonicalize_partition([[0, 1], [2], [3]])
    obs_prefix = ((0, 1), (2, 0), (0, 1), (3, 1))
    obs_suffix = ((1, 1), (0, 0))
    rec = build_example(
        example_id="p0", split="train", names=("A", "B", "C", "D"),
        partition=part, hazard=DEFAULT_H, prefix=obs_prefix, suffix=obs_suffix,
    )
    perm = (2, 0, 3, 1)
    ppart = permute_partition(part, perm)
    ppre = tuple(permute_observations(obs_prefix, perm))
    psuf = tuple(permute_observations(obs_suffix, perm))
    rec2 = build_example(
        example_id="p1", split="train", names=("A", "B", "C", "D"),
        partition=ppart, hazard=DEFAULT_H, prefix=ppre, suffix=psuf,
    )
    expected_legacy = permute_joint(rec["audit"]["q_current_legacy"], perm)
    assert rec2["audit"]["q_current_legacy"] == pytest.approx(expected_legacy, abs=1e-12)
    assert rec2["audit"]["q_current_lex"] == pytest.approx(legacy_joint_to_lex(expected_legacy), abs=1e-12)


def test_independent_unobserved_zero_evidence_update():
    part = all_partitions_of_type("four_singletons")[0]
    prefix = ((0, 1), (0, 0), (0, 1), (0, 1))
    suffix = ((0, 1),)
    rec = build_example(
        example_id="z", split="train", names=("A", "B", "C", "D"),
        partition=part, hazard=DEFAULT_H, prefix=prefix, suffix=suffix,
    )
    q = rec["audit"]["q_current_legacy"]
    r = rec["audit"]["r_joint_legacy"]
    from llm_rl_engine.analysis.paired_revision import marginals

    d = [marginals(q)[k] - marginals(r)[k] for k in range(4)]
    assert d[0] != pytest.approx(0.0, abs=1e-9)
    for k in (1, 2, 3):
        assert d[k] == pytest.approx(0.0, abs=1e-12)


def test_tiny_generation_split_and_family_lock(monkeypatch):
    import llm_rl_engine.analysis.posterior_baseline as pb

    monkeypatch.setattr(pb, "N_TRAIN", 8)
    monkeypatch.setattr(pb, "N_DEV", 4)
    monkeypatch.setattr(pb, "N_SAME", 4)
    monkeypatch.setattr(pb, "N_STRUCT", 4)
    data = generate_dataset(seed=1)
    assert data["n"]["train"] == 8
    assert data["coverage"]["family_split_collisions"] == 0
    owners = {}
    for split, rows in data["splits"].items():
        for rec in rows:
            fam = rec["audit"]["family_id"]
            if fam in owners:
                assert owners[fam] == split
            owners[fam] = split
            assert_clean_text(rec["model_input"]["text"])
            for marker in FORBIDDEN_MODEL_MARKERS:
                assert marker.lower() not in rec["model_input"]["text"].lower()
    struct_types = {r["audit"]["topology_type"] for r in data["splits"]["structural"]}
    assert struct_types <= {"two_pairs", "triple"}
    train_types = {r["audit"]["topology_type"] for r in data["splits"]["train"]}
    assert struct_types.isdisjoint(train_types)


def test_generated_pilot_files_if_present():
    data = REPO / "reports" / "posterior_baseline_preflight" / "data"
    man_path = data / "manifest.json"
    if not man_path.exists():
        pytest.skip("pilot dataset not generated")
    man = json.loads(man_path.read_text())
    assert man["n"]["train"] == 2048
    assert man["coverage"]["family_split_collisions"] == 0
    from llm_rl_engine.analysis.posterior_baseline import sha256_text

    for split in ("train", "dev", "same_topology", "structural"):
        model_path = data / f"{split}_model.jsonl"
        assert sha256_text(model_path.read_text()) == man["file_hashes"][f"{split}_model"]
        line = model_path.read_text().splitlines()[0]
        row = json.loads(line)
        assert "prefix" not in row
        assert "q_current_lex" not in row
        assert "family_id" not in row
        assert "split" not in row
        assert_clean_text(row["text"])
    actions = man["coverage"]["train"]["actions"]
    assert set(actions) >= {"choose_linked", "choose_safe", "diagnose_linked"}
