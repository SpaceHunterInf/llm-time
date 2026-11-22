from __future__ import annotations

import math
from itertools import permutations

import pytest

from llm_rl_engine.analysis.paired_revision import (
    DEFAULT_H,
    N_COMPONENTS,
    N_STATES,
    algebra_numeric_checks,
    all_partitions_of_type,
    brute_transition_row,
    canonicalize_partition,
    evaluate_probes,
    evidence_families,
    filter_observations,
    generate_pairs,
    grad_kl_softmax,
    joint_keys,
    kl_div,
    l_pair_from_marginals,
    leakage_keys,
    marginals,
    objectives_for_pair,
    permute_joint,
    permute_observations,
    permute_partition,
    prior_joint,
    render_partition,
    sequence_log_likelihood,
    showcase_pair,
    softmax,
    solver_input,
    topology_type,
    transition_matrix,
    transition_only,
    unobserved_related_gap,
)


def test_h0_identity_and_h1_uniform():
    part = all_partitions_of_type("one_pair")[0]
    t0 = transition_matrix(part, 0.0)
    for i in range(N_STATES):
        assert t0[i][i] == pytest.approx(1.0)
        assert math.fsum(t0[i]) == pytest.approx(1.0)
    t1 = transition_matrix(part, 1.0)
    for i in range(N_STATES):
        assert math.fsum(t1[i]) == pytest.approx(1.0)
        for j in range(N_STATES):
            assert t1[i][j] == pytest.approx(1.0 / N_STATES)


def test_rows_normalize_for_all_topology_types():
    for kind in ("four_singletons", "one_pair", "two_pairs", "triple"):
        for part in all_partitions_of_type(kind):
            for h in (0.0, DEFAULT_H, 1.0):
                t = transition_matrix(part, h)
                for row in t:
                    assert math.fsum(row) == pytest.approx(1.0)
                    assert all(x >= 0 for x in row)


def test_brute_enumeration_matches_matrix():
    part = canonicalize_partition([[0, 1], [2], [3]])
    t = transition_matrix(part, DEFAULT_H)
    for src in range(N_STATES):
        brute = brute_transition_row(src, part, DEFAULT_H)
        assert brute == pytest.approx(t[src], abs=1e-12)


def test_prefix_suffix_equals_full_replay():
    part = canonicalize_partition([[0, 1], [2], [3]])
    prefix = ((0, 1), (1, 0), (2, 1), (3, 1))
    suffix = ((0, 0), (0, 1), (1, 1))
    full = filter_observations(prefix + suffix, part, DEFAULT_H)
    prev = filter_observations(prefix, part, DEFAULT_H)
    cont = filter_observations(suffix, part, DEFAULT_H, start=prev)
    assert cont == pytest.approx(full, abs=1e-12)


def test_label_permutation_preserves_numbers():
    part = canonicalize_partition([[0, 1], [2], [3]])
    obs = ((0, 1), (0, 0), (2, 1), (1, 1), (3, 0))
    joint = filter_observations(obs, part, DEFAULT_H)
    for perm in list(permutations(range(N_COMPONENTS)))[:8]:
        ppart = permute_partition(part, perm)
        pobs = permute_observations(obs, perm)
        got = filter_observations(pobs, ppart, DEFAULT_H)
        assert got == pytest.approx(permute_joint(joint, perm), abs=1e-12)


def test_independent_unobserved_group_has_zero_evidence_update():
    part = all_partitions_of_type("four_singletons")[0]
    prefix = ((0, 1), (0, 1), (0, 0), (0, 1))
    suffix = ((0, 1), (0, 0))
    prev = filter_observations(prefix, part, DEFAULT_H)
    cur = filter_observations(prefix + suffix, part, DEFAULT_H)
    r = marginals(transition_only(prev, len(suffix), part, DEFAULT_H))
    d = [marginals(cur)[k] - r[k] for k in range(N_COMPONENTS)]
    for k in (1, 2, 3):
        assert d[k] == pytest.approx(0.0, abs=1e-12)
    r_prev = marginals(prev)
    assert marginals(cur)[2] != pytest.approx(r_prev[2], abs=1e-9)


def test_unobserved_related_component_differs_across_partitions():
    pairs = generate_pairs()
    gaps = [unobserved_related_gap(p)["gap"] for p in pairs]
    assert max(gaps) > 1e-6
    show = showcase_pair(pairs)
    assert show["unobserved_related"]["gap"] > 1e-6
    k = show["unobserved_related"]["component"]
    suffix_comps = {row[0] for row in show["suffix"]}
    assert k not in suffix_comps


def test_r_cancels_and_common_bias_hides_in_pair_term():
    checks = algebra_numeric_checks()
    assert checks["r_cancels_abs"] < 1e-12
    assert checks["expansion_abs"] < 1e-12
    assert checks["common_bias_L_pair"] == pytest.approx(0.0, abs=1e-12)
    assert checks["grad_max_abs_err_a"] < 2e-8
    assert checks["grad_max_abs_err_b"] < 2e-8


def test_pair_term_is_error_difference_not_new_labels():
    q_a = prior_joint()
    q_b = softmax([0.2 * i for i in range(N_STATES)])
    qhat_a = softmax([0.1 * (i - 8) for i in range(N_STATES)])
    qhat_b = softmax([0.05 * i for i in range(N_STATES)])
    r_a = [0.3, 0.4, 0.5, 0.6]
    r_b = [0.1, 0.2, 0.7, 0.8]
    pair = l_pair_from_marginals(marginals(qhat_a), marginals(qhat_b), marginals(q_a), marginals(q_b), r_a, r_b)
    ea = [marginals(qhat_a)[k] - marginals(q_a)[k] for k in range(N_COMPONENTS)]
    eb = [marginals(qhat_b)[k] - marginals(q_b)[k] for k in range(N_COMPONENTS)]
    from_errors = sum((ea[k] - eb[k]) ** 2 for k in range(N_COMPONENTS)) / N_COMPONENTS
    assert pair == pytest.approx(from_errors, abs=1e-12)


def test_kl_softmax_grad_matches_teacher_minus_student():
    teacher = prior_joint()
    logits = [0.1, -0.2, 0.3, 0.0] * 4
    g = grad_kl_softmax(teacher, logits)
    assert g == pytest.approx([softmax(logits)[i] - teacher[i] for i in range(N_STATES)], abs=1e-12)
    assert kl_div(teacher, teacher) == pytest.approx(0.0, abs=1e-12)


def test_generate_pairs_split_and_holdout():
    pairs = generate_pairs()
    assert len(pairs) == 128
    counts = {s: sum(p["split"] == s for p in pairs) for s in ("train", "dev", "test")}
    assert counts == {"train": 96, "dev": 16, "test": 16}
    fam_split = {}
    for p in pairs:
        fam_split.setdefault(p["family_id"], set()).add(p["split"])
    assert all(len(v) == 1 for v in fam_split.values())
    for p in pairs:
        types = {p["a"]["solver"]["topology_type"], p["b"]["solver"]["topology_type"]}
        if p["split"] in ("train", "dev"):
            assert types <= {"four_singletons", "one_pair"}
            assert "two_pairs" not in types and "triple" not in types
        else:
            assert types <= {"two_pairs", "triple"}
            assert "four_singletons" not in types and "one_pair" not in types
        assert leakage_keys(p["a"]["solver"]) == []
        assert leakage_keys(p["b"]["solver"]) == []
        assert "q_current" not in solver_input(p["a"])
        assert p["a"]["solver"]["prefix"] == p["b"]["solver"]["prefix"]
        assert p["a"]["solver"]["suffix"] == p["b"]["solver"]["suffix"]
        replay = p["a"]["evaluator"]["replay_from_previous"]
        assert replay == pytest.approx(p["a"]["evaluator"]["q_current"], abs=1e-12)
        assert math.isfinite(p["a"]["evaluator"]["log_likelihood"])
        assert p["a"]["evaluator"]["log_likelihood"] < 0
        text = p["a"]["solver"]["structure_text"].lower()
        assert "which-entry" not in text
        assert "true reset" not in text
        assert "hidden outcome" not in text


def test_names_rotate_and_canonical_render():
    part = canonicalize_partition([[0, 2], [1], [3]])
    text = render_partition(part, ("A", "B", "C", "D"), DEFAULT_H)
    assert "{A, C}" in text or "{C, A}" in text
    assert "Bernoulli(0.03)" in text
    assert topology_type(part) == "one_pair"
    assert len(joint_keys()) == 16


def test_probes_are_losses_not_training():
    pairs = generate_pairs()
    probes = evaluate_probes(pairs, split="train")
    assert probes["exact"]["L_base"] == pytest.approx(0.0, abs=1e-12)
    assert probes["exact"]["L_pair"] == pytest.approx(0.0, abs=1e-12)
    assert probes["common_bias"]["L_pair"] < 1e-4
    assert probes["common_bias"]["L_base"] > 1e-4
    assert probes["scope_swap"]["L_pair"] > probes["common_bias"]["L_pair"]
    assert probes["stale"]["L_base"] > probes["exact"]["L_base"]
    for kind, row in probes.items():
        assert row["max_abs_algebra_gap"] < 1e-12


def test_evidence_families_are_enumerated_and_short():
    fams = evidence_families()
    assert len(fams) == 24
    splits = {f.split for f in fams}
    assert splits == {"train", "dev", "test"}
    for fam in fams:
        n = len(fam.prefix) + len(fam.suffix)
        assert 2 <= n <= 12
        ll = sequence_log_likelihood(fam.prefix + fam.suffix, all_partitions_of_type("four_singletons")[0], fam.hazard)
        assert math.isfinite(ll)
