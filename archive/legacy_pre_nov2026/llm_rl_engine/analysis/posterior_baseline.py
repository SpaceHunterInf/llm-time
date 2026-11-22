"""Ordinary posterior-KL distillation data, renderer, and teacher checks.

Task 008. No paired loss. Historical 128-pair fixtures are not training data.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

from llm_rl_engine.analysis.paired_revision import (
    DEFAULT_H,
    DEFAULT_HIGH,
    DEFAULT_LOW,
    DEFAULT_Q,
    DEFAULT_REDRAW,
    N_COMPONENTS,
    N_STATES,
    all_partitions_of_type,
    canonicalize_partition,
    canonical_json,
    filter_observations,
    group_index,
    kl_div,
    marginals,
    permute_joint,
    permute_observations,
    permute_partition,
    prior_joint,
    sequence_log_likelihood,
    sha256_text,
    state_bits,
    state_index,
    topology_type,
    transition_matrix,
    transition_only,
)
from llm_rl_engine.analysis.selective_revision import decision_values

DATASET_VERSION = "posterior_baseline_v1"
RENDERER_VERSION = "v2"
NUMERIC_ENCODER_VERSION = "v1"
DATASET_SEED = 808
MAX_INPUT_TOKENS = 1024
PAIRWISE_INDEX = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
MAX_SUFFIX = 4
TIMING_SENTENCE = (
    "Before each subsequent observation, each group's reset event is sampled "
    "independently of other groups and of previous events. Apply those resets, "
    "then generate or condition on that observation."
)
N_TRAIN = 2048
N_DEV = 256
N_SAME = 256
N_STRUCT = 256
PREFIX_LEN = (4, 8)
SUFFIX_LEN = (1, 4)
H0_FRACTION = 0.25
NAME_POOL = (
    ("A", "B", "C", "D"),
    ("B", "C", "D", "A"),
    ("W", "X", "Y", "Z"),
    ("P", "Q", "R", "S"),
)
LEX_KEYS = tuple(format(i, "04b") for i in range(N_STATES))
ACTION_MAP = {
    "choose_safe": "choose_safe",
    "choose_B": "choose_linked",
    "diagnose_B": "diagnose_linked",
}
FORBIDDEN_MODEL_MARKERS = (
    "pair_id",
    "family_id",
    "topology_type",
    "split=",
    "q_current",
    "partner",
    "side=",
    "prefix observations",
    "log_likelihood",
    "evidence-driven",
)


def lex_key_to_legacy_index(key: str) -> int:
    if len(key) != N_COMPONENTS or any(ch not in "01" for ch in key):
        raise ValueError(f"bad lex key {key}")
    return state_index([int(ch) for ch in key])


def legacy_joint_to_lex(joint: Sequence[float]) -> list[float]:
    if len(joint) != N_STATES:
        raise ValueError("joint")
    return [float(joint[lex_key_to_legacy_index(key)]) for key in LEX_KEYS]


def lex_joint_to_legacy(lex: Sequence[float]) -> list[float]:
    if len(lex) != N_STATES:
        raise ValueError("lex joint")
    out = [0.0] * N_STATES
    for i, key in enumerate(LEX_KEYS):
        out[lex_key_to_legacy_index(key)] = float(lex[i])
    return out


def lex_dict(joint_legacy: Sequence[float]) -> dict[str, float]:
    return {key: float(joint_legacy[lex_key_to_legacy_index(key)]) for key in LEX_KEYS}


def product_group_kernel_matrix(
    partition: Sequence[Sequence[int]],
    hazard: float,
    *,
    redraw: float = DEFAULT_REDRAW,
) -> list[list[float]]:
    """Independent product of per-group kernels. Does not use the redraw enumerator."""
    if float(redraw) != DEFAULT_REDRAW:
        raise ValueError("product kernel is derived for redraw=0.5")
    h = float(hazard)
    part = canonicalize_partition(partition)
    t = [[0.0] * N_STATES for _ in range(N_STATES)]
    for src in range(N_STATES):
        src_bits = state_bits(src)
        row = []
        for dst in range(N_STATES):
            dst_bits = state_bits(dst)
            p = 1.0
            for group in part:
                src_g = tuple(src_bits[i] for i in group)
                dst_g = tuple(dst_bits[i] for i in group)
                p_g = h / float(1 << len(group))
                if src_g == dst_g:
                    p_g += 1.0 - h
                p *= p_g
            row.append(p)
        z = math.fsum(row)
        t[src] = [x / z for x in row]
    return t


def family_identity(
    prefix: Sequence[tuple[int, int]],
    suffix: Sequence[tuple[int, int]],
) -> str:
    """Canonical observation family under component renaming. Ignores hazard/topology."""
    order: dict[int, int] = {}
    nxt = 0
    for component, _outcome in list(prefix) + list(suffix):
        if component not in order:
            order[component] = nxt
            nxt += 1
    for component in range(N_COMPONENTS):
        if component not in order:
            order[component] = nxt
            nxt += 1
    payload = {
        "boundary": len(prefix),
        "prefix": [(order[c], int(y)) for c, y in prefix],
        "suffix": [(order[c], int(y)) for c, y in suffix],
    }
    return sha256_text(canonical_json(payload))


def individual_input_hash(
    *,
    names: Sequence[str],
    partition: Sequence[Sequence[int]],
    hazard: float,
    q_previous: Sequence[float],
    suffix: Sequence[tuple[int, int]],
    n_prefix: int,
) -> str:
    payload = {
        "hazard": float(hazard),
        "n_prefix": int(n_prefix),
        "names": list(names),
        "partition": [list(g) for g in canonicalize_partition(partition)],
        "q_previous_lex": legacy_joint_to_lex(q_previous),
        "suffix": [list(x) for x in suffix],
    }
    return sha256_text(canonical_json(payload))


def render_model_text(
    *,
    names: Sequence[str],
    partition: Sequence[Sequence[int]],
    hazard: float,
    q_previous: Sequence[float],
    suffix: Sequence[tuple[int, int]],
    n_prefix: int,
    compact: bool = False,
) -> str:
    if len(names) != N_COMPONENTS:
        raise ValueError("names")
    part = canonicalize_partition(partition)
    digits = 8 if compact else 12
    groups = []
    for group in part:
        labels = ", ".join(names[i] for i in group)
        if len(group) == 1:
            groups.append(f"{{{labels}}} has its own Bernoulli({hazard}) reset event.")
        else:
            groups.append(f"{{{labels}}} share one Bernoulli({hazard}) reset event.")
    joint = lex_dict(q_previous)
    joint_txt = json.dumps(
        {k: round(joint[k], digits) for k in LEX_KEYS},
        separators=(",", ":"),
    )
    obs_lines = []
    for i, (component, outcome) in enumerate(suffix):
        t = n_prefix + i + 1
        obs_lines.append(f"t={t}: {names[component]} -> {int(outcome)}")
    return "\n".join(
        [
            f"Components in order: {', '.join(names)}.",
            "Each component is high (1) or low (0).",
            f"Outcome rate is {DEFAULT_HIGH} if high and {DEFAULT_LOW} if low.",
            f"When a group resets, each member independently redraws high with probability {DEFAULT_REDRAW}.",
            "Groups that do not reset keep their current component states.",
            "Every listed group may reset; use the stated group hazards.",
            *groups,
            TIMING_SENTENCE,
            f"previous_state after observation {n_prefix} (keys 0000..1111 = {''.join(names)} left to right):",
            joint_txt,
            "Subsequent observations only:",
            *obs_lines,
            "Return the joint immediately after the last listed observation. No extra transition.",
        ]
    )


def _simulate_observations(
    rng: random.Random,
    partition: Sequence[Sequence[int]],
    hazard: float,
    n_obs: int,
    *,
    q: float = DEFAULT_Q,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
    redraw: float = DEFAULT_REDRAW,
) -> list[tuple[int, int]]:
    part = canonicalize_partition(partition)
    bits = [1 if rng.random() < q else 0 for _ in range(N_COMPONENTS)]
    out: list[tuple[int, int]] = []
    for _ in range(n_obs):
        for group in part:
            if rng.random() < hazard:
                for member in group:
                    bits[member] = 1 if rng.random() < redraw else 0
        component = rng.randrange(N_COMPONENTS)
        rate = high if bits[component] else low
        outcome = 1 if rng.random() < rate else 0
        out.append((component, outcome))
    return out


def _query_component(suffix: Sequence[tuple[int, int]]) -> int:
    return int(suffix[0][0])


def _action_record(probability_high: float) -> dict[str, Any]:
    raw = decision_values(probability_high)
    mapped = {ACTION_MAP[k]: float(v) for k, v in raw["action_values"].items()}
    optimal = ACTION_MAP[raw["optimal_action"]]
    return {
        "action_values": mapped,
        "optimal_action": optimal,
        "net_diagnostic_advantage": raw["net_diagnostic_advantage"],
    }


def _role_flags(
    partition: Sequence[Sequence[int]],
    suffix: Sequence[tuple[int, int]],
) -> dict[str, list[int]]:
    g = group_index(partition)
    observed = sorted({int(c) for c, _ in suffix})
    related = []
    unrelated = []
    for k in range(N_COMPONENTS):
        if k in observed:
            continue
        if any(g[k] == g[o] for o in observed):
            related.append(k)
        else:
            unrelated.append(k)
    return {"observed": observed, "related_unobserved": related, "unrelated": unrelated}


def build_example(
    *,
    example_id: str,
    split: str,
    names: Sequence[str],
    partition: Sequence[Sequence[int]],
    hazard: float,
    prefix: Sequence[tuple[int, int]],
    suffix: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    part = canonicalize_partition(partition)
    q_prev = filter_observations(prefix, part, hazard)
    q_cur = filter_observations(list(prefix) + list(suffix), part, hazard)
    replay = filter_observations(suffix, part, hazard, start=q_prev)
    r_joint = transition_only(q_prev, len(suffix), part, hazard)
    indep = filter_observations(
        list(prefix) + list(suffix),
        all_partitions_of_type("four_singletons")[0],
        hazard,
    )
    query = _query_component(suffix)
    p_linked = marginals(q_cur)[query]
    action = _action_record(p_linked)
    text = render_model_text(
        names=names,
        partition=part,
        hazard=hazard,
        q_previous=q_prev,
        suffix=suffix,
        n_prefix=len(prefix),
    )
    compact = render_model_text(
        names=names,
        partition=part,
        hazard=hazard,
        q_previous=q_prev,
        suffix=suffix,
        n_prefix=len(prefix),
        compact=True,
    )
    model_input = {
        "text": text,
        "text_compact": compact,
        "names": list(names),
        "partition": [list(g) for g in part],
        "hazard": float(hazard),
        "q_init": DEFAULT_Q,
        "low": DEFAULT_LOW,
        "high": DEFAULT_HIGH,
        "redraw": DEFAULT_REDRAW,
        "n_prefix": len(prefix),
        "suffix": [list(x) for x in suffix],
        "q_previous_lex": legacy_joint_to_lex(q_prev),
        "query_component": query,
        "query_name": names[query],
    }
    audit = {
        "example_id": example_id,
        "split": split,
        "family_id": family_identity(prefix, suffix),
        "topology_type": topology_type(part),
        "prefix": [list(x) for x in prefix],
        "q_previous_legacy": list(q_prev),
        "q_current_legacy": list(q_cur),
        "q_current_lex": legacy_joint_to_lex(q_cur),
        "r_joint_legacy": list(r_joint),
        "r_lex": legacy_joint_to_lex(r_joint),
        "independent_lex": legacy_joint_to_lex(indep),
        "stale_lex": legacy_joint_to_lex(q_prev),
        "replay_legacy": list(replay),
        "log_likelihood": sequence_log_likelihood(list(prefix) + list(suffix), part, hazard),
        "marginals_current": marginals(q_cur),
        "roles": _role_flags(part, suffix),
        "action": action,
        "input_hash": individual_input_hash(
            names=names,
            partition=part,
            hazard=hazard,
            q_previous=q_prev,
            suffix=suffix,
            n_prefix=len(prefix),
        ),
        "text_hash": sha256_text(text),
    }
    return {"model_input": model_input, "audit": audit}


def _membership_cycle(kind: str) -> list[tuple[tuple[int, ...], ...]]:
    return list(all_partitions_of_type(kind))


def generate_dataset(seed: int = DATASET_SEED) -> dict[str, Any]:
    rng = random.Random(seed)
    family_split: dict[str, str] = {}
    seen_input: set[str] = set()
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "same_topology": [],
        "structural": [],
    }
    plan = (
        ("train", "four_singletons", N_TRAIN // 2),
        ("train", "one_pair", N_TRAIN // 2),
        ("dev", "four_singletons", N_DEV // 2),
        ("dev", "one_pair", N_DEV // 2),
        ("same_topology", "four_singletons", N_SAME // 2),
        ("same_topology", "one_pair", N_SAME // 2),
        ("structural", "two_pairs", N_STRUCT // 2),
        ("structural", "triple", N_STRUCT // 2),
    )
    attempts = 0
    max_attempts = 200_000
    for split, kind, target in plan:
        n_h0 = int(round(target * H0_FRACTION))
        n_h = target - n_h0
        members = _membership_cycle(kind)
        counts = {(0.0): 0, (DEFAULT_H): 0}
        need = {0.0: n_h0, DEFAULT_H: n_h}
        mi = 0
        while len([e for e in splits[split] if e["audit"]["topology_type"] == kind]) < target:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"could not fill {split} {kind}")
            hazard = 0.0 if counts[0.0] < need[0.0] else DEFAULT_H
            if counts[hazard] >= need[hazard]:
                hazard = 0.0 if counts[0.0] < need[0.0] else DEFAULT_H
            part = members[mi % len(members)]
            mi += 1
            n_pre = rng.randint(*PREFIX_LEN)
            n_suf = rng.randint(*SUFFIX_LEN)
            obs = _simulate_observations(rng, part, hazard, n_pre + n_suf)
            prefix = tuple(obs[:n_pre])
            suffix = tuple(obs[n_pre:])
            fam = family_identity(prefix, suffix)
            owner = family_split.get(fam)
            if owner is not None and owner != split:
                continue
            names = NAME_POOL[rng.randrange(len(NAME_POOL))]
            example_id = f"{split}|{kind}|{len(splits[split])}"
            rec = build_example(
                example_id=example_id,
                split=split,
                names=names,
                partition=part,
                hazard=hazard,
                prefix=prefix,
                suffix=suffix,
            )
            inp_hash = rec["audit"]["input_hash"]
            if inp_hash in seen_input:
                continue
            family_split[fam] = split
            seen_input.add(inp_hash)
            splits[split].append(rec)
            counts[hazard] += 1
    coverage = dataset_coverage(splits)
    return {
        "version": DATASET_VERSION,
        "seed": seed,
        "sampling": {
            "observation_component": "uniform independent of hidden state",
            "outcomes": "Bernoulli from latent rate after group resets",
            "prefix_len": PREFIX_LEN,
            "suffix_len": SUFFIX_LEN,
            "h0_fraction": H0_FRACTION,
            "enumerated": False,
        },
        "n": {k: len(v) for k, v in splits.items()},
        "splits": splits,
        "coverage": coverage,
        "attempts": attempts,
    }


def dataset_coverage(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split, rows in splits.items():
        actions = Counter(r["audit"]["action"]["optimal_action"] for r in rows)
        hazards = Counter(r["model_input"]["hazard"] for r in rows)
        types = Counter(r["audit"]["topology_type"] for r in rows)
        n_rel = sum(1 for r in rows if r["audit"]["roles"]["related_unobserved"])
        n_unrel = sum(1 for r in rows if r["audit"]["roles"]["unrelated"])
        n_obs = sum(1 for r in rows if r["audit"]["roles"]["observed"])
        families = {r["audit"]["family_id"] for r in rows}
        inputs = {r["audit"]["input_hash"] for r in rows}
        out[split] = {
            "n": len(rows),
            "n_families": len(families),
            "n_unique_inputs": len(inputs),
            "actions": dict(actions),
            "hazards": {str(k): v for k, v in hazards.items()},
            "topology_types": dict(types),
            "n_with_observed": n_obs,
            "n_with_related_unobserved": n_rel,
            "n_with_unrelated": n_unrel,
        }
    fam_owner: dict[str, str] = {}
    collisions = 0
    for split, rows in splits.items():
        for r in rows:
            fam = r["audit"]["family_id"]
            prev = fam_owner.get(fam)
            if prev is None:
                fam_owner[fam] = split
            elif prev != split:
                collisions += 1
    out["family_split_collisions"] = collisions
    return out


def public_model_row(rec: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist fields only. No prefix, targets, or split labels."""
    m = rec["model_input"]
    return {
        "text": m["text"],
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
    }


def assert_clean_text(text: str) -> None:
    lower = text.lower()
    for marker in FORBIDDEN_MODEL_MARKERS:
        if marker.lower() in lower:
            raise ValueError(f"model text contains forbidden marker {marker}")


def reference_kls(rec: Mapping[str, Any]) -> dict[str, float]:
    q = rec["audit"]["q_current_legacy"]
    return {
        "exact": kl_div(q, q),
        "stale": kl_div(q, rec["audit"]["q_previous_legacy"]),
        "transition_only": kl_div(q, rec["audit"]["r_joint_legacy"]),
        "independent": kl_div(q, lex_joint_to_legacy(rec["audit"]["independent_lex"])),
    }


def independent_product_joint(component_marginals: Sequence[float]) -> list[float]:
    """Independent Bernoulli joint from four high-bit marginals (legacy index)."""
    if len(component_marginals) != N_COMPONENTS:
        raise ValueError("need four marginals")
    masses = []
    for s in range(N_STATES):
        p = 1.0
        bits = state_bits(s)
        for k, bit in enumerate(bits):
            mk = float(component_marginals[k])
            p *= mk if bit else (1.0 - mk)
        masses.append(p)
    z = math.fsum(masses)
    if z <= 0:
        raise ValueError("zero independent product")
    return [x / z for x in masses]


def same_information_independent(
    q_previous_legacy: Sequence[float],
    suffix: Sequence[tuple[int, int]],
    hazard: float,
) -> list[float]:
    """Singleton-group continuation from the supplied previous joint's product.

    Does not replay the hidden prefix. Historical 008 independent_lex does.
    """
    start = independent_product_joint(marginals(q_previous_legacy))
    singles = all_partitions_of_type("four_singletons")[0]
    return filter_observations(suffix, singles, hazard, start=start)


def numeric_encode(
    *,
    q_previous_lex: Sequence[float],
    partition: Sequence[Sequence[int]],
    hazard: float,
    suffix: Sequence[Sequence[int] | tuple[int, int]],
) -> list[float]:
    """47-d permitted numeric features. No prefix, split, or targets."""
    if len(q_previous_lex) != N_STATES:
        raise ValueError("q_previous_lex")
    joint = [float(x) for x in q_previous_lex]
    z = math.fsum(joint)
    if z <= 0:
        raise ValueError("zero joint")
    joint = [x / z for x in joint]
    g = group_index(partition)
    pairs = [1.0 if g[i] == g[j] else 0.0 for i, j in PAIRWISE_INDEX]
    obs = [0.0] * (MAX_SUFFIX * (N_COMPONENTS + 2))
    for t, item in enumerate(suffix):
        if t >= MAX_SUFFIX:
            raise ValueError("suffix longer than 4")
        component, outcome = int(item[0]), int(item[1])
        base = t * (N_COMPONENTS + 2)
        obs[base + component] = 1.0
        obs[base + N_COMPONENTS] = float(outcome)
        obs[base + N_COMPONENTS + 1] = 1.0
    return joint + pairs + [float(hazard)] + obs


def permute_numeric_features(features: Sequence[float], perm: Sequence[int]) -> list[float]:
    """perm[new] = old component index. Matches permute_joint convention."""
    if sorted(perm) != list(range(N_COMPONENTS)):
        raise ValueError("perm")
    inv = [0] * N_COMPONENTS
    for new_i, old_i in enumerate(perm):
        inv[old_i] = new_i
    joint = list(features[:N_STATES])
    legacy = lex_joint_to_legacy(joint)
    perm_legacy = permute_joint(legacy, perm)
    pairs = list(features[N_STATES : N_STATES + 6])
    old_pair = {PAIRWISE_INDEX[i]: pairs[i] for i in range(6)}
    new_pairs = []
    for a, b in PAIRWISE_INDEX:
        oa, ob = perm[a], perm[b]
        key = (oa, ob) if oa < ob else (ob, oa)
        new_pairs.append(old_pair[key])
    hazard = [float(features[N_STATES + 6])]
    rest = list(features[N_STATES + 7 :])
    new_rest = []
    width = N_COMPONENTS + 2
    for t in range(MAX_SUFFIX):
        block = rest[t * width : (t + 1) * width]
        onehot = block[:N_COMPONENTS]
        outcome, present = block[N_COMPONENTS], block[N_COMPONENTS + 1]
        new_hot = [0.0] * N_COMPONENTS
        for old_c, val in enumerate(onehot):
            new_hot[inv[old_c]] = val
        new_rest.extend(new_hot + [outcome, present])
    return legacy_joint_to_lex(perm_legacy) + new_pairs + hazard + new_rest


def score_prediction(
    pred_lex: Sequence[float],
    teacher_legacy: Sequence[float],
    *,
    partition: Sequence[Sequence[int]],
    suffix: Sequence[Sequence[int] | tuple[int, int]],
    r_legacy: Sequence[float],
) -> dict[str, Any]:
    """Score all four components. Regret uses reference utilities only."""
    pred_legacy = lex_joint_to_legacy(pred_lex)
    teacher_m = marginals(teacher_legacy)
    pred_m = marginals(pred_legacy)
    r_m = marginals(r_legacy)
    roles = _role_flags(partition, [(int(c), int(y)) for c, y in suffix])
    joint_mse = math.fsum((a - b) ** 2 for a, b in zip(pred_legacy, teacher_legacy)) / N_STATES
    joint_kl = kl_div(teacher_legacy, pred_legacy)
    per_comp = []
    for k in range(N_COMPONENTS):
        ref = _action_record(teacher_m[k])
        pred_act = _action_record(pred_m[k])
        regret = ref["action_values"][ref["optimal_action"]] - ref["action_values"][pred_act["optimal_action"]]
        per_comp.append(
            {
                "component": k,
                "abs_error": abs(pred_m[k] - teacher_m[k]),
                "true_d": teacher_m[k] - r_m[k],
                "pred_d": pred_m[k] - r_m[k],
                "optimal_action": ref["optimal_action"],
                "chosen_action": pred_act["optimal_action"],
                "regret": regret,
            }
        )
    role_means = {}
    for role, members in roles.items():
        if not members:
            role_means[role] = None
            continue
        role_means[role] = {
            "abs_error": sum(per_comp[k]["abs_error"] for k in members) / len(members),
            "regret": sum(per_comp[k]["regret"] for k in members) / len(members),
            "n_components": len(members),
        }
    return {
        "joint_kl": joint_kl,
        "joint_mse": joint_mse,
        "per_component": per_comp,
        "roles": roles,
        "role_means": role_means,
        "mean_abs_error": sum(c["abs_error"] for c in per_comp) / N_COMPONENTS,
        "mean_regret": sum(c["regret"] for c in per_comp) / N_COMPONENTS,
    }


def write_dataset(dataset: Mapping[str, Any], output_dir) -> dict[str, str]:
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, rows in dataset["splits"].items():
        model_rows = []
        audit_rows = []
        for rec in rows:
            assert_clean_text(rec["model_input"]["text"])
            model_rows.append(public_model_row(rec))
            audit_rows.append(rec["audit"])
        model_path = output_dir / f"{split}_model.jsonl"
        audit_path = output_dir / f"{split}_audit.jsonl"
        model_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in model_rows))
        audit_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in audit_rows))
        hashes[f"{split}_model"] = sha256_text(model_path.read_text())
        hashes[f"{split}_audit"] = sha256_text(audit_path.read_text())
    manifest = {
        "version": dataset["version"],
        "seed": dataset["seed"],
        "sampling": dataset["sampling"],
        "n": dataset["n"],
        "coverage": dataset["coverage"],
        "file_hashes": hashes,
        "lex_keys": list(LEX_KEYS),
        "max_input_tokens": MAX_INPUT_TOKENS,
        "note": "Pilot data, not a sealed publication test. Do not train on 007 pair fixtures.",
    }
    man_path = output_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    hashes["manifest"] = sha256_text(man_path.read_text())
    return hashes
