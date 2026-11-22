"""Four-component reset-group filter and paired revision-objective prototype.

CPU feasibility only (task 007). No model weights, no API calls, no training.
The paired term adds no teacher labels; it reweights prediction errors.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Iterable, Mapping, Sequence

N_COMPONENTS = 4
N_STATES = 16
DEFAULT_Q = 0.99
DEFAULT_H = 0.03
DEFAULT_LOW = 0.2
DEFAULT_HIGH = 0.8
DEFAULT_REDRAW = 0.5
KL_FLOOR = 1e-15
PAIR_COUNT = 128
TRAIN_PAIRS = 96
DEV_PAIRS = 16
TEST_PAIRS = 16
LAMBDA_DEFAULT = 1.0
TOPOLOGY_TRAIN = ("four_singletons", "one_pair")
TOPOLOGY_TEST = ("two_pairs", "triple")
HIDDEN_EVALUATOR_KEYS = (
    "q_current",
    "r",
    "d",
    "r_joint",
    "log_likelihood",
    "marginals_current",
    "replay_from_previous",
    "evaluator",
)

NAME_ROTATIONS = (
    ("A", "B", "C", "D"),
    ("B", "C", "D", "A"),
    ("C", "D", "A", "B"),
    ("D", "A", "B", "C"),
    ("W", "X", "Y", "Z"),
    ("P", "Q", "R", "S"),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_bits(index: int) -> tuple[int, ...]:
    return tuple((index >> i) & 1 for i in range(N_COMPONENTS))


def state_index(bits: Sequence[int]) -> int:
    if len(bits) != N_COMPONENTS:
        raise ValueError("expected four bits")
    out = 0
    for i, bit in enumerate(bits):
        out |= (int(bit) & 1) << i
    return out


def joint_keys() -> list[str]:
    return ["".join(str(b) for b in state_bits(s)) for s in range(N_STATES)]


def _normalize(masses: Sequence[float]) -> list[float]:
    z = math.fsum(masses)
    if z <= 0 or not math.isfinite(z):
        raise ValueError("zero or nonfinite mass")
    return [x / z for x in masses]


def _check_joint(joint: Sequence[float]) -> None:
    if len(joint) != N_STATES:
        raise ValueError(f"joint must have {N_STATES} masses")
    if any(not math.isfinite(x) or x < 0 for x in joint):
        raise ValueError("joint masses must be finite and nonnegative")


def prior_joint(q: float = DEFAULT_Q) -> list[float]:
    q = float(q)
    joint = []
    for s in range(N_STATES):
        p = 1.0
        for bit in state_bits(s):
            p *= q if bit else (1.0 - q)
        joint.append(p)
    return _normalize(joint)


def canonicalize_partition(groups: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    used: list[int] = []
    frozen = []
    for group in groups:
        g = tuple(sorted(int(x) for x in group))
        if not g:
            raise ValueError("empty group")
        if any(x < 0 or x >= N_COMPONENTS for x in g):
            raise ValueError("component out of range")
        if len(set(g)) != len(g):
            raise ValueError("duplicate member in group")
        used.extend(g)
        frozen.append(g)
    if sorted(used) != list(range(N_COMPONENTS)):
        raise ValueError("partition must cover each component once")
    frozen.sort()
    return tuple(frozen)


def topology_type(partition: Sequence[Sequence[int]]) -> str:
    part = canonicalize_partition(partition)
    sizes = tuple(sorted(len(g) for g in part))
    if sizes == (1, 1, 1, 1):
        return "four_singletons"
    if sizes == (1, 1, 2):
        return "one_pair"
    if sizes == (2, 2):
        return "two_pairs"
    if sizes == (1, 3):
        return "triple"
    raise ValueError(f"unsupported partition sizes {sizes}")


def all_partitions_of_type(kind: str) -> tuple[tuple[tuple[int, ...], ...], ...]:
    comps = range(N_COMPONENTS)
    if kind == "four_singletons":
        return (canonicalize_partition([[i] for i in comps]),)
    if kind == "one_pair":
        out = []
        for i in comps:
            for j in range(i + 1, N_COMPONENTS):
                rest = [[k] for k in comps if k not in (i, j)]
                out.append(canonicalize_partition([[i, j], *rest]))
        return tuple(out)
    if kind == "two_pairs":
        return (
            canonicalize_partition([[0, 1], [2, 3]]),
            canonicalize_partition([[0, 2], [1, 3]]),
            canonicalize_partition([[0, 3], [1, 2]]),
        )
    if kind == "triple":
        out = []
        for sing in comps:
            trip = [k for k in comps if k != sing]
            out.append(canonicalize_partition([trip, [sing]]))
        return tuple(out)
    raise ValueError(kind)


def render_partition(partition: Sequence[Sequence[int]], names: Sequence[str], hazard: float) -> str:
    part = canonicalize_partition(partition)
    if len(names) != N_COMPONENTS:
        raise ValueError("need four names")
    chunks = []
    for group in part:
        labels = ", ".join(names[i] for i in group)
        if len(group) == 1:
            chunks.append(
                f"{{{labels}}} has its own Bernoulli({hazard}) reset event"
            )
        else:
            chunks.append(
                f"{{{labels}}} share one Bernoulli({hazard}) reset event"
            )
    body = ". ".join(chunks)
    return (
        f"{body}. When a group resets, each member independently redraws high with "
        f"probability {DEFAULT_REDRAW} and low with probability {1.0 - DEFAULT_REDRAW}. "
        "Groups that do not reset keep their current component states. "
        "The unrelated-looking groups may still reset; serialize the stated group hazards. "
        "No hidden reset events or future outcomes are provided."
    )


def group_index(partition: Sequence[Sequence[int]]) -> tuple[int, ...]:
    part = canonicalize_partition(partition)
    idx = [0] * N_COMPONENTS
    for g_i, group in enumerate(part):
        for member in group:
            idx[member] = g_i
    return tuple(idx)


def transition_matrix(
    partition: Sequence[Sequence[int]],
    hazard: float,
    *,
    redraw: float = DEFAULT_REDRAW,
) -> list[list[float]]:
    h = float(hazard)
    if not 0.0 <= h <= 1.0:
        raise ValueError("hazard must be in [0, 1]")
    redraw = float(redraw)
    part = canonicalize_partition(partition)
    n_groups = len(part)
    t = [[0.0] * N_STATES for _ in range(N_STATES)]
    n_masks = 1 << n_groups
    for src in range(N_STATES):
        src_bits = state_bits(src)
        for mask in range(n_masks):
            p_mask = 1.0
            for g in range(n_groups):
                p_mask *= h if (mask >> g) & 1 else (1.0 - h)
            if p_mask == 0.0:
                continue
            _accumulate_redraws(t[src], src_bits, part, mask, p_mask, redraw)
        z = math.fsum(t[src])
        if z <= 0:
            raise ValueError("invalid transition row")
        t[src] = [x / z for x in t[src]]
    return t


def _accumulate_redraws(
    row: list[float],
    src_bits: tuple[int, ...],
    part: tuple[tuple[int, ...], ...],
    mask: int,
    p_mask: float,
    redraw: float,
) -> None:
    n_groups = len(part)
    reset_groups = [g for g in range(n_groups) if (mask >> g) & 1]
    if not reset_groups:
        dst = state_index(src_bits)
        row[dst] += p_mask
        return
    sizes = [len(part[g]) for g in reset_groups]
    n_assign = 1
    for s in sizes:
        n_assign *= 1 << s
    for assign in range(n_assign):
        bits = list(src_bits)
        p_draw = 1.0
        cursor = assign
        for g, size in zip(reset_groups, sizes):
            local = cursor & ((1 << size) - 1)
            cursor >>= size
            for j, member in enumerate(part[g]):
                bit = (local >> j) & 1
                bits[member] = bit
                p_draw *= redraw if bit else (1.0 - redraw)
        row[state_index(bits)] += p_mask * p_draw


def brute_transition_row(
    src: int,
    partition: Sequence[Sequence[int]],
    hazard: float,
    *,
    redraw: float = DEFAULT_REDRAW,
) -> list[float]:
    """Enumerate group resets and member redraws independently of the matrix builder."""
    h = float(hazard)
    redraw = float(redraw)
    part = canonicalize_partition(partition)
    src_bits = state_bits(src)
    row = [0.0] * N_STATES
    n_groups = len(part)
    for mask in range(1 << n_groups):
        p_mask = 1.0
        for g in range(n_groups):
            p_mask *= h if (mask >> g) & 1 else (1.0 - h)
        _accumulate_redraws(row, src_bits, part, mask, p_mask, redraw)
    return _normalize(row)


def predict(joint: Sequence[float], matrix: Sequence[Sequence[float]]) -> list[float]:
    _check_joint(joint)
    out = []
    for dst in range(N_STATES):
        out.append(math.fsum(joint[src] * matrix[src][dst] for src in range(N_STATES)))
    return _normalize(out)


def observe(
    joint: Sequence[float],
    component: int,
    outcome: int,
    *,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
) -> list[float]:
    _check_joint(joint)
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")
    if component < 0 or component >= N_COMPONENTS:
        raise ValueError("component out of range")
    like = []
    for s in range(N_STATES):
        bit = state_bits(s)[component]
        rate = high if bit else low
        like.append(rate if outcome == 1 else (1.0 - rate))
    return _normalize([joint[i] * like[i] for i in range(N_STATES)])


def step_filter(
    joint: Sequence[float],
    matrix: Sequence[Sequence[float]],
    component: int,
    outcome: int,
    *,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
) -> list[float]:
    return observe(predict(joint, matrix), component, outcome, low=low, high=high)


def marginals(joint: Sequence[float]) -> list[float]:
    _check_joint(joint)
    out = [0.0] * N_COMPONENTS
    for s, p in enumerate(joint):
        bits = state_bits(s)
        for k, bit in enumerate(bits):
            if bit:
                out[k] += p
    return out


def filter_observations(
    observations: Sequence[tuple[int, int]],
    partition: Sequence[Sequence[int]],
    hazard: float,
    *,
    q: float = DEFAULT_Q,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
    start: Sequence[float] | None = None,
) -> list[float]:
    joint = list(start) if start is not None else prior_joint(q)
    matrix = transition_matrix(partition, hazard)
    for component, outcome in observations:
        joint = step_filter(joint, matrix, component, outcome, low=low, high=high)
    return joint


def transition_only(
    start: Sequence[float],
    n_steps: int,
    partition: Sequence[Sequence[int]],
    hazard: float,
) -> list[float]:
    joint = list(start)
    if n_steps < 0:
        raise ValueError("n_steps")
    if n_steps == 0:
        return _normalize(joint)
    matrix = transition_matrix(partition, hazard)
    for _ in range(n_steps):
        joint = predict(joint, matrix)
    return joint


def sequence_log_likelihood(
    observations: Sequence[tuple[int, int]],
    partition: Sequence[Sequence[int]],
    hazard: float,
    *,
    q: float = DEFAULT_Q,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
) -> float:
    joint = prior_joint(q)
    matrix = transition_matrix(partition, hazard)
    ll = 0.0
    for component, outcome in observations:
        pred = predict(joint, matrix)
        p_high = marginals(pred)[component]
        p_y = (low + (high - low) * p_high) if outcome == 1 else (1.0 - (low + (high - low) * p_high))
        if p_y <= 0:
            raise ValueError("non-positive observation probability")
        ll += math.log(p_y)
        joint = observe(pred, component, outcome, low=low, high=high)
    return ll


def permute_joint(joint: Sequence[float], perm: Sequence[int]) -> list[float]:
    """Relabel components by perm[new] = old."""
    if sorted(perm) != list(range(N_COMPONENTS)):
        raise ValueError("perm")
    out = [0.0] * N_STATES
    for s, p in enumerate(joint):
        old_bits = state_bits(s)
        new_bits = [0] * N_COMPONENTS
        for new_i, old_i in enumerate(perm):
            new_bits[new_i] = old_bits[old_i]
        out[state_index(new_bits)] += p
    return _normalize(out)


def permute_partition(partition: Sequence[Sequence[int]], perm: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """perm[new] = old; map old component ids to new ids."""
    inv = [0] * N_COMPONENTS
    for new_i, old_i in enumerate(perm):
        inv[old_i] = new_i
    return canonicalize_partition([[inv[i] for i in group] for group in partition])


def permute_observations(
    observations: Sequence[tuple[int, int]], perm: Sequence[int]
) -> list[tuple[int, int]]:
    inv = [0] * N_COMPONENTS
    for new_i, old_i in enumerate(perm):
        inv[old_i] = new_i
    return [(inv[c], y) for c, y in observations]


# --- objective algebra ---


def softmax(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    return _normalize(exps)


def kl_div(teacher: Sequence[float], student: Sequence[float]) -> float:
    _check_joint(teacher)
    _check_joint(student)
    total = 0.0
    for q, p in zip(teacher, student):
        if q <= 0:
            continue
        p = max(float(p), KL_FLOOR)
        total += q * (math.log(q) - math.log(p))
    return total


def paired_residual(
    mhat_a: Sequence[float],
    mhat_b: Sequence[float],
    m_a: Sequence[float],
    m_b: Sequence[float],
    r_a: Sequence[float] | None = None,
    r_b: Sequence[float] | None = None,
) -> list[float]:
    """(dhat_a - dhat_b) - (d_a - d_b). r cancels when supplied."""
    if r_a is None or r_b is None:
        return [
            (mhat_a[k] - m_a[k]) - (mhat_b[k] - m_b[k])
            for k in range(len(m_a))
        ]
    dhat_a = [mhat_a[k] - r_a[k] for k in range(len(m_a))]
    dhat_b = [mhat_b[k] - r_b[k] for k in range(len(m_b))]
    d_a = [m_a[k] - r_a[k] for k in range(len(m_a))]
    d_b = [m_b[k] - r_b[k] for k in range(len(m_b))]
    return [(dhat_a[k] - dhat_b[k]) - (d_a[k] - d_b[k]) for k in range(len(m_a))]


def l_pair_from_marginals(
    mhat_a: Sequence[float],
    mhat_b: Sequence[float],
    m_a: Sequence[float],
    m_b: Sequence[float],
    r_a: Sequence[float] | None = None,
    r_b: Sequence[float] | None = None,
) -> float:
    resid = paired_residual(mhat_a, mhat_b, m_a, m_b, r_a, r_b)
    return math.fsum(x * x for x in resid) / len(resid)


def l_mse_term(mhat_a: Sequence[float], mhat_b: Sequence[float], m_a: Sequence[float], m_b: Sequence[float]) -> float:
    ea = [(mhat_a[k] - m_a[k]) ** 2 for k in range(len(m_a))]
    eb = [(mhat_b[k] - m_b[k]) ** 2 for k in range(len(m_b))]
    return 0.5 * (sum(ea) / len(ea) + sum(eb) / len(eb))


def objectives_for_pair(
    q_a: Sequence[float],
    q_b: Sequence[float],
    qhat_a: Sequence[float],
    qhat_b: Sequence[float],
    *,
    r_a: Sequence[float] | None = None,
    r_b: Sequence[float] | None = None,
    lam: float = LAMBDA_DEFAULT,
    lam_mse: float = LAMBDA_DEFAULT,
) -> dict[str, float]:
    m_a = marginals(q_a)
    m_b = marginals(q_b)
    mhat_a = marginals(qhat_a)
    mhat_b = marginals(qhat_b)
    base = 0.5 * (kl_div(q_a, qhat_a) + kl_div(q_b, qhat_b))
    pair = l_pair_from_marginals(mhat_a, mhat_b, m_a, m_b, r_a, r_b)
    mse = l_mse_term(mhat_a, mhat_b, m_a, m_b)
    ea = [mhat_a[k] - m_a[k] for k in range(N_COMPONENTS)]
    eb = [mhat_b[k] - m_b[k] for k in range(N_COMPONENTS)]
    cross = math.fsum(ea[k] * eb[k] for k in range(N_COMPONENTS)) / N_COMPONENTS
    return {
        "L_base": base,
        "L_pair": pair,
        "L_mse": mse,
        "L_cross": cross,
        "L_kl": base,
        "L_kl_mse": base + lam_mse * mse,
        "L_kl_pair": base + lam * pair,
        "pair_minus_twice_mse_plus_cross": pair - (2.0 * mse - 2.0 * cross),
    }


def grad_kl_softmax(teacher: Sequence[float], logits: Sequence[float]) -> list[float]:
    student = softmax(logits)
    return [student[i] - teacher[i] for i in range(N_STATES)]


def grad_pair_softmax(
    logits_a: Sequence[float],
    logits_b: Sequence[float],
    q_a: Sequence[float],
    q_b: Sequence[float],
) -> tuple[list[float], list[float]]:
    qhat_a = softmax(logits_a)
    qhat_b = softmax(logits_b)
    ea = [marginals(qhat_a)[k] - marginals(q_a)[k] for k in range(N_COMPONENTS)]
    eb = [marginals(qhat_b)[k] - marginals(q_b)[k] for k in range(N_COMPONENTS)]
    delta = [ea[k] - eb[k] for k in range(N_COMPONENTS)]
    kdim = float(N_COMPONENTS)
    dL_dqhat_a = [0.0] * N_STATES
    dL_dqhat_b = [0.0] * N_STATES
    for s in range(N_STATES):
        bits = state_bits(s)
        for k, bit in enumerate(bits):
            if bit:
                dL_dqhat_a[s] += 2.0 * delta[k] / kdim
                dL_dqhat_b[s] += -2.0 * delta[k] / kdim
    ga = _softmax_jacobian_t(qhat_a, dL_dqhat_a)
    gb = _softmax_jacobian_t(qhat_b, dL_dqhat_b)
    return ga, gb


def _softmax_jacobian_t(p: Sequence[float], dL_dp: Sequence[float]) -> list[float]:
    dot = math.fsum(p[i] * dL_dp[i] for i in range(len(p)))
    return [p[i] * (dL_dp[i] - dot) for i in range(len(p))]


def finite_diff_grad(
    loss_fn,
    logits: list[float],
    eps: float = 1e-6,
) -> list[float]:
    base = loss_fn(logits)
    out = []
    for i in range(len(logits)):
        logits[i] += eps
        up = loss_fn(logits)
        logits[i] -= 2 * eps
        down = loss_fn(logits)
        logits[i] += eps
        out.append((up - down) / (2 * eps))
    return out


def ipf_marginals(joint: Sequence[float], targets: Sequence[float], rounds: int = 40) -> list[float]:
    p = list(joint)
    _check_joint(p)
    for _ in range(rounds):
        for k, tk in enumerate(targets):
            tk = min(max(float(tk), 1e-12), 1.0 - 1e-12)
            cur = marginals(p)[k]
            cur = min(max(cur, 1e-12), 1.0 - 1e-12)
            scale_hi = tk / cur
            scale_lo = (1.0 - tk) / (1.0 - cur)
            nxt = []
            for s, mass in enumerate(p):
                nxt.append(mass * (scale_hi if state_bits(s)[k] else scale_lo))
            p = _normalize(nxt)
    return p


# --- fixtures ---


@dataclass(frozen=True)
class EvidenceFamily:
    family_id: str
    split: str
    hazard: float
    prefix: tuple[tuple[int, int], ...]
    suffix: tuple[tuple[int, int], ...]
    names: tuple[str, ...]
    note: str


def _obs(component: int, bits: str) -> tuple[tuple[int, int], ...]:
    return tuple((component, int(ch)) for ch in bits)


def evidence_families() -> list[EvidenceFamily]:
    """Enumerated evidence, not draws from the generative model."""
    n0 = NAME_ROTATIONS[0]
    n1 = NAME_ROTATIONS[1]
    n2 = NAME_ROTATIONS[4]
    n3 = NAME_ROTATIONS[5]
    train = [
        EvidenceFamily("T00", "train", 0.0, _obs(0, "1111"), _obs(0, "11"), n0, "h=0 all-positive on 0"),
        EvidenceFamily("T01", "train", 0.0, _obs(0, "0000"), _obs(0, "00"), n1, "h=0 all-negative on 0"),
        EvidenceFamily("T02", "train", 0.0, _obs(0, "1011"), _obs(0, "10"), n2, "h=0 mixed on 0"),
        EvidenceFamily("T03", "train", 0.0, _obs(0, "111") + _obs(1, "1"), _obs(0, "00"), n3, "h=0 two components"),
        EvidenceFamily("T04", "train", DEFAULT_H, _obs(0, "1111"), _obs(0, "1111"), n0, "noisy all-positive 0"),
        EvidenceFamily("T05", "train", DEFAULT_H, _obs(0, "0000"), _obs(0, "0000"), n1, "noisy all-negative 0"),
        EvidenceFamily("T06", "train", DEFAULT_H, _obs(0, "1010"), _obs(0, "10"), n2, "ambiguous 0"),
        EvidenceFamily("T07", "train", DEFAULT_H, _obs(0, "111") + _obs(1, "1"), _obs(0, "11") + _obs(1, "0"), n3, "0 and 1"),
        EvidenceFamily("T08", "train", DEFAULT_H, _obs(0, "11") + _obs(2, "11"), _obs(0, "1") + _obs(3, "1"), n0, "spread"),
        EvidenceFamily("T09", "train", DEFAULT_H, _obs(0, "111111"), _obs(1, "11"), n1, "observe 0 then 1"),
        EvidenceFamily("T10", "train", DEFAULT_H, _obs(0, "000") + _obs(1, "1"), _obs(0, "111"), n2, "flip 0"),
        EvidenceFamily("T11", "train", DEFAULT_H, ((0, 1), (1, 1), (2, 1), (3, 1)), _obs(0, "10"), n3, "all then 0"),
        EvidenceFamily("T12", "train", DEFAULT_H, _obs(1, "1111"), _obs(0, "11"), n0, "observe 1 then 0"),
        EvidenceFamily("T13", "train", DEFAULT_H, ((0, 1), (1, 0), (0, 1), (1, 1)), _obs(2, "11"), n1, "then 2"),
        EvidenceFamily("T14", "train", DEFAULT_H, _obs(0, "1100"), ((0, 1), (0, 0), (1, 1)), n2, "mixed suffix"),
        EvidenceFamily("T15", "train", DEFAULT_H, _obs(3, "1111"), _obs(0, "11"), n3, "3 then 0"),
    ]
    dev = [
        EvidenceFamily("D00", "dev", 0.0, _obs(1, "1010"), _obs(1, "01"), n0, "h=0 on 1"),
        EvidenceFamily("D01", "dev", DEFAULT_H, _obs(0, "111") + _obs(2, "000"), ((0, 1), (2, 0)), n1, "0 vs 2"),
        EvidenceFamily("D02", "dev", DEFAULT_H, _obs(2, "1111"), _obs(2, "00") + _obs(0, "1"), n2, "2 then 0"),
        EvidenceFamily("D03", "dev", DEFAULT_H, ((0, 1), (1, 1), (0, 0), (1, 0), (2, 1)), _obs(0, "01"), n3, "busy prefix"),
    ]
    test = [
        EvidenceFamily("X00", "test", DEFAULT_H, _obs(0, "1111"), _obs(0, "11"), n0, "holdout observe 0"),
        EvidenceFamily("X01", "test", DEFAULT_H, _obs(0, "0000") + _obs(3, "11"), _obs(0, "00"), n1, "0 and singleton 3"),
        EvidenceFamily("X02", "test", 0.0, _obs(0, "111") + _obs(1, "000"), _obs(0, "1"), n2, "h=0 related contrast"),
        EvidenceFamily("X03", "test", DEFAULT_H, ((0, 1), (2, 1), (0, 0), (2, 0), (0, 1), (0, 1)), ((0, 1), (1, 0)), n3, "mixed holdout"),
    ]
    families = train + dev + test
    for fam in families:
        n = len(fam.prefix) + len(fam.suffix)
        if n > 12 or not fam.prefix or not fam.suffix:
            raise ValueError(f"bad family {fam.family_id}")
    return families


def _example_record(
    *,
    pair_id: str,
    side: str,
    family: EvidenceFamily,
    partition: tuple[tuple[int, ...], ...],
    q: float = DEFAULT_Q,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
) -> dict[str, Any]:
    observations = family.prefix + family.suffix
    q_prev = filter_observations(family.prefix, partition, family.hazard, q=q, low=low, high=high)
    q_cur = filter_observations(observations, partition, family.hazard, q=q, low=low, high=high)
    replay = filter_observations(family.suffix, partition, family.hazard, start=q_prev, q=q, low=low, high=high)
    r_joint = transition_only(q_prev, len(family.suffix), partition, family.hazard)
    r = marginals(r_joint)
    m_cur = marginals(q_cur)
    d = [m_cur[k] - r[k] for k in range(N_COMPONENTS)]
    ll = sequence_log_likelihood(observations, partition, family.hazard, q=q, low=low, high=high)
    names = family.names
    solver = {
        "pair_id": pair_id,
        "side": side,
        "family_id": family.family_id,
        "split": family.split,
        "topology_type": topology_type(partition),
        "partition": [list(g) for g in partition],
        "partition_canonical": partition,
        "structure_text": render_partition(partition, names, family.hazard),
        "names": list(names),
        "hazard": family.hazard,
        "q_init": q,
        "low": low,
        "high": high,
        "redraw": DEFAULT_REDRAW,
        "prefix": [list(x) for x in family.prefix],
        "suffix": [list(x) for x in family.suffix],
        "q_previous": q_prev,
        "checkpoint": len(family.prefix),
        "n_suffix": len(family.suffix),
    }
    evaluator = {
        "q_current": q_cur,
        "r": r,
        "d": d,
        "r_joint": r_joint,
        "log_likelihood": ll,
        "marginals_current": m_cur,
        "replay_from_previous": replay,
        "note": family.note,
    }
    return {
        "solver": solver,
        "evaluator": evaluator,
        "identity": {
            "pair_id": pair_id,
            "side": side,
            "family_id": family.family_id,
            "split": family.split,
            "partition_hash": sha256_text(canonical_json(partition)),
            "evidence_hash": sha256_text(canonical_json({"prefix": family.prefix, "suffix": family.suffix, "hazard": family.hazard})),
            "solver_hash": sha256_text(canonical_json(solver)),
        },
    }


def _assert_positive_likelihood(family: EvidenceFamily, partitions: Sequence[tuple[tuple[int, ...], ...]]) -> None:
    obs = family.prefix + family.suffix
    for part in partitions:
        ll = sequence_log_likelihood(obs, part, family.hazard)
        if not math.isfinite(ll):
            raise ValueError(f"nonfinite ll {family.family_id}")


def generate_pairs() -> list[dict[str, Any]]:
    families = {f.family_id: f for f in evidence_families()}
    one_pair = all_partitions_of_type("one_pair")
    four = all_partitions_of_type("four_singletons")[0]
    two_pairs = all_partitions_of_type("two_pairs")
    triples = all_partitions_of_type("triple")
    pairs: list[dict[str, Any]] = []

    train_ids = [f"T{i:02d}" for i in range(16)]
    if len(one_pair) != 6:
        raise RuntimeError("expected 6 one-pair partitions")
    for fam_id in train_ids:
        fam = families[fam_id]
        for part in one_pair:
            pair_id = f"{fam_id}|four_singletons|{topology_type(part)}|{part}"
            _assert_positive_likelihood(fam, (four, part))
            a = _example_record(pair_id=pair_id, side="a", family=fam, partition=four)
            b = _example_record(pair_id=pair_id, side="b", family=fam, partition=part)
            pairs.append(_pack_pair(pair_id, "train", a, b, fam))

    dev_parts = one_pair[:4]
    for fam_id in ("D00", "D01", "D02", "D03"):
        fam = families[fam_id]
        for part in dev_parts:
            pair_id = f"{fam_id}|four_singletons|{part}"
            _assert_positive_likelihood(fam, (four, part))
            a = _example_record(pair_id=pair_id, side="a", family=fam, partition=four)
            b = _example_record(pair_id=pair_id, side="b", family=fam, partition=part)
            pairs.append(_pack_pair(pair_id, "dev", a, b, fam))

    test_combos = (
        (two_pairs[0], triples[3]),
        (two_pairs[0], triples[0]),
        (two_pairs[1], triples[3]),
        (two_pairs[2], triples[2]),
    )
    for fam_id in ("X00", "X01", "X02", "X03"):
        fam = families[fam_id]
        for pa, pb in test_combos:
            pair_id = f"{fam_id}|{pa}|{pb}"
            _assert_positive_likelihood(fam, (pa, pb))
            a = _example_record(pair_id=pair_id, side="a", family=fam, partition=pa)
            b = _example_record(pair_id=pair_id, side="b", family=fam, partition=pb)
            pairs.append(_pack_pair(pair_id, "test", a, b, fam))

    if len(pairs) != PAIR_COUNT:
        raise RuntimeError(f"expected {PAIR_COUNT} pairs, got {len(pairs)}")
    n_train = sum(p["split"] == "train" for p in pairs)
    n_dev = sum(p["split"] == "dev" for p in pairs)
    n_test = sum(p["split"] == "test" for p in pairs)
    if (n_train, n_dev, n_test) != (TRAIN_PAIRS, DEV_PAIRS, TEST_PAIRS):
        raise RuntimeError((n_train, n_dev, n_test))
    return pairs


def _pack_pair(pair_id: str, split: str, a: dict, b: dict, family: EvidenceFamily) -> dict[str, Any]:
    da = a["evaluator"]["d"]
    db = b["evaluator"]["d"]
    d_diff = [da[k] - db[k] for k in range(N_COMPONENTS)]
    return {
        "pair_id": pair_id,
        "split": split,
        "family_id": family.family_id,
        "hazard": family.hazard,
        "enumeration": "hand-specified evidence; not sampled from the generative model",
        "a": a,
        "b": b,
        "d_diff": d_diff,
        "same_evidence": True,
        "previous_states_equal": a["solver"]["q_previous"] == b["solver"]["q_previous"],
        "pair_hash": sha256_text(canonical_json({"a": a["identity"], "b": b["identity"]})),
    }


def solver_input(example: Mapping[str, Any]) -> dict[str, Any]:
    return dict(example["solver"])


def evaluator_targets(example: Mapping[str, Any]) -> dict[str, Any]:
    return dict(example["evaluator"])


def leakage_keys(solver: Mapping[str, Any]) -> list[str]:
    return [k for k in solver if k in HIDDEN_EVALUATOR_KEYS or k == "q_current"]


def unobserved_related_gap(pair: Mapping[str, Any]) -> dict[str, Any]:
    """Largest |d_a - d_b| on a component never observed in the suffix and grouped in exactly one side."""
    suffix_a = {int(c) for c, _ in pair["a"]["solver"]["suffix"]}
    suffix_b = {int(c) for c, _ in pair["b"]["solver"]["suffix"]}
    suffix = suffix_a | suffix_b
    pa = canonicalize_partition(pair["a"]["solver"]["partition"])
    pb = canonicalize_partition(pair["b"]["solver"]["partition"])
    ga = group_index(pa)
    gb = group_index(pb)
    da = pair["a"]["evaluator"]["d"]
    db = pair["b"]["evaluator"]["d"]
    best = None
    for k in range(N_COMPONENTS):
        if k in suffix:
            continue
        a_related = any(k != o and ga[k] == ga[o] for o in suffix)
        b_related = any(k != o and gb[k] == gb[o] for o in suffix)
        if a_related == b_related:
            continue
        gap = abs(da[k] - db[k])
        rec = {"component": k, "gap": gap, "related_in": "a" if a_related else "b", "d_a": da[k], "d_b": db[k]}
        if best is None or gap > best["gap"]:
            best = rec
    return best or {"component": None, "gap": 0.0}


# --- synthetic probes ---


def _uniform_joint() -> list[float]:
    return [1.0 / N_STATES] * N_STATES


def probe_qhat(kind: str, example: Mapping[str, Any], partner: Mapping[str, Any], rng: random.Random) -> list[float]:
    q = example["evaluator"]["q_current"]
    if kind == "exact":
        return list(q)
    if kind == "stale":
        return list(example["solver"]["q_previous"])
    if kind == "independent":
        part = all_partitions_of_type("four_singletons")[0]
        fam_prefix = [tuple(x) for x in example["solver"]["prefix"]]
        fam_suffix = [tuple(x) for x in example["solver"]["suffix"]]
        return filter_observations(
            fam_prefix + fam_suffix,
            part,
            example["solver"]["hazard"],
            q=example["solver"]["q_init"],
            low=example["solver"]["low"],
            high=example["solver"]["high"],
        )
    if kind == "scope_swap":
        return list(partner["evaluator"]["q_current"])
    if kind == "common_bias":
        m = marginals(q)
        targets = [min(max(m[k] + 0.05, 1e-6), 1.0 - 1e-6) for k in range(N_COMPONENTS)]
        return ipf_marginals(q, targets)
    if kind == "random":
        logits = [rng.gauss(0.0, 1.0) for _ in range(N_STATES)]
        return softmax(logits)
    raise ValueError(kind)


def evaluate_probes(pairs: Sequence[Mapping[str, Any]], *, split: str = "train", seed: int = 7) -> dict[str, Any]:
    subset = [p for p in pairs if p["split"] == split]
    kinds = ("exact", "stale", "independent", "scope_swap", "common_bias", "random")
    rng = random.Random(seed)
    out: dict[str, Any] = {}
    for kind in kinds:
        rows = []
        qhats = []
        for pair in subset:
            qa = probe_qhat(kind, pair["a"], pair["b"], rng)
            qb = probe_qhat(kind, pair["b"], pair["a"], rng)
            qhats.append((qa, qb, pair))
            rows.append(
                objectives_for_pair(
                    pair["a"]["evaluator"]["q_current"],
                    pair["b"]["evaluator"]["q_current"],
                    qa,
                    qb,
                    r_a=pair["a"]["evaluator"]["r"],
                    r_b=pair["b"]["evaluator"]["r"],
                )
            )
        shuffled = []
        perm = list(range(len(qhats)))
        rng2 = random.Random(seed + 99)
        rng2.shuffle(perm)
        for i, j in enumerate(perm):
            qa, _, pair_i = qhats[i]
            _, qb, pair_j = qhats[j]
            shuffled.append(
                objectives_for_pair(
                    pair_i["a"]["evaluator"]["q_current"],
                    pair_j["b"]["evaluator"]["q_current"],
                    qa,
                    qb,
                    r_a=pair_i["a"]["evaluator"]["r"],
                    r_b=pair_j["b"]["evaluator"]["r"],
                )
            )
        def mean_key(items, key):
            return sum(r[key] for r in items) / len(items)

        out[kind] = {
            "n": len(rows),
            "L_base": mean_key(rows, "L_base"),
            "L_pair": mean_key(rows, "L_pair"),
            "L_mse": mean_key(rows, "L_mse"),
            "L_kl_mse": mean_key(rows, "L_kl_mse"),
            "L_kl_pair": mean_key(rows, "L_kl_pair"),
            "L_kl_shuffle": mean_key(shuffled, "L_base") + LAMBDA_DEFAULT * mean_key(shuffled, "L_pair"),
            "L_pair_shuffled": mean_key(shuffled, "L_pair"),
            "max_abs_algebra_gap": max(abs(r["pair_minus_twice_mse_plus_cross"]) for r in rows),
        }
    return out


def algebra_numeric_checks() -> dict[str, Any]:
    rng = random.Random(11)
    q_a = softmax([rng.gauss(0, 1) for _ in range(N_STATES)])
    q_b = softmax([rng.gauss(0, 1) for _ in range(N_STATES)])
    r_a = [rng.random() for _ in range(N_COMPONENTS)]
    r_b = [rng.random() for _ in range(N_COMPONENTS)]
    qhat_a = softmax([rng.gauss(0, 1) for _ in range(N_STATES)])
    qhat_b = softmax([rng.gauss(0, 1) for _ in range(N_STATES)])
    m_a, m_b = marginals(q_a), marginals(q_b)
    mhat_a, mhat_b = marginals(qhat_a), marginals(qhat_b)
    with_r = l_pair_from_marginals(mhat_a, mhat_b, m_a, m_b, r_a, r_b)
    without = l_pair_from_marginals(mhat_a, mhat_b, m_a, m_b)
    ea = [mhat_a[k] - m_a[k] for k in range(N_COMPONENTS)]
    eb = [mhat_b[k] - m_b[k] for k in range(N_COMPONENTS)]
    mse = l_mse_term(mhat_a, mhat_b, m_a, m_b)
    cross = math.fsum(ea[k] * eb[k] for k in range(N_COMPONENTS)) / N_COMPONENTS
    expanded = 2.0 * mse - 2.0 * cross

    mhat_a_b = [m_a[k] + 0.04 for k in range(N_COMPONENTS)]
    mhat_b_b = [m_b[k] + 0.04 for k in range(N_COMPONENTS)]
    common = l_pair_from_marginals(mhat_a_b, mhat_b_b, m_a, m_b)

    logits_a = [rng.gauss(0, 0.4) for _ in range(N_STATES)]
    logits_b = [rng.gauss(0, 0.4) for _ in range(N_STATES)]

    def loss_a(z):
        qa = softmax(z)
        qb = softmax(logits_b)
        base = 0.5 * (kl_div(q_a, qa) + kl_div(q_b, qb))
        pair = l_pair_from_marginals(marginals(qa), marginals(qb), m_a, m_b)
        return base + pair

    def loss_b(z):
        qa = softmax(logits_a)
        qb = softmax(z)
        base = 0.5 * (kl_div(q_a, qa) + kl_div(q_b, qb))
        pair = l_pair_from_marginals(marginals(qa), marginals(qb), m_a, m_b)
        return base + pair

    g_kl_a = grad_kl_softmax(q_a, logits_a)
    g_pair_a, g_pair_b = grad_pair_softmax(logits_a, logits_b, q_a, q_b)
    analytic_a = [0.5 * g_kl_a[i] + g_pair_a[i] for i in range(N_STATES)]
    fd_a = finite_diff_grad(lambda z: loss_a(z), list(logits_a))
    g_kl_b = grad_kl_softmax(q_b, logits_b)
    analytic_b = [0.5 * g_kl_b[i] + g_pair_b[i] for i in range(N_STATES)]
    fd_b = finite_diff_grad(lambda z: loss_b(z), list(logits_b))
    err_a = max(abs(analytic_a[i] - fd_a[i]) for i in range(N_STATES))
    err_b = max(abs(analytic_b[i] - fd_b[i]) for i in range(N_STATES))
    return {
        "r_cancels_abs": abs(with_r - without),
        "expansion_abs": abs(with_r - expanded),
        "common_bias_L_pair": common,
        "grad_max_abs_err_a": err_a,
        "grad_max_abs_err_b": err_b,
    }


def fixture_coverage(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list] = {"train": [], "dev": [], "test": []}
    for p in pairs:
        by_split[p["split"]].append(p)
    families = {s: sorted({p["family_id"] for p in rows}) for s, rows in by_split.items()}
    types = {
        s: sorted({p["a"]["solver"]["topology_type"] for p in rows} | {p["b"]["solver"]["topology_type"] for p in rows})
        for s, rows in by_split.items()
    }
    gaps = [unobserved_related_gap(p)["gap"] for p in pairs]
    lls = []
    for p in pairs:
        lls.append(p["a"]["evaluator"]["log_likelihood"])
        lls.append(p["b"]["evaluator"]["log_likelihood"])
    return {
        "n_pairs": len(pairs),
        "by_split": {s: len(v) for s, v in by_split.items()},
        "families": families,
        "topology_types": types,
        "max_unobserved_related_gap": max(gaps),
        "n_pairs_with_related_gap": sum(g > 1e-12 for g in gaps),
        "log_likelihood_min": min(lls),
        "log_likelihood_max": max(lls),
        "n_previous_states_differ": sum(not p["previous_states_equal"] for p in pairs),
        "enumerated": True,
    }


def showcase_pair(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prefer h>0 train pair with a large unobserved-related gap (T04, pair {0,1}).

    h=0 cannot create related-component evidence: the prior is independent and
    groups that never reset stay independent.
    """
    wanted = None
    for p in pairs:
        if p["family_id"] != "T04":
            continue
        part = canonicalize_partition(p["b"]["solver"]["partition"])
        if (0, 1) in part:
            wanted = p
            break
    if wanted is None:
        wanted = pairs[0]
    gap = unobserved_related_gap(wanted)
    a, b = wanted["a"], wanted["b"]
    return {
        "pair_id": wanted["pair_id"],
        "family_id": wanted["family_id"],
        "hazard": wanted["hazard"],
        "prefix": a["solver"]["prefix"],
        "suffix": a["solver"]["suffix"],
        "names": a["solver"]["names"],
        "structure_a": a["solver"]["structure_text"],
        "structure_b": b["solver"]["structure_text"],
        "partition_a": a["solver"]["partition"],
        "partition_b": b["solver"]["partition"],
        "q_previous_a_marginals": marginals(a["solver"]["q_previous"]),
        "q_previous_b_marginals": marginals(b["solver"]["q_previous"]),
        "q_current_a_marginals": a["evaluator"]["marginals_current"],
        "q_current_b_marginals": b["evaluator"]["marginals_current"],
        "r_a": a["evaluator"]["r"],
        "r_b": b["evaluator"]["r"],
        "d_a": a["evaluator"]["d"],
        "d_b": b["evaluator"]["d"],
        "log_likelihood_a": a["evaluator"]["log_likelihood"],
        "log_likelihood_b": b["evaluator"]["log_likelihood"],
        "unobserved_related": gap,
        "previous_states_equal": wanted["previous_states_equal"],
    }
