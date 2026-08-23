from __future__ import annotations

import math
import random
from typing import Any

from llm_rl_engine.envs.bandits import _BanditBase
from llm_rl_engine.types import Action, Observation, StepResult

VALID_FAMILIES = ("constant", "sinusoidal", "linear")


# Task-description modes for prompt ablations (what the LLM is *told*).
# Dynamics are identical; only metadata["description"] changes.
DESCRIPTION_MODES = ("blind", "aware", "minimal")


def staged_bandit_description(n_arms: int, mode: str = "blind") -> str:
    """Public task text for StagedBandit under a description mode."""
    mode = (mode or "blind").lower().strip()
    if mode not in DESCRIPTION_MODES:
        raise ValueError(
            f"Unknown description_mode={mode!r}; expected one of {DESCRIPTION_MODES}"
        )
    if mode == "aware":
        # Explicit coaching: tells the agent that regimes will change.
        # Used only for ablations; NOT the fair default.
        return (
            f"A multi-armed bandit with {n_arms} arms. Reward probabilities "
            "follow a sequence of hidden regimes (stages). Within each stage the "
            "law is fixed (stationary, periodic, or linear ramp), but at unannounced "
            "times the mechanism changes. Exploit while a regime lasts; when rewards "
            "collapse, abandon old assumptions and re-adapt."
        )
    if mode == "minimal":
        return (
            f"A multi-armed bandit with {n_arms} arms. "
            "Choose one arm each step to maximize cumulative reward."
        )
    # blind: no non-stationarity hint and no false "fixed probs" claim
    return (
        f"A multi-armed bandit with {n_arms} arms. Each arm has an unknown "
        "reward probability. Pull one arm per step to maximize cumulative "
        "reward."
    )


class StagedBandit(_BanditBase):
    """Bernoulli bandit with hard-cut stage changes across reward families.

    Within each stage the reward law is fixed (constant / sinusoidal / linear).
    Stage boundaries are unannounced. Designed to test belief falsification after
    successful exploitation of a previous regime.

    ``description_mode`` only affects the text shown to LLM agents:
    - ``blind`` (default): stationary MAB wording; no regime-change hint
    - ``aware``: explicitly states that hidden regimes will change
    - ``minimal``: bare maximize-reward instruction
    """

    def __init__(
        self,
        n_arms: int = 4,
        stage_length: int = 30,
        stages: list[dict[str, Any] | str] | None = None,
        force_distinct_best_arms: bool = True,
        strong_p: float = 0.85,
        weak_p: float = 0.20,
        period_range: tuple[int, int] = (8, 16),
        amplitude_range: tuple[float, float] = (0.15, 0.35),
        base_range: tuple[float, float] = (0.25, 0.65),
        linear_variant: str = "ramp_up",
        invalid_penalty: float = -0.1,
        description_mode: str = "blind",
    ) -> None:
        super().__init__(n_arms=n_arms, invalid_penalty=invalid_penalty)
        self.stage_length = max(int(stage_length), 1)
        self.force_distinct_best_arms = force_distinct_best_arms
        self.strong_p = strong_p
        self.weak_p = weak_p
        self.period_range = period_range
        self.amplitude_range = amplitude_range
        self.base_range = base_range
        self.default_linear_variant = linear_variant
        self.description_mode = (description_mode or "blind").lower().strip()
        if self.description_mode not in DESCRIPTION_MODES:
            raise ValueError(
                f"Unknown description_mode={description_mode!r}; "
                f"expected one of {DESCRIPTION_MODES}"
            )
        raw_stages = stages or [
            {"family": "constant"},
            {"family": "constant"},
            {"family": "constant"},
        ]
        self._stage_templates = [self._normalize_stage_template(s) for s in raw_stages]
        self._n_stages = len(self._stage_templates)
        self._stage_params: list[dict[str, Any]] = []
        self._prob_history: list[list[float]] = []

    @property
    def metadata(self) -> dict[str, Any]:
        families = [s["family"] for s in self._stage_templates]
        return {
            "name": "StagedBandit",
            "description": staged_bandit_description(
                self.n_arms, self.description_mode
            ),
            "description_mode": self.description_mode,
            "action_space": self.action_space.description,
            "n_arms": self.n_arms,
            "stage_length": self.stage_length,
            "n_stages": self._n_stages,
            "stage_families": families,
        }

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng.seed(seed)
        self._stage_params = self._sample_all_stage_params()
        self._step = 0
        self._last_action = None
        self._last_reward = None
        self._done = False
        self._refresh_probs()
        self._prob_history = [list(self._probs)]
        return self._make_observation(
            f"Episode started. Choose an arm from 0 to {self.n_arms - 1}."
        )

    def step(self, action: Action) -> StepResult:
        arm, err = self._validate_arm(action)
        if arm is None:
            return self._invalid_step(err or "Invalid arm.")
        t_action = self._step
        stage_idx = self._stage_index_at(t_action)
        _, result = self._pull_arm(arm)
        result.info["stage_index"] = stage_idx
        result.info["stage_family"] = self._stage_params[stage_idx]["family"]
        result.info["best_arm"] = self._best_arm(
            result.info.get("step_probs") or self._probs
        )
        result.info["stage_start"] = stage_idx * self.stage_length
        result.info["stage_end"] = (stage_idx + 1) * self.stage_length
        return result

    def oracle(self) -> dict[str, Any]:
        # Report state for the *next* action time (self._step), so agents
        # refreshed each turn see the post-boundary regime immediately.
        t = max(self._step, 0)
        stage_idx = self._stage_index_at(t)
        probs = self._probs_at(t) if self._stage_params else list(self._probs)
        params = self._stage_params[stage_idx] if self._stage_params else {}
        return {
            "stage_index": stage_idx,
            "stage_family": params.get("family"),
            "stage_start": stage_idx * self.stage_length,
            "stage_end": (stage_idx + 1) * self.stage_length,
            "stage_length": self.stage_length,
            "n_stages": self._n_stages,
            "stage_boundaries": [
                s * self.stage_length for s in range(1, self._n_stages)
            ],
            "stage_params": list(self._stage_params),
            "current_probs": list(probs),
            "prob_history": list(self._prob_history),
            "best_arm": self._best_arm(probs),
            "step": self._step,
        }

    def _normalize_stage_template(self, stage: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(stage, str):
            stage = {"family": stage}
        family = stage.get("family", "constant")
        if family not in VALID_FAMILIES:
            raise ValueError(f"Unknown stage family {family!r}; expected {VALID_FAMILIES}")
        out = dict(stage)
        out["family"] = family
        return out

    def _sample_all_stage_params(self) -> list[dict[str, Any]]:
        used_best: list[int] = []
        params: list[dict[str, Any]] = []
        for template in self._stage_templates:
            family = template["family"]
            if family == "constant":
                p = self._sample_constant_params(used_best)
            elif family == "sinusoidal":
                p = self._sample_sinusoidal_params(used_best)
            else:
                variant = template.get("linear_variant", self.default_linear_variant)
                p = self._sample_linear_params(used_best, variant)
            if self.force_distinct_best_arms and "best" in p:
                used_best.append(p["best"])
            params.append(p)
        return params

    def _pick_best_arm(self, used_best: list[int]) -> int:
        if self.force_distinct_best_arms:
            candidates = [i for i in range(self.n_arms) if i not in used_best]
            if not candidates:
                candidates = list(range(self.n_arms))
            return self._rng.choice(candidates)
        return self._rng.randint(0, self.n_arms - 1)

    def _sample_constant_params(self, used_best: list[int]) -> dict[str, Any]:
        best = self._pick_best_arm(used_best)
        probs = [self.weak_p] * self.n_arms
        probs[best] = self.strong_p
        return {"family": "constant", "best": best, "probs": probs}

    def _sample_sinusoidal_params(self, used_best: list[int]) -> dict[str, Any]:
        lo_p, hi_p = self.period_range
        lo_b, hi_b = self.base_range
        lo_a, hi_a = self.amplitude_range
        period = self._rng.randint(lo_p, hi_p)
        bases = [self._rng.uniform(lo_b, hi_b) for _ in range(self.n_arms)]
        amps = [self._rng.uniform(lo_a, hi_a) for _ in range(self.n_arms)]
        phases = [self._rng.uniform(0, 2 * math.pi) for _ in range(self.n_arms)]
        # Prefer a distinct mean-best arm when forcing diversity (boost its base).
        preferred = self._pick_best_arm(used_best)
        bases[preferred] = max(bases[preferred], 0.55)
        for i in range(self.n_arms):
            if i != preferred:
                bases[i] = min(bases[i], 0.45)
        return {
            "family": "sinusoidal",
            "best": preferred,
            "period": period,
            "bases": bases,
            "amps": amps,
            "phases": phases,
        }

    def _sample_linear_params(
        self, used_best: list[int], variant: str
    ) -> dict[str, Any]:
        best = self._pick_best_arm(used_best)
        if variant == "ramp_down":
            p0, slope = 0.90, -0.65
        else:
            p0, slope = 0.25, 0.60
        return {
            "family": "linear",
            "best": best,
            "variant": variant,
            "p0": p0,
            "slope": slope,
            "weak_p": self.weak_p,
        }

    def _stage_index_at(self, t: int) -> int:
        if not self._stage_params:
            return 0
        idx = t // self.stage_length
        return max(0, min(idx, self._n_stages - 1))

    def _stage_start(self, stage_idx: int) -> int:
        return stage_idx * self.stage_length

    def _probs_at(self, t: int) -> list[float]:
        stage_idx = self._stage_index_at(t)
        params = self._stage_params[stage_idx]
        family = params["family"]
        if family == "constant":
            return list(params["probs"])
        if family == "sinusoidal":
            return self._sinusoidal_probs(t, params)
        return self._linear_probs(t, params)

    def _sinusoidal_probs(self, t: int, params: dict[str, Any]) -> list[float]:
        t_local = t - self._stage_start(self._stage_index_at(t))
        period = max(int(params["period"]), 1)
        out: list[float] = []
        for i in range(self.n_arms):
            val = params["bases"][i] + params["amps"][i] * math.sin(
                2 * math.pi * t_local / period + params["phases"][i]
            )
            out.append(min(0.95, max(0.05, val)))
        return out

    def _linear_probs(self, t: int, params: dict[str, Any]) -> list[float]:
        stage_idx = self._stage_index_at(t)
        t0 = self._stage_start(stage_idx)
        L = self.stage_length
        u = (t - t0) / max(L - 1, 1)
        best = params["best"]
        p_best = min(0.95, max(0.05, params["p0"] + params["slope"] * u))
        out = [params["weak_p"]] * self.n_arms
        out[best] = p_best
        return out

    def _refresh_probs(self) -> None:
        t = max(self._step, 0)
        self._probs = self._probs_at(t)

    def _on_pre_step(self) -> None:
        self._refresh_probs()
        if self._step > 0:
            self._prob_history.append(list(self._probs))
