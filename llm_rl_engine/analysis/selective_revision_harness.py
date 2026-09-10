"""Dry-run/live harness for the selective-revision pilot.

Dry-run never reads .env, never constructs an API client, and never uses the
network. Live mode requires explicit model, call cap, token caps, verified
pricing, and a USD ceiling that covers the full grid plus the retry reserve.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Protocol

from tqdm import tqdm

from llm_rl_engine.agents.llm import GenerationResult
from llm_rl_engine.analysis.selective_revision_pilot import (
    EvaluatorRecord,
    PilotSpec,
    PublicCase,
    ValidationError,
    build_jobs,
    canonical_json,
    estimate_tokens,
    parse_response,
    prompt_bundle,
    prompt_leaks_evaluator,
    score_response,
    sha256_text,
    spec_from_mapping,
    summarize_scores,
)
from llm_rl_engine.analysis.token_cost import DEFAULT_MODEL_PRICING, estimate_cost_usd
from llm_rl_engine.config.loader import load_config

CREDENTIAL_NAMES = {".env", ".env.example"}
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_MAX_INPUT = 2000
DEFAULT_MAX_OUTPUT = 2048
DEFAULT_THINKING_BUDGET = 0
REPAIR_SUFFIX = (
    "\n\nYour previous response was invalid: {error}\n"
    "Return exactly one valid JSON object with keys probability_high, "
    "predictive_mean, and action. Do not include a rationale."
)


class PilotClient(Protocol):
    def generate(self, prompt: str) -> GenerationResult:
        ...


class BudgetError(ValueError):
    pass


class LiveConfigError(ValueError):
    pass


class IdentityError(ValueError):
    pass


class BookkeepingError(RuntimeError):
    pass


SCIENTIFIC_SCHEMA = "probability_high+predictive_mean+action"
RESPONSE_MIME = "application/json"
SUCCESS_FINISH_REASONS = {"STOP", "stop", "END_TURN", "end_turn"}
IMPLEMENTATION_FILES = (
    "llm_rl_engine/analysis/selective_revision.py",
    "llm_rl_engine/analysis/selective_revision_pilot.py",
    "llm_rl_engine/analysis/selective_revision_harness.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def persist_provider_response(
    output_dir: Path,
    *,
    case_id: str,
    treatment: str,
    attempt_n: int,
    attempt_id: str,
    result: GenerationResult,
    latency_s: float,
    request_text: str,
    request_hash: str,
    generation_settings: Mapping[str, Any],
    job_key_value: str,
    scored: bool = False,
) -> Path:
    """Write the provider payload before scoring. Crash-safe for later resume."""
    usage_status = getattr(result, "usage_status", None) or (
        "available" if (result.input_tokens or result.output_tokens) else "unknown"
    )
    payload = {
        "attempt_id": attempt_id,
        "attempt_n": attempt_n,
        "case_id": case_id,
        "treatment": treatment,
        "job_key": job_key_value,
        "scored": scored,
        "received_at": _utc_now(),
        "latency_s": latency_s,
        "request_text": request_text,
        "request_hash": request_hash,
        "generation_settings": dict(generation_settings),
        "raw_response": result.text,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "thoughts_tokens": int(getattr(result, "thoughts_tokens", 0) or 0),
        "model_version": getattr(result, "model_version", None),
        "finish_reason": getattr(result, "finish_reason", None),
        "usage_status": usage_status,
    }
    path = output_dir / "responses" / f"{case_id}_{treatment}_{attempt_n:04d}.json"
    atomic_write_json(path, payload)
    atomic_write_json(output_dir / "last_response.json", payload)
    return path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_hashes(repo_root: Path) -> dict[str, str]:
    out = {}
    for rel in IMPLEMENTATION_FILES:
        path = repo_root / rel
        out[rel] = file_sha256(path) if path.is_file() else "missing"
    return out


def spec_fingerprint(spec: PilotSpec) -> dict[str, Any]:
    return {
        "schema_version": spec.schema_version,
        "prior_high": spec.prior_high,
        "hazard": spec.hazard,
        "low": spec.low,
        "high": spec.high,
        "safe": spec.safe,
        "batch_length": spec.batch_length,
        "successes": list(spec.successes),
        "costs": list(spec.costs),
        "structures": list(spec.structures),
        "treatments": list(spec.treatments),
    }


def build_scientific_identity(
    spec: PilotSpec,
    settings: LiveSettings,
    repo_root: Path,
) -> dict[str, Any]:
    body = {
        "schema_version": 2,
        "kind": "selective_revision_scientific_identity",
        "model": settings.model,
        "temperature": settings.temperature,
        "thinking_budget": settings.thinking_budget,
        "configured_max_output_tokens": settings.max_output_tokens,
        "provider_max_output_tokens": settings.billed_output_limit,
        "provider_output_limit_note": (
            "Gemini max_output_tokens is set to configured_max_output_tokens + thinking_budget; "
            "it is not an independent visible-text cap."
        ),
        "max_input_tokens": settings.max_input_tokens,
        "response_mime_type": RESPONSE_MIME,
        "output_schema": SCIENTIFIC_SCHEMA,
        "retry_policy": {
            "max_retries": settings.max_retries,
            "repair_includes_reference_answers": False,
        },
        "grid": spec_fingerprint(spec),
        "implementation_hashes": implementation_hashes(repo_root),
    }
    body["effective_hash"] = sha256_text(canonical_json(body))
    return body


def operational_limits(settings: LiveSettings) -> dict[str, Any]:
    return {
        "max_calls": settings.max_calls,
        "max_usd": settings.max_usd,
        "note": "Spending limits may increase without changing scientific identity.",
    }


def check_or_init_identity(
    output_dir: Path,
    identity: Mapping[str, Any],
    settings: LiveSettings,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scientific_identity.json"
    manifest_path = output_dir / "live_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stored = existing.get("effective_hash")
        if not stored:
            raise IdentityError(
                "legacy identity file has no effective_hash; refuse silent resume"
            )
        if stored != identity["effective_hash"]:
            raise IdentityError(
                "scientific identity mismatch; refuse resume into this directory "
                f"(stored={stored[:12]}… new={identity['effective_hash'][:12]}…)"
            )
        ops_path = output_dir / "operational_limits.json"
        if ops_path.exists():
            prev = json.loads(ops_path.read_text(encoding="utf-8"))
            if settings.max_calls < int(prev.get("max_calls") or 0):
                raise IdentityError("max_calls cannot decrease on resume")
            if settings.max_usd + 1e-15 < float(prev.get("max_usd") or 0):
                raise IdentityError("max_usd cannot decrease on resume")
        atomic_write_json(ops_path, operational_limits(settings))
        return
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not existing.get("effective_hash"):
            raise IdentityError(
                "legacy live_manifest.json has no effective_hash; refuse silent resume. "
                "Analyze the directory as-is or start a new output directory."
            )
        if existing.get("effective_hash") != identity["effective_hash"]:
            raise IdentityError("scientific identity mismatch in live_manifest.json")
    atomic_write_json(path, dict(identity))
    atomic_write_json(output_dir / "operational_limits.json", operational_limits(settings))


def generation_settings_payload(settings: LiveSettings) -> dict[str, Any]:
    return {
        "model": settings.model,
        "temperature": settings.temperature,
        "thinking_budget": settings.thinking_budget,
        "provider_max_output_tokens": settings.billed_output_limit,
        "configured_max_output_tokens": settings.max_output_tokens,
        "response_mime_type": RESPONSE_MIME,
    }


def count_request_tokens(
    text: str,
    client: Any,
    cache: dict[str, tuple[int, str]],
) -> tuple[int, str]:
    request_hash = sha256_text(text)
    if request_hash in cache:
        return cache[request_hash]
    counter = getattr(client, "count_tokens", None)
    if callable(counter):
        n = int(counter(text))
        source = "client_count"
    else:
        n = estimate_tokens(text)
        source = "offline_heuristic"
    cache[request_hash] = (n, source)
    return n, source


def find_unscored_response(
    output_dir: Path, case_id: str, treatment: str
) -> dict[str, Any] | None:
    folder = output_dir / "responses"
    if not folder.is_dir():
        return None
    matches = sorted(folder.glob(f"{case_id}_{treatment}_*.json"))
    for path in reversed(matches):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("raw_response") is not None and not payload.get("scored"):
            payload["_path"] = str(path)
            return payload
    return None


def result_from_saved(payload: Mapping[str, Any]) -> GenerationResult:
    usage_status = str(payload.get("usage_status") or "unknown")
    return GenerationResult(
        text=str(payload.get("raw_response") or ""),
        input_tokens=int(payload.get("input_tokens") or 0),
        output_tokens=int(payload.get("output_tokens") or 0),
        total_tokens=int(payload.get("total_tokens") or 0),
        thoughts_tokens=int(payload.get("thoughts_tokens") or 0),
        model_version=payload.get("model_version"),
        finish_reason=payload.get("finish_reason"),
        usage_status=usage_status,
    )


def finish_is_incomplete(finish_reason: str | None, finish_available: bool) -> bool:
    if not finish_available or finish_reason is None:
        return False
    return str(finish_reason) not in SUCCESS_FINISH_REASONS


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def verified_pricing(config: Mapping[str, Any], model: str) -> dict[str, float]:
    pricing = dict(config.get("pricing") or {})
    if "input_per_1m" in pricing and "output_per_1m" in pricing:
        rates = {
            "input_per_1m": float(pricing["input_per_1m"]),
            "output_per_1m": float(pricing["output_per_1m"]),
        }
    else:
        table = DEFAULT_MODEL_PRICING
        if model not in table:
            raise LiveConfigError(
                f"no verified pricing for {model}; set pricing.input_per_1m and output_per_1m"
            )
        rates = dict(table[model])
    if rates["input_per_1m"] < 0 or rates["output_per_1m"] < 0:
        raise LiveConfigError("pricing rates must be nonnegative")
    return rates


def reserved_cost_usd(
    *,
    max_input_tokens: int,
    max_output_tokens: int,
    model: str,
    pricing: Mapping[str, float],
) -> float:
    return estimate_cost_usd(
        model, max_input_tokens, max_output_tokens, {model: dict(pricing)}
    )["total_cost_usd"]


def snapshot_code(root: Path) -> dict[str, Any]:
    """Git HEAD plus hashes of dirty files. Never reads credential files."""
    root = root.resolve()
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain", "-uall"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_head": None, "dirty_file_hashes": {}, "note": "git unavailable"}
    dirty: dict[str, str] = {}
    for line in porcelain.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip()
        name = Path(path).name
        if name in CREDENTIAL_NAMES:
            dirty[path] = "omitted_credentials"
            continue
        target = root / path
        if target.is_file():
            dirty[path] = hashlib.sha256(target.read_bytes()).hexdigest()
        else:
            dirty[path] = "missing"
    return {"git_head": head, "dirty_file_hashes": dirty}


def job_key(
    case_id: str,
    treatment: str,
    effective_hash: str,
    prompt_hash: str,
) -> str:
    return f"{case_id}|{treatment}|{effective_hash}|{prompt_hash}"


def classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "api key" in text or "auth" in text or "permission" in text or "unauthor" in text:
        return "authentication"
    if "quota" in text or "resource exhausted" in text or "rate limit" in text:
        return "quota"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "network" in text or "connection" in text or "dns" in text:
        return "network"
    return "api_failure"


@dataclass(frozen=True)
class LiveSettings:
    model: str
    max_calls: int
    max_usd: float
    max_input_tokens: int
    max_output_tokens: int
    thinking_budget: int
    max_retries: int
    temperature: float
    pricing: dict[str, float]

    @property
    def billed_output_limit(self) -> int:
        """Conservative billed-output cap: visible tokens plus a finite thinking budget."""
        return self.max_output_tokens + self.thinking_budget

    @property
    def reserved_per_attempt(self) -> float:
        return reserved_cost_usd(
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.billed_output_limit,
            model=self.model,
            pricing=self.pricing,
        )


def live_settings_from_config(
    config: Mapping[str, Any], overrides: Mapping[str, Any] | None = None
) -> LiveSettings:
    overrides = dict(overrides or {})
    live = dict(config.get("live") or {})
    merged = {**live, **{k: v for k, v in overrides.items() if v is not None}}
    model = str(merged.get("model") or config.get("model") or DEFAULT_MODEL)
    required = ("max_calls", "max_usd", "max_input_tokens", "max_output_tokens")
    missing = [name for name in required if merged.get(name) is None]
    if missing:
        raise LiveConfigError(
            "live mode requires explicit " + ", ".join(required) + f"; missing {missing}"
        )
    try:
        max_calls = int(merged["max_calls"])
        max_input = int(merged["max_input_tokens"])
        max_output = int(merged["max_output_tokens"])
        thinking = int(merged.get("thinking_budget", DEFAULT_THINKING_BUDGET))
        retries = int(merged.get("max_retries", config.get("max_retries", 1)))
        max_usd = float(merged["max_usd"])
        temperature = float(merged.get("temperature", config.get("temperature", 0.0)))
    except (TypeError, ValueError) as exc:
        raise LiveConfigError(f"invalid live budget inputs: {exc}") from exc
    if max_calls <= 0 or max_input <= 0 or max_output <= 0:
        raise LiveConfigError("max_calls and token limits must be positive")
    if max_usd <= 0 or not math_isfinite(max_usd):
        raise LiveConfigError("max_usd must be a positive finite number")
    if retries < 0:
        raise LiveConfigError("max_retries must be nonnegative")
    if thinking < 0:
        raise LiveConfigError("thinking_budget must be nonnegative")
    pricing = verified_pricing(config, model)
    return LiveSettings(
        model=model,
        max_calls=max_calls,
        max_usd=max_usd,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        thinking_budget=thinking,
        max_retries=retries,
        temperature=temperature,
        pricing=pricing,
    )


def math_isfinite(value: float) -> bool:
    import math

    return math.isfinite(value)


def preflight(
    n_jobs: int,
    settings: LiveSettings,
    prompts: list[str],
    *,
    client: Any = None,
    token_cache: dict[str, tuple[int, str]] | None = None,
) -> dict[str, Any]:
    attempts_needed = n_jobs * (1 + settings.max_retries)
    if settings.max_calls < attempts_needed:
        raise BudgetError(
            f"max_calls={settings.max_calls} does not cover {n_jobs} jobs with "
            f"{settings.max_retries} retries each ({attempts_needed} attempts)"
        )
    reserved = attempts_needed * settings.reserved_per_attempt
    if settings.max_usd + 1e-15 < reserved:
        raise BudgetError(
            f"max_usd={settings.max_usd} does not cover retry-reserve cost {reserved:.6f}"
        )
    cache = token_cache if token_cache is not None else {}
    overflows = []
    sources: list[str] = []
    counted: list[int] = []
    dummy_repair = REPAIR_SUFFIX.format(error="invalid JSON: missing field action")
    for index, prompt in enumerate(prompts):
        n, source = count_request_tokens(prompt, client, cache)
        if settings.max_retries:
            n_repair, _ = count_request_tokens(prompt + dummy_repair, client, cache)
            n = max(n, n_repair)
        counted.append(n)
        sources.append(source)
        if n > settings.max_input_tokens:
            overflows.append({"index": index, "token_count": n, "source": source})
    if overflows:
        raise BudgetError(
            f"{len(overflows)} prompts exceed max_input_tokens={settings.max_input_tokens}"
        )
    return {
        "n_jobs": n_jobs,
        "attempts_reserved": attempts_needed,
        "reserved_usd": reserved,
        "reserved_per_attempt_usd": settings.reserved_per_attempt,
        "max_usd": settings.max_usd,
        "max_calls": settings.max_calls,
        "model": settings.model,
        "max_input_tokens": settings.max_input_tokens,
        "max_output_tokens": settings.max_output_tokens,
        "thinking_budget": settings.thinking_budget,
        "billed_output_limit": settings.billed_output_limit,
        "provider_output_limit_note": (
            "provider max_output_tokens = configured_max_output_tokens + thinking_budget"
        ),
        "token_count_max": max(counted) if counted else 0,
        "token_count_source": sorted(set(sources)),
        "pricing": settings.pricing,
    }


class BoundedGeminiClient:
    """Pilot-only client. Does not alter the historical benchmark GeminiClient."""

    def __init__(
        self,
        model: str,
        *,
        max_output_tokens: int,
        thinking_budget: int = 0,
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
        self.temperature = temperature
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")

    def generate(self, prompt: str) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google.genai import Client, types

        client = Client(api_key=self.api_key)
        billed_limit = self.max_output_tokens + max(self.thinking_budget, 0)
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=billed_limit,
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
            response_mime_type="application/json",
        )
        response = client.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            input_tokens = 0
            billed_output = 0
            thoughts = 0
            total = 0
            usage_status = "unknown"
        else:
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            candidates = int(getattr(usage, "candidates_token_count", 0) or 0)
            thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
            billed_output = candidates + thoughts
            total = int(getattr(usage, "total_token_count", 0) or (input_tokens + billed_output))
            usage_status = "available" if (input_tokens or billed_output) else "unknown"
        text = getattr(response, "text", None) or ""
        finish = None
        candidates_list = getattr(response, "candidates", None) or []
        if candidates_list:
            finish = getattr(candidates_list[0], "finish_reason", None)
            finish = None if finish is None else str(finish).split(".")[-1]
        model_version = getattr(response, "model_version", None)
        if model_version is None:
            model_version = getattr(response, "model", None)
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=billed_output,
            total_tokens=total,
            thoughts_tokens=thoughts,
            model_version=None if model_version is None else str(model_version),
            finish_reason=finish,
            usage_status=usage_status,
        )

    def count_tokens(self, text: str) -> int:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google.genai import Client

        client = Client(api_key=self.api_key)
        result = client.models.count_tokens(model=self.model, contents=text)
        return int(getattr(result, "total_tokens", 0) or 0)


class ReferenceAnswerClient:
    """Test double that returns evaluator targets as a valid model response."""

    def __init__(
        self,
        answers: Mapping[str, dict[str, Any]] | None = None,
        *,
        usage_status: str = "available",
        finish_reason: str | None = "STOP",
        model_version: str | None = "fake-client",
    ) -> None:
        self.answers = dict(answers or {})
        self.prompts: list[str] = []
        self.calls = 0
        self.usage_status = usage_status
        self.finish_reason = finish_reason
        self.model_version = model_version

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def generate(self, prompt: str) -> GenerationResult:
        self.calls += 1
        self.prompts.append(prompt)
        payload = self.answers.get(prompt)
        if payload is None:
            raise RuntimeError("no scripted answer for prompt")
        text = json.dumps(payload, sort_keys=True)
        if self.usage_status == "unknown":
            return GenerationResult(
                text=text,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                thoughts_tokens=0,
                model_version=self.model_version,
                finish_reason=self.finish_reason,
                usage_status="unknown",
            )
        return GenerationResult(
            text=text,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            model_version=self.model_version,
            finish_reason=self.finish_reason,
            usage_status="available",
        )


def empty_ledger() -> dict[str, Any]:
    return {
        "spent_usd": 0.0,
        "reserved_usd": 0.0,
        "attempts": 0,
        "completed_jobs": 0,
        "interrupted_reservations": 0,
    }


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_ledger()
    data = json.loads(path.read_text(encoding="utf-8"))
    base = empty_ledger()
    base.update(data)
    return base


class PilotRun:
    def __init__(
        self,
        output_dir: Path,
        config: Mapping[str, Any],
        spec: PilotSpec,
        *,
        repo_root: Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = dict(config)
        self.spec = spec
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
        self.jobs = build_jobs(spec)
        self.config_hash = sha256_text(canonical_json(self._hashable_config()))
        self.model = str(self.config.get("model") or DEFAULT_MODEL)
        self.identity: dict[str, Any] | None = None

    def bind_settings(self, settings: LiveSettings) -> dict[str, Any]:
        self.model = settings.model
        self.identity = build_scientific_identity(self.spec, settings, self.repo_root)
        return self.identity

    def effective_hash(self) -> str:
        if self.identity and self.identity.get("effective_hash"):
            return str(self.identity["effective_hash"])
        return self.config_hash

    def _hashable_config(self) -> dict[str, Any]:
        data = json.loads(canonical_json(self.config))
        data.pop("live", None)
        return data

    def prepare_records(self) -> list[dict[str, Any]]:
        records = []
        for case, treatment, evaluator in self.jobs:
            bundle = prompt_bundle(case, treatment, evaluator)
            leaks = prompt_leaks_evaluator(bundle["prompt"], evaluator)
            if treatment != "reference" and leaks:
                raise RuntimeError(f"evaluator leak in {case.case_id}/{treatment}: {leaks}")
            records.append(
                {
                    "case": case,
                    "treatment": treatment,
                    "evaluator": evaluator,
                    "bundle": bundle,
                    "key": job_key(
                        case.case_id,
                        treatment,
                        self.effective_hash(),
                        bundle["prompt_hash"],
                    ),
                }
            )
        return records

    def write_dry_run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records = self.prepare_records()
        seen = set()
        unique_cases = []
        for item in records:
            cid = item["case"].case_id
            if cid in seen:
                continue
            seen.add(cid)
            unique_cases.append(item["case"].public_dict())
        evaluator = {
            item["case"].case_id: item["evaluator"].as_dict()
            for item in records
            if item["treatment"] == self.spec.treatments[0]
        }
        prompts_dir = self.output_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        prompt_index = []
        for item in records:
            bundle = item["bundle"]
            name = f"{item['case'].case_id}_{item['treatment']}.txt"
            (prompts_dir / name).write_text(bundle["prompt"], encoding="utf-8")
            public_only = {
                key: bundle[key]
                for key in (
                    "case_id",
                    "treatment",
                    "prompt_hash",
                    "case_hash",
                    "public_inputs",
                    "token_estimate",
                )
            }
            prompt_index.append(public_only)
        n_jobs = len(records)
        max_retries = int(self.config.get("max_retries", 1))
        max_input = int(self.config.get("max_input_tokens", DEFAULT_MAX_INPUT))
        max_output = int(self.config.get("max_output_tokens", DEFAULT_MAX_OUTPUT))
        pricing = verified_pricing(self.config, self.model)
        planned_attempts = n_jobs * (1 + max_retries)
        overflow = [
            item["bundle"]["token_estimate"]
            for item in records
            if item["bundle"]["token_estimate"] > max_input
        ]
        code = snapshot_code(self.repo_root)
        thinking = int(self.config.get("thinking_budget", DEFAULT_THINKING_BUDGET))
        billed_limit = max_output + thinking
        reserved_each = reserved_cost_usd(
            max_input_tokens=max_input,
            max_output_tokens=billed_limit,
            model=self.model,
            pricing=pricing,
        )
        manifest = {
            "schema_version": 2,
            "kind": "selective_revision_fixed_history_pilot_dryrun",
            "model": self.model,
            "config_hash": self.config_hash,
            "effective_hash": self.effective_hash(),
            "n_public_cases": len(unique_cases),
            "n_treatments": len(self.spec.treatments),
            "n_planned_calls": n_jobs,
            "n_planned_attempts_with_retry_reserve": planned_attempts,
            "max_input_tokens": max_input,
            "max_output_tokens": max_output,
            "provider_max_output_tokens": billed_limit,
            "thinking_budget": thinking,
            "token_estimate_kind": "offline_heuristic",
            "pricing": pricing,
            "reserved_usd_per_attempt": reserved_each,
            "reserved_usd_full_grid": planned_attempts * reserved_each,
            "prompt_token_estimate_max": max(item["bundle"]["token_estimate"] for item in records),
            "prompts_exceeding_input_cap": len(overflow),
            "code": code,
            "limitation": (
                "Cost bound is reserved_per_attempt * jobs * (1 + retries) using "
                "max_input_tokens at the input rate and (max_output_tokens + thinking_budget) "
                "at the output rate. thinking_budget=-1 is rejected. Dry-run made no model calls."
            ),
        }
        atomic_write_json(self.output_dir / "manifest.json", manifest)
        atomic_write_json(self.output_dir / "public_cases.json", unique_cases)
        atomic_write_json(self.output_dir / "evaluator.json", evaluator)
        atomic_write_json(self.output_dir / "prompt_index.json", prompt_index)
        atomic_write_json(self.output_dir / "config_snapshot.json", self._hashable_config())
        public_report = {
            "manifest": manifest,
            "jobs": [
                {
                    "key": item["key"],
                    "case_id": item["case"].case_id,
                    "treatment": item["treatment"],
                    "prompt_hash": item["bundle"]["prompt_hash"],
                    "case_hash": item["bundle"]["case_hash"],
                    "token_estimate": item["bundle"]["token_estimate"],
                }
                for item in records
            ],
        }
        atomic_write_json(self.output_dir / "dryrun.json", public_report)
        markdown = render_dryrun_markdown(manifest, unique_cases, records, self.output_dir)
        (self.output_dir / "dryrun.md").write_text(markdown, encoding="utf-8")
        return public_report


def render_dryrun_markdown(
    manifest: Mapping[str, Any],
    cases: list[dict[str, Any]],
    records: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    lines = [
        "# Selective revision pilot: dry-run",
        "",
        "No LLM calls, no `.env` read, no network. This is the fixed-history diagnostic",
        "package, not a model evaluation and not a performance benchmark.",
        "",
        "## Grid",
        "",
        f"- Public cases: {manifest['n_public_cases']} (3 structures × 4 A-success counts × 2 diagnostic costs).",
        f"- Treatments: raw, counts, counts-plus-reference predictions ({manifest['n_treatments']}).",
        f"- Planned decision calls: **{manifest['n_planned_calls']}**.",
        f"- Retry reserve (one retry per call): **{manifest['n_planned_attempts_with_retry_reserve']}** attempts.",
        f"- Requested model ID: `{manifest['model']}`.",
        "",
        "A diagnose choice commits to using the revealed latent rate optimally. This probe",
        "tests valuation of information, not interpretation of diagnostic results.",
        "",
        "## Cost bound",
        "",
        "Each attempt is reserved at `max_input_tokens` input plus the **provider**",
        "output limit (`configured_max_output_tokens + thinking_budget`) at the output",
        "rate. 2048 is not an independently enforced visible-text cap. Interrupted",
        "calls keep their reservation. Dry-run token figures are an offline heuristic.",
        "",
        f"- Input cap: {manifest['max_input_tokens']} tokens",
        f"- Configured max_output_tokens: {manifest['max_output_tokens']}",
        f"- Provider output limit (sum): {manifest.get('provider_max_output_tokens', manifest['max_output_tokens'])}",
        f"- Rates: ${manifest['pricing']['input_per_1m']}/M input, ${manifest['pricing']['output_per_1m']}/M output",
        f"- Reserved per attempt: ${manifest['reserved_usd_per_attempt']:.6f}",
        f"- Reserved for full grid plus retries: **${manifest['reserved_usd_full_grid']:.6f}**",
        f"- Largest prompt token estimate: {manifest['prompt_token_estimate_max']}",
        f"- Prompts exceeding the input cap: {manifest['prompts_exceeding_input_cap']}",
        "",
        "## Integrity",
        "",
        f"- Config hash: `{manifest['config_hash']}`",
        f"- Effective hash: `{manifest.get('effective_hash')}`",
        f"- Git HEAD: `{manifest['code'].get('git_head')}`",
        f"- Dirty files hashed: {len(manifest['code'].get('dirty_file_hashes') or {})} (credentials omitted)",
        "- Public prompts are under `prompts/`. Evaluator targets are in `evaluator.json` and are not sent to the provider.",
        "- Resume key: `case_id|treatment|effective_hash|prompt_hash`.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync python scripts/selective_revision_pilot.py --config llm_rl_engine/configs/pilot_selective_revision_v1.yaml",
        "UV_CACHE_DIR=/tmp/llm-time-uv-cache uv run --no-sync pytest -q tests/test_selective_revision_pilot.py",
        "```",
        "",
        f"Artifacts: `{output_dir}`",
        "",
        "## Limitations",
        "",
        manifest["limitation"],
        "",
        "Labels and rates are held fixed. This grid is not structural-transfer evidence.",
        "Live execution requires a follow-up with explicit model, max_calls, token caps,",
        "and max_usd. No claim of model performance is made here.",
        "",
    ]
    return "\n".join(lines)


def completed_keys(results_path: Path) -> set[str]:
    return {row["key"] for row in load_jsonl(results_path) if row.get("status") == "complete"}


def run_jobs(
    run: PilotRun,
    settings: LiveSettings,
    client: PilotClient,
    *,
    smoke_limit: int | None = None,
    resume: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    run.bind_settings(settings)
    check_or_init_identity(run.output_dir, run.identity, settings)
    records = run.prepare_records()
    if smoke_limit is not None:
        if smoke_limit > 3:
            raise BudgetError("smoke is at most 3 calls")
        records = records[:smoke_limit]
    prompts = [item["bundle"]["prompt"] for item in records]
    token_cache: dict[str, tuple[int, str]] = {}
    preflight_info = preflight(
        len(records), settings, prompts, client=client, token_cache=token_cache
    )
    ledger_path = run.output_dir / "ledger.json"
    attempts_path = run.output_dir / "attempts.jsonl"
    results_path = run.output_dir / "results.jsonl"
    manifest_path = run.output_dir / "live_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "config_hash": run.config_hash,
            "effective_hash": run.effective_hash(),
            "model": settings.model,
            "scientific_identity": run.identity,
            "preflight": preflight_info,
            "code": snapshot_code(run.repo_root),
        },
    )
    ledger = load_ledger(ledger_path)
    absorb_dangling_reservations(ledger)
    atomic_write_json(ledger_path, ledger)
    prior_attempts = load_jsonl(attempts_path)
    done = completed_keys(results_path) if resume else set()
    if not resume and (results_path.exists() or attempts_path.exists()):
        raise BudgetError("refusing to reset an existing run; pass resume=True or a new output directory")
    scores = []
    n_done = sum(1 for item in records if item["key"] in done)
    bar = tqdm(
        total=len(records),
        initial=n_done,
        desc="pilot",
        unit="job",
        disable=not progress,
        file=sys.stderr,
        dynamic_ncols=True,
    )
    try:
        for item in records:
            key = item["key"]
            if key in done:
                continue
            score = _run_one_job(
                item,
                run=run,
                settings=settings,
                client=client,
                ledger=ledger,
                ledger_path=ledger_path,
                attempts_path=attempts_path,
                results_path=results_path,
                prior_attempts=prior_attempts,
            )
            prior_attempts = load_jsonl(attempts_path)
            scores.append(score)
            all_so_far = load_jsonl(results_path)
            summary = summarize_scores(all_so_far)
            snapshot = {
                "summary": summary,
                "ledger": ledger,
                "last": {
                    "case_id": score.get("case_id"),
                    "treatment": score.get("treatment"),
                    "valid": score.get("valid"),
                    "chosen_action": score.get("chosen_action"),
                    "optimal_action": score.get("optimal_action"),
                    "decision_error": score.get("decision_error"),
                    "failure_kind": score.get("failure_kind"),
                    "decision_loss": score.get("decision_loss"),
                },
                "n_results": len(all_so_far),
                "n_jobs": len(records),
                "updated_at": _utc_now(),
            }
            atomic_write_json(run.output_dir / "summary.json", snapshot)
            atomic_write_json(run.output_dir / "progress.json", snapshot)
            bar.set_postfix(
                spent=f"{float(ledger.get('spent_usd') or 0):.3f}",
                valid=summary.get("n_valid"),
                err=summary.get("n_decision_error"),
                parse=summary.get("n_parse_failure"),
                refresh=False,
            )
            bar.update(1)
            if progress:
                line = (
                    f"{score.get('case_id')} {score.get('treatment')} "
                    f"valid={score.get('valid')} chose={score.get('chosen_action')} "
                    f"opt={score.get('optimal_action')} loss={score.get('decision_loss')} "
                    f"fail={score.get('failure_kind')} spent={ledger.get('spent_usd')}"
                )
                tqdm.write(line, file=sys.stderr)
            if score.get("stop_run"):
                break
    finally:
        bar.close()
    all_results = load_jsonl(results_path)
    summary = summarize_scores(all_results)
    atomic_write_json(
        run.output_dir / "summary.json",
        {"summary": summary, "ledger": ledger, "n_results": len(all_results)},
    )
    return {"summary": summary, "ledger": ledger, "n_results": len(all_results), "preflight": preflight_info}


def absorb_dangling_reservations(ledger: dict[str, Any]) -> None:
    """Treat in-flight reservations as spent. Unknown usage is not assumed free."""
    dangling = float(ledger.get("reserved_usd") or 0.0)
    if dangling > 0:
        ledger["spent_usd"] = float(ledger.get("spent_usd") or 0.0) + dangling
        ledger["interrupted_reservations"] = int(ledger.get("interrupted_reservations") or 0) + 1
        ledger["reserved_usd"] = 0.0


def _reserved_attempts_for_key(attempts: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in attempts if row.get("key") == key and row.get("status") == "reserved")


def _score_generation(
    result: GenerationResult,
    *,
    evaluator: EvaluatorRecord,
    spec: PilotSpec,
    settings: LiveSettings,
) -> dict[str, Any]:
    finish = getattr(result, "finish_reason", None)
    finish_available = finish is not None
    incomplete = finish_is_incomplete(finish, finish_available)
    parsed = None
    valid = False
    failure_kind = None
    if incomplete:
        failure_kind = "incomplete_finish"
        try:
            parsed = parse_response(result.text)
        except ValidationError:
            parsed = None
    else:
        try:
            parsed = parse_response(result.text)
            valid = True
        except ValidationError as exc:
            failure_kind = "parse_failure"
            parsed = None
            last = str(exc)
            scored = score_response(
                None, evaluator, valid=False, failure_kind=failure_kind, spec=spec
            )
            scored["parse_error"] = last
            scored["finish_reason"] = finish
            scored["finish_reason_status"] = "available" if finish_available else "unavailable"
            scored["model_version"] = getattr(result, "model_version", None) or "unavailable"
            scored["usage_status"] = getattr(result, "usage_status", None) or "unknown"
            return scored
    scored = score_response(
        parsed, evaluator, valid=valid, failure_kind=failure_kind, spec=spec
    )
    scored["parse_error"] = None if valid else scored.get("parse_error")
    scored["finish_reason"] = finish
    scored["finish_reason_status"] = "available" if finish_available else "unavailable"
    scored["model_version"] = getattr(result, "model_version", None) or "unavailable"
    scored["usage_status"] = getattr(result, "usage_status", None) or "unknown"
    return scored


def _run_one_job(
    item: dict[str, Any],
    *,
    run: PilotRun,
    settings: LiveSettings,
    client: PilotClient,
    ledger: dict[str, Any],
    ledger_path: Path,
    attempts_path: Path,
    results_path: Path,
    prior_attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    case: PublicCase = item["case"]
    treatment: str = item["treatment"]
    evaluator: EvaluatorRecord = item["evaluator"]
    bundle = item["bundle"]
    prompt = bundle["prompt"]
    key = item["key"]
    parsed = None
    failure_kind = None
    last_error = None
    stop_run = False
    valid = False
    attempts_used = 0
    last_scored: dict[str, Any] | None = None
    settings_payload = generation_settings_payload(settings)
    saved = find_unscored_response(run.output_dir, case.case_id, treatment)
    if saved and saved.get("raw_response") is not None:
        result = result_from_saved(saved)
        scored = _score_generation(
            result,
            evaluator=evaluator,
            spec=run.spec,
            settings=settings,
        )
        row = {
            "key": key,
            "status": "complete",
            "case_id": case.case_id,
            "treatment": treatment,
            "model": settings.model,
            "prompt_hash": bundle["prompt_hash"],
            "config_hash": run.config_hash,
            "effective_hash": run.effective_hash(),
            "request_hash": saved.get("request_hash"),
            "recovered_from_saved_response": True,
            "attempts_used": int(saved.get("attempt_n") or 1),
            **scored,
            "stop_run": False,
        }
        append_jsonl(results_path, row)
        ledger["completed_jobs"] += 1
        atomic_write_json(ledger_path, ledger)
        saved["scored"] = True
        path = Path(saved["_path"])
        atomic_write_json(path, {k: v for k, v in saved.items() if k != "_path"})
        return row
    already = _reserved_attempts_for_key(prior_attempts or [], key)
    remaining_tries = settings.max_retries + 1 - already
    if remaining_tries <= 0:
        scored = score_response(
            None, evaluator, valid=False, failure_kind="api_failure", spec=run.spec
        )
        row = {
            "key": key,
            "status": "complete",
            "case_id": case.case_id,
            "treatment": treatment,
            "model": settings.model,
            "prompt_hash": bundle["prompt_hash"],
            "config_hash": run.config_hash,
            "effective_hash": run.effective_hash(),
            "attempts_used": already,
            **scored,
            "stop_run": False,
        }
        append_jsonl(results_path, row)
        ledger["completed_jobs"] += 1
        atomic_write_json(ledger_path, ledger)
        return row
    for retry_index in range(already, already + remaining_tries):
        room_calls = settings.max_calls - ledger["attempts"]
        room_usd = settings.max_usd - ledger["spent_usd"] - ledger["reserved_usd"]
        if room_calls < 1 or room_usd + 1e-15 < settings.reserved_per_attempt:
            raise BudgetError(
                f"pre-call cap hit before {key}: calls left={room_calls}, "
                f"usd room={room_usd}, reserved_per_attempt={settings.reserved_per_attempt}"
            )
        current_prompt = prompt
        if retry_index > 0 and last_error:
            current_prompt = prompt + REPAIR_SUFFIX.format(error=last_error)
            if estimate_tokens(current_prompt) > settings.max_input_tokens:
                failure_kind = "parse_failure"
                break
        attempt_id = f"{key}|attempt{ledger['attempts']}"
        reservation = {
            "attempt_id": attempt_id,
            "key": key,
            "status": "reserved",
            "case_id": case.case_id,
            "treatment": treatment,
            "retry_index": retry_index,
            "reserved_usd": settings.reserved_per_attempt,
            "started_at": _utc_now(),
        }
        ledger["reserved_usd"] += settings.reserved_per_attempt
        ledger["attempts"] += 1
        attempts_used += 1
        atomic_write_json(ledger_path, ledger)
        append_jsonl(attempts_path, reservation)
        started = time.perf_counter()
        request_hash = sha256_text(current_prompt)
        try:
            result = client.generate(current_prompt)
            latency = time.perf_counter() - started
            try:
                persist_provider_response(
                    run.output_dir,
                    case_id=case.case_id,
                    treatment=treatment,
                    attempt_n=ledger["attempts"],
                    attempt_id=attempt_id,
                    result=result,
                    latency_s=latency,
                    request_text=current_prompt,
                    request_hash=request_hash,
                    generation_settings=settings_payload,
                    job_key_value=key,
                    scored=False,
                )
                append_jsonl(
                    run.output_dir / "requests.jsonl",
                    {
                        "attempt_id": attempt_id,
                        "request_hash": request_hash,
                        "generation_settings": settings_payload,
                        "request_text": current_prompt,
                    },
                )
            except Exception as exc:
                raise BookkeepingError(
                    f"provider returned but persistence failed: {exc}"
                ) from exc
            usage_status = getattr(result, "usage_status", None) or "unknown"
            if usage_status == "unknown":
                actual = settings.reserved_per_attempt
            else:
                actual = estimate_cost_usd(
                    settings.model,
                    result.input_tokens,
                    result.output_tokens,
                    {settings.model: settings.pricing},
                )["total_cost_usd"]
                if result.input_tokens > settings.max_input_tokens:
                    raise BudgetError("provider reported input tokens above cap")
            ledger["reserved_usd"] -= settings.reserved_per_attempt
            ledger["spent_usd"] += actual
            completion = {
                **reservation,
                "status": "attempted",
                "finished_at": _utc_now(),
                "latency_s": latency,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "cost_usd": actual,
                "usage_status": usage_status,
                "request_hash": request_hash,
                "exception_category": None,
                "raw_response": result.text,
                "model_version": getattr(result, "model_version", None) or "unavailable",
                "finish_reason": getattr(result, "finish_reason", None),
            }
            thoughts = int(getattr(result, "thoughts_tokens", 0) or 0)
            completion["thoughts_tokens"] = thoughts
            if usage_status != "unknown" and result.output_tokens > settings.billed_output_limit:
                completion["exception_category"] = "output_cap"
                append_jsonl(attempts_path, completion)
                atomic_write_json(ledger_path, ledger)
                raise BudgetError(
                    "billed output exceeded thinking_budget + max_output_tokens; "
                    "dollar ceiling is no longer valid"
                )
            scored_gen = _score_generation(
                result, evaluator=evaluator, spec=run.spec, settings=settings
            )
            last_scored = scored_gen
            valid = bool(scored_gen.get("valid"))
            failure_kind = scored_gen.get("failure_kind")
            last_error = scored_gen.get("parse_error")
            completion["parse_error"] = last_error
            append_jsonl(attempts_path, completion)
            atomic_write_json(ledger_path, ledger)
            if valid:
                break
        except (BudgetError, BookkeepingError, IdentityError):
            raise
        except Exception as exc:
            latency = time.perf_counter() - started
            category = classify_exception(exc)
            ledger["reserved_usd"] -= settings.reserved_per_attempt
            ledger["spent_usd"] += settings.reserved_per_attempt
            ledger["interrupted_reservations"] += 1
            atomic_write_json(ledger_path, ledger)
            append_jsonl(
                attempts_path,
                {
                    **reservation,
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "latency_s": latency,
                    "exception_category": category,
                    "error": f"{type(exc).__name__}: {exc}",
                    "cost_usd": settings.reserved_per_attempt,
                },
            )
            failure_kind = "api_failure"
            last_error = str(exc)
            parsed = None
            valid = False
            if category in {"authentication", "quota"}:
                stop_run = True
                break
            # retry only once and only for non-auth/quota failures, still inside cap
            continue

    scored = last_scored or score_response(
        parsed, evaluator, valid=valid, failure_kind=failure_kind, spec=run.spec
    )
    row = {
        "key": key,
        "status": "complete",
        "case_id": case.case_id,
        "treatment": treatment,
        "model": settings.model,
        "prompt_hash": bundle["prompt_hash"],
        "config_hash": run.config_hash,
        "effective_hash": run.effective_hash(),
        "attempts_used": attempts_used,
        **scored,
        "stop_run": stop_run,
    }
    append_jsonl(results_path, row)
    ledger["completed_jobs"] += 1
    atomic_write_json(ledger_path, ledger)
    leftover = find_unscored_response(run.output_dir, case.case_id, treatment)
    if leftover:
        leftover["scored"] = True
        atomic_write_json(
            Path(leftover["_path"]),
            {k: v for k, v in leftover.items() if k != "_path"},
        )
    return row


def load_pilot_config(path: str | Path) -> tuple[dict[str, Any], PilotSpec]:
    config = load_config(path)
    spec = spec_from_mapping(config)
    if "model" not in config:
        config["model"] = DEFAULT_MODEL
    if "max_input_tokens" not in config:
        config["max_input_tokens"] = DEFAULT_MAX_INPUT
    if "max_output_tokens" not in config:
        config["max_output_tokens"] = DEFAULT_MAX_OUTPUT
    if "thinking_budget" not in config:
        config["thinking_budget"] = DEFAULT_THINKING_BUDGET
    if "max_retries" not in config:
        config["max_retries"] = 1
    return config, spec
