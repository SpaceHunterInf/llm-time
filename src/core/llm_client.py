"""Unified cost-metered LLM client supporting Gemini (`google-genai`) and offline Mock."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.ledger import CostLedger
from src.core.pricing import normalize_model_id


@dataclass
class LLMResponse:
    text: str
    parsed_json: dict[str, Any] | None
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    usd: float
    latency_ms: int
    model_version: str = ""


def load_dotenv_into_environ(dotenv_path: Path) -> None:
    """Load `.env` key-values into `os.environ` without logging or exposing secrets."""
    resolved = dotenv_path.resolve()
    if not resolved.is_file():
        return
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        clean_key = key.strip()
        if clean_key and clean_key not in os.environ:
            os.environ[clean_key] = val.strip().strip('"').strip("'")


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Robustly parse a JSON object from raw or fenced model output."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(stripped[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


class MeteredLLMClient:
    """Wraps Gemini API calls with mandatory `CostLedger` accounting and budget circuit-breaking."""

    def __init__(
        self,
        *,
        model: str,
        ledger: CostLedger,
        dotenv_path: Path | None = None,
        thinking_budget: int | None = 512,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> None:
        self.model = normalize_model_id(model)
        self.ledger = ledger
        self.thinking_budget = thinking_budget
        self.temperature = temperature
        self.max_retries = max_retries
        self._gemini_client: Any = None
        self._plain_text_fallback: bool = "gemma" in self.model.lower()
        self.observed_model_versions: set[str] = set()
        self._tpm_lock = threading.Lock()
        self._tpm_window: deque[tuple[float, int]] = deque()
        # Google Paid Tier 3 caps hosted Gemma-4 at 16,000 input TPM; pace conservatively to 13,200 TPM
        self._input_tpm_cap: int = 13_200 if "gemma" in self.model.lower() else 0

        if dotenv_path is not None:
            load_dotenv_into_environ(dotenv_path)

        if self.model != "mock":
            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is missing from environment / .env.")
            from google import genai

            self._gemini_client = genai.Client(api_key=api_key)

    def _pace_input_tpm(self, est_input_tokens: int) -> None:
        if self._input_tpm_cap <= 0:
            return
        need = max(200, min(self._input_tpm_cap - 500, int(est_input_tokens)))
        while True:
            with self._tpm_lock:
                now = time.time()
                while self._tpm_window and (now - self._tpm_window[0][0]) >= 61.0:
                    self._tpm_window.popleft()
                used = sum(tok for _, tok in self._tpm_window)
                if used + need <= self._input_tpm_cap:
                    self._tpm_window.append((now, need))
                    return
                oldest_ts = self._tpm_window[0][0] if self._tpm_window else now
                wait_s = max(1.5, 61.2 - (now - oldest_ts))
            time.sleep(min(wait_s, 15.0))

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        environment: str,
        method: str,
        mock_responder: Any = None,
    ) -> LLMResponse:
        if self.model == "mock":
            if mock_responder is None:
                payload = {"tool": "finish", "args": {}, "rationale": "mock"}
            else:
                payload = mock_responder(system_prompt, user_prompt)
            raw = json.dumps(payload)
            entry = self.ledger.record_call(
                environment=environment,
                method=method,
                model="mock",
                input_tokens=max(1, len(system_prompt + user_prompt) // 4),
                output_tokens=max(1, len(raw) // 4),
                thinking_tokens=0,
                latency_ms=1,
                status="ok",
            )
            return LLMResponse(
                text=raw,
                parsed_json=payload,
                input_tokens=entry.input_tokens,
                output_tokens=entry.output_tokens,
                thinking_tokens=0,
                usd=entry.usd,
                latency_ms=1,
            )

        from google.genai import types

        last_err: Exception | None = None
        eff_retries = max(self.max_retries, 6 if "gemma" in self.model.lower() else 4)
        for attempt in range(eff_retries + 1):
            est_in = max(100, int(len(system_prompt + user_prompt) / 3.4))
            self._pace_input_tpm(est_in)
            reserved = self.ledger.acquire_reservation(reserve_usd=0.008)
            t0 = time.time()
            try:
                if self._plain_text_fallback:
                    cfg_kwargs = {
                        "temperature": self.temperature,
                        "max_output_tokens": 2048,
                    }
                    contents_payload = f"{system_prompt}\n\n{user_prompt}"
                else:
                    cfg_kwargs = {
                        "system_instruction": system_prompt,
                        "temperature": self.temperature,
                        "response_mime_type": "application/json",
                        "max_output_tokens": 2048,
                    }
                    if self.thinking_budget is not None and "flash-lite" not in self.model and "gemma" not in self.model:
                        cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                            thinking_budget=self.thinking_budget
                        )
                    elif self.thinking_budget is not None and "gemma-4" in self.model:
                        cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                            thinking_budget=self.thinking_budget
                        )
                    contents_payload = user_prompt
                config = types.GenerateContentConfig(**cfg_kwargs)
                resp = self._gemini_client.models.generate_content(
                    model=self.model,
                    contents=contents_payload,
                    config=config,
                )
                latency_ms = int((time.time() - t0) * 1000)
                m_ver = str(getattr(resp, "model_version", "") or self.model)
                self.observed_model_versions.add(m_ver)
                usage = getattr(resp, "usage_metadata", None)
                in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
                out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
                think_tok = int(getattr(usage, "thoughts_token_count", 0) or 0)
                cached_tok = int(getattr(usage, "cached_content_token_count", 0) or 0)
                text = getattr(resp, "text", "") or ""
                parsed = extract_json_object(text)
                status = "ok" if parsed is not None else ("retry" if attempt < eff_retries else "error")
                entry = self.ledger.record_call(
                    environment=environment,
                    method=method,
                    model=self.model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    thinking_tokens=think_tok,
                    cached_tokens=cached_tok,
                    latency_ms=latency_ms,
                    status=status,
                    model_version=m_ver,
                )
                if parsed is not None or attempt == eff_retries:
                    return LLMResponse(
                        text=text,
                        parsed_json=parsed,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        thinking_tokens=think_tok,
                        usd=entry.usd,
                        latency_ms=latency_ms,
                        model_version=m_ver,
                    )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                err_msg = str(exc).lower()
                if not self._plain_text_fallback and any(
                    k in err_msg for k in ("developer instruction", "mime", "thinking", "invalid_argument", "400")
                ):
                    self._plain_text_fallback = True
                latency_ms = int((time.time() - t0) * 1000)
                self.ledger.record_call(
                    environment=environment,
                    method=method,
                    model=self.model,
                    input_tokens=0,
                    output_tokens=0,
                    thinking_tokens=0,
                    latency_ms=latency_ms,
                    status="error" if attempt == eff_retries else "retry",
                    model_version=self.model,
                )
                if attempt < eff_retries:
                    if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
                        m_delay = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", err_msg)
                        sleep_s = (float(m_delay.group(1)) + 3.0) if m_delay else 46.0
                        time.sleep(min(65.0, max(15.0, sleep_s)))
                    else:
                        time.sleep(1.5 * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM request failed after retries: {type(exc).__name__}") from last_err
            finally:
                self.ledger.release_reservation(reserved)

        raise RuntimeError("Unreachable LLM retry termination.")
