"""Thread-safe local HuggingFace GPU client (`Qwen/Qwen3-8B` FP16) with `CostLedger` token metering."""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from src.core.ledger import CostLedger
from src.core.llm_client import LLMResponse, extract_json_object

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_GLOBAL_MODEL_CACHE: dict[str, tuple[Any, Any, threading.Lock]] = {}
_CACHE_LOCK = threading.Lock()


def _get_or_load_hf_model(model_id: str, device: str = "cuda:0") -> tuple[Any, Any, threading.Lock]:
    with _CACHE_LOCK:
        if model_id in _GLOBAL_MODEL_CACHE:
            return _GLOBAL_MODEL_CACHE[model_id]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map=device,
        )
        mdl.eval()
        entry = (tok, mdl, threading.Lock())
        _GLOBAL_MODEL_CACHE[model_id] = entry
        return entry


class LocalHFMeteredClient:
    """Drop-in `MeteredLLMClient`-compatible interface backed by a local open-weight HuggingFace model."""

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-8B",
        ledger: CostLedger,
        device: str = "cuda:0",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        sample_seed: int | None = None,
    ) -> None:
        self.model = model
        self.ledger = ledger
        self.device = device
        self.max_new_tokens = max(64, int(max_new_tokens))
        self.temperature = float(temperature)
        self.sample_seed = sample_seed
        self.observed_model_versions: set[str] = {model}

    def with_sampling(self, *, temperature: float, sample_seed: int) -> "LocalHFMeteredClient":
        """Return a lightweight view sharing the loaded GPU weights with a deterministic per-trial seed."""
        return LocalHFMeteredClient(
            model=self.model,
            ledger=self.ledger,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            sample_seed=sample_seed,
        )

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
            payload = mock_responder(system_prompt, user_prompt) if mock_responder else {"decision": "KEEP"}
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
                usd=0.0,
                latency_ms=1,
                model_version="mock",
            )

        import torch

        tok, mdl, gpu_lock = _get_or_load_hf_model(self.model, self.device)
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            prompt_str = tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        t0 = time.time()
        with gpu_lock:
            inp = tok(prompt_str, return_tensors="pt").to(mdl.device)
            in_len = int(inp["input_ids"].shape[1])
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": tok.eos_token_id,
            }
            if self.temperature > 1e-5:
                if self.sample_seed is not None:
                    torch.cuda.manual_seed_all(int(self.sample_seed))
                    torch.manual_seed(int(self.sample_seed))
                gen_kwargs.update({"do_sample": True, "temperature": self.temperature, "top_p": 0.95})
            else:
                gen_kwargs.update({"do_sample": False})

            with torch.no_grad():
                out = mdl.generate(**inp, **gen_kwargs)
            gen_ids = out[0][in_len:]
            out_len = int(gen_ids.shape[0])
            raw_txt = tok.decode(gen_ids, skip_special_tokens=True).strip()

        latency_ms = int((time.time() - t0) * 1000)
        cleaned = _THINK_RE.sub("", raw_txt).strip()
        parsed = extract_json_object(cleaned)
        entry = self.ledger.record_call(
            environment=environment,
            method=method,
            model=self.model,
            input_tokens=in_len,
            output_tokens=out_len,
            thinking_tokens=0,
            latency_ms=latency_ms,
            status="ok",
            model_version=self.model,
        )
        return LLMResponse(
            text=cleaned,
            parsed_json=parsed,
            input_tokens=in_len,
            output_tokens=out_len,
            thinking_tokens=0,
            usd=entry.usd,
            latency_ms=latency_ms,
            model_version=self.model,
        )
