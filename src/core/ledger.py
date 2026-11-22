"""Append-only global API cost ledger, run progress tracker, and email alerts."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.pricing import estimate_call_usd, normalize_model_id

TOTAL_GEMINI_CREDIT_POOL_USD = 70_000.0
ALLOWED_MONITOR_RECIPIENTS = frozenset({"gs06monitor@gmail.com"})
MAILX_BINARY = "/usr/bin/mailx"


class BudgetExceededError(RuntimeError):
    """Raised when a run attempts to exceed its configured `max_usd` cap."""


@dataclass
class LedgerEntry:
    timestamp_utc: str
    run_id: str
    phase: str
    environment: str
    method: str
    model: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cached_tokens: int
    usd: float
    pricing_verified: bool
    latency_ms: int
    status: str  # "ok" | "retry" | "error"
    model_version: str = ""


def mask_email_for_display(email: str) -> str:
    """Mask PII email address for safe UI rendering (e.g. `gs***@gmail.com`)."""
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    prefix = local[:2] if len(local) >= 2 else local[:1]
    return f"{prefix}***@{domain}"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


class CostLedger:
    """Thread-safe append-only project cost ledger and run circuit breaker."""

    def __init__(
        self,
        results_root: Path,
        run_id: str,
        phase: str = "dev",
        max_usd: float = 5.0,
    ) -> None:
        self.results_root = results_root.resolve()
        self.ledger_dir = self.results_root / "ledger"
        self.ledger_file = self.ledger_dir / "api_ledger.jsonl"
        self.summary_file = self.ledger_dir / "summary.json"
        self.run_dir = self.results_root / "runs" / self._sanitize_id(run_id)
        self.progress_file = self.run_dir / "progress.json"

        self.run_id = self._sanitize_id(run_id)
        self.phase = phase
        self.max_usd = float(max_usd)
        self.started_at = time.time()
        self._lock = threading.Lock()

        self.run_usd = 0.0
        self.in_flight_usd = 0.0
        self.run_calls = 0
        self.run_errors = 0
        self.run_input_tokens = 0
        self.run_output_tokens = 0
        self.run_thinking_tokens = 0

        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.summary_file.exists():
            self.rebuild_global_summary()

    @staticmethod
    def _sanitize_id(raw: str) -> str:
        safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in raw.strip())
        return safe[:96] or "unnamed_run"

    def check_budget(self, reserve_usd: float = 0.005) -> None:
        with self._lock:
            if self.run_usd + self.in_flight_usd + reserve_usd > self.max_usd:
                raise BudgetExceededError(
                    f"Run '{self.run_id}' reached budget guardrail: "
                    f"spent=${self.run_usd:.4f} + in_flight=${self.in_flight_usd:.4f} + "
                    f"reserve=${reserve_usd:.4f} > max_usd=${self.max_usd:.2f}"
                )

    def acquire_reservation(self, reserve_usd: float = 0.004) -> float:
        """Atomically reserve `reserve_usd` for an in-flight concurrent API request."""
        amt = max(0.0, float(reserve_usd))
        with self._lock:
            if self.run_usd + self.in_flight_usd + amt > self.max_usd:
                raise BudgetExceededError(
                    f"Run '{self.run_id}' concurrent budget guardrail: "
                    f"spent=${self.run_usd:.4f} + in_flight=${self.in_flight_usd:.4f} + "
                    f"reserve=${amt:.4f} > max_usd=${self.max_usd:.2f}"
                )
            self.in_flight_usd += amt
        return amt

    def release_reservation(self, reserved_usd: float) -> None:
        with self._lock:
            self.in_flight_usd = max(0.0, self.in_flight_usd - max(0.0, float(reserved_usd)))

    def record_call(
        self,
        *,
        environment: str,
        method: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int = 0,
        cached_tokens: int = 0,
        latency_ms: int = 0,
        status: str = "ok",
        model_version: str = "",
    ) -> LedgerEntry:
        norm_model = normalize_model_id(model)
        usd, verified = estimate_call_usd(
            norm_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            cached_tokens=cached_tokens,
        )
        entry = LedgerEntry(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            run_id=self.run_id,
            phase=self.phase,
            environment=environment,
            method=method,
            model=norm_model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            thinking_tokens=int(thinking_tokens),
            cached_tokens=int(cached_tokens),
            usd=round(usd, 8),
            pricing_verified=verified,
            latency_ms=int(latency_ms),
            status=status,
            model_version=str(model_version or norm_model),
        )
        with self._lock:
            with open(self.ledger_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(entry)) + "\n")
            self.run_usd += usd
            self.run_calls += 1
            if status != "ok":
                self.run_errors += 1
            self.run_input_tokens += int(input_tokens)
            self.run_output_tokens += int(output_tokens)
            self.run_thinking_tokens += int(thinking_tokens)
            self._update_summary_unlocked(entry)
        return entry

    @staticmethod
    def format_progress_bar(
        completed: int,
        total: int,
        elapsed_sec: float,
        eta_sec: float | None,
        run_usd: float = 0.0,
        width: int = 20,
    ) -> str:
        tot = max(1, int(total))
        comp = max(0, min(int(completed), tot))
        pct = 100.0 * comp / tot
        filled = int(round((pct / 100.0) * width))
        bar = "[" + ("#" * filled) + ("-" * (width - filled)) + "]"
        el_m, el_s = divmod(int(round(elapsed_sec)), 60)
        if isinstance(eta_sec, (int, float)) and eta_sec > 0:
            eta_m, eta_s = divmod(int(round(eta_sec)), 60)
            eta_txt = f"~{eta_m}m{eta_s:02d}s"
        else:
            eta_txt = "0m00s" if comp >= tot else "..."
        return (
            f"{bar} {pct:5.1f}% ({comp}/{tot}) | "
            f"elapsed {el_m}m{el_s:02d}s | ETA {eta_txt} | ${run_usd:.4f}"
        )

    def update_progress(
        self,
        *,
        completed_units: int,
        total_units: int,
        status: str = "running",
        current_label: str = "",
        extra: dict[str, Any] | None = None,
        log_every: int = 0,
    ) -> None:
        elapsed = max(0.001, time.time() - self.started_at)
        eta_sec = None
        if completed_units > 0 and total_units > completed_units:
            rate = completed_units / elapsed
            eta_sec = round((total_units - completed_units) / rate, 1)
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "phase": self.phase,
            "status": status,
            "completed_units": int(completed_units),
            "total_units": int(total_units),
            "current_label": current_label,
            "elapsed_sec": round(elapsed, 1),
            "eta_sec": eta_sec,
            "run_usd": round(self.run_usd, 6),
            "max_usd": round(self.max_usd, 2),
            "run_calls": self.run_calls,
            "run_errors": self.run_errors,
            "run_input_tokens": self.run_input_tokens,
            "run_output_tokens": self.run_output_tokens,
            "run_thinking_tokens": self.run_thinking_tokens,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload["metrics_preview"] = extra
        atomic_write_json(self.progress_file, payload)
        if log_every > 0 and (completed_units % log_every == 0 or completed_units >= total_units):
            bar_str = self.format_progress_bar(
                completed_units, total_units, elapsed, eta_sec, self.run_usd
            )
            print(f"  [PROGRESS {self.run_id}] {bar_str} | {current_label}", flush=True)

    def _update_summary_unlocked(self, entry: LedgerEntry) -> None:
        if self.summary_file.exists():
            try:
                summary = json.loads(self.summary_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = self._empty_summary()
        else:
            summary = self._empty_summary()

        summary["total_usd"] = round(float(summary.get("total_usd", 0.0)) + entry.usd, 6)
        summary["remaining_credit_pool_usd"] = round(
            TOTAL_GEMINI_CREDIT_POOL_USD - summary["total_usd"], 4
        )
        summary["total_calls"] = int(summary.get("total_calls", 0)) + 1
        summary["total_input_tokens"] = int(summary.get("total_input_tokens", 0)) + entry.input_tokens
        summary["total_output_tokens"] = int(summary.get("total_output_tokens", 0)) + entry.output_tokens
        summary["total_thinking_tokens"] = (
            int(summary.get("total_thinking_tokens", 0)) + entry.thinking_tokens
        )
        summary["updated_at_utc"] = entry.timestamp_utc

        by_phase = summary.setdefault("by_phase", {})
        by_phase[entry.phase] = round(float(by_phase.get(entry.phase, 0.0)) + entry.usd, 6)

        by_model = summary.setdefault("by_model", {})
        m_rec = by_model.setdefault(
            entry.model,
            {"usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
        )
        m_rec["usd"] = round(float(m_rec["usd"]) + entry.usd, 6)
        m_rec["calls"] = int(m_rec["calls"]) + 1
        m_rec["input_tokens"] = int(m_rec["input_tokens"]) + entry.input_tokens
        m_rec["output_tokens"] = int(m_rec["output_tokens"]) + entry.output_tokens
        m_rec["thinking_tokens"] = int(m_rec["thinking_tokens"]) + entry.thinking_tokens

        atomic_write_json(self.summary_file, summary)

    @staticmethod
    def _empty_summary() -> dict[str, Any]:
        return {
            "credit_pool_usd": TOTAL_GEMINI_CREDIT_POOL_USD,
            "total_usd": 0.0,
            "remaining_credit_pool_usd": TOTAL_GEMINI_CREDIT_POOL_USD,
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_thinking_tokens": 0,
            "monitor_contact_masked": mask_email_for_display("gs06monitor@gmail.com"),
            "by_phase": {},
            "by_model": {},
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def rebuild_global_summary(self) -> dict[str, Any]:
        summary = self._empty_summary()
        if self.ledger_file.exists():
            with open(self.ledger_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    usd = float(row.get("usd", 0.0))
                    summary["total_usd"] = round(summary["total_usd"] + usd, 6)
                    summary["total_calls"] += 1
                    summary["total_input_tokens"] += int(row.get("input_tokens", 0))
                    summary["total_output_tokens"] += int(row.get("output_tokens", 0))
                    summary["total_thinking_tokens"] += int(row.get("thinking_tokens", 0))
                    phase = str(row.get("phase", "dev"))
                    summary["by_phase"][phase] = round(
                        float(summary["by_phase"].get(phase, 0.0)) + usd, 6
                    )
                    model = str(row.get("model", "unknown"))
                    m_rec = summary["by_model"].setdefault(
                        model,
                        {"usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
                    )
                    m_rec["usd"] = round(float(m_rec["usd"]) + usd, 6)
                    m_rec["calls"] += 1
                    m_rec["input_tokens"] += int(row.get("input_tokens", 0))
                    m_rec["output_tokens"] += int(row.get("output_tokens", 0))
                    m_rec["thinking_tokens"] += int(row.get("thinking_tokens", 0))
        summary["remaining_credit_pool_usd"] = round(
            TOTAL_GEMINI_CREDIT_POOL_USD - summary["total_usd"], 4
        )
        atomic_write_json(self.summary_file, summary)
        return summary


def send_monitor_email(subject: str, body: str, recipient: str = "gs06monitor@gmail.com") -> bool:
    """Send status/alert email via allow-listed `/usr/bin/mailx` without shell invocation."""
    if recipient not in ALLOWED_MONITOR_RECIPIENTS:
        raise ValueError("Recipient is not in ALLOWED_MONITOR_RECIPIENTS.")
    if not os.path.isfile(MAILX_BINARY) or not os.access(MAILX_BINARY, os.X_OK):
        return False
    clean_subject = "".join(ch for ch in subject if 32 <= ord(ch) < 127)[:140]
    proc = subprocess.run(
        [MAILX_BINARY, "-s", f"[RSI-Paradigm-Shift] {clean_subject}", recipient],
        input=body[:20_000],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    return proc.returncode == 0
