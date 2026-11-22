"""Live progress and cost ledger for the Gemini diagnostic.

Monitoring output is newline-oriented so a captured log stays readable.
progress.json is replaced atomically. Neither record is an agent prompt.
"""

from __future__ import annotations

import json
import os
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from single_life.gemini_adapter import Usage, attempt_cap_usd, completion_cap_usd


def _money(value: Decimal) -> str:
    return format(value, "f")


class ProgressLedger:
    def __init__(
        self,
        path: Path,
        *,
        planned_smoke: int,
        planned_pilot: int,
        clock: Callable[[], str] | None = None,
        monotonic: Callable[[], float] | None = None,
        heartbeat_seconds: float = 30.0,
        printer: Callable[[str], None] | None = None,
    ) -> None:
        self.path = path
        self.planned_smoke = planned_smoke
        self.planned_pilot = planned_pilot
        self.planned_total = planned_smoke + planned_pilot
        self.attempt_cap = attempt_cap_usd()
        self.cap = completion_cap_usd()
        self.clock = clock or (lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.monotonic = monotonic or time.monotonic
        self.heartbeat_seconds = heartbeat_seconds
        self.printer = printer or (lambda line: print(line, flush=True))
        self.started = self.monotonic()
        self.phase = "preflight"
        self.status = "running"
        self.completed_smoke = 0
        self.completed_pilot = 0
        self.attempts = 0
        self.retries = 0
        self.failures = 0
        self.known_attempts = 0
        self.unknown_attempts = 0
        self.known_spend = Decimal("0")
        self.unknown_reserved = Decimal("0")
        self.known_input = 0
        self.known_output = 0
        self.known_thoughts = 0
        self.in_flight = False
        self._attempt_open = 0
        self.world = ""
        self.condition = ""
        self.step = 0
        self.lines: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start_heartbeat(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._beat, name="gemini-progress", daemon=True)
        self._thread.start()

    def stop_heartbeat(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None

    def _beat(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            with self._lock:
                if self.in_flight and self.status == "running":
                    self._publish(heartbeat=True)

    def locate(self, *, world: str, condition: str, step: int) -> None:
        with self._lock:
            self.world = world
            self.condition = condition
            self.step = step
            self._attempt_open = 0

    def begin_attempt(self) -> None:
        with self._lock:
            projected = self.known_spend + self.unknown_reserved + self.attempt_cap
            if projected > self.cap:
                raise RuntimeError("next attempt would exceed the completion cap")
            self.in_flight = True
            self._attempt_open += 1
            self._publish(heartbeat=False)

    def finish_attempt(self, *, usage: Usage | None, failed: bool) -> None:
        with self._lock:
            self.in_flight = False
            self.attempts += 1
            if self._attempt_open > 1:
                self.retries += 1
            if failed:
                self.failures += 1
            if usage is None or usage.status != "available" or usage.cost_usd is None:
                self.unknown_attempts += 1
                self.unknown_reserved += self.attempt_cap
            else:
                self.known_attempts += 1
                self.known_spend += usage.cost_usd
                self.known_input += usage.input_tokens or 0
                self.known_output += usage.output_tokens or 0
                self.known_thoughts += usage.thoughts_tokens or 0
            self._publish(heartbeat=False)

    def complete_decision(self) -> None:
        with self._lock:
            if self.phase == "smoke":
                self.completed_smoke += 1
            elif self.phase == "pilot":
                self.completed_pilot += 1
            self._attempt_open = 0
            self._publish(heartbeat=False)

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase
            self._publish(heartbeat=False)

    def finish(self, status: str) -> None:
        with self._lock:
            self.status = status
            self.in_flight = False
            self._publish(heartbeat=False)
        self.stop_heartbeat()

    def snapshot(self) -> dict[str, Any]:
        completed = self.completed_smoke + self.completed_pilot
        elapsed = max(0.0, self.monotonic() - self.started)
        remaining_decisions = max(0, self.planned_total - completed)
        eta = None
        if completed > 0 and elapsed > 0:
            eta = elapsed / completed * remaining_decisions
        observed_per_decision = None
        projection_basis = "conservative_until_warmup"
        if completed >= 4:
            observed_per_decision = (self.known_spend + self.unknown_reserved) / Decimal(completed)
            projection_basis = "observed_mean_per_completed_decision"
            projected_remaining = observed_per_decision * Decimal(remaining_decisions)
        else:
            projected_remaining = self.attempt_cap * 2 * Decimal(remaining_decisions)
        inflight = self.attempt_cap if self.in_flight else Decimal("0")
        committed_estimate = self.known_spend + self.unknown_reserved
        projected_final = committed_estimate + inflight + projected_remaining
        headroom = self.cap - committed_estimate - inflight
        percent = 0 if self.planned_total == 0 else (100 * completed) // self.planned_total
        if not (self.status == "complete" and completed == self.planned_total and self.planned_total > 0):
            percent = min(percent, 99)
        return {
            "status": self.status,
            "phase": self.phase,
            "heartbeat": False,
            "planned_smoke": self.planned_smoke,
            "planned_pilot": self.planned_pilot,
            "planned_total": self.planned_total,
            "completed_smoke": self.completed_smoke,
            "completed_pilot": self.completed_pilot,
            "completed_total": completed,
            "percent": percent,
            "attempts": self.attempts,
            "retries": self.retries,
            "failures": self.failures,
            "world": self.world,
            "condition": self.condition,
            "step": self.step,
            "in_flight": self.in_flight,
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": None if eta is None else round(eta, 3),
            "known_attempts": self.known_attempts,
            "unknown_attempts": self.unknown_attempts,
            "known_input_tokens": self.known_input,
            "known_output_tokens_including_thinking": self.known_output,
            "known_thoughts_tokens": self.known_thoughts,
            "known_spend_usd": _money(self.known_spend),
            "unknown_reserved_usd": _money(self.unknown_reserved),
            "inflight_reserved_usd": _money(inflight),
            "projected_remaining_usd": _money(projected_remaining),
            "projected_final_usd": _money(projected_final),
            "projection_basis": projection_basis,
            "observed_usd_per_completed_decision": (
                None if observed_per_decision is None else _money(observed_per_decision)
            ),
            "conservative_completion_bound_usd": _money(self.cap),
            "cap_usd": _money(self.cap),
            "headroom_usd": _money(headroom),
            "billing_note": "usage estimate from published token rates, not a provider invoice",
            "updated_at": self.clock(),
        }

    def render(self, snapshot: dict[str, Any] | None = None) -> str:
        item = snapshot or self.snapshot()
        filled = item["percent"] // 10
        bar = "#" * filled + "-" * (10 - filled)
        eta = "unknown" if item["eta_seconds"] is None else f"{item['eta_seconds']:.0f}s"
        pending = " pending" if item["in_flight"] else ""
        return (
            f"PROGRESS [{bar}] {item['percent']:3d}% "
            f"smoke {item['completed_smoke']}/{item['planned_smoke']} "
            f"pilot {item['completed_pilot']}/{item['planned_pilot']} "
            f"world={item['world'] or '-'} condition={item['condition'] or '-'} "
            f"step={item['step']} attempts={item['attempts']} retries={item['retries']} "
            f"failures={item['failures']} elapsed={item['elapsed_seconds']:.0f}s eta={eta} "
            f"spent={item['known_spend_usd']} unknown_reserved={item['unknown_reserved_usd']} "
            f"inflight={item['inflight_reserved_usd']} remaining={item['projected_remaining_usd']} "
            f"projected_final={item['projected_final_usd']} cap={item['cap_usd']} "
            f"headroom={item['headroom_usd']} basis={item['projection_basis']} "
            f"status={item['status']}{pending}"
        )

    def _publish(self, *, heartbeat: bool) -> None:
        payload = self.snapshot()
        payload["heartbeat"] = heartbeat
        line = self.render(payload)
        payload["last_line"] = line
        self.lines.append(line)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.partial")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)
        self.printer(line)
