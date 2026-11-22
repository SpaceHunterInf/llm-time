"""Progress ledger for task 006.

Ticks, scheduled generation calls, and provider attempts are separate counters.
Frozen and revising lifetimes keep separate planned totals. Token-count requests
are not generation attempts. progress.json is not an agent prompt.
"""

from __future__ import annotations

import json
import os
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from single_life.gemini_adapter import Usage


def _money(value: Decimal) -> str:
    return format(value, "f")


class ProcedureLedger:
    def __init__(
        self,
        path: Path,
        *,
        planned_smoke_calls: int,
        planned_revision_calls: int,
        planned_frozen_calls: int,
        planned_smoke_ticks: int,
        planned_revision_ticks: int,
        planned_frozen_ticks: int,
        planned_references: int,
        attempt_cap: Decimal,
        completion_cap: Decimal,
        clock: Callable[[], str] | None = None,
        monotonic: Callable[[], float] | None = None,
        heartbeat_seconds: float = 30.0,
        printer: Callable[[str], None] | None = None,
    ) -> None:
        self.path = path
        self.planned_smoke_calls = planned_smoke_calls
        self.planned_revision_calls = planned_revision_calls
        self.planned_frozen_calls = planned_frozen_calls
        self.planned_calls = planned_smoke_calls + planned_revision_calls + planned_frozen_calls
        self.planned_smoke_ticks = planned_smoke_ticks
        self.planned_revision_ticks = planned_revision_ticks
        self.planned_frozen_ticks = planned_frozen_ticks
        self.planned_ticks = planned_smoke_ticks + planned_revision_ticks + planned_frozen_ticks
        self.planned_references = planned_references
        self.attempt_cap = attempt_cap
        self.cap = completion_cap
        self.clock = clock or (lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.monotonic = monotonic or time.monotonic
        self.heartbeat_seconds = heartbeat_seconds
        self.printer = printer or (lambda line: print(line, flush=True))
        self.started = self.monotonic()
        self.phase = "preflight"
        self.status = "running"
        self.completed_smoke_calls = 0
        self.completed_revision_calls = 0
        self.completed_frozen_calls = 0
        self.completed_smoke_ticks = 0
        self.completed_revision_ticks = 0
        self.completed_frozen_ticks = 0
        self.completed_references = 0
        self.attempts = 0
        self.retries = 0
        self.failures = 0
        self.token_count_requests = 0
        self.token_count_failures = 0
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
        self.tick = 0
        self.lines: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start_heartbeat(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._beat, name="procedure-progress", daemon=True)
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

    def locate(self, *, world: str, condition: str, tick: int) -> None:
        with self._lock:
            self.world = world
            self.condition = condition
            self.tick = tick
            self._attempt_open = 0

    def begin_token_count(self) -> None:
        with self._lock:
            self.in_flight = True
            self._publish(heartbeat=False)

    def finish_token_count(self, *, ok: bool) -> None:
        with self._lock:
            self.in_flight = False
            self.token_count_requests += 1
            if not ok:
                self.token_count_failures += 1
                self.failures += 1
            self._publish(heartbeat=False)

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

    def complete_call(self) -> None:
        with self._lock:
            if self.phase == "smoke" or self.condition == "smoke":
                self.completed_smoke_calls += 1
            elif self.condition == "revision":
                self.completed_revision_calls += 1
            elif self.condition == "frozen":
                self.completed_frozen_calls += 1
            self._attempt_open = 0
            self._publish(heartbeat=False)

    def complete_tick(self) -> None:
        with self._lock:
            if self.phase == "smoke" or self.condition == "smoke":
                self.completed_smoke_ticks += 1
            elif self.condition == "revision":
                self.completed_revision_ticks += 1
            elif self.condition == "frozen":
                self.completed_frozen_ticks += 1
            self._publish(heartbeat=False)

    def mark_references(self, count: int) -> None:
        with self._lock:
            self.completed_references = count
            self._publish(heartbeat=False)

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase
            self._publish(heartbeat=False)

    def note_failure(self) -> None:
        with self._lock:
            self.failures += 1
            self.in_flight = False
            self._publish(heartbeat=False)

    def finish(self, status: str) -> None:
        with self._lock:
            self.status = status
            self.in_flight = False
            self._publish(heartbeat=False)
        self.stop_heartbeat()

    def snapshot(self) -> dict[str, Any]:
        calls = self.completed_smoke_calls + self.completed_revision_calls + self.completed_frozen_calls
        ticks = self.completed_smoke_ticks + self.completed_revision_ticks + self.completed_frozen_ticks
        elapsed = max(0.0, self.monotonic() - self.started)
        remaining_calls = max(0, self.planned_calls - calls)
        eta = None
        if calls > 0 and elapsed > 0:
            eta = elapsed / calls * remaining_calls
        projection_basis = "two_full_attempts_per_remaining_call"
        if calls >= 4:
            observed = (self.known_spend + self.unknown_reserved) / Decimal(calls)
            projection_basis = "observed_mean_per_completed_call"
            projected_remaining = observed * Decimal(remaining_calls)
        else:
            observed = None
            projected_remaining = self.attempt_cap * 2 * Decimal(remaining_calls)
        inflight = self.attempt_cap if self.in_flight and self._attempt_open else Decimal("0")
        committed = self.known_spend + self.unknown_reserved
        projected_final = committed + inflight + projected_remaining
        headroom = self.cap - committed - inflight
        percent = 0 if self.planned_ticks == 0 else (100 * ticks) // self.planned_ticks
        full = (
            self.status == "complete"
            and ticks == self.planned_ticks
            and calls == self.planned_calls
            and self.planned_ticks > 0
        )
        if not full:
            percent = min(percent, 99)
        return {
            "status": self.status,
            "phase": self.phase,
            "heartbeat": False,
            "planned_ticks": self.planned_ticks,
            "completed_ticks": ticks,
            "planned_smoke_ticks": self.planned_smoke_ticks,
            "completed_smoke_ticks": self.completed_smoke_ticks,
            "planned_revision_ticks": self.planned_revision_ticks,
            "completed_revision_ticks": self.completed_revision_ticks,
            "planned_frozen_ticks": self.planned_frozen_ticks,
            "completed_frozen_ticks": self.completed_frozen_ticks,
            "planned_calls": self.planned_calls,
            "completed_calls": calls,
            "planned_smoke_calls": self.planned_smoke_calls,
            "completed_smoke_calls": self.completed_smoke_calls,
            "planned_revision_calls": self.planned_revision_calls,
            "completed_revision_calls": self.completed_revision_calls,
            "planned_frozen_calls": self.planned_frozen_calls,
            "completed_frozen_calls": self.completed_frozen_calls,
            "planned_references": self.planned_references,
            "completed_references": self.completed_references,
            "percent": percent,
            "attempts": self.attempts,
            "retries": self.retries,
            "failures": self.failures,
            "token_count_requests": self.token_count_requests,
            "token_count_failures": self.token_count_failures,
            "world": self.world,
            "condition": self.condition,
            "tick": self.tick,
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
            "observed_usd_per_completed_call": None if observed is None else _money(observed),
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
            f"ticks {item['completed_ticks']}/{item['planned_ticks']} "
            f"(smoke {item['completed_smoke_ticks']}/{item['planned_smoke_ticks']} "
            f"rev {item['completed_revision_ticks']}/{item['planned_revision_ticks']} "
            f"frz {item['completed_frozen_ticks']}/{item['planned_frozen_ticks']}) "
            f"calls {item['completed_calls']}/{item['planned_calls']} "
            f"(smoke {item['completed_smoke_calls']}/{item['planned_smoke_calls']} "
            f"rev {item['completed_revision_calls']}/{item['planned_revision_calls']} "
            f"frz {item['completed_frozen_calls']}/{item['planned_frozen_calls']}) "
            f"references {item['completed_references']}/{item['planned_references']} "
            f"world={item['world'] or '-'} condition={item['condition'] or '-'} "
            f"tick={item['tick']} attempts={item['attempts']} retries={item['retries']} "
            f"failures={item['failures']} token_counts={item['token_count_requests']} "
            f"elapsed={item['elapsed_seconds']:.0f}s eta={eta} "
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
