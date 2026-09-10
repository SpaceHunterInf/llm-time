"""Live/fake runner for sequential selective revision. Reuses harness identity and persist."""
from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any, Mapping

from tqdm import tqdm

from llm_rl_engine.agents.llm import GenerationResult
from llm_rl_engine.analysis.selective_revision_harness import (
    BookkeepingError,
    BoundedGeminiClient,
    BudgetError,
    IdentityError,
    LiveSettings,
    append_jsonl,
    atomic_write_json,
    check_or_init_identity,
    classify_exception,
    count_request_tokens,
    empty_ledger,
    estimate_cost_usd,
    file_sha256,
    find_unscored_response,
    finish_is_incomplete,
    generation_settings_payload,
    load_jsonl,
    load_ledger,
    persist_provider_response,
    result_from_saved,
    sha256_text,
    snapshot_code,
)
from llm_rl_engine.analysis.selective_revision_pilot import ValidationError, canonical_json
from llm_rl_engine.analysis.selective_revision_sequential import (
    CONDITIONS,
    PROBE_AFTER,
    SequentialTape,
    development_tapes,
    initial_state_payload,
    parse_sequential_response,
    previous_checkpoint,
    prompt_for_job,
    score_sequential,
    sequential_jobs,
    serialize_state,
    state_hash,
)

IMPLEMENTATION_FILES = (
    "llm_rl_engine/analysis/selective_revision.py",
    "llm_rl_engine/analysis/selective_revision_sequential.py",
    "llm_rl_engine/analysis/selective_revision_sequential_run.py",
)
REPAIR_SUFFIX = (
    "\n\nYour previous response was invalid: {error}\n"
    "Return exactly one valid JSON object with keys joint, probability_high, "
    "predictive_mean, and action."
)


def sequential_identity(settings: LiveSettings, repo_root: Path, tape_ids: list[str]) -> dict[str, Any]:
    hashes = {}
    for rel in IMPLEMENTATION_FILES:
        path = repo_root / rel
        hashes[rel] = file_sha256(path) if path.is_file() else "missing"
    body = {
        "schema_version": 1,
        "kind": "sequential_selective_revision_scientific_identity",
        "model": settings.model,
        "temperature": settings.temperature,
        "thinking_budget": settings.thinking_budget,
        "configured_max_output_tokens": settings.max_output_tokens,
        "provider_max_output_tokens": settings.billed_output_limit,
        "max_input_tokens": settings.max_input_tokens,
        "retry_policy": {"max_retries": settings.max_retries},
        "probes": list(PROBE_AFTER),
        "conditions": list(CONDITIONS),
        "tape_ids": tape_ids,
        "implementation_hashes": hashes,
    }
    body["effective_hash"] = sha256_text(canonical_json(body))
    return body


def job_key(tape_id: str, n: int, condition: str, effective_hash: str, prompt_hash: str, prev_hash: str) -> str:
    return f"{tape_id}|n{n}|{condition}|{effective_hash}|{prompt_hash}|{prev_hash}"


def _slot(tape_id: str, n: int, condition: str) -> tuple[str, str]:
    return tape_id, f"n{n}_{condition}"


def run_sequential(
    output_dir: Path,
    settings: LiveSettings,
    client: Any,
    *,
    repo_root: Path,
    smoke_limit: int | None = None,
    resume: bool = True,
    progress: bool = False,
    tapes: list[SequentialTape] | None = None,
) -> dict[str, Any]:
    tapes = list(tapes or development_tapes())
    jobs = sequential_jobs(tapes)
    if smoke_limit is not None:
        if smoke_limit > 3:
            raise BudgetError("smoke is at most 3 calls")
        jobs = jobs[:smoke_limit]
    identity = sequential_identity(settings, repo_root, [t.tape_id for t in tapes])
    check_or_init_identity(output_dir, identity, settings)
    attempts_needed = len(jobs) * (1 + settings.max_retries)
    if settings.max_calls < attempts_needed:
        raise BudgetError(
            f"max_calls={settings.max_calls} does not cover {len(jobs)} jobs with retries"
        )
    reserved = attempts_needed * settings.reserved_per_attempt
    if settings.max_usd + 1e-15 < reserved:
        raise BudgetError(
            f"max_usd={settings.max_usd} does not cover retry-reserve {reserved:.6f}"
        )
    atomic_write_json(
        output_dir / "live_manifest.json",
        {
            "effective_hash": identity["effective_hash"],
            "model": settings.model,
            "scientific_identity": identity,
            "n_jobs": len(jobs),
            "code": snapshot_code(repo_root),
        },
    )
    results_path = output_dir / "results.jsonl"
    attempts_path = output_dir / "attempts.jsonl"
    ledger_path = output_dir / "ledger.json"
    ledger = load_ledger(ledger_path)
    from llm_rl_engine.analysis.selective_revision_harness import absorb_dangling_reservations

    absorb_dangling_reservations(ledger)
    atomic_write_json(ledger_path, ledger)
    completed_rows = [
        row for row in load_jsonl(results_path) if row.get("status") == "complete"
    ] if resume else []
    done = {row["key"] for row in completed_rows}
    completed_by_logical = {
        (row.get("tape_id"), row.get("n"), row.get("condition")): row for row in completed_rows
    }
    prev_self: dict[str, dict[str, Any]] = {}
    invalid_marker: dict[str, bool] = {}
    token_cache: dict[str, tuple[int, str]] = {}
    settings_payload = generation_settings_payload(settings)
    n_already = 0
    for job in jobs:
        key = _key_for(job, identity, prev_self, invalid_marker)
        logical = (job["tape_id"], job["n"], job["condition"])
        if key in done or logical in completed_by_logical:
            n_already += 1
            row = completed_by_logical.get(logical) or {}
            if job["condition"] == "self_state":
                if row.get("valid") and row.get("parsed_joint"):
                    prev_self[job["tape_id"]] = serialize_state(
                        row["parsed_joint"], f"after_obs_{job['n']}"
                    )
                    invalid_marker[job["tape_id"]] = False
                else:
                    invalid_marker[job["tape_id"]] = True
    prev_self = {}
    invalid_marker = {}
    bar = tqdm(
        total=len(jobs),
        initial=n_already,
        desc="sequential",
        unit="job",
        disable=not progress,
        file=sys.stderr,
        dynamic_ncols=True,
    )
    try:
        for job in jobs:
            key = _key_for(job, identity, prev_self, invalid_marker)
            logical = (job["tape_id"], job["n"], job["condition"])
            if key in done or logical in completed_by_logical:
                row = completed_by_logical.get(logical) or {}
                if job["condition"] == "self_state":
                    if row.get("valid") and row.get("parsed_joint"):
                        prev_self[job["tape_id"]] = serialize_state(
                            row["parsed_joint"], f"after_obs_{job['n']}"
                        )
                        invalid_marker[job["tape_id"]] = False
                    else:
                        invalid_marker[job["tape_id"]] = True
                continue
            row = _run_one(
                job,
                output_dir=output_dir,
                settings=settings,
                client=client,
                identity=identity,
                ledger=ledger,
                ledger_path=ledger_path,
                attempts_path=attempts_path,
                results_path=results_path,
                prev_self=prev_self,
                invalid_marker=invalid_marker,
                token_cache=token_cache,
                settings_payload=settings_payload,
            )
            done.add(row["key"])
            if job["condition"] == "self_state":
                if row.get("valid") and row.get("parsed_joint"):
                    prev_self[job["tape_id"]] = serialize_state(
                        row["parsed_joint"], f"after_obs_{job['n']}"
                    )
                    invalid_marker[job["tape_id"]] = False
                else:
                    invalid_marker[job["tape_id"]] = True
            all_rows = load_jsonl(results_path)
            snapshot = {
                "n_results": len(all_rows),
                "n_jobs": len(jobs),
                "ledger": ledger,
                "last": {
                    "tape_id": row.get("tape_id"),
                    "n": row.get("n"),
                    "condition": row.get("condition"),
                    "valid": row.get("valid"),
                    "decision_error": row.get("decision_error"),
                    "decision_loss": row.get("decision_loss"),
                },
            }
            atomic_write_json(output_dir / "progress.json", snapshot)
            atomic_write_json(output_dir / "summary.json", snapshot)
            bar.set_postfix(
                spent=f"{float(ledger.get('spent_usd') or 0):.3f}",
                err=sum(1 for r in all_rows if r.get("decision_error")),
                refresh=False,
            )
            bar.update(1)
            if progress:
                tqdm.write(
                    f"{row.get('tape_id')} n={row.get('n')} {row.get('condition')} "
                    f"valid={row.get('valid')} chose={row.get('chosen_action')} "
                    f"opt={row.get('optimal_action')} loss={row.get('decision_loss')} "
                    f"spent={ledger.get('spent_usd')}",
                    file=sys.stderr,
                )
            if row.get("stop_run"):
                break
    finally:
        bar.close()
    rows = load_jsonl(results_path)
    summary = _summarize(rows)
    atomic_write_json(output_dir / "summary.json", {"summary": summary, "ledger": ledger})
    return {"summary": summary, "ledger": ledger, "n_results": len(rows)}


def _key_for(job, identity, prev_self, invalid_marker) -> str:
    prev = None
    if job["condition"] == "self_state":
        prev = prev_self.get(job["tape_id"]) or initial_state_payload()
    prompt = prompt_for_job(job, prev, invalid_self=bool(invalid_marker.get(job["tape_id"])))
    prev_hash = state_hash(prev) if prev else "none"
    return job_key(
        job["tape_id"],
        job["n"],
        job["condition"],
        identity["effective_hash"],
        sha256_text(prompt),
        prev_hash,
    )


def _run_one(
    job,
    *,
    output_dir: Path,
    settings: LiveSettings,
    client,
    identity: Mapping[str, Any],
    ledger: dict[str, Any],
    ledger_path: Path,
    attempts_path: Path,
    results_path: Path,
    prev_self: dict[str, dict[str, Any]],
    invalid_marker: dict[str, bool],
    token_cache: dict[str, tuple[int, str]],
    settings_payload: Mapping[str, Any],
) -> dict[str, Any]:
    from llm_rl_engine.analysis.selective_revision_harness import _utc_now

    tape_id = job["tape_id"]
    n = int(job["n"])
    condition = job["condition"]
    case_id, treatment = _slot(tape_id, n, condition)
    prev = prev_self.get(tape_id) if condition == "self_state" else None
    prompt = prompt_for_job(job, prev, invalid_self=bool(invalid_marker.get(tape_id)))
    prev_hash = state_hash(prev) if prev else "none"
    key = job_key(
        tape_id, n, condition, identity["effective_hash"], sha256_text(prompt), prev_hash
    )
    saved = find_unscored_response(output_dir, case_id, treatment)
    if saved and saved.get("raw_response") is not None:
        result = result_from_saved(saved)
        row = _score_and_write(
            job, key, result, settings, results_path, ledger, ledger_path, recovered=True
        )
        saved["scored"] = True
        atomic_write_json(Path(saved["_path"]), {k: v for k, v in saved.items() if k != "_path"})
        return row

    last_error = None
    valid = False
    parsed = None
    failure_kind = None
    stop_run = False
    last_result: GenerationResult | None = None
    attempts_used = 0
    for retry_index in range(settings.max_retries + 1):
        room_calls = settings.max_calls - ledger["attempts"]
        room_usd = settings.max_usd - ledger["spent_usd"] - ledger["reserved_usd"]
        if room_calls < 1 or room_usd + 1e-15 < settings.reserved_per_attempt:
            raise BudgetError("pre-call cap hit")
        current = prompt
        if retry_index and last_error:
            current = prompt + REPAIR_SUFFIX.format(error=last_error)
        n_tok, _src = count_request_tokens(current, client, token_cache)
        if n_tok > settings.max_input_tokens:
            raise BudgetError(f"prompt tokens {n_tok} exceed max_input_tokens")
        attempt_id = f"{key}|attempt{ledger['attempts']}"
        ledger["reserved_usd"] += settings.reserved_per_attempt
        ledger["attempts"] += 1
        attempts_used += 1
        atomic_write_json(ledger_path, ledger)
        append_jsonl(
            attempts_path,
            {
                "attempt_id": attempt_id,
                "key": key,
                "status": "reserved",
                "tape_id": tape_id,
                "n": n,
                "condition": condition,
                "retry_index": retry_index,
                "reserved_usd": settings.reserved_per_attempt,
                "started_at": _utc_now(),
            },
        )
        started = time.perf_counter()
        try:
            result = client.generate(current)
            latency = time.perf_counter() - started
            last_result = result
            try:
                persist_provider_response(
                    output_dir,
                    case_id=case_id,
                    treatment=treatment,
                    attempt_n=ledger["attempts"],
                    attempt_id=attempt_id,
                    result=result,
                    latency_s=latency,
                    request_text=current,
                    request_hash=sha256_text(current),
                    generation_settings=settings_payload,
                    job_key_value=key,
                    scored=False,
                )
            except Exception as exc:
                raise BookkeepingError(str(exc)) from exc
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
            ledger["reserved_usd"] -= settings.reserved_per_attempt
            ledger["spent_usd"] += actual
            atomic_write_json(ledger_path, ledger)
            try:
                parsed = parse_sequential_response(result.text)
                valid = True
                failure_kind = None
                last_error = None
            except (ValidationError, ValueError, KeyError) as exc:
                valid = False
                failure_kind = "parse_failure"
                last_error = str(exc)
                parsed = None
            finish = getattr(result, "finish_reason", None)
            if finish_is_incomplete(finish, finish is not None):
                valid = False
                failure_kind = "incomplete_finish"
            append_jsonl(
                attempts_path,
                {
                    "attempt_id": attempt_id,
                    "key": key,
                    "status": "attempted",
                    "cost_usd": actual,
                    "usage_status": usage_status,
                    "parse_error": last_error,
                    "finish_reason": finish,
                    "raw_response": result.text,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "thoughts_tokens": getattr(result, "thoughts_tokens", 0),
                    "latency_s": latency,
                },
            )
            if valid:
                break
        except (BudgetError, BookkeepingError, IdentityError):
            raise
        except Exception as exc:
            category = classify_exception(exc)
            ledger["reserved_usd"] -= settings.reserved_per_attempt
            ledger["spent_usd"] += settings.reserved_per_attempt
            ledger["interrupted_reservations"] = int(ledger.get("interrupted_reservations") or 0) + 1
            atomic_write_json(ledger_path, ledger)
            append_jsonl(
                attempts_path,
                {
                    "attempt_id": attempt_id,
                    "status": "failed",
                    "exception_category": category,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            failure_kind = "api_failure"
            if category in {"authentication", "quota"}:
                stop_run = True
                break
    scored = score_sequential(parsed, job["reference"], valid=valid)
    row = {
        "key": key,
        "status": "complete",
        "tape_id": tape_id,
        "n": n,
        "condition": condition,
        "structure": job["tape"].structure,
        "prev_state_hash": prev_hash,
        "attempts_used": attempts_used,
        "parsed_joint": parsed["joint"] if parsed else None,
        "invalid_self_fallback": bool(invalid_marker.get(tape_id)),
        **scored,
        "failure_kind": failure_kind,
        "stop_run": stop_run,
    }
    append_jsonl(results_path, row)
    ledger["completed_jobs"] = int(ledger.get("completed_jobs") or 0) + 1
    atomic_write_json(ledger_path, ledger)
    leftover = find_unscored_response(output_dir, case_id, treatment)
    if leftover:
        leftover["scored"] = True
        atomic_write_json(Path(leftover["_path"]), {k: v for k, v in leftover.items() if k != "_path"})
    return row


def _score_and_write(job, key, result, settings, results_path, ledger, ledger_path, recovered):
    parsed = None
    valid = False
    failure_kind = None
    try:
        parsed = parse_sequential_response(result.text)
        valid = True
    except (ValidationError, ValueError, KeyError) as exc:
        failure_kind = "parse_failure"
    scored = score_sequential(parsed, job["reference"], valid=valid)
    row = {
        "key": key,
        "status": "complete",
        "tape_id": job["tape_id"],
        "n": job["n"],
        "condition": job["condition"],
        "structure": job["tape"].structure,
        "recovered_from_saved_response": recovered,
        "parsed_joint": parsed["joint"] if parsed else None,
        **scored,
        "failure_kind": failure_kind,
        "stop_run": False,
    }
    append_jsonl(results_path, row)
    ledger["completed_jobs"] = int(ledger.get("completed_jobs") or 0) + 1
    atomic_write_json(ledger_path, ledger)
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("valid")]
    by_cond: dict[str, list] = {c: [] for c in CONDITIONS}
    for r in rows:
        by_cond.setdefault(r.get("condition", "?"), []).append(r)
    def mean_loss(rs):
        if not rs:
            return None
        return sum(float(r.get("decision_loss") or 0) for r in rs) / len(rs)
    paired = []
    index = {(r.get("tape_id"), r.get("n"), r.get("condition")): r for r in rows}
    for r in rows:
        if r.get("condition") != "self_state":
            continue
        rec = index.get((r.get("tape_id"), r.get("n"), "reconstruct"))
        if rec and r.get("valid") and rec.get("valid"):
            paired.append(float(r["decision_loss"]) - float(rec["decision_loss"]))
    return {
        "n_attempted": len(rows),
        "n_valid": len(valid),
        "n_decision_error": sum(1 for r in valid if r.get("decision_error")),
        "n_parse_failure": sum(1 for r in rows if r.get("failure_kind") == "parse_failure"),
        "n_api_failure": sum(1 for r in rows if r.get("failure_kind") == "api_failure"),
        "mean_decision_loss": mean_loss(rows),
        "by_condition_mean_loss": {c: mean_loss(rs) for c, rs in by_cond.items()},
        "by_condition_errors": {
            c: sum(1 for r in rs if r.get("decision_error")) for c, rs in by_cond.items()
        },
        "self_minus_reconstruct_mean": (sum(paired) / len(paired) if paired else None),
        "n_paired_self_reconstruct": len(paired),
    }
