#!/usr/bin/env python3
"""Live CLI Progress Bar & ETA Monitor for Multi-Stream Experiments (`progress.json`)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]

V3_CAMPAIGN_SHARDS = [
    "v3_t1_s401_base", "v3_t1_s401_ours", "v3_t1_s402_base", "v3_t1_s402_ours",
    "v3_t1_s403_base", "v3_t1_s403_ours", "v3_t1_s404_base", "v3_t1_s404_ours",
    "v3_t2_s401_base", "v3_t2_s401_ours", "v3_t2_s402_base", "v3_t2_s402_ours",
    "v3_t2_s403_base", "v3_t2_s403_ours", "v3_t2_s404_base", "v3_t2_s404_ours",
    "v3_st_s401_base", "v3_st_s401_ours", "v3_st_s402_base", "v3_st_s402_ours",
]
SLB_L3_SHARDS = ["slb_b_l3_refs", "slb_b_l3_skillbook", "slb_b_l3_srn", "slb_b_l3_sgeh"]
AUTONOMY_SHARDS = (
    [(f"autonomy_s{s}_flash", 240.0) for s in range(501, 507)]
    + [(f"autonomy_s{s}_hyd_a", 144.0) for s in range(501, 507)]
    + [(f"autonomy_s{s}_hyd_b", 96.0) for s in range(501, 507)]
    + [(f"autonomy_gemma31b_s{s}", 60.0) for s in range(501, 504)]
)


def _bar(pct: float, width: int = 24) -> str:
    clamped = max(0.0, min(100.0, pct))
    filled = int(round((clamped / 100.0) * width))
    return "[" + ("█" * filled) + ("░" * (width - filled)) + "]"


def _fmt_dur(sec: float | None) -> str:
    if sec is None or sec < 0:
        return "—"
    m, s = divmod(int(round(sec)), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def render_snapshot(runs_dir: Path) -> str:
    now = datetime.now(timezone.utc)
    lines = [f"=== RSI-Sprint Live Progress & ETA ({now.strftime('%H:%M:%S UTC')}) ===", ""]

    # 1. Campaign-level progress bars
    v3_units_done = 0.0
    v3_units_total = len(V3_CAMPAIGN_SHARDS) * 252.0
    v3_done_shards = 0
    for rid in V3_CAMPAIGN_SHARDS:
        rdir = runs_dir / rid
        if (rdir / "alphadesk_v3_report.json").is_file():
            v3_units_done += 252.0
            v3_done_shards += 1
        elif (rdir / "progress.json").is_file():
            d = json.loads((rdir / "progress.json").read_text(encoding="utf-8"))
            v3_units_done += min(252.0, float(d.get("completed_units", 0)))
    v3_pct = 100.0 * v3_units_done / v3_units_total
    # Longest serial queue among our 3 parallel v3 worker chains has <= 4 shards; ~9.5m per shard
    rem_v3_shards_max_chain = max(
        sum(1 for r in grp if not (runs_dir / r / "alphadesk_v3_report.json").is_file())
        for grp in (
            ["v3_t1_s403_base", "v3_t1_s403_ours", "v3_t1_s404_base", "v3_t1_s404_ours"],
            ["v3_t2_s403_base", "v3_t2_s403_ours", "v3_t2_s404_base", "v3_t2_s404_ours"],
            ["v3_st_s401_ours", "v3_st_s402_base", "v3_st_s402_ours"],
        )
    )
    est_v3_wall_sec = max(0.0, (rem_v3_shards_max_chain - 0.35) * 570.0) if rem_v3_shards_max_chain > 0 else 0.0
    lines.append(
        f"AlphaDesk-v0.3 (N=4):  {_bar(v3_pct)} {v3_pct:5.1f}% "
        f"({v3_done_shards}/{len(V3_CAMPAIGN_SHARDS)} shards, {int(v3_units_done)}/{int(v3_units_total)} eps) "
        f"| Wall ETA ~{_fmt_dur(est_v3_wall_sec)}"
    )

    slb_units_done = 0.0
    slb_units_total = 96.0 + 3 * 120.0  # refs=96 tasks, 3 learned=120 tasks each
    slb_done_shards = 0
    for rid in SLB_L3_SHARDS:
        rdir = runs_dir / rid
        expected = 96.0 if rid == "slb_b_l3_refs" else 120.0
        if (rdir / "slb_mini_report.json").is_file():
            slb_units_done += expected
            slb_done_shards += 1
        elif (rdir / "progress.json").is_file():
            d = json.loads((rdir / "progress.json").read_text(encoding="utf-8"))
            slb_units_done += min(expected, float(d.get("completed_units", 0)))
    slb_pct = 100.0 * slb_units_done / slb_units_total
    rem_slb_shards = len(SLB_L3_SHARDS) - slb_done_shards
    est_slb_wall_sec = max(0.0, (rem_slb_shards - 0.25) * 480.0) if rem_slb_shards > 0 else 0.0
    lines.append(
        f"ServeLearnBench L3:    {_bar(slb_pct)} {slb_pct:5.1f}% "
        f"({slb_done_shards}/{len(SLB_L3_SHARDS)} shards, {int(slb_units_done)}/{int(slb_units_total)} tasks) "
        f"| Wall ETA ~{_fmt_dur(est_slb_wall_sec)}"
    )

    if any((runs_dir / rid).exists() for rid, _ in AUTONOMY_SHARDS):
        aut_done = 0.0
        aut_tot = sum(u for _, u in AUTONOMY_SHARDS)
        aut_shards_done = 0
        for rid, exp_u in AUTONOMY_SHARDS:
            rdir = runs_dir / rid
            if (rdir / "autonomy_fork_report.json").is_file():
                aut_done += exp_u
                aut_shards_done += 1
            elif (rdir / "progress.json").is_file():
                d = json.loads((rdir / "progress.json").read_text(encoding="utf-8"))
                aut_done += min(exp_u, float(d.get("completed_units", 0)))
        aut_pct = 100.0 * aut_done / aut_tot
        rem_aut = max(0, len(AUTONOMY_SHARDS) - aut_shards_done)
        est_aut_wall = max(0.0, (1.0 - aut_done / aut_tot) * 1500.0) if rem_aut > 0 else 0.0
        lines.append(
            f"Cross-Model Autonomy:  {_bar(aut_pct)} {aut_pct:5.1f}% "
            f"({aut_shards_done}/{len(AUTONOMY_SHARDS)} shards, {int(aut_done)}/{int(aut_tot)} eps) "
            f"| Wall ETA ~{_fmt_dur(est_aut_wall)}"
        )
    lines.append("")
    lines.append("Active Worker Streams (freshly updated < 30 min):")

    active_count = 0
    for pfile in sorted(runs_dir.glob("*/progress.json")):
        try:
            d = json.loads(pfile.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("status") != "running":
            continue
        upd_str = str(d.get("updated_at_utc", ""))
        try:
            upd_dt = datetime.fromisoformat(upd_str)
            age_sec = (now - upd_dt).total_seconds()
        except ValueError:
            age_sec = 0.0
        if age_sec > 1800:
            continue
        active_count += 1
        comp = int(d.get("completed_units", 0))
        tot = max(1, int(d.get("total_units", 1)))
        pct = 100.0 * comp / tot
        el = float(d.get("elapsed_sec", 0.0))
        eta = d.get("eta_sec")
        eta_f = float(eta) if isinstance(eta, (int, float)) else None
        usd = float(d.get("run_usd", 0.0))
        rid = str(d.get("run_id", pfile.parent.name))
        lines.append(
            f"  • {rid:20s} {_bar(pct, 18)} {pct:5.1f}% ({comp:3d}/{tot:<3d}) | "
            f"elapsed {_fmt_dur(el):>6s} | shard ETA {_fmt_dur(eta_f):>6s} | ${usd:.3f}"
        )
    if active_count == 0:
        lines.append("  (No currently active worker streams — all runs idle or completed.)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show live experiment progress bars & ETAs")
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds (0 = print once)")
    args = parser.parse_args()
    runs_dir = ROOT / "results" / "runs"
    if args.watch <= 0:
        print(render_snapshot(runs_dir))
        return
    try:
        while True:
            print("\033[2J\033[H" + render_snapshot(runs_dir), flush=True)
            time.sleep(max(1, args.watch))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
