"""Drive the nine ordinary-KL runs on this machine's GPU. Resume-safe."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "learning_pilot"
RUNS = REPO / "runs" / "learning_pilot"
MODES = ("mlp", "head", "lora")
SEEDS = (0, 1, 2)
STEPS = 192
SPLITS = ("train", "dev", "same_topology", "structural")


def completed(out: Path) -> bool:
    man = out / "manifest.json"
    if not man.exists():
        return False
    try:
        blob = json.loads(man.read_text())
        return int(blob.get("train_state", {}).get("step", 0)) >= STEPS
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(PILOT))


def uv_cmd() -> list[str]:
    return [
        "uv",
        "run",
        "--extra",
        "cu124",
        "python",
        str(REPO / "scripts" / "learning_pilot.py"),
    ]


def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    env_prefix = {
        "HF_HOME": str(REPO / "models" / "hf"),
        "TOKENIZERS_PARALLELISM": "false",
        "UV_CACHE_DIR": "/tmp/llm-time-uv-cache",
    }
    import os
    os.environ.update(env_prefix)
    for mode in MODES:
        for seed in SEEDS:
            out = RUNS / f"{mode}_s{seed}"
            if completed(out):
                print(f"skip complete {out.name}", flush=True)
                continue
            cmd = uv_cmd() + ["fit", "--mode", mode, "--seed", str(seed), "--device", "cuda", "--out-dir", str(out)]
            if (out / "last.pt").exists():
                cmd.append("--resume")
            run(cmd)
    for mode in MODES:
        for seed in SEEDS:
            out = RUNS / f"{mode}_s{seed}"
            if not completed(out):
                raise SystemExit(f"incomplete fit {out}")
            for split in SPLITS:
                dest = out / f"eval_{split}.json"
                if dest.exists():
                    print(f"skip eval {out.name} {split}", flush=True)
                    continue
                extra = ["--allow-holdout"] if split in ("same_topology", "structural") else []
                run(
                    uv_cmd()
                    + ["eval", "--run-dir", str(out), "--split", split, "--device", "cuda"]
                    + extra
                )
    run(uv_cmd() + ["aggregate", "--runs-root", str(RUNS)])
    print("all nine fits + evals done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
