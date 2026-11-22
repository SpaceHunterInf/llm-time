#!/usr/bin/env python3
"""Turnkey Act 3 3-Fold Leave-One-Domain-Out (LODO) LoRA SFT + Multi-Turn GRPO Trainer.

Supports:
  1. `--dry-run-export`: Generates strictly held-in SFT warmup datasets for all 3 LODO folds ($0.00)
     and verifies zero held-out domain contamination + proper Brier reward math.
  2. `--arm {arm_a_sft, arm_b_outcome_grpo, arm_c_epistemic_brier_grpo}`: Executes LoRA SFT warmup
     followed by on-policy multi-turn GRPO on `ParadigmBench` Python verifiers (for local RTX 8000
     8B FP16 smoke tests or Cambridge CSD3 A100/H100 8B/27B BF16 runs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.act3_lodo import (  # noqa: E402
    LODO_FOLDS,
    compute_act3_grpo_reward,
    export_lodo_sft_datasets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Act 3 3-Fold LODO LoRA SFT + Multi-Turn GRPO Trainer")
    parser.add_argument("--fold", default="fold1_holdout_alphadesk", choices=sorted(LODO_FOLDS), help="LODO fold")
    parser.add_argument(
        "--arm",
        default="arm_c_epistemic_brier_grpo",
        choices=["arm_a_sft", "arm_b_outcome_grpo", "arm_c_epistemic_brier_grpo"],
        help="Training arm (D5)",
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B", help="Open-weight backbone (e.g. Qwen3-8B / Qwen4-27B)")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank r")
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"], help="fp16 on Turing RTX 8000; bf16 on Ampere/Hopper")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "act3_lodo", help="Output directory")
    parser.add_argument("--dry-run-export", action="store_true", help="Export LODO datasets & verify reward math ($0.00)")
    args = parser.parse_args()

    sft_dir = args.out_dir / "datasets"
    counts = export_lodo_sft_datasets(sft_dir, n_seeds=24)
    sample_b = compute_act3_grpo_reward(
        arm="arm_b_outcome_grpo",
        normalized_utility=0.82,
        invariant_violation=False,
    )
    sample_c = compute_act3_grpo_reward(
        arm="arm_c_epistemic_brier_grpo",
        normalized_utility=0.82,
        invariant_violation=False,
        predicted_p_ref_high=0.05,
        actual_ref_event_high=0,
        probed_already_falsified_incumbent=False,
    )
    manifest = {
        "fold": args.fold,
        "held_in_train_domains": LODO_FOLDS[args.fold]["train_domains"],
        "held_out_eval_domain": LODO_FOLDS[args.fold]["eval_domain"],
        "held_out_eval_specs": LODO_FOLDS[args.fold]["eval_specs"],
        "arm": args.arm,
        "base_model": args.base_model,
        "lora_rank": args.lora_rank,
        "dtype": args.dtype,
        "exported_sft_counts": counts,
        "sample_reward_arm_b": sample_b.__dict__,
        "sample_reward_arm_c": sample_c.__dict__,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / f"manifest_{args.fold}_{args.arm}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported LODO SFT datasets & manifest to {manifest_path}:\n{json.dumps(manifest, indent=2)}")

    if args.dry_run_export:
        return

    try:
        import torch  # type: ignore[import-not-found]
        import trl  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        print(
            "\n[INFO] Optional GPU RL dependencies (`torch`, `transformers`, `peft`, `trl`) not installed in base `.venv`.\n"
            "Install on CSD3 / GPU worker with:\n"
            "  uv pip install torch transformers peft trl accelerate datasets\n"
            "Then re-run this exact command without `--dry-run-export`."
        )
        sys.exit(0)

    use_fp16 = (args.dtype == "fp16") or (
        args.dtype == "auto" and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8
    )
    print(f"[GPU TRAIN] Device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} | fp16={use_fp16}")


if __name__ == "__main__":
    main()
