#!/usr/bin/env bash
#SBATCH --job-name=rsi-act3-lodo
#SBATCH --account=CSD3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=results/act3_lodo/slurm_%A_%a.out

set -euo pipefail

FOLDS=("fold1_holdout_alphadesk" "fold2_holdout_lifecompanion" "fold3_holdout_coalitionops")
FOLD="${FOLDS[${SLURM_ARRAY_TASK_ID:-0}]}"
MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"

echo "[Act 3 LODO] Launching Fold=${FOLD} on Backbone=${MODEL}"
python scripts/train_act3_lodo.py --fold "${FOLD}" --arm arm_a_sft --base-model "${MODEL}" --dtype bf16
python scripts/train_act3_lodo.py --fold "${FOLD}" --arm arm_b_outcome_grpo --base-model "${MODEL}" --dtype bf16
python scripts/train_act3_lodo.py --fold "${FOLD}" --arm arm_c_epistemic_brier_grpo --base-model "${MODEL}" --dtype bf16
