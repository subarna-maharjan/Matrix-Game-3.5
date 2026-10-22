#!/bin/bash
# Precompute DA3 depth for converted scenes, one shard per array task.
#   NUM_SHARDS=16 sbatch --account=<acct> --array=0-15 tools/slurm/depth_da3_array.slurm <scenes dir>
# Keep --array's task count equal to NUM_SHARDS. Finished scenes (with _DONE)
# are skipped, so re-submitting only redoes what is missing.
#SBATCH --job-name=mg35-depth
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source tools/slurm/lib_env.sh

SCENES=${1:?usage: sbatch --array=0-N tools/slurm/depth_da3_array.slurm <scenes dir>}
export DA3_MODEL_PATH=${DA3_MODEL_PATH:-$PWD/checkpoints/DA3NESTED-GIANT-LARGE-1.1}
"$PY" tools/craftium_depth_da3.py --scenes "$SCENES" --align-to-poses \
    --num-shards "${NUM_SHARDS:-1}" --shard-id "${SLURM_ARRAY_TASK_ID:-0}"
