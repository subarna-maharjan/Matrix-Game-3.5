#!/bin/bash
# Convert one split of the unpacked hudnav-20k dataset into training scenes
# (CPU only; depth comes afterwards from depth_da3_array.slurm).
#
#   sbatch --account=<acct> tools/slurm/prep_split.slurm <split> <out root> [first N]
#   e.g. eval 16 -> data/hudnav20k_mg/val16 ; train -> data/hudnav20k_mg/train
#SBATCH --job-name=mg35-prep
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source tools/slurm/lib_env.sh

SPLIT=${1:?split: train | eval}
OUT=${2:?output root}
FIRST=${3:-}
SRC=${SRC:-data/hudnav-20k/OpenWorldCreativeHud-v0}
SPLITS=${SPLITS:-data/hudnav-20k/motworld-hudnav-20k.splits.json}

FIRST_ARG=()
[ -n "$FIRST" ] && FIRST_ARG=(--first "$FIRST")
"$PY" tools/craftium_to_matrixgame.py --src "$SRC" --splits "$SPLITS" \
    --split "$SPLIT" "${FIRST_ARG[@]}" --out "$OUT" | tail -3
showquota | grep -F "/scratch/user/$USER" || true
