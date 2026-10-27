#!/bin/bash
# Convert one split of the unpacked hudnav-20k dataset into training scenes
# (CPU only; depth comes afterwards from depth_da3_array.slurm).
#
#   sbatch --account=<acct> tools/slurm/prep_split.slurm <split> <out root> [first N]
#   e.g. eval 16 -> data/hudnav20k_mg/val16 ; train -> data/hudnav20k_mg/train
#   # parallel: NUM_SHARDS=8 sbatch --array=0-7 ... prep_split.slurm train <out>
#
# Stops before writing anything if the scratch file quota lacks room for
# ~13 entries per episode (scene folders now + depth later).
#SBATCH --job-name=mg35-prep
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source tools/slurm/lib_env.sh

SPLIT=${1:?split: train | eval}
OUT=${2:?output root}
FIRST=${3:-}
SRC=${SRC:-data/hudnav-20k/OpenWorldCreativeHud-v0}
SPLITS=${SPLITS:-data/hudnav-20k/motworld-hudnav-20k.splits.json}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

# quota guard: room for every not-yet-converted episode of the split (~13
# entries each incl. later depth), so concurrent shards cannot overshoot
N=$("$PY" -c "import json,sys; e=[x for x in json.load(open(sys.argv[1]))['episodes'] if x['split']==sys.argv[2]]; \
n=len(e) if not sys.argv[3] else min(len(e), int(sys.argv[3])); print(n)" "$SPLITS" "$SPLIT" "$FIRST")
DONE=$(ls "$OUT/scenes" 2>/dev/null | wc -l)
read -r _ _ _ USED_F LIM_F <<<"$(showquota | grep -F "/scratch/user/$USER")"
NEED=$(( (N - DONE) * 13 + 5000 ))
echo "[prep] $SPLIT: $N episodes ($DONE converted), shard $SHARD/$NUM_SHARDS; files used $USED_F of $LIM_F, remaining need ~$NEED"
if [ $(( USED_F + NEED )) -gt "$LIM_F" ]; then
  echo "[prep] ABORT: not enough file quota; nothing written"; exit 2
fi

FIRST_ARG=()
[ -n "$FIRST" ] && FIRST_ARG=(--first "$FIRST")
"$PY" tools/craftium_to_matrixgame.py --src "$SRC" --splits "$SPLITS" \
    --split "$SPLIT" "${FIRST_ARG[@]}" --out "$OUT" \
    --num-shards "$NUM_SHARDS" --shard-id "$SHARD" | tail -3
showquota | grep -F "/scratch/user/$USER" || true
