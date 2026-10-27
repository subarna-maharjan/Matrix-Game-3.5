#!/bin/bash
# Login-node watcher: wait for the hudnav-20k restore (RESTORE.md) to finish,
# then submit prep -> DA3 depth -> LoRA training as one Slurm dependency chain.
#
#   nohup tools/slurm/chain_hudnav20k.sh > chain_hudnav20k.log 2>&1 &
#
# Env overrides: ACCOUNT, NUM_SHARDS (16), EXPECTED (episodes after restore).
set -euo pipefail
cd "$(dirname "$0")/../.."

ACCOUNT=${ACCOUNT:-132689576523}
NUM_SHARDS=${NUM_SHARDS:-16}
EXPECTED=${EXPECTED:-19977}   # 7,977 kept + 12,000 restored
SRC_DIR=data/hudnav-20k
LOG=$SRC_DIR/restore.log
OUT=data/hudnav20k_mg

echo "[chain] $(date) waiting for restore ($LOG)"
until grep -q '^exit=' "$LOG" 2>/dev/null; do sleep 60; done
grep -qx 'exit=0' "$LOG" || { echo "[chain] restore failed:"; cat "$LOG"; exit 1; }
N=$(ls "$SRC_DIR/OpenWorldCreativeHud-v0" | wc -l)
echo "[chain] $(date) restore done: $N episodes"
[ "$N" -ge "$EXPECTED" ] || { echo "[chain] expected >= $EXPECTED episodes, stopping"; exit 1; }

sb() { sbatch --parsable --account="$ACCOUNT" "$@"; }

PREP_VAL=$(sb tools/slurm/prep_split.slurm eval "$OUT/val16" 16)
PREP_TRAIN=$(sb tools/slurm/prep_split.slurm train "$OUT/train")
DEPTH_VAL=$(NUM_SHARDS=1 sb --dependency=afterok:$PREP_VAL --array=0 \
    tools/slurm/depth_da3_array.slurm "$OUT/val16/scenes")
DEPTH_TRAIN=$(NUM_SHARDS=$NUM_SHARDS sb --dependency=afterok:$PREP_TRAIN \
    --array=0-$((NUM_SHARDS - 1)) tools/slurm/depth_da3_array.slurm "$OUT/train/scenes")
TRAIN=$(sb --dependency=afterok:$DEPTH_VAL:$DEPTH_TRAIN tools/slurm/train_hudnav20k.slurm)

echo "[chain] prep val=$PREP_VAL train=$PREP_TRAIN"
echo "[chain] depth val=$DEPTH_VAL train=$DEPTH_TRAIN (array 0-$((NUM_SHARDS - 1)))"
echo "[chain] train=$TRAIN (starts after all depth jobs succeed)"
