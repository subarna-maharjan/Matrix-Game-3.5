#!/bin/bash
# Generate a full Craftium episode with Matrix-Game-3.5 from its first frame and
# recorded camera trajectory, then build GT | generated side-by-side + PSNR.
#
#   sbatch --account=<acct> tools/slurm/generate_full_episode.slurm [NAME] [CKPT] [infer.py flags...]
#
#   NAME  output name (default: base)
#   CKPT  DiT checkpoint (default: base first-person model; pass "none" to keep
#         the default while adding flags). A finetune.py LoRA epoch-*.safetensors
#         is merged into the base model first.
#   e.g. native 640x352:  ... generate_full_episode.slurm base_352p none --height 352 --width 640
#
# 600-frame episodes need 8 blocks (block k covers frames 84k+1 .. 84k+84);
# poses past the end are held, and the comparison is trimmed to the episode.
# Fits a 40 GB A100 with MG35_T5_OFFLOAD=1. Without it, multi-block generation OOMs (DA3
# memory registration of each new block ran with the DiT and the 11 GB T5 resident).
#SBATCH --job-name=mg35-generate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

NAME=${1:-base}
CKPT=${2:-}
INFER_EXTRA_ARR=("${@:3}")
# sbatch drops empty arguments, so CKPT may be missing with flags following
if [[ "$CKPT" == --* ]]; then
  INFER_EXTRA_ARR=("${@:2}")
  CKPT=""
elif [ "$CKPT" = none ]; then
  CKPT=""
fi
BLOCKS=${BLOCKS:-8}
DATA=${DATA:-data/hudnav_overfit}
OUT=${OUT:-outputs/full_episode}

nvidia-smi --query-gpu=name,memory.total --format=csv
source tools/slurm/lib_env.sh
MERGED=""
cleanup_merged() { [ -n "$MERGED" ] && rm -f "$MERGED"; _mg35_release_env; }
trap cleanup_merged EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# keep the 11 GB T5 on CPU between prompt encodings so generation fits a 40 GB A100
export MG35_T5_OFFLOAD=1

CKPT_ARGS=()
if [ -n "$CKPT" ]; then
  if "$PY" -c "import sys; from safetensors import safe_open; \
f = safe_open(sys.argv[1], 'pt'); sys.exit(0 if any('.lora_A.' in k for k in f.keys()) else 1)" "$CKPT"; then
    MERGED="${TMPDIR:-/tmp}/merged_${SLURM_JOB_ID}.safetensors"
    "$PY" tools/merge_lora.py --base checkpoints/first-person.safetensors \
        --lora "$CKPT" --out "$MERGED"
    CKPT_ARGS=(--ckpt "$MERGED")
  else
    CKPT_ARGS=(--ckpt "$CKPT")
  fi
fi

"$PY" tools/craftium_eval_inputs.py --root "$DATA" --out "$OUT/inputs"
SCENE_ID=$(ls "$DATA/scenes" | head -1)
IN="$OUT/inputs/$SCENE_ID"
"$PY" infer.py --person first --num-blocks "$BLOCKS" \
    --image "$IN/input.png" --camera "$IN/camera.npz" --prompt-file "$IN/prompt.txt" \
    --output "$OUT" --name "$NAME" "${CKPT_ARGS[@]}" "${INFER_EXTRA_ARR[@]}"
"$PY" tools/log_eval_videos.py --run-dir "$OUT/first_person/$NAME" --gt "$IN/gt.mp4" \
    --video "$NAME=$OUT/first_person/$NAME/result.mp4" --tag full_episode
# result.mp4 is 16 fps and runs past the episode; export 600 frames @ 24 fps,
# at the generated size and at the episode's 640x360
N_GT=$("$PY" -c "import cv2,sys; print(int(cv2.VideoCapture(sys.argv[1]).get(7)))" "$IN/gt.mp4")
"$PY" tools/export_episode_video.py --src "$OUT/first_person/$NAME/result.mp4" \
    --frames "$N_GT" --size 640x360
echo "[generate] done -> $OUT/first_person/$NAME"
