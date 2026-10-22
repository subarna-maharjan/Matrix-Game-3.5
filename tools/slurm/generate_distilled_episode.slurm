#!/bin/bash
# Generate a full Craftium episode with the DISTILLED (3-step causal)
# Matrix-Game-3.5 model from its first frame and recorded camera trajectory.
#
#   sbatch --account=<acct> tools/slurm/generate_distilled_episode.slurm [NAME]
#
# The camera is padded to 1 + 84*BLOCKS poses by holding the last pose (8
# blocks for a 600-frame episode); the result is exported as the episode's
# frame count at 24 fps, plus a GT | distilled side-by-side and PSNR.
# Needs a 48 GB A40: even with MG35_T5_OFFLOAD=1 (T5 on CPU) it peaks above
# 40 GB during DA3 memory registration (DiT + rolling KV cache + DA3).
#SBATCH --job-name=mg35-distilled
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

NAME=${1:-distilled}
BLOCKS=${BLOCKS:-8}
CONFIG=${CONFIG:-configs/infer_distilled_8blocks.yaml}
DATA=${DATA:-data/hudnav_overfit}
OUT=${OUT:-outputs/full_episode}
ENV_TAR=.cache/env/mg35_env.tar
ENV_DIR=/tmp/mg35_env

nvidia-smi --query-gpu=name,memory.total --format=csv
CREATED_ENV=0
(
  flock 9
  if [ ! -x "$ENV_DIR/venv/bin/python" ]; then
    tar -C /tmp -xf "$ENV_TAR"
    touch "$ENV_DIR/.created_by_$SLURM_JOB_ID"
  fi
) 9>/tmp/mg35_env.lock
[ -e "$ENV_DIR/.created_by_$SLURM_JOB_ID" ] && CREATED_ENV=1
cleanup() { [ "$CREATED_ENV" = 1 ] && rm -rf "$ENV_DIR"; }
trap cleanup EXIT
PY="$ENV_DIR/venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# keep the 11 GB T5 on CPU between prompt encodings so generation fits a 40 GB A100
export MG35_T5_OFFLOAD=1

"$PY" tools/craftium_eval_inputs.py --root "$DATA" --out "$OUT/inputs"
SCENE_ID=$(ls "$DATA/scenes" | head -1)
IN="$OUT/inputs/$SCENE_ID"
DST="$OUT/first_person/$NAME"
mkdir -p "$DST"

# hold the last pose so every block has its 84 poses
"$PY" - "$IN/camera.npz" "$DST/camera_padded.npz" "$BLOCKS" <<'EOF'
import sys, numpy as np
src, dst, blocks = sys.argv[1], sys.argv[2], int(sys.argv[3])
z = np.load(src)
e, k = z["extrinsics_c2w"], z["intrinsics"]
need = 1 + 84 * blocks
if len(e) < need:
    pad = need - len(e)
    e = np.concatenate([e, np.repeat(e[-1:], pad, 0)])
    k = np.concatenate([k, np.repeat(k[-1:], pad, 0)])
np.savez(dst, extrinsics_c2w=e, intrinsics=k)
print(f"[distilled] camera {len(z['extrinsics_c2w'])} -> {len(e)} poses")
EOF

"$PY" infer_distilled.py --config "$CONFIG" \
    --checkpoint checkpoints/distilled-first-person.safetensors \
    --image "$IN/input.png" --camera "$DST/camera_padded.npz" \
    --prompt-file "$IN/prompt.txt" --output "$DST/result.mp4"

N_GT=$("$PY" -c "import cv2,sys; print(int(cv2.VideoCapture(sys.argv[1]).get(7)))" "$IN/gt.mp4")
"$PY" tools/export_episode_video.py --src "$DST/result.mp4" --frames "$N_GT" --size 640x360
"$PY" tools/log_eval_videos.py --run-dir "$DST" --gt "$IN/gt.mp4" \
    --video "$NAME=$DST/result_${N_GT}f_24fps.mp4" --tag full_episode
echo "[distilled] done -> $DST"
