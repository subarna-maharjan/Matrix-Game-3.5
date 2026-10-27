#!/bin/bash
# Overfit Matrix-Game-3.5 (LoRA) on ONE Craftium episode and log to W&B.
#
#   sbatch --account=<acct> tools/slurm/overfit_one_episode.slurm [MAX_STEPS] [EVAL_BLOCKS]
#
# Smoke test first (a few steps, 1-block eval), then the real run:
#   sbatch --account=<acct> --time=02:00:00 tools/slurm/overfit_one_episode.slurm 5 1
#   sbatch --account=<acct> tools/slurm/overfit_one_episode.slurm 2000 1
#
# Keep EVAL_BLOCKS at 1 on 40 GB A100s: from the second block on, DA3 memory
# registration OOMs. Full-length videos: tools/slurm/generate_full_episode.slurm (A40).
#
# Optional env vars: RUN_TAG (added to the run/W&B name), TRAIN_EXTRA (extra
# finetune.py flags), INFER_EXTRA (extra infer.py flags), e.g. for 640x352:
#   RUN_TAG=352p TRAIN_EXTRA="--height 352 --width 640" \
#   INFER_EXTRA="--height 352 --width 640" sbatch ... overfit_one_episode.slurm 3000 1
#
# Stages (each skipped when already done): unpack env -> convert episode ->
# DA3 depth -> index -> train (probe loss every PROBE_EVERY steps) -> merge
# LoRA -> generate base / fine-tuned from the episode's first frame ->
# side-by-side video + PSNR logged to the same W&B run.
#SBATCH --job-name=mg35-overfit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=1-12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

MAX_STEPS=${1:-2000}
EVAL_BLOCKS=${2:-1}
PROBE_EVERY=${PROBE_EVERY:-50}
EPISODE=${EPISODE:-hudnav-20k eval 200}
DATA=${DATA:-data/hudnav_overfit}
RUN="overfit${RUN_TAG:+_$RUN_TAG}_${SLURM_JOB_ID}"
read -r -a TRAIN_EXTRA_ARR <<< "${TRAIN_EXTRA:-}"
read -r -a INFER_EXTRA_ARR <<< "${INFER_EXTRA:-}"

# Compute nodes have no direct internet; HPRC's WebProxy module lets W&B
# sync live. The logger's reachability probe is a direct TCP connect that
# ignores the proxy, so request online explicitly (it still falls back to
# offline if wandb.init fails).
WANDB_MODE_ARG=()
if module load WebProxy 2>/dev/null; then
  WANDB_MODE_ARG=(--wandb_mode online)
  echo "[overfit] WebProxy loaded: https_proxy=${https_proxy:-unset}"
fi

nvidia-smi --query-gpu=name,memory.total --format=csv
df -h /tmp | tail -1

source tools/slurm/lib_env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DA3_MODEL_PATH="$PWD/checkpoints/DA3NESTED-GIANT-LARGE-1.1"

# ---- data -----------------------------------------------------------------
"$PY" tools/craftium_to_matrixgame.py --src "$EPISODE" --out "$DATA"
"$PY" tools/craftium_depth_da3.py --scenes "$DATA/scenes" --align-to-poses
"$PY" -m diffsynth.core.data.build_da3_video_index \
    --camera_params_path "$DATA/scenes" --prompt_path "$DATA/captions" \
    --output_path "$DATA/index.sqlite" --min_frame_count 200 \
    --workers 1 --worker_backend serial --overwrite --quiet
SCENE_ID=$(ls "$DATA/scenes" | head -1)

# ---- train ----------------------------------------------------------------
"$PY" -m accelerate.commands.launch --num_processes 1 --num_machines 1 \
    --mixed_precision bf16 --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    finetune.py --config configs/finetune_craftium_first_person.yaml \
    --dataset_index_path "$DATA/index.sqlite" \
    --memory_latent_cache_dir "${TMPDIR:-/tmp}/mg35_memlat_${SLURM_JOB_ID}" \
    --log_dir_name "$RUN" --wandb_run_name "$RUN" "${WANDB_MODE_ARG[@]}" \
    --steps_per_epoch 500 --num_epochs 1000 --save_steps 0 \
    --max_train_steps "$MAX_STEPS" --lr_warmup_steps 50 \
    --probe_every "$PROBE_EVERY" --probe_samples 4 --log_every 10 \
    "${TRAIN_EXTRA_ARR[@]}"
RUN_DIR="outputs/finetune/$RUN"

# ---- evaluate: base vs fine-tuned from the episode's first frame ------------
LATEST=$(ls -v "$RUN_DIR"/epoch-*.safetensors | tail -1)
MERGED="$RUN_DIR/$(basename "$LATEST" .safetensors)-merged.safetensors"
"$PY" tools/merge_lora.py --base checkpoints/first-person.safetensors \
    --lora "$LATEST" --out "$MERGED"
"$PY" tools/craftium_eval_inputs.py --root "$DATA" --out "$RUN_DIR/eval_inputs"
IN="$RUN_DIR/eval_inputs/$SCENE_ID"
for NAME in base finetuned; do
  CKPT_ARGS=()
  [ "$NAME" = finetuned ] && CKPT_ARGS=(--ckpt "$MERGED")
  "$PY" infer.py --person first --num-blocks "$EVAL_BLOCKS" \
      --image "$IN/input.png" --camera "$IN/camera.npz" --prompt-file "$IN/prompt.txt" \
      --output "$RUN_DIR/eval" --name "$NAME" "${CKPT_ARGS[@]}" "${INFER_EXTRA_ARR[@]}"
done
"$PY" tools/log_eval_videos.py --run-dir "$RUN_DIR" --gt "$IN/gt.mp4" \
    --video "base=$RUN_DIR/eval/first_person/base/result.mp4" \
    --video "finetuned=$RUN_DIR/eval/first_person/finetuned/result.mp4"

# merged checkpoint is 10 GB; the LoRA epoch files are enough to rebuild it
rm -f "$MERGED"
echo "[overfit] done -> $RUN_DIR"
