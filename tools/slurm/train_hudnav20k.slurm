#!/bin/bash
# LoRA fine-tune of the Matrix-Game-3.5 base first-person model on hudnav-20k
# (train split), validating on the first 16 eval episodes.
#
#   sbatch --account=<acct> tools/slurm/train_hudnav20k.slurm
#   # resume exactly from any full checkpoint (same W&B run continues):
#   RESUME_CKPT=outputs/finetune/<run>/checkpoints/step-0012345 \
#       sbatch --account=<acct> tools/slurm/train_hudnav20k.slurm
#   # RESUME_CKPT may also be a run directory: its newest complete checkpoint
#   # extend later, e.g. to 40k:  MAX_STEPS=40000 RESUME_CKPT=... sbatch ...
#
# Env overrides: MAX_STEPS (20000), RUN_NAME, CKPT_MIN (30), VAL_MIN (90),
# HEIGHT/WIDTH (352/640), DATA_ROOT (data/hudnav20k_mg), EXTRA (more flags).
# Full checkpoints every CKPT_MIN minutes under <run>/checkpoints/step-*/;
# a resumed job writes into the run directory of the checkpoint it resumes.
#SBATCH --job-name=mg35-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

MAX_STEPS=${MAX_STEPS:-20000}
CKPT_MIN=${CKPT_MIN:-30}
VAL_MIN=${VAL_MIN:-90}
HEIGHT=${HEIGHT:-352}
WIDTH=${WIDTH:-640}
DATA_ROOT=${DATA_ROOT:-data/hudnav20k_mg}
TRAIN_INDEX=$DATA_ROOT/train/index.sqlite
VAL_INDEX=$DATA_ROOT/val16/index.sqlite
read -r -a EXTRA_ARR <<< "${EXTRA:-}"

# compute nodes reach W&B only through HPRC's proxy
WANDB_MODE_ARG=()
if module load WebProxy 2>/dev/null; then WANDB_MODE_ARG=(--wandb_mode online); fi
nvidia-smi --query-gpu=name,memory.total --format=csv
source tools/slurm/lib_env.sh
NGPU=$(nvidia-smi -L | wc -l)

# ---- indexes (built once; delete index.sqlite to rebuild) ------------------
for root in "$DATA_ROOT/train" "$DATA_ROOT/val16"; do
  if [ ! -f "$root/index.sqlite" ]; then
    "$PY" -m diffsynth.core.data.build_da3_video_index \
        --camera_params_path "$root/scenes" --prompt_path "$root/captions" \
        --output_path "$root/index.sqlite" --min_frame_count 200 \
        --workers 16 --quiet
  fi
done
# one epoch = every indexed training episode once (summed over GPUs)
N_TRAIN=$("$PY" -c "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); \
print(c.execute('select count(*) from records where scan_error is null').fetchone()[0])" "$TRAIN_INDEX")
echo "[train] $N_TRAIN training episodes indexed"

# ---- run directory / resume -------------------------------------------------
RESUME_ARGS=()
if [ -n "${RESUME_CKPT:-}" ] && [ -d "$RESUME_CKPT/checkpoints" ]; then
  # a run directory: resume from its newest complete checkpoint
  RESUME_CKPT=$(ls -d "$RESUME_CKPT"/checkpoints/step-[0-9]* 2>/dev/null | grep -v '\.tmp$' | sort | tail -1)
  [ -n "$RESUME_CKPT" ] || { echo "[train] no complete checkpoint to resume"; exit 1; }
fi
if [ -n "${RESUME_CKPT:-}" ]; then
  RUN_DIR=$(dirname "$(dirname "$(realpath "$RESUME_CKPT")")")
  RUN_NAME=$(basename "$RUN_DIR")
  RESUME_ARGS=(--resume_ckpt "$RESUME_CKPT")
  echo "[train] resuming $RUN_NAME from $RESUME_CKPT"
else
  RUN_NAME=${RUN_NAME:-hudnav20k_${HEIGHT}p_lora_${SLURM_JOB_ID}}
fi

"$PY" -m accelerate.commands.launch --num_processes "$NGPU" --num_machines 1 \
    --mixed_precision bf16 --main_process_port $((29500 + SLURM_JOB_ID % 1000)) \
    finetune.py --config configs/finetune_craftium_first_person.yaml \
    --dataset_index_path "$TRAIN_INDEX" --val_index "$VAL_INDEX" \
    --height "$HEIGHT" --width "$WIDTH" \
    --log_dir_name "$RUN_NAME" --wandb_run_name "$RUN_NAME" "${WANDB_MODE_ARG[@]}" \
    --steps_per_epoch "$N_TRAIN" --num_epochs 1000 --save_steps 0 \
    --max_train_steps "$MAX_STEPS" --lr_warmup_steps 200 \
    --ckpt_every_minutes "$CKPT_MIN" --val_every_minutes "$VAL_MIN" \
    --val_draws_per_episode 2 --log_every 10 \
    --dataset_num_workers 4 \
    "${RESUME_ARGS[@]}" "${EXTRA_ARR[@]}"
echo "[train] done -> outputs/finetune/$RUN_NAME"
