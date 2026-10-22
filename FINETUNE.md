# Fine-tuning Matrix-Game-3.5 on Craftium / Minetest data

`finetune.py` trains the **base (bidirectional) first-person model** with the
same flow-matching objective, memory schedule and camera conditioning as the
released training forward pass (LoRA by default). It does not cover the
distilled causal model: its DMD distillation trainer is not part of this
release.

The Slurm scripts under `tools/slurm/` target TAMU HPRC Grace; they unpack a
packed Python env to node-local `/tmp` (`tools/slurm/lib_env.sh`) and take
`--account=<acct>` at submission.

## 1. Convert episodes

Each Craftium episode directory needs `rgb.mp4` and `data.npz`
(`sha256.txt` becomes the scene id when present).

```bash
# a dataset with a splits.json (all episodes in one flat folder)
python tools/craftium_to_matrixgame.py --src data/hudnav-20k/OpenWorldCreativeHud-v0 \
    --splits data/hudnav-20k/motworld-hudnav-20k.splits.json --split train \
    --out data/hudnav20k_mg/train
python tools/craftium_to_matrixgame.py --src data/hudnav-20k/OpenWorldCreativeHud-v0 \
    --splits data/hudnav-20k/motworld-hudnav-20k.splits.json --split eval --first 16 \
    --out data/hudnav20k_mg/val16
# or any folder of episodes
python tools/craftium_to_matrixgame.py --src /path/to/episodes --out data/craftium_mg
```

`tools/slurm/prep_split.slurm <split> <out> [first N]` runs the same on a CPU
node. Poses come from `extrinsics_global` (world-to-camera, OpenCV axes,
metric), inverted to camera-to-world; normalised `intrinsics` are converted to
pixels per frame. Every episode gets one caption (`--caption-text`).

## 2. Precompute depth (GPU)

Training reads memory depth from disk (inference estimates it at runtime).
Depth-Anything-3, aligned to the known camera poses:

```bash
NUM_SHARDS=16 sbatch --account=<acct> --array=0-15 \
    tools/slurm/depth_da3_array.slurm data/hudnav20k_mg/train/scenes
```

Each finished scene gets `depth/video.npz`, `depth/metadata.json` and `_DONE`;
re-submitting skips finished scenes.

## 3. Build the index

```bash
python -m diffsynth.core.data.build_da3_video_index \
    --camera_params_path data/hudnav20k_mg/train/scenes \
    --prompt_path data/hudnav20k_mg/train/captions \
    --output_path data/hudnav20k_mg/train/index.sqlite --min_frame_count 200
```

`train_hudnav20k.slurm` builds missing indexes itself. Check the printed
`kept_count`; rejected scenes list a `scan_error`.

## 4. Train

```bash
sbatch --account=<acct> tools/slurm/train_hudnav20k.slurm
```

Defaults: LoRA rank 64 on the DiT, lr 1e-4 with 200 warm-up steps, 640x352,
2x A100 (one sample per GPU per step), `MAX_STEPS=20000`, a **full checkpoint
every 30 min** and **validation every 90 min** on the first 16 eval episodes.
Override with env vars (`MAX_STEPS`, `CKPT_MIN`, `VAL_MIN`, `HEIGHT`, `WIDTH`,
`DATA_ROOT`, `RUN_NAME`, `EXTRA`). The same options on `finetune.py`:

| Flag | Meaning |
|---|---|
| `--max_train_steps N` | stop after N optimizer steps |
| `--ckpt_every_minutes M` | full checkpoint every M minutes |
| `--resume_ckpt DIR` | resume exactly from a full checkpoint |
| `--val_index PATH` / `--val_every_minutes M` | validation episodes and interval |
| `--val_draws_per_episode D` | fixed draws per validation episode (default 2) |
| `--lr_warmup_steps N` | linear warm-up, then constant LR |
| `--probe_every N` | fixed-draw loss on training clips (for overfitting tests) |
| `--keep_text_encoder_on_gpu` | default keeps T5 on CPU with cached embeddings (~11 GB GPU saved) |

**Checkpoints** go to `outputs/finetune/<run>/checkpoints/step-NNNNNNN/`:
`weights.safetensors` (LoRA, same format as `epoch-*.safetensors`),
`optimizer.pt`, `rng_rank*.pt` and `state.json` (step, epoch, exact data
position, LR schedule, W&B run id). They are written to a temporary folder
and renamed when complete.

**Resume** from any checkpoint, or a run folder for its newest one; the same
W&B run continues:

```bash
RESUME_CKPT=outputs/finetune/<run>/checkpoints/step-0012345 \
    sbatch --account=<acct> tools/slurm/train_hudnav20k.slurm
MAX_STEPS=40000 RESUME_CKPT=outputs/finetune/<run> sbatch ...   # extend a finished run
```

Resuming reproduces the uninterrupted run exactly (data order, noise and
optimizer state). Older runs without full checkpoints resume with
`--resume_from <run dir>` (next epoch boundary, optimizer reset).

**Validation loss** is the flow-matching loss on fixed (clip, timestep, noise)
draws of the validation episodes, so changes reflect the model rather than the
random timestep, and it does not depend on the number of GPUs.

### Weights & Biases

`wandb login` once (stores the key in `~/.netrc`). Compute nodes reach W&B
through HPRC's `WebProxy` module, which the Slurm scripts load. Logged:
`train/loss`, `train/lr`, `train/grad_norm`, `train/loss_mode_*`,
**`val/loss`**, `val_episode/<id>`, `probe/loss`, `ckpt/saved`; the same
scalars are written to `<run>/loss_metrics.jsonl` and `<run>/train_log.jsonl`.

### Overfitting test on one episode

`tools/slurm/overfit_one_episode.slurm` converts one episode, trains with a
probe loss and compares base vs fine-tuned generation (`RUN_TAG`,
`TRAIN_EXTRA`, `INFER_EXTRA` env vars for other resolutions).

## 5. Generate and evaluate against ground truth

```bash
# base model, 8 blocks (600 frames) from the episode's first frame
sbatch --account=<acct> tools/slurm/generate_full_episode.slurm base
# fine-tuned: a LoRA checkpoint is merged into the base model first
sbatch --account=<acct> tools/slurm/generate_full_episode.slurm finetuned \
    outputs/finetune/<run>/checkpoints/step-0020000/weights.safetensors --height 352 --width 640
# side-by-side grid with labels
python tools/make_grid_video.py --out grid.mp4 \
    --video "Ground truth=<episode>/rgb.mp4" \
    --video "Fine-tuned=outputs/full_episode/first_person/finetuned/result_600f_24fps.mp4"
```

Each block covers 84 frames (`1 + 84*k` poses); `infer.py` writes 16 fps and
runs past the episode, so `tools/export_episode_video.py` re-times the result
to the episode's frame count at 24 fps. `tools/merge_lora.py` merges a LoRA
checkpoint for `infer.py --ckpt`.

## GPU memory notes

- **Training** fits a 40 GB A100 because T5 stays on CPU (default).
- **Base-model generation** with more than one block fits a 40 GB A100 only
  with `MG35_T5_OFFLOAD=1` (set in `generate_full_episode.slurm`); without it,
  DA3 memory registration between blocks runs out of memory.
- **Distilled generation** (`generate_distilled_episode.slurm`) runs out of
  memory even on a 48 GB A40 at 1280x704 (its rolling KV cache); it needs an
  80 GB GPU.

## Notes

- **File-count quota.** A converted scene is ~13 files/directories; check
  `showquota` before converting large datasets.
- **Memory-latent cache** is off in the config (one file per memory frame).
- **HUD / held item** are left in the frames; after fine-tuning the model
  reproduces them cleanly.
