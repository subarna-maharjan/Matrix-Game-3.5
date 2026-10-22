#!/usr/bin/env python3
"""Matrix-Game-3.5 fine-tuning (base / bidirectional model, flow-matching SFT).

The release ships the full training forward pass (``WanMosaicPipelineModule``
-> ``FlowMatchSFTLoss``), the training dataset and the model/accelerator
setup in ``examples/wanvideo/pipeline/mosaic/main.py``; only the optimizer
loop was removed. This script reuses ``main()`` unchanged and swaps its
final inference hand-off for the training loop below, so every model and
dataset option stays in sync with the inference path.

Usage (see FINETUNE.md for data preparation):
    accelerate launch --num_processes 2 --mixed_precision bf16 finetune.py \
        --config configs/finetune_craftium_first_person.yaml \
        --dataset_index_path data/craftium_mg/index.sqlite

Extra flags handled here (everything else is a pipeline flag / config key):
    --lr_warmup_steps N   linear LR warm-up over N optimizer steps (default 0)
    --log_every N         print/log the loss every N optimizer steps (default 10)
    --max_train_steps N   stop after N optimizer steps in total
    --ckpt_every_minutes M  save a full training checkpoint every M minutes
                          (LoRA/DiT weights, optimizer, LR schedule, RNG and
                          data position; default 0 = off)
    --resume_ckpt DIR     resume exactly from one of those checkpoints
    --val_index PATH      sqlite index of validation episodes
    --val_every_minutes M  log val/loss every M minutes (and at start/end)
    --val_draws_per_episode D  fixed (window, timestep, noise) draws per
                          validation episode (default 2)
    --probe_every N       every N steps log probe/loss on fixed draws from the
                          training set (default 0 = off)
    --probe_samples K     number of fixed draws in the probe (default 4)
    --keep_text_encoder_on_gpu  keep T5 on the GPU (default: CPU + cached
                          prompt embeddings, saves ~11 GB of GPU memory)

Full checkpoints land in <output_path>/<run>/checkpoints/step-NNNNNNN/
(weights.safetensors, optimizer.pt, rng_rank*.pt, state.json); the weights
file has the same format as epoch-*.safetensors and can be merged with
tools/merge_lora.py. Older runs without full checkpoints can still resume with
--resume_from <run dir> (next epoch boundary, optimizer state reset).
"""

import argparse
import copy
import glob
import importlib
import json
import os
import random
import re
import shutil
import sys
import time

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import numpy as np
import torch
from safetensors.torch import save_file

from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from examples.wanvideo.pipeline.mosaic.config import parse_pipeline_args
from examples.wanvideo.pipeline.mosaic.datasets import build_mosaic_latent_dataset
from examples.wanvideo.pipeline.mosaic.metrics_logger import MosaicMetricsLogger
from examples.wanvideo.pipeline.mosaic.validation import run_mosaic_validation

_CKPT_RE = re.compile(r"^epoch-(\d+)\.safetensors$")
CKPT_FORMAT = 1
FT = argparse.Namespace()


def _parse_finetune_flags(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--ckpt_every_minutes", type=float, default=0.0)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--val_index", type=str, default=None)
    parser.add_argument("--val_every_minutes", type=float, default=0.0)
    parser.add_argument("--val_draws_per_episode", type=int, default=2)
    parser.add_argument("--probe_every", type=int, default=0)
    parser.add_argument("--probe_samples", type=int, default=4)
    parser.add_argument("--keep_text_encoder_on_gpu", action="store_true")
    return parser.parse_known_args(argv)


def _ddp_ignore_text_encoder(model):
    """Must run before accelerator.prepare: DDP rejects a module whose
    parameters span devices unless the CPU-resident part is ignored."""
    model._ddp_params_and_buffers_to_ignore = [
        name for name, _ in model.named_parameters() if name.startswith("pipe.text_encoder.")
    ] + [name for name, _ in model.named_buffers() if name.startswith("pipe.text_encoder.")]


def _text_encoder_to_cpu_with_cache(model, max_entries=512):
    """Run the frozen T5 on CPU and cache prompt embeddings (~11 GB GPU saved).

    The text encoder is only needed to embed captions, which repeat across
    steps; computing each once on CPU and reusing it keeps the 5B DiT's
    activations within a 40 GB GPU. Must run after accelerator.prepare,
    which moves the whole module to the GPU.
    """
    from collections import OrderedDict

    from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder

    pipe = model.pipe
    if getattr(pipe, "text_encoder", None) is None:
        return
    pipe.text_encoder.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cache = OrderedDict()

    @torch.no_grad()
    def process(self, pipe, prompt, positive):
        key = prompt if isinstance(prompt, str) else json.dumps(prompt)
        emb = cache.get(key)
        if emb is None:
            ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
            emb = pipe.text_encoder(ids, mask)
            for i, v in enumerate(mask.gt(0).sum(dim=1).long()):
                emb[:, v:] = 0
            cache[key] = emb
            if len(cache) > max_entries:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        return {"context": emb.to(device=pipe.device, dtype=pipe.torch_dtype)}

    WanVideoUnit_PromptEmbedder.process = process
    print("[finetune] text encoder on CPU with prompt-embedding cache")


# Fixed-draw evaluation (probe / validation) uses an epoch id training never
# reaches, so its clip windows are disjoint from the training stream's draws.
_PROBE_EPOCH = 10**6


@torch.no_grad()
def _fixed_draw_losses(accelerator, model, dataset, indices, seed_base):
    """Loss on fixed (clip window, timestep, noise, memory schedule) draws.

    The per-step training loss is dominated by the random diffusion timestep;
    re-evaluating the same draws every time removes that variance. Every RNG
    the forward touches (torch, python ``random``, the step_count-keyed
    memory schedule / anchor plan) is pinned per draw and restored after.
    Returns {index: (loss, episode name)} for the indices this rank ran.
    """
    module = accelerator.unwrap_model(model)
    saved_epoch = getattr(dataset, "_epoch_id", 0)
    saved_offset = getattr(dataset, "_epoch_offset", 0)
    saved_step = module.step_count
    saved_py = random.getstate()
    devices = [accelerator.device] if accelerator.device.type == "cuda" else []
    out = {}
    try:
        # pin both the per-sample RNG epoch and the per-epoch episode rotation
        # so draw i always maps to the same episode and window
        dataset._epoch_id = _PROBE_EPOCH
        dataset._epoch_offset = 0
        for i in indices:
            data = dataset[i]
            name = str((data.get("info") or {}).get("clean_name", i))
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed_base + i)
                random.seed(seed_base + i)
                module.step_count = _PROBE_EPOCH + i
                out[i] = (float(module(data).float()), name)
    finally:
        dataset._epoch_id = saved_epoch
        dataset._epoch_offset = saved_offset
        module.step_count = saved_step
        random.setstate(saved_py)
    return out


def _distributed_fixed_draw_losses(accelerator, model, dataset, num_draws, seed_base):
    """Split draws across ranks (draw i -> rank i % world) and gather.

    Each draw is seeded by its index alone, so results do not depend on the
    number of GPUs. Every rank returns the full list of (loss, name).
    """
    rank, world = accelerator.process_index, accelerator.num_processes
    local = _fixed_draw_losses(
        accelerator, model, dataset, range(rank, num_draws, world), seed_base
    )
    if world > 1:
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, local)
        merged = {}
        for part in gathered:
            merged.update(part)
        local = merged
    return [local[i] for i in range(num_draws)]


# ---- legacy epoch-boundary resume (runs without full checkpoints) ----------

def _latest_epoch_checkpoint(run_dir):
    best = None
    for path in glob.glob(os.path.join(run_dir, "epoch-*.safetensors")):
        match = _CKPT_RE.match(os.path.basename(path))
        if match and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), path)
    return None if best is None else best[1]


def _resume_overrides(argv):
    """Point the model weights at the checkpoint being resumed.

    ``main()`` only auto-fills ``trained_dit`` when it is empty, but the
    fine-tune config sets it to the base model, so a resume would silently
    restart from the base weights. LoRA runs reload the adapter instead.
    """
    args = parse_pipeline_args(argv)
    if FT.resume_ckpt:
        state = _read_ckpt_state(FT.resume_ckpt)
        weights = os.path.join(FT.resume_ckpt, "weights.safetensors")
        if state.get("lora"):
            print(f"[finetune] resume: LoRA adapter <- {weights}")
            return argv + ["--lora_checkpoint", weights]
        print(f"[finetune] resume: trained_dit <- {weights}")
        return argv + ["--trained_dit", weights]
    if not args.resume_from:
        return argv
    latest = _latest_epoch_checkpoint(args.resume_from)
    if latest is None:
        print(f"[finetune] no epoch checkpoint in {args.resume_from}; starting from config weights")
        return argv
    if args.lora_base_model:
        if args.lora_checkpoint:
            return argv
        print(f"[finetune] resume: LoRA adapter <- {latest}")
        return argv + ["--lora_checkpoint", latest]
    print(f"[finetune] resume: trained_dit <- {latest}")
    return argv + ["--trained_dit", latest]


def _write_train_state(log_dir, epoch_id, global_step, pass_id):
    with open(os.path.join(log_dir, "train_state.json"), "w") as f:
        json.dump(
            {
                "epoch_id": epoch_id,
                "next_epoch": epoch_id + 1,
                "global_step": global_step,
                "dataset_pass_id": pass_id,
            },
            f,
            indent=1,
        )


# ---- full training checkpoints ----------------------------------------------

def _read_ckpt_state(ckpt_dir):
    path = os.path.join(ckpt_dir, "state.json")
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: {ckpt_dir} is not a full checkpoint (no state.json)")
    with open(path) as f:
        state = json.load(f)
    if int(state.get("format", 0)) != CKPT_FORMAT:
        raise SystemExit(f"ERROR: unsupported checkpoint format in {ckpt_dir}")
    return state


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(accelerator, model, optimizer, run_dir, state, remove_prefix):
    """Write <run>/checkpoints/step-NNNNNNN atomically (tmp dir + rename)."""
    final = os.path.join(run_dir, "checkpoints", f"step-{state['global_step']:07d}")
    tmp = final + ".tmp"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
    accelerator.wait_for_everyone()

    torch.save(_rng_state(), os.path.join(tmp, f"rng_rank{accelerator.process_index}.pt"))
    if accelerator.is_main_process:
        module = accelerator.unwrap_model(model)
        weights = {}
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if remove_prefix and name.startswith(remove_prefix):
                name = name[len(remove_prefix):]
            weights[name] = param.detach().to("cpu").contiguous()
        save_file(weights, os.path.join(tmp, "weights.safetensors"))
        torch.save(optimizer.state_dict(), os.path.join(tmp, "optimizer.pt"))
        with open(os.path.join(tmp, "state.json"), "w") as f:
            json.dump(state, f, indent=1)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if os.path.exists(final):
            shutil.rmtree(final)
        os.replace(tmp, final)
        print(f"[finetune] checkpoint -> {final}", flush=True)
    accelerator.wait_for_everyone()
    return final


def _broadcast_flags(accelerator, flags):
    """Rank 0 decides time-based events so every rank acts on the same step."""
    t = torch.tensor([int(bool(f)) for f in flags], device=accelerator.device)
    if accelerator.num_processes > 1:
        torch.distributed.broadcast(t, src=0)
    return [bool(v) for v in t.tolist()]


def _build_val_dataset(args, accelerator):
    if not FT.val_index:
        return None
    val_args = copy.copy(args)
    val_args.dataset_index_path = FT.val_index
    val_args.steps_per_epoch = 1
    val_args.rank = accelerator.process_index
    dataset = build_mosaic_latent_dataset(val_args)
    if accelerator.is_main_process:
        print(f"[finetune] validation: {len(dataset.path)} episodes from {FT.val_index}")
    return dataset


def run_mosaic_training_task(
    accelerator, dataset, model, model_logger, log_dir, args=None, val_dataset=None
):
    """Training loop; same signature as ``runner.run_mosaic_inference_task``."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "No trainable parameters: set trainable_models (e.g. dit) or lora_base_model."
        )
    is_lora = bool(args.lora_base_model)
    if accelerator.is_main_process:
        n_train = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"[finetune] trainable params {n_train / 1e6:.1f}M / {n_total / 1e6:.1f}M "
            f"({'LoRA on ' + args.lora_base_model if is_lora else args.trainable_models})"
        )

    if not FT.keep_text_encoder_on_gpu:
        _ddp_ignore_text_encoder(model)

    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    warmup = max(0, int(FT.lr_warmup_steps))
    schedule = {"type": "constant_with_warmup", "base_lr": float(args.learning_rate),
                "warmup_steps": warmup}

    def lr_at(step):
        # Linear warm-up then constant. Set by hand per optimizer update: a
        # prepared torch scheduler would advance num_processes times per
        # update and distort the warm-up; the schedule is a pure function of
        # the step, so the step in state.json fully restores it.
        return args.learning_rate * (min(1.0, (step + 1) / warmup) if warmup else 1.0)

    # ---- resume state (weights were already loaded via --lora_checkpoint) --
    resume = _read_ckpt_state(FT.resume_ckpt) if FT.resume_ckpt else None
    if resume is not None:
        if int(resume["dataset_pass_id"]) != int(args.dataset_pass_id):
            raise SystemExit(
                f"ERROR: checkpoint dataset_pass_id={resume['dataset_pass_id']} but this run "
                f"uses {args.dataset_pass_id}; the data order would not match"
            )
        optimizer.load_state_dict(
            torch.load(os.path.join(FT.resume_ckpt, "optimizer.pt"), map_location="cpu",
                       weights_only=False)
        )

    num_workers = int(args.dataset_num_workers or 0)
    loader_kwargs = {}
    if num_workers > 0 and args.dataset_prefetch_factor:
        loader_kwargs["prefetch_factor"] = int(args.dataset_prefetch_factor)
    # No sampler shuffle: the dataset already maps index -> a seed-shuffled,
    # per-epoch-rotated episode list with per-index RNG, so a sequential
    # sampler gives an exactly reproducible stream that resumes mid-epoch.
    # A private generator keeps iterator creation (which draws a worker base
    # seed) from consuming the global torch RNG that the training noise uses;
    # otherwise a resumed run's noise stream would be shifted by one draw.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=lambda x: x[0],
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(int(args.seed or 0)),
        **loader_kwargs,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    if not FT.keep_text_encoder_on_gpu:
        _text_encoder_to_cpu_with_cache(accelerator.unwrap_model(model))
    loss_val_dataset = _build_val_dataset(args, accelerator)
    n_val_draws = (
        len(loss_val_dataset.path) * max(1, FT.val_draws_per_episode) if loss_val_dataset else 0
    )

    global_step = int(getattr(args, "_resume_step_count", 0) or 0)
    start_epoch = int(args.start_epoch)
    skip_samples = 0
    if resume is not None:
        global_step = int(resume["global_step"])
        start_epoch = int(resume["epoch_id"])
        skip_samples = int(resume["samples_in_epoch"])
        rng_path = os.path.join(FT.resume_ckpt, f"rng_rank{accelerator.process_index}.pt")
        if int(resume["world_size"]) == accelerator.num_processes and os.path.isfile(rng_path):
            _set_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
        elif accelerator.is_main_process:
            print("[finetune] resume: world size changed; RNG streams re-seeded")
        accelerator.unwrap_model(model).step_count = global_step
        if accelerator.is_main_process:
            print(
                f"[finetune] resume from {FT.resume_ckpt}: step {global_step}, epoch "
                f"{start_epoch}, skipping {skip_samples} samples already seen this epoch"
            )
    model_logger.num_steps = global_step
    save_steps = int(args.save_steps) if args.save_steps else None
    max_steps = FT.max_train_steps
    log_path = os.path.join(log_dir, "train_log.jsonl")
    # wandb (auto: online if logged in and reachable, else offline under
    # <log_dir>/wandb) -> tensorboard -> no-op; main process only.
    metrics = MosaicMetricsLogger(args, log_dir, accelerator.is_main_process)
    metrics._batch_counter = global_step  # wandb step == optimizer step, also on resume
    run = getattr(metrics, "_wandb_run", None)
    wandb_id = run.id if run is not None else None
    if run is not None:
        # lets tools/log_eval_videos.py attach results to this same run
        with open(os.path.join(log_dir, "wandb_run.json"), "w") as f:
            json.dump({"id": run.id, "project": run.project, "entity": run.entity,
                       "mode": metrics.backend}, f)

    def write_log(record):
        if accelerator.is_main_process:
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def run_validation(step):
        if not n_val_draws:
            return
        t0 = time.time()
        draws = _distributed_fixed_draw_losses(
            accelerator, model, loss_val_dataset, n_val_draws, 5000
        )
        per_episode = {}
        for loss, name in draws:
            per_episode.setdefault(name, []).append(loss)
        mean = float(np.mean([loss for loss, _ in draws]))
        if accelerator.is_main_process:
            scalars = {"val/loss": mean}
            scalars.update({f"val_episode/{n}": float(np.mean(v)) for n, v in per_episode.items()})
            metrics.log_scalars(scalars, step=step)
            print(f"[finetune] validation step {step}: val/loss {mean:.4f} over "
                  f"{len(draws)} draws / {len(per_episode)} episodes ({time.time() - t0:.0f}s)",
                  flush=True)
        write_log({"step": step, "val_loss": mean,
                   "val_episodes": {n: float(np.mean(v)) for n, v in per_episode.items()}})

    def maybe_probe(step, force=False):
        if not FT.probe_every or (not force and step % FT.probe_every):
            return
        t0 = time.time()
        draws = _distributed_fixed_draw_losses(accelerator, model, dataset, FT.probe_samples, 1234)
        per_sample = [loss for loss, _ in draws]
        mean = float(np.mean(per_sample))
        if accelerator.is_main_process:
            scalars = {"probe/loss": mean}
            scalars.update({f"probe/loss_{i}": v for i, v in enumerate(per_sample)})
            metrics.log_scalars(scalars, step=step)
            print(f"[finetune] probe step {step}: loss {mean:.4f} ({time.time() - t0:.0f}s)", flush=True)
        write_log({"step": step, "probe_loss": mean, "probe_losses": per_sample})

    samples_per_batch = accelerator.num_processes  # batch size 1 per rank

    def checkpoint(epoch_id, samples_in_epoch):
        state = {
            "format": CKPT_FORMAT,
            "global_step": global_step,
            "epoch_id": epoch_id,
            "samples_in_epoch": samples_in_epoch,
            "dataset_pass_id": int(args.dataset_pass_id),
            "world_size": accelerator.num_processes,
            "lora": is_lora,
            "lr_schedule": dict(schedule, last_step=global_step),
            "wandb_run_id": wandb_id,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = _save_checkpoint(accelerator, model, optimizer, log_dir, state,
                                model_logger.remove_prefix_in_ckpt)
        metrics.log_scalars({"ckpt/saved": 1.0}, step=global_step)
        return path

    maybe_probe(global_step, force=True)
    run_validation(global_step)
    running, running_n, t_log = 0.0, 0, time.time()
    step_loss, step_loss_local, step_n = 0.0, 0.0, 0
    last_ckpt = last_val = time.time()
    stop = False
    epoch_id = start_epoch
    samples_in_epoch = 0

    for epoch_id in range(start_epoch, int(args.num_epochs)):
        pass_id = int(args.dataset_pass_id)
        dataset.set_epoch(
            epoch_id, rank=accelerator.process_index, world_size=accelerator.num_processes
        )
        # Accelerate re-calls dataset.set_epoch(<its own counter>) when the
        # loader starts iterating; sync that counter so resumes keep epoch_id.
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch_id)
        loader = dataloader
        samples_in_epoch = 0
        if skip_samples:
            skip_batches = skip_samples // samples_per_batch
            loader = accelerator.skip_first_batches(dataloader, skip_batches)
            # the skipping loader has its own iteration counter (starting at
            # 0) that it pushes into dataset.set_epoch; keep it on epoch_id
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch_id)
            samples_in_epoch = skip_batches * samples_per_batch
            skip_samples = 0
        for data in loader:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(global_step)
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients and args.max_grad_norm and args.max_grad_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            samples_in_epoch += samples_per_batch
            loss_mean = float(accelerator.reduce(loss.detach().float(), reduction="mean"))
            running += loss_mean
            running_n += 1
            step_loss += loss_mean
            step_loss_local += float(loss.detach())
            step_n += 1
            if not accelerator.sync_gradients:
                continue

            global_step += 1
            metrics.log_train_step(
                loss=step_loss / step_n,
                loss_local=step_loss_local / step_n,
                lr=optimizer.param_groups[0]["lr"],
                epoch=epoch_id,
                global_step=global_step,
                data=data,
                nonfinite=not torch.isfinite(loss.detach()).all().item(),
            )
            if grad_norm is not None:
                metrics.log_scalars({"train/grad_norm": float(grad_norm)}, step=global_step)
            step_loss, step_loss_local, step_n = 0.0, 0.0, 0
            if global_step % max(1, FT.log_every) == 0 and accelerator.is_main_process:
                record = {
                    "step": global_step,
                    "epoch": epoch_id,
                    "loss": running / max(1, running_n),
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": None if grad_norm is None else float(grad_norm),
                    "sec_per_step": (time.time() - t_log) / FT.log_every,
                }
                print(
                    f"[finetune] step {global_step} epoch {epoch_id} "
                    f"loss {record['loss']:.4f} lr {record['lr']:.2e} "
                    f"{record['sec_per_step']:.1f}s/step",
                    flush=True,
                )
                write_log(record)
                running, running_n, t_log = 0.0, 0, time.time()
            maybe_probe(global_step)
            if save_steps and global_step % save_steps == 0:
                model_logger.save_model(accelerator, model, f"step-{global_step}.safetensors")
            if max_steps and global_step >= max_steps:
                stop = True
            now = time.time()
            do_ckpt, do_val = _broadcast_flags(accelerator, [
                FT.ckpt_every_minutes and now - last_ckpt >= 60 * FT.ckpt_every_minutes,
                FT.val_every_minutes and now - last_val >= 60 * FT.val_every_minutes,
            ])
            if do_ckpt or (stop and FT.ckpt_every_minutes):
                checkpoint(epoch_id, samples_in_epoch)
                last_ckpt = time.time()
            if do_val:
                run_validation(global_step)
                last_val = time.time()
            if stop:
                break

        if not stop:
            model_logger.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")
            if accelerator.is_main_process:
                _write_train_state(log_dir, epoch_id, global_step, pass_id)
        if (
            args.run_validation
            and val_dataset is not None
            and (epoch_id + 1) % max(1, int(args.validation_interval)) == 0
        ):
            run_mosaic_validation(
                accelerator, val_dataset, model, log_dir, args=args, epoch_id=epoch_id
            )
            metrics.log_validation_done(epoch_id, global_step)
        if stop:
            break
    if not FT.ckpt_every_minutes:
        model_logger.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")
        if accelerator.is_main_process:
            _write_train_state(log_dir, epoch_id, global_step, int(args.dataset_pass_id))
    run_validation(global_step)
    metrics.finish()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"[finetune] done after {global_step} optimizer steps -> {log_dir}")


def main():
    global FT
    # Same as infer.py: never download, and resolve model ids such as
    # checkpoints/Wan2.2-TI2V-5B relative to the repo instead of ./models.
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", REPO_DIR)
    FT, rest = _parse_finetune_flags(sys.argv[1:])
    if FT.resume_ckpt:
        FT.resume_ckpt = os.path.abspath(FT.resume_ckpt.rstrip("/"))
        state = _read_ckpt_state(FT.resume_ckpt)
        # continue the same W&B run unless the caller chose one explicitly
        if state.get("wandb_run_id") and not os.environ.get("WANDB_RUN_ID"):
            os.environ["WANDB_RUN_ID"] = state["wandb_run_id"]
            os.environ.setdefault("WANDB_RESUME", "allow")
    rest = _resume_overrides(rest)
    sys.argv = [sys.argv[0]] + rest
    # import_module, not "from ...mosaic import main": the package re-exports
    # the main() function under the same name as the submodule.
    mosaic_main = importlib.import_module("examples.wanvideo.pipeline.mosaic.main")
    mosaic_main.run_mosaic_inference_task = run_mosaic_training_task
    mosaic_main.main()


if __name__ == "__main__":
    main()
