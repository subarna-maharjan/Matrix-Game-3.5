#!/usr/bin/env python3
"""Precompute per-frame metric depth for converted scenes with Depth-Anything-3.

Training reads memory depth from ``<scene>/depth/video.npz`` (inference
estimates it at runtime instead), so every training scene needs one. Writes
``frame_xxxxx`` keys in the ``npz_log_u16`` encoding understood by
``diffsynth.core.data.unified_dataset`` plus ``depth/metadata.json``, then
touches ``<scene>/_DONE``.

Depth is stored at ``--depth-scale`` of the video resolution: the memory
fuse resamples it to the latent grid (44x80 at 704x1280) with nearest
neighbour, so full resolution is not needed.

``--align-to-poses`` passes the known camera poses to DA3, which rescales
its depth so it agrees with your metric trajectory (Umeyama scale over each
chunk). Chunks whose camera barely translates fall back to DA3's own metric
scale because the alignment is ill-posed there.

Usage (one GPU; shard across a Slurm array with --num-shards/--shard-id):
    DA3_MODEL_PATH=checkpoints/DA3NESTED-GIANT-LARGE-1.1 \
    python tools/craftium_depth_da3.py --scenes data/craftium_mg/scenes
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
DEPTH_MIN, DEPTH_MAX, BIT_DEPTH = 0.1, 1000.0, 16


def load_estimator(device):
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    if not os.environ.get("DA3_MODEL_PATH"):
        default = REPO_DIR / "checkpoints" / "DA3NESTED-GIANT-LARGE-1.1"
        if not default.exists():
            sys.exit(f"ERROR: set DA3_MODEL_PATH or place DA3 at {default}")
        os.environ["DA3_MODEL_PATH"] = str(default)
    from examples.wanvideo.pipeline.mosaic.video_io import _init_da3_depth_estimator

    return _init_da3_depth_estimator(device=device)


def read_frames(video_path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return frames


def encode_log_u16(depth):
    max_code = float((1 << BIT_DEPTH) - 1)
    d = np.clip(np.nan_to_num(depth, nan=DEPTH_MAX, posinf=DEPTH_MAX), DEPTH_MIN, DEPTH_MAX)
    t = (np.log(d) - np.log(DEPTH_MIN)) / (np.log(DEPTH_MAX) - np.log(DEPTH_MIN))
    return np.round(t * max_code).astype(np.uint16)


def estimate_scene_depth(estimator, scene_dir, args):
    import cv2
    import torch
    from PIL import Image

    frames = read_frames(scene_dir / "rgb" / "video.mp4")
    c2w = np.load(scene_dir / "pose" / "video.npz")["data"].astype(np.float64)
    intr = np.load(scene_dir / "intrinsics" / "video.npz")["data"].astype(np.float64)
    n = min(len(frames), c2w.shape[0])
    h, w = frames[0].shape[:2]
    out_h = max(1, int(round(h * args.depth_scale)))
    out_w = max(1, int(round(w * args.depth_scale)))

    depths = []
    n_aligned = 0
    for start in range(0, n, args.chunk):
        end = min(n, start + args.chunk)
        pil = [Image.fromarray(f) for f in frames[start:end]]
        kwargs = {}
        if args.align_to_poses:
            trans_extent = np.ptp(c2w[start:end, :3, 3], axis=0).max()
            if end - start >= 3 and trans_extent >= args.min_align_translation:
                K = np.zeros((end - start, 3, 3))
                K[:, 0, 0], K[:, 1, 1] = intr[start:end, 0], intr[start:end, 1]
                K[:, 0, 2], K[:, 1, 2] = intr[start:end, 2], intr[start:end, 3]
                K[:, 2, 2] = 1.0
                kwargs = {
                    "extrinsics": np.linalg.inv(c2w[start:end]).astype(np.float32),
                    "intrinsics": K.astype(np.float32),
                    "align_to_input_ext_scale": True,
                }
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                pred = estimator.inference(
                    pil, use_ray_pose=False, process_res=args.process_res, **kwargs
                )
                n_aligned += bool(kwargs)
            except ValueError as exc:
                # Umeyama alignment fails when DA3's own trajectory for the
                # chunk is degenerate (e.g. player standing still)
                if not kwargs:
                    raise
                print(f"  chunk {start}-{end}: pose alignment failed ({exc}); using DA3 metric depth")
                pred = estimator.inference(
                    pil, use_ray_pose=False, process_res=args.process_res
                )
        for d in np.asarray(pred.depth, dtype=np.float32):
            depths.append(cv2.resize(d, (out_w, out_h), interpolation=cv2.INTER_NEAREST))

    depth_dir = scene_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = depth_dir / "video.tmp.npz"
    np.savez_compressed(
        tmp_path, **{f"frame_{i:05d}": encode_log_u16(d) for i, d in enumerate(depths)}
    )
    os.replace(tmp_path, depth_dir / "video.npz")
    with open(depth_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "format": "npz_log_u16",
                "bit_depth": BIT_DEPTH,
                "depth_min": DEPTH_MIN,
                "depth_max": DEPTH_MAX,
                "estimator": "DA3NESTED-GIANT-LARGE-1.1",
                "aligned_to_poses": bool(args.align_to_poses),
                "height": out_h,
                "width": out_w,
            },
            f,
            indent=1,
        )
    (scene_dir / "_DONE").touch()
    stacked = np.stack(depths)
    return len(depths), float(np.median(stacked)), n_aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenes", required=True, help="<out>/scenes from the converter.")
    parser.add_argument("--chunk", type=int, default=32, help="Frames per DA3 call.")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--depth-scale", type=float, default=0.5)
    parser.add_argument("--align-to-poses", action="store_true")
    parser.add_argument(
        "--min-align-translation", type=float, default=1.0,
        help="Metres of camera travel a chunk needs before pose alignment is used.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Defaults to $SLURM_ARRAY_TASK_ID or 0.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    shard_id = args.shard_id
    if shard_id is None:
        shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    scenes = sorted(
        p for p in Path(args.scenes).iterdir()
        if (p / "pose" / "video.npz").is_file() and (p / "rgb" / "video.mp4").exists()
    )
    scenes = scenes[shard_id :: args.num_shards]
    todo = [
        s for s in scenes
        if args.overwrite or not (s / "_DONE").exists() or not (s / "depth" / "video.npz").exists()
    ]
    print(f"[depth] shard {shard_id}/{args.num_shards}: {len(todo)}/{len(scenes)} scenes to process")
    if not todo:
        return

    estimator = load_estimator("cuda")
    failed = 0
    for i, scene_dir in enumerate(todo, start=1):
        t0 = time.time()
        try:
            n, median, n_aligned = estimate_scene_depth(estimator, scene_dir, args)
            print(
                f"[{i}/{len(todo)}] {scene_dir.name}: {n} frames, median depth "
                f"{median:.2f} m, pose-aligned chunks {n_aligned}, {time.time() - t0:.0f}s"
            )
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(todo)}] ERROR {scene_dir.name}: {exc}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
