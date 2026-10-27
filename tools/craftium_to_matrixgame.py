#!/usr/bin/env python3
"""Convert Craftium/Minetest episodes into Matrix-Game-3.5 training scenes.

Each source episode is a directory holding at least ``rgb.mp4`` and
``data.npz`` (Craftium recorder output). ``sha256.txt`` is used as the
scene id when present; ``level_metadata.json`` / ``voxel_t0.glb`` are ignored.

Writes the layout read by ``diffsynth.core.data.da3_video_index``:

    <out>/scenes/<scene_id>/rgb/video.mp4          symlink (or copy) of rgb.mp4
    <out>/scenes/<scene_id>/pose/video.npz         data: (N,4,4) camera-to-world
    <out>/scenes/<scene_id>/intrinsics/video.npz   data: (N,4) [fx,fy,cx,cy] px
    <out>/captions/<scene_id>/video.json           single-segment caption

Depth (``depth/video.npz``) and the ``_DONE`` marker are written afterwards
by ``tools/craftium_depth_da3.py``.

Camera convention: Craftium's ``extrinsics_global`` is world-to-camera in
OpenCV axes (x right, y down, z forward) in a Z-up metric world (1 node =
1 m), so it only needs inverting. ``intrinsics`` are normalised by image
width/height and are converted to pixels here.

Usage:
    python tools/craftium_to_matrixgame.py \
        --src "/path/to/hudnav-20k" --out data/craftium_mg
    # one split of a dataset with a splits.json (episodes in one flat folder)
    python tools/craftium_to_matrixgame.py \
        --src data/hudnav-20k/OpenWorldCreativeHud-v0 \
        --splits data/hudnav-20k/motworld-hudnav-20k.splits.json \
        --split eval --first 16 --out data/hudnav20k_mg/val16
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

DEFAULT_CAPTION = (
    "First-person view exploring a blocky Minetest voxel world with grassy "
    "hills, trees, water and a bright sky. The inventory hotbar is shown at "
    "the bottom of the screen and the player holds a block in the lower "
    "right corner."
)


def episodes_from_splits(root, splits_path, split, first=None):
    """Episode dirs for one split, in the order listed in splits.json."""
    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)
    names = [e["name"] for e in splits["episodes"] if e["split"] == split]
    if first is not None:
        names = names[:first]
    root = Path(root)
    missing = [n for n in names if not (root / n / "rgb.mp4").is_file()]
    if missing:
        sys.exit(f"ERROR: {len(missing)} '{split}' episodes missing under {root}, e.g. {missing[:5]}")
    return [root / n for n in names]


def find_episodes(src_paths):
    episodes = []
    for src in src_paths:
        src = Path(src)
        if (src / "rgb.mp4").is_file() and (src / "data.npz").is_file():
            episodes.append(src)
            continue
        for rgb in sorted(src.rglob("rgb.mp4")):
            if (rgb.parent / "data.npz").is_file():
                episodes.append(rgb.parent)
    return list(dict.fromkeys(episodes))


def scene_id_for(episode_dir):
    sha_path = episode_dir / "sha256.txt"
    if sha_path.is_file():
        text = sha_path.read_text(encoding="utf-8").strip().split()
        if text and re.fullmatch(r"[0-9a-fA-F]{16,128}", text[0]):
            return text[0].lower()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_dir.name).strip("_")


def video_size_and_frames(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return w, h, n, fps


def convert_camera(z, width, height):
    w2c = np.asarray(z["extrinsics_global"], dtype=np.float64)
    if w2c.ndim != 3 or w2c.shape[1:] != (4, 4):
        raise ValueError(f"extrinsics_global must be (N,4,4), got {w2c.shape}")
    c2w = np.linalg.inv(w2c).astype(np.float32)

    K = np.asarray(z["intrinsics"], dtype=np.float64)
    if K.ndim != 3 or K.shape[1:] != (3, 3):
        raise ValueError(f"intrinsics must be (N,3,3), got {K.shape}")
    normalised = float(np.abs(K[:, 0, 2]).max()) <= 1.0
    sx, sy = (float(width), float(height)) if normalised else (1.0, 1.0)
    intr = np.stack(
        [K[:, 0, 0] * sx, K[:, 1, 1] * sy, K[:, 0, 2] * sx, K[:, 1, 2] * sy],
        axis=1,
    ).astype(np.float32)
    return c2w, intr


def check_camera(c2w):
    R = c2w[:, :3, :3].astype(np.float64)
    det = np.linalg.det(R)
    ortho = np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max()
    if np.abs(det - 1.0).max() > 1e-3 or ortho > 1e-3:
        raise ValueError(
            f"camera rotations are not proper rotations "
            f"(det range {det.min():.4f}..{det.max():.4f}, ortho err {ortho:.2e})"
        )


def write_caption(caption_root, scene_id, text):
    cap_dir = caption_root / scene_id
    cap_dir.mkdir(parents=True, exist_ok=True)
    payload = {"detailed": {"0": {"start": "", "dynamic": text}}}
    with open(cap_dir / "video.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def convert_episode(episode_dir, out_root, caption_text, copy_rgb, overwrite):
    scene_id = scene_id_for(episode_dir)
    scene_dir = out_root / "scenes" / scene_id
    pose_path = scene_dir / "pose" / "video.npz"
    intr_path = scene_dir / "intrinsics" / "video.npz"
    if pose_path.exists() and intr_path.exists() and not overwrite:
        return scene_id, "skipped (exists)"

    rgb_src = (episode_dir / "rgb.mp4").resolve()
    width, height, n_video, fps = video_size_and_frames(rgb_src)
    with np.load(episode_dir / "data.npz", allow_pickle=False) as z:
        c2w, intr = convert_camera(z, width, height)
    check_camera(c2w)

    n = min(n_video, c2w.shape[0])
    if n_video != c2w.shape[0]:
        print(
            f"[warn] {scene_id}: {n_video} video frames vs {c2w.shape[0]} poses; "
            f"truncating poses to {n}"
        )
    c2w, intr = c2w[:n], intr[:n]

    for sub in ("rgb", "pose", "intrinsics"):
        (scene_dir / sub).mkdir(parents=True, exist_ok=True)
    rgb_dst = scene_dir / "rgb" / "video.mp4"
    if rgb_dst.exists() or rgb_dst.is_symlink():
        rgb_dst.unlink()
    if copy_rgb:
        shutil.copy2(rgb_src, rgb_dst)
    else:
        rgb_dst.symlink_to(rgb_src)

    np.savez(pose_path, data=c2w)
    np.savez(intr_path, data=intr)
    write_caption(out_root / "captions", scene_id, caption_text)

    fx = intr[:, 0]
    return scene_id, (
        f"{n} frames {width}x{height}@{fps:g}fps "
        f"fx {fx.min():.1f}..{fx.max():.1f}px"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--src", nargs="+", required=True,
        help="Episode directories, or roots searched recursively for rgb.mp4+data.npz.",
    )
    parser.add_argument("--out", required=True, help="Output dataset root.")
    parser.add_argument(
        "--caption-text", default=DEFAULT_CAPTION,
        help="Caption used for every episode (single segment).",
    )
    parser.add_argument(
        "--copy-rgb", action="store_true",
        help="Copy rgb.mp4 instead of symlinking it.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--splits", default=None,
        help="splits.json; with --split, convert only that split (single --src root).",
    )
    parser.add_argument("--split", default=None, help="train | eval | unused")
    parser.add_argument(
        "--first", type=int, default=None,
        help="With --splits: only the first N episodes of the split, in file order.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Convert episodes shard-id, shard-id+num-shards, ...")
    args = parser.parse_args()

    if args.splits:
        if not args.split or len(args.src) != 1:
            sys.exit("ERROR: --splits needs --split and exactly one --src root")
        episodes = episodes_from_splits(args.src[0], args.splits, args.split, args.first)
    else:
        episodes = find_episodes(args.src)
    if args.limit is not None:
        episodes = episodes[: args.limit]
    episodes = episodes[args.shard_id :: args.num_shards]
    if not episodes:
        sys.exit(f"ERROR: no episodes (rgb.mp4 + data.npz) found under {args.src}")

    out_root = Path(args.out).resolve()
    ok = failed = 0
    for i, episode_dir in enumerate(episodes, start=1):
        try:
            scene_id, info = convert_episode(
                episode_dir, out_root, args.caption_text, args.copy_rgb, args.overwrite
            )
            ok += 1
            print(f"[{i}/{len(episodes)}] {scene_id}: {info}")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(episodes)}] ERROR {episode_dir}: {exc}")
    print(f"Done: {ok} converted, {failed} failed -> {out_root}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
