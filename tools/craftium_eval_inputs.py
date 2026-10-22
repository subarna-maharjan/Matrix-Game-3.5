#!/usr/bin/env python3
"""Turn converted Craftium scenes into infer.py inputs for side-by-side eval.

For each scene writes ``<out>/<scene_id>/{input.png, camera.npz, prompt.txt,
gt.mp4}``: the anchor frame, the trajectory from that frame on, the caption
and a symlink to the ground-truth video. Generate with the base model and
with a fine-tuned checkpoint, then compare both against gt.mp4:

    python infer.py --person first --num-blocks 7 \
        --image <out>/<id>/input.png --camera <out>/<id>/camera.npz \
        --prompt-file <out>/<id>/prompt.txt [--ckpt merged.safetensors]

Usage:
    python tools/craftium_eval_inputs.py --root data/craftium_eval --out eval_inputs
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, help="Converter output root (has scenes/, captions/).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--start-frame", type=int, default=0, help="Anchor frame index.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import cv2

    root, out = Path(args.root), Path(args.out)
    scenes = sorted(p for p in (root / "scenes").iterdir() if (p / "pose" / "video.npz").is_file())
    for scene in scenes[: args.limit]:
        dst = out / scene.name
        dst.mkdir(parents=True, exist_ok=True)
        s = args.start_frame
        cap = cv2.VideoCapture(str(scene / "rgb" / "video.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"ERROR {scene.name}: cannot read frame {s}")
            continue
        cv2.imwrite(str(dst / "input.png"), frame)
        c2w = np.load(scene / "pose" / "video.npz")["data"][s:]
        intr = np.load(scene / "intrinsics" / "video.npz")["data"][s:]
        np.savez(dst / "camera.npz", extrinsics_c2w=c2w, intrinsics=intr)
        caption = json.load(open(root / "captions" / scene.name / "video.json"))
        (dst / "prompt.txt").write_text(caption["detailed"]["0"]["dynamic"] + "\n")
        gt = dst / "gt.mp4"
        if gt.is_symlink() or gt.exists():
            gt.unlink()
        gt.symlink_to((scene / "rgb" / "video.mp4").resolve())
        blocks = (len(c2w) - 1) // 84
        print(f"{scene.name}: {len(c2w)} poses from frame {s} -> up to --num-blocks {blocks}")


if __name__ == "__main__":
    main()
