#!/usr/bin/env python3
"""Place videos side by side with a label bar above each panel.

Every panel is center-cropped to the panel aspect ratio and resized to
``--panel`` (default 1280x704, the Matrix-Game generation size), the same
crop-then-resize the pipeline applies to its 640x360 Craftium inputs, so
panels line up pixel for pixel. Inputs should share a frame rate; use
tools/export_episode_video.py first for raw 16 fps infer.py results.

    python tools/make_grid_video.py --out grid.mp4 \
        --video "Ground truth=hudnav-20k eval 200/rgb.mp4" \
        --video "Base model=outputs/.../result_600f_24fps.mp4"
"""

import argparse

import cv2
import imageio
import numpy as np


def read(path, limit):
    cap = cv2.VideoCapture(path)
    frames = []
    while limit is None or len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"ERROR: no frames decoded from {path}")
    return frames


def fit(frame, w, h):
    fh, fw = frame.shape[:2]
    if fw / fh > w / h:
        cw = int(round(fh * w / h))
        x = (fw - cw) // 2
        frame = frame[:, x:x + cw]
    else:
        ch = int(round(fw * h / w))
        y = (fh - ch) // 2
        frame = frame[y:y + ch]
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


def label_bar(text, w, height):
    bar = np.full((height, w, 3), 24, np.uint8)
    scale = 1.1
    while scale > 0.4:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        if tw <= w - 20:
            break
        scale -= 0.1
    cv2.putText(bar, text, ((w - tw) // 2, (height + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return bar


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video", action="append", required=True, help="label=path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--panel", default="1280x704")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    w, h = (int(v) for v in args.panel.lower().split("x"))
    bar_h, gap = 64, 8
    panels = []
    for item in args.video:
        label, path = item.split("=", 1)
        panels.append((label, read(path, args.frames)))
    n = min(len(frames) for _, frames in panels)
    heads = [label_bar(label, w, bar_h) for label, _ in panels]
    sep = np.full((h + bar_h, gap, 3), 255, np.uint8)

    out = []
    for i in range(n):
        cols = []
        for k, (_, frames) in enumerate(panels):
            if k:
                cols.append(sep)
            cols.append(np.vstack([heads[k], fit(frames[i], w, h)]))
        frame = np.hstack(cols)
        cv2.putText(frame, f"{i / args.fps:5.2f}s  frame {i}", (12, bar_h + h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        out.append(frame[..., ::-1])
    imageio.mimwrite(args.out, out, fps=args.fps, codec="libx264", quality=8,
                     macro_block_size=8)
    print(f"{args.out}: {n} frames, {out[0].shape[1]}x{out[0].shape[0]}, {n / args.fps:.2f}s")


if __name__ == "__main__":
    main()
