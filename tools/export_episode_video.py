#!/usr/bin/env python3
"""Re-time and resize an infer.py result to match the source episode.

infer.py writes ``result.mp4`` at 16 fps and includes frames generated past
the end of the trajectory (held last pose). This writes the first ``--frames``
frames at ``--fps`` at the generated size, plus copies at each ``--size``:

    python tools/export_episode_video.py --src .../result.mp4 --frames 600 \
        --size 640x360
    # -> result_600f_24fps.mp4, result_600f_24fps_640x360.mp4

Resizing 1280x704 (or 640x352) to 640x360 stretches the height by ~2%, the
inverse of the 4-row top/bottom crop the pipeline applies to 640x360 input.
"""

import argparse
import os

import cv2
import imageio


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--size", action="append", default=[], help="WxH, repeatable")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.src)
    frames = []
    while len(frames) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame[..., ::-1])
    cap.release()
    if len(frames) < args.frames:
        print(f"[export] note: only {len(frames)} frames available")

    stem = os.path.splitext(args.src)[0] + f"_{len(frames)}f_{args.fps:g}fps"
    outputs = [(f"{stem}.mp4", None)]
    for size in args.size:
        w, h = (int(v) for v in size.lower().split("x"))
        outputs.append((f"{stem}_{w}x{h}.mp4", (w, h)))
    for path, size in outputs:
        out = frames if size is None else [
            cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames
        ]
        imageio.mimwrite(path, out, fps=args.fps, codec="libx264", quality=8,
                         macro_block_size=8)
        h, w = out[0].shape[:2]
        print(f"[export] {path}: {len(out)} frames {w}x{h} @ {args.fps:g} fps "
              f"({len(out) / args.fps:.2f}s)")


if __name__ == "__main__":
    main()
