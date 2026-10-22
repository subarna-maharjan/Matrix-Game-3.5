#!/usr/bin/env python3
"""Compare generated videos with ground truth and log them to a finetune run.

Builds one side-by-side video (ground truth | each generated video), computes
per-video PSNR against the ground truth (frame i of every infer.py result is
aligned with frame i of the ground-truth episode), saves both next to the run
and, when the run has ``wandb_run.json``, attaches them to that W&B run.

    python tools/log_eval_videos.py --run-dir outputs/finetune/<run> \
        --gt eval_inputs/<id>/gt.mp4 \
        --video base=outputs/eval/first_person/base/result.mp4 \
        --video finetuned=outputs/eval/first_person/finetuned/result.mp4
"""

import argparse
import json
import os

import cv2
import numpy as np


def read_video(path, size):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--video", action="append", required=True, help="name=path")
    parser.add_argument("--width", type=int, default=640, help="Per-panel width.")
    parser.add_argument("--tag", default="eval")
    args = parser.parse_args()

    size = (args.width, args.width * 9 // 16)
    gt = read_video(args.gt, size)
    videos = []
    for item in args.video:
        name, path = item.split("=", 1)
        videos.append((name, read_video(path, size)))
    n = min([len(gt)] + [len(v) for _, v in videos])

    scores = {}
    for name, frames in videos:
        # frame 0 is the conditioning frame itself; score the generated ones
        per_frame = [psnr(frames[i], gt[i]) for i in range(1, n)]
        scores[f"{args.tag}/psnr_{name}"] = float(np.mean(per_frame))
    print(json.dumps(scores, indent=1))

    panels = [("ground truth", gt)] + videos
    out = np.concatenate([p[:n] for _, p in panels], axis=2)
    for k, (label, _) in enumerate(panels):
        for frame in out:
            cv2.putText(frame, label, (k * size[0] + 10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
    rgb = out[..., ::-1]

    os.makedirs(args.run_dir, exist_ok=True)
    out_path = os.path.join(args.run_dir, f"{args.tag}_side_by_side.mp4")
    import imageio

    imageio.mimwrite(out_path, list(rgb), fps=24, codec="libx264", quality=7,
                     macro_block_size=8)
    with open(os.path.join(args.run_dir, f"{args.tag}_scores.json"), "w") as f:
        json.dump(scores, f, indent=1)
    print(f"saved {out_path}")

    run_file = os.path.join(args.run_dir, "wandb_run.json")
    if not os.path.exists(run_file):
        print("no wandb_run.json; skipping W&B upload")
        return
    import wandb

    info = json.load(open(run_file))
    mode = "offline" if "offline" in str(info.get("mode")) else "online"
    run = wandb.init(id=info["id"], project=info["project"], entity=info.get("entity") or None,
                     dir=args.run_dir, resume="allow", mode=mode)
    run.log({f"{args.tag}/side_by_side": wandb.Video(out_path, format="mp4"), **scores})
    run.summary.update(scores)
    run.finish()
    print(f"logged to W&B run {info['id']} ({mode})")


if __name__ == "__main__":
    main()
