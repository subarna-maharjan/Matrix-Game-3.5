#!/usr/bin/env python3
"""Merge a finetune.py LoRA checkpoint into a full DiT checkpoint.

The result is a drop-in replacement for first-person.safetensors and can be
passed to ``infer.py --ckpt`` (or ``trained_dit``):

    python tools/merge_lora.py \
        --base checkpoints/first-person.safetensors \
        --lora outputs/finetune/<run>/epoch-3.safetensors \
        --out  outputs/finetune/<run>/epoch-3-merged.safetensors

W' = W + (alpha / rank) * B @ A. finetune.py uses peft's default
alpha = rank, so the scale is 1 unless --alpha is given.
"""

import argparse
import sys

import torch
from safetensors.torch import load_file, save_file

PREFIXES = ("", "pipe.dit.", "dit.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=None,
                        help="LoRA alpha used in training (default: rank).")
    args = parser.parse_args()

    base = load_file(args.base)
    lora = load_file(args.lora)

    # base keys may or may not carry a module prefix
    stripped_to_base = {}
    for key in base:
        for prefix in PREFIXES:
            if prefix and key.startswith(prefix):
                stripped_to_base[key[len(prefix):]] = key
                break
        else:
            stripped_to_base[key] = key

    merged = 0
    for a_key in sorted(k for k in lora if ".lora_A." in k):
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        module = a_key.split(".lora_A.")[0]
        for prefix in PREFIXES:
            if module.startswith(prefix):
                module = module[len(prefix):]
        target = stripped_to_base.get(module + ".weight")
        if target is None or b_key not in lora:
            sys.exit(f"ERROR: cannot match LoRA module {module!r} to a base weight")
        A = lora[a_key].float()
        B = lora[b_key].float()
        rank = A.shape[0]
        scale = (args.alpha if args.alpha is not None else rank) / rank
        W = base[target]
        base[target] = (W.float() + scale * (B @ A)).to(W.dtype)
        merged += 1

    extra = [k for k in lora if ".lora_A." not in k and ".lora_B." not in k]
    if extra:
        print(f"[merge] note: {len(extra)} non-LoRA tensors in --lora were ignored")
    if merged == 0:
        sys.exit("ERROR: no LoRA tensors found in --lora")
    save_file(base, args.out)
    print(f"[merge] merged {merged} LoRA modules -> {args.out}")


if __name__ == "__main__":
    main()
