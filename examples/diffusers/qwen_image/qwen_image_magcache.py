# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""Qwen-Image generation example with optional MagCache runtime."""

import argparse

import torch

from QEfficient import QEffQwenImagePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Qwen image with optional MagCache.")
    parser.add_argument("--prompt", type=str, default="A cinematic photo of a coffee shop street in rain")
    parser.add_argument("--negative-prompt", type=str, default="low quality, blurry")
    parser.add_argument("--width", type=int, default=1664)
    parser.add_argument("--height", type=int, default=928)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--use-magcache", action="store_true", help="Enable MagCache runtime.")
    parser.add_argument("--magcache-thresh", type=float, default=0.06)
    parser.add_argument("--magcache-K", type=int, default=2)
    parser.add_argument("--magcache-retention-ratio", type=float, default=0.2)
    parser.add_argument(
        "--magcache-verbose",
        action="store_true",
        help="Print per-call MagCache diff/decision logs.",
    )
    parser.add_argument("--output", type=str, default="qwen_image_magcache.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipe = QEffQwenImagePipeline.from_pretrained("Qwen/Qwen-Image")
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        parallel_compile=True,
        max_sequence_length=args.max_sequence_length,
        use_magcache=args.use_magcache,
        magcache_thresh=args.magcache_thresh,
        magcache_K=args.magcache_K,
        magcache_retention_ratio=args.magcache_retention_ratio,
        magcache_verbose=args.magcache_verbose,
    )

    output.images[0].save(args.output)
    print(output)


if __name__ == "__main__":
    main()
