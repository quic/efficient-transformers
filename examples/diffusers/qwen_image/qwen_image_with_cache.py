# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""
QwenImage First Cache Example

This script demonstrates how to use the first-block cache optimization for the
QwenImage pipeline. The first cache optimization works by:

1. Running transformer_blocks[0] on CPU before each QAIC inference call
2. Computing the residual (output - input) of the first block
3. Comparing the residual to the previous step's residual
4. If they are similar (below cache_threshold), reusing the cached residuals
   from blocks[1:] instead of recomputing them on QAIC

This can significantly reduce inference time with minimal quality loss,
especially in the middle/later denoising steps where the model output
changes slowly.

Usage:
    python qwen_image_with_cache.py \
        --model_path /path/to/qwen-image-model \
        --prompt "A beautiful sunset over the ocean" \
        --height 464 \
        --width 832 \
        --num_inference_steps 50 \
        --cache_threshold 0.05 \
        --cache_warmup_steps 5

    # Without cache (standard inference):
    python qwen_image_with_cache.py \
        --model_path /path/to/qwen-image-model \
        --prompt "A beautiful sunset over the ocean" \
        --no_cache
"""

import argparse
import time

import torch

from QEfficient.diffusers.pipelines.qwen_image.pipeline_qwenimage import QEFFQwenImagePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="QwenImage inference with first-block cache optimization")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained QwenImage model or HuggingFace model ID",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful sunset over the ocean, photorealistic, high quality",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative text prompt for true CFG (optional)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=464,
        help="Height of the generated image in pixels",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Width of the generated image in pixels",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for guidance-distilled models",
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="True CFG scale (only used when negative_prompt is provided)",
    )

    # Cache arguments
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable first-block cache optimization (standard inference)",
    )
    parser.add_argument(
        "--cache_threshold",
        type=float,
        default=0.05,
        help=(
            "Similarity threshold for cache decision. "
            "Lower values = less aggressive caching (higher quality). "
            "Higher values = more aggressive caching (faster inference). "
            "Typical range: 0.01 - 0.1"
        ),
    )
    parser.add_argument(
        "--cache_warmup_steps",
        type=int,
        default=5,
        help=(
            "Number of initial denoising steps to always compute without cache. "
            "These steps have the largest changes and benefit least from caching."
        ),
    )

    # Compilation arguments
    parser.add_argument(
        "--compile_config",
        type=str,
        default=None,
        help="Path to custom compilation config JSON file",
    )
    parser.add_argument(
        "--parallel_compile",
        action="store_true",
        help="Compile modules in parallel",
    )

    # Output arguments
    parser.add_argument(
        "--output_path",
        type=str,
        default="qwen_image_output.png",
        help="Path to save the generated image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    enable_first_cache = not args.no_cache

    print("=" * 60)
    print("QwenImage First Cache Example")
    print("=" * 60)
    print(f"Model path:          {args.model_path}")
    print(f"Prompt:              {args.prompt}")
    print(f"Resolution:          {args.height}x{args.width}")
    print(f"Inference steps:     {args.num_inference_steps}")
    print(f"First cache:         {'ENABLED' if enable_first_cache else 'DISABLED'}")
    if enable_first_cache:
        print(f"  Cache threshold:   {args.cache_threshold}")
        print(f"  Cache warmup:      {args.cache_warmup_steps} steps")
    print("=" * 60)

    # Load pipeline with first cache enabled/disabled
    print("\nLoading pipeline...")
    pipeline = QEFFQwenImagePipeline.from_pretrained(
        args.model_path,
        enable_first_cache=enable_first_cache,
    )

    # Set up random generator for reproducibility
    generator = torch.Generator().manual_seed(args.seed)

    # Run inference
    print("\nRunning inference...")
    start_time = time.perf_counter()

    result = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
        generator=generator,
        custom_config_path=args.compile_config,
        parallel_compile=args.parallel_compile,
        # Cache parameters (only used when enable_first_cache=True)
        cache_threshold=args.cache_threshold if enable_first_cache else None,
        cache_warmup_steps=args.cache_warmup_steps if enable_first_cache else None,
    )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Save output image
    image = result.images[0]
    image.save(args.output_path)
    print(f"\nImage saved to: {args.output_path}")

    # Print performance summary
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"Total inference time: {total_time:.2f}s")

    if result.pipeline_module:
        for module_perf in result.pipeline_module:
            if isinstance(module_perf.perf, list) and len(module_perf.perf) > 0:
                avg_time = sum(module_perf.perf) / len(module_perf.perf)
                total_module_time = sum(module_perf.perf)
                print(f"\n{module_perf.module_name}:")
                print(f"  Total time:   {total_module_time:.2f}s")
                print(f"  Avg per step: {avg_time:.3f}s")
                print(f"  Steps:        {len(module_perf.perf)}")
            elif isinstance(module_perf.perf, float):
                print(f"\n{module_perf.module_name}:")
                print(f"  Time: {module_perf.perf:.3f}s")

    print("=" * 60)


if __name__ == "__main__":
    main()
