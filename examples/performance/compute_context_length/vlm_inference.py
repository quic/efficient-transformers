# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Vision-Language Model (VLM) inference with Compute Context Length (CCL) optimization.

This example demonstrates how to use CCL optimization for vision-language models.
CCL allows using different context lengths during prefill and decode phases,
reducing memory footprint and computation while maintaining support for longer contexts.
"""

import argparse

import requests
from PIL import Image
from transformers import AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText


def run_model(
    model_name,
    query,
    image_url,
    hf_token=None,
    kv_offload=True,
    prefill_seq_len=32,
    ctx_len=8192,
    ccl_enabled=False,
    comp_ctx_lengths_prefill=None,
    comp_ctx_lengths_decode=None,
    generation_len=128,
    img_size=560,
    num_cores=16,
    num_devices=4,
):
    """
    Run VLM inference with CCL optimization.

    Args:
        model_name: HuggingFace model ID
        query: Text query about the image
        image_url: URL of the image to process
        hf_token: HuggingFace token for gated models
        kv_offload: Enable Dual QPC mode (vision encoder and LM in separate QPCs)
        prefill_seq_len: Prefill sequence length
        ctx_len: Maximum context length
        comp_ctx_lengths_prefill: List of context lengths for prefill phase
        comp_ctx_lengths_decode: List of context lengths for decode phase
        generation_len: Number of tokens to generate
        img_size: Image size for processing
        num_cores: Number of cores for compilation
        num_devices: Number of devices to use
    """
    print(f"Loading model: {model_name}")
    print(f"KV offload (Dual QPC mode): {kv_offload}")
    print("CCL Configuration:")
    print(f"  - Prefill context lengths: {comp_ctx_lengths_prefill}")
    print(f"  - Decode context lengths: {comp_ctx_lengths_decode}")
    print(f"  - Max context length: {ctx_len}")

    ## STEP 1: Load the Processor and Model

    processor = AutoProcessor.from_pretrained(model_name, token=hf_token)

    # `kv_offload` determines Single QPC vs Dual QPC mode:
    # - Single QPC (kv_offload=False): Entire model runs in one QPC
    # - Dual QPC (kv_offload=True): Vision encoder and language model run in separate QPCs
    #   with outputs transferred via host for flexibility

    model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_name,
        token=hf_token,
        attn_implementation="eager",
        kv_offload=kv_offload,
        qaic_config={
            "ccl_enabled": ccl_enabled,
        },
    )

    ## STEP 2: Export & Compile the Model

    print("\nCompiling model...")
    qpc_path = model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        img_size=img_size,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=False,
        comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    )
    print(f"Model compiled successfully to: {qpc_path}")

    ## STEP 3: Load and Process the Inputs for Inference

    print(f"\nLoading image from: {image_url}")
    image = Image.open(requests.get(image_url, stream=True).raw)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ],
        }
    ]
    input_text = [processor.apply_chat_template(messages, add_generation_prompt=True)]

    inputs = processor(
        text=input_text,
        images=image,
        return_tensors="pt",
        add_special_tokens=False,
        padding="max_length",
        max_length=prefill_seq_len,
    )

    ## STEP 4: Run Inference on the Compiled Model

    print(f"\nQuery: {query}")
    print("Generated response:")
    streamer = TextStreamer(processor.tokenizer)
    output_statistics = model.generate(inputs=inputs, streamer=streamer, generation_len=generation_len)

    print(f"Tokens generated: {len(output_statistics.generated_ids[0])}")


def main():
    parser = argparse.ArgumentParser(
        description="Vision-Language Model (VLM) inference with Compute Context Length (CCL) optimization"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="HuggingFace VLM model ID",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Describe this image.",
        help="Text query/question about the image",
    )
    parser.add_argument(
        "--image-url",
        type=str,
        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
        help="URL of the image to process",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for accessing gated models",
    )
    parser.add_argument(
        "--kv-offload",
        action="store_true",
        default=True,
        help="Enable Dual QPC mode (vision encoder and LM in separate QPCs)",
    )
    parser.add_argument(
        "--prefill-seq-len",
        type=int,
        default=32,
        help="Prefill sequence length",
    )
    parser.add_argument(
        "--ctx-len",
        type=int,
        default=8192,
        help="Maximum context length",
    )
    parser.add_argument(
        "--ccl-enabled",
        action="store_true",
        help="Enable compute-context-length (CCL) feature",
    )
    parser.add_argument(
        "--comp-ctx-lengths-prefill",
        type=lambda x: [int(i) for i in x.split(",")],
        default="4096",
        help="Comma-separated list of context lengths for prefill phase (e.g., '4096')",
    )
    parser.add_argument(
        "--comp-ctx-lengths-decode",
        type=lambda x: [int(i) for i in x.split(",")],
        default="6144,8192",
        help="Comma-separated list of context lengths for decode phase (e.g., '6144,8192')",
    )
    parser.add_argument(
        "--generation-len",
        type=int,
        default=128,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=560,
        help="Image size for processing",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=16,
        help="Number of cores for compilation",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=4,
        help="Number of devices to use",
    )
    args = parser.parse_args()

    run_model(
        model_name=args.model_name,
        query=args.query,
        image_url=args.image_url,
        hf_token=args.hf_token,
        kv_offload=args.kv_offload,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        ccl_enabled=args.ccl_enabled,
        comp_ctx_lengths_prefill=args.comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=args.comp_ctx_lengths_decode,
        generation_len=args.generation_len,
        img_size=args.img_size,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
    )


if __name__ == "__main__":
    main()
