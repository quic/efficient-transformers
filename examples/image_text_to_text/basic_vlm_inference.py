# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

import requests
from PIL import Image
from transformers import AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText


def run_model(
    model_name,
    query,
    image_url,
    kv_offload=True,
    prefill_seq_len=32,
    ctx_len=512,
    generation_len=128,
    img_size=336,
    num_cores=16,
    num_devices=1,
):
    ## STEP 1: Load the Processor and Model

    processor = AutoProcessor.from_pretrained(model_name)

    # `kv_offload` determines Single QPC vs Dual QPC mode:
    # - Single QPC (kv_offload=False): Entire model runs in one QPC
    # - Dual QPC (kv_offload=True): Vision encoder and language model run in separate QPCs
    #   with outputs transferred via host for flexibility

    model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_name, attn_implementation="eager", kv_offload=kv_offload
    )

    ## STEP 2: Export & Compile the Model

    model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        img_size=img_size,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=False,
    )

    ## STEP 3: Load and Process the Inputs for Inference
    # Note: the message format would change for different model
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

    streamer = TextStreamer(processor.tokenizer)
    model.generate(inputs=inputs, streamer=streamer, generation_len=generation_len)


def main():
    parser = argparse.ArgumentParser(description="Vision-Language Model (VLM) inference")
    parser.add_argument(
        "--model-name",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
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
        "--kv-offload",
        action="store_true",
        default=True,
        help="Enable Dual QPC mode (vision encoder and LM in separate QPCs)",
    )
    parser.add_argument("--prefill-seq-len", type=int, default=128, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=3000, help="Context length")
    parser.add_argument("--generation-len", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--img-size", type=int, default=336, help="Image size for processing")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices")
    args = parser.parse_args()

    print(f"Running VLM inference with model: {args.model_name}")
    print(f"KV offload (Dual QPC mode): {args.kv_offload}")

    run_model(
        model_name=args.model_name,
        query=args.query,
        image_url=args.image_url,
        kv_offload=args.kv_offload,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        generation_len=args.generation_len,
        img_size=args.img_size,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
    )


if __name__ == "__main__":
    main()
