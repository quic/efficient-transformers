# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Dynamo-based export and inference for Qwen3-VL-MoE on Cloud AI 100.

Requires PyTorch >= 2.13. Install dependencies before running:
    pip install -r examples/dynamo/image_text_to_text/requirements.txt
"""

import argparse

import requests
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.utils import constants


def load_image(image_url: str, width: int, height: int) -> Image.Image:
    """Load a remote image, falling back to a deterministic local image."""
    try:
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")
    except requests.RequestException:
        return Image.new("RGB", (width, height), color=(120, 70, 200))


def maybe_reduce_config(config, *, reduce_layers: bool, vision_depth: int, text_layers: int):
    """Apply the small bring-up config used by the Qwen3-VL-MoE example."""
    if not reduce_layers:
        return config

    config.vision_config.depth = vision_depth
    config.text_config.num_hidden_layers = text_layers
    config.vision_config.deepstack_visual_indexes = [vision_depth - 1]
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Dynamo-based dual-QPC VLM export and inference for Qwen3-VL-MoE on Cloud AI 100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--prompt", type=str, default="Describe all the colors seen in the image.")
    parser.add_argument("--image-url", type=str, default="https://picsum.photos/id/237/536/354")
    parser.add_argument("--height", type=int, default=354)
    parser.add_argument("--width", type=int, default=536)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--generation-len", type=int, default=100)
    parser.add_argument("--num-cores", type=int, default=constants.DEFAULT_AIC_NUM_CORES)
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--mos", type=int, default=1)
    parser.add_argument("--aic-hw-version", type=str, default=constants.DEFAULT_AIC_HW_VERSION)
    parser.add_argument("--vision-depth", type=int, default=9)
    parser.add_argument("--text-layers", type=int, default=1)
    parser.add_argument(
        "--reduce-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a reduced-layer config for faster bring-up.",
    )
    parser.add_argument(
        "--weight-free",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the model on meta tensors and load weights at compile time.",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Compile and run the text path only.",
    )
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name)
    config = maybe_reduce_config(
        config,
        reduce_layers=args.reduce_layers,
        vision_depth=args.vision_depth,
        text_layers=args.text_layers,
    )

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_name,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        weight_free=args.weight_free,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name)

    compile_kwargs = {
        "batch_size": args.batch_size,
        "prefill_seq_len": args.prefill_seq_len,
        "ctx_len": args.ctx_len,
        "num_cores": args.num_cores,
        "num_devices": args.num_devices,
        "height": args.height,
        "width": args.width,
        "mxfp6_matmul": True,
        "aic_enable_depth_first": True,
        "mos": args.mos,
        "use_onnx_subfunctions": True,
        "dynamo": True,
        "aic_hw_version": args.aic_hw_version,
    }
    if args.skip_vision:
        compile_kwargs["skip_vision"] = True
    else:
        compile_kwargs.update({"split_model_io": True, "mxint8_kv_cache": True})

    qpc_paths = qeff_model.compile(**compile_kwargs)
    print(f"Model compiled to: {qpc_paths}")
    print(f"Weight specs: {qeff_model.weight_spec_path}")

    if args.skip_vision:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": args.prompt}],
            }
        ]
        messages = [messages] * args.batch_size
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        image = load_image(args.image_url, args.width, args.height)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        messages = [messages] * args.batch_size
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs,
        prefill_seq_len=args.prefill_seq_len,
        batch_size=args.batch_size,
    )
    output = qeff_model.generate(inputs=inputs, generation_len=args.generation_len)

    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)


if __name__ == "__main__":
    main()
