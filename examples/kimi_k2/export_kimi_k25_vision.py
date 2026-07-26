# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.utils.load_kimi_utils import (
    DEFAULT_MODEL_PATH,
    LOADED_EXPERT_IDS,
    NUM_EXPERTS_PER_TOKEN,
    NUM_TEXT_LAYERS,
    NUM_VISION_LAYERS,
    ensure_torch_fx_import_compatibility,
    load_kimi_k25_class,
    load_layer_subset_model,
    parse_expert_ids,
    prepare_config,
    set_deterministic,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Load Kimi K2.5 with runtime compatibility for transformers==5.5.4.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Load the full model. By default, the script loads a small layer subset for faster startup.",
    )
    parser.add_argument("--num-vision-layers", type=int, default=NUM_VISION_LAYERS)
    parser.add_argument("--num-text-layers", type=int, default=NUM_TEXT_LAYERS)
    parser.add_argument("--expert-ids", type=parse_expert_ids, default=LOADED_EXPERT_IDS)
    parser.add_argument("--num-experts-per-token", type=int, default=NUM_EXPERTS_PER_TOKEN)
    parser.add_argument(
        "--image-url",
        type=str,
        default="https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Image height in pixels for Kimi-K2.5 vision compile. Defaults to the loaded image height.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Image width in pixels for Kimi-K2.5 vision compile. Defaults to the loaded image width.",
    )
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    args = parser.parse_args()
    if (args.image_height is None) != (args.image_width is None):
        parser.error("--image-height and --image-width must be provided together.")
    return args


def main():
    args = parse_args()

    set_deterministic(1234)
    ensure_torch_fx_import_compatibility()
    config = prepare_config(args.model_path)
    kimi_cls = load_kimi_k25_class(args.model_path)

    model_kwargs = {
        "config": config,
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "torch_dtype": torch.float32,
    }

    if args.full_model:
        model, tokenizer, processor = kimi_cls.from_pretrained(str(args.model_path), **model_kwargs)
    elif args.num_vision_layers is not None and args.num_text_layers is not None:
        model, tokenizer, processor = load_layer_subset_model(
            model_path=args.model_path,
            kimi_cls=kimi_cls,
            config=config,
            num_vision_layers=args.num_vision_layers,
            num_text_layers=args.num_text_layers,
            loaded_expert_ids=args.expert_ids,
            num_experts_per_tok=args.num_experts_per_token,
            dtype=torch.float32,
        )
        print(
            "Loaded layer subset: "
            f"vision={model.config.vision_config.vt_num_hidden_layers}, "
            f"text={model.config.text_config.num_hidden_layers}, "
            f"experts={model.config.text_config.n_routed_experts}"
        )
    else:
        raise ValueError("Pass both --num-vision-layers and --num-text-layers to load a layer subset.")

    qaic_config = {"mla_absorption": {"cache_compressed": True, "absorption": False, "online": False}}

    qeff_model = QEFFAutoModelForImageTextToText(model, qaic_config=qaic_config)

    skip_vision = False

    if skip_vision:
        ## TEXT-ONLY MODE ##

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        qeff_model.compile(
            qaic_config=qaic_config,
            prefill_seq_len=1,
            ctx_len=1024,
            num_cores=16,
            num_devices=1,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=False,
            skip_vision=True,  # Skip vision encoder for text-only inference
            mos=1,
        )

        output = qeff_model.generate(inputs=inputs, device_ids=[0], generation_len=10)

        print(output.generated_ids)
        print(tokenizer.batch_decode(output.generated_ids))
        print(output)

    else:
        ## VISION + TEXT MODE ##

        image = Image.open(BytesIO(requests.get(args.image_url).content)).convert("RGB")
        if args.image_height is not None:
            image = image.resize((args.image_width, args.image_height))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image},
                    {"type": "text", "text": args.prompt},
                ],
            },
        ]

        inputs = processor(
            messages=messages,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors="pt",
        )

        inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)

        qeff_model.compile(
            qaic_config=qaic_config,
            prefill_seq_len=1,
            ctx_len=1024,
            num_cores=16,
            num_devices=1,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=False,
            mos=1,
            image_height=image.height,
            image_width=image.width,
        )

        output = qeff_model.generate(inputs=inputs, generation_len=10)

        print(output.generated_ids)
        print(tokenizer.batch_decode(output.generated_ids))
        print(output)


if __name__ == "__main__":
    main()
