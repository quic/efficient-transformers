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
    DEFAULT_MODEL_PATH as MODEL_PATH,
)
from QEfficient.utils.load_kimi_utils import (
    LOADED_EXPERT_IDS,
    NUM_EXPERTS_PER_TOKEN,
    NUM_TEXT_LAYERS,
    NUM_VISION_LAYERS,
    load_kimi_k25_class,
)
from QEfficient.utils.load_kimi_utils import (
    ensure_torch_fx_import_compatibility as _ensure_torch_fx_import_compatibility,
)
from QEfficient.utils.load_kimi_utils import (
    load_layer_subset_model as _load_layer_subset_model,
)
from QEfficient.utils.load_kimi_utils import (
    parse_expert_ids as _parse_expert_ids,
)
from QEfficient.utils.load_kimi_utils import (
    prepare_config as _prepare_config,
)
from QEfficient.utils.load_kimi_utils import (
    set_deterministic as _set_deterministic,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Load Kimi K2.5 with runtime compatibility for transformers==5.5.4.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Load the full model. By default, the script loads a small layer subset for faster startup.",
    )
    parser.add_argument("--num-vision-layers", type=int, default=NUM_VISION_LAYERS)
    parser.add_argument("--num-text-layers", type=int, default=NUM_TEXT_LAYERS)
    parser.add_argument("--expert-ids", type=_parse_expert_ids, default=LOADED_EXPERT_IDS)
    parser.add_argument("--num-experts-per-token", type=int, default=NUM_EXPERTS_PER_TOKEN)
    parser.add_argument(
        "--image-url",
        type=str,
        default="https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png",
    )
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--test", action="store_true", help="Validate ONNX output matches PyTorch image-only forward.")
    return parser.parse_args()


def main():
    args = parse_args()

    _set_deterministic(1234)
    _ensure_torch_fx_import_compatibility()
    config = _prepare_config(args.model_path)
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
        model, tokenizer, processor = _load_layer_subset_model(
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

        ## STEP 3: Compile Model for Text-Only Execution
        # Set skip_vision=True to bypass image processing
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
        ## STEP 4: Prepare Text-Only Input
        # Create a text-only message without any image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                ],
            },
        ]

        ## STEP 5: Process Input with Chat Template
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        ## STEP 6: Run Text-Only Inference
        output = qeff_model.generate(inputs=inputs, device_ids=[0], generation_len=10)

        ## STEP 7: Display Results
        print(output.generated_ids)
        print(tokenizer.batch_decode(output.generated_ids))
        print(output)

    else:
        ## VISION + TEXT MODE ##

        ## STEP 3: Compile Model for Vision+Text Execution
        # Do not set skip_vision (defaults to False) to enable image processing
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
            num_patches=2400,
            h=30,
            w=80,
            # Keep language-side image embedding specialization tightly bounded to
            # actual single-image token count to avoid oversized dynamic VA mapping.
            num_image_tokens=600,
        )

        ## STEP 4: Prepare Image and Text Input
        image = Image.open(BytesIO(requests.get(args.image_url).content)).convert("RGB")

        # Create a message with both image and text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image},
                    {"type": "text", "text": args.prompt},
                ],
            },
        ]

        ## STEP 5: Process Input
        inputs = processor(
            messages=messages,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors="pt",
        )
        # Convert pixel values to float32 for processing
        inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)

        ## STEP 6: Run Vision+Text Inference
        output = qeff_model.generate(inputs=inputs, generation_len=10)

        ## STEP 7: Display Results
        print(output.generated_ids)
        print(tokenizer.batch_decode(output.generated_ids))
        print(output)


if __name__ == "__main__":
    main()
