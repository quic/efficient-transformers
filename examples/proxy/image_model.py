# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""
Simple example: How to enable proxy models for three different vision-language models and generate IO files.
Demonstrates three model types with different execution flows:
1. Standard VLM (LLaVA, Gemma3, granite_vision, etc.)
2. InternVL Model
3. Molmo Model
"""

from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.utils.test_utils import InternProcessor

# Configuration
img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
query = "Describe this image."

# Load image
print("Loading image...")
img = requests.get(img_url, stream=True)
image = Image.open(BytesIO(img.content)).convert("RGB")

# Three models with different execution flows
models = [
    {
        "name": "llava-hf/llava-1.5-7b-hf",
        "type": "Standard VLM",
        "is_intern": False,
        "is_molmo": False,
    },
    {
        "name": "OpenGVLab/InternVL2_5-1B",
        "type": "InternVL",
        "is_intern": True,
        "is_molmo": False,
    },
    {
        "name": "allenai/Molmo-7B-D-0924",
        "type": "Molmo",
        "is_intern": False,
        "is_molmo": True,
    },
]

for model_config in models:
    model_name = model_config["name"]
    model_type = model_config["type"]
    is_intern_model = model_config["is_intern"]
    is_molmo_model = model_config["is_molmo"]

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print(f"TYPE: {model_type}")
    print("=" * 70)

    # Load config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, padding=not is_molmo_model)
    config._attn_implementation = "eager" if (is_intern_model or is_molmo_model) else None

    # ============ EXECUTION FLOW 1: Standard VLM (LLaVA) ============
    compile_kwargs = {}
    if not is_intern_model and not is_molmo_model:
        print("Execution Flow: Standard VLM")

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)

        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image"},
                ],
            }
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt")

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

        # Load proxy model
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(model_name, kv_offload=True, enable_proxy=True)

    # ============ EXECUTION FLOW 2: InternVL Model ============
    elif is_intern_model:
        print("Execution Flow: InternVL")

        model_hf = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            config=config,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        processor = InternProcessor(model_hf, tokenizer)

        # Process image
        image_resized = image.resize((448, 448))
        pixel_value = processor.load_image(image_resized, max_num=12)

        # Prepare prompt
        question = "<image>\n" + query
        messages = []
        roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        prompt = processor(
            pixel_value.unsqueeze(0), [question], messages, roles, num_patches_list=[pixel_value.shape[0]]
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs["pixel_values"] = pixel_value.clone()

        # Load proxy model
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            kv_offload=True,
            enable_proxy=True,
        )

        compile_kwargs["num_patches"] = 1

    # ============ EXECUTION FLOW 3: Molmo Model ============
    else:  # is_molmo_model
        print("Execution Flow: Molmo")

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)

        # Resize image for Molmo
        image_resized = image.resize((536, 354))

        # Process inputs
        inputs = processor.process(images=[image_resized], text=query)
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}

        # Add required fields for Molmo
        inputs["attention_mask"] = torch.ones((inputs["input_ids"].shape), dtype=torch.int64)
        valid = inputs["image_input_idx"] > 0
        valid = valid.reshape(1, -1)
        inputs["valid_idx"] = torch.nonzero(valid)[:, 1].unsqueeze(0)
        inputs["pixel_values"] = inputs.pop("images")

        # Load proxy model
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            kv_offload=True,
            enable_proxy=True,
        )

    # Compile model
    print("Compiling model...")
    qeff_model.compile(num_devices=1, prefill_seq_len=128, ctx_len=2048, **compile_kwargs)

    # Generate with IO files
    outputs = qeff_model.generate(
        inputs=inputs,
        generation_len=10,
        write_io=True,  # Saves input/output tensors to files
    )
    print(f"Output: {outputs}\n")
    print(f"✓ Successfully processed: {model_name}\n")
