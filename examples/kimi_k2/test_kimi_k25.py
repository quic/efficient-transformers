# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from io import BytesIO
from pathlib import Path

import requests
import torch
from export_kimi_k25_vision import (
    LOADED_EXPERT_IDS,
    NUM_EXPERTS_PER_TOKEN,
    _ensure_torch_fx_import_compatibility,
    _load_layer_subset_model,
    _patch_deepseek_init_weights_compat,
    _patch_kimi_tie_weights_compat,
    _prepare_config,
)
from huggingface_hub import snapshot_download
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_NAME = "moonshotai/Kimi-K2.5"
IMAGE_URL = "https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png"
TEXT_PROMPT = "Describe this image."
NUM_VISION_LAYERS = 2
NUM_TEXT_LAYERS = 2
NEW_GENERATION_TOKENS = 10
CTX_LEN = 4096

def _set_deterministic(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

def _resolve_model_path() -> Path:
    return Path(snapshot_download(repo_id=MODEL_NAME, cache_dir=os.environ.get("HF_HUB_CACHE")))

def _prepare_inputs(processor):
    image = Image.open(BytesIO(requests.get(IMAGE_URL, timeout=30).content)).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image},
                {"type": "text", "text": TEXT_PROMPT},
            ],
        }
    ]
    return processor(
        messages=messages,
        add_generation_prompt=True,
        tokenize=False,
        return_tensors="pt",
    )

def _greedy_generate_cpu(model, inputs, max_new_tokens: int):
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    grid_thws = inputs["grid_thws"]

    eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is None and hasattr(model.config, "text_config"):
        eos_token_id = getattr(model.config.text_config, "eos_token_id", None)

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
            ],
            dim=1,
        )

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    return generated_ids[:, -max_new_tokens:]


def check_kimi_k25_pytorch_vs_ai100():
    _set_deterministic(1234)
    _ensure_torch_fx_import_compatibility()
    model_path = _resolve_model_path()
    config = _prepare_config(model_path)
    kimi_cls = get_class_from_dynamic_module("modeling_kimi_k25.KimiK25ForConditionalGeneration", str(model_path))
    _patch_kimi_tie_weights_compat(kimi_cls)
    _patch_deepseek_init_weights_compat(kimi_cls)

    model, tokenizer, processor = _load_layer_subset_model(
        model_path=model_path,
        kimi_cls=kimi_cls,
        config=config,
        num_vision_layers=NUM_VISION_LAYERS,
        num_text_layers=NUM_TEXT_LAYERS,
        loaded_expert_ids=LOADED_EXPERT_IDS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOKEN,
        dtype=torch.float32,
    )

    model.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
    model = model.eval().to("cpu")

    inputs = _prepare_inputs(processor)
    inputs = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = _greedy_generate_cpu(model, inputs, max_new_tokens=NEW_GENERATION_TOKENS)

    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print(
        "Loaded subset:",
        f"vision_layers={model.config.vision_config.vt_num_hidden_layers}",
        f"text_layers={model.config.text_config.num_hidden_layers}",
    )
    print("Prompt:", repr(TEXT_PROMPT))
    print("Generated IDs shape:", tuple(generated_ids.shape))
    print("Output:")
    print(decoded[0] if decoded else "")

    qeff_model = QEFFAutoModelForImageTextToText(
        model,
        kv_offload=True,
        torch_dtype=torch.float32,
    )

    _ = qeff_model.export()
    qeff_model.compile(
        num_devices=1,
        prefill_seq_len=1,  # prompt_len,
        ctx_len=CTX_LEN,
        mxfp6=False,
        num_patches=int(inputs["pixel_values"].shape[0]),
        h=int(inputs["grid_thws"][0, 1].item()),
        w=int(inputs["grid_thws"][0, 2].item()),
        num_image_tokens=600,
        num_cores=16,
    )

    exec_info = qeff_model.generate(inputs=inputs, generation_len=NEW_GENERATION_TOKENS)
    cloud_ai_100_tokens = exec_info.generated_ids[:, :-1]
    decoded = tokenizer.batch_decode(cloud_ai_100_tokens, skip_special_tokens=True)

    print(
        "Loaded subset:",
        f"vision_layers={model.config.vision_config.vt_num_hidden_layers}",
        f"text_layers={model.config.text_config.num_hidden_layers}",
    )
    print("Prompt:", repr(TEXT_PROMPT))
    print("Generated IDs shape:", tuple(cloud_ai_100_tokens.shape))
    print("Output:")
    print(decoded[0] if decoded else "")

    assert (generated_ids == cloud_ai_100_tokens).all(), "Tokens don't match for pytorch HF output and QPC output"


def test_kimi_k25_image_text_to_text_pytorch_vs_ai100():
    torch.manual_seed(42)
    check_kimi_k25_pytorch_vs_ai100()
