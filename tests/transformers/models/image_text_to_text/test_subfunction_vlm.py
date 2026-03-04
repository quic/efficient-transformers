# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import json
from typing import Optional

import onnx
import pytest
import requests
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
)

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText
from QEfficient.utils._utils import get_num_layers_vlm
from QEfficient.utils.test_utils import load_vlm_model, load_vlm_model_from_config

NEW_GENERATION_TOKENS = 10


CONFIG_PATH = "tests/configs/image_text_model_configs.json"

with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    multimodal_models = config_data["image_text_subfunction_models"]

test_mm_models = [model_config["model_name"] for model_config in multimodal_models]
model_config_dict = {model["model_name"]: model for model in multimodal_models}


def has_QwenLayer_function(onnx_path):
    """Check if ONNX model contains QEffqwenlayer function definition."""
    model = onnx.load(onnx_path, load_external_data=False)
    function_names = [f.name for f in model.functions]
    QwenLayer_functions = [name for name in function_names if "QEffQwen2_5_VLDecoderLayer" in name]
    return len(QwenLayer_functions) > 0, QwenLayer_functions


def check_image_text_to_text_subfunction_core(
    model_name: str,
    img_size: int,
    img_url: str,
    query: str,
    prompt_len: int,
    ctx_len: int,
    max_gen_len: int = 20,
    batch_size: int = 1,
    n_layer: int = 1,
    kv_offload: bool = False,
    num_devices: int = 1,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    config: Optional[AutoConfig] = None,
):
    if config is None:
        model_config = {"model_name": model_name}
        model_config["img_size"] = img_size
        config = AutoConfig.from_pretrained(model_config["model_name"], trust_remote_code=True, padding=True)
        config.text_config.num_hidden_layers = n_layer
        config.vision_config.num_hidden_layers = n_layer
        model_hf = load_vlm_model(config)
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            config=config,
        )
    else:
        model_hf = load_vlm_model_from_config(config)
        qeff_model = QEFFAutoModelForImageTextToText(
            copy.deepcopy(model_hf),
            kv_offload=kv_offload,
            config=config,
        )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
    n_layer = get_num_layers_vlm(config)
    image = Image.open(requests.get(img_url, stream=True).raw)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    with_sub_func_onnx = qeff_model.export(use_onnx_subfunctions=True, offload_pt_weights=False)

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    if hasattr(qeff_model.model.config, "model_type") and qeff_model.model.config.model_type == "qwen2_5_vl":
        inputs = qeff_model.model.prepare_inputs_for_generation(
            inputs=inputs, prefill_seq_len=prompt_len, batch_size=batch_size
        )
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    # Verify that the model with subfunctions has QEffQwen2_5_VLDecoderLayer function definition
    has_qwenlayer, qwenlayer_names = has_QwenLayer_function(with_sub_func_onnx[-1])
    assert has_qwenlayer, (
        "Model exported with use_onnx_subfunctions=True should contain QEffQwen2_5_VLDecoderLayer function definition"
    )
    print(f"\nQwenLayer functions found: {qwenlayer_names}")

    qeff_model.compile(
        img_size=img_size,
        num_devices=num_devices,
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        mxfp6=False,
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
    )


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.regular
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])
def test_custom_image_text_to_text_subfunction(model_name, kv_offload):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching with subfunction.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``Qwen/Qwen2.5-VL-3B-Instruct``
    """
    torch.manual_seed(42)
    img_size = model_config_dict[model_name].get("img_size")
    custom_config = model_config_dict[model_name].get("additional_params", {})
    model_type = model_config_dict[model_name].get("model_type", None)
    hf_config = AutoConfig.for_model(model_type, trust_remote_code=True, **custom_config)
    hf_config.name_or_path = model_name
    check_image_text_to_text_subfunction_core(
        model_name=model_name,
        prompt_len=model_config_dict[model_name]["prompt_len"],
        ctx_len=model_config_dict[model_name]["ctx_len"],
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=img_size,
        img_url=model_config_dict[model_name]["img_url"],
        query=model_config_dict[model_name]["text_prompt"],
        n_layer=model_config_dict[model_name]["num_layers"],
        batch_size=model_config_dict[model_name]["batch_size"],
        kv_offload=kv_offload,
        config=hf_config,
    )


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.nightly
@pytest.mark.parametrize("model_name", test_mm_models)
@pytest.mark.parametrize("kv_offload", [True])
def test_image_text_to_text_subfunction(model_name, kv_offload):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model,  without continuous batching with subfunction.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``Qwen/Qwen2.5-VL-3B-Instruct``
    """
    torch.manual_seed(42)
    img_size = model_config_dict[model_name].get("img_size")
    check_image_text_to_text_subfunction_core(
        model_name=model_name,
        prompt_len=model_config_dict[model_name]["prompt_len"],
        ctx_len=model_config_dict[model_name]["ctx_len"],
        max_gen_len=NEW_GENERATION_TOKENS,
        img_size=img_size,
        img_url=model_config_dict[model_name]["img_url"],
        query=model_config_dict[model_name]["text_prompt"],
        n_layer=model_config_dict[model_name]["num_layers"],
        batch_size=model_config_dict[model_name]["batch_size"],
        kv_offload=kv_offload,
    )


"""
Qwen2_5_VLConfig {
  "architectures": [
    "Qwen2_5_VLForConditionalGeneration"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "image_token_id": 151655,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 128000,
  "max_window_layers": 70,
  "model_type": "qwen2_5_vl",
  "num_attention_heads": 16,
  "num_hidden_layers": 36,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "mrope_section": [
      16,
      24,
      24
    ],
    "rope_type": "default",
    "type": "default"
  },
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "text_config": {
    "architectures": [
      "Qwen2_5_VLForConditionalGeneration"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "image_token_id": null,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "layer_types": [
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention",
      "full_attention"
    ],
    "max_position_embeddings": 128000,
    "max_window_layers": 70,
    "model_type": "qwen2_5_vl_text",
    "num_attention_heads": 16,
    "num_hidden_layers": 1,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
      "mrope_section": [
        16,
        24,
        24
      ],
      "rope_type": "default",
      "type": "default"
    },
    "rope_theta": 1000000.0,
    "sliding_window": null,
    "tie_word_embeddings": true,
    "torch_dtype": "bfloat16",
    "use_cache": true,
    "use_sliding_window": false,
    "video_token_id": null,
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
    "vision_token_id": 151654,
    "vocab_size": 151936
  },
  "torch_dtype": "bfloat16",
  "transformers_version": "4.55.0",
  "use_cache": true,
  "use_sliding_window": false,
  "video_token_id": 151656,
  "vision_config": {
    "depth": 32,
    "fullatt_block_indexes": [
      7,
      15,
      23,
      31
    ],
    "hidden_act": "silu",
    "hidden_size": 1280,
    "in_channels": 3,
    "in_chans": 3,
    "initializer_range": 0.02,
    "intermediate_size": 3420,
    "model_type": "qwen2_5_vl",
    "num_heads": 16,
    "num_hidden_layers": 1,
    "out_hidden_size": 2048,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "temporal_patch_size": 2,
    "tokens_per_second": 2,
    "window_size": 112
  },
  "vision_end_token_id": 151653,
  "vision_start_token_id": 151652,
  "vision_token_id": 151654,
  "vocab_size": 151936
}

Qwen2_5_VLForConditionalGeneration(
  (model): Qwen2_5_VLModel(
    (visual): Qwen2_5_VisionTransformerPretrainedModel(
      (patch_embed): Qwen2_5_VisionPatchEmbed(
        (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
      )
      (rotary_pos_emb): Qwen2_5_VisionRotaryEmbedding()
      (blocks): ModuleList(
        (0-31): 32 x Qwen2_5_VLVisionBlock(
          (norm1): Qwen2RMSNorm((1280,), eps=1e-06)
          (norm2): Qwen2RMSNorm((1280,), eps=1e-06)
          (attn): Qwen2_5_VLVisionAttention(
            (qkv): Linear(in_features=1280, out_features=3840, bias=True)
            (proj): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (mlp): Qwen2_5_VLMLP(
            (gate_proj): Linear(in_features=1280, out_features=3420, bias=True)
            (up_proj): Linear(in_features=1280, out_features=3420, bias=True)
            (down_proj): Linear(in_features=3420, out_features=1280, bias=True)
            (act_fn): SiLU()
          )
        )
      )
      (merger): Qwen2_5_VLPatchMerger(
        (ln_q): Qwen2RMSNorm((1280,), eps=1e-06)
        (mlp): Sequential(
          (0): Linear(in_features=5120, out_features=5120, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=5120, out_features=2048, bias=True)
        )
      )
    )
    (language_model): Qwen2_5_VLTextModel(
      (embed_tokens): Embedding(151936, 2048)
      (layers): ModuleList(
        (0): Qwen2_5_VLDecoderLayer(
          (self_attn): Qwen2_5_VLAttention(
            (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
            (k_proj): Linear(in_features=2048, out_features=256, bias=True)
            (v_proj): Linear(in_features=2048, out_features=256, bias=True)
            (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (rotary_emb): Qwen2_5_VLRotaryEmbedding()
          )
          (mlp): Qwen2MLP(
            (gate_proj): Linear(in_features=2048, out_features=11008, bias=False)
            (up_proj): Linear(in_features=2048, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=2048, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
          (post_attention_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
        )
      )
      (norm): Qwen2RMSNorm((2048,), eps=1e-06)
      (rotary_emb): Qwen2_5_VLRotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)

"""
