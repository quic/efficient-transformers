# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import argparse
from typing import Optional

import torch
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from QEfficient.transformers.models.modeling_auto import (
    _QEffAutoModelForImageTextToTextDualQPC,
)
from QEfficient.utils._utils import login_and_download_hf_lm


def duplicate_weights_for_linear_layer(
    layer: torch.nn.Module, orig_kv_heads: int, repeat: int, head_dim: int, hidden_size: int
):
    new_kv_heads = repeat * orig_kv_heads
    layer.weight.data = torch.repeat_interleave(
        layer.weight.data.view(orig_kv_heads, head_dim, hidden_size), repeat, 0
    ).view(new_kv_heads * head_dim, hidden_size)
    if layer.bias is not None:
        layer.bias.data = torch.repeat_interleave(layer.bias.data.view(orig_kv_heads, head_dim), repeat, 0).view(
            new_kv_heads * head_dim
        )


def replicate_kv_heads_vlm(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    prompt: str = "Describe this image.",
    repeat: int = 2,
    num_hidden_layers: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    hidden_size: Optional[int] = None,
):
    """
    Replicate the KV heads for Vision Language Models (language component only).

    ``Mandatory`` Args:
        :model_name (str): Model card name to use (e.g., "Qwen/Qwen2.5-VL-7B-Instruct").
        :prompt (str): Prompt to use for validation.
        :repeat (int): Factor to repeat key-value heads.

    ``Optional`` Args:
        :num_hidden_layers (int): Number of hidden layers to use, default is None.
        :num_attention_heads (int): Number of attention heads, if not passed explicitly then will be picked from config.
        :hidden_size (int): Hidden size to use, if not passed explicitly then will be picked from config.
    """
    # Load the model configuration
    model_base_name = model_name.split("/")[-1]

    # Prepare kwargs for model loading
    model_kwargs = {"attn_implementation": "eager"}

    # Load config
    config = AutoConfig.from_pretrained(model_name)
    if num_hidden_layers:
        config.text_config.num_hidden_layers = num_hidden_layers
        model_kwargs["config"] = config

    pretrained_model_name_or_path = login_and_download_hf_lm(model_name)

    # Load the original transformers model for validation
    print("Loading original transformers model...")
    orig_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, **model_kwargs)

    # Load processor for tokenization
    processor = AutoProcessor.from_pretrained(model_name)
    orig_lang_model = orig_model.language_model
    lang_config = config.text_config
    inputs = processor(text=prompt, return_tensors="pt", add_special_tokens=True)

    # Generate original outputs and tokens from transformers model
    print("\nGenerating with original transformers model...")
    with torch.inference_mode():
        orig_tokens = orig_lang_model(**inputs).last_hidden_state

    # Modify the number of key-value heads in QEfficient model
    orig_kv_heads = lang_config.num_key_value_heads
    new_kv_heads = repeat * orig_kv_heads
    orig_lang_model.config.num_key_value_heads = new_kv_heads

    print(f"\nOriginal KV heads: {orig_kv_heads}")
    print(f"Modified KV heads: {new_kv_heads}")

    # Check if hidden size and number of attention heads are explicitly passed
    if num_attention_heads is None:
        num_attention_heads = lang_config.num_attention_heads
    if hidden_size is None:
        hidden_size = lang_config.hidden_size

    # Update the model's attention layers with new key-value heads
    print(f"\nReplicating KV heads in {len(orig_lang_model.layers)} layers...")
    for block in orig_lang_model.layers:
        attn = block.self_attn
        attn.num_key_value_heads = new_kv_heads
        attn.num_key_value_groups = num_attention_heads // new_kv_heads

        duplicate_weights_for_linear_layer(attn.k_proj, orig_kv_heads, repeat, attn.head_dim, hidden_size)
        duplicate_weights_for_linear_layer(attn.v_proj, orig_kv_heads, repeat, attn.head_dim, hidden_size)

    # Generate modified outputs and tokens from QEfficient model
    print("\nGenerating with modified QEfficient model...")
    with torch.inference_mode():
        mod_tokens = orig_lang_model(**inputs).last_hidden_state

    if not torch.all(orig_tokens == mod_tokens):
        raise RuntimeError(
            "Something went wrong while duplicating KV heads weights, output tokens don't match after modification"
        )

    print("\n Validation successful! Output tokens match.")

    # Export the modified model
    export_dir = f"{model_base_name}-{new_kv_heads}kvheads"
    print(f"\nExporting modified model to {export_dir}...")

    # Export using the qeff_model's export method
    qeff_model = _QEffAutoModelForImageTextToTextDualQPC(orig_model)
    inputs = qeff_model.model.get_dummy_inputs(kv_offload=True)
    dynamic_axes = qeff_model.model.get_onnx_dynamic_axes(kv_offload=True)
    output_names = qeff_model.model.get_output_names(kv_offload=True)
    qeff_model.lang_model.export(
        inputs["lang"], output_names["lang"], dynamic_axes["lang"], export_dir=export_dir, offload_pt_weights=True
    )


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Modify and export KV heads for Vision Language Models (language component)."
    )
    parser.add_argument(
        "--model_name",
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Name of the VLM model to use.",
    )
    parser.add_argument("--prompt", type=str, default="Describe this image.", help="Prompt to use for validation.")
    parser.add_argument("--repeat", type=int, default=2, help="Factor to repeat key-value heads.")
    parser.add_argument(
        "--num_hidden_layers",
        "--num-hidden-layers",
        type=int,
        default=None,
        help="Number of hidden layers to use, default is None",
    )
    parser.add_argument(
        "--num_attention_heads",
        "--num-attention-heads",
        type=int,
        default=None,
        help="Number of attention heads, if not passed explicitly then will be picked from config",
    )
    parser.add_argument(
        "--hidden_size",
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden size to use, if not passed explicitly then will be picked from config",
    )

    args = parser.parse_args()

    replicate_kv_heads_vlm(
        model_name=args.model_name,
        prompt=args.prompt,
        repeat=args.repeat,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        hidden_size=args.hidden_size,
    )
