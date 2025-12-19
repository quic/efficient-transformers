# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers, undo_transformers_quantizers
from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ
from QEfficient.transformers.quantizers.quantizer_compressed_tensors import FP8DeQuantLinear
from QEfficient.utils._utils import login_and_download_hf_lm


def duplicate_weights_for_linear_layer(
    layer: torch.nn.Module, orig_kv_heads: int, repeat: int, head_dim: int, hidden_size: int
):
    new_kv_heads = repeat * orig_kv_heads
    if isinstance(layer, (WQLinear_GEMM, QuantLinearGPTQ)):
        if head_dim % 8 != 0:
            raise ValueError(
                f"the value head_dim={head_dim} is not divisible by 8 which is \
                                according to the assumption that model is 4-bit quantized."
            )
        if hidden_size % layer.group_size != 0:
            raise ValueError(
                f"The value of hidden_size={hidden_size} is not divisible by \
                            K_proj.group_size={layer.group_size}"
            )

        # Duplication of quantized weights
        layer.qweight.data = torch.repeat_interleave(
            layer.qweight.data.view(hidden_size, orig_kv_heads, head_dim // 8), repeat, 1
        ).view(hidden_size, (new_kv_heads * head_dim) // 8)
        # Duplication of quantized zero points
        layer.qzeros.data = torch.repeat_interleave(
            layer.qzeros.data.view(hidden_size // layer.group_size, orig_kv_heads, head_dim // 8),
            repeat,
            1,
        ).view(hidden_size // layer.group_size, (new_kv_heads * head_dim) // 8)
        # Duplication of quantization scales
        layer.scales.data = torch.repeat_interleave(
            layer.scales.data.view(hidden_size // layer.group_size, orig_kv_heads, head_dim),
            repeat,
            1,
        ).view(hidden_size // layer.group_size, new_kv_heads * head_dim)
        layer.out_features = layer.out_features * repeat

    elif isinstance(layer, FP8DeQuantLinear):
        layer.weight.data = torch.repeat_interleave(
            layer.weight.data.view(orig_kv_heads, head_dim, hidden_size), repeat, 0
        ).view(new_kv_heads * head_dim, hidden_size)
        layer.weight_scale.data = torch.repeat_interleave(
            layer.weight_scale.data.view(orig_kv_heads, head_dim), repeat, 0
        ).view(new_kv_heads * head_dim, -1)

    else:
        layer.weight.data = torch.repeat_interleave(
            layer.weight.data.view(orig_kv_heads, head_dim, hidden_size), repeat, 0
        ).view(new_kv_heads * head_dim, hidden_size)
        if layer.bias is not None:
            layer.bias.data = torch.repeat_interleave(layer.bias.data.view(orig_kv_heads, head_dim), repeat, 0).view(
                new_kv_heads * head_dim
            )


def replicate_kv_heads(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    prompt: str = "My name is",
    repeat: int = 2,
    full_batch_size: Optional[int] = None,
    num_hidden_layers: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    hidden_size: Optional[int] = None,
):
    """
    Replicate the KV heads. The script performs the following steps:
    1. Runs inference with the original model.
    2. Replicates the KV heads.
    3. Runs inference on the modified model to validate the changes.
    4. Exports the modified model to ONNX format.

    ``Mandatory`` Args:
        :model_name (str): Model card name to use, default value as meta-llama/Meta-Llama-3-8B-Instruct.
        :prompt (str): Prompt to use for the model, default value as My name is
        :repeat (int): Factor to repeat key-value heads.
    ``Optional`` Args:
        :full_batch_size (int): Set full batch size to enable continuous batching mode, default is None.
        :num_hidden_layers (int): Number of hidden layers to use, default is None.
        :num_attention_heads (int): Number of attention heads, if not passed explicitly then will be picked from config.json.
        :hidden_size (int): Hidden size to use, if not passed explicitly then will be picked from config.json.

    """
    # Load the model and tokenizer
    model_base_name = model_name.split("/")[-1]
    # Replace quantizers for loading Quantized AWQ/GPTQ models on CPU.
    replace_transformers_quantizers()
    # Prepare kwargs for model loading
    model_kwargs = {"attn_implementation": "eager"}

    if num_hidden_layers:
        model_kwargs["num_hidden_layers"] = num_hidden_layers

    pretrained_model_name_or_path = login_and_download_hf_lm(model_name)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **model_kwargs)

    # Undo the effect of replace_transformers_quantizers
    undo_transformers_quantizers()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate original outputs and tokens
    with torch.inference_mode():
        _ = model(**inputs)  # original output
        orig_tokens = model.generate(**inputs, max_new_tokens=10, num_beams=1, do_sample=False)

    # Modify the number of key-value heads
    orig_kv_heads = model.config.num_key_value_heads
    new_kv_heads = repeat * orig_kv_heads
    model.config.num_key_value_heads = new_kv_heads

    print("Original KV heads:", orig_kv_heads)
    print("Modified KV heads:", new_kv_heads)

    # Check if hidden size and number of attention heads are explicitly passed as arguments or not
    if num_attention_heads is None:
        num_attention_heads = model.config.num_attention_heads

    if hidden_size is None:
        hidden_size = model.config.hidden_size

    # Update the model's attention layers with new key-value heads
    for block in model.model.layers:
        attn = block.self_attn
        attn.num_key_value_heads = new_kv_heads
        attn.num_key_value_groups = num_attention_heads // new_kv_heads
        duplicate_weights_for_linear_layer(attn.k_proj, orig_kv_heads, repeat, attn.head_dim, hidden_size)
        duplicate_weights_for_linear_layer(attn.v_proj, orig_kv_heads, repeat, attn.head_dim, hidden_size)

    # Generate modified outputs and tokens
    with torch.inference_mode():
        _ = model(**inputs)  # Modified output
        mod_tokens = model.generate(**inputs, max_new_tokens=10, num_beams=1, do_sample=False)

    # Print the original and modified token outputs
    print("Original:", tokenizer.batch_decode(orig_tokens))
    print("Modified:", tokenizer.batch_decode(mod_tokens))

    if not torch.all(orig_tokens == mod_tokens):
        raise RuntimeError(
            "Something went wrong while duplicating KV heads weights, output token don't match after modification"
        )

    # Export the modified model
    q_model = QEFFAutoModelForCausalLM(model, continuous_batching=(True if full_batch_size else False))
    q_model.export(
        export_dir=f"{model_base_name}-{new_kv_heads}kvheads",
        full_batch_size=(full_batch_size if full_batch_size else None),
    )


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Modify and export a causal language model.")
    parser.add_argument(
        "--model_name",
        "--model-name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Name of the model to use.",
    )
    parser.add_argument("--prompt", type=str, default="My name is", help="Prompt to use for the model.")
    parser.add_argument("--repeat", type=int, default=2, help="Factor to repeat key-value heads.")
    parser.add_argument(
        "--full_batch_size",
        "--full-batch-size",
        type=int,
        default=None,
        help="Set full batch size to enable continuous batching mode, default is None",
    )
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
        help="Number of attention heads, if not passed explicitly then will be picked from config.json",
    )
    parser.add_argument(
        "--hidden_size",
        "--hidden-size",
        type=int,
        default=None,
        help="Hidden size to use, if not passed explicitly then will be picked from config.json",
    )

    args = parser.parse_args()

    replicate_kv_heads(
        model_name=args.model_name,
        prompt=args.prompt,
        repeat=args.repeat,
        full_batch_size=args.full_batch_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        hidden_size=args.hidden_size,
    )
