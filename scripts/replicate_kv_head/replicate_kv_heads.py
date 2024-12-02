# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM, export
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers, undo_transformers_quantizers
from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ


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
    else:
        layer.weight.data = torch.repeat_interleave(
            layer.weight.data.view(orig_kv_heads, head_dim, hidden_size), repeat, 0
        ).view(new_kv_heads * head_dim, hidden_size)


def main(args):
    # Load the model and tokenizer
    model_name = args.model_name
    model_base_name = model_name.split("/")[-1]
    # Replace quantizers for loading Quantized AWQ/GPTQ models on CPU.
    replace_transformers_quantizers()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # num_hidden_layers=1,  # Use for generating smaller model
        attn_implementation="eager",
    )
    # Undo the effect of replace_transformers_quantizers
    undo_transformers_quantizers()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(args.prompt, return_tensors="pt")

    # Generate original outputs and tokens
    with torch.inference_mode():
        _ = model(**inputs)  # original output
        orig_tokens = model.generate(**inputs, max_new_tokens=10, num_beams=1, do_sample=False)

    # Modify the number of key-value heads
    repeat = args.repeat
    orig_kv_heads = model.config.num_key_value_heads
    new_kv_heads = repeat * orig_kv_heads
    model.config.num_key_value_heads = new_kv_heads

    print("Original KV heads:", orig_kv_heads)
    print("Modified KV heads:", new_kv_heads)

    # Update the model's attention layers with new key-value heads
    for block in model.model.layers:
        attn = block.self_attn
        attn.num_key_value_heads = new_kv_heads
        attn.num_key_value_groups = block.self_attn.num_heads // new_kv_heads
        duplicate_weights_for_linear_layer(attn.k_proj, orig_kv_heads, repeat, attn.head_dim, attn.hidden_size)
        duplicate_weights_for_linear_layer(attn.v_proj, orig_kv_heads, repeat, attn.head_dim, attn.hidden_size)

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
    q_model = QEFFAutoModelForCausalLM(model, continuous_batching=(True if args.full_batch_size else False))
    export(
        model_name,
        q_model,
        tokenizer=tokenizer,
        onnx_dir_path=f"{model_base_name}-{new_kv_heads}kvheads",
        full_batch_size=(args.full_batch_size if args.full_batch_size else None),
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

    args = parser.parse_args()
    main(args)
