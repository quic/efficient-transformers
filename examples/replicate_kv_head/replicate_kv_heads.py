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


def main(args):
    # Replace quantizers for loading Quantized AWQ/GPTQ models on CPU.
    replace_transformers_quantizers()
    # Load the model and tokenizer
    model_name = args.model_name
    model_base_name = model_name.split("/")[-1]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,  # num_hidden_layers=1,
        attn_implementation="eager",
    )

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
        k_proj = attn.k_proj
        v_proj = attn.v_proj
        if isinstance(attn.k_proj, (WQLinear_GEMM, QuantLinearGPTQ)):
            if attn.head_dim % 8 != 0:
                raise ValueError(f"the value attn.head_dim={attn.head_dim} is not divisible by 8 which is \
                                 according to the assumption that model is 4-bit quantized.")
            if attn.hidden_size % k_proj.group_size != 0 or attn.hidden_size % v_proj.group_size:
                raise ValueError(f"The value of attn.hidden_size={attn.hidden_size} is not divisible by \
                                K_proj.group_size={k_proj.group_size}")

            # Key projection duplication
            # Duplication of quantized weights
            k_proj.qweight.data = (
                torch.repeat_interleave(
                    k_proj.qweight.data.T.view(orig_kv_heads, attn.head_dim // 8, attn.hidden_size), repeat, 0
                )
                .view((new_kv_heads * attn.head_dim) // 8, attn.hidden_size)
                .T
            )
            # Duplication of quantized zero points
            k_proj.qzeros.data = (
                torch.repeat_interleave(
                    k_proj.qzeros.data.T.view(orig_kv_heads, attn.head_dim // 8, attn.hidden_size // k_proj.group_size),
                    repeat,
                    0,
                )
                .view((new_kv_heads * attn.head_dim) // 8, attn.hidden_size // k_proj.group_size)
                .T
            )
            # Duplication of quantization scales
            k_proj.scales.data = (
                torch.repeat_interleave(
                    k_proj.scales.data.T.view(orig_kv_heads, attn.head_dim, attn.hidden_size // k_proj.group_size),
                    repeat,
                    0,
                )
                .view(new_kv_heads * attn.head_dim, attn.hidden_size // k_proj.group_size)
                .T
            )
            k_proj.out_features = k_proj.out_features * repeat
        else:
            attn.k_proj.weight.data = torch.repeat_interleave(
                attn.k_proj.weight.data.view(orig_kv_heads, attn.head_dim, attn.hidden_size), repeat, 0
            ).view(new_kv_heads * attn.head_dim, attn.hidden_size)

        if isinstance(v_proj, (WQLinear_GEMM, QuantLinearGPTQ)):
            if attn.head_dim % 8 != 0:
                raise ValueError(f"the value attn.head_dim={attn.head_dim} is not divisible by 8 which is \
                                 according to the assumption that model is 4-bit quantized.")
            if attn.hidden_size % v_proj.group_size:
                raise ValueError(f"The value of attn.hidden_size={attn.hidden_size} is not divisible by \
                                v_proj.group_size = {v_proj.group_size}")

            # Value projection duplication
            # Duplication of quantized weights
            v_proj.qweight.data = (
                torch.repeat_interleave(
                    v_proj.qweight.data.T.view(orig_kv_heads, attn.head_dim // 8, attn.hidden_size), repeat, 0
                )
                .view((new_kv_heads * attn.head_dim) // 8, attn.hidden_size)
                .T
            )
            # Duplication of quantized zero points
            v_proj.qzeros.data = (
                torch.repeat_interleave(
                    v_proj.qzeros.data.T.view(orig_kv_heads, attn.head_dim // 8, attn.hidden_size // v_proj.group_size),
                    repeat,
                    0,
                )
                .view((new_kv_heads * attn.head_dim) // 8, attn.hidden_size // v_proj.group_size)
                .T
            )
            # Duplication of quantization scales
            v_proj.scales.data = (
                torch.repeat_interleave(
                    v_proj.scales.data.T.view(orig_kv_heads, attn.head_dim, attn.hidden_size // v_proj.group_size),
                    repeat,
                    0,
                )
                .view(new_kv_heads * attn.head_dim, attn.hidden_size // v_proj.group_size)
                .T
            )
            v_proj.out_features = v_proj.out_features * repeat
        else:
            attn.v_proj.weight.data = torch.repeat_interleave(
                attn.v_proj.weight.data.view(orig_kv_heads, attn.head_dim, attn.hidden_size), repeat, 0
            ).view(new_kv_heads * attn.head_dim, attn.hidden_size)

    # Generate modified outputs and tokens
    with torch.inference_mode():
        _ = model(**inputs)  # Modified output
        mod_tokens = model.generate(**inputs, max_new_tokens=10, num_beams=1, do_sample=False)

    # Print the original and modified token outputs
    print("Original:", tokenizer.batch_decode(orig_tokens))
    print("Modified:", tokenizer.batch_decode(mod_tokens))

    # Export the modified model
    q_model = QEFFAutoModelForCausalLM(model, model_name)
    export(
        model_name,
        q_model,
        tokenizer=tokenizer,
        onnx_dir_path=f"{model_base_name}-{new_kv_heads}kvheads",
    )

    # Undo the effect of replace_transformers_quantizers
    undo_transformers_quantizers()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Modify and export a causal language model.")
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name of the model to use."
    )
    parser.add_argument("--prompt", type=str, default="My name is", help="Prompt to use for the model.")
    parser.add_argument("--repeat", type=int, default=2, help="Factor to repeat key-value heads.")

    args = parser.parse_args()
    main(args)
