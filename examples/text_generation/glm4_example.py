# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

MODEL_ID = "tiny-random/glm-4-moe"
TOKENIZER_ID = "zai-org/GLM-4.7"


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


def optionally_replicate_kv_heads(qeff_model, repeat: int = 1):
    if repeat <= 1:
        return

    orig_kv_heads = qeff_model.model.config.num_key_value_heads
    new_kv_heads = repeat * orig_kv_heads
    qeff_model.model.config.num_key_value_heads = new_kv_heads

    num_attention_heads = qeff_model.model.config.num_attention_heads
    hidden_size = qeff_model.model.config.hidden_size

    for block in qeff_model.model.model.layers:
        attn = block.self_attn
        attn.num_key_value_heads = new_kv_heads
        attn.num_key_value_groups = num_attention_heads // new_kv_heads
        duplicate_weights_for_linear_layer(attn.k_proj, orig_kv_heads, repeat, attn.head_dim, hidden_size)
        duplicate_weights_for_linear_layer(attn.v_proj, orig_kv_heads, repeat, attn.head_dim, hidden_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--replicate-kv-heads", type=int, default=1)
    args = parser.parse_args()

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(args.model_id)
    optionally_replicate_kv_heads(qeff_model, args.replicate_kv_heads)

    qeff_model.compile(
        prefill_seq_len=1,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        mxfp6_matmul=True,
        num_devices=args.num_devices,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
        retain_full_kv=True,
        qaic_config={"enable_blocking": True, "blocking_mode": "kv", "num_kv_blocks": 2},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    qeff_model.generate(prompt=["Once upon a time,"], tokenizer=tokenizer)


if __name__ == "__main__":
    main()
