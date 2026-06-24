# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.base.modeling_qeff import apply_head_pruning, apply_layer_removal

# ---------------------------------------------------------------------------
# Skip-layer config
# List the decoder-layer indices to physically remove before ONNX export.
# deletes the layers and re-indexes KV cache slots — producing a genuinely
# smaller ONNX with no cache-slot divergence.
# Uncomment / add indices to remove additional layers.
# ---------------------------------------------------------------------------
# # ---------------------------------------------------------------------------
# Head pruning config (MHA only — num_kv_heads must equal num_q_heads)
# Keys   : decoder layer index
# Values : list of attention head indices to prune from that layer
#
# Set PRUNE_HEADS = {} to disable head pruning entirely.
# ---------------------------------------------------------------------------
PRUNE_HEADS: dict[int, list[int]] = {
    1: [3, 4, 5, 7, 19, 25, 31],
    2: [15, 27],
    3: [0, 2, 6, 9, 12, 13, 15, 16, 22, 23, 24, 27, 28],
    # 4: [5, 8, 11, 17, 20, 25],
    5: [30],
    6: [2],
    10: [5, 11],
    11: [18],
    # 12: [12, 20, 22],
    14: [14, 21, 25, 31],
    15: [18, 25],
    16: [27],
    17: [3, 24],
    18: [18, 21, 23, 27],
    19: [14, 21],
    20: [2, 8, 9, 18, 23, 24, 31],
    21: [1, 13, 19, 30],
    22: [0, 1, 3, 5, 8, 10, 11, 16, 17, 19, 25, 26, 28, 29],
    23: [7, 12, 23, 24, 28],
    24: [0, 6, 11, 13, 14, 19, 27, 31],
    25: [0, 6, 11, 17, 21, 27, 28],
    26: [0, 2, 6, 8, 10, 13, 14, 15, 23, 24, 28],
    27: [0, 2, 3, 5, 7, 13, 20, 22, 23, 29, 31],
    28: [0, 3, 4, 5, 8, 11, 15, 17, 19, 20, 21, 22, 26, 27, 29],
    29: [2, 3, 7, 8, 13, 25],
    # 30: [0, 14, 20, 22, 23, 27, 28, 31],
 }


def _remap_prune_layers_after_removal(
    prune_heads: dict[int, list[int]],
    removed_layers: list[int],
) -> dict[int, list[int]]:
    """Map original layer indices to post-removal indices for head pruning.

    ``apply_layer_removal`` compacts decoder layers to 0..N-1. When users keep
    ``PRUNE_HEADS`` in original model index space, those indices must be remapped
    before calling ``apply_head_pruning`` on the already-compacted model.
    Any pruned layer that was itself removed is skipped.
    """
    if not prune_heads:
        return {}
    if not removed_layers:
        return dict(prune_heads)

    removed_set = set(removed_layers)
    remapped: dict[int, list[int]] = {}

    for original_layer, heads in sorted(prune_heads.items()):
        if original_layer in removed_set:
            continue
        new_layer = original_layer - sum(1 for layer in removed_set if layer < original_layer)
        remapped[new_layer] = heads

    return remapped

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Basic text generation inference")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen1.5-7B", help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default="Hello, how are you? Response in english. ", help="Input prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context length")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument("--batch-size", type=int, default=1,help="Batch size")
    parser.add_argument("--aic-hw-version", type=str, default="ai100", help="Version of aic hardware")
    parser.add_argument(
        "--remove-hidden-layers",
        type=lambda layers: [int(x) for x in layers.strip("[]").split(",")],
        default=[4,12],required=False,
        help="Layer indices to remove e.g. [2,3,17]",
    )
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=None,
        help="Device IDs (comma-separated) e.g. [0,1]",
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Both transforms must be applied to the raw HF model BEFORE QEff wraps it:
    #   1. apply_layer_removal first — physically deletes layers and re-indexes
    #      layer_idx so KVCacheTransform captures the correct slot numbers.
    #   2. apply_head_pruning second — slices projection weights; reads the
    #      already-updated layer count from config.
    # Loading in float32: QEff exports in float32 and the KV cache dummy
    # tensors in modeling_auto.py are hardcoded float32.
    if args.remove_hidden_layers or PRUNE_HEADS:
        print(f"Loading raw HF model ({args.model_name}) ...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            attn_implementation="eager",
        )

        if args.remove_hidden_layers:
            print(f"Applying layer removal: {args.remove_hidden_layers}")
            apply_layer_removal(hf_model, args.remove_hidden_layers)

        if PRUNE_HEADS:
            remapped_prune_heads = _remap_prune_layers_after_removal(
                PRUNE_HEADS,
                args.remove_hidden_layers,
            )
            print(f"Applying head pruning (remapped): {remapped_prune_heads}")
            apply_head_pruning(hf_model, remapped_prune_heads)

        model = QEFFAutoModelForCausalLM(
            hf_model,
            pretrained_model_name_or_path=args.model_name,
        )
    else:
        model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name)

    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        batch_size=args.batch_size,
        num_devices=(1 if args.device_group is None else len(args.device_group)),
        use_onnx_subfunctions=True,
    )
    print(f"Model compiled to: {qpc_path}")

    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        device_id=args.device_group,
        generation_len=args.generation_len,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")


if __name__ == "__main__":
    main()



