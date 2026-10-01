# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""MiniMax-M3 paged GQA/MSA decode with configurable page counts.

The GQA, indexer, and sparse-attention block tables can use different
logical-page counts. Each value also controls its corresponding physical cache
capacity during export and compilation through ``qaic_config``. Use
``--layout-only`` to inspect the resulting grouped tables without loading the
model or using QAIC.
"""

import argparse
import math
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "MiniMaxAI/MiniMax-M3"


def build_block_table(
    dp: int,
    cp: int,
    batch_size: int,
    num_logical_pages: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Build a DP-major table with one physical page per CP page group."""
    if batch_size % dp:
        raise ValueError("batch_size must be divisible by dp")
    batch_local = batch_size // dp
    num_page_groups = math.ceil(num_logical_pages / cp)
    physical_pages_per_dp = batch_local * num_page_groups
    table = torch.empty((dp, batch_local, num_page_groups), dtype=torch.int32)
    for dp_idx in range(dp):
        table[dp_idx] = torch.randperm(
            physical_pages_per_dp,
            dtype=torch.int32,
            generator=generator,
        ).view(batch_local, num_page_groups)
    return table


def print_page_layout(
    label: str,
    block_table: torch.Tensor,
    cp: int,
    num_logical_pages: int,
) -> None:
    """Print the logical-page mapping used by CP-grouped paged decode."""
    print(
        f"[{label}] logical_pages={num_logical_pages}, cp={cp}, "
        f"page_groups={math.ceil(num_logical_pages / cp)}, table_shape={tuple(block_table.shape)}"
    )
    preview_pages = min(num_logical_pages, max(cp + 1, 4))
    for logical_page in range(preview_pages):
        page_group = logical_page // cp
        owner_cp = logical_page % cp
        physical_page = int(block_table[0, 0, page_group])
        print(f"  logical_page={logical_page}: group={page_group}, owner_cp={owner_cp}, physical_page={physical_page}")


def tokenize_prompt(tokenizer, prompt: str, min_prompt_tokens: int):
    """Repeat the prompt until decode starts at or beyond the requested token position."""
    repetitions = 1
    while True:
        repeated_prompt = " ".join([prompt] * repetitions)
        messages = [[{"role": "user", "content": [{"type": "text", "text": repeated_prompt}]}]]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if inputs["input_ids"].shape[1] >= min_prompt_tokens:
            return inputs
        repetitions *= 2


def expand_batch(inputs, batch_size: int):
    expanded = {}
    for name, value in inputs.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
            expanded[name] = value.repeat((batch_size,) + (1,) * (value.ndim - 1))
        else:
            expanded[name] = value
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniMax-M3 paged MSA decode with independently configurable logical page counts."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--page-block-size", type=int, default=128)
    parser.add_argument(
        "--gqa-num-pages",
        type=int,
        default=64,
        help="Logical pages per sequence in the dense GQA K/V cache.",
    )
    parser.add_argument(
        "--msa-indexer-num-logical-pages",
        type=int,
        default=64,
        help="Logical pages per sequence in the index-key cache.",
    )
    parser.add_argument(
        "--msa-attn-num-logical-pages",
        type=int,
        default=64,
        help="Logical pages per sequence in the full attention K/V cache.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--msa-indexer-dp", type=int, default=2)
    parser.add_argument("--msa-indexer-cp", type=int, default=4)
    parser.add_argument("--msa-attn-dp", type=int, default=2)
    parser.add_argument("--msa-attn-cp", type=int, default=4)
    parser.add_argument("--num-kv-blocks", type=int, default=2)
    parser.add_argument("--indexer-num-blocks", type=int, default=None)
    parser.add_argument("--msa-num-kv-blocks", type=int, default=None)
    parser.add_argument("--indexer-n-head", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=16)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--generation-len", type=int, default=32)
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=129,
        help="Repeat the prompt until decode starts after at least this many tokens; 129 exercises CP owner row 1.",
    )
    parser.add_argument("--block-table-seed", type=int, default=17)
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="Print grouped table shapes and logical-page mappings, then exit before loading the model.",
    )
    parser.add_argument("--skip-generate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--expert-parallel-chunk-size",
        type=int,
        default=256,
        help="MoE expert-parallel chunk size (expert_parallel_chunk_size in moe_config).",
    )
    parser.add_argument(
        "--cores-per-expert",
        type=int,
        default=2,
        help="Number of NSP cores assigned to each expert during decode.",
    )
    parser.add_argument(
        "--no-tree-reduce",
        dest="tree_reduce",
        action="store_false",
        default=True,
        help="Disable tree-reduce for MoE expert-parallel dispatch.",
    )
    parser.add_argument(
        "--enable-proxy", action="store_true", help="Enable QEff proxy transforms during model loading."
    )
    args = parser.parse_args()

    if any(value < 1 for value in (args.msa_indexer_dp, args.msa_indexer_cp, args.msa_attn_dp, args.msa_attn_cp)):
        parser.error("All MSA DP/CP factors must be positive")
    if args.page_block_size < 1:
        parser.error("--page-block-size must be positive")
    if args.min_prompt_tokens < 1:
        parser.error("--min-prompt-tokens must be positive")
    if any(
        value < 1
        for value in (
            args.gqa_num_pages,
            args.msa_indexer_num_logical_pages,
            args.msa_attn_num_logical_pages,
        )
    ):
        parser.error("All logical-page counts must be positive")
    if args.gqa_num_pages * args.page_block_size < args.ctx_len:
        parser.error("GQA logical-page capacity must be at least --ctx-len")
    if args.msa_indexer_num_logical_pages * args.page_block_size < args.ctx_len:
        parser.error("Indexer logical-page capacity must be at least --ctx-len")
    if args.msa_attn_num_logical_pages * args.page_block_size < args.ctx_len:
        parser.error("Attention logical-page capacity must be at least --ctx-len")
    if args.num_kv_blocks < 1:
        parser.error("--num-kv-blocks must be positive")
    if args.indexer_num_blocks is not None and args.indexer_num_blocks < 1:
        parser.error("--indexer-num-blocks must be positive")
    if args.msa_num_kv_blocks is not None and args.msa_num_kv_blocks < 1:
        parser.error("--msa-num-kv-blocks must be positive")
    if args.ctx_len % args.page_block_size:
        parser.error("--ctx-len must be divisible by --page-block-size")
    indexer_num_blocks = args.indexer_num_blocks or args.num_kv_blocks
    indexer_page_divisor = indexer_num_blocks * args.msa_indexer_cp
    if args.msa_indexer_num_logical_pages % indexer_page_divisor:
        parser.error("Indexer logical pages must be divisible by the indexer block count times --msa-indexer-cp")
    if (args.msa_indexer_num_logical_pages // indexer_page_divisor) % args.num_cores:
        parser.error("Indexer page groups per block must be divisible by --num-cores")
    if args.gqa_num_pages % args.num_kv_blocks:
        parser.error("GQA logical pages must be divisible by --num-kv-blocks")
    if (args.gqa_num_pages // args.num_kv_blocks) % args.num_cores:
        parser.error("GQA page groups per block must be divisible by --num-cores")

    dp_lcm = math.lcm(args.msa_indexer_dp, args.msa_attn_dp)
    execution_batch_size = args.batch_size * dp_lcm
    if execution_batch_size % args.msa_indexer_dp or execution_batch_size % args.msa_attn_dp:
        parser.error("DP factors must divide the execution batch size")

    table_generator = torch.Generator(device="cpu").manual_seed(args.block_table_seed)
    gqa_table = build_block_table(
        1,
        1,
        execution_batch_size,
        args.gqa_num_pages,
        table_generator,
    )
    indexer_table = build_block_table(
        args.msa_indexer_dp,
        args.msa_indexer_cp,
        execution_batch_size,
        args.msa_indexer_num_logical_pages,
        table_generator,
    )
    attention_table = build_block_table(
        args.msa_attn_dp,
        args.msa_attn_cp,
        execution_batch_size,
        args.msa_attn_num_logical_pages,
        table_generator,
    )
    print_page_layout("gqa", gqa_table, 1, args.gqa_num_pages)
    print_page_layout(
        "indexer",
        indexer_table,
        args.msa_indexer_cp,
        args.msa_indexer_num_logical_pages,
    )
    print_page_layout(
        "attention",
        attention_table,
        args.msa_attn_cp,
        args.msa_attn_num_logical_pages,
    )
    if args.layout_only:
        return

    config = AutoConfig.from_pretrained(args.model_id)
    if args.num_layers is not None:
        config.text_config.num_hidden_layers = args.num_layers

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        config=config,
        kv_offload=True,
        dtype=torch.float16,
        **({"enable_proxy": True} if args.enable_proxy else {}),
    )
    qaic_config = {
        "blocking_mode": "kv_headpar",
        "num_kv_blocks": args.num_kv_blocks,
        "indexer_num_blocks": args.indexer_num_blocks,
        "msa_num_kv_blocks": args.msa_num_kv_blocks,
        "msa_indexer_dp": args.msa_indexer_dp,
        "msa_indexer_cp": args.msa_indexer_cp,
        "msa_attn_dp": args.msa_attn_dp,
        "msa_attn_cp": args.msa_attn_cp,
        "indexer_n_head": args.indexer_n_head,
        "num_cores_per_device": args.num_cores,
        "paged_kv": True,
        "page_block_size": args.page_block_size,
        "gqa_num_logical_pages": args.gqa_num_pages,
        "msa_indexer_num_logical_pages": args.msa_indexer_num_logical_pages,
        "msa_attn_num_logical_pages": args.msa_attn_num_logical_pages,
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": args.expert_parallel_chunk_size,
            "cores_per_expert": args.cores_per_expert,
            "tree_reduce": args.tree_reduce,
        },
    }

    t0 = time.perf_counter()
    qpc_paths = qeff_model.compile(
        batch_size=execution_batch_size,
        prefill_seq_len=1,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        skip_vision=True,
        node_precision_info=True,
        use_onnx_subfunctions=True,
        mxint8_kv_cache=True,
        qaic_config=qaic_config,
    )
    print(f"[timing] compile: {time.perf_counter() - t0:.2f}s")
    print(f"QPC paths: {qpc_paths}")

    if args.skip_generate:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    inputs = tokenize_prompt(tokenizer, args.prompt, args.min_prompt_tokens)
    prompt_tokens = inputs["input_ids"].shape[1]
    if prompt_tokens + args.generation_len > args.ctx_len:
        parser.error(
            f"Tokenized prompt ({prompt_tokens}) plus generation length ({args.generation_len}) "
            f"exceeds --ctx-len ({args.ctx_len})"
        )
    first_decode_page = prompt_tokens // args.page_block_size
    if first_decode_page >= min(
        args.gqa_num_pages,
        args.msa_indexer_num_logical_pages,
        args.msa_attn_num_logical_pages,
    ):
        parser.error("The first decode position exceeds one of the configured logical-page capacities")
    print(
        f"[decode] prompt_tokens={prompt_tokens}, first_logical_page={first_decode_page}, "
        f"gqa=(group={first_decode_page}, owner_cp=0), "
        f"indexer=(group={first_decode_page // args.msa_indexer_cp}, owner_cp={first_decode_page % args.msa_indexer_cp}), "
        f"attention=(group={first_decode_page // args.msa_attn_cp}, owner_cp={first_decode_page % args.msa_attn_cp})"
    )
    inputs = expand_batch(inputs, execution_batch_size)

    inputs["gqa_block_table"] = gqa_table
    inputs["msa_indexer_block_table"] = indexer_table
    inputs["msa_attn_block_table"] = attention_table

    t0 = time.perf_counter()
    output = qeff_model.generate(inputs=inputs, generation_len=args.generation_len)
    elapsed = time.perf_counter() - t0
    generated = output.generated_ids.shape[-1]
    print(f"[timing] generation: {elapsed:.2f}s ({generated / elapsed:.02f} tok/s)")
    print(tokenizer.batch_decode(output.generated_ids))


if __name__ == "__main__":
    main()
