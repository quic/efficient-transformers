# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
GPT-OSS blocking benchmark example.

Runs two benchmark passes for GPT-OSS:

  1. Decode-only  (prefill_seq_len=1)  — 3 modules in table:
       swa_attention | full_attention_blocked_kv | moe

  2. Prefill-only (prefill_seq_len=32) — 3 modules in table:
       prefill_swa_attention | prefill_full_attention_blocked_kv | prefill_moe

SWA layers are never blocked (mainline skips blocking for sliding-window
attention). Full-attention layers get KV blocking applied.

Usage
-----
# Tiny model (test):
python examples/text_generation/gpt_oss_blocking_benchmark.py

# Real model with explicit block counts:
python examples/text_generation/gpt_oss_blocking_benchmark.py \\
    --model <hf-model-id> \\
    --num-kv-blocks 4 \\
    --ctx-len 4096 \\
    --prefill-seq-len 256
"""

import argparse

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode


def run(model_id: str, prefill_seq_len: int, ctx_len: int, num_kv_blocks: int, num_cores: int, num_devices: int):
    bc = AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=num_kv_blocks)

    # ── Decode-only ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  DECODE-ONLY  (prefill_seq_len=1, ctx_len={ctx_len})")
    print(f"  blocking: KV  num_kv_blocks={num_kv_blocks}")
    print(f"{'=' * 60}")

    m_decode = QEFFAutoModelForCausalLM.from_pretrained(model_id, enable_benchmark=True)
    m_decode.compile(
        prefill_only=False,
        prefill_seq_len=1,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        blocking_config=bc,
    )
    m_decode.generate(tokenizer=None, prompts=[])

    # ── Prefill-only ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  PREFILL-ONLY  (prefill_seq_len={prefill_seq_len}, ctx_len={ctx_len})")
    print(f"  blocking: KV  num_kv_blocks={num_kv_blocks}")
    print(f"{'=' * 60}")

    m_prefill = QEFFAutoModelForCausalLM.from_pretrained(model_id, enable_benchmark=True)
    m_prefill.compile(
        prefill_only=True,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        blocking_config=bc,
    )
    m_prefill.generate(tokenizer=None, prompts=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", default="tiny-random/gpt-oss-bf16", help="HF model id (default: tiny-random/gpt-oss-bf16)"
    )
    parser.add_argument("--prefill-seq-len", type=int, default=32)
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--num-kv-blocks", type=int, default=2, help="Number of KV blocks for KV blocking (default: 2)")
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--num-devices", type=int, default=1)
    args = parser.parse_args()

    run(
        model_id=args.model,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_kv_blocks=args.num_kv_blocks,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
    )


if __name__ == "__main__":
    main()
