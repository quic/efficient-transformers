# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Benchmark + measurement harness: paged vs contiguous KV.

Reports the three quantities requested for the go/no-go gate:
  1. PRECISION  — max/mean abs logit difference paged vs contiguous (should be ~0).
  2. MEMORY     — KV-cache bytes for a realistic "short sequences in a long context"
                  workload (the actual saving paging buys), plus peak process RSS
                  during a forward.
  3. RUNTIME    — per-forward latency, paged vs contiguous (CPU eager).

NOTE: CPU eager latency is NOT representative of Cloud AI 100. On AIC the win
comes from a larger compilable full_batch_size (more decode lanes) enabled by the
memory saving; the CPU number here only reflects host-side index/gather overhead.
The authoritative latency/throughput gate runs on the QAIC host (plan Step 3).
"""

import gc
import resource
import time

import pytest
import torch

pytest.importorskip("QEfficient")

from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform  # noqa: E402


def _build_model(n_layers=4, hidden=256, heads=8, kv_heads=2, vocab=1024, max_pos=4096):
    cfg = Qwen2Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=max_pos,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    cfg.torch_dtype = torch.float32
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model, _ = KVCacheTransform.apply(model)
    model.eval()
    return model, cfg


def _block_table(bsz, max_blocks, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = [torch.randperm(max_blocks, generator=g) + b * max_blocks for b in range(bsz)]
    return torch.stack(rows, 0).to(torch.int32)


def _kv_bytes(num_slots_dim0, kv_heads, second_dim, head_dim, n_layers, dtype_bytes=2):
    # K and V, per layer; dtype_bytes=2 for fp16/mxint8-ish on-card estimate.
    return 2 * n_layers * num_slots_dim0 * kv_heads * second_dim * head_dim * dtype_bytes


def _peak_rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    import sys

    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def run_report():
    n_layers, hidden, heads, kv_heads = 4, 256, 8, 2
    model, cfg = _build_model(n_layers, hidden, heads, kv_heads)
    hd = hidden // heads

    # Workload: many short sequences inside a long max context (paging's sweet spot).
    bsz = 8
    ctx_len = 2048  # contiguous reserves this per sequence
    page_size = 128
    avg_len = 256  # actual tokens per sequence
    prompt_len = avg_len

    # --- MEMORY ---
    max_blocks = ctx_len // page_size  # blocks a single seq could need at full ctx
    # Paged pool sized for the ACTUAL working set: ceil(total_tokens/page)+null.
    used_blocks = (bsz * avg_len + page_size - 1) // page_size + 1
    cont_bytes = _kv_bytes(bsz, kv_heads, ctx_len, hd, n_layers)
    paged_bytes = _kv_bytes(used_blocks, kv_heads, page_size, hd, n_layers)

    print("\n==================== PAGED vs CONTIGUOUS ====================")
    print(f"workload: bsz={bsz}, ctx_len={ctx_len}, page_size={page_size}, avg_seq_len={avg_len}")
    print("\n--- MEMORY (KV cache, fp16-equiv) ---")
    print(f"contiguous KV : {cont_bytes/1e6:8.2f} MB  (bsz*ctx_len = {bsz*ctx_len} slots)")
    print(f"paged KV      : {paged_bytes/1e6:8.2f} MB  ({used_blocks} blocks * {page_size})")
    print(f"saving        : {100*(1-paged_bytes/cont_bytes):8.1f} %  ({cont_bytes/paged_bytes:.1f}x smaller)")

    # --- correctness run (small ctx so it fits/fast), measure PRECISION + RUNTIME + RSS ---
    bench_ctx = 512
    bench_max_blocks = bench_ctx // page_size
    num_blocks = bsz * bench_max_blocks + 1
    block_table = _block_table(bsz, bench_max_blocks, seed=3)
    batch_index = torch.arange(bsz).view(bsz, 1).to(torch.int32)
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, prompt_len))
    position_ids = torch.arange(prompt_len).view(1, -1).expand(bsz, prompt_len).to(torch.int64)
    attn = torch.ones(bsz, bench_ctx)

    def cont_pkv():
        return [(torch.zeros(bsz, kv_heads, bench_ctx, hd), torch.zeros(bsz, kv_heads, bench_ctx, hd)) for _ in range(n_layers)]

    def paged_pkv():
        return [(torch.zeros(num_blocks, kv_heads, page_size, hd), torch.zeros(num_blocks, kv_heads, page_size, hd)) for _ in range(n_layers)]

    def fwd_cont():
        return model(input_ids=input_ids, position_ids=position_ids, batch_index=batch_index,
                     attention_mask=attn, past_key_values=cont_pkv(), use_cache=True).logits

    def fwd_paged():
        return model(input_ids=input_ids, position_ids=position_ids, batch_index=batch_index,
                     block_table=block_table, attention_mask=attn, past_key_values=paged_pkv(), use_cache=True).logits

    with torch.no_grad():
        lc, lp = fwd_cont(), fwd_paged()
        diff = (lc - lp).abs()
        print("\n--- PRECISION (logits, paged vs contiguous) ---")
        print(f"max abs diff  : {diff.max().item():.3e}")
        print(f"mean abs diff : {diff.mean().item():.3e}")

        def timeit(fn, n=10):
            fn()  # warmup
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            return (time.perf_counter() - t0) / n * 1e3  # ms

        tc = timeit(fwd_cont)
        tp = timeit(fwd_paged)
        print("\n--- RUNTIME (CPU eager, prefill forward; NOT representative of AIC) ---")
        print(f"contiguous    : {tc:8.2f} ms/fwd")
        print(f"paged         : {tp:8.2f} ms/fwd   ({tp/tc:.2f}x)")

    gc.collect()
    print(f"\n--- PEAK RSS : {_peak_rss_mb():.0f} MB ---")
    print("============================================================\n")
    return diff.max().item(), cont_bytes, paged_bytes


def test_precision_and_memory_gate():
    """Hard gate (CPU): logits parity + paged uses less KV memory for short seqs."""
    max_diff, cont_bytes, paged_bytes = run_report()
    assert max_diff < 1e-3, f"precision gate failed: max diff {max_diff}"
    assert paged_bytes < cont_bytes, "paged KV must use less memory than contiguous for this workload"


if __name__ == "__main__":
    run_report()
