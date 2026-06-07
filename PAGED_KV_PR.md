# Paged (block-table) KV cache for Cloud AI 100 — Qwen2

## Summary

Adds **vLLM-style paged attention** (block-table KV) to QEfficient, so KV is stored
in a shared **block pool** `[num_blocks, num_heads, page_size, head_dim]` instead of
reserving a full `[full_batch_size, num_heads, ctx_len, head_dim]` slot per sequence.
A request's token at logical position `p` is resolved through a per-request
`block_table`:

```
physical_block = block_table[seq, p // page_size]
intra_offset   = p %  page_size
```

This removes the per-sequence over-allocation (waste drops from `max_model_len` to
`≤ page_size`), so the same on-card memory affords a larger compiled
`full_batch_size` → more decode lanes → higher throughput. Reference model: **Qwen2 /
Qwen2.5**. The core (ops + cache) is model-agnostic; only the per-model `block_table`
threading is model-specific.

## What changed

**Core (≈430 LoC):**
- `QEfficient/customop/ctx_paged_scatter_gather.py` — new `CtxScatterPagedFunc` /
  `CtxGatherPagedFunc` ops (block-pool ScatterND/GatherND, same op vocabulary as the
  continuous-batching ops; only the index construction differs).
- `QEfficient/transformers/cache_utils.py` — `QEffPagedDynamicLayer` /
  `QEffPagedDynamicCache` (block-pool cache; null-block routing for padding;
  CCL-aware gather; defensive clamp).
- `QEfficient/transformers/models/qwen2/modeling_qwen2.py` — thread optional
  `block_table` through `QEffQwen2ForCausalLM → Model → DecoderLayer → Attention`;
  select the paged cache when `block_table` is supplied; paged-mode guards.
- `QEfficient/blocking/attention_blocking.py` — thread `block_table` into the shared
  `past_key_value_update` / `generic_blocked_attention_interface` (centralizes paged
  support for any model that adopts the threading).
- `QEfficient/customop/__init__.py` — export the new ops.

**Tests (`tests/customop/`):** `test_paged_kv_parity.py`, `test_paged_cache_layer.py`,
`test_paged_qwen2_e2e.py`, `test_paged_onnx_export.py`, `test_paged_qwen2_onnx.py`,
`bench_paged_vs_contiguous.py`, `_metrics.py`.

## Why this is correct (test commands + results)

CPU venv (mac/Linux): `python tests/customop/<name>.py`. All pass.

- **Eager parity (bit-exact vs contiguous):**
  - ops `data` round-trip vs contiguous KV across prefill + decode, shuffled
    block_table, multi-sequence isolation, CCL, indirection negative-control →
    `max_abs_diff = 0.0`.
  - paged cache layer vs the proven `QEffDynamicLayer` (CB) across prefill + 10
    decode steps (page-boundary crossing) + CCL + different-length seqs with padding →
    `0.0`.
  - full Qwen2 model logits, paged vs contiguous, prefill + 3-step decode (cache
    persistence across calls) → `0.0`.
- **ONNX symbolic numerics (the AIC-consumed path):** export the paged ops, run under
  onnxruntime, compare to eager → `max_abs_diff = 0.0`. (This caught and fixed two
  real symbolic-only bugs: an int32/int64 ScatterND-index Concat mismatch and a
  non-scalar `Range`.)
- **Full-model structural export:** `block_table` is a real ONNX graph input of the
  whole Qwen2 model; paged scatter/gather ops present.
- **Measured (every test reports time / peak-RSS / precision):** memory — paged
  `2.23 MB` vs contiguous `16.78 MB` for short-seqs-in-long-context (**7.5× / 86.7%
  smaller**); CPU-eager latency ≈ `1.0×` (host-side; not the AIC metric).

## NOT in this PR (box-gated — needs a QAIC host)

1. **Production export-plumbing** in `QEFFAutoModelForCausalLM.export()`: a `paged_kv`
   flag that shapes `past_key_values` as the block pool and adds `block_table` to
   `example_inputs` / `dynamic_axes` / specializations (the runtime threading is
   already export-ready; this is the pipeline wiring).
2. **AIC compile** of the paged QPC + on-card **accuracy** and **throughput/FBS**
   go/no-go (plan Step 3). CPU-eager and onnxruntime-ops numerics are bit-exact, but
   the authoritative full-graph numeric/perf gate is the AIC compiler + card.
3. **vLLM-QAIC plugin** changes (separate repo): un-disable paging in `platform.py`
   (`block_size = page_size`, re-enable prefix caching), feed vLLM's block_table into
   the QPC in `model_runner.py`. (This also re-populates `Request.block_hashes`,
   unblocking the later Mooncake-Store work.)

## Known pre-existing (not introduced here)
- `modeling_qwen2.py` blocking branch passes `comp_ctx_length=` (singular) where the
  helper expects `comp_ctx_lengths=`; silently drops CCL in the blocked path. Upstream
  bug, left untouched (out of scope).
- The contiguous `QEffDynamicLayer` masks `v_out` with a hardcoded float32 zero
  (dtype risk for fp16/bf16). The paged path uses `v_out.dtype`; the contiguous path
  is left as-is.

## AI assistance
This change was developed with AI assistance (Claude). Every source change is covered
by a CPU test; reviews were performed and findings addressed. A human must review and
run the box-gated steps before production use.
