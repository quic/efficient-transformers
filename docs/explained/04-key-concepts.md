# Key concepts

The [architecture doc](03-architecture.md) showed *where* transformation happens.
This doc explains the two ideas that actually make a transformed model correct
and fast on the accelerator:

1. **Model-block reimplementation** — rewriting a few forward methods for static,
   export-friendly, fp16-safe execution.
2. **The static gather/scatter KV cache** — replacing HF's growing cache with a
   fixed-size buffer.

We use **Llama** as the worked example, because most other families follow the
same pattern.

---

## 1. Model-block reimplementation

Recall from Stage 0 that QEfficient swaps only the class pointer of a few blocks
and keeps the original weights. For Llama, the swapped classes live in
`QEfficient/transformers/models/llama/modeling_llama.py`. Each one is a near-copy
of upstream HF with a small, deliberate set of changes. Every class docstring
even states *"The only differences are …"*. Here is what actually changes and
why.

### Static, precomputed RoPE

Upstream HF recomputes the rotary position embedding (the `cos`/`sin` tables)
on every forward pass. That's fine eagerly, but on an AOT-compiled graph it's
wasted work baked into the graph.

QEfficient precomputes the `cos`/`sin` tables once into constant buffers (built
in the `__qeff_init__` hook that fires right after the class swap). At runtime
the attention forward just **indexes** those buffers by `position_ids` — a
static table lookup instead of per-step trigonometry. This is also why
`position_ids` is an explicit graph input (see Stage 1).

### Boolean causal masks with finite fill values

Upstream masking adds a large-magnitude float tensor to the attention scores.
QEfficient instead builds a **boolean** causal mask (comparing key indices
against query indices) and applies it with `torch.where`, filling masked
positions with a finite minimum value. Two reasons: boolean comparisons export
more cleanly than constructing a big additive float tensor, and using a finite
fill (rather than `-inf` / `float32.min`) avoids extreme constants that
destabilize the fp16 compiled graph.

### Last-token slicing before the LM head

During decode you only need logits for the *last* position, but naively the LM
head would run over the whole sequence. QEfficient computes the last valid index
from `position_ids` and gathers just that one hidden state before applying
`lm_head`, so the (large) vocabulary projection runs on a single token instead of
the full sequence.

### Fused, numerically-stable normalization

Instead of emitting RMSNorm as a decomposed `pow → mean → rsqrt → mul` subgraph
(which the compiler would run in whatever precision, risking overflow in the
squared-sum), `CustomOpsTransform` swaps every `*RMSNorm` for `CustomRMSNormAIC`
— a single fused custom op (`com.qti.aisw.onnx::CustomRMSNorm`) that the compiler
executes as one numerically-stable kernel (it casts to fp32 internally for the
reduction). Gemma's variant handles its "weight stored as zeros, +1 at runtime"
convention inside the fused op so both normal and weight-free export stay
correct.

### fp16 overflow/underflow handling

fp16's max value is 65504; bf16/fp32 activations above that become `inf` once the
graph is converted to fp16, which corrupts everything downstream. Models that
need it (Gemma-4 is the clearest example, in
`QEfficient/transformers/models/gemma4/modeling_gemma4.py`) add **export-time
only** safety: clamp hidden states into fp16 range, do residual adds in fp32 then
clamp before casting back, use `-65504` instead of `float32.min` as the mask
fill, and replace any `inf`/`nan` clip bounds with finite fallbacks. These guards
are gated on "is this an ONNX export?" so eager PyTorch numerics are unchanged —
preserving parity.

---

## 2. The static gather/scatter KV cache

This is the single most important adaptation, and the reason a growing HF cache
can't just be moved to the card.

### The problem with `DynamicCache`

HF's `DynamicCache` **concatenates** each step's new key/value onto the stored
tensors, so the cache grows one token longer every decode step. Its shape is
different at every step. An AOT compiler needs fixed shapes, so this cannot be
compiled directly.

### The fixed-buffer solution

QEfficient's cache (`QEfficient/transformers/cache_utils.py`, primarily
`QEffDynamicCache` and its per-layer `QEffDynamicLayer`) pre-allocates a
**fixed** buffer of shape `[batch, kv_heads, ctx_len, head_dim]` and never
changes its shape. Each decode step does two operations, keyed by `position_ids`:

- **Scatter (write).** Write the new key/value into the buffer *at the token's
  position* using a `ScatterND`-based custom op (`ctx_scatter`), or the
  continuous-batching variant `ctx_scatter_cb` which also takes `batch_index`
  to pick the right cache line.
- **Gather (read).** Read back only the *valid* window of the buffer (positions
  up to the current maximum) using a `GatherND`-based custom op (`ctx_gather`),
  masking out and zeroing the not-yet-written slots so uninitialized memory never
  pollutes attention.

These custom ops live in `QEfficient/customop/ctx_scatter_gather.py` (with a
`_cb` continuous-batching file) and are defined in the `com.qualcomm.cloud` ONNX
domain that the QAIC compiler recognizes. The `compute-context-length` (CCL)
mechanism lets the gather return only the active context width, so attention
computes over just the tokens that exist rather than the full `ctx_len`.

There are specialized variants for different execution modes — read-only (prefill
paths), batch-folded reads (reshaping `[FBS, Hkv, T, D]` → `[1, FBS*Hkv, T, D]`
so the compiler can pattern-match its fast on-chip gather), MLA/compressed caches
for DeepSeek, and hybrid/sliding-window caches — but they all share the same
core idea: **a fixed buffer with position-keyed scatter/gather instead of a
growing concatenation.**

### Why this matters end to end

Because the buffer is fixed-shape, the graph compiles. Because the writes/reads
are position-keyed custom ops the compiler recognizes, they're fast. And because
these tensors are exported as `*_RetainedState` outputs and *skipped* by the
runtime session (see [Stage 3](03-architecture.md#stage-3--run-on-device-generate)),
the cache stays on the card between steps. The static cache and the retained-state
runtime are two halves of the same design.

---

## Beyond the basics

Everything above is the foundation. On top of it, QEfficient layers a set of
optional features — each documented with runnable recipes in
[Features Enablement](../source/features_enablement.md) and catalogued in
[Supported Features](../source/supported_features.rst):

- **Continuous batching** — dynamic request batching with slot recycling.
- **Speculative decoding** — draft/target models and multi-projection heads
  (also block-diffusion "DFlash" drafting).
- **Multi-device (MDP)** — tensor-parallel and pipeline-parallel / disaggregated
  serving across cards.
- **Quantization** — MXFP6 weights, MXINT8 KV cache, FP8, and loading
  pre-quantized AWQ/GPTQ checkpoints.
- **PEFT / LoRA** — single-adapter and multi-adapter ("Finite-LoRA") inference.
- **VLM dual-QPC** — splitting the vision encoder and language model into
  separate compiled programs (`kv_offload`).
- **QNN compilation** — the alternative Qualcomm QNN SDK backend.
- **On-device sampling** — running sampling on the card to cut host round-trips.

---

**Where this lives in the code**

- Llama reimplementation (worked example):
  `QEfficient/transformers/models/llama/modeling_llama.py`
- Fused norm custom op: `QEfficient/customop/rms_norm.py`
- fp16 clamping example: `QEfficient/transformers/models/gemma4/modeling_gemma4.py`
- Static KV cache: `QEfficient/transformers/cache_utils.py`
- Scatter/gather custom ops: `QEfficient/customop/ctx_scatter_gather.py`,
  `QEfficient/customop/ctx_scatter_gather_cb.py`

**Next:** [Optimization techniques](06-optimization-techniques.md) — the full
catalog of what QEfficient does to run models efficiently.
