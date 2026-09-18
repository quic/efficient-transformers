# Optimization techniques

QEfficient's job is not just to *run* a Hugging Face model on Qualcomm Cloud AI
(QAIC) hardware — it's to run it *efficiently*. This page catalogs the concrete
optimization techniques the library uses, and explains each in a little detail:
what it does, why it helps on this hardware, and where it lives in the code.

The techniques fall into four groups:

- **Graph & numeric optimizations** — applied when the model is transformed and
  exported, so the compiled graph is smaller and numerically stable.
- **Memory & KV-cache optimizations** — keeping the (large) attention cache cheap
  to store and read.
- **Decode-throughput optimizations** — techniques that reduce host round-trips
  or produce more than one token per model call.
- **Compute-distribution optimizations** — spreading work across cores and
  multiple cards.

Many of these build on the two foundations from
[Key concepts](04-key-concepts.md) (block reimplementation + the static KV
cache) and the four-stage pipeline in [Architecture](03-architecture.md). Where a
feature has a runnable recipe, it's linked from
[Features Enablement](../source/features_enablement.md).

---

## Graph & numeric optimizations

### Precomputed RoPE (static rotary embeddings)

Upstream HF recomputes the rotary `cos`/`sin` tables on every forward. QEfficient
computes them once into constant buffers (in the `__qeff_init__` hook that runs
right after a block's class is swapped), so at runtime attention just **indexes**
those tables by `position_ids` instead of doing per-step trigonometry. The saved
work is baked out of the compiled graph entirely. This is why `position_ids` is
an explicit graph input.

*Where:* `QEfficient/transformers/models/llama/modeling_llama.py` (and every other
`modeling_<arch>.py`).

### Fused RMSNorm custom op

Rather than emit normalization as a decomposed `pow → mean → rsqrt → mul`
subgraph, `CustomOpsTransform` swaps every `*RMSNorm` for a single fused
`CustomRMSNorm` op (in the `com.qti.aisw.onnx` domain) that the compiler runs as
one numerically-stable kernel, casting to fp32 internally for the reduction. One
fused op compiles better and avoids overflow in the squared-sum than several
small ops.

*Where:* `QEfficient/customop/rms_norm.py`,
`QEfficient/transformers/models/pytorch_transforms.py` (`CustomOpsTransform`).

### fp16-safe execution (overflow/underflow handling)

The device runs in fp16, whose max value is 65504 — bf16/fp32 activations above
that become `inf` when the graph is converted, corrupting everything downstream.
Models that need it add **export-time-only** guards: clamp hidden states into
fp16 range, perform residual adds in fp32 then clamp before casting back, use a
finite `-65504` mask fill instead of `float32.min`/`-inf`, and replace any
`inf`/`nan` clip bounds with finite fallbacks. The guards are gated on "is this
an ONNX export?" so eager PyTorch numerics — and therefore parity — are
unchanged.

*Where:* `QEfficient/transformers/models/gemma4/modeling_gemma4.py` is the
clearest example; MoE clamped-GLU variants live in
`QEfficient/transformers/moe/profiles.py`.

### Boolean causal masks

Instead of adding a large-magnitude additive float mask to attention scores,
QEfficient builds a **boolean** causal mask (comparing key vs query indices) and
applies it with `torch.where` and a finite fill value. Boolean comparisons export
more cleanly and avoid the extreme constants that destabilize the fp16 graph.

*Where:* `QEfficient/transformers/modeling_attn_mask_utils.py` and the per-model
attention forwards.

### Last-token slicing before the LM head

During decode you only need logits for the last position. QEfficient computes the
last valid index from `position_ids` and gathers just that one hidden state
before `lm_head`, so the large vocabulary projection runs on a single token
instead of the whole sequence.

*Where:* the `forward` of each `…ForCausalLM` wrapper (e.g. `modeling_llama.py`).

### ONNX sub-functions (`use_onnx_subfunctions`)

A decoder stack is N identical blocks. With sub-functions enabled, the block is
exported **once** as an ONNX local function and *called* N times, instead of
inlining N copies into the graph. The compiler then processes one block body
rather than N — a large compile-time and compile-memory saving on many-layer
models. The model advertises its repeated block via `get_submodules_for_export`;
enabling the flag adds the `-sub-functions` compiler option.

*Where:* `QEfficient/utils/export_utils.py`, `QEfficient/base/onnx_transforms.py`,
per-model `get_submodules_for_export`. Enable via
`compile(..., use_onnx_subfunctions=True)` (or the CLI `--use-onnx-subfunctions`).

---

## Memory & KV-cache optimizations

### Static gather/scatter KV cache

The foundational one (covered in depth in [Key concepts](04-key-concepts.md)):
instead of HF's cache that grows by concatenation, QEfficient uses a **fixed-size**
`[batch, kv_heads, ctx_len, head_dim]` buffer and writes/reads it with
position-keyed `ScatterND`/`GatherND` custom ops. Fixed shapes are what make the
graph compilable; the position-keyed ops keep it fast.

*Where:* `QEfficient/transformers/cache_utils.py`,
`QEfficient/customop/ctx_scatter_gather.py`.

### On-device retention of the KV cache

The KV tensors are exported as `*_RetainedState` outputs and marked *retained* at
compile time. At runtime the session **skips** those buffers, so the cache stays
resident on the card and simply persists across `run()` calls — prefill and
decode only exchange `input_ids`/`position_ids`/`logits`. The large cache never
crosses the host boundary, which is the single biggest decode speedup.

*Where:* `QEfficient/generation/cloud_infer.py`,
`QEfficient/generation/text_generation_inference.py`.

### Compute Context Length (CCL)

Early in a sequence you've only produced a few tokens, so attending over the full
allocated `ctx_len` is wasted work. CCL generates a set of increasing
"compute context length" buckets (e.g. `[1024, 2048, …, ctx_len]`) and emits one
compile specialization per bucket. At runtime the generator picks the smallest
bucket larger than the current position and passes a `comp_ctx_lengths` tensor of
that size; the attention mask is truncated to it and the KV-cache gather returns
only that many positions — so the QKᵀ/softmax/PV matmuls run over a smaller
sequence dimension (less compute, smaller on-chip footprint) until the sequence
grows into the next bucket.

*Where:* `QEfficient/utils/check_ccl_specializations.py`,
`QEfficient/transformers/cache_utils.py` (`read_only`), threaded through
`modeling_auto.py` and the runtime. Recipe:
[compute_context_length example](https://github.com/quic/efficient-transformers/tree/main/examples/performance/compute_context_length).

### BlockedKV attention

For long-context decode, a full KV cache can exceed the on-chip VTCM budget.
BlockedKV **tiles** the attention computation along head / query / KV-sequence /
batch dimensions so each tile's working set (Q tile + KV tile + scores) fits under
the VTCM threshold, reading the cache block-by-block. A configurator can
auto-pick the number of blocks by choosing the fewest blocks whose per-core
footprint stays under budget.

*Where:* `QEfficient/blocking/` (`attention_blocking.py`,
`blocked_attention_forwards.py`, `blocking_configurator.py`). Controlled via
`qaic_config["blocking_mode"]` (plus optional block-count overrides).

### KV precision compression (MXINT8 KV cache)

`mxint8_kv_cache` compresses the KV cache to MXINT8 on device, making cache reads
and updates cheaper. Mechanically it isn't a matmul flag — it changes the
**custom-IO dtype** of the retained KV buffers (each `past_key`/`past_value`
retained-state tensor is tagged `mxint8` instead of the model dtype) and is also
passed to the compiler.

*Where:* `QEfficient/transformers/models/modeling_auto.py`
(`_add_retained_state_custom_io`, `custom_io` construction),
`QEfficient/base/modeling_qeff.py`.

### Prefix / prefill caching

Reuses already-computed KV for a shared prompt prefix across continuous-batching
slots instead of re-prefilling it. It works by giving the retained-state KV
buffers a named infix (`kv_cache_prefix`) so buffers can be paired/shared across
QPC I/O; a new batch slot can then reuse the populated cache positions for an
identical prefix. (Currently gated to run with chunked prefill.)

*Where:* `QEfficient/utils/_utils.py` (`apply_kv_cache_prefix`),
`QEfficient/transformers/models/modeling_auto.py`.

---

## Decode-throughput optimizations

### Chunked prefill

Long prompts are padded to a multiple of the prefill sequence length and run in
**chunks** through the prefill specialization, rather than requiring one enormous
fixed prefill graph. This bounds the prefill graph size and lets prompts longer
than a single prefill window be processed.

*Where:* `QEfficient/generation/text_generation_inference.py` (`run_prefill`).

### Continuous batching

Rather than waiting for the slowest sequence in a batch, continuous batching
recycles decode **slots**: when one sequence finishes, the next queued prompt is
prefilled into that slot, and a `batch_index` input selects which on-device cache
line it uses. This keeps the card busy and improves throughput/latency under
concurrent load. Enable it by compiling with `full_batch_size` (and *not* passing
a fixed `batch_size`).

*Where:* `QEfficient/generation/text_generation_inference.py`
(`run_continuous_batching_decode`),
`QEfficient/customop/ctx_scatter_gather_cb.py`. Recipe:
[continuous batching](../source/features_enablement.md).

### On-device sampling

Normally token sampling runs on the host, adding a device→host→device round-trip
each step. On-device sampling runs the whole pipeline — repetition/presence
penalties, temperature, Top-K, Top-P, Min-P, and random sampling via the
Gumbel-Max trick — as ONNX ops **inside the QPC**, so the graph outputs
`next_tokens` (and optionally `probs`) instead of raw logits. This removes the
per-step host round-trip and improves throughput. A greedy-only configuration
short-circuits to just `argmax`.

*Where:* `QEfficient/transformers/sampler/sampler.py`,
`QEfficient/utils/sampler_utils.py`. Recipe:
[on_device_sampling example](https://github.com/quic/efficient-transformers/blob/main/examples/on_device_sampling.py).

### Speculative decoding (draft-based, multi-projection, DFlash)

Speculative decoding produces several candidate tokens per expensive target-model
pass, verifying them in one shot:

- **Draft-based** — a small draft model proposes `k` tokens ahead; the large
  target model validates them in a single call.
- **Multi-projection heads (SpD)** — post-attention projection heads on the base
  model speculate tokens ahead without a separate draft model.
- **DFlash** — a lightweight draft predicts a whole block of tokens jointly using
  a two-stream (context/noise) attention mechanism, verified by the target in one
  pass.
- **Prompt-lookup decoding** — proposes continuations by matching against
  overlapping spans already present in the prompt/generated text (no model
  needed).

All amortize the memory-bound decode step by validating multiple tokens together.

*Where:* `QEfficient/transformers/spd/`,
`QEfficient/generation/dflash_generation.py`, `QEfficient/utils/spd_utils.py`.
Recipe: [draft-based speculative decoding](../source/features_enablement.md).

---

## Compute-distribution optimizations

### Multi-Device Partitioning (MDP): tensor- and pipeline-parallel

For models too large for one card — or to go faster — MDP splits the compiled
program across devices. QEfficient generates either a **tensor-slice** config
(split each layer's tensors across devices) or a **pipeline-parallel /
disaggregated** config (assign layer ranges to devices), passed to the compiler
as a partition config. In the public API you pass `num_devices` and
`mdp_num_partitions`; the per-partition tensor-slice count is derived internally.
Disaggregated serving additionally splits prefill and decode (and, for VLMs, the
vision encoder vs the language model) into separate programs.

*Where:* `QEfficient/compile/mdp_generator.py`, `QEfficient/compile/`.

### KV-head replication

For GQA/MQA models (fewer KV heads than attention heads), replicating the KV
projection weights creates more KV heads, which distribute better across cores
and tensor-slices on this hardware — improving utilization versus a handful of
shared KV heads. The transform duplicates each `k_proj`/`v_proj` `n_repeat` times
and updates the head counts, with checks that the new count divides the attention
heads and that the model isn't MLA.

*Where:* `QEfficient/transformers/models/pytorch_transforms.py`
(`ReplicateKVHeadTransform`). Controlled via `num_replicate_kv_heads`; a
standalone [script](https://github.com/quic/efficient-transformers/tree/main/scripts/replicate_kv_head)
also exists.

### MoE execution flavours

Mixture-of-Experts blocks have several execution "flavours" because prefill (many
tokens) and decode (one/few tokens) favor opposite strategies, and quantized
experts need their own path:

- **`decode_bmm`** — decode-optimal: gather only the top-k selected expert
  weights per token and do a batched matmul over just those, then weight and sum.
- **`simple_loop`** — one masked pass per expert over all tokens; a
  simple/robust fallback (also used for quantized experts).
- **`expert_parallel`** — prefill-optimal: pack the tokens routed to each expert
  into contiguous chunks, run the expert MLP on the packed tokens, scatter back,
  and reduce across cards. Maps experts across pipeline stages/devices.
- **`legacy_ffn_blocking`** — an older env-var-gated path that blocks the FFN
  token dimension, superseded by the flavour system.

The default is `expert_parallel` for prefill and `decode_bmm` for decode, with
`simple_loop` as fallback; it's overridable via `qaic_config["moe_config"]`.

*Where:* `QEfficient/transformers/moe/` (`flavours.py`, `qflavours.py`,
`block.py`, `profiles.py`). Support matrix:
[Supported Features](../source/supported_features.rst).

---

## Weight quantization

Two of these compress **weights** (as opposed to the KV-cache compression above):

- **MXFP6 matmul weights (`mxfp6_matmul`)** — a compile-time flag telling the
  compiler to store MatMul weights as 6-bit micro-scaled float, shrinking the
  weight footprint and speeding up matmuls. It changes weight encoding only, not
  graph I/O.
- **Loading pre-quantized checkpoints (AWQ / GPTQ / FP8 / compressed-tensors)** —
  QEfficient's quantizer path loads models already quantized upstream, unpacking
  int4/uint4 weights and inserting blockwise `DequantizeLinear` custom ops so they
  run correctly on device.

*Where:* `QEfficient/base/modeling_qeff.py` (compile flags),
`QEfficient/customop/quantization_ops.py`,
`QEfficient/transformers/quantizers/`.

---

## Quick reference

| Technique | Group | Turn it on with |
| --- | --- | --- |
| Precomputed RoPE, fused RMSNorm, fp16 clamping, boolean masks, last-token slice | Graph/numeric | Automatic (part of the transforms) |
| ONNX sub-functions | Graph/numeric | `use_onnx_subfunctions=True` |
| Static gather/scatter KV cache + on-device retention | Memory/KV | Automatic |
| Compute Context Length (CCL) | Memory/KV | `comp_ctx_lengths` compile arg |
| BlockedKV attention | Memory/KV | `qaic_config["blocking_mode"]` |
| MXINT8 KV cache | Memory/KV | `mxint8_kv_cache=True` |
| Prefix/prefill caching | Memory/KV | `kv_cache_prefix` (with chunking) |
| Chunked prefill | Decode | Automatic |
| Continuous batching | Decode | `full_batch_size=N` |
| On-device sampling | Decode | `qaic_config` sampler options |
| Speculative decoding / DFlash / prompt-lookup | Decode | `num_speculative_tokens`, `qaic_config` |
| MDP (tensor/pipeline parallel) | Distribution | `num_devices`, `mdp_num_partitions` |
| KV-head replication | Distribution | `num_replicate_kv_heads` |
| MoE flavours | Distribution | Automatic; `qaic_config["moe_config"]` |
| MXFP6 weights | Quantization | `mxfp6_matmul=True` |
| AWQ / GPTQ / FP8 loading | Quantization | Automatic for quantized checkpoints |

---

**Where this lives in the code** (top-level pointers)

- Transforms & custom ops: `QEfficient/transformers/models/pytorch_transforms.py`,
  `QEfficient/customop/`
- KV cache & CCL: `QEfficient/transformers/cache_utils.py`,
  `QEfficient/utils/check_ccl_specializations.py`
- Blocking: `QEfficient/blocking/`
- Sampling: `QEfficient/transformers/sampler/`
- MoE: `QEfficient/transformers/moe/`
- Compile / MDP / quantization flags: `QEfficient/base/modeling_qeff.py`,
  `QEfficient/compile/`
- Runtime (retention, chunked prefill, continuous batching):
  `QEfficient/generation/text_generation_inference.py`,
  `QEfficient/generation/cloud_infer.py`

**Back to:** [Understanding QEfficient index](index.md) ·
**See also:** [Key concepts](04-key-concepts.md),
[Features Enablement](../source/features_enablement.md)
