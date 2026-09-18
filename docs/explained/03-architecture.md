# Architecture

This is the core document. It walks the full QEfficient pipeline from a Hugging
Face model to running tokens on a Qualcomm accelerator, one stage at a time.

The pipeline has four stages, driven by the auto-model API you already saw:

```
                      from_pretrained()          compile()                        generate()
                            │                        │                                 │
  HF model card  ──▶  ┌───────────┐  ──────▶  ┌────────────┐  ──────▶  ┌──────────┐  ─────▶  tokens
   or local path      │  Stage 0  │           │  Stage 1   │           │ Stage 2  │          │
                      │  Load &   │           │  Export to │           │ Compile  │          │
                      │ transform │           │   ONNX     │           │ to QPC   │          │
                      └───────────┘           └────────────┘           └──────────┘          │
                       PyTorch nn.Module        model.onnx              programqpc.bin        │
                                                                                        ┌──────────┐
                                                                                        │ Stage 3  │
                                                                                        │  Run on  │
                                                                                        │  device  │
                                                                                        └──────────┘
```

> Note: `export()` runs implicitly as part of `compile()`. You normally call
> `from_pretrained` → `compile` → `generate`; the ONNX export happens inside
> compile when needed. (Calling `.export()` directly is deprecated.)

Everything hangs off a base class, `QEFFBaseModel`
(`QEfficient/base/modeling_qeff.py`), which every task-specific wrapper
(`QEFFAutoModelForCausalLM`, etc.) subclasses. It holds the PyTorch module and
the ordered transform lists, and it owns the export/compile/cache lifecycle.

---

## Stage 0 — Load & transform (`from_pretrained`)

QEfficient loads the model through the standard HF auto class, with two
deliberate settings: `attn_implementation="eager"` (so attention is plain
PyTorch that exports cleanly, not a fused kernel) and `use_cache=True` (so the
model produces KV outputs that can be retained on device).

Then it applies an **ordered list of transforms** — `self._pytorch_transforms`,
declared per task class — to rewrite the in-memory `nn.Module` into an
accelerator-friendly form. The base runner loops the list and applies each in
turn.

There are **three transform mechanisms** (base classes in
`QEfficient/base/pytorch_transforms.py`):

1. **`ModuleMappingTransform` — in-place class reassignment.**
   The key trick: it walks the module tree and, for any module whose type is in
   a `{HFClass: QEffClass}` mapping, does `module.__class__ = QEffClass`. Only
   the Python class *pointer* is swapped — all weights, buffers, and state are
   preserved. The QEff subclass simply supplies a new `forward`. If the
   replacement defines `__qeff_init__`, it's called right after the swap to build
   extra state (e.g. precomputed RoPE buffers). This is the primary mechanism and
   the reason divergence from HF stays tiny.

2. **`ExternalModuleMapperTransform` — method rebinding.**
   Used for "remote code" models whose classes aren't importable when the
   registry is built. Instead of swapping the class, it rebinds individual
   methods onto the instance (keyed by class object or by class-name string).

3. **`ModuleMutatorTransform` — build/rewrite a module.**
   For transforms that must actually rewrite weights, not just re-point a class —
   for example `ReplicateKVHeadTransform`, which duplicates K/V projection
   weights to expand GQA/MQA KV heads.

The main registries live in
`QEfficient/transformers/models/pytorch_transforms.py`:

- **`CustomOpsTransform`** — swaps each family's RMSNorm/norm for a fused QAIC
  custom-op version.
- **`KVCacheTransform`** — the large registry (~40 model families) that swaps
  attention, decoder-layer, model, and rotary-embedding classes for their QEff
  reimplementations. This is what installs the static-cache attention and
  precomputed RoPE.
- **`KVCacheExternalModuleMapperTransform`** — the method-rebinding equivalent
  for remote-code models (InternVL, Molmo, DeepseekV3, …).
- Quantization mappers (AWQ/GPTQ/int4) run first; speculative-decoding and
  sampler transforms are applied conditionally based on `qaic_config`.

Some transforms are **compile-parameter-dependent** (they need to know things
like device count or blocking config) and run later, at export time, rather than
at load: KV-head replication, attention blocking, and MoE optimization.

**Result of Stage 0:** the same weights, running through QEff `forward` methods
that use static shapes, precomputed RoPE, fused norms, and a static KV cache.

---

## Stage 1 — Export to ONNX (`export`)

ONNX is the interchange format between PyTorch and the Qualcomm compiler. This
stage traces the transformed model into an ONNX graph with a **precise, stable
I/O contract** that the rest of the pipeline depends on.

**Inputs** the graph declares:

- `input_ids` — shape `(batch, seq_len)`, int64.
- `position_ids` — shape `(batch, seq_len)`, int64. (Explicit positions, because
  RoPE is a static lookup indexed by position.)
- Per-layer past cache, expanded from `past_key_values` into named tensors:
  `past_key.{i}` / `past_value.{i}` for each layer `i` (4-D:
  `[batch, kv_heads, ctx_len, head_dim]`). Encoder-decoder models use
  `past_{key,value}_{self,cross}.{i}`; MLA models (e.g. DeepSeek) use
  `compressed_kv.{i}` / `k_pe.{i}`.
- Optional inputs enabled by features: `batch_index` (continuous batching),
  `num_logits_to_keep` (speculative decoding / TLM), `comp_ctx_lengths`
  (compute-context-length), sampler tensors, `lora_ids`.

**Outputs** the graph declares:

- `logits` — `(batch, seq_len, vocab)`. (Or `probs`/`next_tokens` when on-device
  sampling is enabled.)
- One `past_key.{i}_RetainedState` / `past_value.{i}_RetainedState` per layer.
  **These `_RetainedState` outputs are the KV tensors the device will persist
  between steps** — remember this name; it's the linchpin of Stage 3.

**Dynamic axes** mark which dimensions vary: batch (dim 0), sequence length
(dim 1 of the ids), and context length (the cache dim). These become the
"symbols" the compiler specializes on in Stage 2.

The exporter supports three backends — a default TorchScript `torch.onnx.export`
path, a newer `dynamo` path, and a `weight_free` path (traces with meta tensors
and keeps weights outside the ONNX graph via a `weight_spec.json` sidecar, for
very large models). After export, a small set of **ONNX transforms** run on the
graph — most importantly fp16 clipping (clamp constants into fp16 range) and
tensor splitting (externalize large weight tensors).

**Caching:** the export directory is named with a **hash** (SHA-256 of the model
config diff, the applied transforms, the I/O names, dynamic axes, and export
flags, truncated to 16 hex chars). If the `.onnx` already exists for that hash,
export is skipped entirely.

**Result of Stage 1:** `QEFF_HOME/<arch>/<model>-<export_hash>/<model>.onnx`
(plus any external weight data and metadata sidecars).

---

## Stage 2 — Compile to QPC (`compile`)

This stage turns the ONNX graph into a **QPC** — the compiled binary that runs on
the card — by invoking the Qualcomm compiler
(`/opt/qti-aic/exec/qaic-compile`).

The important compile-time concepts:

- **Specializations.** The compiler builds specialized graphs for fixed shape
  combinations. QEfficient generates (at least) a **Prefill** specialization
  (process the whole prompt) and a **Decode** specialization (process one new
  token), each a dict of `batch_size` / `seq_len` / `ctx_len` (plus
  `full_batch_size` for continuous batching, extra decode specs for speculative
  decoding, etc.). These are written to `specializations.json` and passed to the
  compiler; the runtime reads the same file back to recover its shapes.

- **`custom_io` — KV precision map.** Each `past_*.{i}_RetainedState` tensor is
  mapped to its on-device dtype (the model dtype, or `mxint8` when MXINT8
  KV-cache compression is on). Written to `custom_io.yaml` and passed to the
  compiler. This is *how the KV cache precision gets set on the device.*

- **Compiler flags.** Options become `-flag=value` arguments — e.g.
  `convert_to_fp16=True` → `-convert-to-fp16`, `aic_num_cores=16` →
  `-aic-num-cores=16`, `mxfp6_matmul=True` → MXFP6 weight compression,
  `use_onnx_subfunctions` → `-sub-functions`.

- **MDP (Multi-Device Partitioning).** For multi-card execution, QEfficient
  generates a partition config — either a **tensor-slice** template (split each
  layer across devices) or a **pipeline-parallel / disaggregated** config
  (assign layer ranges to devices) — passed via `-mdp-load-partition-config`.
  In the public API you pass `num_devices` and `mdp_num_partitions`; the
  tensor-slice count per partition is derived internally.

- **QNN path.** If `enable_qnn=True`, compilation is delegated to the Qualcomm
  QNN SDK backend instead of the default `qaic-compile` path.

**Caching:** a **compile hash** is computed over the full compiler command,
specializations, custom-IO, MDP config, and speculative-token / prefill-only
flags. If `qpc-<compile_hash>/qpc/programqpc.bin` already exists, compilation is
skipped.

**Result of Stage 2:** a QPC package directory
`.../qpc-<compile_hash>/qpc/programqpc.bin`, alongside `specializations.json`,
`custom_io.yaml`, the MDP config, and metadata sidecars (`hashed_compile_params.json`,
`qconfig.json`).

---

## Stage 3 — Run on device (`generate`)

`generate` loads the QPC into a runtime session and drives the prefill/decode
loop.

- **`QAICInferenceSession`** (`QEfficient/generation/cloud_infer.py`) is the
  low-level wrapper over the Qualcomm runtime. It loads the QPC, reads the I/O
  descriptor into a name→buffer map, allocates a device buffer per binding, and
  exposes a `run(inputs)` call that copies numpy inputs in, executes, and reads
  outputs back.

- **The retained-state trick.** This is the heart of on-device efficiency. The
  KV cache tensors (`*_RetainedState`) were marked *retained* at compile time.
  At runtime the session **skips** those buffers — it does not read them back to
  the host or write them from the host. As a result the KV cache **stays
  resident on the card** and simply persists across successive `run()` calls.
  Prefill and decode only exchange `input_ids` / `position_ids` / `logits`; the
  (large) cache never crosses the host boundary. This is what makes decode fast.

- **Prefill → decode.** `run_prefill` tokenizes and pads the prompt to a
  multiple of the prefill sequence length and runs it in chunks, producing the
  first token and populating the on-device cache. `run_decode` then loops: run
  one token, take the argmax of `logits` (or sampler output), advance
  `position_ids`, store the token, and stop when all sequences hit EOS.

- **Continuous batching** reuses decode "slots": when one sequence finishes, the
  next queued prompt is prefilled into that slot and `batch_index` selects which
  on-device cache line it uses — keeping the card busy without waiting for the
  slowest sequence.

**Result of Stage 3:** generated token ids/text plus performance metrics
(prefill time, decode throughput, total time).

---

## On-disk cache layout (summary)

Both expensive stages are content-addressed by hash, so re-running with the same
configuration is nearly free:

```
$QEFF_HOME/
└── <model_architecture>/
    └── <model_name>-<export_hash>/          ← Stage 1 output
        ├── <model_name>.onnx
        ├── (external weight data, weight_spec.json)
        ├── hashed_export_params.json
        └── qpc-<compile_hash>/              ← Stage 2 output
            ├── qpc/programqpc.bin
            ├── specializations.json
            ├── custom_io.yaml
            ├── hashed_compile_params.json
            └── qconfig.json
```

`QEFF_HOME` defaults to `~/.cache/qeff_cache` (overridable via `QEFF_HOME` or
`XDG_CACHE_HOME` — see [Quick Start](../source/quick_start.md)).

---

**Where this lives in the code**

- Lifecycle base class: `QEfficient/base/modeling_qeff.py`
  (`QEFFBaseModel`, `_export`, `_compile`, `transform`, `get_onnx_path`)
- Auto-model glue (I/O contract, specializations): `QEfficient/transformers/models/modeling_auto.py`
- Transform base classes: `QEfficient/base/pytorch_transforms.py`
- Transform registries: `QEfficient/transformers/models/pytorch_transforms.py`
- Export backends: `QEfficient/exporter/`, `QEfficient/utils/export_utils.py`
- Compile helpers / MDP / QNN: `QEfficient/compile/`
- Runtime: `QEfficient/generation/cloud_infer.py`,
  `QEfficient/generation/text_generation_inference.py`

**Next:** [Key concepts](04-key-concepts.md) — the two ideas that make the
transformed model work.
