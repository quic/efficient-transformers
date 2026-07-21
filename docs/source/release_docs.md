# Efficient Transformer Library - 1.22.0 Release Notes

Welcome to the official release of **Efficient Transformer Library v1.22.0**! This release delivers major new model support including Gemma 4, Kimi-K2, Qwen3-VL, and GLM4-MOE, a full rebase to Transformers v5.5.4, Python 3.12 support, advanced attention blocking, Multi-head Latent Attention (MLA), and numerous performance and export improvements.

> ✅ All features and models listed below are available on the [`release/v1.22.0`](https://github.com/quic/efficient-transformers/tree/release/v1.22.0) branch and [`mainline`](https://github.com/quic/efficient-transformers/tree/main).

---

## Newly Supported Models

- **Gemma 4**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText) (multimodal)
  - Covers both dense and MoE configurations
  - Optimized chunked prefill and disaggregated (vision/language split) serving
  - [Gemma 4 Example Scripts](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/gemma_vision/gemma4)

- **Qwen3-VL (Vision Language)**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - Dense and MoE variants supported
  - Continuous batching enabled
  - Disaggregated VL-MoE serving (three-QPC split)
  - [Qwen3-VL Example Scripts](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/qwen3vl)

- **Qwen3.5 / Qwen3.6 VLM Series**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - Linear attention variants for Qwen3.5 and Qwen3.6 architectures
  - Continuous batching and disaggregated serving support
  - [Qwen3.5 Example Scripts](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/qwen3_5)

- **GLM4-MOE**
  - Executable via [`QEFFAutoModelForCausalLM`](#QEFFAutoModelForCausalLM)
  - Disaggregated prefill and decode support with chunked prefill
  - KV-blocked attention and ONNX subfunction export
  - [GLM4-MOE Disaggregated Example](https://github.com/quic/efficient-transformers/blob/main/examples/disagg_serving/glm4_moe_disagg_mode_with_chunking.py)



---

## Deprecations

- **Llama 3.2 Vision (mllama)**: Support for `meta-llama/Llama-3.2-11B-Vision` has been deprecated. Users should migrate to supported VLM alternatives.

---

## Key Features & Enhancements

- **Framework Upgrades**
  - Transformers rebased to `v5.5.4` — modeling wrappers, `cache_utils`, and export flows aligned; PyTorch/ORT parity restored
  - Intermediate rebase to `v4.57.3` also landed, unifying subfunction changes
  - Python `3.12` support added (while retaining `3.10` and `3.11` compatibility)


- **Causal Attention Blocking**
  - New KV, Q, Head, and Batch blocking strategies for CausalLM models
  - Supported architectures: Llama3, GPT-OSS, Gemma, Gemma2, Granite, GraniteMOE, Mistral
  - Configurable via `qaic_config` with `enable_blocking: True`; disabled by default
  - Automatic blocking configurator with `skip_kv` enabled by default
  - Optimized nested loop implementation for attention blocking

- **NSP-Blocked Expert Dispatch for MoE Prefill**
  - Batched packed-prefix expert dispatch for Qwen3MOE and GPT-OSS chunked prefill
  - Configurable via `num_cores` and `moe_prefill_packed_chunk_size` in compile API
  - Replaces sequential per-expert loop for improved throughput

- **First-Block Caching for Diffusers (Wan)**
  - Block caching infrastructure for Wan non-unified transformer
  - Enable via `enable_first_block_cache=True` and `first_block_cache_downsample_factor`
  - Supports `Wan-AI/Wan2.2-T2V-A14B-Diffusers` at 720p

- **Named Graph Specializations**
  - `specializations.json` now carries meaningful graph names: `Prefill`, `Decode`, `Vision`, `Encoder`, `Embedding`
  - Automatic name inference based on input key patterns

- **Multiple TLM Decode Specializations**
  - `QEFFAutoModelForCausalLM.compile()` now accepts `num_speculative_tokens` as `List[int]`
  - Compiles multiple decode specializations in one QPC for per-step dispatch to the cheapest kernel
  - Backward compatible: plain `int` input still works

- **CCL Support for Disaggregated Prefill**
  - Compute-Context-Length (CCL) now supported during the prefill phase of disaggregated serving
  - Enables compilation at maximum context length with efficient TTFT for smaller prompt lengths

- **FP16/BF16 Export & Compile for VLMs**
  - Added `fp16`/`bf16` precision export and compile support for Vision Language Models

- **Proxy Model Export**
  - New `enable_proxy=True` flag to export a proxy model (disables Embedding Layer and LM Head)
  - Supported across text, embedding, vision, and audio models
  - Proxy-gated ONNX transforms: `FP16ClipTransform` and `SplitTensorsTransform` now only applied when `enable_proxy=True`

- **On-Device Sampling for Qwen3**
  - Extended on-device sampling support to `Qwen3ForCausalLM`

- **ONNX Subfunction Support for VLMs**
  - `use_onnx_subfunctions=True` now supported for Vision Language Models
  - RoPE indexing hoisted out of decoder subfunctions for cached text models (performance improvement)
  - Subfunction changes for Qwen2.5 models

- **Weight Offloading Improvements**
  - Improved weight offloading to handle plain tensor attributes using `to_empty()`
  - Fixes for large model export OOM risk

- **Large Model ONNX Export**
  - `SplitTensorsTransform` added to `QEFFAutoModel` to prevent >2GB protobuf export failures
  - Improved proxy-only clip/split transform gating

- **Compiler Toolchain Update**
  - Compilation updated from deprecated `qaic-exec` to `qaic-compile`
  - Removed deprecated `-compile-only` compiler flag

- **Disaggregated Serving Enhancements**
  - ONNX reuse fix for disaggregated compile and diffusion modules
  - Qwen3-MoE disaggregated mode support
  - CCL input precision changed from `int8` to `int64` to align with compiler

- **Bug Fixes**
  - Fixed MLA KV head replication for correct head count handling
  - Fixed RoPE dtype for `custom_dtype` export
  - Fixed `fp16`/`bf16` export and compile for Qwen3-VL and Qwen3-VL-MoE models
  - Fixed continuous batching inconsistency for Qwen2.5-VL
  - Fixed Gemma3 sliding window issue
  - Fixed CCL support during disaggregated serving decode phase
  - Fixed `device_ids` type handling for single-device `qaicrt.Program`
  - Clarified CausalLM decode-only flag handling and fixed PL=1 specialization naming
  - Removed `custom_io` passing for BF16 dtype case

---

## Example Scripts

- [Gemma 4 ](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/gemma_vision/gemma4)
- [Qwen3-VL](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/qwen3vl)
- [Qwen3.5 VLM](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/qwen3_5)
- [Qwen3.5-MoE](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/qwen3_5_moe)
- [Qwen3-VL-MoE](https://github.com/quic/efficient-transformers/tree/main/examples/image_text_to_text/models/qwen3_vl_moe)
- [GLM4-MOE Disaggregated](https://github.com/quic/efficient-transformers/blob/main/examples/disagg_serving/glm4_moe_disagg_mode_with_chunking.py)
- [Sequence Classification](https://github.com/quic/efficient-transformers/tree/main/examples/sequence_classification)
- [Wan Image-to-Video](https://github.com/quic/efficient-transformers/tree/main/examples/diffusers/wan)
