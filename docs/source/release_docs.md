# Efficient Transformer Library - 1.21.0 Release Notes

Welcome to the official release of **Efficient Transformer Library v1.21.0**! This release introduces advanced attention mechanisms, expanded model support, optimized serving capabilities, and significant improvements to fine-tuning and deployment workflows.

> ✅ All features and models listed below are available on the [`release/1.21.0`](https://github.com/quic/efficient-transformers/tree/release/v1.21.0) branch and [`mainline`](https://github.com/quic/efficient-transformers/tree/main).

---

## Newly Supported Models

- **Flux (Diffusers - Image Generation)**
  - Diffusion-based image generation model
  - Blocking support for optimized attention patterns
  - Full integration with diffusers pipeline
  - [Flux Example Scripts](https://github.com/quic/efficient-transformers/blob/main/examples/diffusers/flux/)

- **WAN (Diffusers - Video Generation)**
  - Wide-Area Network Lightning support for distributed inference
  - Enhanced serving capabilities for large-scale deployments
  - Full diffusers integration
  - [WAN Example Scripts](https://github.com/quic/efficient-transformers/blob/main/examples/diffusers/wan/)

- **Qwen2.5-VL (Vision Language)**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - Multi-image prompt support
  - Continuous batching enabled
  - [Qwen2.5-VL Example Script](https://github.com/quic/efficient-transformers/blob/main/examples/image_text_to_text/qwen2_5_vl/)


- **GPT-OSS (Decode-Only)**
  - Executable via [`QEffAutoModelForCausalLM`](#QEffAutoModelForCausalLM)
  - Separate prefill and decode compilation supported
  - Disaggregated serving ready
  - [GPT-OSS Example Scripts](https://github.com/quic/efficient-transformers/blob/main/examples/disagg_serving/)


- **Molmo**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - Multi-modal capabilities

- **InternVL 3.5 Series**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - Full Vision-Language support
  - Multi-image handling with continuous batching

- **Qwen3-MOE (Mixture of Experts)**
  - Executable via [`QEffAutoModelForCausalLM`](#QEffAutoModelForCausalLM)
  - Efficient expert routing

- **Wav2Vec2 (Audio)**
  - Speech recognition and audio feature extraction
  - On-device audio processing support

---

## Key Features & Enhancements

- **Framework Upgrades**: Transformers `4.55`, PyTorch `2.7.0+cpu`, Torchvision `0.22.0+cpu`
- **Python Support**: Now requires Python `>=3.9`
- **ONNX Opset**: Updated to version `17` for broader operator support
- **Advanced Attention**: Flux blocking support, BlockedKV attention for CausalLM models
- **Diffusers Integration**: Full support for diffuser-based image generation models
- **WAN Lightning Support**: Enhanced distributed serving across wide-area networks
- **Automatic CCL Generation**: Automatic Compute-Context-Length list generation for prefill/decode phases
- **Prefill/Decode Separation**: Independent compilation and optimization for disaggregated serving
- **Continuous Batching (VLMs)**: Extended to Vision Language Models with multi-image handling
- **ONNX Sub-Functions**: New ONNX sub-function export feature for AutoModelForCausalLM
- **Guided Decoding**: Constrained generation support for on-device sampling
- **Memory Profiling**: Built-in utilities for optimization analysis
- **KV Cache Optimization**: Improved cache handling for long sequences
- **Checkpoint Resume**: Resume fine-tuning from specific epochs with corrected loss tracking

---

## Embedding Model Upgrades

- **Multi-Sequence Length Support**: Auto-selects optimal graph at runtime
- **Enhanced Pooling**: Flexible pooling strategies for various embedding tasks
- **On-Device Execution**: Improved on-device inference for embedding models

---

## Fine-Tuning Support

- **Checkpoint Management**: Resume from epochs with proper state restoration
- **Enhanced Loss Tracking**: Corrected data type handling for accurate loss computation
- **Custom Dataset Support**: Improved handling with better tokenization
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Device-Aware Scaling**: Optimized GradScaler for multi-device training
- **Comprehensive Testing**: Unit tests for fine-tuning workflows

---


# Efficient Transformer Library - 1.20.0 Release Notes

Welcome to the official release of **Efficient Transformer Library v1.20.0**! This release introduces advanced attention mechanisms, expanded model support, optimized serving capabilities, and significant improvements to fine-tuning and deployment workflows.

> ✅ All features and models listed below are available on the [`release/1.20.0`](https://github.com/quic/efficient-transformers/tree/release/v1.20.0) branch and [`mainline`](https://github.com/quic/efficient-transformers/tree/main).

---


## Newly Supported Models

- **Llama-4-Scout-17B-16E-Instruct**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - Text & Image+Text support
  - Chunk attention, Single/Dual QPC support
  - Multi-image prompts enabled via VLLM interface
  - [Llama4 Example Script](https://github.com/quic/efficient-transformers/blob/main/examples/image_text_to_text/models/llama_vision/single_image.py)

- **Grok-1**
  - Executable via [`QEffAutoModelForCausalLM`](#QEffAutoModelForCausalLM)

- **Gemma3**
  - Executable via [`QEFFAutoModelForImageTextToText`](#QEFFAutoModelForImageTextToText)
  - Text & Image+Text support
  - Sliding window support
  - [Gemma3 Example Script](https://github.com/quic/efficient-transformers/blob/main/examples/image_text_to_text/models/gemma_vision/inference.py)


- **SwiftKV (Llama-3.1-SwiftKV-8B-Instruct)**
  - Executable via [`QEffAutoModelForCausalLM`](#QEffAutoModelForCausalLM)
  - Supports both continuous and non-continuous batching

- **GGUF Models**
  - Executable via [`QEffAutoModelForCausalLM`](#QEffAutoModelForCausalLM)
  - Execution support (non-quantized)
  - [Example Script](https://github.com/quic/efficient-transformers/blob/main/examples/text_generation/gguf_models.py)

- **FP8 Compressed Quantization**
  - Support for [`Llama-3.3-70B-Instruct-FP8-Dynamic`](https://huggingface.co/Infermatic/Llama-3.3-70B-Instruct-FP8-Dynamic)

---

## Key Features & Enhancements

- **Transformer Upgrade**: Now using version `4.51.3`
- **SpD & Multi-Projection Heads**: Token speculation via post-attention projections
- **I/O Encryption**: `--io-encrypt` flag support in compile/infer APIs
- **Separate Prefill/Decode Compilation**: For disaggregated serving
- **On-Device Sampling**: Supported using VLLM, which reduces host-device latency for CausalLM models

---

## Embedding Model Upgrades

- **Flexible Pooling**: Choose from standard or custom strategies
- **Sentence Embedding**: Now runs directly on AI100
- **Multi-Seq Length Compilation**: Auto-selects optimal graph at runtime

---

## Fine-Tuning Support

- BERT fine-tuning support with templates and documentation
- Gradient checkpointing, device-aware `GradScaler`, and CLI `--help` added
