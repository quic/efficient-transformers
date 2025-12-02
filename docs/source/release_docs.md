# Efficient Transformer Library - 1.20.0 Release Notes

Welcome to the official release of **Efficient Transformer Library v1.20.0**! This release brings a host of new model integrations, performance enhancements, and fine-tuning capabilities to accelerate your AI development.

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
