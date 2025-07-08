# 🚀 Efficient Transformer Library - Release 1.20.0 (Beta)

Welcome to the official release of **Efficient Transformer Library v1.20.0**! This release brings a host of new model integrations, performance enhancements, and fine-tuning capabilities to accelerate your AI development.

> ✅ All features and models listed below are available on the `release/1.20.0` branch and `mainline`.

---

## 🧠 Newly Supported Models

- **Llama-4-Scout-17B-16E-Instruct**
  - Text & Image+Text support
  - Chunk attention, Single/Dual QPC support
  - Multi-image prompts enabled via VLLM interface

- **Grok-1**
  - Executable via `QEffAutoModelForCausalLM`

- **Gemma3**
  - Text & Image+Text support
  - Sliding window support

- **Granite Vision & MOE Models**
  - Sample scripts available

- **SwiftKV (Llama-3.1-SwiftKV-8B-Instruct)**
  - Supports both continuous and non-continuous batching

- **GGUF Models**
  - Execution support (non-quantized)

- **FP8 Compressed Quantization**
  - Support for `Llama-3.3-70B-Instruct-FP8-Dynamic`

---

## ✨ Key Features & Enhancements

- **Transformer Upgrade**: Now using version `4.51.3`
- **SpD & Multi-Projection Heads**: Token speculation via post-attention projections
- **I/O Encryption**: `--io-encrypt` flag support in compile/infer APIs
- **Separate Prefill/Decode Compilation**: For disaggregated serving
- **On-Device Sampling**: Reduces host-device latency for CausalLM models

---

## 🔍 Embedding Model Upgrades

- **Flexible Pooling**: Choose from standard or custom strategies
- **Sentence Embedding**: Now runs directly on AI100
- **Multi-Seq Length Compilation**: Auto-selects optimal graph at runtime

---

## 🛠️ Fine-Tuning Support

- BERT fine-tuning support with templates and documentation
- Gradient checkpointing, device-aware `GradScaler`, and CLI `--help` added

---
Thank you for using Efficient Transformers! For more details, refer to the full documentation.