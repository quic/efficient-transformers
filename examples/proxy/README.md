# Proxy Models Examples

## Overview

This directory contains examples demonstrating how to enable and use **proxy models** in QEfficient. Proxy models replace specific layers (embeddings and LM heads) with dummy layers, enabling efficient model export and IO file generation for downstream optimization and validation.

## What is a Proxy Model?

A proxy model is a modified version of a transformer model where:
- **Embedding layers** are replaced with proxy stubs that transform token IDs into embeddings
- **Language model (LM) head layers** are replaced with proxy implementations that convert hidden states to logits

### Benefits
- **Simplified model export**: Easier to export models for compilation and deployment
- **IO file generation**: Automatically save input/output tensors for validation and debugging


## Enabling Proxy Mode

To enable proxy models, use the `enable_proxy=True` parameter when loading a model:

```python
from QEfficient import QEFFAutoModelForCausalLM

model = QEFFAutoModelForCausalLM.from_pretrained(
    model_name, 
    enable_proxy=True
)
```

### Saving Input/Output Files

Generate IO files during inference using `write_io=True`:

```python
model.generate(
    inputs=...,
    write_io=True  # Saves input/output tensors to .npy files
)
```

## Example Files

### 1. **text_model.py** - Text Generation (Causal Language Models)
Demonstrates proxy model usage with GPT2 for text generation.

**Key Features:**
- Loads a causal language model with proxy enabled
- Compiles the model for inference
- Generates text with IO file output

**Usage:**
```bash
python text_model.py
```

**Model:** `openai-community/gpt2`

---

### 2. **embedding_model.py** - Text Embeddings
Shows how to enable proxy mode for embedding models that extract sentence/text embeddings.

**Key Features:**
- Loads an embedding model with proxy enabled
- Supports pooling strategies (mean, CLS, etc.)
- Generates embeddings with IO file output

**Usage:**
```bash
python embedding_model.py
```

**Model:** `BAAI/bge-base-en-v1.5`

---

### 3. **audio_model.py** - Audio Processing
Demonstrates proxy models for two popular audio model types:

#### a) Speech-to-Seq2Seq (Whisper)
- Transcribes audio to text using encoder-decoder architecture
- Model: `openai/whisper-tiny`

#### b) CTC (Connectionist Temporal Classification) - Wav2Vec2
- Direct audio-to-text transcription
- Model: `facebook/wav2vec2-base`

**Key Features:**
- Processes audio samples with automatic feature extraction
- Supports both Seq2Seq and CTC-based models
- Generates IO files for validation

**Usage:**
```bash
python audio_model.py
```

---

### 4. **image_model.py** - Vision-Language Models (Multimodal)
Demonstrates proxy models for advanced vision-language models with three different execution flows.

#### Supported Model Types:

1. **Standard VLM** (LLaVA, Gemma3, Granite Vision)
   - Standard image-to-text architecture
   - Model: `llava-hf/llava-1.5-7b-hf`

2. **InternVL**
   - Advanced vision-language model with custom architecture
   - Model: `OpenGVLab/InternVL2_5-1B`

3. **Molmo**
   - Open-source multimodal model
   - Model: `allenai/Molmo-7B-D-0924`

**Key Features:**
- Handles image and text inputs
- Supports multiple VLM architectures with different preprocessing pipelines
- Generates captions/descriptions with IO file output
- KV cache offloading support (`kv_offload=True`)

**Usage:**
```bash
python image_model.py
```

---

## Generated IO Files

When `write_io=True`, the model generates files in the qeff models directory:
- `*.npy` files: NumPy arrays containing input/output tensors
- File names indicate tensor type and layer depth
- **Use case**: Validate model outputs, compare with baseline implementations, debug inference issues






---

## References

- [QEfficient Documentation](https://quic.github.io/efficient-transformers/index.html)
- [Model Hub](https://huggingface.co/models)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

