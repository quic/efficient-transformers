# Compute Context Length (CCL) Examples

Examples demonstrating Compute Context Length (CCL) optimization for efficient inference on Qualcomm Cloud AI 100.

## What is CCL?

Compute Context Length (CCL) is a performance optimization feature that allows models to use different context lengths during different phases of inference:

- **Prefill Phase**: Processing the initial prompt with optimized context lengths
- **Decode Phase**: Generating new tokens with dynamically adjusted context lengths

This optimization provides:
- **Memory Efficiency**: Uses smaller context lengths when possible
- **Performance Optimization**: Reduces computation for shorter sequences
- **Flexible Scaling**: Adapts context length based on actual sequence position
- **Hardware Optimization**: Optimized for Qualcomm Cloud AI 100 accelerators

## Authentication

For private/gated models, export your HuggingFace token:
```bash
export HF_TOKEN=<your_huggingface_token>
```

## Quick Start

### Text-Only Models

Run basic CCL inference with default settings:
```bash
python basic_inference.py
```

Customize with command-line arguments:
```bash
python basic_inference.py \
    --model-name meta-llama/Llama-3.2-1B \
    --prompt "Hello, how are you?" \
    --ctx-len 1024 \
    --comp-ctx-lengths-prefill "256,500" \
    --comp-ctx-lengths-decode "512,1024" \
    --generation-len 100
```

### Vision-Language Models

Run VLM inference with CCL:
```bash
python vlm_inference.py
```

Customize with command-line arguments:
```bash
python vlm_inference.py \
    --model-name meta-llama/Llama-3.2-11B-Vision-Instruct \
    --query "Describe this image" \
    --image-url "https://..." \
    --comp-ctx-lengths-prefill "4096" \
    --comp-ctx-lengths-decode "6144,8192" \
    --ctx-len 8192
```

## Available Examples

### Text-Only Models

#### basic_inference.py
Basic CCL usage with text-only language models.

**Supported Models:**
- Llama (3.2, 3.3, swiftkv)
- Gemma/Gemma-2
- Mistral
- Phi/Phi-3
- Qwen
- Granite
- GPT-2, GPT-J
- CodeGen
- OLMo-2
- Mistral/Mixtral
- Qwen2
- Falcon

**Command-Line Arguments:**
- `--model-name`: HuggingFace model ID (default: meta-llama/Llama-3.2-1B)
- `--prompt`: Input prompt (default: "My name is ")
- `--ctx-len`: Maximum context length (default: 1024)
- `--comp-ctx-lengths-prefill`: Comma-separated prefill context lengths (default: 256,500)
- `--comp-ctx-lengths-decode`: Comma-separated decode context lengths (default: 512,1024)
- `--generation-len`: Number of tokens to generate (default: 128)
- `--continuous-batching`: Enable continuous batching mode
- `--num-cores`: Number of cores (default: 16)
- `--num-devices`: Number of devices (default: 1)

**Usage Examples:**
```bash
# Basic usage with defaults
python basic_inference.py

# Custom model and prompt
python basic_inference.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --prompt "Explain quantum computing"

# With continuous batching
python basic_inference.py \
    --continuous-batching \
    --full-batch-size 4

# Larger context with progressive CCL
python basic_inference.py \
    --ctx-len 4096 \
    --comp-ctx-lengths-prefill "1024,2048" \
    --comp-ctx-lengths-decode "2048,3072,4096"
```

**Python API:**
```python
from transformers import AutoTokenizer
from QEfficient import QEFFAutoModelForCausalLM

model = QEFFAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    qaic_config={
        "comp_ctx_lengths_prefill": [256, 500],
        "comp_ctx_lengths_decode": [512, 1024],
        "ctx_len": 1024,  # Required for CCL validation
    },
)
```

#### gpt_oss.py
CCL for GPT-OSS MoE models with prefill_seq_len=1 optimization.

**Usage:**
```bash
python gpt_oss.py
```

**Note:** For MoE models, both prefill and decode CCL lists can be similar when using prefill_seq_len=1.

### Vision-Language Models

#### vlm_inference.py
General VLM inference with CCL optimization.

**Usage:**
```bash
python vlm_inference.py
```

#### gemma3.py
CCL for Gemma-3 multimodal models (4B/27B).

**Usage:**
```bash
python gemma3.py
```

#### granite_vision.py
CCL for IBM Granite Vision models.

**Usage:**
```bash
python granite_vision.py
```

#### internvl.py
CCL for InternVL2.5 models with custom processor.

**Usage:**
```bash
python internvl.py
```

#### llama4.py
CCL for Llama-4 Scout vision-language models.

**Usage:**
```bash
python llama4.py
```

#### llama4_cb.py
CCL for Llama-4 with continuous batching.

**Usage:**
```bash
python llama4_cb.py
```

#### llama4_multi_image.py
CCL for Llama-4 with multiple images.

**Usage:**
```bash
python llama4_multi_image.py
```

#### mistral3.py
CCL for Mistral-Small-3.1 vision models.

**Usage:**
```bash
python mistral3.py
```

#### molmo.py
CCL for Molmo-7B multimodal models.

**Usage:**
```bash
python molmo.py
```

#### qwen2_5_vl.py
CCL for Qwen2.5-VL models (32B).

**Usage:**
```bash
python qwen2_5_vl.py
```

#### qwen2_5_vl_cb.py
CCL for Qwen2.5-VL with continuous batching.

**Usage:**
```bash
python qwen2_5_vl_cb.py
```

## Configuration Guidelines

### Choosing CCL Values

1. **Prefill Context Lengths** (`comp_ctx_lengths_prefill`):
   - Start with smaller values (e.g., [256, 512, 1024])
   - Should be less than or equal to your prefill_seq_len
   - Gradually increase based on prompt chunk position

2. **Decode Context Lengths** (`comp_ctx_lengths_decode`):
   - Start from a value based on expected prompt length
   - Include intermediate steps (e.g., [512, 1024, 2048, ctx_len])
   - Final value should match ctx_len

3. **Context Length** (`ctx_len`):
   - Maximum context length for the model
   - Required parameter for CCL validation
   - Should match your model's maximum supported length

### Example Configurations

**Small Context (1K-2K):**
```python
ctx_len = 2048
comp_ctx_lengths_prefill = [256, 512]
comp_ctx_lengths_decode = [1024, ctx_len]
```

**Medium Context (4K-8K):**
```python
ctx_len = 8192
comp_ctx_lengths_prefill = [3072, 4096]
comp_ctx_lengths_decode = [4096, 6144, ctx_len]
```

**Large Context (16K+):**
```python
ctx_len = 16384
comp_ctx_lengths_prefill = [4096, 8192]
comp_ctx_lengths_decode = [8192, 12288, ctx_len]
```

## Performance Tips

1. **Memory Optimization**: Use smaller CCL values for prefill to reduce memory footprint
2. **Progressive Scaling**: Include intermediate CCL values in decode list for smooth transitions
3. **Vision Models**: Larger prefill contexts needed for image embeddings
4. **Continuous Batching**: CCL works seamlessly with CB for dynamic workloads
5. **MoE Models**: Consider prefill_seq_len=1 for optimal performance

## Common Patterns

### Text-Only Model
```python
model = QEFFAutoModelForCausalLM.from_pretrained(
    model_name,
    qaic_config={
        "comp_ctx_lengths_prefill": [256, 500],
        "comp_ctx_lengths_decode": [512, 1024],
        "ctx_len": 1024,
    },
)
```

### Vision-Language Model
```python
model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_name,
    kv_offload=True,
    qaic_config={
        "comp_ctx_lengths_prefill": [3072],
        "comp_ctx_lengths_decode": [4096, 8192],
        "ctx_len": 8192,
    },
)
```

### Continuous Batching
```python
model = QEFFAutoModelForCausalLM.from_pretrained(
    model_name,
    continuous_batching=True,
    qaic_config={
        "comp_ctx_lengths_prefill": [256, 500],
        "comp_ctx_lengths_decode": [512, 1024],
        "ctx_len": 1024,
    },
)
```

## Documentation

- [QEff Auto Classes](https://quic.github.io/efficient-transformers/source/qeff_autoclasses.html)
- [Performance Features](https://quic.github.io/efficient-transformers/source/features_enablement.html)
- [Quick Start Guide](https://quic.github.io/efficient-transformers/source/quick_start.html)
