# PEFT Examples

Examples for running Parameter-Efficient Fine-Tuning (PEFT) models with LoRA adapters on Qualcomm Cloud AI 100.


## Authentication

For private/gated models, export your HuggingFace token:
```bash
export HF_TOKEN=<your_huggingface_token>
```

## Supported Models

**QEff Auto Class:** `QEffAutoPeftModelForCausalLM`

PEFT/LoRA adapters work with any supported base model architecture. 

Popular base models include:
- Llama
- Mistral, Mixtral


## Available Examples

### single_adapter.py
Load and use a single LoRA adapter with a base model.

**Usage:**
```python
python single_adapter.py
```

This example:
- Loads Mistral-7B base model with a LoRA adapter
- Demonstrates adapter switching
- Shows inference with different adapters (magicoder, tldr, gsm8k, agnews)

### multi_adapter.py
Use multiple LoRA adapters with continuous batching.

**Usage:**
```python
python multi_adapter.py
```

This example:
- Runs multiple adapters simultaneously in one batch
- Demonstrates continuous batching with `full_batch_size=4`
- Shows different prompts using different adapters in the same batch

## Key Features

### Single Adapter Mode
- Load one LoRA adapter at a time
- Switch between adapters dynamically
- Suitable for single-task inference

### Multi-Adapter Mode (Continuous Batching)
- Run multiple adapters simultaneously
- Different prompts can use different adapters in the same batch
- Efficient for multi-task scenarios
- Requires `continuous_batching=True` and `finite_adapters=True`

## Adapter Management

```python
# Load adapter
qeff_model.load_adapter("predibase/adapter_name", "adapter_name")

# Set active adapter
qeff_model.set_adapter("adapter_name")

# Unload adapter
qeff_model.unload_adapter("adapter_name")
```

## Documentation

- [QEff Auto Classes](https://quic.github.io/efficient-transformers/qeff_autoclasses.html)
- [Validated Base Models](https://quic.github.io/efficient-transformers/validate.html#text-only-language-models)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Quick Start Guide](https://quic.github.io/efficient-transformers/quick_start.html)
