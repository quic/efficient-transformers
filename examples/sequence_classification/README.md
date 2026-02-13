# Sequence Classification Examples

This directory contains examples demonstrating how to use QEfficient for sequence classification tasks on Cloud AI 100 hardware.

## Overview

Sequence classification models are used to classify text inputs into predefined categories. Common use cases include:
- Sentiment analysis
- Spam detection
- Prompt injection detection
- Content moderation

## Supported Models

QEfficient supports sequence classification models through the `QEFFAutoModelForSequenceClassification` class. Currently validated models include:

- **meta-llama/Llama-Prompt-Guard-2-22M**: A DeBERTa-v2 based model for detecting malicious prompts

## Examples

### Basic Inference (`basic_inference.py`)

Demonstrates the complete workflow for running sequence classification on Cloud AI 100:

1. Load a pre-trained model and tokenizer
2. Prepare input text
3. Compile the model for Cloud AI 100
4. Run inference and get predictions

**Usage:**
```bash
python basic_inference.py
```

**Key Features:**
- Simple end-to-end example
- Supports multiple sequence lengths for compilation
- Demonstrates how to interpret classification results

## Quick Start

```python
from transformers import AutoTokenizer
from QEfficient import QEFFAutoModelForSequenceClassification

# Load model and tokenizer
model_id = "meta-llama/Llama-Prompt-Guard-2-22M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = QEFFAutoModelForSequenceClassification.from_pretrained(model_id)

# Prepare input
text = "Your text here"
inputs = tokenizer(text, return_tensors="pt")

# Compile for Cloud AI 100
model.compile(num_cores=16, seq_len=32)

# Run inference
output = model.generate(inputs)
predicted_class = output["logits"].argmax().item()
print(f"Predicted class: {model.model.config.id2label[predicted_class]}")
```

## Compilation Options

The `compile()` method supports various options:

- `num_cores`: Number of cores to use (default: 16)
- `seq_len`: Sequence length(s) for compilation. Can be:
  - Single integer: `seq_len=32`
  - List of integers for multiple specializations: `seq_len=[16, 32, 64, 128]`
- `batch_size`: Batch size (default: 1)
- `num_devices`: Number of devices (default: 1)
- `mxfp6_matmul`: Enable MXFP6 compression (default: False)

## Performance Tips

1. **Multiple Sequence Lengths**: Compile with multiple sequence lengths to handle variable input sizes efficiently
2. **Batch Processing**: For processing multiple inputs, use appropriate batch sizes
3. **Core Allocation**: Adjust `num_cores` based on your Cloud AI 100 SKU

## Additional Resources

- [QEfficient Documentation](https://quic.github.io/efficient-transformers/)
- [Validated Models](../../docs/source/validate.md)
- [API Reference](../../docs/source/qeff_autoclasses.md)
