# MiniMax-M3 Decode-Only Inference on AI200 Servers

This guide provides instructions for running the MiniMax-M3 VLM (Vision-Language Model) in decode-only mode on Qualcomm Cloud AI 200 servers with replicate KV-head support.

## Overview

The `minimax_m3_decode_only.py` script demonstrates:
- MiniMax-M3 VLM decode-only inference (PL=1)
- API layerwise compilation
- Replicate KV-head optimization for AI200
- MXFP6 matmul and MXINT8 KV cache support

## Prerequisites

- Access to AI200 servers
- GitHub CLI (`gh`) installed and authenticated
- Python environment with QEfficient installed

## Setup Instructions

### 1. Checkout the PR with Replicate KV-Head Changes

This PR (1127) contains the replicate KV-head changes required for optimal performance:

```bash
gh pr checkout 1127
```

### 2. Upgrade Transformers

Ensure you have the latest version of transformers:

```bash
pip install transformers --upgrade
```

### 3. Run the Script

#### Basic Usage (Default Configuration)

```bash
python examples/text_generation/minimax_m3_decode_only.py
```

This will run with default settings:
- Model: `MiniMaxAI/MiniMax-M3`
- Context length: 1024
- Number of devices: 24
- Number of cores: 4
- Generation length: 32
- Prompt: "Tell me about yourself."
- **Replicate KV-heads: 8** (optimized for AI200)

#### Advanced Usage with Custom Parameters

```bash
python examples/text_generation/minimax_m3_decode_only.py \
    --model-id MiniMaxAI/MiniMax-M3 \
    --ctx-len 2048 \
    --num-devices 24 \
    --num-cores 4 \
    --generation-len 64 \
    --prompt "Describe the architecture of transformers." \
    --layerwise-window-size 1
```

#### Compile Only (Skip Generation)

To compile the model without running inference:

```bash
python examples/text_generation/minimax_m3_decode_only.py --skip-generate
```

## Configuration Details

### Replicate KV-Head Feature

The script uses `qaic_config={"num_replicate_kv_heads": 8}` in two places:

1. **Model initialization** (`QEFFAutoModelForImageTextToText.from_pretrained`):
   - Configures the model to use 8 replicate KV-heads

2. **Compilation** (`qeff_model.compile`):
   - Ensures the compiled QPC uses the same KV-head replication

This optimization improves performance on AI200 hardware by replicating key-value heads across multiple processing units.

### Compilation Settings

The script compiles with the following AI200-optimized settings:
- `batch_size=1`: Single batch inference
- `prefill_seq_len=1`: Decode-only mode (no prefill)
- `mxfp6_matmul=True`: Mixed-precision FP6 matrix multiplication
- `mxint8_kv_cache=True`: INT8 KV cache for memory efficiency
- `aic_hw_version='ai200'`: Target AI200 hardware
- `skip_vision=True`: Skip vision encoder (text-only generation)
- `kv_offload=True`: Offload KV cache to optimize memory

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-id` | str | `MiniMaxAI/MiniMax-M3` | HuggingFace model ID |
| `--ctx-len` | int | 1024 | Context length for generation |
| `--num-devices` | int | 24 | Number of AI200 devices to use |
| `--num-cores` | int | 4 | Number of cores per device |
| `--generation-len` | int | 32 | Number of tokens to generate |
| `--prompt` | str | "Tell me about yourself." | Input prompt for generation |
| `--layerwise-window-size` | int | 1 | Window size for layerwise compilation |
| `--num-layers` | int | None | Override number of model layers (for testing) |
| `--skip-generate` | flag | False | Compile only, skip inference |

## Expected Output

The script will:
1. Load the MiniMax-M3 model configuration
2. Initialize the model with replicate KV-heads
3. Compile the model to QPC (Qualcomm Program Container)
4. Print QPC paths
5. Run inference with the provided prompt
6. Output:
   - Generated text object
   - Generated token IDs
   - Decoded text response

Example output:
```
QPC paths: {'lang_decode_qpc_path': '/path/to/qpc'}
<GenerationOutput object>
[Performance Numbers]
[token_ids...]
['Generated response text...']
```

