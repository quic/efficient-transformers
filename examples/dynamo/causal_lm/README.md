# Dynamo Causal LM Examples

Examples for exporting and running Causal LM models on Cloud AI 100 using the **dynamo** (`torch.export`) export path. This path requires PyTorch 2.13+ and produces more optimized ONNX graphs than the legacy TorchScript exporter.

## Prerequisites

### 1. Install QEfficient
```bash
pip install -e .
```

### 2. Install dynamo dependencies (PyTorch 2.13)
```bash
pip install -r examples/dynamo/causal_lm/requirements.txt
```

> **Note:** `requirements.txt` installs PyTorch 2.13 CPU wheels and `onnxscript`/`compressed-tensors`. These override any existing torch installation. For x86_64 and aarch64 — Python 3.9–3.12.

### 3. HuggingFace authentication (gated models)
```bash
export HF_TOKEN=<your_token>
```

---

## Available Examples

### `basic_dynamo_inference.py`

End-to-end example: export via `torch.export` (dynamo), compile to QPC, and run generation on Cloud AI 100.

**Usage:**
```bash
python examples/dynamo/causal_lm/basic_dynamo_inference.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --prompt "My name is" \
    --prefill-seq-len 32 \
    --ctx-len 128 \
    --num-cores 16
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `--model-name` | `Qwen/Qwen2-1.5B-Instruct` | HuggingFace model ID |
| `--prompt` | `"My name is"` | Input prompt |
| `--prefill-seq-len` | `32` | Prefill sequence length |
| `--ctx-len` | `128` | KV-cache context length |
| `--generation-len` | `100` | New tokens to generate |
| `--num-cores` | `16` | Number of AI 100 cores |
| `--aic-hw-version` | `ai100` | Hardware version |
| `--num-hidden-layers` | `-1` | Override model depth (for debugging) |
| `--device-group` | `None` | Device IDs, e.g. `[0,1]` |

This example:
- Loads the model via `QEFFAutoModelForCausalLM.from_pretrained`
- Exports using `torch.export` (`dynamo=True`) with ONNX subfunctions enabled
- Compiles to a QPC binary for Cloud AI 100
- Runs token generation and prints the output

---

## Additional Resources

- [QEfficient Documentation](../../../docs/source/index.rst)
- [Text Generation Examples (TorchScript path)](../../text_generation/README.md)
- [Validated Models](../../../docs/source/validate.md)
