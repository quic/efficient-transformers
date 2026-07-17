# Weight-Free ONNX Export

Weight-free export lets you generate an ONNX graph for a causal LM **without loading
model weights into CPU RAM**. The QAIC compiler loads weights directly from the original
safetensors checkpoint at compile time, using a mapping file (`weight_spec.json`) that
is built during export.

---

## Why weight-free?

Standard export loads all weights into RAM before exporting:

```
Load weights → RAM  →  torch.onnx.export  →  ONNX (weights embedded)  →  QAIC compile
```

For a 405B model this requires ~800 GB of RAM just to run the export step. Weight-free
export avoids this entirely:

```
Meta-device model (0 RAM)  →  torch.onnx.export  →  ONNX (structure only)
                                                       + weight_spec.json
                                                       + prepared_checkpoint/
                                                             ↓
                                                       QAIC compile (compiler loads weights)
```

The model is built on PyTorch's **meta device** — tensors have shape and dtype but no
storage. The ONNX graph captures the compute graph only. The `weight_spec.json` file
maps every ONNX initializer name to the corresponding key and file in the safetensors
checkpoint, so the compiler can resolve weights at compile time.

---

## How to enable

Pass `use_weight_free_export=True` to `compile()` or `export()`. The model must be
constructed on meta device first:

```python
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from QEfficient import QEFFAutoModelForCausalLM

model_name = "meta-llama/Llama-3.1-8B-Instruct"
config = AutoConfig.from_pretrained(model_name)
config.dtype = torch.float32

# Build model skeleton on meta device — no weights in RAM
with init_empty_weights():
    meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")

model = QEFFAutoModelForCausalLM(
    meta_model,
    pretrained_model_name_or_path=model_name,   # used to locate checkpoint
)

qpc_path = model.compile(
    prefill_seq_len=1,
    ctx_len=4096,
    num_cores=16,
    use_dynamo=True,
    use_weight_free_export=True,          # ← enable weight-free
)
```

See `examples/text_generation/weight_free/export_compile_infer.py` for a runnable
end-to-end example with timing output.

---

## What happens on the first run

### 1. Weights download

If the model is not yet in the HF cache (`~/.cache/huggingface/hub/`), the exporter
calls `snapshot_download()` to fetch the safetensors shards. Models that only have
`.bin` checkpoints on the Hub are automatically downloaded and converted to safetensors
before proceeding.

Set `HF_HUB_CACHE` to control where downloads land. Set
`HF_HUB_ENABLE_HF_TRANSFER=1` for faster parallel downloads via the `hf_transfer`
library.

### 2. Checkpoint preparation (one-time conversion)

The `CheckpointTransformPipeline` processes the raw checkpoint and writes a
**prepared checkpoint** next to the ONNX output:

```
<compile_dir>/
  <model_name>/
    model.onnx              ← ONNX graph (no embedded weights)
    weight_spec.json        ← ONNX initializer name → checkpoint key mapping
    prepared_checkpoint/    ← processed safetensors (one-time output)
      model-part-0001.safetensors
      model-part-0002.safetensors
      model.safetensors.index.json
```

The pipeline runs the following transforms in order:

| Transform | What it does |
|---|---|
| `BinToSafetensorsTransform` | Converts legacy `.bin` shards to safetensors (skipped if already safetensors) |
| `DtypeConversionTransform` | Casts all tensors to `float32` to match the ONNX graph dtype |
| `MoEExpertStackingTransform` | Stacks per-expert weight tensors `[ffn, hidden]` × E → `[E, ffn, hidden]` for MoE models (Mixtral, Qwen3-MoE, GraniteMoE, GLM4-MoE, GPT-OSS) |

**This is a one-time operation.** On subsequent runs, the exporter detects the existing
`prepared_checkpoint/` and skips the transform pipeline entirely.

### 3. ONNX export

`torch.onnx.export` traces the meta-device model using the dynamo path. Because the
model has no real weights, the ONNX initializers reference empty meta tensors. The
`_promote_initializers_and_build_spec()` step then:

1. Walks every ONNX initializer (weight tensor) in the graph
2. Resolves its corresponding key in the prepared checkpoint using `_find_checkpoint_key()`
   (five lookup strategies including Mixtral `mlp` → `block_sparse_moe` rename handling)
3. Removes the tensor data from the ONNX graph
4. Writes the mapping to `weight_spec.json`

### 4. QAIC compile

The QAIC compiler reads `model.onnx`, resolves weights via `weight_spec.json` from the
`prepared_checkpoint/`, fuses and quantizes, and produces the QPC binary.

---

## Reusing prepared checkpoints across runs and machines

The `prepared_checkpoint/` directory is **fully portable**. Once generated, it can be:

- **Reused on the same machine** — subsequent export/compile calls detect it and skip
  re-conversion automatically.
- **Copied to another machine** — copy the entire `<compile_dir>/<model_name>/`
  directory (which contains `model.onnx`, `weight_spec.json`, and
  `prepared_checkpoint/`) to the target machine. Point `compile_dir` at the same path
  and run `compile()` — the QAIC compiler step runs locally without re-downloading or
  re-converting weights.
- **Shared via network storage** — mount the directory on multiple machines and compile
  in parallel.

### Detecting reuse

The exporter checks for an existing `prepared_checkpoint/` before running the pipeline:

```python
if (prepared_checkpoint_dir / "model.safetensors.index.json").exists():
    # skip — checkpoint already prepared
    return prepared_checkpoint_dir
```

There is no hash or version check. If you update the model weights (e.g. new model
revision), delete `prepared_checkpoint/` manually to force re-conversion.

---

## Memory savings

| Model | Standard export RAM | Weight-free export RAM |
|---|---|---|
| Llama 3.1 8B | ~32 GB | ~2 GB (model structure only) |
| Llama 3.1 70B | ~280 GB | ~2 GB |
| Llama 3.1 405B | ~800 GB | ~2 GB |
| Mixtral 8×7B | ~100 GB | ~2 GB |
| Qwen3-235B-A22B | ~500 GB | ~2 GB |

The ~2 GB baseline covers the Python process, PyTorch runtime, and the ONNX graph
structure (no weight data).

---

## Supported models

Weight-free export has been validated on:

| Architecture | Example model |
|---|---|
| LlamaForCausalLM | meta-llama/Llama-3.1-8B-Instruct |
| MistralForCausalLM | mistralai/Mistral-7B-Instruct-v0.1 |
| MixtralForCausalLM | mistralai/Mixtral-8x7B-Instruct-v0.1 |
| Qwen2ForCausalLM | Qwen/Qwen2-1.5B-Instruct |
| Qwen3ForCausalLM | Qwen/Qwen3-8B |
| Qwen3MoeForCausalLM | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| Phi3ForCausalLM | microsoft/Phi-3-mini-4k-instruct |
| GemmaForCausalLM | google/gemma-2-2b |
| Olmo2ForCausalLM | allenai/OLMo-2-0425-1B |
| GPT2LMHeadModel | openai-community/gpt2 |
| GPTBigCodeForCausalLM | bigcode/starcoder2-3b |
| GPTJForCausalLM | EleutherAI/gpt-j-6b |
| FalconForCausalLM | tiiuae/falcon-7b |
| GraniteForCausalLM | ibm-granite/granite-3.1-8b-instruct |
| GraniteMoeForCausalLM | hf-internal-testing/tiny-random-GraniteMoeForCausalLM |
| GptOssForCausalLM | openai/gpt-oss-20b |

---

## Key source files

| File | Purpose |
|---|---|
| `QEfficient/exporter/weight_free/core.py` | Main export logic, checkpoint key resolution, `weight_spec.json` generation |
| `QEfficient/exporter/weight_free/spec.py` | `weight_spec.json` format, read/write helpers |
| `QEfficient/exporter/weight_free/transforms.py` | `CheckpointTransformPipeline` and all checkpoint transforms |
| `QEfficient/base/modeling_qeff.py` | `use_weight_free_export` wiring in `_export()` |
| `examples/text_generation/weight_free/export_compile_infer.py` | Single-model end-to-end example |
| `examples/text_generation/weight_free/continuous_batching.py` | Continuous batching example |
