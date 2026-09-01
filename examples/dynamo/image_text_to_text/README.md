# Dynamo Image-Text-to-Text Example

This folder contains a Qwen3-VL-MoE script for exporting, compiling, and
running a dual-QPC VLM on Cloud AI 100 using the **dynamo** (`torch.export`)
export path.

The default path uses weight-free export. The Hugging Face model is built on
meta tensors and the QAIC compiler loads weights from the checkpoint through
`weight_spec.json`.

## Prerequisites

Install QEfficient from the repository root:

```bash
pip install -e .
```

Install dynamo and VLM dependencies:

```bash
pip install -r examples/dynamo/image_text_to_text/requirements.txt
```

For private or gated Hugging Face models:

```bash
export HF_TOKEN=<your_token>
```

For local artifact placement:

```bash
export HF_HUB_CACHE=/home/huggingface_hub
export HF_HUB_ENABLE_HF_TRANSFER=1
export QEFF_HOME=/path/to/qeff_artifacts
export QEFF_CHECKPOINT_HOME=/path/to/qeff_prepared_checkpoints
```

## Qwen3-VL-MoE

Default usage:

```bash
python examples/dynamo/image_text_to_text/qwen3_vl_moe_dynamo_inference.py
```

The default run uses a reduced bring-up config:

- `vision_config.depth = 9`
- `text_config.num_hidden_layers = 1`
- `vision_config.deepstack_visual_indexes = [8]`

Disable that reduction with:

```bash
python examples/dynamo/image_text_to_text/qwen3_vl_moe_dynamo_inference.py --no-reduce-layers
```

Disable weight-free export only for explicit experimentation:

```bash
python examples/dynamo/image_text_to_text/qwen3_vl_moe_dynamo_inference.py --no-weight-free
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--model-name` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | Hugging Face model ID |
| `--prompt` | `"Describe all the colors seen in the image."` | Text prompt |
| `--image-url` | `https://picsum.photos/id/237/536/354` | Image URL |
| `--height` | `354` | Compile image height |
| `--width` | `536` | Compile image width |
| `--batch-size` | `1` | Compile/runtime batch size |
| `--prefill-seq-len` | `128` | Prefill sequence length |
| `--ctx-len` | `4096` | KV-cache context length |
| `--generation-len` | `100` | New tokens to generate |
| `--num-cores` | hardware default | Number of AI cores |
| `--num-devices` | `4` | Number of devices |
| `--mos` | `1` | Compiler MOS setting |
| `--aic-hw-version` | hardware default | AIC hardware version |
| `--reduce-layers` / `--no-reduce-layers` | `True` | Use the reduced bring-up config |
| `--weight-free` / `--no-weight-free` | `True` | Enable weight-free export |
| `--skip-vision` | `False` | Compile and run text path only |
