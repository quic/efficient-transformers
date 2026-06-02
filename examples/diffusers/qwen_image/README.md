# Qwen Image Generation Examples

This directory contains examples showing how to use `QEffQwenImagePipeline` to generate images with Qwen Image models.

## Overview

Qwen Image is a text-to-image diffusion model. These examples demonstrate end-to-end image generation with QEfficient export, compile, and execution flow for Qualcomm Cloud AI 100.

## Files

- **`qwen_image_example.py`** - Basic image generation example
- **`qwen_image_custom.py`** - Advanced example with customization options
- **`qwen_config.json`** - Configuration file for transformer and VAE modules

## Quick Start

### Basic Usage

```python
from QEfficient import QEffQwenImagePipeline
import torch

pipeline = QEffQwenImagePipeline.from_pretrained("Qwen/Qwen-Image")

output = pipeline(
    prompt="A cinematic photo of a coffee shop street in rain",
    negative_prompt="low quality, blurry",
    width=1664,
    height=928,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
    parallel_compile=True,
    max_sequence_length=128,
)

output.images[0].save("qwen_output.png")
```

Run the basic example:

```bash
python qwen_image_example.py
```

## Advanced Customization

`qwen_image_custom.py` includes common customizations.

### 1. Custom model components

```python
pipeline = QEffQwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image",
    transformer=custom_transformer,
    vae=custom_vae,
    text_encoder=custom_text_encoder,
    tokenizer=custom_tokenizer,
)
```

### 2. Custom scheduler

```python
pipeline.scheduler = custom_scheduler.from_config(pipeline.scheduler.config)
```

### 3. Reduce layers for faster iteration

```python
original_blocks = pipeline.transformer.model.transformer_blocks
pipeline.transformer.model.transformer_blocks = torch.nn.ModuleList([original_blocks[0], original_blocks[1]])
pipeline.transformer.model.config.num_layers = 2
```

### 4. Compile with custom configuration

```python
pipeline.compile(
    compile_config="examples/diffusers/qwen_image/qwen_config.json",
    parallel=True,
    height=928,
    width=1664,
    use_onnx_subfunctions=False,
)
```

### 5. Skip export/compile if precompiled QPC exists

Update `qwen_config.json` with prebuilt `qpc_path` in `execute` for each module:

```json
"execute": {
  "device_ids": null,
  "qpc_path": "<QPC_PATH>"
}
```

### 6. Runtime custom config

```python
output = pipeline(
    prompt="A modern storefront at golden hour",
    negative_prompt="low quality",
    width=1664,
    height=928,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
    custom_config_path="examples/diffusers/qwen_image/qwen_config.json",
    parallel_compile=True,
    max_sequence_length=128,
    use_onnx_subfunctions=True,
)
```

Run the advanced example:

```bash
python qwen_image_custom.py
```

## Configuration File

`qwen_config.json` controls specialization, compile, and execute settings for:

- `transformer`
- `vae_decoder`

### Common parameter groups

#### Specializations
- `batch_size`
- `cl` (image token length for transformer)
- `seq_length` (text sequence length)
- `latent_height`, `latent_width` (VAE decode shape)

#### Compilation
- `onnx_path`: Path to pre-exported ONNX model (null for auto-export)
- `compile_dir`: Directory for compiled artifacts (null for auto-generation)
- `mdp_ts_num_devices`: Number of devices for model data parallelism
- `mxfp6_matmul`: Enable MXFP6 quantization for matrix multiplication
- `convert_to_fp16`: Convert model to FP16 precision
- `aic_num_cores`: Number of AI cores to use
- `mos`: Multi-output streaming
- `mdts-mos`: Multi-device tensor slicing with MOS (transformer only)
- `aic-enable-depth-first`: Enable depth-first compilation

#### Execute
- `device_ids`: List of device IDs to use (null for auto-selection)
- `qpc_path` : compiled qpc path, to skip recompilation (null by default)

## Key Generation Parameters

- **`prompt`**: Positive prompt string
- **`negative_prompt`**: Negative prompt string
- **`width`**, **`height`**: Output image size
- **`num_inference_steps`**: Number of denoising steps
- **`true_cfg_scale`**: Classifier-free guidance scale
- **`max_sequence_length`**: Text sequence length limit
- **`generator`**: Seeded torch generator for reproducibility
- **`parallel_compile`**: Compile multiple modules in parallel
- **`use_onnx_subfunctions`**: Enable ONNX modular export (experimental)

## Output

Pipeline output contains generated images and performance metadata.

```python
print(output)
image = output.images[0]
image.save("output.png")
```

## References

- [Qwen Image Model Card](https://huggingface.co/Qwen/Qwen-Image)
- [QEfficient Documentation](../../../README.md)