# FLUX.1-schnell Image Generation Examples

This directory contains examples demonstrating how to use the QEffFluxPipeline to generate images using the FLUX.1-schnell model from Black Forest Labs.

## Overview

FLUX.1-schnell is a fast, distilled version of the FLUX.1 text-to-image model optimized for speed with minimal quality loss. These examples show how to leverage Qualcomm Cloud AI 100 acceleration for efficient image generation.

## Files

- **`flux_1_schnell.py`** - Basic example showing simple image generation
- **`flux_1_shnell_custom.py`** - Advanced example with customization options
- **`flux_config.json`** - Configuration file for pipeline modules

## Quick Start

### Basic Usage

The simplest way to generate images with FLUX.1-schnell:

```python
from QEfficient import QEffFluxPipeline
import torch

# Initialize pipeline
pipeline = QEffFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")

# Generate image
output = pipeline(
    prompt="A laughing girl",
    height=1024,
    width=1024,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.manual_seed(42),
    parallel_compile=True,
    use_onnx_subfunctions=False,
)

# Save image
output.images[0].save("girl_laughing.png")
```

Run the basic example:
```bash
python flux_1_schnell.py
```

## Advanced Customization

The `flux_1_shnell_custom.py` example demonstrates several advanced features:

### 1. Custom Model Components

You can provide custom text encoders, transformers, and tokenizers:

```python
pipeline = QEffFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    text_encoder=custom_text_encoder,
    transformer=custom_transformer,
    tokenizer=custom_tokenizer,
)
```

### 2. Custom Scheduler

Replace the default scheduler with your own:

```python
pipeline.scheduler = custom_scheduler.from_config(pipeline.scheduler.config)
```

### 3. Reduce Model Layers for Faster Inference

Trade quality for speed by reducing transformer blocks:

```python
original_blocks = pipeline.transformer.model.transformer_blocks
org_single_blocks = pipeline.transformer.model.single_transformer_blocks
pipeline.transformer.model.transformer_blocks = torch.nn.ModuleList([original_blocks[0]])
pipeline.transformer.model.single_transformer_blocks = torch.nn.ModuleList([org_single_blocks[0]])
pipeline.transformer.model.config['num_layers'] = 1
pipeline.transformer.model.config['num_single_layers'] = 1
```

### 4. Pre-compile with Custom Configuration

Compile the model separately before generation:

```python
pipeline.compile(
    compile_config="examples/diffusers/flux/flux_config.json",
    height=512,
    width=512,
    use_onnx_subfunctions=False
)
```

### 5. Runtime Configuration

Use custom configuration during generation:

```python
output = pipeline(
    prompt="A girl laughing",
    custom_config_path="examples/diffusers/flux/flux_config.json",
    height=1024,
    width=1024,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=torch.manual_seed(42),
    parallel_compile=True,
    use_onnx_subfunctions=False,
)
```

Run the advanced example:
```bash
python flux_1_shnell_custom.py
```

## Configuration File

The `flux_config.json` file controls compilation and execution settings for each pipeline module:

### Module Structure

The configuration includes four main modules:

1. **text_encoder** (CLIP) - Encodes text prompts (77 token sequence)
2. **text_encoder_2** (T5) - Secondary text encoder (256 token sequence)
3. **transformer** - Core diffusion transformer model
4. **vae_decoder** - Decodes latents to images

### Configuration Parameters

Each module has three sections:

#### Specializations
- `batch_size`: Batch size for inference
- `seq_len`: Sequence length for text encoders
- `steps`: Number of inference steps (transformer only)
- `channels`: Number of channels (VAE decoder only)

#### Compilation
- `onnx_path`: Path to pre-exported ONNX model (null for auto-export)
- `compile_dir`: Directory for compiled artifacts (null for auto-generation)
- `mdp_ts_num_devices`: Number of devices for model data parallelism
- `mxfp6_matmul`: Enable MXFP6 quantization for matrix multiplication
- `convert_to_fp16`: Convert model to FP16 precision
- `aic_num_cores`: Number of AI cores to use
- `mos`: Multi-output streaming (transformer only)
- `mdts-mos`: Multi-device tensor slicing with MOS (transformer only)
- `aic-enable-depth-first`: Enable depth-first compilation (VAE only)

#### Execute
- `device_ids`: List of device IDs to use (null for auto-selection)

### Example Configuration Snippet

```json
{
  "transformer": {
    "specializations": {
      "batch_size": 1,
      "seq_len": 256,
      "steps": 1
    },
    "compilation": {
      "mdp_ts_num_devices": 4,
      "mxfp6_matmul": true,
      "convert_to_fp16": true,
      "aic_num_cores": 16
    },
    "execute": {
      "device_ids": null
    }
  }
}
```

## Key Parameters

### Generation Parameters

- **`prompt`** (str): Text description of the image to generate
- **`height`** (int): Output image height in pixels (default: 1024)
- **`width`** (int): Output image width in pixels (default: 1024)
- **`guidance_scale`** (float): Classifier-free guidance scale (0.0 for schnell)
- **`num_inference_steps`** (int): Number of denoising steps (4 recommended for schnell)
- **`max_sequence_length`** (int): Maximum text sequence length (256 recommended)
- **`generator`** (torch.Generator): Random seed for reproducibility
- **`parallel_compile`** (bool): Enable parallel compilation of modules
- **`use_onnx_subfunctions`** (bool): Enable ONNX modular export (experimental)

### Performance Tuning

- **Faster inference**: Reduce `num_inference_steps` or model layers
- **Better quality**: Increase `num_inference_steps` or use full model
- **Memory optimization**: Adjust `mdp_ts_num_devices` in config
- **Precision trade-offs**: Toggle `mxfp6_matmul` and `convert_to_fp16`

## Output

The pipeline returns an output object containing:
- `images`: List of generated PIL Image objects
- Performance metrics (timing information)

Example output:
```python
print(output)  # Displays performance information
image = output.images[0]  # Access the generated image
image.save("output.png")  # Save to disk
```

## Hardware Requirements

- Qualcomm Cloud AI 100 accelerator
- Sufficient memory for model compilation and execution
- Multiple devices recommended for optimal transformer performance (see `mdp_ts_num_devices`)

## Notes

- FLUX.1-schnell is optimized for 4-step generation with `guidance_scale=0.0`
- The transformer module benefits most from multi-device parallelism
- ONNX subfunctions (`use_onnx_subfunctions=True`) is experimental and may improve compile time but is not recommended for production use
- Custom configurations allow fine-tuning for specific hardware setups

## Troubleshooting

- **Out of memory**: Reduce image dimensions or increase `mdp_ts_num_devices`
- **Slow compilation**: Enable `parallel_compile=True`
- **Quality issues**: Ensure using recommended parameters (4 steps, guidance_scale=0.0)
- **Device errors**: Check `device_ids` in config or set to `null` for auto-selection

## References

- [FLUX.1 Model Card](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [QEfficient Documentation](../../../README.md)
- [Diffusers Pipeline Guide](../../README.md)
