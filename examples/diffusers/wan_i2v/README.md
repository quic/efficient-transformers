# WAN 2.2 Image-to-Video Generation Examples

This directory contains examples demonstrating how to use the QEffWanImageToVideoPipeline to generate videos from input images using the WAN 2.2 Image-to-Video model with Lightning LoRA optimization.

## Overview

WAN 2.2 Image-to-Video (I2V) is a diffusion model that transforms static images into dynamic videos by adding temporal motion while preserving the original image content. The model uses dual-stage processing with Lightning LoRA for fast 4-step inference, making it ideal for efficient video generation on Qualcomm Cloud AI 100 hardware.

### Key Features

- **Image-to-Video Generation**: Transform any image into a dynamic video sequence
- **Lightning LoRA Integration**: Fast 4-step inference with high-quality results
- **Dual-Stage Processing**: Separate high and low noise transformers for optimal quality
- **Flexible Resolution Support**: 180p, 480p, and 720p with attention blocking
- **Hardware Acceleration**: Optimized for Qualcomm Cloud AI 100 inference

## Files

- **`wan_lightning_i2v.py`** - Basic I2V example with Lightning LoRA and simple progress callback
- **`wan_i2v_custom.py`** - Enhanced example with customization options
- **`wan_i2v_config.json`** - Configuration file for I2V pipeline compilation

**Note**: Update compilation configuration for desired configurations

## Quick Start

### 1. Basic Image-to-Video Generation

The simplest way to generate videos from images:

```python
from QEfficient import QEffWanImageToVideoPipeline
import torch
from diffusers.utils import export_to_video, load_image

# Initialize pipeline
pipeline = QEffWanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers")

# Load input image
image = load_image("path/to/your/image.jpg")

# Generate video
output = pipeline(
    image=image,
    prompt="A beautiful scene with gentle motion",
    num_frames=81,
    height=480,
    width=832,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.manual_seed(42),
)

# Export video
frames = output.images[0]
export_to_video(frames, "output.mp4", fps=16)
```

### 2. Run Basic Example

```bash
cd examples/diffusers/wan_i2v
python wan_lightning_i2v.py
```

### 3. Run Enhanced Custom Example

```bash
python wan_i2v_custom.py
```

## Image Input Requirements

### Supported Formats
- **JPEG** (.jpg, .jpeg)
- **PNG** (.png)
- **WebP** (.webp)
- **Remote URLs** (HTTP/HTTPS)

### Resolution Guidelines

The model works best with images that match the target video resolution:

| Target Resolution | Recommended Input Size |
|------------------|----------------------|
| **180p** | 192×320 |
| **480p** | 480×832 |
| **720p** | 720×1280 |

### Image Preprocessing

The pipeline automatically handles:
- **Aspect ratio preservation** with intelligent cropping
- **Resolution adjustment** to match model requirements
- **Format conversion** to RGB if needed


## Lightning LoRA Integration

### Automatic LoRA Loading

Both example scripts automatically download and load Lightning LoRA adapters:

```python
from huggingface_hub import hf_hub_download
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers

# Download Lightning LoRAs
high_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
)
low_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
)

# Load and apply LoRAs
def load_wan_lora(path: str):
    return _convert_non_diffusers_wan_lora_to_diffusers(safetensors.torch.load_file(path))

pipeline.transformer.model.transformer_high.load_lora_adapter(
    load_wan_lora(high_noise_lora_path), adapter_name="high_noise"
)
pipeline.transformer.model.transformer_high.set_adapters(["high_noise"], weights=[1.0])

pipeline.transformer.model.transformer_low.load_lora_adapter(
    load_wan_lora(low_noise_lora_path), adapter_name="low_noise"
)
pipeline.transformer.model.transformer_low.set_adapters(["low_noise"], weights=[1.0])
```

## Blocking Guide
### Attention Blocking Modes

- **`default`**: Head-only blocking
- **`kv`**: Head blocking with key-value blocking
- **`q`**: Head blocking with query blocking
- **`qkv`**: Head blocking with query, key, and value blocking

#### Use below environment variables for attention blocking
#### 180p Configuration
```bash
ATTENTION_BLOCKING_MODE=kv head_block_size=16 num_kv_blocks=3 python wan_lightning_i2v.py
```

#### 480p Configuration
```bash
ATTENTION_BLOCKING_MODE=qkv head_block_size=16 num_kv_blocks=21 num_q_blocks=2 python wan_lightning_i2v.py
```

#### 720p Configuration
```bash
ATTENTION_BLOCKING_MODE=qkv head_block_size=16 num_kv_blocks=48 num_q_blocks=5 python wan_lightning_i2v.py
```

## Key Parameters

#### Generation Parameters
- **`image`** (PIL.Image or str): Input image or path/URL
- **`prompt`** (str): Text description to guide video generation
- **`num_frames`** (int): Number of video frames (default: 81)
- **`height`** (int): Output video height in pixels
- **`width`** (int): Output video width in pixels
- **`guidance_scale`** (float): Guidance strength (1.0 for Lightning)
- **`num_inference_steps`** (int): Denoising steps (4 for Lightning)
- **`generator`** (torch.Generator): Random seed for reproducibility

#### Performance Parameters
- **`custom_config_path`** (str): Path to configuration JSON file
- **`use_onnx_subfunctions`** (bool): Enable ONNX modular export
- **`parallel_compile`** (bool): Enable parallel compilation

## Advanced Features

### 1. Layer Reduction for Speed

Reduce transformer layers for faster inference:

```python
# Configure for single-layer model (fastest)
pipeline.transformer.model.transformer_high.config['num_layers'] = 1
pipeline.transformer.model.transformer_low.config['num_layers'] = 1

# Reduce transformer blocks
original_blocks = pipeline.transformer.model.transformer_high.blocks
pipeline.transformer.model.transformer_high.blocks = torch.nn.ModuleList(
    [original_blocks[i] for i in range(0, 1)]
)

org_blocks = pipeline.transformer.model.transformer_low.blocks
pipeline.transformer.model.transformer_low.blocks = torch.nn.ModuleList(
    [org_blocks[i] for i in range(0, 1)]
)
```

### 2. For Multiple Images

Process multiple images efficiently:

```python
images = [load_image(path) for path in image_paths]
outputs = []

for i, image in enumerate(images):
    output = pipeline(
        image=image,
        prompt=prompts[i],
        # ... other parameters
    )
    outputs.append(output)
```

### 3. Custom Compilation

To compile the model for desired resolution:

```python
# Compile with custom configuration
pipeline.compile(
    compile_config="wan_i2v_config.json", # update compilation flags for desired config
    parallel=True,
    height=480,
    width=832,
    num_frames=81,
    use_onnx_subfunctions=True,
)
```
### 4. Skip export, compilation if pre-compiled qpc exist
Update custom config with qpc in execute of corresponding module.
```
"execute":
          {
           "device_ids": null,
           "qpc_path" : "<QPC_PATH>"
          }
```

### 5. Callback Functions for Progress Monitoring

Monitor generation progress and save intermediate drafts - refer **save_draft_callback** in wan I2V custom.py

```python
def save_draft_callback(pipeline, step, timestep, callback_kwargs):
    # Refer wan I2V custom.py script

# Use callback in pipeline
output = pipeline(
    image=image,
    prompt=prompt,
    # ... other parameters
    callback_on_step_end=save_draft_callback,
    callback_on_step_end_tensor_inputs=["latents"],
)
```

## References

- [WAN 2.2 I2V Model Card](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
- [Lightning LoRA](https://huggingface.co/lightx2v/Wan2.2-Lightning)
- [QEfficient Documentation](../../../README.md)
- [Diffusers Library](https://huggingface.co/docs/diffusers)
