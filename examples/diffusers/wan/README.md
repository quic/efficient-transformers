# WAN 2.2 Text-to-Video Generation Examples

This directory contains examples demonstrating how to use the QEffWanPipeline to generate videos using the WAN 2.2 text-to-video model with Lightning LoRA optimization.

## Overview

WAN 2.2 is a text-to-video diffusion model that uses dual-stage processing for high-quality video generation. These examples show how to leverage Qualcomm Cloud AI 100 acceleration for efficient video generation with Lightning LoRA for fast 4-step inference.

## Files

- **`wan_lightning.py`** - Complete example with Lightning LoRA for fast video generation
- **`wan_config.json`** - Configuration file for transformer module compilation

## Quick Start

### Basic Usage

The simplest way to generate videos with WAN 2.2 Lightning:
### 1. Load Model
```python
from QEfficient import QEffWanPipeline
import torch
from diffusers.utils import export_to_video

# Initialize pipeline
pipeline = QEffWanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
```

### 2. Lightning LoRA Integration

Load high and low noise LoRA adapters for fast 4-step generation:

```python
from huggingface_hub import hf_hub_download
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
import safetensors.torch

# Download Lightning LoRAs
high_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
)
low_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
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


### 3. Compile API

To compile the model for desired resolution:

```python
# Compile with custom configuration
pipeline.compile(
    compile_config="examples/diffusers/wan/wan_config.json",
    parallel=True,
    height=480,
    width=832,
    num_frames=81,
    use_onnx_subfunctions=False,
)
```

### 4. Generate video
```python
output = pipeline(
    prompt="A cat playing in a sunny garden",
    num_frames=81,
    height=480,
    width=832,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.manual_seed(42),
    parallel_compile=True,
    use_onnx_subfunctions=False,
)

# Export video
frames = output.images[0]
export_to_video(frames, "cat_garden.mp4", fps=16)
```

Run the Lightning example:
```bash
python wan_lightning.py
```

## Advanced Customization


### 1. Reduce Model Layers for Faster Inference


```python
# Reduce to 2 layers for faster inference
pipeline.transformer.model.transformer_high.config['num_layers'] = 2
pipeline.transformer.model.transformer_low.config['num_layers']= 2

original_blocks = pipeline.transformer.model.transformer_high.blocks
org_blocks = pipeline.transformer.model.transformer_low.blocks

pipeline.transformer.model.transformer_high.blocks = torch.nn.ModuleList(
    [original_blocks[i] for i in range(0, pipeline.transformer.model.transformer_high.config.num_layers)]
)
pipeline.transformer.model.transformer_low.blocks = torch.nn.ModuleList(
    [org_blocks[i] for i in range(0, pipeline.transformer.model.transformer_low.config.num_layers)]
)
```

### 2. To Run with Blocking

Use environment variables to enable attention blocking:

```bash
# For 180p Generation (192x320) with HKV blocking
ATTENTION_BLOCKING_MODE=kv head_block_size=16 num_kv_blocks=3 python wan_lightning.py

# For 480p Generation (480x832) with HQKV blocking
ATTENTION_BLOCKING_MODE=qkv head_block_size=16 num_kv_blocks=21 num_q_blocks=2 python wan_lightning.py

# for 720P Generation (720x1280) with HQKV blocking
ATTENTION_BLOCKING_MODE=qkv head_block_size=16 num_kv_blocks=48 num_q_blocks=5 python wan_lightning.py
```

### Blocking Modes

Head blocking is common in all modes

- **`kv`**: Block key-value processing (along with Head blocking)
- **`q`**: Block query processing (along with Head blocking)
- **`qkv`**: Block query, key, and value (along with Head blocking)
- **`default`**: Head-only blocking


## Configuration File

The `wan_config.json` file controls compilation settings for the transformer module:

### Module Structure

The configuration includes dual specializations for WAN's high and low noise models:

```json
{
  "transformer": {
    "specializations":[
        {
            "batch_size": "1",
            "num_channels": "16",
            "steps": "1",
            "sequence_length": "512",
            "model_type": "1"
        },
        {
            "batch_size": "1",
            "num_channels": "16",
            "steps": "1",
            "sequence_length": "512",
            "model_type": "2"
        }
    ]
}
}
```

### Configuration Parameters

#### Specializations
- `batch_size`: Batch size for inference
- `num_channels`: Number of latent channels (16 for WAN)
- `sequence_length` : Sequence length of text encoder 512
- `model_type`: 1 for high noise model, 2 for low noise model

#### Compilation
- `mdp_ts_num_devices`: Number of devices for model parallelism (16 recommended)
- `mxfp6_matmul`: Enable MXFP6 quantization for matrix multiplication
- `convert_to_fp16`: Convert model to FP16 precision
- `aic_num_cores`: Number of AI cores to use (16 recommended)
- `mos`: Degree of weight splitting done across cores (1 is recommended)
- `mdts_mos`: Degree of weight splitting done across multi-device tensor slices (1 is recommended)

## Key Parameters

### Generation Parameters

- **`prompt`** (str): Text description of the video to generate
- **`num_frames`** (int): Number of video frames (default: 81)
- **`height`** (int): Output video height in pixels (default: 480)
- **`width`** (int): Output video width in pixels (default: 832)
- **`guidance_scale`** (float): Guidance scale for high noise stage (1.0 for Lightning)
- **`guidance_scale_2`** (float): Guidance scale for low noise stage (1.0 for Lightning)
- **`num_inference_steps`** (int): Number of denoising steps (4 for Lightning)
- **`generator`** (torch.Generator): Random seed for reproducibility
- **`parallel_compile`** (bool): Enable parallel compilation of modules
- **`use_onnx_subfunctions`** (bool): Enable ONNX modular export


## Output

The pipeline returns an output object containing:
- `images`: List of video frames as PIL Image objects
- Performance metrics (timing information)

Example output:
```python
print(output)  # Displays performance information
frames = output.images[0]  # Access the generated video frames
export_to_video(frames, "output.mp4", fps=16)  # Export to MP4
```

## Notes

- WAN 2.2 Lightning is optimized for 4-step generation with `guidance_scale=1.0`
- The transformer uses dual-stage processing (high/low noise models)
- Attention blocking is essential for higher resolutions (480p+)


## References

- [WAN 2.2 Model Card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
- [Lightning LoRA](https://huggingface.co/lightx2v/Wan2.2-Lightning)
- [QEfficient Documentation](../../../README.md)
