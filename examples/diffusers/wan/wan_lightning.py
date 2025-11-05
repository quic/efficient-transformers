
import torch
import onnx
from pathlib import Path
from huggingface_hub import hf_hub_download
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
import safetensors.torch

from QEfficient import QEFFWanPipeline

# Load the pipe
pipeline = QEFFWanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

# Download the LoRAs
high_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors"
)
low_noise_lora_path = hf_hub_download(
    repo_id="lightx2v/Wan2.2-Lightning",
    filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
)

# LoRA conversion
def load_wan_lora(path: str):
    return _convert_non_diffusers_wan_lora_to_diffusers(
        safetensors.torch.load_file(path)
    )

# Load into the transformers
pipeline.transformer.model.load_lora_adapter(load_wan_lora(high_noise_lora_path), adapter_name="high_noise")
pipeline.transformer.model.set_adapters(["high_noise"], weights=[1.0])

pipeline.transformer_2.model.load_lora_adapter(load_wan_lora(low_noise_lora_path), adapter_name="low_noise")
pipeline.transformer_2.model.set_adapters(["low_noise"], weights=[1.0])

### for 2 layer model
pipeline.transformer.model.config.num_layers = 2 # for  2 layers
pipeline.transformer_2.model.config.num_layers = 2 # for 2 layers
original_blocks = pipeline.transformer.model.blocks
org_blocks = pipeline.transformer_2.model.blocks
pipeline.transformer.model.blocks = torch.nn.ModuleList([original_blocks[i] for i in range(0,pipeline.transformer.model.config.num_layers)])
pipeline.transformer_2.model.blocks = torch.nn.ModuleList([org_blocks[i] for i in range(0,pipeline.transformer_2.model.config.num_layers)])

pipeline = pipeline.to("cpu")

transformer1_onnx_path, transformer2_onnx_path = pipeline.export()
print(f"high_noise model onnx path : {transformer1_onnx_path},\nLow_noise model onnx path : {transformer2_onnx_path} ")