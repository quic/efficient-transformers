import hashlib
import os
from pathlib import Path
from typing import Optional, Union
from venv import logger
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.utils.hub_utils import _check_legacy_sharding_variant_format
from huggingface_hub import read_dduf_file
from diffusers.pipelines.pipeline_loading_utils import _identify_model_variants
import torch
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.diffusers.pipelines.pipeline_utils import QEffTextEncoder, QEffUNet, QEffVAE,QEffSafetyChecker
from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform, KVCacheExternalModuleMapperTransform, KVCacheTransform
from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform
from QEfficient.utils import constants
from QEfficient.utils.cache import to_hashable
import torch.nn as nn

from diffusers.models.attention_processor import AttnProcessor


from transformers.models.clip.modeling_clip import CLIPAttention

class QEFFStableDiffusionPipeline():
    _hf_auto_class = StableDiffusionPipeline
    
    def __init__(self, model, *args, **kwargs):
        self.text_encoder = QEffTextEncoder(model)
        self.unet=QEffUNet(model)
        
        # VAE Encoder
        self.vae_encoder=QEffVAE(model, "encoder")
        self.vae_encoder.model.forward = lambda sample, return_dict: self.vae_encoder.model.encode(sample, return_dict)[0].sample()
        
        # VAE Decoder
        self.vae_decoder=QEffVAE(model, "decoder")
        self.vae_decoder.model.forward = lambda latent_sample, return_dict: self.vae_decoder.model.decode(latent_sample, return_dict)
        
        # Saftey Checker
        self.safety_checker= QEffSafetyChecker(model)
        self.safety_checker.model.forward = model.safety_checker.forward_onnx
        
        self.pretrained_model_name_or_path = kwargs.get("pretrained_model_name_or_path", None)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        kwargs.update({"attn_implementation": "eager"})
        model= cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float32, **kwargs)
        return cls(model, pretrained_model_name_or_path)
    
    def export(self, export_dir: Optional[str] = None) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        ``Optional`` Args:
           :export_dir (str, optional): The directory path to store ONNX-graph.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """
        
        # Text encoder export 
        
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN

        example_inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "attention_mask": torch.ones((bs, seq_len), dtype=torch.int64),
        }

        dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}, "attention_mask": {0: "batch_size", 1: "seq_len"}}

        output_names = ["last_hidden_state", "pooler_output"]

        # self.text_encoder.model.set_attn_processor(AttnProcessor())
        
        config = self.text_encoder.model.text_model.config
        for layer in self.text_encoder.model.text_model.encoder.layers:
            layer.self_attn = CLIPAttention(config)

        
        self.text_encoder_onnx_path = self.text_encoder.export(
            example_inputs,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
        )
        
        # UNET Export
        
        print("######################  Text Encoder Exported #####################")
        
        unet_example_input={
            "sample": torch.randn(bs, self.unet.model.in_channels, self.unet.model.config.sample_size, self.unet.model.config.sample_size),
            "timestep": torch.tensor([1]),
            "encoder_hidden_states": torch.randn(bs, seq_len, self.unet.model.config.cross_attention_dim),
            "return_dict": False,  
        }
                
        output_names=["out_sample"]
        
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"}
        }
        self.unet.model.set_attn_processor(AttnProcessor())

        self.unet_onnx_path = self.unet.export(
            unet_example_input,
            output_names,
            dynamic_axes,
            export_dir=export_dir,
        )
        
        print("######################  UNet Exported #####################")
        
        
        vae_encoder_input={
            "sample": torch.randn(bs, 3, 512, 512),
            "return_dict": False,  
        }
        
        output_names=["latent_sample"]
        
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        }
        
        self.vae_encoder.model.set_attn_processor(AttnProcessor())

        self.vae_encoder_onnx_path = self.vae_encoder.export(
            vae_encoder_input,
            output_names,
            dynamic_axes,
            export_dir=None,
        )
        
        print("######################  VAE Encoder Exported #####################")
        
        vae_decoder_input={
            "latent_sample": torch.randn(bs, 4, 64, 64),
            "return_dict": False,  
        }
        
        output_names=["sample"]
        
        dynamic_axes={
            "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        }
        
        self.vae_decoder.model.set_attn_processor(AttnProcessor())
        
        self.vae_decoder_onnx_path = self.vae_decoder.export(
            vae_decoder_input,
            output_names,
            dynamic_axes,
            export_dir=None,
        )
        
        print("######################  VAE Decoder Exported #####################")
        
        saftey_checker_input={
            "clip_input": torch.randn(bs, 3, 224, 224),
            "images": torch.randn(bs, 512, 512, 3)
            }
        output_names=["out_images", "has_nsfw_concepts"]
        
        dynamic_axes={
            "clip_input": {0: "batch", 1: "channels", 2: "clip_height", 3: "clip_width"},
            "images": {0: "batch", 1: "height", 2: "width", 3: "channels"},
        }
        
        # self.safety_checker.model.set_attn_processor(AttnProcessor())
        
        for layer in self.safety_checker.model.vision_model.vision_model.encoder.layers:            
            config = self.safety_checker.model.config.vision_config
            layer.self_attn = CLIPAttention(config)
            # Replace with eager version

        
        self.safety_checker_onnx_path = self.safety_checker.export(
            saftey_checker_input,
            output_names,
            dynamic_axes,
            export_dir=None,
        )
        
        print("######################  Safety Checker Exported #####################")
        
        
        # return self.text_encoder_onnx_path, self.unet_onnx_path, self.vae_encoder_onnx_path, self.vae_decoder_onnx_path, self.safety_checker_onnx_path
     
        
    def compile():
        pass
    @property
    def model_name(self) -> str:
        pass
    @property
    def model_hash(self) -> str:
        pass
        
        
    