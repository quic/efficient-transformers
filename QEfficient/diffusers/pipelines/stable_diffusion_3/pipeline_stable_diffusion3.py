import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from venv import logger
from diffusers import DiffusionPipeline, StableDiffusion3Pipeline, OnnxStableDiffusionPipeline
from diffusers.utils.hub_utils import _check_legacy_sharding_variant_format
from huggingface_hub import read_dduf_file
from diffusers.pipelines.pipeline_loading_utils import _identify_model_variants
import torch
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.diffusers.pipelines.pipeline_utils import QEffTextEncoder, QEffUNet, QEffVAE,QEffSafetyChecker, QEffSD3Transformer2DModel
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform, KVCacheExternalModuleMapperTransform, KVCacheTransform
from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform, GPTQToMatmulNbitsTransform
from QEfficient.utils import constants
from QEfficient.utils.cache import to_hashable
import torch.nn as nn
from diffusers.pipelines.onnx_utils import OnnxRuntimeModel
from diffusers import AutoencoderKL
import numpy as np
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from diffusers.models.attention_processor import AttnProcessor


from transformers.models.clip.modeling_clip import CLIPAttention

class QEFFStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    _hf_auto_class = StableDiffusion3Pipeline

    def __init__(self, model, *args, **kwargs):
        
        # super().__init__(*args, **kwargs)
        self.tokenizer = model.tokenizer
        self.tokenizer_2 = model.tokenizer_2
        self.tokenizer_3 = model.tokenizer_3
        self.tokenizer_max_length = model.tokenizer_max_length

        
        self.scheduler = model.scheduler
        self.feature_extractor = model.feature_extractor
        
        self.text_encoder = QEffTextEncoder(model.text_encoder)
        
        self.text_encoder_2= QEffTextEncoder(model.text_encoder_2)
        
        self.text_encoder_3= QEffTextEncoder(model.text_encoder_3)
        
        self.transformer =  QEffSD3Transformer2DModel(model.transformer)
        
        # VAE Decoder
        self.vae_decode=QEffVAE(model, "decoder")
        self.vae_decode.model.forward = lambda latent_sample, return_dict: self.vae_decode.model.decode(latent_sample, return_dict)
        
        self.vae_scale_factor = 2 ** (len(model.vae.config.block_out_channels) - 1) if getattr(model, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=model.vae_scale_factor)

        self.t_max_length = (
            model.tokenizer.model_max_length if hasattr(model, "tokenizer") and model.tokenizer is not None else 77
        )
        self.default_sample_size = (
            model.transformer.config.sample_size
            if hasattr(model, "transformer") and model.transformer is not None
            else 128
        )
        self.patch_size = (
            model.transformer.config.patch_size if hasattr(model, "transformer") and model.transformer is not None else 2
        )
        
        
        
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        kwargs.update({"attn_implementation": "eager"})
        model= cls._hf_auto_class.from_pretrained(pretrained_model_name_or_path,token=, torch_dtype=torch.float32, **kwargs)
        model.to("cpu")
        return cls(model, pretrained_model_name_or_path)
    
    def export(self, export_dir: Optional[str] = None) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        ``Optional`` Args:
           :export_dir (str, optional): The directory path to store ONNX-graph.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """
        
        # Export text_encoder
        # TEXT ENCODER
        
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = self.tokenizer.model_max_length

        example_inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "output_hidden_states": True,
        }

        dynamic_axes = {"input_ids": {0: "batch_size", 1: "seq_len"}}

        output_names=["pooler_output", "last_hidden_state"]
       
        for i in range(0,13):
            output_names.append("hidden_states_"+str(i) )

        self.text_encoder_onnx_path= self.text_encoder.export(
            inputs=example_inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
        )
        
        print("######################  TEXT ENCODER EXPORTED ######################")
        
        # TEXT ENCODER 2
        example_inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "output_hidden_states": True,
        }
        
        dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}}
        
        output_names=["pooler_output", "last_hidden_state"]
        
        for i in range(0,33):
            output_names.append("hidden_states_"+str(i) )
        
        self.text_encoder_2_onnx_path= self.text_encoder_2.export(
            inputs=example_inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
        )
        
        print("######################  TEXT ENCODER 2 EXPORTED ######################")
        
        
        # T5 TEXT ENCODER
        example_inputs = {
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64)
        }
        
        dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}}
        
        output_names=["last_hidden_state"]
        
        self.text_encoder_3_onnx_path = self.text_encoder_3.export(
            inputs=example_inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
        )
        
        print("######################  TEXT ENCODER 3 EXPORTED ######################")
        
        example_inputs={
            "hidden_states": torch.randn((2,
                                        self.transformer.model.config.in_channels,
                                        self.transformer.model.config.sample_size,
                                        self.transformer.model.config.sample_size,
                                        ),dtype=torch.float32
                                        ),
            "encoder_hidden_states": torch.randn((2,
                                                 seq_len,
                                                 self.transformer.model.config.joint_attention_dim
                                                 ),
                                                dtype=torch.float32),
            "pooled_projections": torch.randn((2, self.transformer.model.config.pooled_projection_dim), dtype=torch.float32),
            "timestep":  torch.randint(0, 20, (2,), dtype=torch.float32), 
        }
            
        output_names=["output"]
        
        dynamic_axes={
            "hidden_states": {0: "batch_size", 1: "latent_channels", 2: "latent_height", 3: "latent_width"},
            "encoder_hidden_states": {0: "batch_size", 1: "seq_len"},
            "pooled_projections": {0: "batch_size"},
        }
        
        self.transformer_onnx_path = self.transformer.export(
            inputs=example_inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=export_dir,
        )
        
        print("######################  TRANSFORMER EXPORTED ######################")
        
        # # VAE decode
        # vae_decoder_input={
        #     "latent_sample": torch.randn(bs, 4, 64, 64),
        #     "return_dict": False,  
        # }
        
        # output_names=["sample"]
        
        # dynamic_axes={
        #     "latent_sample": {0: "batch_size", 1: "channels", 2: "height", 3: "width"},
        # }
                
        # self.vae_decoder_onnx_path = self.vae_decode.export(
        #     vae_decoder_input,
        #     output_names,
        #     dynamic_axes,
        #     export_dir=None,
        # )
    def compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        seq_len: Union[int, List[int]] = 32,
        batch_size: int = 1,
        num_devices: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        **compiler_options,
    ) -> str:   
       
        # Compile text_encoder
        
        # Make specilization
        
        seq_len= self.tokenizer.model_max_length
        
        specializations = [
            {"batch_size": batch_size, "seq_len": seq_len},
        ]
        
        self.text_encoder_compile_path=self.text_encoder._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            **compiler_options,
        )    
        
        print("######################  Text Encoder Compiled #####################")
        
        # Compile text encoder 2
        
        specializations = [
            {"batch_size": batch_size, "seq_len": seq_len},
        ]   

        self.text_encoder_2_compile_path=self.text_encoder_2._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            **compiler_options,
        )    
        
        print("######################  Text Encoder 2 Compiled #####################")
        
        # Compile text_encoder 3
        seq_len= 256
        
        specializations = [
            {"batch_size": batch_size, "seq_len": seq_len},
        ]   
        
        self.text_encoder_3_compile_path=self.text_encoder_3._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            **compiler_options,
        )    
        print("######################  Text Encoder 3 Compiled #####################")
        
        # Compile transformer
        
        specializations=[
            {
            "batch_size": 2*batch_size,
            "latent_channels": self.transformer.model.config.in_channels,
            "latent_height": self.transformer.model.config.sample_size,
            "latent_width": self.transformer.model.config.sample_size,
            "seq_len": 333,
            }
            ]
        self.trasformers_compile_path=self.transformer._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=2,
            aic_num_cores=num_cores,
            **compiler_options,
        )
        print("######################  Transformer Compiled #####################")

    
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
        device_ids: List[int] = [0],
    ):


        if clip_model_index == 0:
            text_encoder = self.text_encoder
            tokenizer = self.tokenizer
        else:
            text_encoder = self.text_encoder_2
            tokenizer = self.tokenizer_2

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        ##### AI 100 related changes ######
        
        if clip_model_index==0:
            if self.text_encoder.qpc_session is None:
                self.text_encoder.qpc_session = QAICInferenceSession(str(self.text_encoder_compile_path))

            text_encoder_output={
                "pooler_output": np.random.rand(batch_size, 768).astype(np.int32),
                "last_hidden_state": np.random.rand(batch_size, 77, 768).astype(np.int32),
            }
            
            for i in range(0,13):
                text_encoder_output[f"hidden_states_{i}"] = np.random.rand(batch_size, 77, 768).astype(np.int32)
            self.text_encoder.qpc_session.set_buffers(text_encoder_output)
            
            prompt_embeds_pytorch = self.text_encoder.model(text_input_ids.to(device), output_hidden_states=True)
            aic_text_input={"input_ids": text_input_ids.numpy().astype(np.int64)}
            aic_embeddings= self.text_encoder.qpc_session.run(aic_text_input)
            
            ## CHECK ACC ##        
            pt_pooled_embed=prompt_embeds_pytorch[0].detach().numpy()
            aic_pooled_embed=aic_embeddings['pooler_output']
            mad = np.mean(np.abs(pt_pooled_embed - aic_pooled_embed))
            ### END CHECK ACC ##
                       
        else:
            if self.text_encoder_2.qpc_session is None:
                self.text_encoder_2.qpc_session = QAICInferenceSession(str(self.text_encoder_2_compile_path))
        
            text_encoder_2_output={
                "pooler_output": np.random.rand(batch_size, 1280).astype(np.int32),
                "last_hidden_state": np.random.rand(batch_size, 77, 1280).astype(np.int32),
            }
            
            for i in range(0,33):
                text_encoder_2_output[f"hidden_states_{i}"] = np.random.rand(batch_size, 77, 1280).astype(np.int32) 
            
            self.text_encoder_2.qpc_session.set_buffers(text_encoder_2_output)
            
            prompt_embeds_pytorch=self.text_encoder_2.model(text_input_ids.to(device), output_hidden_states=True)
            aic_text_input={"input_ids": text_input_ids.numpy().astype(np.int64)}
            aic_embeddings= self.text_encoder_2.qpc_session.run(aic_text_input)
            
            ## CHECK ACC ##        
            pt_pooled_embed=prompt_embeds_pytorch[0].detach().numpy()
            aic_pooled_embed=aic_embeddings['pooler_output']
            mad = np.mean(np.abs(pt_pooled_embed - aic_pooled_embed))
            ### END CHECK ACC ##
                    
        # prompt_embeds_pytorch = self.text_encoder.model(text_input_ids.to(device), output_hidden_states=True)
        # aic_text_input={"input_ids": text_input_ids.numpy().astype(np.int64)}
        # aic_embeddings= self.text_encoder.qpc_session.run(aic_text_input)
        
        # ## CHECK ACC ##        
        # pt_pooled_embed=prompt_embeds_pytorch[0].detach().numpy()
        # aic_pooled_embed=aic_embeddings['pooler_output']
        # mad = np.mean(np.abs(pt_pooled_embed - aic_pooled_embed))
        # ### END CHECK ACC ##
        
        print("CLIP pooled embed MAD: ", mad)
        
        pooled_prompt_embeds = torch.tensor(aic_pooled_embed)

        prompt_embeds = torch.tensor(aic_embeddings[list(aic_embeddings.keys())[-2]])
        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds    
    
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )
        # if self.text_encoder_3.qpc_session is None:
        #         self.text_encoder_3.qpc_session = QAICInferenceSession(str(self.text_encoder_3_compile_path))
        
        prompt_embeds = self.text_encoder_3.model(text_input_ids.to(device))[0]
        # aic_text_input={"input_ids": text_input_ids.numpy().astype(np.int64)}
        # aic_embeddings= self.text_encoder_3.qpc_session.run(aic_text_input)
        
        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
    ):
        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    
        
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        sigmas: Optional[List[float]] = None,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
        vae_type = "vae",
    ):


        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.model.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            "cpu",
            generator,
            latents,
        )
        
        # 5. Prepare timesteps
        scheduler_kwargs = {}
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            "cpu",
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        ###### AIC related changes of transformers ######
        if self.transformer.qpc_session is None:
            self.transformer.qpc_session=QAICInferenceSession(str("/home/amitraj/.cache/qeff_models/SD3Transformer2DModel-e3b0c44298fc1c14/qpc-ef39c368c29e943d/qpc"), [4,5])
            
            output_buffer={
                "output": np.random.rand(2*batch_size, num_channels_latents, self.default_sample_size, self.default_sample_size).astype(np.int32),
            }
        
            self.transformer.qpc_session.set_buffers(output_buffer)
         
        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # noise_pred = self.transformer.model(
                #     hidden_states=latent_model_input,
                #     timestep=timestep,
                #     encoder_hidden_states=prompt_embeds,
                #     pooled_projections=pooled_prompt_embeds,
                #     joint_attention_kwargs=self.joint_attention_kwargs,
                #     return_dict=False,
                # )[0]
                
                
                noise_pred=self.transformer.qpc_session.run(
                    {"encoder_hidden_states": prompt_embeds.detach().numpy(),
                     "pooled_projections": pooled_prompt_embeds.numpy(),
                     "timestep": timestep.numpy(),
                     "hidden_states": latent_model_input.numpy()
                     }
                )

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    should_skip_layers = (
                        True
                        if i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                        else False
                    ) 

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
  
        
        
        
        
        

        
        
         
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
            
        
        
        
