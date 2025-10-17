# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union
from venv import logger

import numpy as np
import torch
import regex as re

from diffusers import WanPipeline
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.video_processor import VideoProcessor
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from transformers import AutoTokenizer, UMT5EncoderModel

from QEfficient.diffusers.pipelines.pipeline_utils import  QEffTextEncoder, QEffClipTextEncoder, QEffVAE, QEffWanTransformerModel
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants

class QEFFWanPipeline(WanPipeline):
    _hf_auto_class = WanPipeline

    r"""
    Pipeline for text-to-video generation using Wan.
    A QEfficient-optimized Wan pipeline, inheriting from `diffusers.WanPipeline`.

    This class integrates QEfficient components (e.g., optimized models for umt5 text encoders,
    wan transformer, and VAE) to enhance performance, particularly for deployment on Qualcomm AI hardware.
    It provides methods for text-to-video generation leveraging these optimized components.
    """

    # model_cpu_offload_seq = "text_encoder->transformer->transformer_2->vae"
    # _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    # _optional_components = ["transformer", "transformer_2"]
            # self,
        # tokenizer: AutoTokenizer,
        # text_encoder: UMT5EncoderModel,
        # vae: AutoencoderKLWan,
        # scheduler: FlowMatchEulerDiscreteScheduler,
        # transformer: Optional[WanTransformer3DModel] = None,
        # transformer_2: Optional[WanTransformer3DModel] = None,
        # boundary_ratio: Optional[float] = None,
        # expand_timesteps: bool = False,  # Wan2.2 ti2v

    def __init__(self, model, *args, **kwargs):
        
        # Required by diffusers for serialization and device management
        self.model = model
        self.args = args
        self.kwargs = kwargs

        self.text_encoder = model.text_encoder  # UMT5EncoderModel ##TODO : update with  UMT5 encoder 
        self.transformer = QEffWanTransformerModel(model.transformer)
        self.vae_decode = QEffVAE(model, "decoder") # TODO check and compile
        self.tokenizer = model.tokenizer
        self.text_encoder.tokenizer = model.tokenizer
        self.scheduler = model.scheduler
        # import pdb; pdb.set_trace()
        # super().__init__(tokenizer=self.tokenizer, text_encoder=self.text_encoder, vae=self.vae_decode, scheduler=self.scheduler) # taken everything from parent


        self.register_modules(
            vae=self.vae_decode,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer ,
            scheduler=self.scheduler,
            transformer_2=model.transformer_2,
        )
        # import pdb; pdb.set_trace()
        boundary_ratio = self.kwargs.get("boundary_ratio", None)
        expand_timesteps = self.kwargs.get("expand_timesteps", True) ##TODO :  not used this part of code in onboarding
        self.register_to_config(boundary_ratio=boundary_ratio)
        self.register_to_config(expand_timesteps=expand_timesteps)
        self.vae_scale_factor_temporal = self.vae_decode.model.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae_decode.model.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        """
        Instantiate a QEFFWanTransformer3DModel from pretrained Diffusers models.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                The path to the pretrained model or its name.
            **kwargs (additional keyword arguments):
                Additional arguments that can be passed to the underlying `StableDiffusion3Pipeline.from_pretrained`
                method.
        """
        model = cls._hf_auto_class.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float32,
            **kwargs,
        )
        model.to("cpu")
        return cls(model, pretrained_model_name_or_path)
    
    @property
    def components(self):
        return {
            "text_encoder": self.text_encoder,
            "transformer": self.transformer,
            "transformer_2": getattr(self, "transformer_2", None),
            "vae": self.vae_decode,
            "tokenizer": self.tokenizer,
            "scheduler": self.scheduler,
        }
        
    
    def export(self, export_dir: Optional[str] = None) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        ``Optional`` Args:
           :export_dir (str, optional): The directory path to store ONNX-graph.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """

        # text_encoder - umt5 ##TODO: update once umt5 modeling is available
        # example_inputs_text_encoder, dynamic_axes_text_encoder, output_names_text_encoder = (
        #     self.text_encoder.get_onnx_config(seq_len = self.tokenizer.model_max_length)
        # )
        # self.text_encoder.export(
        #     inputs=example_inputs_text_encoder,
        #     output_names=output_names_text_encoder,
        #     dynamic_axes=dynamic_axes_text_encoder,
        #     export_dir=export_dir,
        # )

        # transformers
        example_inputs_transformer, dynamic_axes_transformer, output_names_transformer = (
            self.transformer.get_onnx_config()
        )
        self.transformer.export(
            inputs=example_inputs_transformer,
            output_names=output_names_transformer,
            dynamic_axes=dynamic_axes_transformer,
            export_dir=export_dir,
        )

        # vae
        example_inputs_vae, dynamic_axes_vae, output_names_vae = self.vae_decode.get_onnx_config()
        self.vae_decoder_onnx_path = self.vae_decode.export(
            example_inputs_vae,
            output_names_vae,
            dynamic_axes_vae,
            export_dir=export_dir,
        )


    def compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        seq_len: Union[int, List[int]] = 512,
        batch_size: int = 1,
        num_devices_text_encoder: int = 1,
        num_devices_transformer: int = 16,
        num_devices_vae_decoder: int = 1,
        num_cores: int = 16,  # FIXME: Make this mandatory arg
        mxfp6_matmul: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compiles the ONNX graphs of the different model components for deployment on Qualcomm AI hardware.

        This method takes the ONNX paths of the text encoders, transformer, and VAE decoder,
        and compiles them into an optimized format for inference.

        Args:
            onnx_path (`str`, *optional*):
                The base directory where ONNX files were exported. If None, it assumes the ONNX
                paths are already set as attributes (e.g., `self.text_encoder_onnx_path`).
                This parameter is currently not fully utilized as individual ONNX paths are derived
                from the `export` method.
            compile_dir (`str`, *optional*):
                The directory path to store the compiled artifacts. If None, a default location
                might be used by the underlying compilation process.
            seq_len (`Union[int, List[int]]`, *optional*, defaults to 32):
                The sequence length(s) to use for compiling the text encoders. Can be a single
                integer or a list of integers for multiple sequence lengths.
            batch_size (`int`, *optional*, defaults to 1):
                The batch size to use for compilation.
            num_devices_text_encoder (`int`, *optional*, defaults to 1):
                The number of AI devices to deploy the text encoder models on.
            num_devices_transformer (`int`, *optional*, defaults to 4):
                The number of AI devices to deploy the transformer model on.
            num_devices_vae_decoder (`int`, *optional*, defaults to 1):
                The number of AI devices to deploy the VAE decoder model on.
            num_cores (`int`, *optional*, defaults to 16):
                The number of cores to use for compilation. This argument is currently marked
                as `FIXME: Make this mandatory arg`.
            mxfp6_matmul (`bool`, *optional*, defaults to `False`):
                If `True`, enables mixed-precision floating-point 6-bit matrix multiplication
                optimization during compilation.
            **compiler_options:
                Additional keyword arguments to pass to the underlying compiler.

        Returns:
            `str`: A message indicating the compilation status or path to compiled artifacts.
            (Note: The current implementation might need to return specific paths for each compiled model).
        """
        # if any(
        #     path is None
        #     for path in [
        #         # self.text_encoder.onnx_path,
        #         self.transformer.onnx_path,
        #         self.vae_decode.onnx_path,
        #     ]
        # ):
        #     self.export()
        # text_encoder - umt5
        # specializations_text_encoder = self.text_encoder.get_specializations(
        #     batch_size, self.tokenizer.model_max_length
        # )

        # self.text_encoder_compile_path = self.text_encoder._compile(
        #     onnx_path,
        #     compile_dir,
        #     compile_only=True,
        #     specializations=specializations_text_encoder,
        #     convert_to_fp16=True,
        #     mxfp6_matmul=mxfp6_matmul,
        #     mdp_ts_num_devices=num_devices_text_encoder,
        #     aic_num_cores=num_cores,
        #     **compiler_options,
        # )


        # transformer
        # import pdb; pdb.set_trace()
        specializations_transformer = self.transformer.get_specializations(batch_size, seq_len)
        compiler_options = {"mos": 1, "mdts-mos":1}
        # self.trasformer_compile_path = "/home/vtirumal/wan_onboard/Wan2.2_5B-Diffusers-/qpcs/transformer/qpc_transformer/"
        self.trasformer_compile_path = self.transformer._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations_transformer,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices_transformer,
            aic_num_cores=num_cores,
            **compiler_options,
        )

        # vae
        # specializations_vae = self.vae_decode.get_specializations(batch_size)
        # self.vae_decoder_compile_path = self.vae_decode._compile(
        #     onnx_path,
        #     compile_dir,
        #     compile_only=True,
        #     specializations=specializations_vae,
        #     convert_to_fp16=True,
        #     mdp_ts_num_devices=num_devices_vae_decoder,
        # )
    
    # def _get_t5_prompt_embeds(
    #     self,
    #     prompt: Union[str, List[str]] = None,
    #     num_videos_per_prompt: int = 1,
    #     max_sequence_length: int = 226,
    #     device: Optional[torch.device] = None,
    #     dtype: Optional[torch.dtype] = None,
    # ):
    #     device = device or self._execution_device
    #     dtype = dtype or self.text_encoder.dtype

    #     prompt = [prompt] if isinstance(prompt, str) else prompt
    #     prompt = [prompt_clean(u) for u in prompt]
    #     batch_size = len(prompt)

    #     text_inputs = self.tokenizer(
    #         prompt,
    #         padding="max_length",
    #         max_length=max_sequence_length,
    #         truncation=True,
    #         add_special_tokens=True,
    #         return_attention_mask=True,
    #         return_tensors="pt",
    #     )
    #     text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    #     seq_lens = mask.gt(0).sum(dim=1).long()

    #     prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    #     prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    #     prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    #     prompt_embeds = torch.stack(
    #         [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    #     )

    #     # duplicate text embeddings for each generation per prompt, using mps friendly method
    #     _, seq_len, _ = prompt_embeds.shape
    #     prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    #     prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    #     return prompt_embeds

    # def encode_prompt(
    #     self,
    #     prompt: Union[str, List[str]],
    #     negative_prompt: Optional[Union[str, List[str]]] = None,
    #     do_classifier_free_guidance: bool = True,
    #     num_videos_per_prompt: int = 1,
    #     prompt_embeds: Optional[torch.Tensor] = None,
    #     negative_prompt_embeds: Optional[torch.Tensor] = None,
    #     max_sequence_length: int = 226,
    #     device: Optional[torch.device] = None,
    #     dtype: Optional[torch.dtype] = None,
    # ):
    #     r"""
    #     Encodes the prompt into text encoder hidden states.

    #     Args:
    #         prompt (`str` or `List[str]`, *optional*):
    #             prompt to be encoded
    #         negative_prompt (`str` or `List[str]`, *optional*):
    #             The prompt or prompts not to guide the image generation. If not defined, one has to pass
    #             `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
    #             less than `1`).
    #         do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
    #             Whether to use classifier free guidance or not.
    #         num_videos_per_prompt (`int`, *optional*, defaults to 1):
    #             Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
    #         prompt_embeds (`torch.Tensor`, *optional*):
    #             Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
    #             provided, text embeddings will be generated from `prompt` input argument.
    #         negative_prompt_embeds (`torch.Tensor`, *optional*):
    #             Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
    #             weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
    #             argument.
    #         device: (`torch.device`, *optional*):
    #             torch device
    #         dtype: (`torch.dtype`, *optional*):
    #             torch dtype
    #     """
    #     device = device or self._execution_device

    #     prompt = [prompt] if isinstance(prompt, str) else prompt
    #     if prompt is not None:
    #         batch_size = len(prompt)
    #     else:
    #         batch_size = prompt_embeds.shape[0]

    #     if prompt_embeds is None:
    #         prompt_embeds = self._get_t5_prompt_embeds(
    #             prompt=prompt,
    #             num_videos_per_prompt=num_videos_per_prompt,
    #             max_sequence_length=max_sequence_length,
    #             device=device,
    #             dtype=dtype,
    #         )

    #     if do_classifier_free_guidance and negative_prompt_embeds is None:
    #         negative_prompt = negative_prompt or ""
    #         negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

    #         if prompt is not None and type(prompt) is not type(negative_prompt):
    #             raise TypeError(
    #                 f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
    #                 f" {type(prompt)}."
    #             )
    #         elif batch_size != len(negative_prompt):
    #             raise ValueError(
    #                 f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
    #                 f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
    #                 " the batch size of `prompt`."
    #             )

    #         negative_prompt_embeds = self._get_t5_prompt_embeds(
    #             prompt=negative_prompt,
    #             num_videos_per_prompt=num_videos_per_prompt,
    #             max_sequence_length=max_sequence_length,
    #             device=device,
    #             dtype=dtype,
    #         )

    #     return prompt_embeds, negative_prompt_embeds


    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 61, #81
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        # if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        #     callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )
        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.model.dtype  # if self.transformer is not None else self.transformer_2.dtype update it to self.transformer_2.model.dtype for 14 B
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = (
            self.transformer.model.config.in_channels
            # if self.transformer is not None
            # else self.transformer_2.model.config.in_channels
        )

        # import pdb; pdb.set_trace()
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        # 6. Denoising loop
        ###### AIC related changes of transformers ######
        if self.transformer.qpc_session is None:
            self.transformer.qpc_session = QAICInferenceSession(str(self.trasformer_compile_path)) #, device_ids=device_ids_transformer)
        
        output_buffer = {
            "output": np.random.rand(
                batch_size, 6240, 192 #self.transformer.model.config.joint_attention_dim , self.transformer.model.config.in_channels
            ).astype(np.int32),
        }
        self.transformer.qpc_session.set_buffers(output_buffer)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer.model
                    current_guidance_scale = guidance_scale
                else:
                    print("NOT expected for wan 5 B ")
                #     # low-noise stage in wan2.2
                #     current_model = self.transformer_2.model
                #     current_guidance_scale = guidance_scale_2

                latent_model_input = latents.to(transformer_dtype)
                if self.config.expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                batch_size, num_channels, num_frames, height, width = latents.shape #  modeling
                p_t, p_h, p_w = self.transformer.model.config.patch_size
                post_patch_num_frames = num_frames // p_t
                post_patch_height = height // p_h
                post_patch_width = width // p_w

                # patch_states = self.transformer.patch_embedding(latent_model_input)
                # import pdb; pdb.set_trace()
                rotary_emb = self.transformer.model.rope(latent_model_input)
                rotary_emb = torch.cat(rotary_emb, dim=0)
                ts_seq_len = None 
                # ts_seq_len = timestep.shape[1]
                timestep = timestep.flatten()

                temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.transformer.model.condition_embedder(
                    timestep, prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=ts_seq_len
                )
                temb, timestep_proj, encoder_hidden_states_neg, encoder_hidden_states_image = self.transformer.model.condition_embedder(
                    timestep, negative_prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=ts_seq_len
                )
                # timestep_proj = timestep_proj.unflatten(2, (6, -1)) # for 5 B rnew_app.py ##TODO: cross check once 
                timestep_proj = timestep_proj.unflatten(1, (6, -1))
                # import pdb; pdb.set_trace()
                inputs_aic = {
                    "hidden_states": latents.detach().numpy(),
                    "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
                    "rotary_emb": rotary_emb.detach().numpy(),
                    "temb": temb.detach().numpy(),
                    "timestep_proj": timestep_proj.detach().numpy()
                }

                inputs_aic2 = {
                    "hidden_states": latents.detach().numpy(),
                    "encoder_hidden_states": encoder_hidden_states_neg.detach().numpy(),
                    "rotary_emb": rotary_emb.detach().numpy(),
                    "temb": temb.detach().numpy(),
                    "timestep_proj": timestep_proj.detach().numpy()
                }

                # import pdb; pdb.set_trace()

                # with current_model.cache_context("cond"):
                #     noise_pred_torch = current_model(
                #         hidden_states=latent_model_input,
                #         # timestep=timestep,
                #         encoder_hidden_states=encoder_hidden_states,
                #         rotary_emb=rotary_emb,
                #         temb=temb,
                #         timestep_proj=timestep_proj,
                #         attention_kwargs=attention_kwargs,
                #         return_dict=False,
                #     )[0]

                start_time = time.time()
                outputs = self.transformer.qpc_session.run(inputs_aic)
                end_time = time.time()
                print(f"Time : {end_time - start_time:.2f} seconds")

                # noise_pred = torch.from_numpy(outputs["output"])
                hidden_states = torch.tensor(outputs["output"])

                hidden_states = hidden_states.reshape(
                    batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                )

                hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                noise_pred = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


                if self.do_classifier_free_guidance:
                    # with current_model.cache_context("uncond"):
                    #     noise_uncond_pytorch = current_model(
                    #         hidden_states=latent_model_input,
                    #         timestep=timestep,
                    #         encoder_hidden_states=negative_prompt_embeds,
                    #         attention_kwargs=attention_kwargs,
                    #         return_dict=False,
                    #     )[0]
                    start_time = time.time()
                    outputs = self.transformer.qpc_session.run(inputs_aic2)
                    end_time = time.time()
                    print(f"Time : {end_time - start_time:.2f} seconds")

                    # noise_uncond = torch.from_numpy(outputs["output"])
                    hidden_states = torch.tensor(outputs["output"])

                    hidden_states = hidden_states.reshape(
                        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                    )

                    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                    noise_uncond = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # if callback_on_step_end is not None:
                #     callback_kwargs = {}
                #     for k in callback_on_step_end_tensor_inputs:
                #         callback_kwargs[k] = locals()[k]
                #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                #     latents = callback_outputs.pop("latents", latents)
                #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                #     negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae_decode.model.dtype)
            latents_mean = (
                torch.tensor(self.vae_decode.model.config.latents_mean)
                .view(1, self.vae_decode.model.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_decode.model.config.latents_std).view(1, self.vae_decode.model.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            # import pdb; pdb.set_trace()
            video = self.model.vae.decode(latents, return_dict=False)[0] #TODO: to enable aic with qpc self.vae_decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video.detach())
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)