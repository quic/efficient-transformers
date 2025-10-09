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
from diffusers import FluxPipeline
from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

from QEfficient.diffusers.pipelines.pipeline_utils import  QEffTextEncoder, QEffClipTextEncoder, QEffVAE, QEffFluxTransformerModel
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants

class QEFFFluxPipeline(FluxPipeline):
    _hf_auto_class = FluxPipeline
    """
    A QEfficient-optimized Flux pipeline, inheriting from `diffusers.FluxPipeline`.

    This class integrates QEfficient components (e.g., optimized models for Clip, t5 text encoders,
    flux transformer, and VAE) to enhance performance, particularly for deployment on Qualcomm AI hardware.
    It provides methods for text-to-image generation leveraging these optimized components.
    """

    def __init__(self, model, *args, **kwargs):
        self.text_encoder = QEffClipTextEncoder(model.text_encoder)
        self.text_encoder_2 = QEffTextEncoder(model.text_encoder_2)
        self.transformer = QEffFluxTransformerModel(model.transformer)
        self.vae_decode = QEffVAE(model, "decoder")

        self.tokenizer = model.tokenizer
        self.text_encoder.tokenizer = model.tokenizer
        self.text_encoder_2.tokenizer = model.tokenizer_2
        self.tokenizer_max_length = model.tokenizer_max_length
        self.scheduler = model.scheduler

        self.register_modules(
            vae=self.vae_decode,
            text_encoder= self.text_encoder,
            text_encoder_2= self.text_encoder_2,
            tokenizer= self.tokenizer ,
            tokenizer_2= self.text_encoder_2.tokenizer,
            transformer=self.transformer,
            scheduler=self.scheduler,
        )

        self.vae_decode.model.forward = lambda latent_sample, return_dict: self.vae_decode.model.decode(
            latent_sample, return_dict
        )

        self.vae_scale_factor = (
            2 ** (len(model.vae.config.block_out_channels) - 1) if getattr(model, "vae", None) else 8
        )
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        self.t_max_length = (
            model.tokenizer.model_max_length if hasattr(model, "tokenizer") and model.tokenizer is not None else 77
        )
        self.default_sample_size = 128

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        """
        Instantiate a QEffFluxTransformer2DModel from pretrained Diffusers models.

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

    def export(self, export_dir: Optional[str] = None) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        ``Optional`` Args:
           :export_dir (str, optional): The directory path to store ONNX-graph.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """

        # text_encoder - CLIP
        example_inputs_text_encoder, dynamic_axes_text_encoder, output_names_text_encoder = (
            self.text_encoder.get_onnx_config(seq_len = self.tokenizer.model_max_length)
        )
        self.text_encoder.export(
            inputs=example_inputs_text_encoder,
            output_names=output_names_text_encoder,
            dynamic_axes=dynamic_axes_text_encoder,
            export_dir=export_dir,
        )

        # text_encoder_2 - T5
        example_inputs_text_encoder_2, dynamic_axes_text_encoder_2, output_names_text_encoder_2 = (
            self.text_encoder_2.get_onnx_config(seq_len = self.text_encoder_2.tokenizer.model_max_length)
        )
        self.text_encoder_2.export(
            inputs=example_inputs_text_encoder_2,
            output_names=output_names_text_encoder_2,
            dynamic_axes=dynamic_axes_text_encoder_2,
            export_dir=export_dir,
        )

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
        seq_len: Union[int, List[int]] = 256,
        batch_size: int = 1,
        num_devices_text_encoder: int = 1,
        num_devices_transformer: int = 4,
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
        if any(
            path is None
            for path in [
                self.text_encoder.onnx_path,
                self.text_encoder_2.onnx_path,
                self.transformer.onnx_path,
                self.vae_decode.onnx_path,
            ]
        ):
            self.export()
        # text_encoder - CLIP
        specializations_text_encoder = self.text_encoder.get_specializations(
            batch_size, self.tokenizer.model_max_length
        )

        # self.text_encoder_compile_path = "<your clip qpc>"
        self.text_encoder_compile_path = self.text_encoder._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations_text_encoder,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices_text_encoder,
            aic_num_cores=num_cores,
            **compiler_options,
        )

        # text encoder 2 - T5
        specializations_text_encoder_2 = self.text_encoder_2.get_specializations(
            batch_size, seq_len
        )

        self.text_encoder_2_compile_path = self.text_encoder_2._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations_text_encoder_2,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices_text_encoder,
            aic_num_cores=num_cores,
            **compiler_options,
        )

        # transformer
        specializations_transformer = self.transformer.get_specializations(batch_size, seq_len)
        compiler_options = {"mos": 1, "mdts-mos":1}
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
        specializations_vae = self.vae_decode.get_specializations(batch_size)
        self.vae_decoder_compile_path = self.vae_decode._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations_vae,
            convert_to_fp16=True,
            mdp_ts_num_devices=num_devices_vae_decoder,
        )


    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device_ids: List[int] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Get T5 prompt embeddings for the given prompt(s).

        Args:
            prompt (Union[str, List[str]], optional): The input prompt(s) to encode.
            num_images_per_prompt (int, defaults to 1): Number of images to generate per prompt.
            max_sequence_length (int, defaults to 256): Maximum sequence length for tokenization.
            device ids (Optional[torch.device], optional): The device to place tensors on QAIC device ids.
            dtype (Optional[torch.dtype], optional): The data type for tensors.

        Returns:
            torch.Tensor: The T5 prompt embeddings with shape (batch_size * num_images_per_prompt, seq_len, hidden_size).
        """


        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        embed_dim = 4096

        text_inputs = self.text_encoder_2.tokenizer(
            prompt,
            padding="max_length",
            max_length= max_sequence_length,
            truncation= True,
            return_length= False,
            return_overflowing_tokens= False,
            return_tensors= "pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.text_encoder_2.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.text_encoder_2.tokenizer.batch_decode(untruncated_ids[:,self.text_encoder_2.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" { self.text_encoder_2.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if self.text_encoder_2.qpc_session is None:
            self.text_encoder_2.qpc_session = QAICInferenceSession(str(self.text_encoder_2_compile_path), device_ids=device_ids)

        text_encoder_2_output = {
            "last_hidden_state": np.random.rand(batch_size, max_sequence_length, embed_dim).astype(np.int32),
        }
        self.text_encoder_2.qpc_session.set_buffers(text_encoder_2_output)

        aic_text_input = {"input_ids": text_input_ids.numpy().astype(np.int64)}
        prompt_embeds = torch.tensor(self.text_encoder_2.qpc_session.run(aic_text_input)["last_hidden_state"])

        self.text_encoder_2.qpc_session.deactivate()

        # # # AIC Testing
        # prompt_embeds_pytorch = self.text_encoder_2.model(text_input_ids, output_hidden_states=False)
        # mad = torch.abs(prompt_embeds_pytorch["last_hidden_state"] - prompt_embeds).mean()
        # print(">>>>>>>>>>>> MAD for text-encoder-2 - T5 => Pytorch vs AI 100:", mad)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds


    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device_ids: List[int] = None,
    ):
        """
        Get CLIP prompt embeddings for a given text encoder and tokenizer.

        Args:
            prompt (Union[str, List[str]]): The input prompt(s) to encode.
            num_images_per_prompt (Optional[int], defaults to 1): Number of images to generate per prompt.
            device_ids (List[int], optional): List of device IDs to use for inference.

        Returns:
            - prompt_embd_text_encoder: The prompt embeddings from the text encoder.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        embed_dim = 768

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )

        if self.text_encoder.qpc_session is None:
            self.text_encoder.qpc_session = QAICInferenceSession(str(self.text_encoder_compile_path), device_ids=device_ids)

        text_encoder_output = {
            "pooler_output": np.random.rand(batch_size, embed_dim).astype(np.int32),
            "last_hidden_state": np.random.rand(batch_size, self.tokenizer_max_length, embed_dim).astype(np.int32),
        }

        self.text_encoder.qpc_session.set_buffers(text_encoder_output)

        aic_text_input = {"input_ids": text_input_ids.numpy().astype(np.int64)}
        aic_embeddings = self.text_encoder.qpc_session.run(aic_text_input)
        aic_text_encoder_emb = aic_embeddings["pooler_output"]

        self.text_encoder.qpc_session.deactivate() #To deactivate CLIP instance

        # # # [TEMP] CHECK ACC # #
        # prompt_embeds_pytorch = self.text_encoder.model(text_input_ids, output_hidden_states=False)
        # pt_pooled_embed = prompt_embeds_pytorch["pooler_output"].detach().numpy()
        # mad = np.mean(np.abs(pt_pooled_embed - aic_text_encoder_emb))
        # print(f">>>>>>>>>>>> CLIP text encoder pooled embed MAD: ", mad) ## 0.0043082903 ##TODO : Clean up
        ### END CHECK ACC ###

        # Use pooled output of CLIPTextModel
        prompt_embeds = torch.tensor(aic_embeddings["pooler_output"])
        # prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        device_ids: List[int] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
    ):
        r"""
        Encode the given prompts into text embeddings using the two text encoders (CLIP and T5).

        This method processes prompts through multiple text encoders to generate embeddings suitable
        for Flux pipeline. It handles both positive and negative prompts for
        classifier-free guidance.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device_ids=device_ids,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device_ids=device_ids,
            )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3)
        return prompt_embeds, pooled_prompt_embeds, text_ids


    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        qpc_path: str = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and
                `negative_prompt` is provided.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with `prompt` at the expense of lower image quality.

                Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to
                the [paper](https://huggingface.co/papers/2210.03142) to learn more.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.

        Examples:
            ```python
            # Basic text-to-image generation
            from QEfficient import QEFFFluxPipeline
            pipeline = QEFFFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
            pipeline.compile(num_devices_text_encoder=1, num_devices_transformer=4, num_devices_vae_decoder=1)

            generator = torch.manual_seed(42)
            # NOTE: guidance_scale <=1 is not supported
            image = pipeline("A cat holding a sign that says hello world",
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
                generator=generator).images[0]
            image.save("flux-schnell_aic.png")
            ```
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        device = 'cpu'

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.model.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        ###### AIC related changes of transformers ######
        if self.transformer.qpc_session is None:
            self.transformer.qpc_session = QAICInferenceSession(str(self.trasformer_compile_path))

        output_buffer = {
            "output": np.random.rand(
                batch_size, self.transformer.model.config.joint_attention_dim , self.transformer.model.config.in_channels
            ).astype(np.int32),
        }

        self.transformer.qpc_session.set_buffers(output_buffer)

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                temb = self.transformer.model.time_text_embed(timestep, pooled_prompt_embeds)

                adaln_emb = []
                for i in range(19):
                    f1 = self.transformer.model.transformer_blocks[i].norm1.linear(self.transformer.model.transformer_blocks[i].norm1.silu(temb)).chunk(6, dim=1)
                    f2 = self.transformer.model.transformer_blocks[i].norm1_context.linear(self.transformer.model.transformer_blocks[i].norm1_context.silu(temb)).chunk(6, dim=1)
                    adaln_emb.append(torch.cat(list(f1) + list(f2)))

                adaln_dual_emb = torch.stack(adaln_emb)

                adaln_emb = []

                for i in range(38):
                    f1 = self.transformer.model.single_transformer_blocks[i].norm.linear(self.transformer.model.single_transformer_blocks[i].norm.silu(temb)).chunk(3, dim=1)
                    adaln_emb.append(torch.cat(list(f1)))

                adaln_single_emb = torch.stack(adaln_emb)

                temp = self.transformer.model.norm_out
                adaln_out =  temp.linear(temp.silu(temb))

                timestep = timestep / 1000

                inputs_aic = {"hidden_states": latents.detach().numpy(),
                "encoder_hidden_states": prompt_embeds.detach().numpy(),
                "pooled_projections": pooled_prompt_embeds.detach().numpy(),
                "timestep": timestep.detach().numpy(),
                "img_ids": latent_image_ids.detach().numpy(),
                "txt_ids": text_ids.detach().numpy(),
                "adaln_emb": adaln_dual_emb.detach().numpy(),
                "adaln_single_emb": adaln_single_emb.detach().numpy(),
                "adaln_out": adaln_out.detach().numpy()}

                # noise_pred_torch = self.transformer.model(
                #     hidden_states=latents,
                #     encoder_hidden_states = prompt_embeds,
                #     pooled_projections=pooled_prompt_embeds,
                #     timestep=torch.tensor(timestep),
                #     img_ids = latent_image_ids,
                #     txt_ids = text_ids,
                #     adaln_emb = adaln_dual_emb,
                #     adaln_single_emb=adaln_single_emb,
                #     adaln_out = adaln_out,
                #     return_dict=False,
                # )[0]

                start_time = time.time()
                outputs = self.transformer.qpc_session.run(inputs_aic)
                end_time = time.time()
                print(f"Time : {end_time - start_time:.2f} seconds")

                noise_pred = torch.from_numpy(outputs["output"])

                # # # ###### ACCURACY TESTING #######
                # mad=np.mean(np.abs(noise_pred_torch.detach().numpy()-outputs['output']))
                # print(f">>>>>>>>> at t = {t} FLUX transfromer model MAD:{mad}")

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae_decode.model.scaling_factor) + self.vae_decode.model.shift_factor


            self.transformer.qpc_session.deactivate()

            if self.vae_decode.qpc_session is None:
                self.vae_decode.qpc_session = QAICInferenceSession(str(self.vae_decoder_compile_path))

            output_buffer = {
                "sample": np.random.rand(
                    batch_size, 3, self.vae_decode.model.config.sample_size, self.vae_decode.model.config.sample_size
                ).astype(np.int32)
            }
            self.vae_decode.qpc_session.set_buffers(output_buffer)

            inputs = {"latent_sample": latents.numpy()}
            image = self.vae_decode.qpc_session.run(inputs)
            self.vae_decode.qpc_session.deactivate()

            ###### ACCURACY TESTING #######
            # image_torch = self.vae_decode.model(latents, return_dict=False)[0]
            # mad= np.mean(np.abs(image['sample']-image_torch.detach().numpy()))
            # print(">>>>>>>>>>>> VAE mad: ",mad)

            image_tensor = torch.from_numpy(image['sample'])
            image = self.image_processor.postprocess(image_tensor, output_type=output_type)

        # Offload all models
        # self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
