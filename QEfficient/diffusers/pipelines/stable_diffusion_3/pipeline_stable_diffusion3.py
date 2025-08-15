# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from typing import Any, Callable, Dict, List, Optional, Union
from venv import logger

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

from QEfficient.diffusers.pipelines.pipeline_utils import QEffSD3Transformer2DModel, QEffTextEncoder, QEffVAE
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants


class QEFFStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    _hf_auto_class = StableDiffusion3Pipeline
    """
    A QEfficient-optimized Stable Diffusion 3 pipeline, inheriting from `diffusers.StableDiffusion3Pipeline`.

    This class integrates QEfficient components (e.g., optimized models for text encoder,
    transformer, and VAE) to enhance performance, particularly for deployment on Qualcomm AI hardware.
    It provides methods for text-to-image generation leveraging these optimized components.
    """

    def __init__(self, model, *args, **kwargs):
        self.text_encoder = QEffTextEncoder(model.text_encoder)
        self.text_encoder_2 = QEffTextEncoder(model.text_encoder_2)
        self.text_encoder_3 = QEffTextEncoder(model.text_encoder_3)
        self.transformer = QEffSD3Transformer2DModel(model.transformer)
        self.vae_decode = QEffVAE(model, "decoder")

        self.tokenizer = model.tokenizer
        self.text_encoder.tokenizer = model.tokenizer
        self.text_encoder_2.tokenizer = model.tokenizer_2
        self.text_encoder_3.tokenizer = model.tokenizer_3
        self.tokenizer_max_length = model.tokenizer_max_length
        self.scheduler = model.scheduler

        self.vae_decode.model.forward = lambda latent_sample, return_dict: self.vae_decode.model.decode(
            latent_sample, return_dict
        )

        self.vae_scale_factor = (
            2 ** (len(model.vae.config.block_out_channels) - 1) if getattr(model, "vae", None) else 8
        )
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
            model.transformer.config.patch_size
            if hasattr(model, "transformer") and model.transformer is not None
            else 2
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        """
        Instantiate a QEFFStableDiffusion3Pipeline from pretrained Diffusers models.

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

        # text_encoder
        example_inputs_text_encoder, dynamic_axes_text_encoder, output_names_text_encoder = (
            self.text_encoder.get_onnx_config()
        )

        for i in range(0, 13):
            output_names_text_encoder.append("hidden_states_" + str(i))
        self.text_encoder.export(
            inputs=example_inputs_text_encoder,
            output_names=output_names_text_encoder,
            dynamic_axes=dynamic_axes_text_encoder,
            export_dir=export_dir,
        )

        # text_encoder_2
        example_inputs_text_encoder_2, dynamic_axes_text_encoder_2, output_names_text_encoder_2 = (
            self.text_encoder_2.get_onnx_config()
        )

        for i in range(0, 33):
            output_names_text_encoder_2.append("hidden_states_" + str(i))

        self.text_encoder_2.export(
            inputs=example_inputs_text_encoder_2,
            output_names=output_names_text_encoder_2,
            dynamic_axes=dynamic_axes_text_encoder_2,
            export_dir=export_dir,
        )

        # t5_text_encoder
        example_inputs_text_encoder_3, dynamic_axes_text_encoder_3, output_names_text_encoder_3 = (
            self.text_encoder_3.get_onnx_config()
        )

        with torch.no_grad():
            prev_sf = 1
            for i in range(len(self.text_encoder_3.model.encoder.block)):
                wosf = constants.WO_SFS[i]
                self.text_encoder_3.model.encoder.block[i].layer[0].SelfAttention.o.weight *= 1 / wosf
                self.text_encoder_3.model.encoder.block[i].layer[0].scaling_factor *= prev_sf / wosf
                self.text_encoder_3.model.encoder.block[i].layer[1].DenseReluDense.wo.weight *= 1 / wosf
                prev_sf = wosf

        self.text_encoder_3.export(
            inputs=example_inputs_text_encoder_3,
            output_names=output_names_text_encoder_3,
            dynamic_axes=dynamic_axes_text_encoder_3,
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
        seq_len: Union[int, List[int]] = 32,
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
                self.text_encoder_3.onnx_path,
                self.transformer.onnx_path,
                self.vae_decode.onnx_path,
            ]
        ):
            self.export()

        # text_encoder
        specializations_text_encoder = self.text_encoder.get_specializations(
            batch_size, self.tokenizer.model_max_length
        )

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

        # text encoder 2
        specializations_text_encoder_2 = self.text_encoder_2.get_specializations(
            batch_size, self.tokenizer.model_max_length
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

        # text_encoder 3
        specializations_text_encoder_3 = self.text_encoder_3.get_specializations(batch_size, 256)

        self.text_encoder_3_compile_path = self.text_encoder_3._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations_text_encoder_3,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices_text_encoder,
            aic_num_cores=num_cores,
            **compiler_options,
        )

        # transformer
        specializations_transformer = self.transformer.get_specializations(batch_size, 333)

        compiler_options = {"mos": 1, "ols": 2}
        self.trasformers_compile_path = self.transformer._compile(
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

    def _get_clip_prompt_embeds(
        self,
        text_encoder,
        tokenizer,
        clip_index: bool,
        prompt: Union[str, List[str]],
        num_images_per_prompt: Optional[int] = 1,
        clip_skip: Optional[int] = None,
        device_ids: List[int] = None,
    ):
        """
        Get CLIP prompt embeddings for a given text encoder and tokenizer.

        Args:
            text_encoder: The QEffTextEncoder instance to use for encoding.
            tokenizer: The tokenizer to use for text preprocessing.
            clip_index (int): Index of the CLIP model (0 or 1) to determine embedding dimensions and hidden state range.
            prompt (Union[str, List[str]]): The input prompt(s) to encode.
            num_images_per_prompt (Optional[int], defaults to 1): Number of images to generate per prompt.
            clip_skip (Optional[int], optional): Number of layers to skip from the end when extracting hidden states.
            device_ids (List[int], optional): List of device IDs to use for inference.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - prompt_embd_text_encoder: The prompt embeddings from the text encoder.
                - pooled_prompt_embeds_text_encoder: The pooled prompt embeddings.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # to pick correct hidden_state range for each clip model
        hidden_state_range = 33 if clip_index else 13

        # choose embed_dim based on the clip model index.
        embed_dim = 1280 if clip_index else 768

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
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

        if text_encoder.qpc_session is None:
            text_encoder.qpc_session = QAICInferenceSession(text_encoder.qpc_path, device_ids=device_ids)

        text_encoder_output = {
            "pooler_output": np.random.rand(batch_size, embed_dim).astype(np.int32),
            "last_hidden_state": np.random.rand(batch_size, self.tokenizer_max_length, embed_dim).astype(np.int32),
        }

        for i in range(0, hidden_state_range):
            text_encoder_output[f"hidden_states_{i}"] = np.random.rand(
                batch_size, self.tokenizer_max_length, embed_dim
            ).astype(np.int32)
        text_encoder.qpc_session.set_buffers(text_encoder_output)

        aic_text_input = {"input_ids": text_input_ids.numpy().astype(np.int64)}
        aic_embeddings = text_encoder.qpc_session.run(aic_text_input)
        aic_text_encoder_emb = aic_embeddings["pooler_output"]

        ## [TEMP] CHECK ACC ##
        # prompt_embeds_pytorch = text_encoder.model(text_input_ids, output_hidden_states=True)
        # pt_pooled_embed = prompt_embeds_pytorch[0].detach().numpy()
        # mad = np.mean(np.abs(pt_pooled_embed - aic_text_encoder_emb))
        # print(f"CLIP text encoder {clip_index}- pooled embed MAD: ", mad)
        ### END CHECK ACC ##

        pooled_prompt_embeds = torch.tensor(aic_text_encoder_emb)
        if clip_skip is None:
            prompt_embd_text_encoder = torch.tensor(aic_embeddings[list(aic_embeddings.keys())[-2]])
        else:
            prompt_embd_text_encoder = torch.tensor(aic_embeddings[list(aic_embeddings.keys())[-(clip_skip + 2)]])
        _, seq_len, _ = prompt_embd_text_encoder.shape

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embd_text_encoder = prompt_embd_text_encoder.repeat(1, num_images_per_prompt, 1)
        prompt_embd_text_encoder = prompt_embd_text_encoder.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds_text_encoder = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds_text_encoder = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embd_text_encoder, pooled_prompt_embeds_text_encoder

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Get T5 prompt embeddings for the given prompt(s).

        Args:
            prompt (Union[str, List[str]], optional): The input prompt(s) to encode.
            num_images_per_prompt (int, defaults to 1): Number of images to generate per prompt.
            max_sequence_length (int, defaults to 256): Maximum sequence length for tokenization.
            device (Optional[torch.device], optional): The device to place tensors on.
            dtype (Optional[torch.dtype], optional): The data type for tensors.

        Returns:
            torch.Tensor: The T5 prompt embeddings with shape (batch_size * num_images_per_prompt, seq_len, hidden_size).
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.text_encoder_3.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.text_encoder_3.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.text_encoder_3.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )
        if self.text_encoder_3.qpc_session is None:
            self.text_encoder_3.qpc_session = QAICInferenceSession(str(self.text_encoder_3_compile_path))

        aic_text_input = {"input_ids": text_input_ids.numpy().astype(np.int64)}
        prompt_embeds = torch.tensor(self.text_encoder_3.qpc_session.run(aic_text_input)["last_hidden_state"])

        # AIC Testing
        # prompt_embeds_torch = self.text_encoder_3.model(text_input_ids.to(device))[0]
        # mad = torch.abs(prompt_embeds - aic_embeddings).mean()
        # print("Clip text-encoder-3 Pytorch vs AI 100:", mad)

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
        device_ids: List[int] = None,
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
    ):
        """
        Encode the given prompts into text embeddings using the three text encoders (CLIP and T5).

        This method processes prompts through multiple text encoders to generate embeddings suitable
        for Stable Diffusion 3 generation. It handles both positive and negative prompts for
        classifier-free guidance.

        Args:
            prompt (Union[str, List[str]]): The primary prompt(s) to encode.
            prompt_2 (Union[str, List[str]]): The secondary prompt(s) for the second CLIP encoder.
            prompt_3 (Union[str, List[str]]): The tertiary prompt(s) for the T5 encoder.
            device_ids (List[int], optional): List of device IDs to use for inference.
            num_images_per_prompt (int, defaults to 1): Number of images to generate per prompt.
            do_classifier_free_guidance (bool, defaults to True): Whether to use classifier-free guidance.
            negative_prompt (Optional[Union[str, List[str]]], optional): The negative prompt(s) to encode.
            negative_prompt_2 (Optional[Union[str, List[str]]], optional): The negative prompt(s) for the second CLIP encoder.
            negative_prompt_3 (Optional[Union[str, List[str]]], optional): The negative prompt(s) for the T5 encoder.
            prompt_embeds (Optional[torch.FloatTensor], optional): Pre-computed prompt embeddings.
            negative_prompt_embeds (Optional[torch.FloatTensor], optional): Pre-computed negative prompt embeddings.
            pooled_prompt_embeds (Optional[torch.FloatTensor], optional): Pre-computed pooled prompt embeddings.
            negative_pooled_prompt_embeds (Optional[torch.FloatTensor], optional): Pre-computed negative pooled prompt embeddings.
            clip_skip (Optional[int], optional): Number of layers to skip from the end when extracting CLIP hidden states.
            max_sequence_length (int, defaults to 256): Maximum sequence length for T5 tokenization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - prompt_embeds: The combined prompt embeddings from all encoders.
                - negative_prompt_embeds: The combined negative prompt embeddings (if classifier-free guidance is enabled).
                - pooled_prompt_embeds: The pooled prompt embeddings from CLIP encoders.
                - negative_pooled_prompt_embeds: The pooled negative prompt embeddings (if classifier-free guidance is enabled).
        """
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
                self.text_encoder,
                self.text_encoder.tokenizer,
                clip_index=0,
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                device_ids=device_ids,
            )

            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                self.text_encoder_2,
                self.text_encoder_2.tokenizer,
                clip_index=1,
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                device_ids=device_ids,
            )

            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

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
                self.text_encoder,
                self.text_encoder.tokenizer,
                clip_index=0,
                prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                device_ids=device_ids,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                self.text_encoder_2,
                self.text_encoder_2.tokenizer,
                clip_index=1,
                prompt=negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                device_ids=device_ids,
            )

            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)

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
    ):
        """
        Generate images from text prompts using the QEfficient-optimized Stable Diffusion 3 pipeline.

        This method performs text-to-image generation by encoding the input prompts through multiple
        text encoders, running the diffusion process with the transformer model, and decoding the
        final latents to images using the VAE decoder. All components are optimized for Qualcomm AI hardware.

        Args:
            prompt (Union[str, List[str]], optional): The primary text prompt(s) to guide image generation.
            prompt_2 (Optional[Union[str, List[str]]], optional): Secondary prompt(s) for the second CLIP encoder.
                If None, defaults to `prompt`.
            prompt_3 (Optional[Union[str, List[str]]], optional): Tertiary prompt(s) for the T5 encoder.
                If None, defaults to `prompt`.
            height (Optional[int], optional): Height of the generated image in pixels. If None, uses default
                sample size scaled by VAE scale factor.
            width (Optional[int], optional): Width of the generated image in pixels. If None, uses default
                sample size scaled by VAE scale factor.
            num_inference_steps (int, defaults to 28): Number of denoising steps during generation.
            timesteps (List[int], optional): Custom timesteps to use for denoising. If provided, overrides
                `num_inference_steps`.
            guidance_scale (float, defaults to 7.0): Guidance scale for classifier-free guidance. Higher values
                result in images more closely linked to the prompt at the expense of lower image quality.
            negative_prompt (Optional[Union[str, List[str]]], optional): Negative prompt(s) to guide what not
                to include in image generation.
            negative_prompt_2 (Optional[Union[str, List[str]]], optional): Negative prompt(s) for the second
                CLIP encoder.
            negative_prompt_3 (Optional[Union[str, List[str]]], optional): Negative prompt(s) for the T5 encoder.
            num_images_per_prompt (Optional[int], defaults to 1): Number of images to generate per prompt.
            generator (Optional[Union[torch.Generator, List[torch.Generator]]], optional): Random number
                generator(s) for reproducible generation.
            latents (Optional[torch.FloatTensor], optional): Pre-generated noisy latents sampled from a Gaussian
                distribution to be used as inputs for image generation.
            prompt_embeds (Optional[torch.FloatTensor], optional): Pre-generated text embeddings. Can be used
                to easily tweak text inputs (prompt weighting).
            negative_prompt_embeds (Optional[torch.FloatTensor], optional): Pre-generated negative text embeddings.
            pooled_prompt_embeds (Optional[torch.FloatTensor], optional): Pre-generated pooled text embeddings.
            negative_pooled_prompt_embeds (Optional[torch.FloatTensor], optional): Pre-generated negative pooled
                text embeddings.
            output_type (Optional[str], defaults to "pil"): Output format of the generated images. Choose between
                "pil", "np", "pt", or "latent".
            return_dict (bool, defaults to True): Whether to return a `StableDiffusion3PipelineOutput` instead
                of a plain tuple.
            joint_attention_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to pass
                to the joint attention layers.
            clip_skip (Optional[int], optional): Number of layers to skip from the end when extracting CLIP
                hidden states.
            callback_on_step_end (Optional[Callable[[int, int, Dict], None]], optional): Callback function
                called at the end of each denoising step.
            callback_on_step_end_tensor_inputs (List[str], defaults to ["latents"]): List of tensor inputs
                to pass to the callback function.
            max_sequence_length (int, defaults to 256): Maximum sequence length for T5 text encoder tokenization.

        Returns:
            Union[StableDiffusion3PipelineOutput, Tuple]: If `return_dict` is True, returns a
            `StableDiffusion3PipelineOutput` object containing the generated images. Otherwise,
            returns a tuple with the generated images.

        Examples:
            ```python
            # Basic text-to-image generation
            from QEfficient import QEFFStableDiffusion3Pipeline

            pipeline = QEFFStableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large")
            pipeline.compile(num_devices_text_encoder=1, num_devices_transformer=4, num_devices_vae_decoder=1)

            # NOTE: guidance_scale <=1 is not supported
            image = pipeline("A girl laughing", num_inference_steps=28, guidance_scale=2.0).images[0]
            image.save("girl_laughing.png")
            ```
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        device = "cpu"

        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

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

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.model.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        ###### AIC related changes of transformers ######
        if self.transformer.qpc_session is None:
            self.transformer.qpc_session = QAICInferenceSession(str(self.transformer.qpc_path))

            output_buffer = {
                "output": np.random.rand(
                    2 * batch_size, num_channels_latents, self.default_sample_size, self.default_sample_size
                ).astype(np.int32),
            }

            self.transformer.qpc_session.set_buffers(output_buffer)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                timestep = np.array([t], dtype=np.int64)

                # noise_pred_torch = self.transformer.model(
                #     hidden_states=latent_model_input,
                #     timestep=torch.tensor(timestep),
                #     encoder_hidden_states=prompt_embeds,
                #     pooled_projections=pooled_prompt_embeds,
                #     joint_attention_kwargs=self.joint_attention_kwargs,
                #     return_dict=False,
                # )[0]

                noise_pred = self.transformer.qpc_session.run(
                    {
                        "encoder_hidden_states": prompt_embeds.detach().numpy(),
                        "pooled_projections": pooled_prompt_embeds.numpy(),
                        "timestep": timestep,
                        "hidden_states": latent_model_input.numpy(),
                    }
                )

                # ###### ACCURACY TESTING #######
                # mad=np.mean(np.abs(noise_pred_torch.detach().numpy()-noise_pred['output']))
                # print("transfromer model MAD:", mad)

                noise_pred = torch.tensor(noise_pred["output"])

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

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
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = (
                latents / self.vae_decode.model.config.scaling_factor
            ) + self.vae_decode.model.config.shift_factor

            # image_torch = self.vae_decode.model(latents, return_dict=False)[0]

            vae_session = QAICInferenceSession(str(self.vae_decoder_compile_path))

            output_buffer = {
                "sample": np.random.rand(
                    batch_size, 3, self.vae_decode.model.config.sample_size, self.vae_decode.model.config.sample_size
                ).astype(np.int32)
            }

            vae_session.set_buffers(output_buffer)
            inputs = {"latent_sample": latents.numpy()}
            image = vae_session.run(inputs)
            # mad= np.mean(np.abs(image['sample']-image_torch.detach().numpy()))
            # print("VAE mad: ",mad)
            image = self.image_processor.postprocess(torch.tensor(image["sample"]), output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
