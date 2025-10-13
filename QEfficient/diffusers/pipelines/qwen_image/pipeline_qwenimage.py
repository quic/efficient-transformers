# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import QwenImagePipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from QEfficient.diffusers.pipelines.pipeline_utils import (
    QEffQwenImageTransformer2DModel,
    QEffTextEncoder,
    QEffVAE,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants

class QEFFQwenImagePipeline(QwenImagePipeline):
    _hf_auto_class = QwenImagePipeline
    """
    A QEfficient-optimized QwenImage pipeline, inheriting from `diffusers.QwenImagePipeline`.

    This class integrates QEfficient components (e.g., optimized models for text encoder,
    transformer, and VAE) to enhance performance, particularly for deployment on Qualcomm AI hardware.
    It provides methods for text-to-image generation leveraging these optimized components.
    """

    def __init__(self, model, *args, **kwargs):
        self.text_encoder = QEffTextEncoder(model.text_encoder)
        self.text_encoder.tokenizer = model.tokenizer
        self.transformer = QEffQwenImageTransformer2DModel(model.transformer)
        self.vae_decode = QEffVAE(model, "decoder")

        self.tokenizer = model.tokenizer
        self.tokenizer_max_length = model.tokenizer_max_length
        self.scheduler = model.scheduler
        self.prompt_template_encode = model.prompt_template_encode
        self.prompt_template_encode_start_idx = model.prompt_template_encode_start_idx

        self.vae_decode.model.forward = lambda latent_sample, return_dict: self.vae_decode.model.decode(
            latent_sample, return_dict
        )

        self.vae_scale_factor = (
            2 ** len(model.vae.temperal_downsample) if getattr(model, "vae", None) else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = model.default_sample_size

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs
    ):
        """
        Instantiate a QEFFQwenImagePipeline from pretrained Diffusers models.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                The path to the pretrained model or its name.
            **kwargs (additional keyword arguments):
                Additional arguments that can be passed to the underlying `QwenImagePipeline.from_pretrained`
                method.
        """
        model = cls._hf_auto_class.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float32,
            **kwargs,
        )
        model.to("cpu")
        return cls(
            model=model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
        )

    def export(self, export_dir: Optional[str] = None) -> str:
        """
        Exports the model to ``ONNX`` format using ``torch.onnx.export``.

        ``Optional`` Args:
           :export_dir (str, optional): The directory path to store ONNX-graph.

        Returns:
            :str: Path of the generated ``ONNX`` graph.
        """

        # text_encoder (Qwen2_5_VL)
        # example_inputs_text_encoder, dynamic_axes_text_encoder, output_names_text_encoder = (
        #     self.text_encoder.get_onnx_config()
        # )

        
        # self.text_encoder.export(
        #     inputs=example_inputs_text_encoder,
        #     output_names=output_names_text_encoder,
        #     dynamic_axes=dynamic_axes_text_encoder,
        #     export_dir=export_dir,
        # )

        # transformer
        example_inputs_transformer, dynamic_axes_transformer, output_names_transformer = (
            self.transformer.get_onnx_config()
        )
        
        self.transformer.export(
            inputs=example_inputs_transformer,
            output_names=output_names_transformer,
            dynamic_axes=dynamic_axes_transformer,
            export_dir=export_dir,
        )
        print("Exported transformers")
        # vae
        # example_inputs_vae, dynamic_axes_vae, output_names_vae = self.vae_decode.get_onnx_config()

        # # Update VAE config for QwenImage (5D tensor with temporal dimension)
        # example_inputs_vae = {
        #     "latent_sample": torch.randn(
        #         constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
        #         self.vae_decode.model.config.z_dim,
        #         1,
        #         64,
        #         64,
        #     ),
        #     "return_dict": False,
        # }
        # dynamic_axes_vae = {
        #     "latent_sample": {0: "batch_size", 2: "frames", 3: "height", 4: "width"},
        # }

        # self.vae_decoder_onnx_path = self.vae_decode.export(
        #     example_inputs_vae,
        #     output_names_vae,
        #     dynamic_axes_vae,
        #     export_dir=export_dir,
        # )
        
        # print("Exported VAE Decode")

    def compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        batch_size: int = 1,
        num_devices_text_encoder: int = 1,
        num_devices_transformer: int = 4,
        num_devices_vae_decoder: int = 1,
        num_cores: int = 16,
        mxfp6_matmul: bool = False,
        **compiler_options,
    ) -> str:
        """
        Compiles the ONNX graphs of the different model components for deployment on Qualcomm AI hardware.

        This method takes the ONNX paths of the text encoder, transformer, and VAE decoder,
        and compiles them into an optimized format for inference.

        Args:
            onnx_path (`str`, *optional*):
                The base directory where ONNX files were exported.
            compile_dir (`str`, *optional*):
                The directory path to store the compiled artifacts.
            batch_size (`int`, *optional*, defaults to 1):
                The batch size to use for compilation.
            num_devices_text_encoder (`int`, *optional*, defaults to 1):
                The number of AI devices to deploy the text encoder model on.
            num_devices_transformer (`int`, *optional*, defaults to 4):
                The number of AI devices to deploy the transformer model on.
            num_devices_vae_decoder (`int`, *optional*, defaults to 1):
                The number of AI devices to deploy the VAE decoder model on.
            num_cores (`int`, *optional*, defaults to 16):
                The number of cores to use for compilation.
            mxfp6_matmul (`bool`, *optional*, defaults to `False`):
                If `True`, enables mixed-precision floating-point 6-bit matrix multiplication
                optimization during compilation.
            **compiler_options:
                Additional keyword arguments to pass to the underlying compiler.

        Returns:
            `str`: A message indicating the compilation status or path to compiled artifacts.
        """
        if any(
            path is None
            for path in [
                self.text_encoder.onnx_path,
                self.transformer.onnx_path,
                self.vae_decode.onnx_path,
            ]
        ):
            self.export()

        # # text_encoder
        # specializations_text_encoder = self.text_encoder.get_specializations(
        #     batch_size, self.tokenizer_max_length
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

        # transformer - calculate latent sequence length based on default image size
        # height = self.default_sample_size * self.vae_scale_factor
        # width = self.default_sample_size * self.vae_scale_factor
        # latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        # latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        # latent_seq_len = (latent_height // 2) * (latent_width // 2)
        latent_seq_len=6032
        batch_size=1
        text_seq_len=126
        
        specializations = [
            {
                "batch_size": batch_size,
                "latent_seq_len": latent_seq_len,
                "text_seq_len": text_seq_len,
            }
        ]
        
        # specializations_transformer = self.transformer.get_specializations(
        #     batch_size, latent_seq_len, 512
        # )

        compiler_options_transformer = {"mos": 1, "ols": 2,"mdts-mos":1}
        self.transformer_compile_path = self.transformer._compile(
            onnx_path,
            compile_dir,
            compile_only=True,
            specializations=specializations,
            convert_to_fp16=True,
            mxfp6_matmul=mxfp6_matmul,
            mdp_ts_num_devices=num_devices_transformer,
            aic_num_cores=num_cores,
            **compiler_options_transformer,
        )

        # # vae
        # specializations_vae = self.vae_decode.get_specializations(batch_size)

        # self.vae_decoder_compile_path = self.vae_decode._compile(
        #     onnx_path,
        #     compile_dir,
        #     compile_only=True,
        #     specializations=specializations_vae,
        #     convert_to_fp16=True,
        #     mdp_ts_num_devices=num_devices_vae_decoder,
        # )

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """Extract hidden states based on attention mask."""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        device_ids: List[int] = None,
    ):
        """
        Get Qwen prompt embeddings for the given prompt(s) using QAICInferenceSession.

        Args:
            prompt (Union[str, List[str]], optional): The input prompt(s) to encode.
            device (Optional[torch.device], optional): The device to place tensors on.
            dtype (Optional[torch.dtype], optional): The data type for tensors.
            device_ids (List[int], optional): List of device IDs to use for inference.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The prompt embeddings and attention mask.
        """
        device = device or "cpu"
        dtype = dtype or torch.float32

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        )
        
        # HACK: Currently working on Pytorch
        encoder_hidden_states = self.text_encoder.model(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        
        
        # if self.text_encoder.qpc_session is None:
        #     self.text_encoder.qpc_session = QAICInferenceSession(self.text_encoder.qpc_path, device_ids=device_ids)

        # aic_text_input = {
        #     "input_ids": txt_tokens.input_ids.numpy().astype(np.int64),
        #     "attention_mask": txt_tokens.attention_mask.numpy().astype(np.int64),
        # }
        # aic_embeddings = self.text_encoder.qpc_session.run(aic_text_input)
        # hidden_states = torch.tensor(aic_embeddings["last_hidden_state"])

        
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
        device_ids: List[int] = None,
    ):
        """
        Encode the given prompts into text embeddings using the Qwen text encoder.

        Args:
            prompt (Union[str, List[str]]): The prompt(s) to encode.
            device (Optional[torch.device], optional): The device to place tensors on.
            num_images_per_prompt (int, defaults to 1): Number of images to generate per prompt.
            prompt_embeds (Optional[torch.Tensor], optional): Pre-computed prompt embeddings.
            prompt_embeds_mask (Optional[torch.Tensor], optional): Pre-computed prompt embeddings mask.
            max_sequence_length (int, defaults to 1024): Maximum sequence length for tokenization.
            device_ids (List[int], optional): List of device IDs to use for inference.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The prompt embeddings and attention mask.
        """
        device = device or "cpu"

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, device, device_ids=device_ids)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        """
        Generate images from text prompts using the QEfficient-optimized QwenImage pipeline.

        This method performs text-to-image generation by encoding the input prompts through the
        Qwen text encoder, running the diffusion process with the transformer model, and decoding
        the final latents to images using the VAE decoder. All components are optimized for
        Qualcomm AI hardware.

        Args:
            prompt (Union[str, List[str]], optional): The text prompt(s) to guide image generation.
            negative_prompt (Union[str, List[str]], optional): Negative prompt(s) for true CFG.
            true_cfg_scale (float, defaults to 4.0): Scale for true classifier-free guidance.
            height (Optional[int], optional): Height of the generated image in pixels.
            width (Optional[int], optional): Width of the generated image in pixels.
            num_inference_steps (int, defaults to 50): Number of denoising steps.
            sigmas (Optional[List[float]], optional): Custom sigmas for the denoising process.
            guidance_scale (float, defaults to 1.0): Guidance scale (for future guidance-distilled models).
            num_images_per_prompt (int, defaults to 1): Number of images to generate per prompt.
            generator (Optional[Union[torch.Generator, List[torch.Generator]]], optional): Random generator(s).
            latents (Optional[torch.Tensor], optional): Pre-generated noisy latents.
            prompt_embeds (Optional[torch.Tensor], optional): Pre-generated text embeddings.
            prompt_embeds_mask (Optional[torch.Tensor], optional): Pre-generated text embeddings mask.
            negative_prompt_embeds (Optional[torch.Tensor], optional): Pre-generated negative text embeddings.
            negative_prompt_embeds_mask (Optional[torch.Tensor], optional): Pre-generated negative text embeddings mask.
            output_type (Optional[str], defaults to "pil"): Output format ("pil", "np", "pt", or "latent").
            return_dict (bool, defaults to True): Whether to return a QwenImagePipelineOutput.
            attention_kwargs (Optional[Dict[str, Any]], optional): Additional attention kwargs.
            callback_on_step_end (Optional[Callable], optional): Callback function at end of each step.
            callback_on_step_end_tensor_inputs (List[str], defaults to ["latents"]): Tensor inputs for callback.
            max_sequence_length (int, defaults to 512): Maximum sequence length for text encoder.

        Returns:
            Union[QwenImagePipelineOutput, Tuple]: Generated images.

        Examples:
            ```python
            from QEfficient import QEFFQwenImagePipeline

            pipeline = QEFFQwenImagePipeline.from_pretrained("Qwen/Qwen-Image")
            pipeline.compile(num_devices_text_encoder=1, num_devices_transformer=4, num_devices_vae_decoder=1)

            image = pipeline("A cat holding a sign that says hello world", num_inference_steps=50).images[0]
            image.save("qwenimage.png")
            ```
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        device = "cpu"

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.model.config.in_channels // 4
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
        img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift

        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.model.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        # Initialize transformer session
        if self.transformer.qpc_session is None:
            self.transformer.qpc_session = QAICInferenceSession(str(self.transformer.qpc_path))

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = (t.expand(latents.shape[0]) / 1000).numpy().astype(np.float32)

                # Conditional pass
                transformer_inputs = {
                    "hidden_states": latents.numpy().astype(np.float32),
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds.numpy().astype(np.float32),
                    "encoder_hidden_states_mask": prompt_embeds_mask.numpy().astype(np.int64),
                }
                if guidance is not None:
                    transformer_inputs["guidance"] = guidance.numpy().astype(np.float32)

                noise_pred = self.transformer.qpc_session.run(transformer_inputs)
                noise_pred = torch.tensor(noise_pred["output"])

                if do_true_cfg:
                    # Unconditional pass
                    transformer_inputs_uncond = {
                        "hidden_states": latents.numpy().astype(np.float32),
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds.numpy().astype(np.float32),
                        "encoder_hidden_states_mask": negative_prompt_embeds_mask.numpy().astype(np.int64),
                    }
                    if guidance is not None:
                        transformer_inputs_uncond["guidance"] = guidance.numpy().astype(np.float32)

                    neg_noise_pred = self.transformer.qpc_session.run(transformer_inputs_uncond)
                    neg_noise_pred = torch.tensor(neg_noise_pred["output"])

                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae_decode.model.dtype)
            latents_mean = (
                torch.tensor(self.vae_decode.model.config.latents_mean)
                .view(1, self.vae_decode.model.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_decode.model.config.latents_std).view(
                1, self.vae_decode.model.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

            # VAE decode
            vae_session = QAICInferenceSession(str(self.vae_decoder_compile_path))
            inputs = {"latent_sample": latents.numpy().astype(np.float32)}
            image = vae_session.run(inputs)
            image = self.image_processor.postprocess(torch.tensor(image["sample"])[:, :, 0], output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
