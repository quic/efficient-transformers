# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""
QEfficient QWEN image Pipeline Implementation

This module provides an optimized implementation of QWEN image pipeline
for high-performance text-to-image generation on Qualcomm AI hardware.

TODO: 1. Update Qwen text encoder, Vae decoder to Qaic; present running on cpu
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import QwenImagePipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from tqdm import tqdm

from QEfficient.diffusers.pipelines.pipeline_module import (
    QEffQwenImageTransformer2DModel,
    QEffVAE,
)
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ONNX_SUBFUNCTION_MODULE,
    ModulePerf,
    QEffPipelineOutput,
    calculate_compressed_latent_dimension,
    compile_modules_parallel,
    compile_modules_sequential,
    config_manager,
    set_execute_params,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants


class QEFFQwenImagePipeline(QwenImagePipeline):
    """
    #TODO : update docs
    A QEfficient-optimized QwenImage pipeline, inheriting from `diffusers.QwenImagePipeline`.

    This class integrates QEfficient components (e.g., optimized models for text encoder,
    transformer, and VAE) to enhance performance, particularly for deployment on Qualcomm AI hardware.
    It provides methods for text-to-image generation leveraging these optimized components.
    """

    _hf_auto_class = QwenImagePipeline

    def __init__(self, model, **kwargs):
        "#TODO : update docs"
        self.model = model
        self.kwargs = kwargs

        self.text_encoder = model.text_encoder  # TODO: Text encoder on QAIC
        self.transformer = QEffQwenImageTransformer2DModel(model.transformer)
        self.vae_decoder = QEffVAE(model.vae, "decoder")  # TODO make as vae_decoder

        # Store all modules in a dictionary for easy iteration during export/compile
        self.modules = {
            "transformer": self.transformer,
            "vae_decoder": self.vae_decoder,
        }

        # Copy tokenizers and scheduler from the original model
        self.tokenizer = model.tokenizer
        self.text_encoder.tokenizer = model.tokenizer  # TODO check and remove if not utlising it
        self.tokenizer_max_length = model.tokenizer_max_length
        self.scheduler = model.scheduler

        self.prompt_template_encode = model.prompt_template_encode
        self.prompt_template_encode_start_idx = model.prompt_template_encode_start_idx

        self.vae_decoder.model.forward = lambda latent_sample, return_dict: self.vae_decoder.model.decode(
            latent_sample, return_dict
        )

        self.vae_scale_factor = 2 ** len(model.vae.temperal_downsample) if getattr(model, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = model.default_sample_size

        self.vae_decoder.get_onnx_params = self.vae_decoder.get_video_onnx_params

    @property
    def do_classifier_free_guidance(self):
        """
        Determine if classifier-free guidance should be used.

        Returns:
            bool: True if CFG should be applied based on current guidance scales
        """
        return self._guidance_scale > 1.0 and (self._guidance_scale_2 is None or self._guidance_scale_2 > 1.0)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
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
            device_map="cpu",
            **kwargs,
        )
        return cls(
            model=model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs,
        )

    def export(
        self,
        export_dir: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        """
        Export all pipeline modules to ONNX format for deployment preparation.

        This method systematically exports the unified transformer to ONNX format with
        video-specific configurations including temporal dimensions, dynamic axes, and
        optimization settings. The export process prepares the model for subsequent
        compilation to QPC format for efficient inference on QAIC hardware.

        Args:
            export_dir (str, optional): Target directory for saving ONNX model files. If None,
                uses the default export directory structure. The directory will be created
                if it doesn't exist.
            use_onnx_subfunctions (bool, default=False): Whether to enable ONNX subfunction
                optimization for supported modules. This can optimize the graph structure
                and improve compilation efficiency for complex models like the transformer.

        Returns:
            str: Absolute path to the export directory containing all ONNX model files.

        Raises:
            RuntimeError: If ONNX export fails for any module
            OSError: If there are issues creating the export directory or writing files
            ValueError: If module configurations are invalid

        Example:
            >>> pipeline = QEffWanPipeline.from_pretrained("path/to/wan/model")
            >>> export_path = pipeline.export(
            ...     export_dir="/path/to/export",
            ...     use_onnx_subfunctions=True
            ... )
        """
        # Export each module with video-specific parameters
        for module_name, module_obj in tqdm(self.modules.items(), desc="Exporting modules", unit="module"):
            # Get ONNX export configuration with video dimensions
            example_inputs, dynamic_axes, output_names = module_obj.get_onnx_params()

            # Prepare export parameters
            export_params = {
                "inputs": example_inputs,
                "output_names": output_names,
                "dynamic_axes": dynamic_axes,
                "export_dir": export_dir,
            }

            # Enable ONNX subfunctions for supported modules if requested
            if use_onnx_subfunctions and module_name in ONNX_SUBFUNCTION_MODULE:
                export_params["use_onnx_subfunctions"] = True

            if module_obj.qpc_path is None:
                module_obj.export(**export_params)

    @staticmethod
    def get_default_config_path():
        """
        Get the default configuration file path for WAN pipeline.

        Returns:
            str: Path to the default WAN configuration JSON file.
        """
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/qwen_config.json")

    def compile(
        self,
        compile_config: Optional[str] = None,
        parallel: bool = False,
        height: int = constants.WAN_ONNX_EXPORT_HEIGHT_180P,
        width: int = constants.WAN_ONNX_EXPORT_WIDTH_180P,
        use_onnx_subfunctions: bool = False,
    ) -> str:
        """
        Compiles the ONNX graphs of the different model components for deployment on Qualcomm AI hardware.

        This method takes the ONNX paths of the transformer and compiles them into an optimized format
        for inference using JSON-based configuration.

        Args:
            compile_config (str, optional): Path to a JSON configuration file containing
                compilation settings, device mappings, and optimization parameters. If None,
                uses the default configuration.
            parallel (bool, default=False): Compilation mode selection:
                - True: Compile modules in parallel using ThreadPoolExecutor for faster processing
                - False: Compile modules sequentially for lower resource usage
            height (int, default=192): Target image height in pixels.
            width (int, default=320): Target image width in pixels.
            use_onnx_subfunctions (bool, default=False): Whether to export models with ONNX
                subfunctions before compilation if not already exported.

        Raises:
            RuntimeError: If compilation fails for any module or if QAIC compiler is not available
            FileNotFoundError: If ONNX models haven't been exported or config file is missing
            ValueError: If configuration parameters are invalid
            OSError: If there are issues with file I/O during compilation

        Example:
            >>> pipeline = QEffWanPipeline.from_pretrained("path/to/wan/model")
            >>> # Sequential compilation with default config
            >>> pipeline.compile(height=480, width=832, num_frames=81)
            >>>
            >>> # Parallel compilation with custom config
            >>> pipeline.compile(
            ...     compile_config="/path/to/custom_config.json",
            ...     parallel=True,
            ...     height=480,
            ...     width=832,
            ... )
        """
        # Load compilation configuration
        config_manager(self, config_source=compile_config, use_onnx_subfunctions=use_onnx_subfunctions)

        # Set device IDs, qpc path if precompiled qpc exist
        set_execute_params(self)

        # Ensure all modules are exported to ONNX before compilation
        if any(
            path is None
            for path in [
                self.transformer.onnx_path,
            ]
        ):
            self.export(use_onnx_subfunctions=use_onnx_subfunctions)

        # Calculate compressed latent dimension using utility function
        cl, latent_height, latent_width = calculate_compressed_latent_dimension(
            height, width, self.model.vae_scale_factor
        )

        # Prepare dynamic specialization updates based on video dimensions
        specialization_updates = {
            "transformer": {
                "latent_seq_len": cl,  # TODO : Make it dynamic
                # "cl": cl,  # Compressed latent dimension
                # "latent_height": latent_height,  # Latent space height
                # "latent_width": latent_width,  # Latent space width
            },
            "vae_decoder": {
                "latent_frames": 1,
                "latent_height": latent_height,
                "latent_width": latent_width,
            },
        }

        # Use generic utility functions for compilation
        if parallel:
            compile_modules_parallel(self.modules, self.custom_config, specialization_updates)
        else:
            compile_modules_sequential(self.modules, self.custom_config, specialization_updates)

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
        custom_config_path: Optional[str] = None,
        parallel_compile: bool = False,
        use_onnx_subfunctions: bool = False,
    ):
        """
        # TODO update docs
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
        device = "cpu"
        height = height
        width = width

        # Compile models with custom configuration if needed
        self.compile(
            compile_config=custom_config_path,
            parallel=parallel_compile,
            use_onnx_subfunctions=use_onnx_subfunctions,
            height=height,
            width=width,
        )

        # 1. Check inputs
        self.model.check_inputs(
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

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        # # Initialize transformer session
        if self.transformer.qpc_session is None:
            self.transformer.qpc_session = QAICInferenceSession(str(self.transformer.qpc_path))

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        transformer_perf = []
        cfg_perf = []

        # rotary emb
        qaic_image_rotary_emb = self.transformer.model.pos_embed(img_shapes, txt_seq_lens, device="cpu")
        qaic_img_freqs_cos, qaic_img_freqs_sin, qaic_txt_freqs_cos, qaic_txt_freqs_sin = qaic_image_rotary_emb

        img_rotary_emb = torch.cat([qaic_img_freqs_cos, qaic_img_freqs_sin], dim=-1)  # [6032, 128]
        txt_rotary_emb = torch.cat([qaic_txt_freqs_cos, qaic_txt_freqs_sin], dim=-1)  # [126, 128]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = (t.expand(latents.shape[0]) / 1000).detach().numpy().astype(np.float32)

                # Conditional pass
                transformer_inputs = {
                    "hidden_states": latents.detach().numpy().astype(np.float32),
                    "encoder_hidden_states": prompt_embeds.detach().numpy().astype(np.float32),
                    "img_rotary_emb": img_rotary_emb.detach().numpy().astype(np.float32),
                    "txt_rotary_emb": txt_rotary_emb.detach().numpy().astype(np.float32),
                    "timestep": timestep,
                }
                if guidance is not None:
                    transformer_inputs["guidance"] = guidance.numpy().astype(np.float32)

                # Run transformer inference and measure time
                start_transformer_step_time = time.perf_counter()
                noise_pred = self.transformer.qpc_session.run(transformer_inputs)
                end_transformer_step_time = time.perf_counter()
                transformer_perf.append(end_transformer_step_time - start_transformer_step_time)

                noise_pred = torch.tensor(noise_pred["output"])

                if do_true_cfg:
                    # Unconditional pass
                    transformer_inputs_uncond = {
                        "hidden_states": latents.detach().numpy().astype(np.float32),
                        "encoder_hidden_states": negative_prompt_embeds.detach().numpy().astype(np.float32),
                        "timestep": timestep,
                    }
                    if guidance is not None:
                        transformer_inputs_uncond["guidance"] = guidance.numpy().astype(np.float32)

                    start_cfg_step_time = time.perf_counter()
                    neg_noise_pred = self.transformer.qpc_session.run(transformer_inputs_uncond)
                    end_cfg_step_time = time.perf_counter()
                    cfg_perf.append(end_cfg_step_time - start_cfg_step_time)

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

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae_decoder.model.dtype)
            latents_mean = (
                torch.tensor(self.vae_decoder.model.config.latents_mean)
                .view(1, self.vae_decoder.model.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_decoder.model.config.latents_std).view(
                1, self.vae_decoder.model.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

            ########## QAIC
            # Initialize VAE decoder inference session
            if self.vae_decoder.qpc_session is None:
                self.vae_decoder.qpc_session = QAICInferenceSession(
                    str(self.vae_decoder.qpc_path), device_ids=self.vae_decoder.device_ids
                )

            # Allocate output buffer for VAE decoder
            output_buffer = {"sample": np.random.rand(batch_size, 3, 1, height, width).astype(np.int32)}
            self.vae_decoder.qpc_session.set_buffers(output_buffer)

            # Run VAE decoder inference and measure time
            inputs = {"latent_sample": latents.numpy()}
            start_decode_time = time.perf_counter()
            image = self.vae_decoder.qpc_session.run(inputs)
            end_decode_time = time.perf_counter()
            vae_decoder_perf = end_decode_time - start_decode_time

            image_tensor = torch.from_numpy(image["sample"])
            image_tensor = image_tensor[:, :, 0]
            image = self.image_processor.postprocess(image_tensor, output_type=output_type)

        if not return_dict:
            return (image,)

        # Build performance metrics
        perf_metrics = [
            ModulePerf(module_name="transformer", perf=transformer_perf),
            ModulePerf(module_name="vae_decoder", perf=vae_decoder_perf),
        ]

        return QEffPipelineOutput(
            pipeline_module=perf_metrics,
            images=image,
        )
