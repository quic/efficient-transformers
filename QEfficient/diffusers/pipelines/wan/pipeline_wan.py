# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""
QEfficient WAN Pipeline Implementation

This module provides an optimized implementation of the WAN pipeline
for high-performance text-to-video generation on Qualcomm AI hardware.
The pipeline supports WAN 2.2 architectures with unified transformer.

TODO: 1. Update umt5 to Qaic; present running on cpu
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import WanPipeline
from tqdm import tqdm

from QEfficient.diffusers.pipelines.pipeline_module import QEffVAE, QEffWanUnifiedTransformer
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ONNX_SUBFUNCTION_MODULE,
    ModulePerf,
    QEffPipelineOutput,
    QEffWanUnifiedWrapper,
    calculate_latent_dimensions_with_frames,
    compile_modules_parallel,
    compile_modules_sequential,
    config_manager,
    set_module_device_ids,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants
from QEfficient.utils.logging_utils import logger


class QEffWanPipeline:
    """
    QEfficient-optimized WAN pipeline for high-performance text-to-video generation on Qualcomm AI hardware.

    This pipeline provides an optimized implementation of the WAN diffusion model
    specifically designed for deployment on Qualcomm AI Cloud (QAIC) devices. It extends the original
    HuggingFace WAN model with QEfficient-optimized components that can be exported to ONNX format
    and compiled into Qualcomm Program Container (QPC) files for efficient video generation.

    The pipeline supports the complete WAN workflow including:
    - UMT5 text encoding for rich semantic understanding
    - Unified transformer architecture: Combines multiple transformer stages into a single optimized model
    - VAE decoding for final video output
    - Performance monitoring and hardware optimization

    Attributes:
        text_encoder: UMT5 text encoder for semantic text understanding (TODO: QEfficient optimization)
        unified_wrapper (QEffWanUnifiedWrapper): Wrapper combining transformer stages
        transformer (QEffWanUnifiedTransformer): Optimized unified transformer for denoising
        vae_decode: VAE decoder for latent-to-video conversion
        modules (Dict[str, Any]): Dictionary of pipeline modules for batch operations
        model (WanPipeline): Original HuggingFace WAN model reference
        tokenizer: Text tokenizer for preprocessing
        scheduler: Diffusion scheduler for timestep management

    Example:
        >>> from QEfficient.diffusers.pipelines.wan import QEffWanPipeline
        >>> pipeline = QEffWanPipeline.from_pretrained("path/to/wan/model")
        >>> videos = pipeline(
        ...     prompt="A cat playing in a garden",
        ...     height=480,
        ...     width=832,
        ...     num_frames=81,
        ...     num_inference_steps=4
        ... )
        >>> # Save generated video
        >>> videos.images[0].save("generated_video.mp4")
    """

    _hf_auto_class = WanPipeline

    def __init__(self, model, **kwargs):
        """
        Initialize the QEfficient WAN pipeline.

        This pipeline provides an optimized implementation of the WAN text-to-video model
        for deployment on Qualcomm AI hardware. It wraps the original HuggingFace WAN model
        components with QEfficient-optimized versions that can be exported to ONNX and compiled
        for QAIC devices.

        Args:
            model: Pre-loaded WanPipeline model with transformer and transformer_2 components
            **kwargs: Additional keyword arguments including configuration parameters
        """
        # Store original model and configuration
        self.model = model
        self.kwargs = kwargs
        self.custom_config = None

        # Text encoder (TODO: Replace with QEfficient UMT5 optimization)
        self.text_encoder = model.text_encoder

        # Create unified transformer wrapper combining dual-stage models(high, low noise DiTs)
        self.unified_wrapper = QEffWanUnifiedWrapper(model.transformer, model.transformer_2)
        self.transformer = QEffWanUnifiedTransformer(self.unified_wrapper)

        # VAE decoder for latent-to-video conversion
        self.vae_decoder = QEffVAE(model.vae, "decoder")
        # Store all modules in a dictionary for easy iteration during export/compile
        # TODO: add text encoder on QAIC
        self.modules = {"transformer": self.transformer, "vae_decoder": self.vae_decoder}

        # Copy tokenizers and scheduler from the original model
        self.tokenizer = model.tokenizer
        self.text_encoder.tokenizer = model.tokenizer
        self.scheduler = model.scheduler

        self.vae_decoder.model.forward = lambda latent_sample, return_dict: self.vae_decoder.model.decode(
            latent_sample, return_dict
        )

        self.vae_decoder.get_onnx_params = self.vae_decoder.get_video_onnx_params
        # Extract patch dimensions from transformer configuration
        _, self.patch_height, self.patch_width = self.transformer.model.config.patch_size

    @property
    def do_classifier_free_guidance(self):
        """
        Determine if classifier-free guidance should be used.

        Returns:
            bool: True if CFG should be applied based on current guidance scales
        """
        return self._guidance_scale > 1.0 and (self._guidance_scale_2 is None or self._guidance_scale_2 > 1.0)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        **kwargs,
    ):
        """
        Load a pretrained WAN model from HuggingFace Hub or local path and wrap it with QEfficient optimizations.

        This class method provides a convenient way to instantiate a QEffWanPipeline from a pretrained
        WAN model. It automatically loads the base WanPipeline model in float32 precision on CPU
        and wraps all components with QEfficient-optimized versions for QAIC deployment.

        Args:
            pretrained_model_name_or_path (str or os.PathLike): Either a HuggingFace model identifier
                or a local path to a saved WAN model directory. Should contain transformer, transformer_2,
                text_encoder, and VAE components.
            **kwargs: Additional keyword arguments passed to WanPipeline.from_pretrained().

        Returns:
            QEffWanPipeline: A fully initialized pipeline instance with QEfficient-optimized components
                ready for export, compilation, and inference on QAIC devices.

        Raises:
            ValueError: If the model path is invalid or model cannot be loaded
            OSError: If there are issues accessing the model files
            RuntimeError: If model initialization fails

        Example:
            >>> # Load from HuggingFace Hub
            >>> pipeline = QEffWanPipeline.from_pretrained("path/to/wan/model")
            >>>
            >>> # Load from local path
            >>> pipeline = QEffWanPipeline.from_pretrained("/local/path/to/wan")
            >>>
            >>> # Load with custom cache directory
            >>> pipeline = QEffWanPipeline.from_pretrained(
            ...     "wan-model-id",
            ...     cache_dir="/custom/cache/dir"
            ... )
        """
        # Load the base WAN model in float32 on CPU for optimization
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

            module_obj.export(**export_params)

    @staticmethod
    def get_default_config_path():
        """
        Get the default configuration file path for WAN pipeline.

        Returns:
            str: Path to the default WAN configuration JSON file.
        """
        return os.path.join(os.path.dirname(__file__), "wan_config.json")

    def compile(
        self,
        compile_config: Optional[str] = None,
        parallel: bool = False,
        height: int = constants.WAN_ONNX_EXPORT_HEIGHT_180P,
        width: int = constants.WAN_ONNX_EXPORT_WIDTH_180P,
        num_frames: int = constants.WAN_ONNX_EXPORT_FRAMES,
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
            num_frames (int, deafult=81) : Target num of frames in pixel space
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
            ...     num_frames=81
            ... )
        """
        # Ensure all modules are exported to ONNX before compilation
        if any(
            path is None
            for path in [
                self.transformer.onnx_path,
                self.vae_decoder.onnx_path,
            ]
        ):
            self.export(use_onnx_subfunctions=use_onnx_subfunctions)

        # Load compilation configuration
        config_manager(self, config_source=compile_config, use_onnx_subfunctions=use_onnx_subfunctions)

        # Configure pipeline dimensions and calculate compressed latent parameters
        cl, latent_height, latent_width, latent_frames = calculate_latent_dimensions_with_frames(
            height,
            width,
            num_frames,
            self.model.vae.config.scale_factor_spatial,
            self.model.vae.config.scale_factor_temporal,
            self.patch_height,
            self.patch_width,
        )
        # Prepare dynamic specialization updates based on video dimensions
        specialization_updates = {
            "transformer": [
                # high noise
                {
                    "cl": cl,  # Compressed latent dimension
                    "latent_height": latent_height,  # Latent space height
                    "latent_width": latent_width,  # Latent space width
                    "latent_frames": latent_frames,  # Latent frames
                },
                # low noise
                {
                    "cl": cl,  # Compressed latent dimension
                    "latent_height": latent_height,  # Latent space height
                    "latent_width": latent_width,  # Latent space width
                    "latent_frames": latent_frames,  # Latent frames
                },
            ],
            "vae_decoder": {
                "latent_frames": latent_frames,
                "latent_height": latent_height,
                "latent_width": latent_width,
            },
        }

        # Use generic utility functions for compilation
        logger.warning('For VAE compilation use QAIC_COMPILER_OPTS_UNSUPPORTED="-aic-hmx-conv3d" ')
        if parallel:
            compile_modules_parallel(self.modules, self.custom_config, specialization_updates)
        else:
            compile_modules_sequential(self.modules, self.custom_config, specialization_updates)

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
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
        callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None]]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        custom_config_path: Optional[str] = None,
        use_onnx_subfunctions: bool = False,
        parallel_compile: bool = True,
    ):
        """
        Generate videos from text prompts using the QEfficient-optimized WAN pipeline on QAIC hardware.

        This is the main entry point for text-to-video generation. It orchestrates the complete WAN
        diffusion pipeline optimized for Qualcomm AI Cloud devices.

        Args:
            prompt (str or List[str]): Primary text prompt(s) describing the desired video content.
                Required unless `prompt_embeds` is provided.
            negative_prompt (str or List[str], optional): Negative prompt(s) describing what to avoid
                in the generated video. Used with classifier-free guidance.
            height (int, optional): Target video height in pixels. Must be divisible by VAE scale factor.
                Default: 480.
            width (int, optional): Target video width in pixels. Must be divisible by VAE scale factor.
                Default: 832.
            num_frames (int, optional): Number of video frames to generate. Must satisfy temporal
                divisibility requirements. Default: 81.
            num_inference_steps (int, optional): Number of denoising steps. More steps generally
                improve quality but increase generation time. Default: 50.
            guidance_scale (float, optional): Guidance scale for classifier-free guidance. Default: 3.0.
            guidance_scale_2 (float, optional): Guidance scale for low-noise stage in WAN 2.2.
                If None, uses guidance_scale value.
            num_videos_per_prompt (int, optional): Number of videos to generate per prompt. Default: 1.
            generator (torch.Generator or List[torch.Generator], optional): Random generator for
                reproducible generation.
            latents (torch.Tensor, optional): Pre-generated latent tensors. If None, random latents
                are generated based on video dimensions.
            prompt_embeds (torch.Tensor, optional): Pre-computed text embeddings from UMT5 encoder.
                Shape: [batch, seq_len, hidden_dim].
            negative_prompt_embeds (torch.Tensor, optional): Pre-computed negative text embeddings.
            output_type (str, optional): Output format. Options: "np" (default), "pil", or "latent".
            return_dict (bool, optional): Whether to return a dictionary or tuple. Default: True.
            attention_kwargs (Dict[str, Any], optional): Additional attention arguments for transformer.
            callback_on_step_end (Callable, optional): Callback function executed after each denoising step.
            callback_on_step_end_tensor_inputs (List[str], optional): Tensor names to pass to callback.
                Default: ["latents"].
            max_sequence_length (int, optional): Maximum token sequence length for text encoder. Default: 512.
            custom_config_path (str, optional): Path to custom JSON configuration file for compilation.
            use_onnx_subfunctions (bool, optional): Whether to export transformer blocks as ONNX subfunctions.
                Default: False.
            parallel_compile (bool, optional): Whether to compile modules in parallel. Default: True.

        Returns:
            QEffPipelineOutput: A dataclass containing:
                - images: Generated video(s) in the format specified by `output_type`
                - pipeline_module: Performance metrics for each pipeline component

        Raises:
            ValueError: If input validation fails or parameters are incompatible
            RuntimeError: If compilation fails or QAIC devices are unavailable
            FileNotFoundError: If custom config file is specified but not found

        Example:
            >>> from QEfficient.diffusers.pipelines.wan import QEffWanPipeline
            >>> pipeline = QEffWanPipeline.from_pretrained("path/to/wan/model")
            >>> result = pipeline(
            ...     prompt="A cat playing in a sunny garden",
            ...     height=480,
            ...     width=832,
            ...     num_frames=81,
            ...     num_inference_steps=4,
            ...     guidance_scale=3.0
            ... )
            >>> # Save generated video
            >>> result.images[0].save("cat_garden.mp4")
        """
        device = "cpu"

        # Compile models with custom configuration if needed
        self.compile(
            compile_config=custom_config_path,
            parallel=parallel_compile,
            use_onnx_subfunctions=use_onnx_subfunctions,
            height=height,
            width=width,
            num_frames=num_frames,
        )

        # Set device IDs for all modules based on configuration
        set_module_device_ids(self)

        # Step 1: Validate all inputs
        self.model.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        # Ensure num_frames satisfies temporal divisibility requirements
        if num_frames % self.model.vae.config.scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.model.vae.config.scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames // self.model.vae.config.scale_factor_temporal * self.model.vae.config.scale_factor_temporal
                + 1
            )
        num_frames = max(num_frames, 1)

        if self.model.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        # Initialize pipeline state
        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2 if guidance_scale_2 is not None else guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # Step 2: Determine batch size from inputs
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Step 3: Encode input prompts using UMT5 text encoder
        # TODO: Update UMT5 on QAIC
        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Convert embeddings to transformer dtype for compatibility
        transformer_dtype = self.transformer.model.transformer_high.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # Step 4: Prepare timesteps for denoising process
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Step 5: Prepare initial latent variables for video generation
        num_channels_latents = self.transformer.model.config.in_channels

        latents = self.model.prepare_latents(
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

        # Create mask for temporal processing (used in expand_timesteps mode)
        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # Step 6: Configure dual-stage processing for WAN 2.2
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Calculate boundary timestep for stage switching in WAN 2.2
        if self.model.config.boundary_ratio is not None:
            boundary_timestep = self.model.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        # Step 7: Initialize QAIC inference session for transformer
        if self.transformer.qpc_session is None:
            self.transformer.qpc_session = QAICInferenceSession(
                str(self.transformer.qpc_path), device_ids=self.transformer.device_ids
            )

        # Calculate compressed latent dimension for transformer buffer allocation
        cl, _, _, _ = calculate_latent_dimensions_with_frames(
            height,
            width,
            num_frames,
            self.model.vae.config.scale_factor_spatial,
            self.model.vae.config.scale_factor_temporal,
            self.patch_height,
            self.patch_width,
        )
        # Allocate output buffer for QAIC inference
        output_buffer = {
            "output": np.random.rand(
                batch_size,
                cl,  # Compressed latent dimension
                constants.WAN_DIT_OUT_CHANNELS,
            ).astype(np.int32),
        }
        self.transformer.qpc_session.set_buffers(output_buffer)
        transformer_perf = []

        # Step 8: Denoising loop with dual-stage processing
        with self.model.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue

                self._current_timestep = t

                # Determine which model to use based on boundary timestep
                if boundary_timestep is None or t >= boundary_timestep:
                    # High-noise stage
                    current_model = self.transformer.model.transformer_high
                    current_guidance_scale = guidance_scale
                    model_type = torch.ones(1, dtype=torch.int64)  # High-noise model indicator
                else:
                    # Low-noise stage
                    current_model = self.transformer.model.transformer_low
                    current_guidance_scale = guidance_scale_2
                    model_type = torch.ones(2, dtype=torch.int64)  # Low-noise model indicator

                # Prepare latent input with proper dtype
                latent_model_input = latents.to(transformer_dtype)

                # Handle timestep expansion for temporal consistency
                if self.model.config.expand_timesteps:
                    # Expand timesteps spatially for better temporal modeling
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    # Standard timestep broadcasting
                    timestep = t.expand(latents.shape[0])

                # Extract dimensions for patch processing
                batch_size, num_channels, num_frames, height, width = latents.shape
                p_t, p_h, p_w = current_model.config.patch_size
                post_patch_num_frames = num_frames // p_t
                post_patch_height = height // p_h
                post_patch_width = width // p_w

                # Generate rotary position embeddings
                rotary_emb = current_model.rope(latent_model_input)
                rotary_emb = torch.cat(rotary_emb, dim=0)
                ts_seq_len = None
                timestep = timestep.flatten()

                # Generate conditioning embeddings (time + text)
                temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
                    current_model.condition_embedder(
                        timestep, prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=ts_seq_len
                    )
                )

                # Generate negative conditioning for classifier-free guidance
                if self.do_classifier_free_guidance:
                    temb, timestep_proj, encoder_hidden_states_neg, encoder_hidden_states_image = (
                        current_model.condition_embedder(
                            timestep,
                            negative_prompt_embeds,
                            encoder_hidden_states_image=None,
                            timestep_seq_len=ts_seq_len,
                        )
                    )

                # Reshape timestep projection for transformer input
                timestep_proj = timestep_proj.unflatten(1, (6, -1))

                # Prepare inputs for QAIC inference
                inputs_aic = {
                    "hidden_states": latents.detach().numpy(),
                    "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
                    "rotary_emb": rotary_emb.detach().numpy(),
                    "temb": temb.detach().numpy(),
                    "timestep_proj": timestep_proj.detach().numpy(),
                    "tsp": model_type.detach().numpy(),  # Transformer stage pointer
                }

                # Prepare negative inputs for classifier-free guidance
                if self.do_classifier_free_guidance:
                    inputs_aic2 = {
                        "hidden_states": latents.detach().numpy(),
                        "encoder_hidden_states": encoder_hidden_states_neg.detach().numpy(),
                        "rotary_emb": rotary_emb.detach().numpy(),
                        "temb": temb.detach().numpy(),
                        "timestep_proj": timestep_proj.detach().numpy(),
                    }

                # Run conditional prediction with caching context
                with current_model.cache_context("cond"):
                    # QAIC inference for conditional prediction
                    start_transformer_step_time = time.perf_counter()
                    outputs = self.transformer.qpc_session.run(inputs_aic)
                    end_transformer_step_time = time.perf_counter()
                    transformer_perf.append(end_transformer_step_time - start_transformer_step_time)
                    print(f"DIT {i} time {end_transformer_step_time - start_transformer_step_time:.2f} seconds")

                    # Process transformer output
                    hidden_states = torch.tensor(outputs["output"])

                    # Reshape output from patches back to video format
                    hidden_states = hidden_states.reshape(
                        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                    )

                    # Permute dimensions to reconstruct video tensor
                    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                    noise_pred = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                # Run unconditional prediction for classifier-free guidance
                if self.do_classifier_free_guidance:  # Note: CFG is False for WAN Lightning
                    with current_model.cache_context("uncond"):
                        # QAIC inference for unconditional prediction
                        start_transformer_step_time = time.perf_counter()
                        outputs = self.transformer.qpc_session.run(inputs_aic2)
                        end_transformer_step_time = time.perf_counter()
                        transformer_perf.append(end_transformer_step_time - start_transformer_step_time)

                        # Process unconditional output
                        hidden_states = torch.tensor(outputs["output"])

                        # Reshape unconditional output
                        hidden_states = hidden_states.reshape(
                            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                        )

                        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                        noise_uncond = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

                        # Apply classifier-free guidance
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # Update latents using scheduler (x_t -> x_t-1)
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Execute callback if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        # Step 9: Decode latents to video
        if not output_type == "latent":
            # Prepare latents for VAE decoding
            latents = latents.to(self.vae_decoder.model.dtype)

            # Apply VAE normalization (denormalization)
            latents_mean = (
                torch.tensor(self.vae_decoder.model.config.latents_mean)
                .view(1, self.vae_decoder.model.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_decoder.model.config.latents_std).view(
                1, self.vae_decoder.model.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

            # Initialize VAE decoder inference session
            if self.vae_decoder.qpc_session is None:
                self.vae_decoder.qpc_session = QAICInferenceSession(
                    str(self.vae_decoder.qpc_path), device_ids=self.vae_decoder.device_ids
                )

            # Allocate output buffer for VAE decoder
            output_buffer = {"sample": np.random.rand(batch_size, 3, num_frames, height, width).astype(np.int32)}

            inputs = {"latent_sample": latents.numpy()}

            start_decode_time = time.perf_counter()
            video = self.vae_decoder.qpc_session.run(inputs)
            end_decode_time = time.perf_counter()
            vae_decoder_perf = end_decode_time - start_decode_time

            # Post-process video for output
            video_tensor = torch.from_numpy(video["sample"])
            video = self.model.video_processor.postprocess_video(video_tensor)
        else:
            video = latents

        # Step 10: Collect performance metrics
        perf_data = {
            "transformer": transformer_perf,  # Unified transformer (QAIC)
            "vae_decoder": vae_decoder_perf,
        }

        # Build performance metrics for output
        perf_metrics = [ModulePerf(module_name=name, perf=perf_data[name]) for name in perf_data.keys()]

        return QEffPipelineOutput(
            pipeline_module=perf_metrics,
            images=video,
        )
