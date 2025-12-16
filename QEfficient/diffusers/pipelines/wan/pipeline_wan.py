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

TODO: To enable VAE, UMT5 on QAIC
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import WanPipeline
from diffusers.video_processor import VideoProcessor

from QEfficient.diffusers.pipelines.pipeline_module import QEffWanUnifiedTransformer, QEffWanUnifiedWrapper
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ONNX_SUBFUNCTION_MODULE,
    ModulePerf,
    QEffPipelineOutput,
    compile_modules_parallel,
    compile_modules_sequential,
    config_manager,
    set_module_device_ids,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants
from QEfficient.utils.logging_utils import logger


class QEffWanPipeline(WanPipeline):
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
        video_processor (VideoProcessor): Video post-processing utilities

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
        # Required by diffusers for serialization and device management
        self.model = model
        self.kwargs = kwargs
        self.custom_config = None

        # Text encoder (TODO: Replace with QEfficient UMT5 optimization)
        self.text_encoder = model.text_encoder

        # Create unified transformer wrapper combining dual-stage models(high, low noise DiTs)
        self.unified_wrapper = QEffWanUnifiedWrapper(model.transformer, model.transformer_2)
        self.transformer = QEffWanUnifiedTransformer(self.unified_wrapper)

        # VAE decoder for latent-to-video conversion
        self.vae_decode = model.vae

        # Store all modules in a dictionary for easy iteration during export/compile
        self.modules = {"transformer": self.transformer}

        # Copy tokenizers and scheduler from the original model
        self.tokenizer = model.tokenizer
        self.text_encoder.tokenizer = model.tokenizer
        self.scheduler = model.scheduler

        # Register components with diffusers framework
        # TODO: Clean up register_modules (verify all model.config are working correctly)
        self.register_modules(
            text_encoder=self.text_encoder,
            scheduler=self.scheduler,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            vae=self.vae_decode,
        )

        # Register configuration parameters
        self.register_to_config(boundary_ratio=self.model.config.boundary_ratio)
        self.register_to_config(expand_timesteps=self.model.config.expand_timesteps)

        # Configure VAE scale factors for video processing
        self.vae_scale_factor_temporal = (
            self.vae_decode.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        )
        self.vae_scale_factor_spatial = self.vae_decode.config.scale_factor_spatial if getattr(self, "vae", None) else 8

        # Initialize video processor for frame handling and post-processing
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # Extract patch dimensions from transformer configuration
        _, self.patch_height, self.patch_width = self.transformer.model.config.patch_size

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

    @property
    def components(self):
        """
        Get dictionary of all pipeline components.

        Returns:
            Dict[str, Any]: Dictionary containing all pipeline components including
                text_encoder, transformer, vae, tokenizer, and scheduler.
        """
        return {
            "text_encoder": self.text_encoder,
            "transformer": self.transformer,
            "vae": self.vae_decode,
            "tokenizer": self.tokenizer,
            "scheduler": self.scheduler,
        }

    def configure_height_width_cl_latents_hw(self, height, width):
        """
        Configure pipeline dimensions and calculate compressed latent parameters.

        This method sets the target video dimensions and calculates the corresponding
        compressed latent dimension (cl) and latent space dimensions for buffer allocation
        and transformer processing.

        Args:
            height (int): Target video height in pixels
            width (int): Target video width in pixels

        Note:
            - Updates self.height, self.width, self.cl, self.latent_height, self.latent_width
            - Used internally before compilation and inference
        """
        self.height = height
        self.width = width
        self.cl, self.latent_height, self.latent_width = self.calculate_compressed_latent_dimension(height, width)

    def calculate_compressed_latent_dimension(self, height, width):
        """
        Calculate the compressed latent dimension for transformer buffer allocation.

        This method computes the compressed sequence length (cl) that the transformer
        will process, based on the target video dimensions, VAE scale factors, and
        patch sizes. This is crucial for proper buffer allocation in QAIC inference.

        Args:
            height (int): Target video height in pixels
            width (int): Target video width in pixels

        Returns:
            tuple: (cl, latent_height, latent_width)
                - cl (int): Compressed latent dimension for transformer input
                - latent_height (int): Height in latent space
                - latent_width (int): Width in latent space

        Mathematical Formula:
            latent_height = height // vae_scale_factor_spatial
            latent_width = width // vae_scale_factor_spatial
            cl = (latent_height // patch_height) * (latent_width // patch_width) * latent_frames

        Note:
            - Uses constants.WAN_ONNX_EXPORT_LATENT_FRAMES for temporal dimension (21 frames for 81 input frames)
            - TODO: Calculate latent frames dynamically based on input num_frames
        """
        # Calculate latent space dimensions after VAE encoding
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        # Calculate compressed sequence length for transformer processing
        # TODO: Calculate latent frames based on actual frames passed in pipeline call
        cl = (
            latent_height // self.patch_height * latent_width // self.patch_width
        ) * constants.WAN_ONNX_EXPORT_LATENT_FRAMES

        return cl, latent_height, latent_width

    def export(
        self,
        height: int = constants.WAN_ONNX_EXPORT_HEIGHT_180P,
        width: int = constants.WAN_ONNX_EXPORT_WIDTH_180P,
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
            height (int, default=192): Export height in pixels. Users can export at low
                resolution and compile for higher resolution later for flexibility.
            width (int, default=320): Export width in pixels. Should maintain aspect ratio
                appropriate for the target use case.
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
            ...     height=480,
            ...     width=832,
            ...     export_dir="/path/to/export",
            ...     use_onnx_subfunctions=True
            ... )
        """
        # Calculate compressed latent dimensions for export configuration
        export_cl, export_latent_height, export_latent_width = self.calculate_compressed_latent_dimension(height, width)

        # Export each module with video-specific parameters
        for module_name, module_obj in self.modules.items():
            # Get ONNX export configuration with video dimensions
            example_inputs, dynamic_axes, output_names = module_obj.get_onnx_params(
                batch_size=1,
                seq_length=512,  # Text sequence length
                cl=export_cl,  # Compressed latent dimension
                latent_height=export_latent_height,
                latent_width=export_latent_width,
            )

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

            # Export with performance timing
            start_time = time.perf_counter()
            module_obj.export(**export_params)
            end_time = time.perf_counter()
            print(f"{module_name} export took {end_time - start_time:.2f} seconds")

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
            >>> pipeline.compile(height=480, width=832)
            >>>
            >>> # Parallel compilation with custom config
            >>> pipeline.compile(
            ...     compile_config="/path/to/custom_config.json",
            ...     parallel=True
            ... )
        """
        # Ensure all modules are exported to ONNX before compilation
        if any(
            path is None
            for path in [
                # self.text_encoder.onnx_path,  # TODO: Enable when UMT5 is optimized
                self.transformer.onnx_path,
                # self.vae_decode.onnx_path,  # TODO: Enable when VAE is optimized
            ]
        ):
            self.export(use_onnx_subfunctions=use_onnx_subfunctions)

        # Load compilation configuration
        config_manager(self, config_source=compile_config)

        # Prepare dynamic specialization updates based on video dimensions
        specialization_updates = {
            "transformer": {
                "cl": self.cl,  # Compressed latent dimension
                "latent_height": self.latent_height,  # Latent space height
                "latent_width": self.latent_width,  # Latent space width
            }
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

        # Configure pipeline dimensions and calculate compressed latent parameters
        self.configure_height_width_cl_latents_hw(height, width)

        # Compile models with custom configuration if needed
        self.compile(
            compile_config=custom_config_path,
            parallel=parallel_compile,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

        # Set device IDs for all modules based on configuration
        set_module_device_ids(self)

        # Step 1: Validate all inputs
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

        # Ensure num_frames satisfies temporal divisibility requirements
        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        # Initialize pipeline state
        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
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
        start_encoder_time = time.perf_counter()
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
        end_encoder_time = time.perf_counter()
        text_encoder_perf = end_encoder_time - start_encoder_time

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

        # Create mask for temporal processing (used in expand_timesteps mode)
        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # Step 6: Configure dual-stage processing for WAN 2.2
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Calculate boundary timestep for stage switching in WAN 2.2
        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        # Step 7: Initialize QAIC inference session for transformer
        if self.transformer.qpc_session is None:
            self.transformer.qpc_session = QAICInferenceSession(
                str(self.transformer.qpc_path), device_ids=self.transformer.device_ids
            )

        # Allocate output buffer for QAIC inference
        output_buffer = {
            "output": np.random.rand(
                batch_size,
                self.cl,  # Compressed latent dimension
                64,
                # TODO: Use self.transformer.model.config.joint_attention_dim and in_channels
            ).astype(np.int32),
        }
        self.transformer.qpc_session.set_buffers(output_buffer)
        transformer_perf = []

        # Step 8: Denoising loop with dual-stage processing
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
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
                if self.config.expand_timesteps:
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
                    noise_pred = torch.from_numpy(outputs["output"])
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

        # Step 9: Decode latents to video (unless output_type is "latent")
        if not output_type == "latent":
            # Prepare latents for VAE decoding
            latents = latents.to(self.vae_decode.dtype)

            # Apply VAE normalization (denormalization)
            latents_mean = (
                torch.tensor(self.vae_decode.config.latents_mean)
                .view(1, self.vae_decode.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae_decode.config.latents_std).view(
                1, self.vae_decode.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

            # TODO: Enable VAE on QAIC
            # VAE Decode latents to video using CPU (temporary)
            start_decode_time = time.perf_counter()
            video = self.model.vae.decode(latents, return_dict=False)[0]  # CPU fallback
            end_decode_time = time.perf_counter()
            vae_decode_perf = end_decode_time - start_decode_time

            # Post-process video for output
            video = self.video_processor.postprocess_video(video.detach())
        else:
            video = latents

        # Step 10: Collect performance metrics
        perf_data = {
            "umt5": text_encoder_perf,  # UMT5 text encoder (CPU)
            "transformer": transformer_perf,  # Unified transformer (QAIC)
            "vae_decoder": vae_decode_perf,  # VAE decoder (CPU)
        }

        # Build performance metrics for output
        perf_metrics = [ModulePerf(module_name=name, perf=perf_data[name]) for name in perf_data.keys()]

        return QEffPipelineOutput(
            pipeline_module=perf_metrics,
            images=video,
        )
