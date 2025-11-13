# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from tqdm import tqdm

from QEfficient.diffusers.pipelines.pipeline_module import (
    QEffFluxTransformerModel,
    QEffTextEncoder,
    QEffVAE,
)
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ModulePerf,
    QEffPipelineOutput,
    config_manager,
    set_module_device_ids,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.logging_utils import logger


class QEFFFluxPipeline(FluxPipeline):
    """
    QEfficient-optimized Flux pipeline for text-to-image generation on Qualcomm AI hardware.

    Attributes:
        text_encoder (QEffTextEncoder): Optimized CLIP text encoder
        text_encoder_2 (QEffTextEncoder): Optimized T5 text encoder
        transformer (QEffFluxTransformerModel): Optimized Flux transformer
        vae_decode (QEffVAE): Optimized VAE decoder
        modules (Dict): Dictionary of all pipeline modules for iteration
    """

    _hf_auto_class = FluxPipeline

    def __init__(self, model, use_onnx_function: bool, *args, **kwargs):
        """
        Initialize the QEfficient Flux pipeline.

        Args:
            model: Pre-loaded FluxPipeline model
            use_onnx_function (bool): Whether to export transformer blocks as ONNX functions
            **kwargs: Additional arguments including height and width
        """
        # Wrap model components with QEfficient optimized versions
        self.text_encoder = QEffTextEncoder(model.text_encoder)
        self.text_encoder_2 = QEffTextEncoder(model.text_encoder_2)
        self.transformer = QEffFluxTransformerModel(model.transformer, use_onnx_function=use_onnx_function)
        self.vae_decode = QEffVAE(model, "decoder")
        self.use_onnx_function = use_onnx_function

        # Store all modules in a dictionary for easy iteration during export/compile
        self.modules = {
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "transformer": self.transformer,
            "vae_decoder": self.vae_decode,
        }

        # Copy tokenizers and scheduler from the original model
        self.tokenizer = model.tokenizer
        self.text_encoder.tokenizer = model.tokenizer
        self.text_encoder_2.tokenizer = model.tokenizer_2
        self.tokenizer_max_length = model.tokenizer_max_length
        self.scheduler = model.scheduler

        # Set default image dimensions
        self.height = kwargs.get("height", 256)
        self.width = kwargs.get("width", 256)

        # Override VAE forward method to use decode directly
        self.vae_decode.model.forward = lambda latent_sample, return_dict: self.vae_decode.model.decode(
            latent_sample, return_dict
        )

        # Calculate VAE scale factor from model config
        self.vae_scale_factor = (
            2 ** (len(model.vae.config.block_out_channels) - 1) if getattr(model, "vae", None) else 8
        )

        # Flux uses 2x2 patches, so multiply scale factor by patch size
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        # Set tokenizer max length with fallback
        self.t_max_length = (
            model.tokenizer.model_max_length if hasattr(model, "tokenizer") and model.tokenizer is not None else 77
        )

        # Calculate latent dimensions based on image size and VAE scale factor
        self.default_sample_size = 128
        self.latent_height = self.height // self.vae_scale_factor
        self.latent_width = self.width // self.vae_scale_factor
        # cl = compressed latent dimension (divided by 4 for Flux's 2x2 packing)
        self.cl = (self.latent_height * self.latent_width) // 4

        # Sync max position embeddings between text encoders
        self.text_encoder_2.model.config.max_position_embeddings = (
            self.text_encoder.model.config.max_position_embeddings
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        use_onnx_function: bool = False,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        **kwargs,
    ):
        """
        Load a pretrained Flux model and wrap it with QEfficient optimizations.

        Args:
            pretrained_model_name_or_path (str or os.PathLike): HuggingFace model ID or local path
            use_onnx_function (bool): Whether to export transformer blocks as ONNX functions
            height (int): Target image height (default: 512)
            width (int): Target image width (default: 512)
            **kwargs: Additional arguments passed to FluxPipeline.from_pretrained

        Returns:
            QEFFFluxPipeline: Initialized pipeline instance
        """
        # Load the base Flux model in float32 on CPU
        model = cls._hf_auto_class.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float32,
            **kwargs,
        )
        model.to("cpu")

        return cls(
            model=model,
            use_onnx_function=use_onnx_function,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            height=height,
            width=width,
            **kwargs,
        )

    def export(self, export_dir: Optional[str] = None) -> str:
        """
        Export all pipeline modules to ONNX format.

        This method iterates through all modules (text encoders, transformer, VAE decoder)
        and exports each to ONNX using their respective configurations.

        Args:
            export_dir (str, optional): Directory to save ONNX models. If None, uses default.

        Returns:
            str: Path to the export directory
        """
        for module_name, module_obj in tqdm(self.modules.items(), desc="Exporting modules", unit="module"):
            # Get ONNX export configuration for this module
            example_inputs, dynamic_axes, output_names = module_obj.get_onnx_config()

            export_kwargs = {}
            # Special handling for transformer: export blocks as functions if enabled
            if module_name == "transformer" and self.use_onnx_function:
                export_kwargs = {
                    "export_modules_as_functions": self.transformer.model._block_classes,
                }

            # Export the module to ONNX
            module_obj.export(
                inputs=example_inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_dir=export_dir,
                export_kwargs=export_kwargs,
            )

    @staticmethod
    def get_default_config_path() -> str:
        """
        Get the path to the default Flux pipeline configuration file.

        Returns:
            str: Absolute path to flux_config.json
        """
        return os.path.join(os.path.dirname(__file__), "flux_config.json")

    def compile(self, compile_config: Optional[str] = None) -> None:
        """
        Compile ONNX models for deployment on Qualcomm AI hardware.

        This method compiles all pipeline modules (text encoders, transformer, VAE decoder)
        into optimized QPC (Qualcomm Program Container) format for inference on QAIC devices.

        Args:
            compile_config (str, optional): Path to JSON configuration file.
                                           If None, uses default configuration.
        """
        # Ensure all modules are exported to ONNX before compilation
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

        # Load compilation configuration
        if self.custom_config is None:
            config_manager(self, config_source=compile_config)

        # Compile each module with its specific configuration
        for module_name, module_obj in tqdm(self.modules.items(), desc="Compiling modules", unit="module"):
            module_config = self.custom_config["modules"]
            specializations = module_config[module_name]["specializations"]
            compile_kwargs = module_config[module_name]["compilation"]

            # Set dynamic specialization values based on image dimensions
            if module_name == "transformer":
                specializations["cl"] = self.cl
            elif module_name == "vae_decoder":
                specializations["latent_height"] = self.latent_height
                specializations["latent_width"] = self.latent_width

            # Compile the module to QPC format
            module_obj.compile(specializations=[specializations], **compile_kwargs)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device_ids: Optional[List[int]] = None,
    ):
        """
        Encode prompts using the T5 text encoder.

        Args:
            prompt (str or List[str]): Input prompt(s) to encode
            num_images_per_prompt (int): Number of images to generate per prompt
            max_sequence_length (int): Maximum token sequence length (default: 512)
            device_ids (List[int], optional): QAIC device IDs for inference

        Returns:
            tuple: (prompt_embeds, inference_time)
                - prompt_embeds (torch.Tensor): Encoded embeddings [batch*num_images, seq_len, 4096]
                - inference_time (float): T5 encoder inference time in seconds
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        embed_dim = 4096  # T5 embedding dimension

        # Tokenize prompts with padding and truncation
        text_inputs = self.text_encoder_2.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        # Check for truncation and warn user
        untruncated_ids = self.text_encoder_2.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.text_encoder_2.tokenizer.batch_decode(
                untruncated_ids[:, self.text_encoder_2.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                f"The following part of your input was truncated because `max_sequence_length` is set to "
                f"{self.text_encoder_2.tokenizer.model_max_length} tokens: {removed_text}"
            )

        # Initialize QAIC inference session if not already created
        if self.text_encoder_2.qpc_session is None:
            self.text_encoder_2.qpc_session = QAICInferenceSession(
                str(self.text_encoder_2.qpc_path), device_ids=device_ids
            )

        # Allocate output buffers for QAIC inference
        text_encoder_2_output = {
            "last_hidden_state": np.random.rand(batch_size, max_sequence_length, embed_dim).astype(np.float32),
        }
        self.text_encoder_2.qpc_session.set_buffers(text_encoder_2_output)

        # Prepare input for QAIC inference
        aic_text_input = {"input_ids": text_input_ids.numpy().astype(np.int64)}

        # Run T5 encoder inference and measure time
        start_t5_time = time.time()
        prompt_embeds = torch.tensor(self.text_encoder_2.qpc_session.run(aic_text_input)["last_hidden_state"])
        end_t5_time = time.time()
        text_encoder_2_perf = end_t5_time - start_t5_time

        # Duplicate embeddings for multiple images per prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, text_encoder_2_perf

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device_ids: Optional[List[int]] = None,
    ):
        """
        Encode prompts using the CLIP text encoder.

        Args:
            prompt (str or List[str]): Input prompt(s) to encode
            num_images_per_prompt (int): Number of images to generate per prompt
            device_ids (List[int], optional): QAIC device IDs for inference

        Returns:
            tuple: (pooled_prompt_embeds, inference_time)
                - pooled_prompt_embeds (torch.Tensor): Pooled embeddings [batch*num_images, 768]
                - inference_time (float): CLIP encoder inference time in seconds
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        embed_dim = 768  # CLIP embedding dimension

        # Tokenize prompts
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

        # Check for truncation and warn user
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                f"The following part of your input was truncated because CLIP can only handle sequences up to "
                f"{self.tokenizer_max_length} tokens: {removed_text}"
            )

        # Initialize QAIC inference session if not already created
        if self.text_encoder.qpc_session is None:
            self.text_encoder.qpc_session = QAICInferenceSession(str(self.text_encoder.qpc_path), device_ids=device_ids)

        # Allocate output buffers for QAIC inference
        text_encoder_output = {
            "last_hidden_state": np.random.rand(batch_size, self.tokenizer_max_length, embed_dim).astype(np.float32),
            "pooler_output": np.random.rand(batch_size, embed_dim).astype(np.float32),
        }
        self.text_encoder.qpc_session.set_buffers(text_encoder_output)

        # Prepare input for QAIC inference
        aic_text_input = {"input_ids": text_input_ids.numpy().astype(np.int64)}

        # Run CLIP encoder inference and measure time
        start_text_encoder_time = time.time()
        aic_embeddings = self.text_encoder.qpc_session.run(aic_text_input)
        end_text_encoder_time = time.time()
        text_encoder_perf = end_text_encoder_time - start_text_encoder_time

        # Extract pooled output (used for conditioning in Flux)
        prompt_embeds = torch.tensor(aic_embeddings["pooler_output"])

        # Duplicate embeddings for multiple images per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, text_encoder_perf

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
    ):
        """
        Encode prompts using both CLIP and T5 text encoders.

        Flux uses a dual text encoder setup:
        - CLIP provides pooled embeddings for global conditioning
        - T5 provides sequence embeddings for detailed text understanding

        Args:
            prompt (str or List[str]): Primary prompt(s)
            prompt_2 (str or List[str], optional): Secondary prompt(s) for T5. If None, uses primary prompt
            num_images_per_prompt (int): Number of images to generate per prompt
            prompt_embeds (torch.FloatTensor, optional): Pre-computed T5 embeddings
            pooled_prompt_embeds (torch.FloatTensor, optional): Pre-computed CLIP pooled embeddings
            max_sequence_length (int): Maximum sequence length for T5 tokenization

        Returns:
            tuple: (prompt_embeds, pooled_prompt_embeds, text_ids, encoder_perf_times)
                - prompt_embeds: T5 sequence embeddings
                - pooled_prompt_embeds: CLIP pooled embeddings
                - text_ids: Position IDs for text tokens
                - encoder_perf_times: List of [CLIP_time, T5_time]
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            # Use primary prompt for both encoders if secondary not provided
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # Encode with CLIP (returns pooled embeddings)
            pooled_prompt_embeds, text_encoder_perf = self._get_clip_prompt_embeds(
                prompt=prompt,
                device_ids=self.text_encoder.device_ids,
                num_images_per_prompt=num_images_per_prompt,
            )

            # Encode with T5 (returns sequence embeddings)
            prompt_embeds, text_encoder_2_perf = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device_ids=self.text_encoder_2.device_ids,
            )

        # Create text position IDs (required by Flux transformer)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3)

        return prompt_embeds, pooled_prompt_embeds, text_ids, [text_encoder_perf, text_encoder_2_perf]

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
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
        custom_config_path: Optional[str] = None,
    ):
        """
        Generate images from text prompts using the Flux pipeline.

        This is the main entry point for image generation. It orchestrates the entire pipeline:
        1. Validates inputs and loads configuration
        2. Encodes prompts using CLIP and T5
        3. Prepares latents and timesteps
        4. Runs denoising loop with transformer
        5. Decodes latents to images with VAE

        Args:
            prompt (str or List[str]): Text prompt(s) for image generation
            prompt_2 (str or List[str], optional): Secondary prompt for T5 encoder
            negative_prompt (str or List[str], optional): Negative prompt for classifier-free guidance
            negative_prompt_2 (str or List[str], optional): Secondary negative prompt
            true_cfg_scale (float): True CFG scale (default: 1.0, disabled)
            num_inference_steps (int): Number of denoising steps (default: 28)
            timesteps (List[int], optional): Custom timestep schedule
            guidance_scale (float): Guidance scale for generation (default: 3.5)
            num_images_per_prompt (int): Number of images per prompt (default: 1)
            generator (torch.Generator, optional): Random generator for reproducibility
            latents (torch.FloatTensor, optional): Pre-generated latents
            prompt_embeds (torch.FloatTensor, optional): Pre-computed prompt embeddings
            pooled_prompt_embeds (torch.FloatTensor, optional): Pre-computed pooled embeddings
            negative_prompt_embeds (torch.FloatTensor, optional): Pre-computed negative embeddings
            negative_pooled_prompt_embeds (torch.FloatTensor, optional): Pre-computed negative pooled embeddings
            output_type (str): Output format - "pil", "np", or "latent" (default: "pil")
            return_dict (bool): Whether to return QEffPipelineOutput object (default: True)
            joint_attention_kwargs (dict, optional): Additional attention processor kwargs
            callback_on_step_end (Callable, optional): Callback function after each step
            callback_on_step_end_tensor_inputs (List[str]): Tensors to pass to callback
            max_sequence_length (int): Maximum sequence length for T5 (default: 512)
            custom_config_path (str, optional): Path to custom compilation config

        Returns:
            QEffPipelineOutput or tuple: Generated images and performance metrics
        """
        device = "cpu"

        # Step 1: Load configuration and compile models if needed
        if custom_config_path is not None:
            config_manager(self, custom_config_path)
            set_module_device_ids(self)

        self.compile(compile_config=custom_config_path)

        # Validate all inputs
        self.check_inputs(
            prompt,
            prompt_2,
            self.height,
            self.width,
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

        # Step 2: Determine batch size from inputs
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Step 3: Encode prompts with both text encoders
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        (prompt_embeds, pooled_prompt_embeds, text_ids, text_encoder_perf) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        # Encode negative prompts if using true classifier-free guidance
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

        # Step 4: Prepare timesteps for denoising
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Step 5: Prepare initial latents
        num_channels_latents = self.transformer.model.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            self.height,
            self.width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Step 6: Initialize transformer inference session
        if self.transformer.qpc_session is None:
            self.transformer.qpc_session = QAICInferenceSession(
                str(self.transformer.qpc_path), device_ids=self.transformer.device_ids
            )

        # Allocate output buffer for transformer
        output_buffer = {
            "output": np.random.rand(batch_size, self.cl, self.transformer.model.config.in_channels).astype(np.float32),
        }
        self.transformer.qpc_session.set_buffers(output_buffer)

        transformer_perf = []
        self.scheduler.set_begin_index(0)

        # Step 7: Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Prepare timestep embedding
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                temb = self.transformer.model.time_text_embed(timestep, pooled_prompt_embeds)

                # Compute AdaLN (Adaptive Layer Normalization) embeddings for dual transformer blocks
                adaln_emb = []
                for block_idx in range(len(self.transformer.model.transformer_blocks)):
                    block = self.transformer.model.transformer_blocks[block_idx]
                    # Process through norm1 and norm1_context
                    f1 = block.norm1.linear(block.norm1.silu(temb)).chunk(6, dim=1)
                    f2 = block.norm1_context.linear(block.norm1_context.silu(temb)).chunk(6, dim=1)
                    adaln_emb.append(torch.cat(list(f1) + list(f2)))
                adaln_dual_emb = torch.stack(adaln_emb)

                # Compute AdaLN embeddings for single transformer blocks
                adaln_emb = []
                for block_idx in range(len(self.transformer.model.single_transformer_blocks)):
                    block = self.transformer.model.single_transformer_blocks[block_idx]
                    f1 = block.norm.linear(block.norm.silu(temb)).chunk(3, dim=1)
                    adaln_emb.append(torch.cat(list(f1)))
                adaln_single_emb = torch.stack(adaln_emb)

                # Compute output AdaLN embedding
                temp = self.transformer.model.norm_out
                adaln_out = temp.linear(temp.silu(temb))

                # Normalize timestep to [0, 1] range
                timestep = timestep / 1000

                # Prepare all inputs for transformer inference
                inputs_aic = {
                    "hidden_states": latents.detach().numpy(),
                    "encoder_hidden_states": prompt_embeds.detach().numpy(),
                    "pooled_projections": pooled_prompt_embeds.detach().numpy(),
                    "timestep": timestep.detach().numpy(),
                    "img_ids": latent_image_ids.detach().numpy(),
                    "txt_ids": text_ids.detach().numpy(),
                    "adaln_emb": adaln_dual_emb.detach().numpy(),
                    "adaln_single_emb": adaln_single_emb.detach().numpy(),
                    "adaln_out": adaln_out.detach().numpy(),
                }

                # Run transformer inference and measure time
                start_transformer_step_time = time.time()
                outputs = self.transformer.qpc_session.run(inputs_aic)
                end_transformer_step_time = time.time()
                transformer_perf.append(end_transformer_step_time - start_transformer_step_time)

                noise_pred = torch.from_numpy(outputs["output"])

                # Update latents using scheduler (x_t -> x_t-1)
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Handle dtype mismatch (workaround for MPS backend bug)
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                # Execute callback if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Step 8: Decode latents to images (unless output_type is "latent")
        if output_type == "latent":
            image = latents
        else:
            # Unpack and denormalize latents
            latents = self._unpack_latents(latents, self.height, self.width, self.vae_scale_factor)
            latents = (latents / self.vae_decode.model.scaling_factor) + self.vae_decode.model.shift_factor

            # Initialize VAE decoder inference session
            if self.vae_decode.qpc_session is None:
                self.vae_decode.qpc_session = QAICInferenceSession(
                    str(self.vae_decode.qpc_path), device_ids=self.vae_decode.device_ids
                )

            # Allocate output buffer for VAE decoder
            output_buffer = {"sample": np.random.rand(batch_size, 3, self.height, self.width).astype(np.int32)}
            self.vae_decode.qpc_session.set_buffers(output_buffer)

            # Run VAE decoder inference and measure time
            inputs = {"latent_sample": latents.numpy()}
            start_decode_time = time.time()
            image = self.vae_decode.qpc_session.run(inputs)
            end_decode_time = time.time()
            vae_decode_perf = end_decode_time - start_decode_time

            # Post-process image
            image_tensor = torch.from_numpy(image["sample"])
            image = self.image_processor.postprocess(image_tensor, output_type=output_type)

            # Build performance metrics
            perf_metrics = [
                ModulePerf(module_name="text_encoder", perf=text_encoder_perf[0]),
                ModulePerf(module_name="text_encoder_2", perf=text_encoder_perf[1]),
                ModulePerf(module_name="transformer", perf=transformer_perf),
                ModulePerf(module_name="vae_decoder", perf=vae_decode_perf),
            ]

            return QEffPipelineOutput(
                pipeline_module=perf_metrics,
                images=image,
            )
