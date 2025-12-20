# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Test for wan pipeline
# TODO : 1. Add pytest for call method
         2. See if we reduce height and width
        3. Keep test for Sub fn as default once sdk supports
"""

import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pytest
import safetensors.torch
import torch
from diffusers import WanPipeline
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download

from QEfficient import QEffWanPipeline
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ModulePerf,
    QEffPipelineOutput,
    calculate_latent_dimensions_with_frames,
    set_module_device_ids,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants
from QEfficient.utils._utils import load_json
from tests.diffusers.diffusers_utils import DiffusersTestUtils, MADValidator

# Test Configuration for 192x320 resolution with 1 layer
CONFIG_PATH = "tests/diffusers/wan_test_config.json"
INITIAL_TEST_CONFIG = load_json(CONFIG_PATH)


def wan_pipeline_call_with_mad_validation(
    pipeline,
    pytorch_pipeline,
    height: int = 192,
    width: int = 320,
    num_frames: int = 81,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    num_inference_steps: int = 2,
    guidance_scale: float = 1.0,
    guidance_scale_2: Optional[float] = None,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "np",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    custom_config_path: Optional[str] = None,
    use_onnx_subfunctions: bool = False,
    parallel_compile: bool = True,
    mad_tolerances: Dict[str, float] = None,
):
    """
    Pipeline call function that replicates the exact flow of pipeline_wan.py.__call__()
    while adding comprehensive MAD validation for transformer modules only.

    This function follows the EXACT same structure as QEffWanPipeline.__call__()
    but adds MAD validation hooks for transformer testing.
    """
    # Initialize MAD validator
    mad_validator = MADValidator(tolerances=mad_tolerances)

    device = "cpu"

    # Step 1: Compile() (export and compile)
    pipeline.cl, pipeline.latent_height, pipeline.latent_width, pipeline.latent_frames = (
        calculate_latent_dimensions_with_frames(
            height,
            width,
            num_frames,
            pipeline.model.vae.config.scale_factor_spatial,
            pipeline.model.vae.config.scale_factor_temporal,
            pipeline.patch_height,
            pipeline.patch_width,
        )
    )
    pipeline.compile(
        compile_config=custom_config_path,
        parallel=parallel_compile,
        height=height,
        width=width,
        num_frames=num_frames,
        use_onnx_subfunctions=use_onnx_subfunctions,
    )

    set_module_device_ids(pipeline)

    # Step 2: Check inputs
    pipeline.model.check_inputs(
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds,
        negative_prompt_embeds,
        callback_on_step_end_tensor_inputs,
        guidance_scale_2,
    )

    if num_frames % pipeline.model.vae.config.scale_factor_temporal != 1:
        num_frames = (
            num_frames
            // pipeline.model.vae.config.scale_factor_temporal
            * pipeline.model.vae.config.scale_factor_temporal
            + 1
        )
    num_frames = max(num_frames, 1)

    if pipeline.model.config.boundary_ratio is not None and guidance_scale_2 is None:
        guidance_scale_2 = guidance_scale

    pipeline._guidance_scale = guidance_scale
    pipeline._guidance_scale_2 = guidance_scale_2
    pipeline._attention_kwargs = attention_kwargs
    pipeline._current_timestep = None
    pipeline._interrupt = False

    # Step 3: Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Step 4: Encode input prompt(using CPU text encoder for now)
    prompt_embeds, negative_prompt_embeds = pipeline.model.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipeline.do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    # Get PyTorch reference prompt embeddings
    # For standard WAN pipeline, CFG is determined by presence of negative prompts
    do_classifier_free_guidance = negative_prompt is not None
    pytorch_prompt_embeds, pytorch_negative_prompt_embeds = pytorch_pipeline.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    transformer_dtype = pipeline.transformer.model.transformer_high.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    pytorch_prompt_embeds = pytorch_prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
        pytorch_negative_prompt_embeds = pytorch_negative_prompt_embeds.to(transformer_dtype)

    # Step 5: Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # Step 6: Prepare latent variables
    num_channels_latents = pipeline.transformer.model.config.in_channels
    latents = pipeline.model.prepare_latents(
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

    # Step 7: Setup transformer inference session
    if pipeline.transformer.qpc_session is None:
        pipeline.transformer.qpc_session = QAICInferenceSession(
            str(pipeline.transformer.qpc_path), device_ids=pipeline.transformer.device_ids
        )

    output_buffer = {
        "output": np.random.rand(
            batch_size,
            pipeline.cl,
            constants.WAN_DIT_OUT_CHANNELS,
        ).astype(np.int32),
    }
    pipeline.transformer.qpc_session.set_buffers(output_buffer)
    transformer_perf = []

    # Step 8: Denoising loop with transformer MAD validation
    if pipeline.model.config.boundary_ratio is not None:
        boundary_timestep = pipeline.model.config.boundary_ratio * pipeline.scheduler.config.num_train_timesteps
    else:
        boundary_timestep = None

    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order

    with pipeline.model.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipeline._interrupt:
                continue

            pipeline._current_timestep = t

            # Determine which transformer to use (high or low noise)
            if boundary_timestep is None or t >= boundary_timestep:
                # High-noise stage
                current_model = pipeline.transformer.model.transformer_high
                pytorch_current_model = pytorch_pipeline.transformer
                model_type = torch.ones(1, dtype=torch.int64)
                model_name = "transformer_high"
            else:
                # Low-noise stage
                current_model = pipeline.transformer.model.transformer_low
                pytorch_current_model = pytorch_pipeline.transformer_2
                model_type = torch.ones(2, dtype=torch.int64)
                model_name = "transformer_low"

            latent_model_input = latents.to(transformer_dtype)
            if pipeline.model.config.expand_timesteps:
                temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                timestep = t.expand(latents.shape[0])

            batch_size, num_channels, num_frames, height, width = latents.shape
            p_t, p_h, p_w = current_model.config.patch_size
            post_patch_num_frames = num_frames // p_t
            post_patch_height = height // p_h
            post_patch_width = width // p_w

            # Prepare transformer inputs
            rotary_emb = current_model.rope(latent_model_input)
            rotary_emb = torch.cat(rotary_emb, dim=0)
            ts_seq_len = None
            timestep = timestep.flatten()

            temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = current_model.condition_embedder(
                timestep, prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=ts_seq_len
            )

            timestep_proj = timestep_proj.unflatten(1, (6, -1))

            # Prepare inputs for QAIC inference
            inputs_aic = {
                "hidden_states": latents.detach().numpy(),
                "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
                "rotary_emb": rotary_emb.detach().numpy(),
                "temb": temb.detach().numpy(),
                "timestep_proj": timestep_proj.detach().numpy(),
                "tsp": model_type.detach().numpy(),
            }

            # PyTorch reference inference (standard WAN transformer has different signature)
            noise_pred_torch = pytorch_current_model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=pytorch_prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

            # QAIC inference
            with current_model.cache_context("cond"):
                start_transformer_step_time = time.time()
                outputs = pipeline.transformer.qpc_session.run(inputs_aic)
                end_transformer_step_time = time.time()
                transformer_perf.append(end_transformer_step_time - start_transformer_step_time)

                hidden_states = torch.tensor(outputs["output"])
                hidden_states = hidden_states.reshape(
                    batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                )
                hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                noise_pred = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

            # Transformer MAD validation
            print(f" Performing MAD validation for {model_name} at step {i}...")
            mad_validator.validate_module_mad(
                noise_pred_torch.detach().cpu().numpy(),
                noise_pred.detach().cpu().numpy(),
                model_name,
                f"step {i} (t={t.item():.1f})",
            )

            # Update latents using scheduler
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Update progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    # Step 9: Decode latents to video (using CPU VAE for now)
    if not output_type == "latent":
        latents = latents.to(pipeline.vae_decode.dtype)
        latents_mean = (
            torch.tensor(pipeline.vae_decode.config.latents_mean)
            .view(1, pipeline.vae_decode.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipeline.vae_decode.config.latents_std).view(
            1, pipeline.vae_decode.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean

        video = pipeline.model.vae.decode(latents, return_dict=False)[0]

        video = pipeline.model.video_processor.postprocess_video(video.detach())
    else:
        video = latents

    # Build performance metrics
    perf_metrics = [
        ModulePerf(module_name="transformer", perf=transformer_perf),
    ]

    return QEffPipelineOutput(
        pipeline_module=perf_metrics,
        images=video,
    )


@pytest.fixture(scope="session")
def wan_pipeline():
    """Setup compiled WAN pipeline for testing with LoRA adapters and 2 layers total"""
    config = INITIAL_TEST_CONFIG["model_setup"]

    def load_wan_lora(path: str):
        return _convert_non_diffusers_wan_lora_to_diffusers(safetensors.torch.load_file(path))

    # Download and load LoRA adapters
    high_noise_lora_path = hf_hub_download(
        repo_id="lightx2v/Wan2.2-Lightning",
        filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
    )
    low_noise_lora_path = hf_hub_download(
        repo_id="lightx2v/Wan2.2-Lightning",
        filename="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
    )

    # Load PyTorch reference pipeline
    pytorch_pipeline = WanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

    # Load into the transformers
    pytorch_pipeline.transformer.load_lora_adapter(load_wan_lora(high_noise_lora_path), adapter_name="high_noise")
    pytorch_pipeline.transformer.set_adapters(["high_noise"], weights=[1.0])

    pytorch_pipeline.transformer_2.load_lora_adapter(load_wan_lora(low_noise_lora_path), adapter_name="low_noise")
    pytorch_pipeline.transformer_2.set_adapters(["low_noise"], weights=[1.0])

    # ### for 2 layer model
    pytorch_pipeline.transformer.config.num_layers = config["num_transformer_layers_high"]
    pytorch_pipeline.transformer_2.config.num_layers = config["num_transformer_layers_low"]
    original_blocks = pytorch_pipeline.transformer.blocks
    org_blocks = pytorch_pipeline.transformer_2.blocks
    pytorch_pipeline.transformer.blocks = torch.nn.ModuleList(
        [original_blocks[i] for i in range(0, pytorch_pipeline.transformer.config.num_layers)]
    )
    pytorch_pipeline.transformer_2.blocks = torch.nn.ModuleList(
        [org_blocks[i] for i in range(0, pytorch_pipeline.transformer_2.config.num_layers)]
    )

    # Load QEff WAN pipeline
    pipeline = QEffWanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

    # Load LoRA adapters into transformers
    pipeline.transformer.model.transformer_high.load_lora_adapter(
        load_wan_lora(high_noise_lora_path), adapter_name="high_noise"
    )
    pipeline.transformer.model.transformer_high.set_adapters(["high_noise"], weights=[1.0])
    pipeline.transformer.model.transformer_low.load_lora_adapter(
        load_wan_lora(low_noise_lora_path), adapter_name="low_noise"
    )
    pipeline.transformer.model.transformer_low.set_adapters(["low_noise"], weights=[1.0])

    # Reduce to 1 layer (1 high, 1 low) for testing
    pipeline.transformer.model.transformer_high.config.num_layers = config["num_transformer_layers_high"]
    pipeline.transformer.model.transformer_low.config.num_layers = config["num_transformer_layers_low"]

    original_blocks_high = pipeline.transformer.model.transformer_high.blocks
    original_blocks_low = pipeline.transformer.model.transformer_low.blocks

    pipeline.transformer.model.transformer_high.blocks = torch.nn.ModuleList(
        [original_blocks_high[i] for i in range(0, config["num_transformer_layers_high"])]
    )
    pipeline.transformer.model.transformer_low.blocks = torch.nn.ModuleList(
        [original_blocks_low[i] for i in range(0, config["num_transformer_layers_low"])]
    )

    return pipeline, pytorch_pipeline


@pytest.mark.diffusion_models
@pytest.mark.on_qaic
@pytest.mark.wan
def test_wan_pipeline(wan_pipeline):
    """
    Comprehensive WAN pipeline test that focuses on transformer validation:
    - 192x320 resolution - 2 transformer layers total (1 high + 1 low)
    - MAD validation for transformer modules only
    - Functional video generation test
    - Export/compilation checks for transformer
    - Returns QEffPipelineOutput with performance metrics
    """
    pipeline, pytorch_pipeline = wan_pipeline
    config = INITIAL_TEST_CONFIG

    # Print test header
    DiffusersTestUtils.print_test_header(
        f"WAN PIPELINE TEST - {config['model_setup']['height']}x{config['model_setup']['width']} Resolution, {config['model_setup']['num_frames']} Frames, 2 Layers Total",
        config,
    )

    # Test parameters
    test_prompt = config["pipeline_params"]["test_prompt"]
    num_inference_steps = config["pipeline_params"]["num_inference_steps"]
    guidance_scale = config["pipeline_params"]["guidance_scale"]
    guidance_scale_2 = config["pipeline_params"]["guidance_scale_2"]
    max_sequence_length = config["pipeline_params"]["max_sequence_length"]
    num_frames = config["model_setup"]["num_frames"]

    # Generate with MAD validation
    generator = torch.manual_seed(42)
    start_time = time.time()

    try:
        # Run the pipeline with integrated MAD validation (focuses on transformer)
        result = wan_pipeline_call_with_mad_validation(
            pipeline,
            pytorch_pipeline,
            height=config["model_setup"]["height"],
            width=config["model_setup"]["width"],
            num_frames=num_frames,
            prompt=test_prompt,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            custom_config_path=CONFIG_PATH,
            generator=generator,
            mad_tolerances=config["mad_validation"]["tolerances"],
            parallel_compile=True,
            return_dict=True,
        )

        execution_time = time.time() - start_time

        # Validate video generation
        if config["pipeline_params"]["validate_gen_video"]:
            assert result is not None, "Pipeline returned None"
            assert hasattr(result, "images"), "Result missing 'images' attribute"
            assert len(result.images) > 0, "No video frames generated"

            generated_video = result.images[0]
            assert len(generated_video) == num_frames, f"Expected {num_frames} frames, got {len(generated_video)}"

            # Validate first frame properties
            first_frame = generated_video[0]
            expected_size = (config["model_setup"]["width"], config["model_setup"]["height"])

            # Convert numpy array to PIL Image if needed for validation
            if isinstance(first_frame, np.ndarray):
                from PIL import Image

                if first_frame.dtype != np.uint8:
                    first_frame = (first_frame * 255).astype(np.uint8)
                if len(first_frame.shape) == 3 and first_frame.shape[0] == 3:
                    first_frame = first_frame.transpose(1, 2, 0)
                first_frame = Image.fromarray(first_frame)

            # Validate video frame properties
            frame_validation = DiffusersTestUtils.validate_image_generation(
                first_frame, expected_size, config["pipeline_params"]["min_video_variance"]
            )

            print("\n VIDEO VALIDATION PASSED")
            print(f"   - Frame count: {len(generated_video)}")
            print(f"   - Frame size: {frame_validation['size']}")
            print(f"   - Frame mode: {frame_validation['mode']}")
            print(f"   - Frame variance: {frame_validation['variance']:.2f}")
            print(f"   - Mean pixel value: {frame_validation['mean_pixel_value']:.2f}")

            # Save result as video
            frames = result.images[0]
            export_to_video(frames, "test_wan_output_t2v.mp4", fps=16)
            print("\n VIDEO SAVED: test_wan_output_t2v.mp4")
            print(result)

        if config["validation_checks"]["onnx_export"]:
            # Check if transformer ONNX file exists
            print("\n ONNX Export Validation:")
            if hasattr(pipeline.transformer, "onnx_path") and pipeline.transformer.onnx_path:
                DiffusersTestUtils.check_file_exists(str(pipeline.transformer.onnx_path), "transformer ONNX")

        if config["validation_checks"]["compilation"]:
            # Check if transformer QPC file exists
            print("\n Compilation Validation:")
            if hasattr(pipeline.transformer, "qpc_path") and pipeline.transformer.qpc_path:
                DiffusersTestUtils.check_file_exists(str(pipeline.transformer.qpc_path), "transformer QPC")

        # Print test summary
        print(f"\nTotal execution time: {execution_time:.4f}s")
        print(" WAN TRANSFORMER TEST COMPLETED SUCCESSFULLY")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        raise


if __name__ == "__main__":
    # This allows running the test file directly for debugging
    pytest.main([__file__, "-v", "-s", "-m", "wan"])
# pytest tests/diffusers/test_wan.py -m wan -v -s --tb=short
