# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Test for wan pipeline
# TODO : 1. Add pytest for call method
"""

import copy
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
from transformers import T5TokenizerFast, UMT5EncoderModel

from QEfficient import QEffWanPipeline
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ModulePerf,
    QEffPipelineOutput,
    calculate_latent_dimensions_with_frames,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants
from QEfficient.utils._utils import load_json
from tests.diffusers.diffusers_utils import DiffusersTestUtils, MADValidator

# Test Configuration for 48 x 64 resolution with 1 layer
CONFIG_PATH = "tests/diffusers/wan_test_config.json"
NON_UNIFIED_CONFIG_PATH = "tests/diffusers/wan_test_non_unified_config.json"
INITIAL_TEST_CONFIG = load_json(CONFIG_PATH)
NON_UNIFIED_TEST_CONFIG = load_json(NON_UNIFIED_CONFIG_PATH)
TEST_SEED = 42


def wan_pipeline_call_with_mad_validation(
    pipeline,
    pytorch_pipeline,
    height: int = 48,
    width: int = 64,
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
    cache_threshold_high: Optional[float] = None,
    cache_threshold_low: Optional[float] = None,
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

    if pipeline.use_unified:
        transformer_dtype = pipeline.transformer.model.transformer_high.dtype
    else:
        transformer_dtype = pipeline.transformer_high.model.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    pytorch_prompt_embeds = pytorch_prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
        pytorch_negative_prompt_embeds = pytorch_negative_prompt_embeds.to(transformer_dtype)

    # Step 5: Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    reference_scheduler = copy.deepcopy(pipeline.scheduler)

    # Step 6: Prepare latent variables
    num_channels_latents = pipeline.model.transformer.config.in_channels
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
    latents_torch = latents.clone()

    # Step 7: Setup transformer inference session(s)
    output_buffer = {"output": np.zeros((batch_size, pipeline.cl, constants.WAN_DIT_OUT_CHANNELS), dtype=np.int32)}
    if pipeline.use_unified:
        if pipeline.transformer.qpc_session is None:
            pipeline.transformer.qpc_session = QAICInferenceSession(
                str(pipeline.transformer.qpc_path), device_ids=pipeline.transformer.device_ids
            )
        pipeline.transformer.qpc_session.set_buffers(output_buffer)
    else:
        if pipeline.transformer_high.qpc_session is None:
            pipeline.transformer_high.qpc_session = QAICInferenceSession(
                str(pipeline.transformer_high.qpc_path), device_ids=pipeline.transformer_high.device_ids
            )
        if pipeline.transformer_low.qpc_session is None:
            pipeline.transformer_low.qpc_session = QAICInferenceSession(
                str(pipeline.transformer_low.qpc_path), device_ids=pipeline.transformer_low.device_ids
            )
        pipeline.transformer_high.qpc_session.set_buffers(output_buffer)
        pipeline.transformer_low.qpc_session.set_buffers(output_buffer)
        if getattr(pipeline, "enable_first_block_cache", False):
            pipeline.transformer_high.qpc_session.skip_buffers(
                [
                    tensor_name
                    for tensor_name in (
                        pipeline.transformer_high.qpc_session.input_names
                        + pipeline.transformer_high.qpc_session.output_names
                    )
                    if tensor_name.startswith("prev_") or tensor_name.endswith("_RetainedState")
                ]
            )
            pipeline.transformer_low.qpc_session.skip_buffers(
                [
                    tensor_name
                    for tensor_name in (
                        pipeline.transformer_low.qpc_session.input_names
                        + pipeline.transformer_low.qpc_session.output_names
                    )
                    if tensor_name.startswith("prev_") or tensor_name.endswith("_RetainedState")
                ]
            )
    transformer_perf = []

    # Step 8: Denoising loop with transformer MAD validation
    if pipeline.model.config.boundary_ratio is not None:
        boundary_timestep = pipeline.model.config.boundary_ratio * pipeline.scheduler.config.num_train_timesteps
    else:
        boundary_timestep = None

    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
    low_stage_counter = 0

    with pipeline.model.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipeline._interrupt:
                continue

            pipeline._current_timestep = t

            # Determine which transformer to use (high or low noise)
            if boundary_timestep is None or t >= boundary_timestep:
                # High-noise stage
                pytorch_current_model = pytorch_pipeline.transformer
                if pipeline.use_unified:
                    current_model = pipeline.transformer.model.transformer_high
                    current_qpc_session = pipeline.transformer.qpc_session
                    model_type = torch.ones(1, dtype=torch.int64)
                else:
                    current_model = pipeline.transformer_high.model
                    current_qpc_session = pipeline.transformer_high.qpc_session
                    model_type = None
            else:
                # Low-noise stage
                pytorch_current_model = pytorch_pipeline.transformer_2
                if pipeline.use_unified:
                    current_model = pipeline.transformer.model.transformer_low
                    current_qpc_session = pipeline.transformer.qpc_session
                    model_type = torch.ones(2, dtype=torch.int64)
                else:
                    current_model = pipeline.transformer_low.model
                    current_qpc_session = pipeline.transformer_low.qpc_session
                    model_type = None
                    low_stage_counter += 1

            latent_model_input = latents.to(transformer_dtype)
            if pipeline.model.config.expand_timesteps:
                temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                timestep = t.expand(latents.shape[0])

            batch_size, num_channels, latent_num_frames, latent_height, latent_width = latents.shape
            p_t, p_h, p_w = current_model.config.patch_size
            post_patch_num_frames = latent_num_frames // p_t
            post_patch_height = latent_height // p_h
            post_patch_width = latent_width // p_w

            # Prepare transformer inputs
            rotary_emb = pytorch_current_model.rope(latent_model_input)
            rotary_emb = torch.cat(rotary_emb, dim=0)
            ts_seq_len = None
            timestep = timestep.flatten()

            temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
                pytorch_current_model.condition_embedder(
                    timestep, prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=ts_seq_len
                )
            )

            timestep_proj = timestep_proj.unflatten(1, (6, -1))

            # Prepare inputs for QAIC inference
            inputs_aic = {
                "hidden_states": latent_model_input.detach().numpy(),
                "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
                "rotary_emb": rotary_emb.detach().numpy(),
                "temb": temb.detach().numpy(),
                "timestep_proj": timestep_proj.detach().numpy(),
            }
            if model_type is not None:
                inputs_aic["tsp"] = model_type.detach().numpy()
            if getattr(pipeline, "enable_first_block_cache", False):
                if boundary_timestep is None or t >= boundary_timestep:
                    stage_cache_threshold = 0.0 if cache_threshold_high is None else cache_threshold_high
                else:
                    low_threshold = 0.0 if cache_threshold_low is None else cache_threshold_low
                    stage_cache_threshold = 0.0 if low_stage_counter < 3 else low_threshold
                inputs_aic["cache_threshold"] = np.array(stage_cache_threshold, dtype=np.float32)

            # PyTorch reference inference (standard WAN transformer has different signature)
            noise_pred_torch = pytorch_current_model(
                hidden_states=latents_torch.to(transformer_dtype),
                timestep=timestep,
                encoder_hidden_states=pytorch_prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

            # QAIC inference
            with current_model.cache_context("cond"):
                start_transformer_step_time = time.time()
                outputs = current_qpc_session.run(inputs_aic)
                end_transformer_step_time = time.time()
                transformer_perf.append(end_transformer_step_time - start_transformer_step_time)

                hidden_states = torch.tensor(outputs["output"])
                hidden_states = hidden_states.reshape(
                    batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
                )
                hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
                noise_pred = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

            # Update latents using scheduler
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents_torch = reference_scheduler.step(noise_pred_torch, t, latents_torch, return_dict=False)[0]

            # Update progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    print(" Performing MAD validation for transformer final latent output...")
    mad_validator.validate_module_mad(
        latents_torch.detach().cpu().numpy(),
        latents.detach().cpu().numpy(),
        "transformer",
        f"final latents after {num_inference_steps} steps",
    )

    if pipeline.use_unified:
        pipeline.transformer.qpc_session.deactivate()  # deactivate transformer qpc session
    else:
        pipeline.transformer_high.qpc_session.deactivate()
        pipeline.transformer_low.qpc_session.deactivate()
    # Step 9: Decode latents to video QAIC VAE decoder
    latents = latents.to(pipeline.vae_decoder.model.dtype)
    latents_mean = (
        torch.tensor(pipeline.vae_decoder.model.config.latents_mean)
        .view(1, pipeline.vae_decoder.model.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipeline.vae_decoder.model.latents_std).view(
        1, pipeline.vae_decoder.model.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean

    # Initialize VAE decoder inference session
    if pipeline.vae_decoder.qpc_session is None:
        pipeline.vae_decoder.qpc_session = QAICInferenceSession(
            str(pipeline.vae_decoder.qpc_path), device_ids=pipeline.vae_decoder.device_ids
        )
    # MAD Validation for VAE decoder - PyTorch reference inference
    video_torch = pytorch_pipeline.vae.decode(latents, return_dict=False)[0]

    # Allocate output buffer for VAE decoder
    output_buffer = {"sample": np.zeros((batch_size, 3, num_frames, height, width), dtype=np.int32)}
    pipeline.vae_decoder.qpc_session.set_buffers(output_buffer)

    # Run VAE decoder inference and measure time
    inputs = {"latent_sample": latents.numpy()}
    start_decode_time = time.perf_counter()
    video = pipeline.vae_decoder.qpc_session.run(inputs)
    end_decode_time = time.perf_counter()
    vae_decoder_perf = end_decode_time - start_decode_time
    pipeline.vae_decoder.qpc_session.deactivate()  # deactivate vae decoder qpc session

    # VAE decoder MAD validation
    print(" Performing MAD validation for VAE decoder...")
    mad_validator.validate_module_mad(
        video_torch.detach().cpu().numpy(), video["sample"], "vae_decoder", "video decoding"
    )

    # Post-process video for output
    video_tensor = torch.from_numpy(video["sample"])
    video = pipeline.model.video_processor.postprocess_video(video_tensor)

    # Build performance metrics
    perf_data = {
        "transformer": transformer_perf,
        "vae_decoder": vae_decoder_perf,
    }

    # Convert metrics to pipeline output format.
    perf_metrics = [ModulePerf(module_name=name, perf=perf_data[name]) for name in perf_data.keys()]

    return QEffPipelineOutput(
        pipeline_module=perf_metrics,
        images=video,
    )


def _build_wan_pipeline(use_unified: bool = True, enable_first_block_cache: bool = False):
    """Build the WAN pipeline with random weights/ dummy config."""
    torch.manual_seed(TEST_SEED)
    np.random.seed(TEST_SEED)

    config = INITIAL_TEST_CONFIG["model_setup"]
    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    pipe_cfg = WanPipeline.load_config(model_id)

    vae_config = AutoencoderKLWan.load_config(model_id, subfolder="vae")
    t_config = WanTransformer3DModel.load_config(model_id, subfolder="transformer")
    tiny_vae_config = dict(vae_config)
    # Reduced configs
    tiny_vae_config.update(
        {
            "num_res_blocks": 1,
            "base_dim": 16,
            "dim_mult": [1, 1, 2, 2],
            "z_dim": 16,
            "temperal_downsample": [False, True, True],
        }
    )
    vae = AutoencoderKLWan.from_config(tiny_vae_config)
    # vae = AutoencoderKLWan.from_config(vae_config)  # Uncomment to use full VAE config.

    # Keep transformer shallow for quicker tests.
    t_config["num_layers"] = config["num_transformer_layers_high"]
    transformer_high = WanTransformer3DModel.from_config(t_config)
    transformer_low = WanTransformer3DModel.from_config(t_config)

    # Random-init text encoder and scheduler from config to avoid model weight downloads.
    text_encoder_cfg = UMT5EncoderModel.config_class.from_pretrained(model_id, subfolder="text_encoder")
    text_encoder = UMT5EncoderModel(text_encoder_cfg)
    tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
    scheduler_cfg = FlowMatchEulerDiscreteScheduler.load_config(model_id, subfolder="scheduler")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_cfg)

    pytorch_pipeline = WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        transformer=transformer_high,
        transformer_2=transformer_low,
        boundary_ratio=pipe_cfg.get("boundary_ratio"),
        expand_timesteps=pipe_cfg.get("expand_timesteps", False),
    )
    vae.eval()
    transformer_high.eval()
    transformer_low.eval()
    text_encoder.eval()

    pytorch_pipeline_copy = copy.deepcopy(pytorch_pipeline)
    pipeline = QEffWanPipeline(
        pytorch_pipeline_copy,
        use_unified=use_unified,
        enable_first_block_cache=enable_first_block_cache,
    )

    return pipeline, pytorch_pipeline


@pytest.fixture(scope="session")
def wan_pipeline():
    return _build_wan_pipeline(use_unified=True)


@pytest.fixture(scope="session")
def wan_pipeline_non_unified():
    return _build_wan_pipeline(use_unified=False)


@pytest.fixture(scope="session")
def wan_pipeline_non_unified_first_block_cache():
    return _build_wan_pipeline(use_unified=False, enable_first_block_cache=True)


@pytest.mark.diffusion_models
@pytest.mark.on_qaic
@pytest.mark.wan
def test_wan_pipeline(wan_pipeline):
    _run_wan_pipeline_test_case(
        wan_pipeline,
        INITIAL_TEST_CONFIG,
        CONFIG_PATH,
        "WAN PIPELINE TEST",
    )


@pytest.mark.diffusion_models
@pytest.mark.on_qaic
@pytest.mark.wan
def test_wan_pipeline_non_unified(wan_pipeline_non_unified):
    _run_wan_pipeline_test_case(
        wan_pipeline_non_unified,
        NON_UNIFIED_TEST_CONFIG,
        NON_UNIFIED_CONFIG_PATH,
        "WAN PIPELINE NON-UNIFIED TEST",
    )


@pytest.mark.diffusion_models
@pytest.mark.on_qaic
@pytest.mark.wan
def test_wan_pipeline_non_unified_first_block_cache(wan_pipeline_non_unified_first_block_cache):
    _run_wan_pipeline_test_case(
        wan_pipeline_non_unified_first_block_cache,
        NON_UNIFIED_TEST_CONFIG,
        NON_UNIFIED_CONFIG_PATH,
        "WAN PIPELINE NON-UNIFIED FIRST-BLOCK-CACHE TEST",
        pipeline_call_overrides={
            "cache_threshold_high": 0.0,
            "cache_threshold_low": 0.0,
        },
    )


def _run_wan_pipeline_test_case(
    wan_pipeline_data,
    config,
    compile_config_path: str,
    test_label: str,
    pipeline_call_overrides: Optional[Dict[str, Any]] = None,
):
    """
    Comprehensive WAN pipeline test case runner that focuses on transformer validation:
    - 45p - 48x64 resolution - 2 transformer layers total (1 high + 1 low)
    - MAD validation for transformer modules only
    - Functional video generation test
    - Export/compilation checks for transformer and VAE decoder
    - Returns QEffPipelineOutput with performance metrics
    """
    pipeline, pytorch_pipeline = wan_pipeline_data

    # Print test header
    DiffusersTestUtils.print_test_header(
        f"{test_label} - {config['model_setup']['height']}x{config['model_setup']['width']} Resolution, {config['model_setup']['num_frames']} Frames, 2 Layers Total",
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
    generator = torch.Generator(device="cpu").manual_seed(TEST_SEED)
    start_time = time.time()

    try:
        pipeline_call_overrides = pipeline_call_overrides or {}
        # Run the pipeline with integrated MAD validation
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
            custom_config_path=compile_config_path,
            generator=generator,
            mad_tolerances=config["mad_validation"]["tolerances"],
            use_onnx_subfunctions=config["pipeline_params"]["use_onnx_subfunctions"],
            parallel_compile=True,
            return_dict=True,
            **pipeline_call_overrides,
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
            print("\n ONNX Export Validation:")
            if pipeline.use_unified:
                DiffusersTestUtils.check_file_exists(str(pipeline.transformer.onnx_path), "transformer ONNX")
            else:
                DiffusersTestUtils.check_file_exists(str(pipeline.transformer_high.onnx_path), "transformer_high ONNX")
                DiffusersTestUtils.check_file_exists(str(pipeline.transformer_low.onnx_path), "transformer_low ONNX")

        if config["validation_checks"]["compilation"]:
            print("\n Compilation Validation:")
            if pipeline.use_unified:
                DiffusersTestUtils.check_file_exists(str(pipeline.transformer.qpc_path), "transformer QPC")
            else:
                DiffusersTestUtils.check_file_exists(str(pipeline.transformer_high.qpc_path), "transformer_high QPC")
                DiffusersTestUtils.check_file_exists(str(pipeline.transformer_low.qpc_path), "transformer_low QPC")

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
