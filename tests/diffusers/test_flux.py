# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pytest
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from QEfficient import QEffFluxPipeline
from QEfficient.diffusers.pipelines.pipeline_utils import (
    ModulePerf,
    QEffPipelineOutput,
)
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils._utils import load_json
from tests.diffusers.diffusers_utils import DiffusersTestUtils, MADValidator

# Test Configuration for 256x256 resolution with 2 layers # update mad tolerance
CONFIG_PATH = "tests/diffusers/flux_test_config.json"
INITIAL_TEST_CONFIG = load_json(CONFIG_PATH)


def flux_pipeline_call_with_mad_validation(
    pipeline,
    pytorch_pipeline,
    height: int = 256,
    width: int = 256,
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
    use_onnx_subfunctions: bool = False,
    parallel_compile: bool = False,
    mad_tolerances: Dict[str, float] = None,
):
    """
    Pipeline call function that replicates the exact flow of pipeline_flux.py.__call__()
    while adding comprehensive MAD validation at each step.

    This function follows the EXACT same structure as QEffFluxPipeline.__call__()
    but adds MAD validation hooks throughout the process.
    """
    # Initialize MAD validator
    mad_validator = MADValidator(tolerances=mad_tolerances)

    device = "cpu"

    # Step 1: Load configuration, compile models
    pipeline.compile(
        compile_config=custom_config_path,
        parallel=parallel_compile,
        use_onnx_subfunctions=use_onnx_subfunctions,
        height=height,
        width=width,
    )

    # Validate all inputs
    pipeline.model.check_inputs(
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

    # Set pipeline attributes
    pipeline._guidance_scale = guidance_scale
    pipeline._interrupt = False
    batch_size = INITIAL_TEST_CONFIG["modules"]["transformer"]["specializations"]["batch_size"]

    # Step 3: Encode prompts with both text encoders
    # Use pipeline's encode_prompt method
    (t5_qaic_prompt_embeds, clip_qaic_pooled_prompt_embeds, text_ids, text_encoder_perf) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    (t5_torch_prompt_embeds, clip_torch_pooled_prompt_embeds, text_ids) = pytorch_pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    # Deactivate text encoder qpc sessions
    pipeline.text_encoder.qpc_session.deactivate()
    pipeline.text_encoder_2.qpc_session.deactivate()

    # MAD Validation for Text Encoders
    print(" Performing MAD validation for text encoders...")
    mad_validator.validate_module_mad(
        clip_qaic_pooled_prompt_embeds, clip_torch_pooled_prompt_embeds, module_name="clip_text_encoder"
    )
    mad_validator.validate_module_mad(t5_torch_prompt_embeds, t5_qaic_prompt_embeds, "t5_text_encoder")

    # Step 4: Prepare timesteps for denoising
    timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_inference_steps, device, timesteps)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)

    # Step 5: Prepare initial latents
    num_channels_latents = pipeline.transformer.model.config.in_channels // 4
    latents, latent_image_ids = pipeline.model.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        t5_qaic_prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # Step 6: Initialize transformer inference session
    if pipeline.transformer.qpc_session is None:
        pipeline.transformer.qpc_session = QAICInferenceSession(
            str(pipeline.transformer.qpc_path), device_ids=pipeline.transformer.device_ids
        )

    # Calculate compressed latent dimension (cl) for transformer buffer allocation
    from QEfficient.diffusers.pipelines.pipeline_utils import calculate_compressed_latent_dimension

    cl, _, _ = calculate_compressed_latent_dimension(height, width, pipeline.model.vae_scale_factor)

    # Allocate output buffer for transformer
    output_buffer = {
        "output": np.random.rand(batch_size, cl, pipeline.transformer.model.config.in_channels).astype(np.float32),
    }
    pipeline.transformer.qpc_session.set_buffers(output_buffer)

    transformer_perf = []
    pipeline.scheduler.set_begin_index(0)

    # Step 7: Denoising loop
    with pipeline.model.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipeline._interrupt:
                continue

            # Prepare timestep embedding
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            temb = pipeline.transformer.model.time_text_embed(timestep, clip_qaic_pooled_prompt_embeds)

            # Compute AdaLN embeddings for dual transformer blocks
            adaln_emb = []
            for block_idx in range(len(pipeline.transformer.model.transformer_blocks)):
                block = pipeline.transformer.model.transformer_blocks[block_idx]
                f1 = block.norm1.linear(block.norm1.silu(temb)).chunk(6, dim=1)
                f2 = block.norm1_context.linear(block.norm1_context.silu(temb)).chunk(6, dim=1)
                adaln_emb.append(torch.cat(list(f1) + list(f2)))
            adaln_dual_emb = torch.stack(adaln_emb)

            # Compute AdaLN embeddings for single transformer blocks
            adaln_emb = []
            for block_idx in range(len(pipeline.transformer.model.single_transformer_blocks)):
                block = pipeline.transformer.model.single_transformer_blocks[block_idx]
                f1 = block.norm.linear(block.norm.silu(temb)).chunk(3, dim=1)
                adaln_emb.append(torch.cat(list(f1)))
            adaln_single_emb = torch.stack(adaln_emb)

            # Compute output AdaLN embedding
            temp = pipeline.transformer.model.norm_out
            adaln_out = temp.linear(temp.silu(temb))

            # Normalize timestep to [0, 1] range
            timestep = timestep / 1000

            # Prepare all inputs for transformer inference
            inputs_aic = {
                "hidden_states": latents.detach().numpy(),
                "encoder_hidden_states": t5_qaic_prompt_embeds.detach().numpy(),
                "pooled_projections": clip_qaic_pooled_prompt_embeds.detach().numpy(),
                "timestep": timestep.detach().numpy(),
                "img_ids": latent_image_ids.detach().numpy(),
                "txt_ids": text_ids.detach().numpy(),
                "adaln_emb": adaln_dual_emb.detach().numpy(),
                "adaln_single_emb": adaln_single_emb.detach().numpy(),
                "adaln_out": adaln_out.detach().numpy(),
            }

            # MAD Validation for Transformer - PyTorch reference inference
            noise_pred_torch = pytorch_pipeline.transformer(
                hidden_states=latents,
                encoder_hidden_states=t5_torch_prompt_embeds,
                pooled_projections=clip_torch_pooled_prompt_embeds,
                timestep=torch.tensor(timestep),
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                return_dict=False,
            )[0]

            # Run transformer inference and measure time
            start_transformer_step_time = time.time()
            outputs = pipeline.transformer.qpc_session.run(inputs_aic)
            end_transformer_step_time = time.time()
            transformer_perf.append(end_transformer_step_time - start_transformer_step_time)

            noise_pred = torch.from_numpy(outputs["output"])

            # Transformer MAD validation
            mad_validator.validate_module_mad(
                noise_pred_torch.detach().cpu().numpy(),
                outputs["output"],
                "transformer",
                f"step {i} (t={t.item():.1f})",
            )

            # Update latents using scheduler
            latents_dtype = latents.dtype
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Handle dtype mismatch
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

            # Update progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    # Step 8: Decode latents to images
    if output_type == "latent":
        image = latents
        vae_decode_perf = 0.0  # No VAE decoding for latent output
    else:
        # Unpack and denormalize latents
        latents = pipeline.model._unpack_latents(latents, height, width, pipeline.model.vae_scale_factor)

        # Denormalize latents
        latents = (latents / pipeline.vae_decode.model.scaling_factor) + pipeline.vae_decode.model.shift_factor
        # Initialize VAE decoder inference session
        if pipeline.vae_decode.qpc_session is None:
            pipeline.vae_decode.qpc_session = QAICInferenceSession(
                str(pipeline.vae_decode.qpc_path), device_ids=pipeline.vae_decode.device_ids
            )

        # Allocate output buffer for VAE decoder
        output_buffer = {"sample": np.random.rand(batch_size, 3, height, width).astype(np.float32)}
        pipeline.vae_decode.qpc_session.set_buffers(output_buffer)

        # MAD Validation for VAE
        # PyTorch reference inference
        image_torch = pytorch_pipeline.vae.decode(latents, return_dict=False)[0]

        # Run VAE decoder inference and measure time
        inputs = {"latent_sample": latents.numpy()}
        start_decode_time = time.time()
        image = pipeline.vae_decode.qpc_session.run(inputs)
        end_decode_time = time.time()
        vae_decode_perf = end_decode_time - start_decode_time

        # VAE MAD validation
        mad_validator.validate_module_mad(image_torch.detach().cpu().numpy(), image["sample"], "vae_decoder")

        # Post-process image
        image_tensor = torch.from_numpy(image["sample"])
        image = pipeline.model.image_processor.postprocess(image_tensor, output_type=output_type)

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


@pytest.fixture(scope="session")
def flux_pipeline():
    """Setup compiled Flux pipeline for testing"""
    config = INITIAL_TEST_CONFIG["model_setup"]

    pipeline = QEffFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")

    # Reduce to 2 layers for testing
    original_blocks = pipeline.transformer.model.transformer_blocks
    org_single_blocks = pipeline.transformer.model.single_transformer_blocks

    pipeline.transformer.model.config["num_layers"] = config["num_transformer_layers"]
    pipeline.transformer.model.config["num_single_layers"] = config["num_single_layers"]
    pipeline.transformer.model.transformer_blocks = torch.nn.ModuleList(
        [original_blocks[i] for i in range(0, pipeline.transformer.model.config["num_layers"])]
    )
    pipeline.transformer.model.single_transformer_blocks = torch.nn.ModuleList(
        [org_single_blocks[i] for i in range(0, pipeline.transformer.model.config["num_single_layers"])]
    )

    ### Pytorch pipeline
    pytorch_pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
    original_blocks_pt = pytorch_pipeline.transformer.transformer_blocks
    org_single_blocks_pt = pytorch_pipeline.transformer.single_transformer_blocks
    pytorch_pipeline.transformer.transformer_blocks = torch.nn.ModuleList(
        [original_blocks_pt[i] for i in range(0, pipeline.transformer.model.config["num_layers"])]
    )
    pytorch_pipeline.transformer.single_transformer_blocks = torch.nn.ModuleList(
        [org_single_blocks_pt[i] for i in range(0, pipeline.transformer.model.config["num_single_layers"])]
    )
    return pipeline, pytorch_pipeline


@pytest.mark.diffusion_models
@pytest.mark.on_qaic
def test_flux_pipeline(flux_pipeline):
    """
    Comprehensive Flux pipeline test that follows the exact same flow as pipeline_flux.py:
    - 256x256 resolution - 2 transformer layers
    - MAD validation
    - Functional image generation test
    - Export/compilation checks
    - Returns QEffPipelineOutput with performance metrics
    """
    pipeline, pytorch_pipeline = flux_pipeline
    config = INITIAL_TEST_CONFIG

    # Print test header
    DiffusersTestUtils.print_test_header(
        f"FLUX PIPELINE TEST - {config['model_setup']['height']}x{config['model_setup']['width']} Resolution, {config['model_setup']['num_transformer_layers']} Layers",
        config,
    )

    # Test parameters
    test_prompt = config["pipeline_params"]["test_prompt"]
    num_inference_steps = config["pipeline_params"]["num_inference_steps"]
    guidance_scale = config["pipeline_params"]["guidance_scale"]
    max_sequence_length = config["pipeline_params"]["max_sequence_length"]

    # Generate with MAD validation
    generator = torch.manual_seed(42)
    start_time = time.time()

    try:
        # Run the pipeline with integrated MAD validation (follows exact pipeline flow)
        result = flux_pipeline_call_with_mad_validation(
            pipeline,
            pytorch_pipeline,
            height=config["model_setup"]["height"],
            width=config["model_setup"]["width"],
            prompt=test_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            custom_config_path=CONFIG_PATH,
            generator=generator,
            mad_tolerances=config["mad_validation"]["tolerances"],
            use_onnx_subfunctions=config["pipeline_params"]["use_onnx_subfunctions"],
            parallel_compile=True,
            return_dict=True,
        )

        execution_time = time.time() - start_time

        # Validate image generation
        if config["pipeline_params"]["validate_gen_img"]:
            assert result is not None, "Pipeline returned None"
            assert hasattr(result, "images"), "Result missing 'images' attribute"
            assert len(result.images) > 0, "No images generated"

            generated_image = result.images[0]
            expected_size = (config["model_setup"]["height"], config["model_setup"]["width"])
            # Validate image properties using utilities
            image_validation = DiffusersTestUtils.validate_image_generation(
                generated_image, expected_size, config["pipeline_params"]["min_image_variance"]
            )

            print("\n IMAGE VALIDATION PASSED")
            print(f"   - Size: {image_validation['size']}")
            print(f"   - Mode: {image_validation['mode']}")
            print(f"   - Variance: {image_validation['variance']:.2f}")
            print(f"   - Mean pixel value: {image_validation['mean_pixel_value']:.2f}")
            file_path = "test_flux_256x256_2layers.png"
            # Save test image
            generated_image.save(file_path)

            if os.path.exists(file_path):
                print(f"Image saved successfully at: {file_path}")
            else:
                print("Image was not saved.")

        if config["validation_checks"]["onnx_export"]:
            # Check if ONNX files exist (basic check)
            print("\n ONNX Export Validation:")
            for module_name in ["text_encoder", "text_encoder_2", "transformer", "vae_decode"]:
                module_obj = getattr(pipeline, module_name, None)
                if module_obj and hasattr(module_obj, "onnx_path") and module_obj.onnx_path:
                    DiffusersTestUtils.check_file_exists(str(module_obj.onnx_path), f"{module_name} ONNX")

        if config["validation_checks"]["compilation"]:
            # Check if QPC files exist (basic check)
            print("\n Compilation Validation:")
            for module_name in ["text_encoder", "text_encoder_2", "transformer", "vae_decode"]:
                module_obj = getattr(pipeline, module_name, None)
                if module_obj and hasattr(module_obj, "qpc_path") and module_obj.qpc_path:
                    DiffusersTestUtils.check_file_exists(str(module_obj.qpc_path), f"{module_name} QPC")

        # Print test summary using utilities
        print(f"\nTotal execution time: {execution_time:.4f}s")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        raise


if __name__ == "__main__":
    # This allows running the test file directly for debugging
    pytest.main([__file__, "-v", "-s", "-m", "flux"])
# pytest tests/diffusers/test_flux.py -m flux -v -s --tb=short