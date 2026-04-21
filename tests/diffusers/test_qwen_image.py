# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import time
from typing import Dict, List, Optional, Union

import numpy as np
import pytest
import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from QEfficient import QEFFQwenImagePipeline
from QEfficient.diffusers.pipelines.pipeline_utils import ModulePerf, QEffPipelineOutput
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils._utils import load_json
from tests.diffusers.diffusers_utils import DiffusersTestUtils, MADValidator

CONFIG_PATH = "tests/diffusers/qwen_image_test_config.json"
INITIAL_TEST_CONFIG = load_json(CONFIG_PATH)


class DummyTokenizer:
    model_max_length = 1024


class DummyTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32


def qwen_pipeline_call_with_mad_validation(
    pipeline,
    pytorch_pipeline,
    height: int = 64,
    width: int = 64,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    true_cfg_scale: float = 1.0,
    num_inference_steps: int = 2,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 1.0,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    max_sequence_length: int = 128,
    custom_config_path: Optional[str] = None,
    parallel_compile: bool = True,
    use_onnx_subfunctions: bool = False,
    mad_tolerances: Dict[str, float] = None,
):
    """
    Replicate QEFFQwenImagePipeline.__call__ flow and validate MAD for transformer and VAE.
    """
    mad_validator = MADValidator(tolerances=mad_tolerances)
    device = "cpu"

    # Step 1: Compile/export
    pipeline.compile(
        compile_config=custom_config_path,
        parallel=parallel_compile,
        height=height,
        width=width,
        use_onnx_subfunctions=use_onnx_subfunctions,
    )

    # Step 2: Validate input contract
    pipeline.model.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=max_sequence_length,
    )

    pipeline._guidance_scale = guidance_scale
    pipeline._current_timestep = None
    pipeline._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = 1

    has_neg_prompt = negative_prompt is not None
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

    # Step 3: Use deterministic synthetic prompt embeddings for both QAIC and torch reference.
    torch.manual_seed(123)
    embed_dim = pipeline.transformer.model.config.joint_attention_dim
    qaic_prompt_embeds = torch.randn(
        batch_size * num_images_per_prompt, max_sequence_length, embed_dim, dtype=torch.float32
    )
    qaic_prompt_embeds_mask = torch.ones(batch_size * num_images_per_prompt, max_sequence_length, dtype=torch.int64)

    torch_prompt_embeds = qaic_prompt_embeds.clone()
    torch_prompt_embeds_mask = qaic_prompt_embeds_mask.clone()

    if do_true_cfg:
        qaic_negative_prompt_embeds = torch.randn(
            batch_size * num_images_per_prompt,
            max_sequence_length,
            embed_dim,
            dtype=torch.float32,
        )
        qaic_negative_prompt_embeds_mask = torch.ones(
            batch_size * num_images_per_prompt, max_sequence_length, dtype=torch.int64
        )
    else:
        qaic_negative_prompt_embeds = None
        qaic_negative_prompt_embeds_mask = None

    # Step 4: Latents and timesteps
    num_channels_latents = pipeline.transformer.model.config.in_channels // 4
    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        qaic_prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # QEff path currently uses nested shape format for pos_embed utility.
    img_shapes = [[(1, height // pipeline.vae_scale_factor // 2, width // pipeline.vae_scale_factor // 2)]] * batch_size
    # Diffusers PyTorch transformer expects List[Tuple[int, int, int]].
    torch_img_shapes = [
        (1, height // pipeline.vae_scale_factor // 2, width // pipeline.vae_scale_factor // 2)
    ] * batch_size

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)

    txt_seq_lens = [max_sequence_length]

    if pipeline.transformer.qpc_session is None:
        pipeline.transformer.qpc_session = QAICInferenceSession(str(pipeline.transformer.qpc_path))

    pipeline.scheduler.set_begin_index(0)
    transformer_perf = []

    qaic_image_rotary_emb = pipeline.transformer.model.pos_embed(img_shapes, txt_seq_lens, device="cpu")
    qaic_img_freqs_cos, qaic_img_freqs_sin, qaic_txt_freqs_cos, qaic_txt_freqs_sin = qaic_image_rotary_emb

    img_rotary_emb = torch.cat([qaic_img_freqs_cos, qaic_img_freqs_sin], dim=-1)
    txt_rotary_emb = torch.cat([qaic_txt_freqs_cos, qaic_txt_freqs_sin], dim=-1)

    # Step 5: Denoising + transformer MAD
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipeline.interrupt:
                continue

            timestep = (t.expand(latents.shape[0]) / 1000).detach().numpy().astype(np.float32)

            transformer_inputs = {
                "hidden_states": latents.detach().numpy().astype(np.float32),
                "encoder_hidden_states": qaic_prompt_embeds.detach().numpy().astype(np.float32),
                "encoder_hidden_states_mask": qaic_prompt_embeds_mask.detach().numpy().astype(np.int64),
                "img_rotary_emb": img_rotary_emb.detach().numpy().astype(np.float32),
                "txt_rotary_emb": txt_rotary_emb.detach().numpy().astype(np.float32),
                "timestep": timestep,
            }

            timestep_torch = torch.from_numpy(timestep).to(dtype=torch.float32)
            noise_pred_torch = pytorch_pipeline.transformer(
                hidden_states=latents,
                encoder_hidden_states=torch_prompt_embeds,
                encoder_hidden_states_mask=torch_prompt_embeds_mask,
                timestep=timestep_torch,
                img_shapes=torch_img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]

            start_transformer_step_time = time.perf_counter()
            outputs = pipeline.transformer.qpc_session.run(transformer_inputs)
            end_transformer_step_time = time.perf_counter()
            transformer_perf.append(end_transformer_step_time - start_transformer_step_time)

            mad_validator.validate_module_mad(
                noise_pred_torch.detach().cpu().numpy(),
                outputs["output"],
                module_name="transformer",
                step_info=f"step {i} (t={t.item():.6f})",
            )

            noise_pred = torch.from_numpy(outputs["output"])

            if do_true_cfg:
                transformer_inputs_uncond = {
                    "hidden_states": latents.detach().numpy().astype(np.float32),
                    "encoder_hidden_states": qaic_negative_prompt_embeds.detach().numpy().astype(np.float32),
                    "encoder_hidden_states_mask": qaic_negative_prompt_embeds_mask.detach().numpy().astype(np.int64),
                    "img_rotary_emb": img_rotary_emb.detach().numpy().astype(np.float32),
                    "txt_rotary_emb": txt_rotary_emb.detach().numpy().astype(np.float32),
                    "timestep": timestep,
                }
                neg_noise_pred = pipeline.transformer.qpc_session.run(transformer_inputs_uncond)
                neg_noise_pred = torch.from_numpy(neg_noise_pred["output"])
                comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                noise_pred = comb_pred * (cond_norm / noise_norm)

            latents_dtype = latents.dtype
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    pipeline.transformer.qpc_session.deactivate()

    # Step 6: VAE decode + MAD
    latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    latents = latents.to(pipeline.vae_decoder.model.dtype)

    latents_mean = (
        torch.tensor(pipeline.vae_decoder.model.config.latents_mean)
        .view(1, pipeline.vae_decoder.model.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipeline.vae_decoder.model.config.latents_std).view(
        1, pipeline.vae_decoder.model.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean

    if pipeline.vae_decoder.qpc_session is None:
        pipeline.vae_decoder.qpc_session = QAICInferenceSession(
            str(pipeline.vae_decoder.qpc_path), device_ids=pipeline.vae_decoder.device_ids
        )

    output_buffer = {"sample": np.random.rand(batch_size, 3, 1, height, width).astype(np.int32)}
    pipeline.vae_decoder.qpc_session.set_buffers(output_buffer)

    image_torch = pytorch_pipeline.vae.decode(latents, return_dict=False)[0]

    inputs = {"latent_sample": latents.numpy()}
    start_decode_time = time.perf_counter()
    image = pipeline.vae_decoder.qpc_session.run(inputs)
    end_decode_time = time.perf_counter()
    vae_decoder_perf = end_decode_time - start_decode_time

    mad_validator.validate_module_mad(
        image_torch.detach().cpu().numpy(),
        image["sample"],
        module_name="vae_decoder",
    )

    pipeline.vae_decoder.qpc_session.deactivate()

    image_tensor = torch.from_numpy(image["sample"])
    image_tensor = image_tensor[:, :, 0]
    image = pipeline.image_processor.postprocess(image_tensor, output_type=output_type)

    perf_metrics = [
        ModulePerf(module_name="transformer", perf=transformer_perf),
        ModulePerf(module_name="vae_decoder", perf=vae_decoder_perf),
    ]

    return QEffPipelineOutput(pipeline_module=perf_metrics, images=image)


@pytest.fixture(scope="session")
def qwen_image_pipeline():
    """Setup tiny random-init Qwen Image pipelines for QAIC vs PyTorch validation."""
    torch.manual_seed(42)
    np.random.seed(42)
    config = INITIAL_TEST_CONFIG["model_setup"]

    transformer = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=64,
        out_channels=16,
        num_layers=config["num_transformer_layers"],
        attention_head_dim=config["attention_head_dim"],
        num_attention_heads=config["num_attention_heads"],
        joint_attention_dim=config["joint_attention_dim"],
        axes_dims_rope=(2, 2, 4),
    )

    vae = AutoencoderKLQwenImage(
        base_dim=16,
        z_dim=16,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[False],
    )

    scheduler = FlowMatchEulerDiscreteScheduler()

    pytorch_pipeline = QwenImagePipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=DummyTextEncoder(),
        tokenizer=DummyTokenizer(),
        transformer=transformer,
    )

    pipeline = QEFFQwenImagePipeline(copy.deepcopy(pytorch_pipeline))

    # Match WAN test style: run reference and wrapped modules in eval mode.
    pytorch_pipeline.transformer.eval()
    pytorch_pipeline.vae.eval()
    pipeline.transformer.model.eval()
    pipeline.vae_decoder.model.eval()

    # Align export inputs with tiny test model dims; the production helper uses
    # hardcoded Qwen-Image sizes (e.g., encoder_hidden_dim=3584), which break
    # for this reduced config.
    def _test_get_onnx_params(self):
        bs = 1
        cl = 256
        seq_length = INITIAL_TEST_CONFIG["model_setup"]["max_sequence_length"]
        hidden_dim = self.model.config.in_channels
        encoder_hidden_dim = self.model.config.joint_attention_dim
        rot_dim = sum(self.model.config.axes_dims_rope)

        example_inputs = {
            "hidden_states": torch.randn(bs, cl, hidden_dim, dtype=torch.float32),
            "encoder_hidden_states": torch.randn(bs, seq_length, encoder_hidden_dim, dtype=torch.float32),
            "encoder_hidden_states_mask": torch.ones(bs, seq_length, dtype=torch.int64),
            "txt_seq_lens": torch.tensor([seq_length], dtype=torch.int64),
            "img_rotary_emb": torch.randn(cl, rot_dim, dtype=torch.float32),
            "txt_rotary_emb": torch.randn(seq_length, rot_dim, dtype=torch.float32),
            "timestep": torch.tensor([1.0], dtype=torch.float32),
        }

        dynamic_axes = {
            "hidden_states": {0: "batch_size", 1: "cl"},
            "encoder_hidden_states": {0: "batch_size", 1: "seq_length"},
            "encoder_hidden_states_mask": {0: "batch_size", 1: "seq_length"},
            "img_rotary_emb": {0: "cl"},
            "txt_rotary_emb": {0: "seq_length"},
        }

        return example_inputs, dynamic_axes, ["output"]

    pipeline.transformer.get_onnx_params = _test_get_onnx_params.__get__(
        pipeline.transformer, type(pipeline.transformer)
    )
    return pipeline, pytorch_pipeline


@pytest.mark.qwen_image
@pytest.mark.diffusion_models
@pytest.mark.on_qaic
def test_qwen_image_pipeline(qwen_image_pipeline):
    """Qwen Image pipeline test with transformer and VAE MAD validation."""
    pipeline, pytorch_pipeline = qwen_image_pipeline
    config = INITIAL_TEST_CONFIG

    DiffusersTestUtils.print_test_header(
        f"QWEN IMAGE PIPELINE TEST - {config['model_setup']['height']}x{config['model_setup']['width']} Resolution",
        config,
    )

    generator = torch.manual_seed(42)
    start_time = time.time()

    result = qwen_pipeline_call_with_mad_validation(
        pipeline=pipeline,
        pytorch_pipeline=pytorch_pipeline,
        height=config["model_setup"]["height"],
        width=config["model_setup"]["width"],
        prompt=config["pipeline_params"]["test_prompt"],
        guidance_scale=config["pipeline_params"]["guidance_scale"],
        true_cfg_scale=config["pipeline_params"]["true_cfg_scale"],
        num_inference_steps=config["pipeline_params"]["num_inference_steps"],
        max_sequence_length=config["model_setup"]["max_sequence_length"],
        custom_config_path=CONFIG_PATH,
        generator=generator,
        mad_tolerances=config["mad_validation"]["tolerances"],
        use_onnx_subfunctions=config["pipeline_params"]["use_onnx_subfunctions"],
        parallel_compile=True,
    )

    execution_time = time.time() - start_time

    if config["validation_checks"]["image_generation"]:
        assert result is not None, "Pipeline returned None"
        assert hasattr(result, "images"), "Result missing 'images' attribute"
        assert len(result.images) > 0, "No images generated"

        generated_image = result.images[0]
        expected_size = (config["model_setup"]["width"], config["model_setup"]["height"])
        image_validation = DiffusersTestUtils.validate_image_generation(
            generated_image, expected_size, config["pipeline_params"]["min_image_variance"]
        )

        print("\n IMAGE VALIDATION PASSED")
        print(f"   - Size: {image_validation['size']}")
        print(f"   - Mode: {image_validation['mode']}")
        print(f"   - Variance: {image_validation['variance']:.2f}")
        print(f"   - Mean pixel value: {image_validation['mean_pixel_value']:.2f}")

    if config["validation_checks"]["onnx_export"]:
        print("\n ONNX Export Validation:")
        for module_name in ["transformer", "vae_decoder"]:
            module_obj = getattr(pipeline, module_name, None)
            if module_obj and hasattr(module_obj, "onnx_path") and module_obj.onnx_path:
                DiffusersTestUtils.check_file_exists(str(module_obj.onnx_path), f"{module_name} ONNX")

    if config["validation_checks"]["compilation"]:
        print("\n Compilation Validation:")
        for module_name in ["transformer", "vae_decoder"]:
            module_obj = getattr(pipeline, module_name, None)
            if module_obj and hasattr(module_obj, "qpc_path") and module_obj.qpc_path:
                DiffusersTestUtils.check_file_exists(str(module_obj.qpc_path), f"{module_name} QPC")

    print(f"\nTotal execution time: {execution_time:.4f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "qwen_image"])
