# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import json
import os

import numpy as np
import pytest
import torch
from diffusers import FluxPipeline, WanImageToVideoPipeline, WanPipeline
from diffusers.utils import load_image

from QEfficient import QEffFluxPipeline, QEffWanImageToVideoPipeline, QEffWanPipeline

from ..nightly_utils import (
    compare_with_golden,
    make_golden_key,
    measure_peak_ram,
    run_or_load_golden,
)

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    _config = json.load(f)

flux_models = _config["diffuser_flux_models"]
wan_models = _config["diffuser_wan_models"]
wan_i2v_models = _config["diffuser_wan_i2v_models"]

TEST_SEED = 42


def _skip_if_no_artifacts(diffuser_model_artifacts, model_id, dtype_key):
    """Skip generate if export+compile artifacts are missing."""
    entry = diffuser_model_artifacts.get(model_id, {}).get(dtype_key, {})
    if not entry.get("qpc_paths"):
        pytest.skip(f"No QPC artifacts for {model_id} [{dtype_key}]. Run export+compile first.")
    return entry


def _image_to_float32(img):
    """Normalise PIL image or numpy array to float32 in [0,1]."""
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def _frames_to_float32(frames):
    """Stack a list of frames (PIL or numpy) to (F, H, W, C) float32."""
    return np.stack([_image_to_float32(f) for f in frames])


# Flux image generation and golden MAD validation
def _generate_flux(model_id, diffuser_model_artifacts, get_pipeline_config, enable_first_block_cache=False, pipeline_call_overrides=None):
    """Generate Flux image and compare against golden PyTorch output."""
    dtype_key = "fp32_fbc" if enable_first_block_cache else "fp32"
    entry = _skip_if_no_artifacts(diffuser_model_artifacts, model_id, dtype_key)
    cfg = get_pipeline_config["diffuser_flux_configs"]
    pipeline_params = cfg["pipeline_params"]
    model_setup = cfg["model_setup"]
    compile_config = cfg["compile_config"]

    pytorch_pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
    pipeline = QEffFluxPipeline(model=copy.deepcopy(pytorch_pipeline), enable_first_block_cache=enable_first_block_cache)

    pipeline_call_overrides = pipeline_call_overrides or {}
    generator = torch.Generator(device="cpu").manual_seed(TEST_SEED)

    with measure_peak_ram() as ram:
        result = pipeline(
            prompt=pipeline_params["test_prompt"],
            height=model_setup["height"],
            width=model_setup["width"],
            guidance_scale=pipeline_params["guidance_scale"],
            num_inference_steps=pipeline_params["num_inference_steps"],
            max_sequence_length=pipeline_params["max_sequence_length"],
            generator=generator,
            custom_config_path=compile_config,
            use_onnx_subfunctions=pipeline_params["use_onnx_subfunctions"],
            **pipeline_call_overrides,
        )

    assert result is not None and hasattr(result, "images") and len(result.images) > 0
    qaic_arr = _image_to_float32(result.images[0])

    golden_key = make_golden_key(
        dtype="fp32",
        config_params={
            "height": model_setup["height"],
            "width": model_setup["width"],
            "num_inference_steps": pipeline_params["num_inference_steps"],
            "guidance_scale": pipeline_params["guidance_scale"],
        },
        extra_tags={"fbc": int(enable_first_block_cache)},
    )

    def _run_pytorch():
        gen = torch.Generator(device="cpu").manual_seed(TEST_SEED)
        ref = pytorch_pipeline(
            prompt=pipeline_params["test_prompt"],
            height=model_setup["height"],
            width=model_setup["width"],
            guidance_scale=pipeline_params["guidance_scale"],
            num_inference_steps=pipeline_params["num_inference_steps"],
            max_sequence_length=pipeline_params["max_sequence_length"],
            generator=gen,
            output_type="np",
        )
        return {"pytorch_images": ref.images[0].astype(float).tolist()}

    golden = run_or_load_golden(
        category="diffuser_models",
        model_name=model_id,
        golden_key=golden_key,
        run_pytorch_fn=_run_pytorch,
        config_fp=compile_config,
    )

    tolerance = cfg["mad_validation"]["tolerances"].get("end_to_end", 0.35)
    comparison = compare_with_golden(
        qpc_output={"pytorch_images": qaic_arr.tolist()},
        golden={"pytorch_images": golden.get("pytorch_images", [])},
        tolerance=tolerance,
    )
    print(f"\n[GOLDEN] passed={comparison['passed']} peak_ram_mb={ram['peak_mb']:.1f} details={comparison['per_key']}")

    diffuser_model_artifacts[model_id][dtype_key].update({
        "generate_peak_ram_mb": round(ram["peak_mb"], 2),
        "golden_comparison": comparison,
        "image_shape": list(qaic_arr.shape),
        "image_mean": round(float(qaic_arr.mean()), 6),
        "image_max": round(float(qaic_arr.max()), 6),
    })

    assert comparison["passed"], f"QPC image differs from golden PyTorch: {comparison['per_key']}"


@pytest.mark.diffusion_models
@pytest.mark.on_qaic
@pytest.mark.flux
@pytest.mark.parametrize("model_id", flux_models)
def test_generate_flux(model_id, diffuser_model_artifacts, get_pipeline_config):
    """FP32 Flux image generation with golden MAD validation."""
    _generate_flux(model_id, diffuser_model_artifacts, get_pipeline_config)


@pytest.mark.diffusion_models
@pytest.mark.on_qaic
@pytest.mark.flux
@pytest.mark.parametrize("model_id", flux_models)
def test_generate_flux_first_block_cache(model_id, diffuser_model_artifacts, get_pipeline_config):
    """FP32 Flux image generation with first_block_cache and golden MAD validation."""
    _generate_flux(model_id, diffuser_model_artifacts, get_pipeline_config, enable_first_block_cache=True,
                   pipeline_call_overrides={"cache_threshold": 0.0})


# WAN T2V video generation and golden MAD validation
def _generate_wan(model_id, diffuser_model_artifacts, get_pipeline_config, use_unified=False, enable_first_block_cache=False,
                  pipeline_call_overrides=None):
    """Generate WAN T2V video and compare against golden PyTorch output."""
    dtype_key = "fp32" + ("_unified" if use_unified else "_non_unified") + ("_fbc" if enable_first_block_cache else "")
    cfg_key = "diffuser_wan_configs" if use_unified else "diffuser_wan_non_unified_configs"
    entry = _skip_if_no_artifacts(diffuser_model_artifacts, model_id, dtype_key)
    cfg = get_pipeline_config[cfg_key]
    pipeline_params = cfg["pipeline_params"]
    model_setup = cfg["model_setup"]
    compile_config = cfg["compile_config"]

    pytorch_pipeline = WanPipeline.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
    pipeline = QEffWanPipeline(
        model=copy.deepcopy(pytorch_pipeline),
        use_unified=use_unified,
        enable_first_block_cache=enable_first_block_cache,
    )

    pipeline_call_overrides = pipeline_call_overrides or {}
    generator = torch.Generator(device="cpu").manual_seed(TEST_SEED)

    with measure_peak_ram() as ram:
        result = pipeline(
            prompt=pipeline_params["test_prompt"],
            height=model_setup["height"],
            width=model_setup["width"],
            num_frames=model_setup["num_frames"],
            guidance_scale=pipeline_params["guidance_scale"],
            guidance_scale_2=pipeline_params["guidance_scale_2"],
            num_inference_steps=pipeline_params["num_inference_steps"],
            max_sequence_length=pipeline_params["max_sequence_length"],
            generator=generator,
            custom_config_path=compile_config,
            use_onnx_subfunctions=pipeline_params["use_onnx_subfunctions"],
            **pipeline_call_overrides,
        )

    assert result is not None and hasattr(result, "images") and len(result.images) > 0
    qaic_frames = _frames_to_float32(result.images[0])

    golden_key = make_golden_key(
        dtype="fp32",
        config_params={
            "height": model_setup["height"],
            "width": model_setup["width"],
            "num_frames": model_setup["num_frames"],
            "num_inference_steps": pipeline_params["num_inference_steps"],
        },
        extra_tags={"unified": int(use_unified), "fbc": int(enable_first_block_cache)},
    )

    def _run_pytorch():
        gen = torch.Generator(device="cpu").manual_seed(TEST_SEED)
        ref = pytorch_pipeline(
            prompt=pipeline_params["test_prompt"],
            height=model_setup["height"],
            width=model_setup["width"],
            num_frames=model_setup["num_frames"],
            guidance_scale=pipeline_params["guidance_scale"],
            guidance_scale_2=pipeline_params["guidance_scale_2"],
            num_inference_steps=pipeline_params["num_inference_steps"],
            max_sequence_length=pipeline_params["max_sequence_length"],
            generator=gen,
            output_type="np",
        )
        frames = ref.frames[0] if hasattr(ref, "frames") else ref.images[0]
        return {"pytorch_frames": _frames_to_float32(frames).tolist()}

    golden = run_or_load_golden(
        category="diffuser_models",
        model_name=model_id,
        golden_key=golden_key,
        run_pytorch_fn=_run_pytorch,
        config_fp=compile_config,
    )

    tolerance = cfg["mad_validation"]["tolerances"].get("end_to_end", 0.35)
    comparison = compare_with_golden(
        qpc_output={"pytorch_frames": qaic_frames.tolist()},
        golden={"pytorch_frames": golden.get("pytorch_frames", [])},
        tolerance=tolerance,
    )
    print(f"\n[GOLDEN] passed={comparison['passed']} peak_ram_mb={ram['peak_mb']:.1f} details={comparison['per_key']}")

    diffuser_model_artifacts[model_id][dtype_key].update({
        "generate_peak_ram_mb": round(ram["peak_mb"], 2),
        "golden_comparison": comparison,
        "image_shape": list(qaic_frames.shape),
        "image_mean": round(float(qaic_frames.mean()), 6),
        "image_max": round(float(qaic_frames.max()), 6),
    })

    assert comparison["passed"], f"QPC video differs from golden PyTorch: {comparison['per_key']}"


@pytest.mark.diffusion_models
@pytest.mark.on_qaic
@pytest.mark.wan
@pytest.mark.parametrize("model_id", wan_models)
def test_generate_wan_non_unified(model_id, diffuser_model_artifacts, get_pipeline_config):
    """FP32 WAN T2V non-unified generate with golden MAD validation."""
    _generate_wan(model_id, diffuser_model_artifacts, get_pipeline_config, use_unified=False)


@pytest.mark.diffusion_models
@pytest.mark.on_qaic
@pytest.mark.wan
@pytest.mark.parametrize("model_id", wan_models)
def test_generate_wan_non_unified_first_block_cache(model_id, diffuser_model_artifacts, get_pipeline_config):
    """FP32 WAN T2V non-unified with first_block_cache generate and golden MAD validation."""
    _generate_wan(model_id, diffuser_model_artifacts, get_pipeline_config, use_unified=False, enable_first_block_cache=True,
                  pipeline_call_overrides={"cache_threshold_high": 0.0, "cache_threshold_low": 0.0})


# WAN I2V video generation and golden MAD validation
def _generate_wan_i2v(model_id, diffuser_model_artifacts, get_pipeline_config):
    """Generate WAN I2V video and compare against golden PyTorch output."""
    dtype_key = "fp32"
    entry = _skip_if_no_artifacts(diffuser_model_artifacts, model_id, dtype_key)
    cfg = get_pipeline_config["diffuser_wan_i2v_configs"]
    pipeline_params = cfg["pipeline_params"]
    model_setup = cfg["model_setup"]
    compile_config = cfg["compile_config"]

    pytorch_pipeline = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
    pipeline = QEffWanImageToVideoPipeline(copy.deepcopy(pytorch_pipeline))
    shared_vae = pipeline.model.vae
    pipeline.vae_encoder.model = copy.deepcopy(shared_vae)
    pipeline.vae_decoder.model = copy.deepcopy(shared_vae)

    # Reuse height/width saved by export+compile
    height = entry.get("height")
    width  = entry.get("width")
    if height is None or width is None:
        # Recalculate if export+compile was not run first
        image = load_image(pipeline_params["test_image_url"])
        max_area = model_setup["max_area"]
        aspect_ratio = image.height / image.width
        mod_value = pipeline.model.vae.config.scale_factor_spatial * pipeline.model.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width  = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image  = image.resize((width, height))
    else:
        image = load_image(pipeline_params["test_image_url"]).resize((width, height))

    generator = torch.Generator(device="cpu").manual_seed(TEST_SEED)

    with measure_peak_ram() as ram:
        result = pipeline(
            image=image,
            prompt=pipeline_params["test_prompt"],
            height=height,
            width=width,
            num_frames=model_setup["num_frames"],
            guidance_scale=pipeline_params["guidance_scale"],
            guidance_scale_2=pipeline_params["guidance_scale_2"],
            num_inference_steps=pipeline_params["num_inference_steps"],
            max_sequence_length=pipeline_params["max_sequence_length"],
            generator=generator,
            custom_config_path=compile_config,
            use_onnx_subfunctions=pipeline_params["use_onnx_subfunctions"],
        )

    assert result is not None and hasattr(result, "images") and len(result.images) > 0
    qaic_frames = _frames_to_float32(result.images[0])

    golden_key = make_golden_key(
        dtype="fp32",
        config_params={
            "height": height,
            "width": width,
            "num_frames": model_setup["num_frames"],
            "num_inference_steps": pipeline_params["num_inference_steps"],
        },
        extra_tags={},
    )

    def _run_pytorch():
        gen = torch.Generator(device="cpu").manual_seed(TEST_SEED)
        ref = pytorch_pipeline(
            image=image,
            prompt=pipeline_params["test_prompt"],
            height=height,
            width=width,
            num_frames=model_setup["num_frames"],
            guidance_scale=pipeline_params["guidance_scale"],
            guidance_scale_2=pipeline_params["guidance_scale_2"],
            num_inference_steps=pipeline_params["num_inference_steps"],
            max_sequence_length=pipeline_params["max_sequence_length"],
            generator=gen,
            output_type="np",
        )
        frames = ref.frames[0] if hasattr(ref, "frames") else ref.images[0]
        return {"pytorch_frames": _frames_to_float32(frames).tolist()}

    golden = run_or_load_golden(
        category="diffuser_models",
        model_name=model_id,
        golden_key=golden_key,
        run_pytorch_fn=_run_pytorch,
        config_fp=compile_config,
    )

    tolerance = cfg["mad_validation"]["tolerances"].get("end_to_end", 0.35)
    comparison = compare_with_golden(
        qpc_output={"pytorch_frames": qaic_frames.tolist()},
        golden={"pytorch_frames": golden.get("pytorch_frames", [])},
        tolerance=tolerance,
    )
    print(f"\n[GOLDEN] passed={comparison['passed']} peak_ram_mb={ram['peak_mb']:.1f} details={comparison['per_key']}")

    diffuser_model_artifacts[model_id][dtype_key].update({
        "generate_peak_ram_mb": round(ram["peak_mb"], 2),
        "golden_comparison": comparison,
        "image_shape": list(qaic_frames.shape),
        "image_mean": round(float(qaic_frames.mean()), 6),
        "image_max": round(float(qaic_frames.max()), 6),
    })

    assert comparison["passed"], f"QPC video differs from golden PyTorch: {comparison['per_key']}"


@pytest.mark.diffusion_models
@pytest.mark.on_qaic
@pytest.mark.wan_i2v
@pytest.mark.parametrize("model_id", wan_i2v_models)
def test_generate_wan_i2v(model_id, diffuser_model_artifacts, get_pipeline_config):
    """FP32 WAN I2V generate with golden MAD validation."""
    _generate_wan_i2v(model_id, diffuser_model_artifacts, get_pipeline_config)
