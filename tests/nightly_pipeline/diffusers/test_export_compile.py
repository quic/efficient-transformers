# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import json
import os
import time

import numpy as np
import pytest
import torch
from diffusers import FluxPipeline, WanImageToVideoPipeline, WanPipeline

from QEfficient import QEffFluxPipeline, QEffWanImageToVideoPipeline, QEffWanPipeline

from ..nightly_utils import get_file_or_dir_size, get_onnx_dir_size, measure_peak_ram

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    _config = json.load(f)

flux_models = _config["diffuser_flux_models"]
wan_models = _config["diffuser_wan_models"]
wan_i2v_models = _config["diffuser_wan_i2v_models"]


# Flux
def _export_compile_flux(model_id, diffuser_model_artifacts, get_pipeline_config, enable_first_block_cache=False):
    """Export + compile Flux pipeline and store metrics."""
    dtype_key = "fp32_fbc" if enable_first_block_cache else "fp32"
    cfg = get_pipeline_config["diffuser_flux_configs"]
    model_setup = cfg["model_setup"]
    use_onnx_subfunctions = cfg["pipeline_params"]["use_onnx_subfunctions"]
    compile_config = cfg["compile_config"]

    with measure_peak_ram() as ram:
        load_start = time.time()
        pipeline = QEffFluxPipeline(
            model=FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu"),
            enable_first_block_cache=enable_first_block_cache,
        )
        load_time = time.time() - load_start

        export_start = time.time()
        pipeline.export(use_onnx_subfunctions=use_onnx_subfunctions)
        export_time = time.time() - export_start

        compile_start = time.time()
        pipeline.compile(
            compile_config=compile_config,
            parallel=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
            height=model_setup["height"],
            width=model_setup["width"],
        )
        compile_time = time.time() - compile_start

    modules = {"text_encoder": pipeline.text_encoder, "text_encoder_2": pipeline.text_encoder_2,
               "transformer": pipeline.transformer, "vae_decode": pipeline.vae_decode}
    onnx_paths = {name: str(m.onnx_path) for name, m in modules.items() if m.onnx_path}
    qpc_paths  = {name: str(m.qpc_path)  for name, m in modules.items() if m.qpc_path}
    onnx_dirs  = {name: os.path.dirname(p) for name, p in onnx_paths.items()}

    diffuser_model_artifacts.setdefault(model_id, {}).setdefault(dtype_key, {}).update({
        "load_time": load_time,
        "export_time": export_time,
        "compile_time": compile_time,
        "peak_ram_mb": round(ram["peak_mb"], 2),
        "onnx_and_qpc_dir": onnx_dirs,
        "size": {name: get_file_or_dir_size(d) for name, d in onnx_dirs.items()},
        "onnx_paths": onnx_paths,
        "onnx_size": {name: get_onnx_dir_size(d) for name, d in onnx_dirs.items()},
        "qpc_paths": qpc_paths,
        "qpc_size":  {name: get_file_or_dir_size(p) for name, p in qpc_paths.items()},
    })


@pytest.mark.diffusion_models
@pytest.mark.non_qaic
@pytest.mark.flux
@pytest.mark.parametrize("model_id", flux_models)
def test_export_compile_flux(model_id, diffuser_model_artifacts, get_pipeline_config):
    """FP32 Flux export + compile."""
    _export_compile_flux(model_id, diffuser_model_artifacts, get_pipeline_config)


@pytest.mark.diffusion_models
@pytest.mark.non_qaic
@pytest.mark.flux
@pytest.mark.parametrize("model_id", flux_models)
def test_export_compile_flux_first_block_cache(model_id, diffuser_model_artifacts, get_pipeline_config):
    """FP32 Flux export + compile with first_block_cache."""
    _export_compile_flux(model_id, diffuser_model_artifacts, get_pipeline_config, enable_first_block_cache=True)


#  WAN T2V
def _export_compile_wan(model_id, diffuser_model_artifacts, get_pipeline_config, use_unified=True, enable_first_block_cache=False):
    """Export + compile WAN T2V pipeline and store metrics."""
    dtype_key = "fp32" + ("_unified" if use_unified else "_non_unified") + ("_fbc" if enable_first_block_cache else "")
    cfg_key = "diffuser_wan_configs" if use_unified else "diffuser_wan_non_unified_configs"
    cfg = get_pipeline_config[cfg_key]
    model_setup = cfg["model_setup"]
    use_onnx_subfunctions = cfg["pipeline_params"]["use_onnx_subfunctions"]
    compile_config = cfg["compile_config"]

    with measure_peak_ram() as ram:
        load_start = time.time()
        pipeline = QEffWanPipeline(
            model=WanPipeline.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu"),
            use_unified=use_unified,
            enable_first_block_cache=enable_first_block_cache,
        )
        load_time = time.time() - load_start

        export_start = time.time()
        pipeline.export(use_onnx_subfunctions=use_onnx_subfunctions)
        export_time = time.time() - export_start

        compile_start = time.time()
        pipeline.compile(
            compile_config=compile_config,
            parallel=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
            height=model_setup["height"],
            width=model_setup["width"],
            num_frames=model_setup["num_frames"],
        )
        compile_time = time.time() - compile_start

    if use_unified:
        modules = {"transformer": pipeline.transformer, "vae_decoder": pipeline.vae_decoder}
    else:
        modules = {"transformer_high": pipeline.transformer_high, "transformer_low": pipeline.transformer_low,
                   "vae_decoder": pipeline.vae_decoder}
    onnx_paths = {name: str(m.onnx_path) for name, m in modules.items() if m.onnx_path}
    qpc_paths  = {name: str(m.qpc_path)  for name, m in modules.items() if m.qpc_path}
    onnx_dirs  = {name: os.path.dirname(p) for name, p in onnx_paths.items()}

    diffuser_model_artifacts.setdefault(model_id, {}).setdefault(dtype_key, {}).update({
        "load_time": load_time,
        "export_time": export_time,
        "compile_time": compile_time,
        "peak_ram_mb": round(ram["peak_mb"], 2),
        "onnx_and_qpc_dir": onnx_dirs,
        "size": {name: get_file_or_dir_size(d) for name, d in onnx_dirs.items()},
        "onnx_paths": onnx_paths,
        "onnx_size": {name: get_onnx_dir_size(d) for name, d in onnx_dirs.items()},
        "qpc_paths": qpc_paths,
        "qpc_size":  {name: get_file_or_dir_size(p) for name, p in qpc_paths.items()},
    })


@pytest.mark.diffusion_models
@pytest.mark.non_qaic
@pytest.mark.wan
@pytest.mark.parametrize("model_id", wan_models)
def test_export_compile_wan_non_unified(model_id, diffuser_model_artifacts, get_pipeline_config):
    """FP32 WAN T2V export + compile, non-unified transformers."""
    _export_compile_wan(model_id, diffuser_model_artifacts, get_pipeline_config, use_unified=False)


@pytest.mark.diffusion_models
@pytest.mark.non_qaic
@pytest.mark.wan
@pytest.mark.parametrize("model_id", wan_models)
def test_export_compile_wan_non_unified_first_block_cache(model_id, diffuser_model_artifacts, get_pipeline_config):
    """FP32 WAN T2V export + compile, non-unified with first_block_cache."""
    _export_compile_wan(model_id, diffuser_model_artifacts, get_pipeline_config, use_unified=False, enable_first_block_cache=True)


# WAN I2V
def _export_compile_wan_i2v(model_id, diffuser_model_artifacts, get_pipeline_config):
    """Export + compile WAN I2V pipeline and store metrics."""
    dtype_key = "fp32"
    cfg = get_pipeline_config["diffuser_wan_i2v_configs"]
    pipeline_params = cfg["pipeline_params"]
    model_setup = cfg["model_setup"]
    use_onnx_subfunctions = pipeline_params["use_onnx_subfunctions"]
    compile_config = cfg["compile_config"]

    with measure_peak_ram() as ram:
        load_start = time.time()
        pipeline = QEffWanImageToVideoPipeline(
            WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
        )
        shared_vae = pipeline.model.vae
        pipeline.vae_encoder.model = copy.deepcopy(shared_vae)
        pipeline.vae_decoder.model = copy.deepcopy(shared_vae)
        load_time = time.time() - load_start

        # Dynamic sizing from config max_area
        from diffusers.utils import load_image
        image = load_image(pipeline_params["test_image_url"])
        max_area = model_setup["max_area"]
        aspect_ratio = image.height / image.width
        mod_value = pipeline.model.vae.config.scale_factor_spatial * pipeline.model.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width  = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

        export_start = time.time()
        pipeline.export(use_onnx_subfunctions=use_onnx_subfunctions)
        export_time = time.time() - export_start

        compile_start = time.time()
        pipeline.compile(
            compile_config=compile_config,
            parallel=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
            height=height,
            width=width,
            num_frames=model_setup["num_frames"],
        )
        compile_time = time.time() - compile_start

    modules = {"vae_encoder": pipeline.vae_encoder, "transformer": pipeline.transformer,
               "vae_decoder": pipeline.vae_decoder}
    onnx_paths = {name: str(m.onnx_path) for name, m in modules.items() if m.onnx_path}
    qpc_paths  = {name: str(m.qpc_path)  for name, m in modules.items() if m.qpc_path}
    onnx_dirs  = {name: os.path.dirname(p) for name, p in onnx_paths.items()}

    diffuser_model_artifacts.setdefault(model_id, {}).setdefault(dtype_key, {}).update({
        "load_time": load_time,
        "export_time": export_time,
        "compile_time": compile_time,
        "peak_ram_mb": round(ram["peak_mb"], 2),
        "height": height,
        "width": width,
        "onnx_and_qpc_dir": onnx_dirs,
        "size": {name: get_file_or_dir_size(d) for name, d in onnx_dirs.items()},
        "onnx_paths": onnx_paths,
        "onnx_size": {name: get_onnx_dir_size(d) for name, d in onnx_dirs.items()},
        "qpc_paths": qpc_paths,
        "qpc_size":  {name: get_file_or_dir_size(p) for name, p in qpc_paths.items()},
    })


@pytest.mark.diffusion_models
@pytest.mark.non_qaic
@pytest.mark.wan_i2v
@pytest.mark.parametrize("model_id", wan_i2v_models)
def test_export_compile_wan_i2v(model_id, diffuser_model_artifacts, get_pipeline_config):
    """FP32 WAN I2V export + compile."""
    _export_compile_wan_i2v(model_id, diffuser_model_artifacts, get_pipeline_config)
