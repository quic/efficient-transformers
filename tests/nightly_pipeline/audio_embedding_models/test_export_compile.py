# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import json
import os
import time

import pytest
import torch

from QEfficient import QEFFAutoModelForCTC

from ..model_age_utils import filter_models_for_nightly
from ..nightly_utils import get_file_or_dir_size, get_onnx_dir_size, measure_peak_ram, pre_export_compile_utils

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = filter_models_for_nightly(config["audio_embedding_models"], "audio_embedding_models")


def _export_compile_audio_embedding_model(
    model_name, get_pipeline_config, audio_embedding_model_artifacts, torch_dtype, dtype_key="fp32"
):
    """Common export and compile logic for CTC audio embedding models."""
    export_params, compile_params = pre_export_compile_utils(
        model_name, "audio_embedding_model_configs", get_pipeline_config
    )

    with measure_peak_ram() as ram:
        # Export loading time
        print(f"\nLoading model for export: {model_name} (dtype={torch_dtype})")
        export_load_start = time.time()
        qeff_model = QEFFAutoModelForCTC.from_pretrained(model_name, torch_dtype=torch_dtype)
        export_loading_time = time.time() - export_load_start
        print(f"\nModel loading is done for model: {model_name} in {export_loading_time:.2f} seconds.")

        # Export time
        print(f"\nExporting for model: {model_name}")
        export_start = time.time()
        onnx_path = qeff_model.export(**export_params)
        export_time = time.time() - export_start
        print(f"\nExport is done for model: {model_name} and onnx_path: {onnx_path} in {export_time:.2f} seconds.")

        # Compile
        print(f"\nCompiling for model: {model_name}")
        compile_start = time.time()
        qpc_path = qeff_model.compile(onnx_path=onnx_path, **compile_params)
        compile_time = time.time() - compile_start
        print(f"\nCompilation is done for model: {model_name} and qpc path: {qpc_path} in {compile_time:.2f} seconds.")

    onnx_and_qpc_dir = os.path.dirname(onnx_path)
    audio_embedding_model_artifacts.setdefault(model_name, {}).setdefault(dtype_key, {}).update(
        {
            "load_time": export_loading_time,
            "export_time": export_time,
            "compile_time": compile_time,
            "peak_ram_mb": round(ram["peak_mb"], 2),
            "onnx_and_qpc_dir": onnx_and_qpc_dir,
            "size": get_file_or_dir_size(onnx_and_qpc_dir),
            "onnx_path": onnx_path,
            "onnx_size": get_onnx_dir_size(onnx_and_qpc_dir),
            "qpc_path": qpc_path,
            "qpc_size": get_file_or_dir_size(qpc_path),
        }
    )


# Config 1: FP32
@pytest.mark.parametrize("model_name", test_models)
def test_export_compile_audio_embedding_model(model_name, get_pipeline_config, audio_embedding_model_artifacts):
    """FP32 export + compile."""
    _export_compile_audio_embedding_model(
        model_name, get_pipeline_config, audio_embedding_model_artifacts, torch.float32
    )


# Config 2: FP16
@pytest.mark.parametrize("model_name", test_models)
def test_export_compile_audio_embedding_model_fp16(model_name, get_pipeline_config, audio_embedding_model_artifacts):
    """FP16 export + compile."""
    _export_compile_audio_embedding_model(
        model_name, get_pipeline_config, audio_embedding_model_artifacts, torch.float16, dtype_key="fp16"
    )
