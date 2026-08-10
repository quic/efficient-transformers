# -----------------------------------------------------------------------------

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------


import json
import os
import time

import pytest
import torch

from QEfficient import QEFFAutoModel

from ..model_age_utils import filter_models_for_nightly
from ..nightly_utils import get_file_or_dir_size, measure_peak_ram, pre_export_compile_utils

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = filter_models_for_nightly(config["embedding_models"], "embedding_models")

poolings = ["mean", "max", "cls", "avg", None]


def _export_compile_embedding_model(model_name, pooling, get_pipeline_config, embedding_model_artifacts, torch_dtype, seq_len=32, dtype_key="fp32"):
    """Common export and compile logic for embedding models.

    Artifacts are stored under a nested structure:
        artifacts[model_name][dtype_key][pooling_key] = {onnx_path, qpc_path, ...}
    This avoids key collisions across dtype/pooling combinations.
    """
    export_params, compile_params = pre_export_compile_utils(model_name, "embedding_model_configs", get_pipeline_config)

    # Initialize nested model entry
    pooling_key = str(pooling) if pooling is not None else "None"
    embedding_model_artifacts.setdefault(model_name, {}).setdefault(dtype_key, {}).setdefault(pooling_key, {})

    with measure_peak_ram() as ram:
        # Export loading time
        print(f"\nLoading model for export: {model_name} pooling={pooling} dtype_key={dtype_key} seq_len={seq_len}")
        export_load_start = time.time()
        qeff_model = QEFFAutoModel.from_pretrained(model_name, pooling=pooling, torch_dtype=torch_dtype, attn_implementation="eager", trust_remote_code=True)

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
        qpc_path = qeff_model.compile(onnx_path=onnx_path, seq_len=seq_len, **compile_params)
        compile_time = time.time() - compile_start
        print(f"\nCompilation is done for model: {model_name} and qpc path: {qpc_path} in {compile_time:.2f} seconds.")

    embedding_model_artifacts[model_name][dtype_key][pooling_key].update(
        {
            "export_loading_time": export_loading_time,
            "onnx_path": onnx_path,
            "onnx_size": get_file_or_dir_size(onnx_path),
            "export_time": export_time,
            "qpc_path": qpc_path,
            "qpc_size": get_file_or_dir_size(qpc_path),
            "compile_time": compile_time,
            "peak_ram_mb": round(ram["peak_mb"], 2),
        }
    )


# Config 1: FP32, all poolings, single seq_len
@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_export_compile_embedding_model(model_name, pooling, get_pipeline_config, embedding_model_artifacts):
    """FP32 export + compile, all pooling variants."""
    _export_compile_embedding_model(model_name, pooling, get_pipeline_config, embedding_model_artifacts, torch.float32, dtype_key="fp32")


# Config 2: FP32, all poolings, multi seq_len
@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_export_compile_embedding_model_multiseqlen(model_name, pooling, get_pipeline_config, embedding_model_artifacts):
    """FP32 export + compile, multi seq_len."""
    _export_compile_embedding_model(model_name, pooling, get_pipeline_config, embedding_model_artifacts, torch.float32, seq_len=[32, 20], dtype_key="fp32_multiseqlen")


# Config 3: FP16, all poolings, single seq_len
@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_export_compile_embedding_model_fp16(model_name, pooling, get_pipeline_config, embedding_model_artifacts):
    """FP16 export + compile, all pooling variants."""
    _export_compile_embedding_model(model_name, pooling, get_pipeline_config, embedding_model_artifacts, torch.float16, dtype_key="fp16")


# Config 4: FP16, all poolings, multi seq_len
@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("pooling", poolings)
def test_export_compile_embedding_model_fp16_multiseqlen(model_name, pooling, get_pipeline_config, embedding_model_artifacts):
    """FP16 export + compile, multi seq_len."""
    _export_compile_embedding_model(model_name, pooling, get_pipeline_config, embedding_model_artifacts, torch.float16, seq_len=[32, 20], dtype_key="fp16_multiseqlen")
