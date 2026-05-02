# -----------------------------------------------------------------------------

# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------


import json
import os
import time

import pytest
import torch

from QEfficient import QEFFAutoModel as AutoModel

from ..nightly_utils import NIGHTLY_SKIPPED_MODELS


def max_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply max pooling to the last hidden states."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    last_hidden_states[input_mask_expanded == 0] = -1e9
    return torch.max(last_hidden_states, 1)[0]


model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["embedding_models"]


@pytest.mark.parametrize("model_name", test_models)
@pytest.mark.parametrize("pooling", [None])
def test_export_compile_embedding_model(model_name, pooling, get_model_config, embedding_model_artifacts):
    """Test export and compile for embedding models."""
    if model_name in NIGHTLY_SKIPPED_MODELS:
        pytest.skip(f"Skipping {model_name} as it is in nightly skipped models list.")

    config, pipeline_configs = get_model_config
    export_params = pipeline_configs["embedding_model_configs"][0].get("export_params", {})
    compile_params = pipeline_configs["embedding_model_configs"][0].get("compile_params", {})

    # Initialize model entry
    if model_name not in embedding_model_artifacts:
        embedding_model_artifacts[model_name] = {}

    # Export loading time
    print(f"\nLoading model for export: {model_name}")
    export_load_start = time.time()
    if pooling == "max":
        qeff_model = AutoModel.from_pretrained(model_name, pooling=max_pooling, trust_remote_code=True)
    elif pooling == "mean":
        qeff_model = AutoModel.from_pretrained(model_name, pooling="mean", trust_remote_code=True)
    else:
        qeff_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    export_loading_time = time.time() - export_load_start

    # Export time
    print(f"\nExporting for model: {model_name}")
    export_start = time.time()
    onnx_path = qeff_model.export(**export_params)
    export_time = time.time() - export_start
    print(f"\nExport is done for model: {model_name} and onnx_path: {onnx_path}")

    # Compile
    print(f"\nCompiling for model: {model_name}")
    compile_start = time.time()
    qpc_path = qeff_model.compile(onnx_path=onnx_path, **compile_params)
    compile_time = time.time() - compile_start
    print(f"\nCompilation is done for model: {model_name} and qpc path: {qpc_path}")

    embedding_model_artifacts[model_name].update(
        {
            "export_loading_time": export_loading_time,
            "onnx_path": onnx_path,
            "export_time": export_time,
            "qpc_path": qpc_path,
            "compile_time": compile_time,
        }
    )
