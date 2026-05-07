# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os

import pytest
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

from ..nightly_utils import get_onnx_and_qpc_size, pre_generate_utils

model_config_path = os.path.join(os.path.dirname(__file__), "../configs/validated_models.json")
with open(model_config_path, "r") as f:
    config = json.load(f)

test_models = config["causal_lm_models"]


@pytest.mark.parametrize("model_name", test_models)
def test_generate_causal_lm(model_name, causal_model_artifacts, get_pipeline_config):

    compile_params, generate_params = pre_generate_utils(
        model_name, "causal_pipeline_configs", get_pipeline_config, causal_model_artifacts
    )

    onnx_path = causal_model_artifacts[model_name].get("onnx_path")

    print(f"\nLoading model for generation: {model_name}")
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    _ = qeff_model.compile(onnx_path=onnx_path, **compile_params)

    print(f"\nGenerating for model: {model_name}")
    exec_info = qeff_model.generate(tokenizer=tokenizer, **generate_params)

    print(f"\nGeneration complete for model: {model_name}")

    onnx_and_qpc_dir = os.path.dirname(onnx_path)
    size = get_onnx_and_qpc_size(onnx_and_qpc_dir)
    # Store all metrics and execution info
    causal_model_artifacts[model_name].update(
        {
            "batch_size": exec_info.batch_size,
            "generated_texts": exec_info.generated_texts,
            "generated_ids": exec_info.generated_ids[0][0][:20],  # Converted to list by conftest serializer
            "onnx_and_qpc_dir": onnx_and_qpc_dir,
            "size": size,
            "perf_metrics": {
                "prefill_time": exec_info.perf_metrics.prefill_time,
                "decode_perf": exec_info.perf_metrics.decode_perf,
                "total_perf": exec_info.perf_metrics.total_perf,
                "total_time": exec_info.perf_metrics.total_time,
            },
        }
    )
