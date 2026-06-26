# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import pytest
import torch
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText

MODEL_IDS = {
    "qwen3_vl_moe": "tiny-random/qwen3-vl-moe",
    "qwen3_5_moe": "tiny-random/qwen3.5-moe",
    "qwen3_moe": "tiny-random/qwen3-moe",
}

COMMON_LAYERWISE_CB_COMPILE_KWARGS = {
    "num_cores": 16,
    "num_devices": 1,
    "full_batch_size": 4,
    "aic_enable_depth_first": True,
    "split_retained_state_io": True,
    "use_onnx_subfunctions": True,
    "prefill_only": True,
    "mos": 1,
    "layerwise": True,
    "layerwise_window_size": 1,
}

VLM_LAYERWISE_CB_COMPILE_KWARGS = {
    **COMMON_LAYERWISE_CB_COMPILE_KWARGS,
    "prefill_seq_len": 32,
    "ctx_len": 4096,
    "batch_size": 1,
    "height": 354,
    "width": 536,
    "skip_vision": True,
}

LAYERWISE_CB_COMPILE_OVERRIDES = {
    "qwen3_vl_moe": {
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
    },
    "qwen3_5_moe": {
        "mxfp6_matmul": False,
    },
    "qwen3_moe": {
        "num_cores": 8,
        "prefill_seq_len": 4,
        "ctx_len": 128,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "offload_pt_weights": False,
        "num_speculative_tokens": None,
        "enable_chunking": True,
    },
}


def _build_compile_kwargs(model_type):
    if model_type in {"qwen3_vl_moe", "qwen3_5_moe"}:
        return {**VLM_LAYERWISE_CB_COMPILE_KWARGS, **LAYERWISE_CB_COMPILE_OVERRIDES[model_type]}
    return {**COMMON_LAYERWISE_CB_COMPILE_KWARGS, **LAYERWISE_CB_COMPILE_OVERRIDES[model_type]}


def _load_qeff_model(model_type, model_id, config):
    if model_type == "qwen3_moe":
        return QEFFAutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            layerwise=True,
            continuous_batching=True,
        )

    config.torch_dtype = "float32"
    return QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        torch_dtype=torch.float32,
        layerwise=True,
        continuous_batching=True,
    )


@pytest.mark.on_qaic
@pytest.mark.regular
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    [
        pytest.param("qwen3_vl_moe", MODEL_IDS["qwen3_vl_moe"]),
        pytest.param("qwen3_5_moe", MODEL_IDS["qwen3_5_moe"]),
        pytest.param("qwen3_moe", MODEL_IDS["qwen3_moe"]),
    ],
)
def test_qwen_layerwise_continuous_batching_compile_only(manual_cleanup, model_type, model_id):
    config = AutoConfig.from_pretrained(model_id)
    compile_kwargs = _build_compile_kwargs(model_type)
    qeff_model = _load_qeff_model(model_type, model_id, config)
    qpc_path = qeff_model.compile(**compile_kwargs)
    assert qpc_path is not None
    if qeff_model.onnx_path is not None:
        manual_cleanup(qeff_model.onnx_path)
