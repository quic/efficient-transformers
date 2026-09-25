# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Weight-free disaggregated prefill/decode compile tests."""

from __future__ import annotations

import pytest
from transformers import AutoConfig

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

from ._helpers import skip_on_model_fetch_error


DISAGG_MODEL_PARAMS = [
    pytest.param("glm4_moe", "tiny-random/glm-4-moe", id="glm4-moe"),
    pytest.param("qwen3_moe", "tiny-random/qwen3-moe", id="qwen3-moe"),
    pytest.param("gpt_oss", "tiny-random/gpt-oss-mxfp4", id="gpt-oss"),
]

CTX_LEN = 128
PREFILL_SEQ_LEN = 32
MOE_PREFILL_PACKED_CHUNK_SIZE = 16


def _load_weight_free_model(model_id: str, continuous_batching: bool = False):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    config.num_hidden_layers = 2
    return QEFFAutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        weight_free=True,
        continuous_batching=continuous_batching,
    )


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", DISAGG_MODEL_PARAMS)
def test_weight_free_disaggregated_prefill_and_decode(model_type, model_id, tmp_export_dir):
    """Compile weight-free decode and chunked-prefill QPCs for disaggregated serving."""
    try:
        qeff_model = _load_weight_free_model(model_id)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    moe_config = {"moe_config": {"expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE}}
    common = {
        "ctx_len": CTX_LEN,
        "num_cores": 4,
        "num_devices": 1,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "use_onnx_subfunctions": True,
        "offload_pt_weights": False,
        "retain_full_kv": True,
    }

    decode_qpc = qeff_model.compile(
        compile_dir=str(tmp_export_dir / f"{model_type}_decode"),
        prefill_seq_len=1,
        **common,
    )
    prefill_qpc = qeff_model.compile(
        compile_dir=str(tmp_export_dir / f"{model_type}_prefill"),
        prefill_seq_len=PREFILL_SEQ_LEN,
        prefill_only=True,
        enable_chunking=True,
        qaic_config=moe_config,
        **common,
    )

    assert decode_qpc
    assert prefill_qpc


@pytest.mark.weight_free
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_type,model_id", DISAGG_MODEL_PARAMS)
def test_weight_free_disaggregated_continuous_batching(model_type, model_id, tmp_export_dir):
    """Compile CB decode and chunked-prefill QPCs for weight-free disaggregated serving."""
    try:
        qeff_model = _load_weight_free_model(model_id, continuous_batching=True)
    except Exception as exc:
        skip_on_model_fetch_error(exc, model_id)

    full_batch_size = 2
    common = {
        "ctx_len": CTX_LEN,
        "full_batch_size": full_batch_size,
        "num_cores": 4,
        "num_devices": 1,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "split_retained_state_io": True,
        "retain_full_kv": True,
        "use_onnx_subfunctions": True,
        "offload_pt_weights": False,
    }

    decode_qpc = qeff_model.compile(
        compile_dir=str(tmp_export_dir / f"{model_type}_cb_decode"),
        prefill_seq_len=1,
        **common,
    )
    prefill_qpc = qeff_model.compile(
        compile_dir=str(tmp_export_dir / f"{model_type}_cb_prefill"),
        prefill_seq_len=PREFILL_SEQ_LEN,
        prefill_only=True,
        enable_chunking=True,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE}},
        **common,
    )

    assert decode_qpc
    assert prefill_qpc
