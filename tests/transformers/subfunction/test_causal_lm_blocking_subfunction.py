# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from typing import Optional

import onnx
import pytest
from transformers import AutoConfig

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.test_utils import ModelConfig, load_hf_causal_lm_model

test_models_blockedKV_dict = {
    "unsloth/gemma-2b": "optimum-intel-internal-testing/tiny-random-gemma2",
    "unsloth/gemma-2-2b": "optimum-intel-internal-testing/tiny-random-gemma2",
    "ibm-granite/granite-3.1-1b-a400m-base": "optimum-intel-internal-testing/tiny-random-GraniteMoeForCausalLM",
    "meta-llama/Llama-3.2-1B": "optimum-intel-internal-testing/tiny-random-LlamaForCausalLM",
    "wtang06/mpt-125m-c4": "optimum-intel-internal-testing/tiny-random-MptForCausalLM",
    "bigcode/starcoder2-3b": "optimum-intel-internal-testing/tiny-random-Starcoder2ForCausalLM",
}

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    test_models_blockedKV = list(test_models_blockedKV_dict.values())
else:
    test_models_blockedKV = list(test_models_blockedKV_dict.keys())


def check_blockedKV_onnx_function_count_with_subfunction(
    model_name: str, n_layer: int = -1, config: Optional[AutoConfig] = None
):
    """
    Export twice with `use_onnx_subfunctions=True`:
      1) without blocking
      2) with KV blocking (NUM_KV_BLOCKS=2)

    Verify that the number of ONNX functions is identical.
    """
    # Export with subfunctions, NO blocking
    model_no_block = load_hf_causal_lm_model(model_name, num_hidden_layers=n_layer, config=config)
    qeff_no_block = QEFFAutoModelForCausalLM(model_no_block, pretrained_model_name_or_path=model_name, qaic_config=None)
    qeff_no_block.export(use_onnx_subfunctions=True, offload_pt_weights=False)
    onnx_no_block = onnx.load(qeff_no_block.onnx_path, load_external_data=False)
    num_functions_no_block = len(onnx_no_block.functions)
    # Export with subfunctions, WITH KV blocking
    NUM_KV_BLOCKS = 2
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)

    model_kv_block = load_hf_causal_lm_model(model_name, num_hidden_layers=n_layer, config=config)
    qeff_kv_block = QEFFAutoModelForCausalLM(
        model_kv_block, pretrained_model_name_or_path=model_name, qaic_config=qaic_config
    )
    qeff_kv_block.export(use_onnx_subfunctions=True, offload_pt_weights=False)
    onnx_kv_block = onnx.load(qeff_kv_block.onnx_path, load_external_data=False)
    num_functions_kv_block = len(onnx_kv_block.functions)

    assert num_functions_no_block == num_functions_kv_block


@pytest.mark.feature
@pytest.mark.parametrize("model_name", test_models_blockedKV)
def test_blockedKV_onnx_function_count_with_subfunction(model_name):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")

    check_blockedKV_onnx_function_count_with_subfunction(model_name)
