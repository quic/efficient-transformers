# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
from collections import Counter

import onnx
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.device_utils import get_available_device_id

torch.manual_seed(42)

configs = [
    ("gpt2", 256, 2, 4, 128, 512, 127, {}),
    # ("codegen", 256, 2, 4, 128, 512, 127, {"rotary_dim": 16}),
    # ("falcon", 256, 2, 4, 128, 512, 127, {}),
    # ("gptj", 256, 2, 4, 128, 512, 127, {"rotary_dim": 16}),
    # ("llama", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    # ("mistral", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    # ("mixtral", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    # ("mpt", 256, 2, 4, 128, 512, 127, {}),
    # ("phi", 256, 2, 4, 128, 512, 127, {}),
    # ("phi3", 256, 2, 4, 128, 512, 127, {"pad_token_id": 0}),
    # ("qwen2", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    # ("qwen3", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    # ("starcoder2", 256, 2, 4, 128, 512, 127, {}),
    # ("granite", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    # ("olmo2", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    # ("gpt_oss", 256, 3, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    # ("qwen3_moe", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    # ("granitemoe", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
]

configs = [
    AutoConfig.for_model(
        model_name,
        max_position_embeddings=max_position_embeddings,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        **additional_params,
    )
    for (
        model_name,
        max_position_embeddings,
        num_hidden_layers,
        num_attention_heads,
        hidden_size,
        intermediate_size,
        vocab_size,
        additional_params,
    ) in configs
]

model_kwargs = {"attn_implementation": "eager"}
config_ids = [x.model_type for x in configs]


def get_function(onnx_path):
    """Check if ONNX model contains QEffGPT2Block function definition."""
    model = onnx.load(onnx_path, load_external_data=False)
    function_names = [f.name for f in model.functions]
    return function_names

@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_subfunction_vs_nonsubfunction(config, tmp_path):
    # tokenizer = AutoTokenizer.from_pretrained(config.model_type)
    model_0_0 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(config, **model_kwargs), cb=False)
    tmp_path = "/home/abhishek/.cache/qeff_models/temp_onnx"
    # Export with subfunctions enabled
    with_sub_func_onnx = model_0_0.export(tmp_path, use_onnx_subfunctions=True, offload_pt_weights=False)

    print(f"{config.model_type} is going on...")
    # Verify that the model with subfunctions has QEffGPT2Block function definition
    functions_names = get_function(with_sub_func_onnx)
    if len(functions_names) != 12:
        raise AssertionError(
            f"function definition, but found {len(functions_names)} functions: {functions_names}"
        )
    
    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")
    compile_params = {"prefill_seq_len": 8, "ctx_len": 16}
    model_0_0.compile(onnx_path=with_sub_func_onnx, **compile_params, use_onnx_subfunctions=True)