# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

torch.manual_seed(42)

configs = [
    # ("gpt2", 256, 4, 4, 128, 512, 127, {}),
    # ("llama", 256, 4, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    # ("gpt_oss", 256, 3, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    #  ("mistral", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    # ("qwen3", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("phi3", 256, 8, 4, 128, 512, 127, {"pad_token_id": 0}),
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


@pytest.mark.on_qaic
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_subfunction_dynamo_gpt2(config, tmp_path):
    model = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(config, **model_kwargs), cb=False)

    _ = model.export(
        "test_models",
        use_onnx_subfunctions=True,
        use_dynamo=True,
        offload_pt_weights=False,
    )

    # TODO: Add correct verification checks
    # onnx_model = onnx.load(onnx_path, load_external_data=False)
    # assert any("QEffGPT2Block" in func.name for func in onnx_model.functions), (
    #     "Expected QEffGPT2Block functions in ONNX model when use_dynamo=True and use_onnx_subfunctions=True"
    # )
    # assert any("QEffGPT2Block" in node.op_type for node in onnx_model.graph.node), (
    #     "Expected QEffGPT2Block function calls in graph when use_dynamo=True and use_onnx_subfunctions=True"
    # )
