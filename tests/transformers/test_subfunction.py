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

torch.manual_seed(42)

configs = [
    ("gpt2", 256, 2, 4, 128, 512, 127, {}),
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


def has_gpt2block_function(onnx_path):
    """Check if ONNX model contains QEffGPT2Block function definition."""
    model = onnx.load(onnx_path, load_external_data=False)
    function_names = [f.name for f in model.functions]
    gpt2block_functions = [name for name in function_names if "QEffGPT2Block" in name]
    return len(gpt2block_functions) > 0, gpt2block_functions


def get_gpt2block_call_count(onnx_path):
    """Get count of QEffGPT2Block function calls in the ONNX model graph."""
    model = onnx.load(onnx_path, load_external_data=False)
    calls = Counter([n.op_type for n in model.graph.node])
    gpt2block_calls = {k: v for k, v in calls.items() if "QEffGPT2Block" in k}
    return gpt2block_calls


@pytest.mark.on_qaic
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_subfunction_vs_nonsubfunction(config, tmp_path):
    # tokenizer = AutoTokenizer.from_pretrained(config.model_type)
    model_0_0 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(config, **model_kwargs), cb=False)

    # Export with subfunctions enabled
    with_sub_func_onnx = model_0_0.export(tmp_path, use_onnx_subfunctions=True, offload_pt_weights=False)

    # Export without subfunctions
    without_sub_func_onnx = model_0_0.export(tmp_path, use_onnx_subfunctions=False)

    # Verify that the model with subfunctions has QEffGPT2Block function definition
    has_gpt2block, gpt2block_names = has_gpt2block_function(with_sub_func_onnx)
    assert has_gpt2block, (
        "Model exported with use_onnx_subfunctions=True should contain QEffGPT2Block function definition"
    )
    print(f"\nGpt2Block functions found: {gpt2block_names}")

    # Verify that the model without subfunctions has no QEffGPT2Block function definition
    has_gpt2block_without, _ = has_gpt2block_function(without_sub_func_onnx)
    assert not has_gpt2block_without, (
        "Model exported with use_onnx_subfunctions=False should not contain QEffGPT2Block function definition"
    )

    # Get QEffGPT2Block call counts
    gpt2block_calls_with_sub = get_gpt2block_call_count(with_sub_func_onnx)
    gpt2block_calls_without_sub = get_gpt2block_call_count(without_sub_func_onnx)

    print(f"\nGpt2Block call counts with subfunctions: {gpt2block_calls_with_sub}")
    print(f"QEffGPT2Block call counts without subfunctions: {gpt2block_calls_without_sub}")

    # Verify that QEffGPT2Block function calls exist in the subfunction model
    assert len(gpt2block_calls_with_sub) > 0, (
        "Expected to find QEffGPT2Block function calls in graph when use_onnx_subfunctions=True"
    )

    # Verify that QEffGPT2Block function calls do NOT exist in the non-subfunction model
    assert len(gpt2block_calls_without_sub) == 0, (
        "Expected NO QEffGPT2Block function calls in graph when use_onnx_subfunctions=False"
    )

    # TODO: Re-enable this check when generation is fully deterministic
    # Compile and test generation to ensure functional equivalence
    # compile_params = {"prefill_seq_len": 8, "ctx_len": 16}

    # model_0_0.compile(onnx_path=with_sub_func_onnx, **compile_params, use_onnx_subfunctions=True)
    # generation_00 = model_0_0.generate(prompts=["Help me with this"], tokenizer=tokenizer)

    # model_0_0.compile(onnx_path=without_sub_func_onnx, **compile_params)
    # generation_01 = model_0_0.generate(prompts=["Help me with this"], tokenizer=tokenizer)

    # Verify that both models produce the same output
    # assert generation_00.generated_texts == generation_01.generated_texts, (
    #    "Models with and without subfunctions should produce identical outputs"
    # )
