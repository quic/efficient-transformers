# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import hashlib

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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


@pytest.mark.on_qaic
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_subfunction_vs_nonsubfunction(config, tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(config.model_type)
    model_0_0 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(config, **model_kwargs), cb=False)
    # model_0_0 = QEFFAutoModelForCausalLM.from_pretrained(config.model_type)

    with_sub_func_onnx = model_0_0.export(tmp_path, use_onnx_subfunctions=True, offload_pt_weights=False)
    hash_0_0 = model_0_0.export_hash

    without_sub_func_onnx = model_0_0.export(tmp_path, use_onnx_subfunctions=False)
    hash_0_1 = model_0_0.export_hash

    # Test that the export hash changes when use_onnx_subfunction is toggled, indicating different parameters are used
    assert hash_0_0 != hash_0_1

    # Test that the exported ONNX files hash are different by comparing their hashes when use_onnx_subfunction is toggled
    with_sub_func_onnx_hash = hashlib.sha256(open(with_sub_func_onnx, "rb").read()).hexdigest()
    without_sub_func_onnx_hash = hashlib.sha256(open(without_sub_func_onnx, "rb").read()).hexdigest()
    assert with_sub_func_onnx_hash != without_sub_func_onnx_hash

    compile_params = {"prefill_seq_len": 8, "ctx_len": 16}
    model_0_0.compile(onnx_path=with_sub_func_onnx, **compile_params)
    generation_00 = model_0_0.generate(prompts=["Help me with this"], tokenizer=tokenizer)

    model_0_0.compile(onnx_path=without_sub_func_onnx, **compile_params)
    generation_01 = model_0_0.generate(prompts=["Help me with this"], tokenizer=tokenizer)
    assert generation_00.generated_texts == generation_01.generated_texts
