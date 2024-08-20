# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import numpy as np
import onnxruntime as ort
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.customop.matmulnbits import QuantLinearORT
from QEfficient.transformers.quantizers import replace_transformers_quantizers
from QEfficient.transformers.quantizers.quant_transforms import AwqToOnnxTransform
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner

quant_models_test = [
    "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",
]


@pytest.mark.parametrize("model_card_name", quant_models_test)
def test_quant_models_load_and_run(model_card_name: str):
    replace_transformers_quantizers()
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_card_name, attn_implementation="eager", low_cpu_mem_usage=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_card_name)
    api_runner = ApiRunner(
        1,
        tokenizer,
        model_hf.config,
        Constants.INPUT_STR,
        Constants.PROMPT_LEN,
        Constants.CTX_LEN,
    )

    pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, model_card_name)

    pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

    assert (pytorch_hf_tokens == pytorch_kv_tokens).all()

    print("done")


# FIXME: add test for module_mutator also add test for awqtoonnxtransofrm
#        remove the below and above test, add the model to test_causal_lm
@pytest.mark.parametrize("model_card_name", quant_models_test)
def test_awq_unpack(model_card_name: str):
    replace_transformers_quantizers()
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_card_name, attn_implementation="eager", low_cpu_mem_usage=False
    )

    wqlinear = model_hf.model.layers[0].self_attn.q_proj

    rand_inputs = torch.rand(4, 2048)
    # rand_inputs = torch.rand(4, 896)
    orig_out = wqlinear(rand_inputs)
    # fp16_weight, scales, zeros = wqlinear.unpack()

    fp16_weight_awq, scales_awq, zeros_awq = AwqToOnnxTransform.unpack_and_dequantize_awq(
        wqlinear.qweight, wqlinear.qzeros, wqlinear.scales, wqlinear.w_bit, wqlinear.group_size
    )
    # assert torch.mean(torch.abs(fp16_weight - fp16_weight_awq)) == 0
    # assert torch.mean(torch.abs(scales - scales_awq)) == 0
    # assert torch.mean(torch.abs((zeros - zeros_awq).float())) == 0

    wqlinear.weight = fp16_weight_awq
    new_module = QuantLinearORT(
        wqlinear.w_bit, wqlinear.group_size, wqlinear.in_features, wqlinear.out_features, wqlinear.bias is not None
    )
    new_module.bias = wqlinear.bias if wqlinear.bias is not None else None
    new_module.pack(wqlinear, scales_awq.T, zeros_awq.T, wqlinear.g_idx)

    del wqlinear

    new_out = new_module(rand_inputs)
    # import ipdb; ipdb.set_trace()
    assert torch.mean(torch.abs(orig_out - new_out)) < 0.02

    model_path = "qeff_models/test/tmp.onnx"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.onnx.export(new_module, rand_inputs, model_path)

    session = ort.InferenceSession(model_path)
    # import ipdb; ipdb.set_trace()
    onnx_out = session.run(None, {"inputs.1": rand_inputs.detach().numpy()})

    assert np.mean(np.abs(new_out.detach().numpy() - onnx_out[0])) < 1e-6


# FIXME: Change this test to only test AWQTransform without loading the model; directly initiaize WQLinear
@pytest.mark.parametrize("model_card_name", quant_models_test)
def test_awq_transform(model_card_name: str):
    replace_transformers_quantizers()
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_card_name, attn_implementation="eager", low_cpu_mem_usage=False
    )

    wqlinear = model_hf.model.layers[0].self_attn.q_proj
    rand_data = torch.rand(4, 2048)
    old_out = wqlinear(rand_data)
    new_module, transformed = AwqToOnnxTransform.apply(wqlinear)
    assert transformed
    new_out = new_module(rand_data)

    assert torch.mean(torch.abs(old_out - new_out)) < 0.02

    # FIXME: Move this part to customop test
    model_path = "qeff_models/test/tmp.onnx"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.onnx.export(new_module, rand_data, model_path)

    session = ort.InferenceSession(model_path)
    # import ipdb; ipdb.set_trace()
    onnx_out = session.run(None, {"inputs.1": rand_data.detach().numpy()})

    assert np.mean(np.abs(new_out.detach().numpy() - onnx_out[0])) < 1e-6
