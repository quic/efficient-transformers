# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import numpy as np
import pytest

from QEfficient.compile.compile_helper import compile_kv_model_on_cloud_ai_100
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.device_utils import get_available_device_id
from QEfficient.utils.run_utils import ApiRunner
from tests.utils import load_pytorch_model, replace_transformers_quantizers

test_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpt2",
    "Salesforce/codegen-350M-mono",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "tiiuae/falcon-7b",
    "Qwen/Qwen2-0.5B",
    "bigcode/starcoder2-3b",
    "Felladrin/Minueza-32M-Base",
    "wtang06/mpt-125m-c4",
    "hakurei/gpt-j-random-tinier",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-2b",
    "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",  # AWQ model
    "TheBloke/Llama-2-7B-GPTQ",  # GPTQ model
    "ibm-granite/granite-20b-code-base",
]


@pytest.mark.causal_lm
@pytest.mark.parametrize("model_name", test_models)
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name):
    """
    Test function to validate the model before and after KV changes on Pytorch
    :param model_name: Name of model.
    """
    replace_transformers_quantizers()
    if model_name == "microsoft/Phi-3-mini-4k-instruct":
        n_layer = 2  # test only 2 layer models
    else:
        n_layer = 1

    model_config = {"model_name": model_name}
    model_config["n_layer"] = n_layer

    model_hf, _ = load_pytorch_model(model_config)

    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
    config = model_hf.config
    batch_size = len(Constants.INPUT_STR)
    api_runner = ApiRunner(
        batch_size,
        tokenizer,
        config,
        Constants.INPUT_STR,
        Constants.PROMPT_LEN,
        Constants.CTX_LEN,
    )

    pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, f"{model_name}")

    pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

    assert (
        pytorch_hf_tokens == pytorch_kv_tokens
    ).all(), "Tokens don't match for HF PyTorch model output and KV PyTorch model output"

    onnx_model_path = qeff_model.export()
    ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path)

    assert (pytorch_kv_tokens == ort_tokens).all(), "Tokens don't match for ONNXRT output and PyTorch output."

    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")

    base_path = os.path.dirname(onnx_model_path)
    tests_qpc_dir = os.path.join(base_path, "tests_qpc")
    os.makedirs(tests_qpc_dir, exist_ok=True)

    _, test_qpcs_path = compile_kv_model_on_cloud_ai_100(
        onnx_path=onnx_model_path,
        specializations_json="scripts/specializations.json",
        num_cores=14,
        base_path=tests_qpc_dir,
        mxfp6=False,
        custom_io_path=os.path.join(base_path, "custom_io_fp16.yaml"),
        aic_enable_depth_first=False,
    )

    cloud_ai_100_tokens = api_runner.run_kv_model_on_cloud_ai_100(test_qpcs_path)
    gen_len = ort_tokens.shape[-1]
    assert (
        ort_tokens == cloud_ai_100_tokens[:, :gen_len]
    ).all(), "Tokens don't match for ONNXRT output and Cloud AI 100 output."

    # testing for CB models
    model_hf, _ = load_pytorch_model(model_config)
    full_batch_size = 1
    api_runner = ApiRunner(
        batch_size,
        tokenizer,
        config,
        Constants.INPUT_STR,
        Constants.PROMPT_LEN,
        Constants.CTX_LEN,
        full_batch_size,
    )

    pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch_CB(model_hf)
    pytorch_hf_tokens = np.vstack(pytorch_hf_tokens)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, f"{model_name}")
    onnx_model_path = qeff_model.export()

    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")

    base_path = os.path.dirname(onnx_model_path)
    tests_qpc_dir = os.path.join(base_path, "tests_qpc_cb")
    os.makedirs(tests_qpc_dir, exist_ok=True)

    _, test_qpcs_path = compile_kv_model_on_cloud_ai_100(
        onnx_path=onnx_model_path,
        specializations_json="scripts/specializations.json",
        num_cores=14,
        base_path=tests_qpc_dir,
        mxfp6=False,
        custom_io_path=os.path.join(base_path, "custom_io_fp16.yaml"),
        aic_enable_depth_first=False,
    )

    cloud_ai_100_tokens = api_runner.run_kv_model_on_cloud_ai_100(test_qpcs_path)

    pytorch_hf_tokens = pytorch_hf_tokens[:, : api_runner.gen_len]
    cloud_ai_100_tokens = cloud_ai_100_tokens[:, : api_runner.gen_len]

    assert (
        pytorch_hf_tokens == cloud_ai_100_tokens
    ).all(), "Tokens don't match for  HF PyTorch model output and Cloud AI 100 output."
