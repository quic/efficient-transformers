# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from typing import Optional

import numpy as np
import pytest
from transformers import AutoModelForCausalLM

from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils import hf_download
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.device_utils import get_available_device_id
from QEfficient.utils.run_utils import ApiRunner

test_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpt2",
    "Salesforce/codegen-350M-mono",
    "microsoft/Phi-3-mini-4k-instruct",
    "tiiuae/falcon-7b",
    "Qwen/Qwen2-0.5B",
    "bigcode/starcoder2-3b",
    "Felladrin/Minueza-32M-Base",
    "wtang06/mpt-125m-c4",
    "hakurei/gpt-j-random-tinier",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3.2-1B",
    "unsloth/gemma-2b",
    "unsloth/gemma-2-2b",
    "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",  # AWQ model
    "TheBloke/Llama-2-7B-GPTQ",  # GPTQ model
    "ibm-granite/granite-20b-code-base",
    # "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",  # naive-quantized compressed-tensor FP8 model per-channel weight, per-token activations
    "neuralmagic/Llama-3.2-3B-Instruct-FP8",  # float quantized compressed-tensor per tensor both weight and activations
    "neuralmagic/Qwen2-0.5B-Instruct-FP8",  # fp8 quant method, static, with lm head ignored
    "ibm-granite/granite-3.1-2b-instruct",
    "ibm-granite/granite-guardian-3.1-2b",
    "pfnet/plamo-13b-instruct",
]

spd_test_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


def load_causal_lm_model(model_config):
    """
    Function to load model from huggingface and transform to KV model
    --------

    :model_config: Dict

    :return model_hf, params
    """
    model_path = hf_download(
        repo_id=model_config["model_name"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_cache=True,
        num_hidden_layers=model_config["n_layer"],
        attn_implementation="eager",
        low_cpu_mem_usage=False,
    )  # Run models for single layers only
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


def check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    prompt_len: int = Constants.PROMPT_LEN,
    ctx_len: int = Constants.CTX_LEN,
    n_layer: int = 1,
    num_speculative_tokens: Optional[int] = None,
    prefill_only: Optional[bool] = None,
):
    """
    Validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
        :prompt_len (int): Prompt length for the model to compile.
        :ctx_len (int): Maximum context length to compile the model.
        :n_layers (int): Number of layers for the Model.
    """
    replace_transformers_quantizers()
    model_config = {"model_name": model_name}
    model_config["n_layer"] = n_layer

    model_hf, _ = load_causal_lm_model(model_config)

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

    is_tlm = False if num_speculative_tokens is None else True
    qeff_model = QEFFAutoModelForCausalLM(model_hf, is_tlm=is_tlm)

    pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

    assert (pytorch_hf_tokens == pytorch_kv_tokens).all(), (
        "Tokens don't match for HF PyTorch model output and KV PyTorch model output"
    )

    onnx_model_path = qeff_model.export()
    ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=is_tlm)
    gen_len = ort_tokens.shape[-1]

    assert (pytorch_kv_tokens == ort_tokens).all(), "Tokens don't match for ONNXRT output and PyTorch output."

    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")

    qpc_path = qeff_model.compile(
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_cores=14,
        mxfp6=False,
        aic_enable_depth_first=False,
        num_speculative_tokens=num_speculative_tokens,
        prefill_only=prefill_only,
    )
    exec_info = qeff_model.generate(tokenizer, prompts=Constants.INPUT_STR)
    cloud_ai_100_tokens = exec_info.generated_ids[0][
        :, :gen_len
    ]  # Because we always run for single input and single batch size
    if prefill_only:
        assert (ort_tokens[0][0] == cloud_ai_100_tokens[0][0]).all(), (
            "prefill run output tokens don't match for ONNXRT output and Cloud AI 100 output."
        )
    else:
        assert (ort_tokens == cloud_ai_100_tokens).all(), (
            "Tokens don't match for ONNXRT output and Cloud AI 100 output."
        )
        assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))
    if prefill_only is not None:
        return
    # testing for CB models
    model_hf, _ = load_causal_lm_model(model_config)
    full_batch_size = 4
    fbs_prompts = Constants.INPUT_STR * 4
    api_runner = ApiRunner(
        batch_size,
        tokenizer,
        config,
        fbs_prompts,
        Constants.PROMPT_LEN,
        Constants.CTX_LEN,
        full_batch_size,
    )

    pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch_CB(model_hf)
    pytorch_hf_tokens = np.vstack(pytorch_hf_tokens)

    qeff_model = QEFFAutoModelForCausalLM(model_hf, continuous_batching=True, is_tlm=is_tlm)
    onnx_model_path = qeff_model.export()

    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")

    # TODO: add prefill_only tests
    qpc_path = qeff_model.compile(
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_cores=14,
        mxfp6=False,
        aic_enable_depth_first=False,
        full_batch_size=full_batch_size,
        num_speculative_tokens=num_speculative_tokens,
    )
    exec_info_fbs = qeff_model.generate(tokenizer, prompts=fbs_prompts)

    assert all(
        [
            all(pt_token[:24] == cloud_token[:24])
            for pt_token, cloud_token in zip(pytorch_hf_tokens, exec_info_fbs.generated_ids)
        ]
    ), "Tokens don't match for  HF PyTorch model output and Cloud AI 100 output."
    assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))


# FIXME: there should be a CB test here
@pytest.mark.parametrize("model_name", ["gpt2"], ids=lambda x: x)
def test_causal_lm_export_with_deprecated_api(model_name):
    model_config = {"model_name": model_name}
    model_config["n_layer"] = 1
    model, _ = load_causal_lm_model(model_config)
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
    qeff_model = QEFFAutoModelForCausalLM(model)
    new_api_onnx_model_path = qeff_model.export()
    _, old_api_onnx_model_path = qualcomm_efficient_converter(
        model_name=model_name, model_kv=qeff_model, tokenizer=tokenizer
    )

    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=model.config,
        prompt=Constants.INPUT_STR,
        prompt_len=Constants.PROMPT_LEN,
        ctx_len=Constants.CTX_LEN,
    )

    new_api_ort_tokens = api_runner.run_kv_model_on_ort(new_api_onnx_model_path)
    old_api_ort_tokens = api_runner.run_kv_model_on_ort(old_api_onnx_model_path)

    assert (new_api_ort_tokens == old_api_ort_tokens).all(), (
        "New API output does not match old API output for ONNX export function"
    )


@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    if model_name == "microsoft/Phi-3-mini-4k-instruct":
        n_layer = 2  # test only 2 layer models
    else:
        n_layer = 1

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, n_layer=n_layer)


@pytest.mark.skip()  # remove when the SDK 1.20.0 issue solved for compiling this model
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", spd_test_models)
def test_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100(model_name):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """

    if model_name == "microsoft/Phi-3-mini-4k-instruct":
        n_layer = 2  # test only 2 layer models
    else:
        n_layer = 1

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer=n_layer, num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS
    )


@pytest.mark.on_qaic
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_pl1():
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model for a prompt length of 1, both with and without continuous batching.
    """
    model_name = "gpt2"
    prompt_len = 1

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, prompt_len=prompt_len)


@pytest.mark.on_qaic
def test_prefiill_only_pytorch_vs_kv_vs_ort_vs_ai100():
    model_name = "gpt2"
    n_layer = 1
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, n_layer=n_layer, prefill_only=True)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, n_layer=n_layer, prefill_only=False)
