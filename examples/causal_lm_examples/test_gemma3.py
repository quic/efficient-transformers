# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from typing import Optional

import pytest
from transformers import Gemma3ForCausalLM

from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils._utils import create_json, load_hf_tokenizer
from QEfficient.utils.constants import Constants, QnnConstants
from QEfficient.utils.device_utils import get_available_device_id
from QEfficient.utils.run_utils import ApiRunner

test_models_qaic = [
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "gpt2",
    # "Salesforce/codegen-350M-mono",
    # "microsoft/Phi-3-mini-4k-instruct",
    # "tiiuae/falcon-7b",
    # "Qwen/Qwen2-0.5B",
    # "bigcode/starcoder2-3b",
    # "Felladrin/Minueza-32M-Base",
    "google/gemma-3-4b-it",
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "wtang06/mpt-125m-c4",
    # "hakurei/gpt-j-random-tinier",
    # "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "meta-llama/Llama-3.2-1B",
    # "unsloth/gemma-2b",
    # "unsloth/gemma-2-2b",
    # "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",  # AWQ model
    # "TheBloke/Llama-2-7B-GPTQ",  # GPTQ model
    # "ibm-granite/granite-20b-code-base",
    # "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",  # naive-quantized compressed-tensor FP8 model per-channel weight, per-token activations
    # "neuralmagic/Llama-3.2-3B-Instruct-FP8",  # float quantized compressed-tensor per tensor both weight and activations
    # "neuralmagic/Qwen2-0.5B-Instruct-FP8",  # fp8 quant method, static, with lm head ignored
    # "ibm-granite/granite-3.1-2b-instruct",
    # "ibm-granite/granite-guardian-3.1-2b",
]

test_models_qnn = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3.2-1B",
    "unsloth/gemma-2b",
    "ibm-granite/granite-guardian-3.1-2b",
]

spd_test_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2-0.5B",
]


def load_causal_lm_model(model_config):
    """
    Function to load model from huggingface and transform to KV model
    --------

    :model_config: Dict

    :return model_hf, params
    """
    # model_path = hf_download(
    #     repo_id=model_config["model_name"],
    #     ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    # )
    model_hf = Gemma3ForCausalLM.from_pretrained(
        model_config["model_name"],
        use_cache=True,
        # num_hidden_layers=26,
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
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
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

    model_hf, _ = load_causal_lm_model(model_config)

    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
    config = model_hf.config
    batch_size = len(Constants.INPUT_STR)
    api_runner = ApiRunner(batch_size, tokenizer, config, ["Describe the transformers architecture in LLMs"], 16, 80)

    pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    is_tlm = False if num_speculative_tokens is None else True
    qeff_model = QEFFAutoModelForCausalLM(model_hf, is_tlm=is_tlm)

    pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    print(pytorch_hf_tokens)
    print(pytorch_kv_tokens)
    # assert (pytorch_hf_tokens == pytorch_kv_tokens).all(), (
    #     "Tokens don't match for HF PyTorch model output and KV PyTorch model output"
    # )
    onnx_model_path = qeff_model.export()
    ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=is_tlm)
    # gen_len = ort_tokens.shape[-1]
    print("ort tokens", ort_tokens)
    # assert (pytorch_kv_tokens == ort_tokens).all(), "Tokens don't match for ONNXRT output and PyTorch output."

    if not get_available_device_id():
        pytest.skip("No available devices to run model on Cloud AI 100")
    qeff_model.compile(
        prefill_seq_len=16,
        ctx_len=128,
        num_cores=16,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        node_precision_info="fp32.yaml",
    )
    exec_info = qeff_model.generate(tokenizer, prompts="Describe the transformers architecture in LLMs")
    cloud_ai_100_tokens = exec_info.generated_ids[0][
        :, :
    ]  # Because we always run for single input and single batch size
    # import ipdb; ipdb.set_trace()
    print("ai 100 tokens", cloud_ai_100_tokens)
    return


# FIXME: there should be a CB test here
@pytest.mark.skip()
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
@pytest.mark.parametrize("model_name", test_models_qaic)
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


@pytest.mark.skip
@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.parametrize("model_name", test_models_qnn)
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_qnn(model_name):
    """
    QNN Compilation Test
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    if model_name == "microsoft/Phi-3-mini-4k-instruct":
        n_layer = 2  # test only 2 layer models
    else:
        n_layer = 1

    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer=n_layer, enable_qnn=True, qnn_config=qnn_config_json_path
    )


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


@pytest.mark.skip
@pytest.mark.on_qaic
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_pl1():
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model for a prompt length of 1, both with and without continuous batching.
    """
    model_name = "gpt2"
    prompt_len = 1

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, prompt_len=prompt_len)


@pytest.mark.skip
@pytest.mark.on_qaic
@pytest.mark.qnn
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_pl1_qnn():
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model for a prompt length of 1, both with and without continuous batching.
    """
    model_name = "gpt2"
    prompt_len = 1

    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, prompt_len=prompt_len, enable_qnn=True, qnn_config=qnn_config_json_path
    )


@pytest.mark.skip
@pytest.mark.on_qaic
def test_prefiill_only_pytorch_vs_kv_vs_ort_vs_ai100():
    model_name = "gpt2"
    n_layer = 1
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, n_layer=n_layer, prefill_only=True)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, n_layer=n_layer, prefill_only=False)


@pytest.mark.skip
@pytest.mark.on_qaic
@pytest.mark.qnn
def test_prefiill_only_pytorch_vs_kv_vs_ort_vs_ai100_qnn():
    model_name = "gpt2"
    n_layer = 1

    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name, n_layer=n_layer, prefill_only=True, enable_qnn=True, qnn_config=qnn_config_json_path
    )

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name, n_layer=n_layer, prefill_only=False, enable_qnn=True, qnn_config=qnn_config_json_path
    )
