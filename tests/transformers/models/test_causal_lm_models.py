# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import os
from typing import Optional

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils import hf_download
from QEfficient.utils._utils import create_json, load_hf_tokenizer
from QEfficient.utils.constants import Constants, QnnConstants
from QEfficient.utils.device_utils import get_available_device_id
from QEfficient.utils.run_utils import ApiRunner
from QEfficient.utils.test_utils import ModelConfig

test_models_causal = [
    "openai/gpt-oss-20b",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gpt2",
    "Salesforce/codegen-350M-mono",
    "microsoft/Phi-3-mini-4k-instruct",
    "tiiuae/falcon-7b",
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen3-0.6B",
    "bigcode/starcoder2-3b",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
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
    "hpcai-tech/grok-1",
    "Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
    "allenai/OLMo-2-0425-1B",
]

test_models_qnn = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3.2-1B",
    "unsloth/gemma-2b",
    "ibm-granite/granite-guardian-3.1-2b",
]

test_models_spd = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2-0.5B",
]

test_models_blockedKV = [
    # "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.2-1B",
]


def get_custom_n_layers(model_name):
    """
    Function to set number layers of the variuos types of models such as swiftkv models and others
    --------

    :model_name: str

    :return n_layer
    """
    if model_name in {"microsoft/Phi-3-mini-4k-instruct", "neuralmagic/Qwen2-0.5B-Instruct-FP8", "openai/gpt-oss-20b"}:
        return 2
    elif model_name in ModelConfig.SWIFTKV_MODELS:
        return None
    return 1


def load_causal_lm_model(model_name, n_layer=1, config=None):
    """
    Function to load model from huggingface and transform to KV model
    --------

    :model_name: str
    :n_layer: int
    :config: Autoconfig

    :return model_hf, params
    """
    torch.manual_seed(42)
    model_path = hf_download(
        repo_id=model_name,
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    if config is None:  # If custom config is not provided, load the model config from Hugging Face
        if n_layer is not None:
            # If n_layer is specified, load the model with that many layers
            model_hf = AutoModelForCausalLM.from_pretrained(
                model_path,
                use_cache=True,
                num_hidden_layers=n_layer,
                attn_implementation="eager",
                low_cpu_mem_usage=False,
                trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
            )
        else:
            # If n_layer is not specified, load the model without specifying the number of layers
            model_hf = AutoModelForCausalLM.from_pretrained(
                model_path,
                use_cache=True,
                attn_implementation="eager",
                low_cpu_mem_usage=False,
                trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
            )
    else:  # If custom config is provided, load the model using the config
        model_hf = AutoModelForCausalLM.from_config(
            config,
            attn_implementation="eager",
            trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        )
    # Convert to FP32 if model is in BF16 or in FP16
    torch_dtype = getattr(model_hf.config, "torch_dtype", None)
    if torch_dtype == torch.bfloat16 or torch_dtype == torch.float16:
        model_hf = model_hf.to(torch.float32)

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
    config: Optional[AutoConfig] = None,
    pytorch_hf_tokens: Optional[list] = None,
    qaic_config: Optional[dict] = None,
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
    if config is None:
        n_layer = get_custom_n_layers(model_name)
        model_hf, _ = load_causal_lm_model(model_name, n_layer=n_layer)
    else:
        model_hf, _ = load_causal_lm_model(model_name, config=config)

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
    if model_name not in ModelConfig.SWIFTKV_MODELS and model_name not in ModelConfig.EXTERNAL_MODELS:
        pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    is_tlm = False if num_speculative_tokens is None else True
    qeff_model = QEFFAutoModelForCausalLM(
        copy.deepcopy(model_hf), is_tlm=is_tlm, pretrained_model_name_or_path=model_name, qaic_config=qaic_config
    )
    pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

    if model_name not in ModelConfig.SWIFTKV_MODELS:
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
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
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

    if model_name not in ModelConfig.SWIFTKV_MODELS and model_name not in ModelConfig.EXTERNAL_MODELS:
        pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch_CB(model_hf)
        pytorch_hf_tokens = np.vstack(pytorch_hf_tokens)

    if model_name in ModelConfig.EXTERNAL_MODELS:
        pytorch_hf_tokens = [pytorch_hf_tokens for _ in range(full_batch_size)]

    qeff_model = QEFFAutoModelForCausalLM(
        model_hf,
        continuous_batching=True,
        is_tlm=is_tlm,
        pretrained_model_name_or_path=model_name,
        qaic_config=qaic_config,
    )
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
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
    )
    exec_info_fbs = qeff_model.generate(tokenizer, prompts=fbs_prompts)

    if model_name in ModelConfig.SWIFTKV_MODELS:
        assert all(
            [
                all(ort_token[:24] == cloud_token[:24])
                for ort_token, cloud_token in zip(ort_tokens, exec_info_fbs.generated_ids)
            ]
        ), "Tokens don't match for  HF PyTorch model output and Cloud AI 100 output."
    else:
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
    model, _ = load_causal_lm_model(model_name, n_layer=1)
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
    qeff_model = QEFFAutoModelForCausalLM(model, model_name=model_name, pretrained_model_name_or_path=model_name)
    new_api_onnx_model_path = qeff_model.export()

    # Again loading model since the export moves model to meta device
    model, _ = load_causal_lm_model(model_name, n_layer=1)
    qeff_model = QEFFAutoModelForCausalLM(model, model_name=model_name, pretrained_model_name_or_path=model_name)
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
@pytest.mark.regular
@pytest.mark.parametrize("model_name", test_models_causal)
def test_custom_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, custom_causal_model_config_dict):
    """
    Test function to validate the dummy PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    config = custom_causal_model_config_dict.get(model_name)

    # Using fixed reference tokens for external models for specific test cases.
    # These tokens are hardcoded, therefore will not match if the model config changes.
    pytorch_hf_tokens = None
    if model_name in ModelConfig.EXTERNAL_MODELS:
        pytorch_hf_tokens = ModelConfig.EXTERNAL_MODELS[model_name]["pytorch_hf_tokens_custom_case"]

    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, n_layer=n_layer, pytorch_hf_tokens=pytorch_hf_tokens)
    else:
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, config=config, pytorch_hf_tokens=pytorch_hf_tokens)


@pytest.mark.nightly
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    n_layer = get_custom_n_layers(model_name)

    # Using fixed reference tokens for external models for specific test cases.
    # These tokens are hardcoded, therefore will not match if the model config changes.
    pytorch_hf_tokens = None
    if model_name in ModelConfig.EXTERNAL_MODELS:
        pytorch_hf_tokens = ModelConfig.EXTERNAL_MODELS[model_name]["pytorch_hf_tokens_normal_case"]

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer=n_layer, pytorch_hf_tokens=pytorch_hf_tokens
    )


@pytest.mark.on_qaic
@pytest.mark.regular
@pytest.mark.qnn
@pytest.mark.parametrize("model_name", test_models_qnn)
def test_custom_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_qnn(model_name, custom_causal_model_config_dict):
    """
    QNN Setup
    Test function to validate the dummy PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    config = custom_causal_model_config_dict.get(model_name)
    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name, enable_qnn=True, qnn_config=qnn_config_json_path, config=config
    )


@pytest.mark.nightly
@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.parametrize("model_name", test_models_qnn)
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_qnn(model_name):
    """
    QNN Setup
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)
    n_layer = get_custom_n_layers(model_name)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name, n_layer=n_layer, enable_qnn=True, qnn_config=qnn_config_json_path
    )


@pytest.mark.regular
@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.parametrize("model_name", test_models_spd)
def test_custom_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, custom_causal_model_config_dict):
    """
    Test function to validate the dummy PyTorch model for speculative decoding, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    config = custom_causal_model_config_dict.get(model_name)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS,
        config=config,
    )


@pytest.mark.nightly
@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models_spd)
def test_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100(model_name):
    """
    Test function to validate the PyTorch model for speculative decoding, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    n_layer = get_custom_n_layers(model_name)

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


@pytest.mark.on_qaic
def test_prefiill_only_pytorch_vs_kv_vs_ort_vs_ai100():
    model_name = "gpt2"
    n_layer = 1
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, n_layer=n_layer, prefill_only=True)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, n_layer=n_layer, prefill_only=False)


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


@pytest.mark.parametrize("model_name", test_models_blockedKV)
def test_causal_blockedKV_pytorch_vs_kv_vs_ort_vs_ai100(model_name):
    """
    Test function to validate the PyTorch model for KV blocking, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    n_layer = get_custom_n_layers(model_name)

    qaic_config = dict(num_kv_blocks=Constants.NUM_KV_BLOCKS)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, n_layer=n_layer, qaic_config=qaic_config)


@pytest.mark.parametrize("model_name", test_models_blockedKV)
def test_causal_nonBlockedKV_pytorch_vs_kv_vs_ort_vs_ai100(model_name):
    """
    Test function to validate the PyTorch model for KV blocking, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    n_layer = get_custom_n_layers(model_name)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, n_layer=n_layer)
