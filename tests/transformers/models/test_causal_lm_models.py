# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import os
from typing import List, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient.exporter.export_hf_to_cloud_ai_100 import qualcomm_efficient_converter
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils import hf_download
from QEfficient.utils._utils import create_json, load_hf_tokenizer
from QEfficient.utils.constants import Constants, QnnConstants
from QEfficient.utils.device_utils import get_available_device_id
from QEfficient.utils.run_utils import ApiRunner

extrenal_models = {"hpcai-tech/grok-1"}
test_models_qaic = [
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
    "hpcai-tech/grok-1",
]

test_dummy_model_configs = [
    # model_name, model_type, max_position_embeddings, num_hidden_layers, num_attention_heads, hidden_size, intermediate_size, vocab_size, additional_params
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", 128, 1, 2, 64, 256, 32000, {"num_key_value_heads": 1}),
    ("gpt2", "gpt2", 128, 1, 2, 64, 256, 50257, {"num_key_value_heads": 1}),
    (
        "Salesforce/codegen-350M-mono",
        "codegen",
        128,
        1,
        4,
        64,
        256,
        51200,
        {"num_key_value_heads": 1, "rotary_dim": 16},
    ),
    # ("microsoft/Phi-3-mini-4k-instruct","phi3", 128, 1, 2, 64, 256, 32064, {}), ouput not matching
    ("tiiuae/falcon-7b", "falcon", 128, 1, 2, 64, 256, 65024, {"num_key_value_heads": 1}),
    ("Qwen/Qwen2-0.5B", "qwen2", 128, 1, 2, 64, 256, 151936, {"num_key_value_heads": 1}),
    ("bigcode/starcoder2-3b", "starcoder2", 128, 1, 2, 64, 256, 49152, {"num_key_value_heads": 1}),
    ("Felladrin/Minueza-32M-Base", "mistral", 128, 1, 2, 64, 256, 32002, {"num_key_value_heads": 1}),
    ("wtang06/mpt-125m-c4", "mpt", 128, 1, 2, 64, 256, 50368, {}),
    ("hakurei/gpt-j-random-tinier", "gptj", 128, 1, 2, 64, 256, 50400, {"num_key_value_heads": 1, "rotary_dim": 16}),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "mixtral", 128, 1, 2, 64, 256, 32000, {"num_key_value_heads": 1}),
    (
        "meta-llama/Llama-3.2-1B",
        "llama",
        128,
        1,
        2,
        64,
        256,
        128256,
        {
            "num_key_value_heads": 1,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        },
    ),
    (
        "unsloth/gemma-2b",
        "gemma",
        128,
        1,
        2,
        64,
        256,
        256000,
        {"num_key_value_heads": 1, "_name_or_path": "unsloth/gemma-2b"},
    ),
    (
        "unsloth/gemma-2-2b",
        "gemma2",
        128,
        1,
        2,
        64,
        256,
        256000,
        {"num_key_value_heads": 1, "_name_or_path": "unsloth/gemma-2-2b"},
    ),
    # ("TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ", "llama",  128, 1, 2, 64, 256, 32003, {"num_key_value_heads": 1, "architectures": ["LlamaForCausalLM"], "pad_token_id": 0}),
    # ("TheBloke/Llama-2-7B-GPTQ", "llama", 128, 1, 2, 64, 256, 32000, {"num_key_value_heads": 2}),
    (
        "ibm-granite/granite-20b-code-base",
        "gpt_bigcode",
        128,
        1,
        2,
        64,
        256,
        49152,
        {"num_key_value_heads": 1, "activation_function": "gelu", "architectures": ["GPTBigCodeForCausalLM"]},
    ),
    # ("neuralmagic/Llama-3.2-3B-Instruct-FP8", "llama", 128, 1, 2, 64, 256, 128256, {"num_key_value_heads": 2}),
    # ("neuralmagic/Qwen2-0.5B-Instruct-FP8", "qwen2", 128, 1, 2, 64, 256, 151936, {"num_key_value_heads": 1,"quantization_config": {"activation_scheme": "static","ignored_layers": [  "lm_head" ],"quant_method": "fp8"}}),
    # ("ibm-granite/granite-3.1-2b-instruct", "granite", 128, 1, 2, 64, 256, 49155, {"num_key_value_heads": 2}),
    ("ibm-granite/granite-guardian-3.1-2b", "granite", 128, 1, 2, 64, 256, 49155, {"num_key_value_heads": 1}),
]


def get_model_configs_and_names(configs: List[tuple]):
    configs = [
        (
            AutoConfig.for_model(
                model_type,
                max_position_embeddings=max_position_embeddings,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                vocab_size=vocab_size,
                **additional_params,
            ),
            model_name,
        )
        for (
            model_name,
            model_type,
            max_position_embeddings,
            num_hidden_layers,
            num_attention_heads,
            hidden_size,
            intermediate_size,
            vocab_size,
            additional_params,
        ) in configs
    ]
    names = [y for (_, y) in configs]
    return configs, names


test_dummy_model_configs, test_dummy_model_names = get_model_configs_and_names(test_dummy_model_configs)

test_models_qnn = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3.2-1B",
    "unsloth/gemma-2b",
    "ibm-granite/granite-guardian-3.1-2b",
]
test_dummy_model_configs_qnn = [
    # model_name, model_type, max_position_embeddings, num_hidden_layers, num_attention_heads, hidden_size, intermediate_size, vocab_size, additional_params
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "mixtral", 128, 1, 2, 64, 256, 32000, {"num_key_value_heads": 1}),
    (
        "meta-llama/Llama-3.2-1B",
        "llama",
        128,
        1,
        2,
        64,
        256,
        128256,
        {
            "num_key_value_heads": 1,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        },
    ),
    (
        "unsloth/gemma-2b",
        "gemma",
        128,
        1,
        2,
        64,
        256,
        256000,
        {"num_key_value_heads": 1, "_name_or_path": "unsloth/gemma-2b"},
    ),
    ("ibm-granite/granite-guardian-3.1-2b", "granite", 128, 1, 2, 64, 256, 49155, {"num_key_value_heads": 1}),
]
test_dummy_model_configs_qnn, test_dummy_model_names_qnn = get_model_configs_and_names(test_dummy_model_configs_qnn)

spd_test_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2-0.5B",
]
test_dummy_model_configs_spd = [
    # model_name, model_type, max_position_embeddings, num_hidden_layers, num_attention_heads, hidden_size, intermediate_size, vocab_size, additional_params
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", 128, 1, 2, 64, 256, 32000, {"num_key_value_heads": 1}),
    ("Qwen/Qwen2-0.5B", "qwen2", 128, 1, 2, 64, 256, 151936, {"num_key_value_heads": 1}),
]
test_dummy_model_configs_spd, test_dummy_model_names_spd = get_model_configs_and_names(test_dummy_model_configs_spd)


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
        trust_remote_code=model_config["model_name"] in extrenal_models,
    )
    # Convert to FP32 if model is in BF16
    if getattr(model_hf.config, "torch_dtype", None) == torch.bfloat16:
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
    model_hf: Optional[nn.Module] = None,
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
    if model_hf is None:
        model_hf, _ = load_causal_lm_model(model_config)
    model_hf_cb = copy.deepcopy(model_hf)

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
    qeff_model = QEFFAutoModelForCausalLM(model_hf, is_tlm=is_tlm, pretrained_model_name_or_path=model_name)
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
    model_hf = model_hf_cb
    model_hf.eval()
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

    qeff_model = QEFFAutoModelForCausalLM(
        model_hf, continuous_batching=True, is_tlm=is_tlm, pretrained_model_name_or_path=model_name
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
    qeff_model = QEFFAutoModelForCausalLM(model, model_name=model_name, pretrained_model_name_or_path=model_name)
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
@pytest.mark.regular
@pytest.mark.parametrize(
    "test_dummy_model_config, test_dummy_model_name", test_dummy_model_configs, ids=test_dummy_model_names
)
def test_dummy_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(test_dummy_model_config, test_dummy_model_name):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """

    torch.manual_seed(42)
    model_hf = AutoModelForCausalLM.from_config(
        test_dummy_model_config,
        attn_implementation="eager",
    )
    model_hf.eval()
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(test_dummy_model_name, model_hf=model_hf)


@pytest.mark.nightly
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


@pytest.mark.on_qaic
@pytest.mark.regular
@pytest.mark.qnn
@pytest.mark.parametrize(
    "test_dummy_model_config_qnn, test_dummy_model_name_qnn",
    test_dummy_model_configs_qnn,
    ids=test_dummy_model_names_qnn,
)
def test_dummy_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_qnn(test_dummy_model_config_qnn, test_dummy_model_name_qnn):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """

    torch.manual_seed(42)
    model_hf = AutoModelForCausalLM.from_config(
        test_dummy_model_config_qnn,
        attn_implementation="eager",
    )
    model_hf.eval()
    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        test_dummy_model_name_qnn, enable_qnn=True, qnn_config=qnn_config_json_path, model_hf=model_hf
    )


@pytest.mark.nightly
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
@pytest.mark.regular
@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.parametrize(
    "test_dummy_model_config_spd, test_dummy_model_name_spd",
    test_dummy_model_configs_spd,
    ids=test_dummy_model_names_spd,
)
def test_dummy_causal_tlm_pytorch_vs_kv_vs_ort_vs_ai100(test_dummy_model_config_spd, test_dummy_model_name_spd):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """

    torch.manual_seed(42)
    model_hf = AutoModelForCausalLM.from_config(
        test_dummy_model_config_spd,
        attn_implementation="eager",
    )
    model_hf.eval()

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=test_dummy_model_name_spd, num_speculative_tokens=Constants.NUM_SPECULATIVE_TOKENS, model_hf=model_hf
    )


@pytest.mark.skip()  # remove when the SDK 1.20.0 issue solved for compiling this model
@pytest.mark.nightly
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
    model_name = "gpt2"  # hf-internal-testing/tiny-random-gpt2
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
