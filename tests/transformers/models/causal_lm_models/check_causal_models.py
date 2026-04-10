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
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils import hf_download
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner
from QEfficient.utils.test_utils import ModelConfig

from ..check_model_results import dump_and_compare_results


def get_hf_config_from_custom_config(model_name, additional_params={}):
    """
    Function to get HF config from custom config file
    --------
    :model_name: str
    :additional_params: dict

    :return config
    """
    hf_config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS, **additional_params
    )
    return hf_config


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


def load_causal_lm_model(model_name, n_layer=-1, config=None):
    """
    Function to load model from huggingface or dummy models
    --------

    :model_name: str
    :n_layer: int
    :config: Autoconfig

    :return model_hf
    """
    model_path = hf_download(
        repo_id=model_name,
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    if config is None:
        kwargs = {
            "attn_implementation": "eager",
            "low_cpu_mem_usage": False,
            "use_cache": True,
        }
        if n_layer > 0:
            kwargs["num_hidden_layers"] = n_layer
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
            **kwargs,
        )
    else:
        model_hf = AutoModelForCausalLM.from_config(
            config,
            attn_implementation="eager",
            trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        )
    # Convert to FP32 if model is in BF16 or in FP16
    torch_dtype = getattr(model_hf.config, "torch_dtype", None)
    if torch_dtype == torch.bfloat16 or torch_dtype == torch.float16:
        model_hf = model_hf.to(torch.float32)
    model_hf.eval()
    return model_hf


def check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    manual_cleanup: callable,
    num_devices: int = 1,
    continuous_batching: bool = False,
    prompt_len: int = Constants.PROMPT_LEN,
    ctx_len: int = Constants.CTX_LEN,
    n_layer: int = -1,
    num_speculative_tokens: Optional[int] = None,
    prefill_only: Optional[bool] = None,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    config: Optional[AutoConfig] = None,
    pytorch_hf_tokens: Optional[list] = None,
    qaic_config: Optional[dict] = None,
    retain_full_kv: Optional[bool] = None,
    compare_results: bool = False,
):

    torch.manual_seed(42)
    replace_transformers_quantizers()
    model_hf = load_causal_lm_model(model_name, n_layer=n_layer, config=config)
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
    config = model_hf.config
    batch_size = len(Constants.INPUT_STR)
    prompts = Constants.INPUT_STR * 4 if continuous_batching else Constants.INPUT_STR
    full_batch_size = 4
    gen_len = 24
    is_tlm = False if num_speculative_tokens is None else True
    pytorch_hf_tokens = None
    pytorch_kv_tokens = None
    ort_tokens = None

    api_runner = ApiRunner(
        batch_size,
        tokenizer,
        config,
        prompts,
        Constants.PROMPT_LEN,
        Constants.CTX_LEN,
        full_batch_size if continuous_batching else None,
    )
    qeff_model = QEFFAutoModelForCausalLM(
        copy.deepcopy(model_hf),
        is_tlm=is_tlm,
        pretrained_model_name_or_path=model_name,
        continuous_batching=continuous_batching,
        qaic_config=qaic_config,
    )

    if continuous_batching is False:
        pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

    if model_name not in ModelConfig.SWIFTKV_MODELS and model_name not in ModelConfig.EXTERNAL_MODELS:
        if continuous_batching:
            pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch_CB(model_hf)
            pytorch_hf_tokens = np.vstack(pytorch_hf_tokens)
        else:
            pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    onnx_model_path = qeff_model.export()
    if continuous_batching is False:
        ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=is_tlm)
        gen_len = ort_tokens.shape[-1]

    if pytorch_hf_tokens is not None and ort_tokens is not None:
        assert (pytorch_hf_tokens == ort_tokens).all(), (
            "Tokens don't match for HF PyTorch model output and ONNXRT output."
        )

    if pytorch_kv_tokens is not None and ort_tokens is not None:
        assert (pytorch_kv_tokens == ort_tokens).all(), "Tokens don't match for ONNXRT output and PyTorch output."

    compiler_options = {}
    if continuous_batching and prompt_len == 1:
        prefill_spec = {
            "batch_size": batch_size,
            "seq_len": 1,
            "ctx_len": ctx_len,
            "full_batch_size": full_batch_size,
            "sliding_window": 128,
        }
        decode_spec = {
            "batch_size": full_batch_size,
            "seq_len": 1,
            "ctx_len": ctx_len,
            "full_batch_size": full_batch_size,
            "sliding_window": 128,
        }
        compiler_options["specializations"] = [prefill_spec, decode_spec]

    qpc_path = qeff_model.compile(
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_devices=num_devices,
        mxfp6=False,
        aic_enable_depth_first=False,
        num_speculative_tokens=num_speculative_tokens,
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
        retain_full_kv=retain_full_kv,
        prefill_only=prefill_only,
        batch_size=batch_size if continuous_batching else 1,
        full_batch_size=full_batch_size if continuous_batching else None,
        **compiler_options,
    )
    assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))

    # Generate
    exec_info = qeff_model.generate(tokenizer, prompts=prompts)

    if continuous_batching:
        cloud_ai_100_tokens = exec_info.generated_ids
        if model_name in ModelConfig.SWIFTKV_MODELS or model_name in ModelConfig.EXTERNAL_MODELS:
            api_runner = ApiRunner(
                batch_size, tokenizer, config, Constants.INPUT_STR, Constants.PROMPT_LEN, Constants.CTX_LEN
            )
            ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=is_tlm)
            assert all(
                [
                    all(ort_token[:24] == cloud_token[:24])
                    for ort_token, cloud_token in zip(ort_tokens, cloud_ai_100_tokens)
                ]
            ), "Tokens don't match for  HF PyTorch model output and Cloud AI 100 output."
        else:
            assert all(
                [
                    all(pt_token[:24] == cloud_token[:24])
                    for pt_token, cloud_token in zip(pytorch_hf_tokens, cloud_ai_100_tokens)
                ]
            ), "Tokens don't match for  HF PyTorch model output and Cloud AI 100 output."
    else:
        cloud_ai_100_tokens = exec_info.generated_ids[0][:, :gen_len]
        if prefill_only:
            assert (ort_tokens[0][0] == cloud_ai_100_tokens[0][0]).all(), (
                "prefill run output tokens don't match for ONNXRT output and Cloud AI 100 output."
            )
        else:
            assert (ort_tokens == cloud_ai_100_tokens).all(), (
                "Tokens don't match for ONNXRT output and Cloud AI 100 output."
            )

    manual_cleanup(onnx_model_path)  # Clean up the model files after the tests are done.
    if compare_results is False:
        return
    # Compare results for full model only.
    compile_params = {
        "prefill_seq_len": prompt_len,
        "ctx_len": ctx_len,
        "num_devices": num_devices,
        "mxfp6": False,
        "aic_enable_depth_first": False,
        "num_speculative_tokens": num_speculative_tokens,
        "enable_qnn": enable_qnn,
        "qnn_config": qnn_config,
        "retain_full_kv": retain_full_kv,
        "prefill_only": prefill_only,
        "batch_size": batch_size if continuous_batching else 1,
        "full_batch_size": full_batch_size if continuous_batching else None,
        "compiler_options": compiler_options,
    }
    assert dump_and_compare_results(
        model_name,
        compile_params,
        "causal_lm_model_results.json",
        cloud_ai_100_tokens,
        exec_info,
        pytorch_hf_tokens,
        pytorch_kv_tokens,
        ort_tokens,
    )
