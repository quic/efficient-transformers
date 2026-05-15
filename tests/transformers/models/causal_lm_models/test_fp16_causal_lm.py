# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import json
import os
from typing import Optional

import pytest
import torch
from transformers import AutoConfig

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner
from QEfficient.utils.test_utils import ModelConfig, load_hf_causal_lm_model

from .check_causal_models import (
    get_custom_n_layers,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/causal_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    causal_lm_fp16_test_models = config_data["causal_lm_fp16_test_models"]
test_models = [model["model_name"] for model in causal_lm_fp16_test_models]
model_config_dict = {model["model_name"]: model for model in causal_lm_fp16_test_models}


def check_causal_lm_pytorch_vs_kv_vs_ai100(
    model_name: str,
    manual_cleanup: callable,
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
    retain_full_kv: Optional[bool] = None,
    torch_dtype: Optional[torch.dtype] = torch.float32,
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

    model_hf = load_hf_causal_lm_model(model_name, num_hidden_layers=n_layer, config=config, torch_dtype=torch_dtype)
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
        dtype=torch_dtype,
    )

    if model_name not in ModelConfig.SWIFTKV_MODELS and model_name not in ModelConfig.EXTERNAL_MODELS:
        pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    is_tlm = False if num_speculative_tokens is None else True
    qeff_model = QEFFAutoModelForCausalLM(
        copy.deepcopy(model_hf), is_tlm=is_tlm, pretrained_model_name_or_path=model_name, qaic_config=qaic_config
    )

    pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

    if model_name not in ModelConfig.SWIFTKV_MODELS and model_name not in ModelConfig.EXTERNAL_MODELS:
        assert (pytorch_hf_tokens == pytorch_kv_tokens).all(), (
            "Tokens don't match for HF PyTorch model output and KV PyTorch model output"
        )
    qeff_model.export()
    qpc_path = qeff_model.compile(
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_cores=16,
        mxfp6=False,
        aic_hw_version="ai100",
        aic_enable_depth_first=False,
        num_speculative_tokens=num_speculative_tokens,
        prefill_only=prefill_only,
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
    )
    exec_info = qeff_model.generate(tokenizer, prompts=Constants.INPUT_STR)
    gen_len = pytorch_kv_tokens.shape[-1]
    cloud_ai_100_tokens = exec_info.generated_ids[0][
        :, :gen_len
    ]  # Because we always run for single input and single batch size
    if prefill_only:
        assert (pytorch_hf_tokens[0][0] == cloud_ai_100_tokens[0][0]).all(), (
            "prefill run output tokens don't match for ONNXRT output and Cloud AI 100 output."
        )
    else:
        assert (pytorch_hf_tokens == cloud_ai_100_tokens).all(), (
            "Tokens don't match for ONNXRT output and Cloud AI 100 output."
        )
        assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))
    if prefill_only is not None:
        return

    assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))
    manual_cleanup(qeff_model.onnx_path)


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models)
def test_full_fp16_causal_lm_pytorch_vs_kv_vs_ai100(model_name, manual_cleanup):
    torch.manual_seed(42)
    check_causal_lm_pytorch_vs_kv_vs_ai100(
        model_name=model_name, torch_dtype=torch.float16, manual_cleanup=manual_cleanup
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models)
def test_few_fp16_causal_lm_pytorch_vs_kv_vs_ai100(model_name, manual_cleanup):
    torch.manual_seed(42)
    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ai100(
        model_name=model_name, n_layer=n_layer, torch_dtype=torch.float16, manual_cleanup=manual_cleanup
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models)
def test_dummy_fp16_causal_lm_pytorch_vs_kv_vs_ai100(model_name, manual_cleanup):
    torch.manual_seed(42)
    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        check_causal_lm_pytorch_vs_kv_vs_ai100(
            model_name, n_layer=n_layer, torch_dtype=torch.float16, manual_cleanup=manual_cleanup
        )
    else:
        check_causal_lm_pytorch_vs_kv_vs_ai100(
            model_name, config=hf_config, torch_dtype=torch.float16, manual_cleanup=manual_cleanup
        )
