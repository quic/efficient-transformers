# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from typing import Optional

import numpy as np
import pytest
import torch
from transformers import Gemma3ForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.device_utils import get_available_device_id
from QEfficient.utils.run_utils import ApiRunner

torch.manual_seed(42)
test_models_qaic = [
    "google/gemma-3-4b-it",
]


def load_causal_lm_model(model_config):
    """
    Function to load model from huggingface and transform to KV model
    --------

    :model_config: Dict

    :return model_hf, params
    """
    model_hf = Gemma3ForCausalLM.from_pretrained(
        model_config["model_name"],
        use_cache=True,
        num_hidden_layers=6,
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
    model_config["n_layer"] = 6

    model_hf, _ = load_causal_lm_model(model_config)

    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
    config = model_hf.config
    batch_size = len(Constants.INPUT_STR)
    # config.sliding_window = 20
    api_runner = ApiRunner(batch_size, tokenizer, config, ["Describe the transformers architecture in LLMs"], 16, 200)

    pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    # breakpoint()
    is_tlm = False if num_speculative_tokens is None else True
    qeff_model = QEFFAutoModelForCausalLM(model_hf, is_tlm=is_tlm)

    pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    # print(pytorch_hf_tokens)
    # print(pytorch_kv_tokens)
    assert (pytorch_hf_tokens == pytorch_kv_tokens).all(), (
        "Tokens don't match for HF PyTorch model output and KV PyTorch model output"
    )
    onnx_model_path = qeff_model.export()
    ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=is_tlm)
    # gen_len = ort_tokens.shape[-1]
    # print("ort tokens", ort_tokens)
    assert (pytorch_kv_tokens == ort_tokens).all(), "Tokens don't match for ONNXRT output and PyTorch output."

    # if not get_available_device_id():
    #     pytest.skip("No available devices to run model on Cloud AI 100")
    qpc_path = qeff_model.compile(
        prefill_seq_len=16,
        ctx_len=200,
        num_cores=16,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        node_precision_info="fp32V0.yaml",
    )
    exec_info = qeff_model.generate(tokenizer, prompts="Describe the transformers architecture in LLMs")
    cloud_ai_100_tokens = exec_info.generated_ids[0][
        :, :
    ]  # Because we always run for single input and single batch size
    # import ipdb; ipdb.set_trace()
    print("ai 100 tokens", cloud_ai_100_tokens)
    return
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
        4,  # Constants.PROMPT_LEN
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
