# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Any, Dict

import pytest
from transformers.models.llama import LlamaForCausalLM

from QEfficient import QEFFAutoModelForCausalLM

model_name_to_params_dict: Dict[str, Dict[str, Any]] = {
    # "gpt2": {
    #     "qeff_class": QEFFAutoModelForCausalLM,
    #     "hf_class": GPT2LMHeadModel,
    #     "prompt": "Equator is"
    # },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "qeff_class": QEFFAutoModelForCausalLM,
        "hf_class": LlamaForCausalLM,
        "prompt": "Equator is",
    }
}
model_names = model_name_to_params_dict.keys()


# FIXME: Add test cases for passing cache_dir, pretrained_model_path instead of card name, etc., Passing other kwargs
@pytest.mark.parametrize("model_name", model_names)
def test_qeff_auto_model_for_causal_lm(model_name: str):
    model: QEFFAutoModelForCausalLM = QEFFAutoModelForCausalLM.from_pretrained(model_name)  # type: ignore
    assert isinstance(model.model, model_name_to_params_dict[model_name]["hf_class"])  # type: ignore
    model.export_and_compile(
        num_cores=14,
        device_group=[
            0,
        ],
        batch_size=1,
        prompt_len=32,
        ctx_len=128,
        mxfp6=True,
        mxint8=False,
        mos=-1,
        aic_enable_depth_first=False,
        qpc_dir_suffix="vllm",
        onnx_dir_suffix="qserve",
        full_batch_size=1,
    )
    # FIXME: Enable later
    # QEfficient.cloud_ai_100_exec_kv(batch_size=1, tokenizer=model.tokenizer, qpc_path=qpc_dir_path, prompt=["My name is", ], input_len=3)
