# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Any, Dict

import pytest
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

import QEfficient
from QEfficient import QEFFAutoModelForCausalLM, QEFFCommonLoader

model_name_to_params_dict : Dict[str, Dict[str, Any]] = {
    "gpt2": {
        "qeff_class": QEFFAutoModelForCausalLM,
        "hf_class": GPT2LMHeadModel,
        "prompt": "Equator is"
    },
    
}
model_names = model_name_to_params_dict.keys()

#FIXME: Add test cases for passing cache_dir, pretrained_model_path instead of card name, etc., Passing other kwargs
@pytest.mark.parametrize("model_name", model_names)
def test_qeff_auto_model_for_causal_lm(model_name: str):
    model = QEFFCommonLoader.from_pretrained(model_name)
    assert isinstance(model, model_name_to_params_dict[model_name]['qeff_class'])
    assert isinstance(model.model, model_name_to_params_dict[model_name]['hf_class']) # type: ignore

    # Run transform
    QEfficient.transform(model)
    print(model)
