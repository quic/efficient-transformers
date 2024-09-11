# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest
from peft import IA3Config, LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient import QEffAutoPeftModelForCausalLM

configs = [
    pytest.param(
        AutoConfig.for_model(
            "llama", num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, hidden_size=128
        ),
        LoraConfig(target_modules=["q_proj", "v_proj"]),
        id="llama-2l-4h-2kvh-128d-qv",
    ),
    pytest.param(
        AutoConfig.for_model("mistral", num_hidden_layers=2, num_attention_heads=4, hidden_size=128),
        LoraConfig(target_modules=["q_proj", "v_proj"]),
        id="mistral-2l-4h-128d-qv",
    ),
]


@pytest.mark.parametrize("base_config,adapter_config", configs)
def test_auto_peft_model_for_causal_lm_init(base_config, adapter_config):
    base_model = AutoModelForCausalLM.from_config(base_config, attn_implementation="eager")
    ia3_model = get_peft_model(base_model, IA3Config())

    with pytest.raises(TypeError):
        QEffAutoPeftModelForCausalLM(base_model)

    with pytest.raises(NotImplementedError):
        QEffAutoPeftModelForCausalLM(ia3_model)

    base_model = AutoModelForCausalLM.from_config(base_config, attn_implementation="eager")
    lora_model = get_peft_model(base_model, adapter_config, "testAdapter101")
    lora_model.add_adapter("testAdapter102", adapter_config)
    qeff_model = QEffAutoPeftModelForCausalLM(lora_model)
    assert set(qeff_model.adapter_weights.keys()) == {"testAdapter101", "testAdapter102"}


@pytest.mark.parametrize("base_config,adapter_config", configs)
def test_auto_peft_model_for_causal_lm_from_pretrained(base_config, adapter_config, tmp_path):
    base_path = tmp_path / "base"
    adapter_path = tmp_path / "adapter"
    adapter_name = "testAdapter103"

    base_model = AutoModelForCausalLM.from_config(base_config, attn_implementation="eager")
    base_model.save_pretrained(base_path)

    base_model = AutoModelForCausalLM.from_pretrained(base_path)
    lora_model = get_peft_model(base_model, adapter_config, adapter_name)
    lora_model.save_pretrained(adapter_path)

    qeff_model = QEffAutoPeftModelForCausalLM.from_pretrained(adapter_path / adapter_name, adapter_name)
    assert set(qeff_model.adapter_weights.keys()) == {"testAdapter103"}

    with pytest.raises(NotImplementedError):
        QEffAutoPeftModelForCausalLM.from_pretrained(adapter_path / adapter_name, full_batch_size=4)
