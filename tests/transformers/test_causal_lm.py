# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy

import pytest
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

configs = [
    # name, max_position_embeddings, num_hidden_layers, num_attention_heads, hidden_size, intermediate_size, vocab_size, additional_params
    ("codegen", 256, 2, 4, 128, 512, 127, {"rotary_dim": 16}),
    ("falcon", 256, 2, 4, 128, 512, 127, {}),
    ("gpt2", 256, 2, 4, 128, 512, 127, {}),
    ("gptj", 256, 2, 4, 128, 512, 127, {"rotary_dim": 16}),
    ("llama", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("mistral", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("mixtral", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("mpt", 256, 2, 4, 128, 512, 127, {}),
    ("phi", 256, 2, 4, 128, 512, 127, {}),
    ("phi3", 256, 2, 4, 128, 512, 127, {"pad_token_id": 0}),
    ("qwen2", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("starcoder2", 256, 2, 4, 128, 512, 127, {}),
]

configs = [
    AutoConfig.for_model(
        model_name,
        max_position_embeddings=max_position_embeddings,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        **additional_params,
    )
    for (
        model_name,
        max_position_embeddings,
        num_hidden_layers,
        num_attention_heads,
        hidden_size,
        intermediate_size,
        vocab_size,
        additional_params,
    ) in configs
]

model_kwargs = {"attn_implementation": "eager"}


def config_id(config):
    return config.model_type


@pytest.mark.parametrize("config", configs, ids=config_id)
def test_causal_lm_init(config):
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(model)
    with pytest.raises(TypeError):
        QEFFAutoModelForCausalLM(AutoModel.from_config(config, **model_kwargs))
    assert qeff_model.model.__class__.__name__.startswith("QEff")


@pytest.mark.parametrize("config", configs, ids=config_id)
def test_causal_lm_pretrained(config, tmp_path):
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    model.save_pretrained(tmp_path)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(tmp_path)
    assert qeff_model.model.__class__.__name__.startswith("QEff")


@pytest.mark.parametrize("config", configs, ids=config_id)
def test_causal_lm_hash(config):
    hash_0_0 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(config, **model_kwargs)).model_hash
    hash_0_1 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(config, **model_kwargs)).model_hash

    assert hash_0_0 == hash_0_1

    cfg1 = copy.deepcopy(config)
    cfg1.num_hidden_layers -= 1
    hash_1_0 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(cfg1, **model_kwargs)).model_hash
    cfg2 = copy.deepcopy(config)
    cfg2.num_hidden_layers -= 1
    hash_1_1 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(cfg2, **model_kwargs)).model_hash
    assert hash_1_0 == hash_1_1

    assert hash_0_0 != hash_1_0
