import copy

import pytest
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

configs = [
    AutoConfig.for_model("codegen", num_hidden_layers=2, num_attention_heads=4, hidden_size=128),
    AutoConfig.for_model("falcon", num_hidden_layers=2, num_attention_heads=4, hidden_size=128),
    AutoConfig.for_model("gpt2", num_hidden_layers=2, num_attention_heads=4, hidden_size=128),
    AutoConfig.for_model("gptj", num_hidden_layers=2, num_attention_heads=4, hidden_size=128),
    AutoConfig.for_model("llama", num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, hidden_size=128),
    AutoConfig.for_model("mistral", num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, hidden_size=128),
    AutoConfig.for_model("mixtral", num_hidden_layers=2, num_attention_heads=4, hidden_size=128),
    AutoConfig.for_model("mpt", num_hidden_layers=2, num_attention_heads=4, hidden_size=128),
    AutoConfig.for_model("phi", num_hidden_layers=2, num_attention_heads=4, hidden_size=128),
    AutoConfig.for_model("phi3", num_hidden_layers=2, num_attention_heads=4, hidden_size=128),
    AutoConfig.for_model("qwen2", num_hidden_layers=2, num_attention_heads=4, hidden_size=128),
    AutoConfig.for_model("starcoder2", num_hidden_layers=2, num_attention_heads=4, hidden_size=128),
]


def config_id(config):
    return config.model_type


@pytest.mark.parametrize("config", configs, ids=config_id)
def test_causal_lm_init(config):
    model = AutoModelForCausalLM.from_config(config)
    qeff_model = QEFFAutoModelForCausalLM(model)
    with pytest.raises(TypeError):
        QEFFAutoModelForCausalLM(AutoModel.from_config(config))
    assert qeff_model.model.__class__.__name__.startswith("QEff")


@pytest.mark.parametrize("config", configs, ids=config_id)
def test_causal_lm_pretrained(config, tmp_path):
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(tmp_path)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(tmp_path)
    assert qeff_model.model.__class__.__name__.startswith("QEff")


@pytest.mark.parametrize("config", configs, ids=config_id)
def test_causal_lm_hash(config):
    hash_0_0 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(config)).model_hash
    hash_0_1 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(config)).model_hash

    assert hash_0_0 == hash_0_1

    cfg1 = copy.deepcopy(config)
    cfg1.num_hidden_layers -= 1
    hash_1_0 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(cfg1)).model_hash
    cfg2 = AutoConfig.for_model(
        config.model_type,
        num_hidden_layers=config.num_hidden_layers - 1,
        num_attention_heads=config.num_attention_heads,
        hidden_size=config.hidden_size,
    )
    if hasattr(cfg2, "num_key_value_heads"):
        cfg2.num_key_value_heads = cfg1.num_key_value_heads
    hash_1_1 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(cfg2)).model_hash
    assert hash_1_0 == hash_1_1

    assert hash_0_0 != hash_1_0
