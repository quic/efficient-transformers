# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
from time import perf_counter

import onnx
import pytest
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM, QEFFAutoModelForCausalLMwithCB

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


@pytest.mark.parametrize(
    "QEFFAutoClass", [QEFFAutoModelForCausalLM, QEFFAutoModelForCausalLMwithCB], ids=["nocb", "cb"]
)
@pytest.mark.parametrize("config", configs, ids=config_id)
def test_causal_lm_init(QEFFAutoClass, config):
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoClass(model)
    with pytest.raises(TypeError):
        QEFFAutoClass(AutoModel.from_config(config, **model_kwargs))
    assert qeff_model.model.__class__.__name__.startswith("QEff")


@pytest.mark.parametrize(
    "QEFFAutoClass", [QEFFAutoModelForCausalLM, QEFFAutoModelForCausalLMwithCB], ids=["nocb", "cb"]
)
@pytest.mark.parametrize("config", configs, ids=config_id)
def test_causal_lm_pretrained(QEFFAutoClass, config, tmp_path):
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    model.save_pretrained(tmp_path)

    qeff_model = QEFFAutoClass.from_pretrained(tmp_path)
    assert qeff_model.model.__class__.__name__.startswith("QEff")


@pytest.mark.parametrize(
    "QEFFAutoClass", [QEFFAutoModelForCausalLM, QEFFAutoModelForCausalLMwithCB], ids=["nocb", "cb"]
)
@pytest.mark.parametrize("config", configs, ids=config_id)
def test_causal_lm_hash(QEFFAutoClass, config):
    hash_0_0 = QEFFAutoClass(AutoModelForCausalLM.from_config(config, **model_kwargs)).model_hash
    hash_0_1 = QEFFAutoClass(AutoModelForCausalLM.from_config(config, **model_kwargs)).model_hash

    assert hash_0_0 == hash_0_1

    cfg1 = copy.deepcopy(config)
    cfg1.num_hidden_layers -= 1
    hash_1_0 = QEFFAutoClass(AutoModelForCausalLM.from_config(cfg1, **model_kwargs)).model_hash
    cfg2 = copy.deepcopy(config)
    cfg2.num_hidden_layers -= 1
    hash_1_1 = QEFFAutoClass(AutoModelForCausalLM.from_config(cfg2, **model_kwargs)).model_hash
    assert hash_1_0 == hash_1_1

    assert hash_0_0 != hash_1_0


@pytest.mark.parametrize("config", configs, ids=config_id)
def test_causal_lm_export(config, tmp_path):
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(model)
    start = perf_counter()
    qeff_model.export(tmp_path)
    end = perf_counter()
    export_time_0 = end - start
    model_path = tmp_path.with_name(tmp_path.name + "-" + qeff_model.model_hash)
    assert model_path.is_dir()
    assert qeff_model.onnx_path.is_file()
    assert qeff_model.onnx_path.relative_to(model_path).parts == (qeff_model.model_name + ".onnx",)

    # Check if the KV-cache inputs and outputs are created
    onnx_model = onnx.load(qeff_model.onnx_path, load_external_data=False)
    retained_output_names = {
        x.name[: -len("_RetainedState")] for x in onnx_model.graph.output if x.name.endswith("_RetainedState")
    }
    retained_output_names.issubset({x.name for x in onnx_model.graph.input})

    start = perf_counter()
    qeff_model.export(tmp_path)
    end = perf_counter()
    export_time_1 = end - start
    assert export_time_1 < 0.01 * export_time_0


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("QEfficient.base.modeling_qeff.QEFF_HOME", tmp_path)
    yield tmp_path


@pytest.mark.parametrize("config", configs, ids=config_id)
def test_causal_lm_compile(config, tmp_cache):
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(model)
    start = perf_counter()
    qeff_model.compile(prefill_seq_len=8, ctx_len=16)
    end = perf_counter()
    compile_time_0 = end - start
    model_path = tmp_cache / (qeff_model.model_name + "-" + qeff_model.model_hash)

    # Check if ONNX is exported properly
    assert model_path.is_dir()
    assert qeff_model.onnx_path.is_file()
    assert qeff_model.onnx_path.relative_to(model_path).parts == (qeff_model.model_name + ".onnx",)

    # Check if QPC is compiled properly
    assert qeff_model.qpc_path.is_dir()
    assert (qeff_model.qpc_path / "programqpc.bin").is_file()
    assert qeff_model.qpc_path.relative_to(tmp_cache).parts[0] == qeff_model.model_name + "-" + qeff_model.model_hash

    # Check if there is no re-compilation
    start = perf_counter()
    qeff_model.compile(prefill_seq_len=8, ctx_len=16)
    end = perf_counter()
    compile_time_1 = end - start
    assert compile_time_1 < 0.01 * compile_time_0