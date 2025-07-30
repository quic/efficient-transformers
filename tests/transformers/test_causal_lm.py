# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import os
from time import perf_counter

import onnx
import pytest
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

configs = [
    # name, max_position_embeddings, num_hidden_layers, num_attention_heads, hidden_size, intermediate_size, vocab_size, additional_params
    ("gpt2", 256, 2, 4, 128, 512, 127, {}),
    ("codegen", 256, 2, 4, 128, 512, 127, {"rotary_dim": 16}),
    ("falcon", 256, 2, 4, 128, 512, 127, {}),
    ("gptj", 256, 2, 4, 128, 512, 127, {"rotary_dim": 16}),
    ("llama", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("mistral", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("mixtral", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("mpt", 256, 2, 4, 128, 512, 127, {}),
    ("phi", 256, 2, 4, 128, 512, 127, {}),
    ("phi3", 256, 2, 4, 128, 512, 127, {"pad_token_id": 0}),
    ("qwen2", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("starcoder2", 256, 2, 4, 128, 512, 127, {}),
    ("granite", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("olmo2", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
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
config_ids = [x.model_type for x in configs]

model_kwargs = {"attn_implementation": "eager"}


@pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
def test_causal_lm_unsupported(cb):
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model("opt"))
    with pytest.warns():
        QEFFAutoModelForCausalLM(model, cb)


@pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_causal_lm_init(config, cb):
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(model, cb)
    with pytest.raises(TypeError):
        QEFFAutoModelForCausalLM(AutoModel.from_config(config, **model_kwargs), cb)
    assert qeff_model.model.__class__.__name__.startswith("QEff")


@pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_causal_lm_pretrained(config, cb, tmp_path):
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    model.save_pretrained(tmp_path)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(tmp_path, cb)
    assert qeff_model.model.__class__.__name__.startswith("QEff")


@pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_causal_lm_hash(config, cb):
    hash_0_0 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(config, **model_kwargs), cb).model_hash
    hash_0_1 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(config, **model_kwargs), cb).model_hash

    assert hash_0_0 == hash_0_1

    cfg1 = copy.deepcopy(config)
    cfg1.num_hidden_layers -= 1
    hash_1_0 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(cfg1, **model_kwargs), cb).model_hash
    cfg2 = copy.deepcopy(config)
    cfg2.num_hidden_layers -= 1
    hash_1_1 = QEFFAutoModelForCausalLM(AutoModelForCausalLM.from_config(cfg2, **model_kwargs), cb).model_hash
    assert hash_1_0 == hash_1_1

    assert hash_0_0 != hash_1_0

    if cb:
        hash_0_no_cb = QEFFAutoModelForCausalLM(
            AutoModelForCausalLM.from_config(config, **model_kwargs), False
        ).model_hash
        assert hash_0_0 != hash_0_no_cb


@pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_causal_lm_export(config, cb, tmp_path):
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(model, cb)
    qeff_model.export(tmp_path)
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

    # Check if there is no re-export
    start = perf_counter()
    qeff_model.export(tmp_path)
    end = perf_counter()
    export_time = end - start
    assert export_time < 2.0


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("QEfficient.base.modeling_qeff.QEFF_HOME", tmp_path)
    yield tmp_path


@pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_causal_lm_compile(config, cb, tmp_cache):
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForCausalLM(model, cb)
    compile_params = {"prefill_seq_len": 8, "ctx_len": 16}
    if cb:
        compile_params["full_batch_size"] = 32
        compile_params["batch_size"] = 8
    qeff_model.compile(**compile_params)
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
    qeff_model.compile(**compile_params)
    end = perf_counter()
    compile_time = end - start
    assert compile_time < 2.0
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))
