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
from transformers import AutoConfig, AutoModel, AutoModelForSpeechSeq2Seq

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForSpeechSeq2Seq

configs = [
    # name, max_source_positions, num_hidden_layers, num_attention_heads, hidden_size, encoder_ffn_dim, vocab_size, additional_params
    ("whisper", 1500, 4, 6, 384, 1536, 51865, {}),
]

configs = [
    AutoConfig.for_model(
        model_name,
        max_source_positions=max_source_positions,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        encoder_ffn_dim=encoder_ffn_dim,
        vocab_size=vocab_size,
        **additional_params,
    )
    for (
        model_name,
        max_source_positions,
        num_hidden_layers,
        num_attention_heads,
        hidden_size,
        encoder_ffn_dim,
        vocab_size,
        additional_params,
    ) in configs
]
config_ids = [x.model_type for x in configs]

model_kwargs = {"attn_implementation": "eager"}


def test_seq2seq_unsupported():
    model = AutoModelForSpeechSeq2Seq.from_config(AutoConfig.for_model("speech_to_text"))
    with pytest.warns():
        QEFFAutoModelForSpeechSeq2Seq(model)


@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_seq2seq_init(config):
    model = AutoModelForSpeechSeq2Seq.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
    with pytest.raises(TypeError):
        QEFFAutoModelForSpeechSeq2Seq(AutoModel.from_config(config, **model_kwargs))
    assert qeff_model.model.model.__class__.__name__.startswith("QEff")


@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_seq2seq_pretrained(config, tmp_path):
    model = AutoModelForSpeechSeq2Seq.from_config(config, **model_kwargs)
    model.save_pretrained(tmp_path)

    qeff_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(tmp_path)
    assert qeff_model.model.model.__class__.__name__.startswith("QEff")


@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_seq2seq_hash(config):
    hash_0_0 = QEFFAutoModelForSpeechSeq2Seq(AutoModelForSpeechSeq2Seq.from_config(config, **model_kwargs)).model_hash
    hash_0_1 = QEFFAutoModelForSpeechSeq2Seq(AutoModelForSpeechSeq2Seq.from_config(config, **model_kwargs)).model_hash

    assert hash_0_0 == hash_0_1

    cfg1 = copy.deepcopy(config)
    cfg1.num_hidden_layers -= 1
    hash_1_0 = QEFFAutoModelForSpeechSeq2Seq(AutoModelForSpeechSeq2Seq.from_config(cfg1, **model_kwargs)).model_hash
    cfg2 = copy.deepcopy(config)
    cfg2.num_hidden_layers -= 1
    hash_1_1 = QEFFAutoModelForSpeechSeq2Seq(AutoModelForSpeechSeq2Seq.from_config(cfg2, **model_kwargs)).model_hash
    assert hash_1_0 == hash_1_1
    assert hash_0_0 != hash_1_0


@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_seq2seq_export(config, tmp_path):
    model = AutoModelForSpeechSeq2Seq.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
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


# disable compile testing, compile not validated
# @pytest.mark.parametrize("cb", [False, True], ids=["nocb", "cb"])
# @pytest.mark.parametrize("config", configs, ids=config_ids)
# def test_causal_lm_compile(config, cb, tmp_cache):
#     model = AutoModelForSpeechSeq2Seq.from_config(config, **model_kwargs)
#     qeff_model = QEFFAutoModelForSpeechSeq2Seq(model, cb)
#     compile_params = {"prefill_seq_len": 8, "ctx_len": 16}
#     if cb:
#         compile_params["full_batch_size"] = 32
#         compile_params["batch_size"] = 8
#     qeff_model.compile(**compile_params)
#     model_path = tmp_cache / (qeff_model.model_name + "-" + qeff_model.model_hash)

#     # Check if ONNX is exported properly
#     assert model_path.is_dir()
#     assert qeff_model.onnx_path.is_file()
#     assert qeff_model.onnx_path.relative_to(model_path).parts == (qeff_model.model_name + ".onnx",)

#     # Check if QPC is compiled properly
#     assert qeff_model.qpc_path.is_dir()
#     assert (qeff_model.qpc_path / "programqpc.bin").is_file()
#     assert qeff_model.qpc_path.relative_to(tmp_cache).parts[0] == qeff_model.model_name + "-" + qeff_model.model_hash

#     # Check if there is no re-compilation
#     start = perf_counter()
#     qeff_model.compile(**compile_params)
#     end = perf_counter()
#     compile_time = end - start
#     assert compile_time < 2.0
