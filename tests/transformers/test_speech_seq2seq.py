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
from transformers import AutoConfig, AutoModel, AutoModelForSpeechSeq2Seq

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForSpeechSeq2Seq
from QEfficient.utils.hash_utils import hash_dict_params

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
def test_seq2seq_export_and_hash(config, tmp_path):
    model_0_0 = QEFFAutoModelForSpeechSeq2Seq(AutoModelForSpeechSeq2Seq.from_config(config, **model_kwargs))
    model_0_0.export(tmp_path)
    model_path = tmp_path.with_name(tmp_path.name + "-" + model_0_0.export_hash)
    assert model_path.is_dir()
    assert model_0_0.onnx_path.is_file()
    assert model_0_0.onnx_path.relative_to(model_path).parts == (model_0_0.model_name + ".onnx",)

    # Check if the KV-cache inputs and outputs are created
    onnx_model = onnx.load(model_0_0.onnx_path, load_external_data=False)
    retained_output_names = {
        x.name[: -len("_RetainedState")] for x in onnx_model.graph.output if x.name.endswith("_RetainedState")
    }
    retained_output_names.issubset({x.name for x in onnx_model.graph.input})

    # Check if there is no re-export
    start = perf_counter()
    model_0_0.export(tmp_path)
    end = perf_counter()
    export_time = end - start
    assert export_time < 2.0

    # Check if the hashing is happening properly.
    hash_0_0 = model_0_0.export_hash
    model_0_1 = QEFFAutoModelForSpeechSeq2Seq(AutoModelForSpeechSeq2Seq.from_config(config, **model_kwargs))
    model_0_1.export(tmp_path)
    hash_0_1 = model_0_1.export_hash

    assert hash_0_0 == hash_0_1

    cfg1 = copy.deepcopy(config)
    cfg1.num_hidden_layers += 1
    model_1_0 = QEFFAutoModelForSpeechSeq2Seq(AutoModelForSpeechSeq2Seq.from_config(cfg1, **model_kwargs))
    model_1_0.export(tmp_path)
    hash_1_0 = model_1_0.export_hash

    cfg2 = copy.deepcopy(config)
    cfg2.num_hidden_layers += 1
    model_1_1 = QEFFAutoModelForSpeechSeq2Seq(AutoModelForSpeechSeq2Seq.from_config(cfg2, **model_kwargs))
    model_1_1.export(tmp_path)
    hash_1_1 = model_1_1.export_hash

    assert hash_1_0 == hash_1_1
    assert hash_0_0 != hash_1_0


@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_seq2seq_hash_creation(config, tmp_path):
    model = AutoModelForSpeechSeq2Seq.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
    qeff_model.export(tmp_path)
    hash_params = {}
    hash_params["config"] = qeff_model.model.config.to_diff_dict()
    hash_params["peft_config"] = None
    hash_params["applied_transform_names"] = qeff_model._transform_names()
    hash_params["qeff_auto_class"] = qeff_model.__class__.__name__

    export_params = {}
    export_params["output_names"] = qeff_model.model.get_output_names()
    export_params["dynamic_axes"] = qeff_model.model.get_onnx_dynamic_axes()
    hash_params["export_params"] = export_params
    manual_hash = hash_dict_params(hash_params)

    assert manual_hash == qeff_model.export_hash


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("QEfficient.utils.export_utils.QEFF_HOME", tmp_path)
    yield tmp_path


# disable compile testing, compile not validated
@pytest.mark.parametrize("config", configs, ids=config_ids)
def test_causal_lm_compile(config, tmp_cache):
    model = AutoModelForSpeechSeq2Seq.from_config(config, **model_kwargs)
    qeff_model = QEFFAutoModelForSpeechSeq2Seq(model)
    qeff_model.compile()
    model_path = tmp_cache / qeff_model.model_name / (qeff_model.model_name + "-" + qeff_model.export_hash)

    # Check if ONNX is exported properly
    assert model_path.is_dir()
    assert qeff_model.onnx_path.is_file()
    assert qeff_model.onnx_path.relative_to(model_path).parts == (qeff_model.model_name + ".onnx",)

    # Check if QPC is compiled properly
    assert qeff_model.qpc_path.is_dir()
    assert (qeff_model.qpc_path / "programqpc.bin").is_file()
    assert qeff_model.qpc_path.relative_to(tmp_cache).parts[1] == qeff_model.model_name + "-" + qeff_model.export_hash

    # Check if there is no re-compilation
    start = perf_counter()
    qeff_model.compile()
    end = perf_counter()
    compile_time = end - start
    assert compile_time < 2.0
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))
