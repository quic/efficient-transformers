# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.safetensor_materializer import (
    load_safetensor_checkpoint_into_model,
    materialize_safetensor_checkpoint,
    prepare_safetensor_streaming_from_pretrained_source,
)


def _write_tiny_safetensor_checkpoint(path):
    path.mkdir()
    with (path / "config.json").open("w") as handle:
        json.dump(
            {
                "architectures": ["TinyForCausalLM"],
                "model_type": "tiny",
                "torch_dtype": "bfloat16",
                "text_config": {"torch_dtype": "bfloat16"},
            },
            handle,
        )
    save_file(
        {
            "bf16_weight": torch.ones(2, 3, dtype=torch.bfloat16),
            "fp32_weight": torch.ones(3, 2, dtype=torch.float32),
            "int_weight": torch.ones(1, dtype=torch.int64),
        },
        path / "model.safetensors",
        metadata={"format": "pt"},
    )


def _write_sharded_tiny_safetensor_checkpoint(path):
    path.mkdir()
    with (path / "config.json").open("w") as handle:
        json.dump(
            {
                "architectures": ["TinyForCausalLM"],
                "model_type": "tiny",
                "torch_dtype": "bfloat16",
                "text_config": {"torch_dtype": "bfloat16"},
            },
            handle,
        )
    save_file(
        {
            "bf16_weight": torch.ones(2, 3, dtype=torch.bfloat16),
            "int_weight": torch.ones(1, dtype=torch.int64),
        },
        path / "model-00001-of-00002.safetensors",
        metadata={"format": "pt"},
    )
    save_file(
        {"fp32_weight": torch.ones(3, 2, dtype=torch.float32)},
        path / "model-00002-of-00002.safetensors",
        metadata={"format": "pt"},
    )
    with (path / "model.safetensors.index.json").open("w") as handle:
        json.dump(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "bf16_weight": "model-00001-of-00002.safetensors",
                    "int_weight": "model-00001-of-00002.safetensors",
                    "fp32_weight": "model-00002-of-00002.safetensors",
                },
            },
            handle,
        )


def test_materialize_safetensor_checkpoint_converts_floating_tensors(tmp_path):
    source_path = tmp_path / "source"
    _write_tiny_safetensor_checkpoint(source_path)

    materialized = materialize_safetensor_checkpoint(
        source_path,
        torch.float16,
        cache_dir=tmp_path / "qeff_materialized",
    )

    assert materialized.materialized
    assert materialized.path != source_path
    assert (materialized.path / "qeff_materialized_checkpoint.json").exists()

    with safe_open(materialized.path / "model.safetensors", framework="pt", device="cpu") as reader:
        assert reader.get_tensor("bf16_weight").dtype == torch.float16
        assert reader.get_tensor("fp32_weight").dtype == torch.float16
        assert reader.get_tensor("int_weight").dtype == torch.int64

    with (materialized.path / "config.json").open() as handle:
        config = json.load(handle)
    assert config["torch_dtype"] == "float16"
    assert config["text_config"]["torch_dtype"] == "float16"

    with safe_open(source_path / "model.safetensors", framework="pt", device="cpu") as reader:
        assert reader.get_tensor("bf16_weight").dtype == torch.bfloat16


def test_materialize_safetensor_checkpoint_converts_shards_in_parallel(tmp_path, monkeypatch):
    source_path = tmp_path / "source"
    _write_sharded_tiny_safetensor_checkpoint(source_path)
    monkeypatch.setattr("QEfficient.utils.safetensor_materializer._SAFETENSOR_PARALLEL_WORKERS", 2)

    materialized = materialize_safetensor_checkpoint(
        source_path,
        torch.float16,
        cache_dir=tmp_path / "qeff_materialized",
    )

    with safe_open(materialized.path / "model-00001-of-00002.safetensors", framework="pt", device="cpu") as reader:
        assert reader.get_tensor("bf16_weight").dtype == torch.float16
        assert reader.get_tensor("int_weight").dtype == torch.int64
    with safe_open(materialized.path / "model-00002-of-00002.safetensors", framework="pt", device="cpu") as reader:
        assert reader.get_tensor("fp32_weight").dtype == torch.float16
    with (materialized.path / "model.safetensors.index.json").open() as handle:
        index = json.load(handle)
    assert index["metadata"]["total_size"] > 0


def test_loader_hook_skips_streaming_without_explicit_dtype(tmp_path, monkeypatch):
    source_path = tmp_path / "source"
    _write_tiny_safetensor_checkpoint(source_path)
    monkeypatch.setenv("QEFF_MATERIALIZED_CHECKPOINT_DIR", str(tmp_path / "qeff_materialized"))
    kwargs = {"low_cpu_mem_usage": False}

    prepared_path = prepare_safetensor_streaming_from_pretrained_source(str(source_path), kwargs)

    assert prepared_path == str(source_path)
    assert kwargs["torch_dtype"] == torch.float32
    assert kwargs["low_cpu_mem_usage"] is False
    assert "use_safetensors" not in kwargs
    assert not (tmp_path / "qeff_materialized").exists()


def test_loader_hook_streams_source_checkpoint_without_materialized_copy(tmp_path, monkeypatch):
    source_path = tmp_path / "source"
    _write_tiny_safetensor_checkpoint(source_path)
    monkeypatch.setenv("QEFF_MATERIALIZED_CHECKPOINT_DIR", str(tmp_path / "qeff_materialized"))
    kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": False}

    prepared_path = prepare_safetensor_streaming_from_pretrained_source(str(source_path), kwargs)

    assert prepared_path == str(source_path.resolve())
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["use_safetensors"] is True
    assert not (tmp_path / "qeff_materialized").exists()
    with safe_open(f"{prepared_path}/model.safetensors", framework="pt", device="cpu") as reader:
        assert reader.get_tensor("bf16_weight").dtype == torch.bfloat16


def test_streaming_loader_populates_meta_model_from_safetensors(tmp_path):
    source_path = tmp_path / "source"
    config = AutoConfig.for_model(
        "gpt2",
        n_layer=1,
        n_head=2,
        n_embd=8,
        vocab_size=17,
        n_positions=16,
        n_ctx=16,
    )
    source_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").to(torch.float16)
    source_model.save_pretrained(source_path, safe_serialization=True)

    with torch.device("meta"):
        target_model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation="eager",
            torch_dtype=torch.float16,
        )

    result = load_safetensor_checkpoint_into_model(target_model, source_path, target_dtype=torch.float16)

    assert result.loaded_tensor_count > 0
    assert not result.missing_parameter_names
    assert all(parameter.device.type == "cpu" for parameter in target_model.parameters())
    assert next(target_model.parameters()).dtype == torch.float16
    assert torch.equal(source_model.transformer.wte.weight, target_model.transformer.wte.weight)


def test_from_pretrained_streams_fp16_checkpoint_without_materialized_copy(tmp_path, monkeypatch):
    source_path = tmp_path / "source"
    materialized_dir = tmp_path / "qeff_materialized"
    monkeypatch.setenv("QEFF_MATERIALIZED_CHECKPOINT_DIR", str(materialized_dir))
    config = AutoConfig.for_model(
        "gpt2",
        n_layer=1,
        n_head=2,
        n_embd=8,
        vocab_size=17,
        n_positions=16,
        n_ctx=16,
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").to(torch.bfloat16)
    model.save_pretrained(source_path, safe_serialization=True)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(str(source_path), torch_dtype=torch.float16)

    assert next(qeff_model.model.parameters()).dtype == torch.float16
    assert not materialized_dir.exists()
