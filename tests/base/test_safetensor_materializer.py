# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import (
    QEFFAutoModelForCausalLM,
    _prepare_weight_materialized_from_pretrained_source,
)
from QEfficient.utils.safetensor_materializer import materialize_safetensor_checkpoint


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


def test_materialized_loader_hook_uses_streaming_load_kwargs(tmp_path):
    source_path = tmp_path / "source"
    _write_tiny_safetensor_checkpoint(source_path)
    kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": False,
        "qeff_materialize_weights": True,
        "qeff_materialized_checkpoint_dir": tmp_path / "qeff_materialized",
    }

    prepared_path = _prepare_weight_materialized_from_pretrained_source(str(source_path), kwargs)

    assert prepared_path != str(source_path)
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["use_safetensors"] is True
    assert "qeff_materialize_weights" not in kwargs
    with safe_open(f"{prepared_path}/model.safetensors", framework="pt", device="cpu") as reader:
        assert reader.get_tensor("bf16_weight").dtype == torch.float16


def test_from_pretrained_loads_materialized_checkpoint(tmp_path):
    source_path = tmp_path / "source"
    materialized_dir = tmp_path / "qeff_materialized"
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

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        str(source_path),
        torch_dtype=torch.float16,
        qeff_materialize_weights=True,
        qeff_materialized_checkpoint_dir=materialized_dir,
    )

    assert next(qeff_model.model.parameters()).dtype == torch.float16
    assert any(materialized_dir.iterdir())
