# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Fast unit coverage for Qwen3-VL embedding helpers."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from QEfficient.transformers.models.qwen3_vl._embedding_utils import (
    QEffQwen3VLEmbedder,
    configure_embedding_model_config,
    format_model_input,
)

CONFIG_PATH = "tests/configs/image_text_model_configs.json"


def _load_embedding_model_configs():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config_data = json.load(file)
    return config_data.get("image_text_embedding_models", [])


def _dummy_config():
    return SimpleNamespace(
        use_cache=False,
        text_config=SimpleNamespace(use_cache=False, num_hidden_layers=32),
        vision_config=SimpleNamespace(depth=20, deepstack_visual_indexes=[-1, 3, 25], patch_size=16),
    )


class _DummyInnerModel:
    def __init__(self):
        self.config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=16))


class _DummyQEffModel:
    def __init__(self):
        self.model = _DummyInnerModel()


@pytest.mark.embedding
def test_embedding_model_list_is_present():
    model_configs = _load_embedding_model_configs()
    assert model_configs, (
        "image_text_embedding_models is empty. Add embedding entries in tests/configs/image_text_model_configs.json."
    )


@pytest.mark.embedding
def test_configure_embedding_model_config_sets_expected_fields():
    cfg = _dummy_config()
    configure_embedding_model_config(
        config=cfg,
        num_hidden_layers=1,
        vision_depth=9,
        deepstack_index=99,
        export_embedding=True,
    )

    assert cfg.use_cache is True
    assert cfg.text_config.use_cache is True
    assert int(cfg.text_config.num_hidden_layers) == 1
    assert int(cfg.vision_config.depth) == 9
    assert cfg.vision_config.deepstack_visual_indexes == [8]
    assert cfg.export_embedding is True


@pytest.mark.embedding
def test_format_model_input_adds_default_null_payload():
    conversation = format_model_input()
    assert len(conversation) == 2
    user_content = conversation[1]["content"]
    assert user_content and user_content[0]["type"] == "text"
    assert user_content[0]["text"] == "NULL"
    assert conversation[0]["content"][0]["text"].endswith(".")


@pytest.mark.embedding
def test_qwen3_vl_embedder_dummy_process_smoke(monkeypatch):
    embedder = QEffQwen3VLEmbedder(processor=None, model=_DummyQEffModel())

    contexts = [{"tokenized": {"kind": "image"}}, {"tokenized": {"kind": "text"}}]

    def _fake_collect_contexts(_inputs):
        return contexts, 8, 6, 10

    def _fake_prepare_qeff_inputs(qeff_model, tokenized_inputs, prefill_seq_len):
        del qeff_model
        prepared = {
            "input_ids": torch.arange(8, dtype=torch.int64).unsqueeze(0),
            "position_ids": torch.arange(prefill_seq_len, dtype=torch.int64).reshape(1, 1, prefill_seq_len),
        }
        if tokenized_inputs.get("kind") == "image":
            prepared["pixel_values"] = torch.ones((1, 3, 2, 2), dtype=torch.float32)
            prepared["image_grid_thw"] = torch.zeros((1, 1, 2, 2), dtype=torch.int64)
        return prepared, 8

    def _fake_run_ai100_vision(vision_qpc_path, prepared_inputs):
        del vision_qpc_path, prepared_inputs
        return {"vision_RetainedState": np.ones((1, 2), dtype=np.float16)}

    def _fake_run_ai100_prefill(prepared_inputs, vision_outputs, lang_qpc_path):
        del vision_outputs, lang_qpc_path
        if "pixel_values" in prepared_inputs:
            return np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        return np.array([[2.0, 1.0, 0.5, 1.0]], dtype=np.float32)

    monkeypatch.setattr(embedder, "_collect_contexts", _fake_collect_contexts)
    monkeypatch.setattr(QEffQwen3VLEmbedder, "_prepare_qeff_inputs", staticmethod(_fake_prepare_qeff_inputs))
    monkeypatch.setattr(QEffQwen3VLEmbedder, "_run_ai100_vision", staticmethod(_fake_run_ai100_vision))
    monkeypatch.setattr(QEffQwen3VLEmbedder, "_run_ai100_prefill", staticmethod(_fake_run_ai100_prefill))

    compile_specs = embedder.get_compile_specs(inputs=[{}, {}], ctx_len=64, prefill_seq_len=12)
    assert compile_specs == {"prefill_seq_len": 12, "ctx_len": 64, "img_size": 160, "height": 96, "width": 160}

    embeddings = embedder.process(
        inputs=[{}, {}],
        qpc_paths={"vision_qpc_path": "dummy_vision", "lang_qpc_path": "dummy_lang"},
        prefill_seq_len=12,
        normalize=True,
    )
    assert tuple(embeddings.shape) == (2, 4)
    norms = torch.linalg.norm(embeddings, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
