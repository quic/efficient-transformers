# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Generic unit coverage for image-text reranker model entries.

This test is intentionally model-list driven:
  - Add/remove reranker models only in tests/configs/reranker_model_configs.json
  - The same unit checks run for every configured reranker model
"""

import copy
import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from transformers import AutoConfig

from QEfficient.utils.test_utils import set_num_layers_vlm

CONFIG_PATH = "tests/configs/reranker_model_configs.json"


def _load_reranker_model_configs() -> List[Dict]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


RERANKER_MODEL_CONFIGS = _load_reranker_model_configs()


def _config_from_hf_or_skip(model_name: str):
    try:
        return AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Skipping {model_name}: unable to load HF config ({type(exc).__name__}: {exc})")


def _vision_num_layers(config) -> int:
    if hasattr(config.vision_config, "num_hidden_layers"):
        return int(config.vision_config.num_hidden_layers)
    if hasattr(config.vision_config, "depth"):
        return int(config.vision_config.depth)
    raise AssertionError("vision_config is missing num_hidden_layers/depth")


def test_reranker_model_list_is_present():
    assert RERANKER_MODEL_CONFIGS, (
        "reranker_model_configs.json is empty. Add reranker entries in tests/configs/reranker_model_configs.json."
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_cfg",
    RERANKER_MODEL_CONFIGS,
    ids=[cfg["model_name"] for cfg in RERANKER_MODEL_CONFIGS],
)
def test_reranker_config_reduction_keeps_valid_deepstack(model_cfg: Dict):
    model_name = model_cfg["model_name"]
    target_layers = int(model_cfg["num_layers"])
    assert target_layers > 0, f"{model_name}: num_layers must be > 0"

    cfg = _config_from_hf_or_skip(model_name)
    reduced_cfg = set_num_layers_vlm(copy.deepcopy(cfg), n_layer=target_layers)

    assert hasattr(reduced_cfg, "vision_config"), f"{model_name}: missing vision_config"
    assert hasattr(reduced_cfg, "text_config"), f"{model_name}: missing text_config"
    assert int(reduced_cfg.text_config.num_hidden_layers) == target_layers
    assert _vision_num_layers(reduced_cfg) == target_layers

    if hasattr(reduced_cfg.vision_config, "deepstack_visual_indexes"):
        deepstack_idxs = list(reduced_cfg.vision_config.deepstack_visual_indexes)
        assert deepstack_idxs, f"{model_name}: deepstack_visual_indexes must not be empty after layer reduction"
        assert min(deepstack_idxs) >= 0, f"{model_name}: deepstack indexes must be non-negative"
        assert max(deepstack_idxs) < _vision_num_layers(reduced_cfg), (
            f"{model_name}: deepstack indexes must be in [0, vision_num_layers)"
        )


# ---------------------------------------------------------------------------
# Tests: kv_offload=False (single QPC) runtime dispatch in QEffQwen3VLReranker
# ---------------------------------------------------------------------------


def _make_dummy_reranker():
    """Build a minimal QEffQwen3VLReranker with mocked internals."""
    # Import the reranker class from the examples directory via importlib
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "reranker_model",
        Path(__file__).parents[4] / "examples" / "reranker" / "qwen3vl" / "reranker_model.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Stub heavy dependencies so the module loads without hardware
    sys.modules.setdefault("QEfficient.generation.cloud_infer", MagicMock())
    sys.modules.setdefault("QEfficient.transformers.models.qwen3_vl._reranker_utils", MagicMock())
    spec.loader.exec_module(mod)
    return mod.QEffQwen3VLReranker


@pytest.fixture()
def reranker_cls():
    return _make_dummy_reranker()


def _fake_prepared_inputs(has_image: bool, prefill_len: int = 8):
    inputs = {
        "input_ids": torch.ones((1, prefill_len), dtype=torch.int64),
        "position_ids": torch.arange(prefill_len).reshape(1, 1, prefill_len).expand(4, 1, -1),
    }
    if has_image:
        inputs["pixel_values"] = torch.zeros((748, 1536), dtype=torch.float32)
        inputs["image_grid_thw"] = torch.zeros((1, 1, 22, 34), dtype=torch.int64)
    return inputs


def test_reranker_process_dispatches_to_dual_qpc(reranker_cls, monkeypatch):
    """process() with dict qpc_paths uses the dual-QPC path."""
    reranker = object.__new__(reranker_cls)
    reranker.yes_token_id = 0
    reranker.no_token_id = 1

    fake_logits = np.zeros((1, 1, 10), dtype=np.float32)
    fake_logits[0, 0, 0] = 2.0  # yes logit > no logit → score > 0.5

    monkeypatch.setattr(reranker, "_collect_contexts", lambda _: ([{"tokenized": {}}], 4, 22, 34))
    monkeypatch.setattr(reranker, "_prepare_inputs", lambda tok, prefill_seq_len: _fake_prepared_inputs(True))
    monkeypatch.setattr(
        reranker_cls, "_run_ai100_vision", staticmethod(lambda pi, vision_qpc_path: {"v": np.zeros((1,))})
    )
    monkeypatch.setattr(
        reranker_cls,
        "_run_ai100_prefill",
        staticmethod(lambda pi, vision_template, lang_qpc_path, vision_qpc_path: fake_logits),
    )
    monkeypatch.setattr(reranker_cls, "_score_from_logits", staticmethod(lambda logits, y, n: 0.88))

    scores = reranker.process(
        inputs={},
        qpc_paths={"vision_qpc_path": "v.qpc", "lang_qpc_path": "l.qpc"},
        prefill_seq_len=8,
    )
    assert scores == [0.88]


def test_reranker_process_dispatches_to_single_qpc(reranker_cls, monkeypatch):
    """process() with a non-dict qpc_paths uses the single-QPC path."""
    reranker = object.__new__(reranker_cls)
    reranker.yes_token_id = 0
    reranker.no_token_id = 1

    fake_logits = np.zeros((1, 1, 10), dtype=np.float32)

    monkeypatch.setattr(reranker, "_collect_contexts", lambda _: ([{"tokenized": {}}], 4, 22, 34))
    monkeypatch.setattr(reranker, "_prepare_inputs", lambda tok, prefill_seq_len: _fake_prepared_inputs(False))
    monkeypatch.setattr(
        reranker_cls,
        "_run_ai100_single_qpc_prefill",
        staticmethod(lambda pi, qpc_path: fake_logits),
    )
    monkeypatch.setattr(reranker_cls, "_score_from_logits", staticmethod(lambda logits, y, n: 0.72))

    scores = reranker.process(inputs={}, qpc_paths="/path/to/single.qpc", prefill_seq_len=8)
    assert scores == [0.72]


def test_reranker_process_single_qpc_with_pathlib(reranker_cls, monkeypatch):
    """Single QPC path also accepts a pathlib.Path object."""
    reranker = object.__new__(reranker_cls)
    reranker.yes_token_id = 0
    reranker.no_token_id = 1

    monkeypatch.setattr(reranker, "_collect_contexts", lambda _: ([{"tokenized": {}}], 4, 22, 34))
    monkeypatch.setattr(reranker, "_prepare_inputs", lambda tok, prefill_seq_len: _fake_prepared_inputs(True))
    monkeypatch.setattr(
        reranker_cls,
        "_run_ai100_single_qpc_prefill",
        staticmethod(lambda pi, qpc_path: np.zeros((1, 1, 10), dtype=np.float32)),
    )
    monkeypatch.setattr(reranker_cls, "_score_from_logits", staticmethod(lambda logits, y, n: 0.5))

    scores = reranker.process(inputs={}, qpc_paths=Path("/tmp/model.qpc"), prefill_seq_len=8)
    assert scores == [0.5]
