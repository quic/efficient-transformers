# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Generic unit coverage for image-text reranker model entries.

This test is intentionally model-list driven:
  - Add/remove reranker models only in tests/configs/image_text_model_configs.json
  - The same unit checks run for every configured reranker model
"""

import copy
import json
import os
from typing import Dict, List

import pytest
from transformers import AutoConfig

from QEfficient.utils.test_utils import set_num_layers_vlm

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/image_text_model_configs.json")


def _load_reranker_model_configs() -> List[Dict]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config_data = json.load(file)
    return config_data.get("image_text_reranker_models", [])


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
        "image_text_reranker_models is empty. Add reranker entries in tests/configs/image_text_model_configs.json."
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
