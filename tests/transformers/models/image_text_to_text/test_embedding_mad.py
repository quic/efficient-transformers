# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from QEfficient.utils.test_utils import load_vlm_model

CONFIG_PATH = "tests/configs/image_text_model_configs.json"

DEFAULT_MAD_MAX = 1e-3

EXAMPLE_QUERIES = [
    {"text": "A woman playing with her dog on a beach at sunset."},
]

EXAMPLE_DOCUMENTS = [
    {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
]

with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    embedding_models = config_data["image_text_embedding_models"]

test_embedding_models = [model_config["model_name"] for model_config in embedding_models]
embedding_model_config_dict = {model["model_name"]: model for model in embedding_models}


def _load_embedder_cls():
    repo_root = Path(__file__).resolve().parents[4]
    example_path = repo_root / "examples/image_text_to_text/models/qwen3vl/embedding/qwen3_vl_embedding.py"
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_example", str(example_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.QEffQwen3VLEmbedder


def _resolve_model_source(model_name_or_path: str) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    return snapshot_download(repo_id=model_name_or_path)


def _compute_cpu_embeddings(model_hf, embedder, model_inputs: List[Dict[str, Any]]) -> torch.Tensor:
    embedding_rows = []
    for entry in model_inputs:
        conversation = embedder.format_model_input(
            text=entry.get("text"),
            image=entry.get("image"),
            video=entry.get("video"),
            instruction=entry.get("instruction"),
        )
        tokenized = embedder._tokenize_conversation(conversation)
        hf_inputs = {}
        for key, value in tokenized.items():
            hf_inputs[key] = value.to(next(model_hf.parameters()).device) if torch.is_tensor(value) else value

        with torch.no_grad():
            last_hidden_state = model_hf.model(**hf_inputs).last_hidden_state

        last_idx = tokenized["input_ids"].shape[1] - 1
        row = last_hidden_state[:, last_idx : last_idx + 1, :].reshape(last_hidden_state.shape[0], -1)
        embedding_rows.append(row.detach().cpu().to(torch.float32))

    embeddings = torch.cat(embedding_rows, dim=0)
    return F.normalize(embeddings, p=2, dim=-1)


@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.regular
@pytest.mark.parametrize("model_name", test_embedding_models)
def test_qwen3_vl_embedding_cpu_vs_ai100_mad_parity(model_name):
    torch.manual_seed(42)
    model_cfg = embedding_model_config_dict[model_name]
    model_source = _resolve_model_source(model_name)

    config = AutoConfig.from_pretrained(model_source, trust_remote_code=True, padding=True)
    if hasattr(config, "use_cache"):
        config.use_cache = True
    if hasattr(config, "text_config") and hasattr(config.text_config, "use_cache"):
        config.text_config.use_cache = True

    config.text_config.num_hidden_layers = model_cfg["num_layers"]
    config.vision_config.depth = model_cfg["vision_depth"]
    config.vision_config.deepstack_visual_indexes = [model_cfg["deepstack_index"]]

    model_hf = load_vlm_model(config)
    model_hf.eval()

    QEffQwen3VLEmbedder = _load_embedder_cls()
    embedder = QEffQwen3VLEmbedder(
        model_name_or_path=model_source,
        ctx_len=model_cfg["ctx_len"],
        num_cores=16,
        num_devices=1,
        compile_prefill_seq_len=model_cfg.get("compile_prefill_seq_len", None),
        num_hidden_layers=model_cfg["num_layers"],
        vision_depth=model_cfg["vision_depth"],
        deepstack_index=model_cfg["deepstack_index"],
    )

    model_inputs = EXAMPLE_QUERIES + EXAMPLE_DOCUMENTS
    cpu_embeddings = _compute_cpu_embeddings(model_hf=model_hf, embedder=embedder, model_inputs=model_inputs)
    ai100_embeddings = embedder.process(model_inputs, normalize=True)

    diff = torch.abs(cpu_embeddings - ai100_embeddings)
    mad_mean = float(diff.mean().item())
    mad_max = float(diff.max().item())
    threshold = float(model_cfg.get("mad_max_threshold", DEFAULT_MAD_MAX))

    print(f"[MAD] CPU vs AI100 mean={mad_mean:.6e}, max={mad_max:.6e}")
    assert mad_max <= threshold, (
        f"CPU vs AI100 MAD max {mad_max:.6e} exceeds threshold {threshold:.6e}. "
        f"Check prompt formatting, tokenization, prompt-length handling, and AI100 compile args."
    )
