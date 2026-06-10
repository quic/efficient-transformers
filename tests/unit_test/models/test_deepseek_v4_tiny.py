# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""4-stage validation for ``deepseek_v4`` (deepseek-ai/DeepSeek-V4-Pro).

Stages 1-3 run on CPU; Stage 4 needs AI 100 hardware (``@pytest.mark.on_qaic``).
Uses ``AutoConfig.for_model("deepseek_v4", **dummy)`` so no full-weight
download is needed.
"""

import copy
import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "causal_model_configs.json")
with open(_CONFIG_FILE) as _f:
    _all = json.load(_f)
_V4_ENTRY = next(e for e in _all["causal_lm_models"] if e["model_type"] == "deepseek_v4")

MODEL_NAME = _V4_ENTRY["model_name"]
TOKENIZER_ID = _V4_ENTRY.get("tokenizer_id", "gpt2")
DUMMY_PARAMS = _V4_ENTRY["additional_params"]

PROMPT_LEN = 8
CTX_LEN = 32
PROMPT = "My name is"


def _build_dummy_model():
    config = AutoConfig.for_model("deepseek_v4", **DUMMY_PARAMS)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager", torch_dtype=torch.float32)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.02)
    model.eval()
    return model


def _encode_in_vocab(tokenizer, prompt, vocab_size, prompt_len):
    ids = tokenizer.encode(prompt, return_tensors="pt")
    ids = ids[:, :prompt_len]
    if ids.shape[1] < prompt_len:
        pad = torch.zeros(ids.shape[0], prompt_len - ids.shape[1], dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    return ids % vocab_size


@pytest.fixture(scope="module")
def hf_model():
    return _build_dummy_model()


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


@pytest.fixture(scope="module")
def input_ids(tokenizer):
    return _encode_in_vocab(tokenizer, PROMPT, DUMMY_PARAMS["vocab_size"], PROMPT_LEN)


def test_stage1_hf_pytorch_forward(hf_model, input_ids):
    with torch.no_grad():
        out = hf_model(input_ids=input_ids)
    assert out.logits.shape == (1, PROMPT_LEN, DUMMY_PARAMS["vocab_size"])
    assert torch.isfinite(out.logits).all()


def test_stage2_qeff_transform_resolves_classes(hf_model):
    from QEfficient.transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        QEffDeepseekV4Attention,
        QEffDeepseekV4DecoderLayer,
        QEffDeepseekV4ForCausalLM,
        QEffDeepseekV4Model,
        QEffDeepseekV4SparseMoeBlock,
    )

    hf_copy = copy.deepcopy(hf_model)
    qeff = QEFFAutoModelForCausalLM(hf_copy, continuous_batching=False, pretrained_model_name_or_path=MODEL_NAME)
    qeff.transform(ctx_len=CTX_LEN, seq_len=PROMPT_LEN, batch_size=1, num_devices=1)

    assert isinstance(qeff.model, QEffDeepseekV4ForCausalLM)
    inner = qeff.model.model
    assert isinstance(inner, QEffDeepseekV4Model)
    for layer in inner.layers:
        assert isinstance(layer, QEffDeepseekV4DecoderLayer)
        assert isinstance(layer.self_attn, QEffDeepseekV4Attention)
        if hasattr(layer, "mlp") and not isinstance(layer.mlp, torch.nn.Identity):
            assert isinstance(layer.mlp, QEffDeepseekV4SparseMoeBlock)


def test_stage3_onnx_export(hf_model, tmp_path):
    hf_copy = copy.deepcopy(hf_model)
    qeff = QEFFAutoModelForCausalLM(hf_copy, continuous_batching=False, pretrained_model_name_or_path=MODEL_NAME)
    onnx_path = qeff.export(export_dir=str(tmp_path / "onnx"))
    assert Path(onnx_path).is_file()


@pytest.mark.on_qaic
def test_stage4_compile_and_generate(hf_model, tokenizer, tmp_path):
    hf_copy = copy.deepcopy(hf_model)
    qeff = QEFFAutoModelForCausalLM(hf_copy, continuous_batching=False, pretrained_model_name_or_path=MODEL_NAME)

    qpc_path = qeff.compile(
        prefill_seq_len=PROMPT_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=1,
        mxfp6=False,
        aic_enable_depth_first=False,
        compile_dir=str(tmp_path / "qpc"),
    )
    assert Path(qpc_path).is_dir()
    assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    exec_info = qeff.generate(tokenizer, prompts=[PROMPT])
    assert exec_info.generated_ids is not None
    assert exec_info.generated_ids[0].shape[-1] >= 1
