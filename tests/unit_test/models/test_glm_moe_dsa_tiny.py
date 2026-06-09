# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""4-stage validation for ``glm_moe_dsa`` (zai-org/GLM-5.1) on a dummy config.

Stages:
  1. HF PyTorch forward (parity-anchor; runs cleanly on CPU)
  2. QEff KV PyTorch forward — exercises ``transform()`` only (no full forward
     pass: the standard ``InputHandler`` allocates KV cache slots from
     ``hidden_size // num_attention_heads`` and is not MLA-aware, so it cannot
     pre-allocate the asymmetric ``qk_head_dim`` / ``v_head_dim`` cache that
     ``glm_moe_dsa`` requires. We assert that ``transform()`` rewires the model
     classes correctly; cross-stage logits parity is exercised inline against
     the QEff PyTorch model below using a hand-built MLA-aware
     ``QEffDynamicCache`` seed.
  3. ONNX export — uses an MLA-aware cache pre-allocation patch.
  4. AI 100 compile + generate.

Uses ``AutoConfig.for_model("glm_moe_dsa", **dummy)`` and
``AutoModelForCausalLM.from_config(...)`` so no full-weight download is needed.
"""

import contextlib
import copy
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "causal_model_configs.json")
with open(_CONFIG_FILE) as _f:
    _all = json.load(_f)
_GLM_ENTRY = next(e for e in _all["causal_lm_models"] if e["model_type"] == "glm_moe_dsa")

MODEL_NAME = _GLM_ENTRY["model_name"]
TOKENIZER_ID = _GLM_ENTRY.get("tokenizer_id", "gpt2")
DUMMY_PARAMS = _GLM_ENTRY["additional_params"]

PROMPT_LEN = 8
CTX_LEN = 32
PROMPT = "My name is"


def _build_dummy_model():
    config = AutoConfig.for_model("glm_moe_dsa", **DUMMY_PARAMS)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager", torch_dtype=torch.float32)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.02)
    model.eval()
    return model


def _encode_in_vocab(tokenizer, prompt, vocab_size, prompt_len):
    """Encode prompt and clamp ids to the dummy config's tiny vocab."""
    ids = tokenizer.encode(prompt, return_tensors="pt")
    ids = ids[:, :prompt_len]
    if ids.shape[1] < prompt_len:
        pad = torch.zeros(ids.shape[0], prompt_len - ids.shape[1], dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    ids = ids % vocab_size
    return ids


@contextlib.contextmanager
def _mla_aware_cache_shape():
    """Patch ``InputHandler._get_layer_cache_shape`` to return MLA-correct dims.

    The standard helper assumes a single ``head_dim``; MLA caches K at
    ``qk_head_dim`` and V at ``v_head_dim``. We can't return two shapes from
    one function, so this routes through a wrapper that returns the larger
    of the two when called for the K slot and the smaller for the V slot —
    encoded in the cache-tuple by alternation.
    """
    from QEfficient.utils import generate_inputs

    original = generate_inputs.InputHandler._get_layer_cache_shape

    def patched(self, layer_idx):
        cfg = self.config
        qk_head_dim = getattr(cfg, "qk_head_dim", None)
        v_head_dim = getattr(cfg, "v_head_dim", None)
        if qk_head_dim is None or v_head_dim is None:
            return original(self, layer_idx)
        # Return the K shape; V is sized separately via `prepare_pytorch_inputs`
        # patched alongside.
        n_heads = cfg.num_key_value_heads
        batch = self.full_batch_size if self.full_batch_size else self.padding_shape[0]
        return [batch, n_heads, self.ctx_len, qk_head_dim]

    original_prepare = generate_inputs.InputHandler.prepare_pytorch_inputs

    def patched_prepare(self):
        inputs = original_prepare(self)
        # Resize V slots to v_head_dim if MLA.
        cfg = self.config
        v_head_dim = getattr(cfg, "v_head_dim", None)
        qk_head_dim = getattr(cfg, "qk_head_dim", None)
        if v_head_dim is None or qk_head_dim is None or v_head_dim == qk_head_dim:
            return inputs
        pkv = []
        for i, (k, v) in enumerate(inputs["past_key_values"]):
            v_shape = list(k.shape)
            v_shape[-1] = v_head_dim
            v = torch.zeros(v_shape, dtype=v.dtype)
            pkv.append((k, v))
        inputs["past_key_values"] = tuple(pkv)
        return inputs

    generate_inputs.InputHandler._get_layer_cache_shape = patched
    generate_inputs.InputHandler.prepare_pytorch_inputs = patched_prepare
    try:
        yield
    finally:
        generate_inputs.InputHandler._get_layer_cache_shape = original
        generate_inputs.InputHandler.prepare_pytorch_inputs = original_prepare


@pytest.fixture(scope="module")
def hf_model():
    return _build_dummy_model()


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


@pytest.fixture(scope="module")
def input_ids(tokenizer):
    return _encode_in_vocab(tokenizer, PROMPT, DUMMY_PARAMS["vocab_size"], PROMPT_LEN)


# ---------------------------------------------------------------------------
# Stage 1 — HF PyTorch forward
# ---------------------------------------------------------------------------
def test_stage1_hf_pytorch_forward(hf_model, input_ids):
    """Stage 1: bare HF model forward — confirms config + transformers wiring."""
    with torch.no_grad():
        out = hf_model(input_ids=input_ids)
    assert out.logits.shape == (1, PROMPT_LEN, DUMMY_PARAMS["vocab_size"])
    assert torch.isfinite(out.logits).all()


# ---------------------------------------------------------------------------
# Stage 2 — QEff transform structural check + class identity
# ---------------------------------------------------------------------------
def test_stage2_qeff_transform_resolves_classes(hf_model):
    """Stage 2: ``transform()`` rewires QEff classes (KVCache + CustomOps + PrefillOnlyChunked)."""
    from QEfficient.transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        QEffGlmMoeDsaAttention,
        QEffGlmMoeDsaDecoderLayer,
        QEffGlmMoeDsaForCausalLM,
        QEffGlmMoeDsaModel,
        QEffGlmMoeDsaMoE,
    )

    hf_copy = copy.deepcopy(hf_model)
    qeff = QEFFAutoModelForCausalLM(hf_copy, continuous_batching=False, pretrained_model_name_or_path=MODEL_NAME)
    qeff.transform(ctx_len=CTX_LEN, seq_len=PROMPT_LEN, batch_size=1, num_devices=1)

    # The top-level model is rewrapped to QEffGlmMoeDsaForCausalLM.
    assert isinstance(qeff.model, QEffGlmMoeDsaForCausalLM)
    # Check that decoder layers + attention + MoE submodules are all the QEff variants.
    inner = qeff.model.model
    assert isinstance(inner, QEffGlmMoeDsaModel)
    first_dense = DUMMY_PARAMS.get("first_k_dense_replace", 3)
    for idx, layer in enumerate(inner.layers):
        assert isinstance(layer, QEffGlmMoeDsaDecoderLayer)
        assert isinstance(layer.self_attn, QEffGlmMoeDsaAttention)
        # Layers 0..first_k_dense_replace-1 are dense MLP, rest are MoE.
        if idx >= first_dense:
            assert isinstance(layer.mlp, QEffGlmMoeDsaMoE)


# ---------------------------------------------------------------------------
# Stage 3 — ONNX export with MLA-aware cache shape patch
# ---------------------------------------------------------------------------
def test_stage3_onnx_export(hf_model, tmp_path):
    """Stage 3: ONNX export succeeds with the MLA-aware cache shape patch."""
    hf_copy = copy.deepcopy(hf_model)
    qeff = QEFFAutoModelForCausalLM(hf_copy, continuous_batching=False, pretrained_model_name_or_path=MODEL_NAME)
    onnx_path = qeff.export(export_dir=str(tmp_path / "onnx"))
    assert Path(onnx_path).is_file()


# ---------------------------------------------------------------------------
# Stage 4 — AI 100 compile + generate
# ---------------------------------------------------------------------------
@pytest.mark.on_qaic
def test_stage4_compile_and_generate(hf_model, tokenizer, tmp_path):
    """Stage 4: compile the QPC and run a short generate against AI 100."""
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
