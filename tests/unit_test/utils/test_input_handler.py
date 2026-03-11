# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Tests for InputHandler: prepare_pytorch_inputs, update_pytorch_inputs,
prepare_ort_inputs, update_ort_inputs, update_ort_outputs.

All tests run on CPU only. Tests that require a tokenizer download are
automatically skipped if the network is unavailable.
"""

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from QEfficient.utils.generate_inputs import InputHandler

CTX_LEN = 32
VOCAB_SIZE = 500


def _get_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        return tok
    except Exception:
        pytest.skip("Cannot load gpt2 tokenizer (network unavailable)")


def _make_tiny_gpt2_config(tokenizer):
    return GPT2Config(
        n_layer=2, n_head=2, n_embd=64,
        vocab_size=tokenizer.vocab_size, n_positions=CTX_LEN, n_ctx=CTX_LEN,
    )


def _make_handler(tokenizer, config, prompt=None, prompt_len=8, ctx_len=CTX_LEN):
    if prompt is None:
        prompt = ["Hello world"]
    return InputHandler(
        batch_size=1, tokenizer=tokenizer, config=config,
        prompt=prompt, prompt_len=prompt_len, ctx_len=ctx_len, full_batch_size=None,
    )


class TestInputHandlerConstruction:
    def test_construction_succeeds(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler = _make_handler(tok, cfg)
        assert handler is not None

    def test_construction_with_multiple_prompts(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler = InputHandler(
            batch_size=2, tokenizer=tok, config=cfg,
            prompt=["Hello world", "The capital of France"],
            prompt_len=8, ctx_len=CTX_LEN, full_batch_size=None,
        )
        assert handler is not None

    def test_construction_with_longer_ctx_len(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler = _make_handler(tok, cfg, ctx_len=64)
        assert handler is not None


class TestPreparePytorchInputs:
    def test_returns_dict(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        inputs = _make_handler(tok, cfg).prepare_pytorch_inputs()
        assert hasattr(inputs, "__getitem__") and hasattr(inputs, "keys")

    def test_has_input_ids(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        inputs = _make_handler(tok, cfg, prompt_len=8).prepare_pytorch_inputs()
        assert "input_ids" in inputs

    def test_has_position_ids(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        inputs = _make_handler(tok, cfg, prompt_len=8).prepare_pytorch_inputs()
        assert "position_ids" in inputs

    def test_has_past_key_values(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        inputs = _make_handler(tok, cfg, prompt_len=8).prepare_pytorch_inputs()
        assert "past_key_values" in inputs

    def test_input_ids_shape(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        prompt_len = 8
        inputs = _make_handler(tok, cfg, prompt_len=prompt_len).prepare_pytorch_inputs()
        assert inputs["input_ids"].shape[0] == 1
        assert inputs["input_ids"].shape[1] == prompt_len

    def test_position_ids_shape(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        prompt_len = 8
        inputs = _make_handler(tok, cfg, prompt_len=prompt_len).prepare_pytorch_inputs()
        assert inputs["position_ids"].shape == (1, prompt_len)

    def test_position_ids_are_sequential(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        inputs = _make_handler(tok, cfg, prompt_len=8).prepare_pytorch_inputs()
        pos = inputs["position_ids"].squeeze()
        valid_pos = pos[pos >= 0]
        assert len(valid_pos) > 0
        if len(valid_pos) > 1:
            diffs = valid_pos[1:] - valid_pos[:-1]
            assert (diffs > 0).all(), f"Position IDs are not strictly increasing: {valid_pos}"

    def test_past_key_values_has_correct_number_of_layers(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        inputs = _make_handler(tok, cfg).prepare_pytorch_inputs()
        assert len(inputs["past_key_values"]) == cfg.n_layer

    def test_past_key_values_are_zero_initialized(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        inputs = _make_handler(tok, cfg).prepare_pytorch_inputs()
        for layer_idx, (k, v) in enumerate(inputs["past_key_values"]):
            assert torch.all(k == 0), f"Layer {layer_idx} key cache is not zero-initialized"
            assert torch.all(v == 0), f"Layer {layer_idx} value cache is not zero-initialized"

    def test_past_key_values_ctx_len_dimension(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        inputs = _make_handler(tok, cfg, ctx_len=CTX_LEN).prepare_pytorch_inputs()
        for layer_idx, (k, v) in enumerate(inputs["past_key_values"]):
            assert k.shape[2] == CTX_LEN, f"Layer {layer_idx} key cache ctx_len={k.shape[2]}, expected {CTX_LEN}"
            assert v.shape[2] == CTX_LEN, f"Layer {layer_idx} value cache ctx_len={v.shape[2]}, expected {CTX_LEN}"

    def test_input_ids_are_valid_token_ids(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        inputs = _make_handler(tok, cfg).prepare_pytorch_inputs()
        ids = inputs["input_ids"]
        assert (ids >= 0).all(), "Negative token IDs found"
        assert (ids < tok.vocab_size).all(), "Token IDs exceed vocab_size"


class TestUpdatePytorchInputs:
    def _run_prefill(self, tok, cfg, prompt_len=8):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
        model = GPT2LMHeadModel(cfg).eval()
        qeff_model = QEFFAutoModelForCausalLM(model)
        handler = _make_handler(tok, cfg, prompt_len=prompt_len)
        inputs = handler.prepare_pytorch_inputs()
        with torch.no_grad():
            outputs = qeff_model.model(**inputs)
        return handler, inputs, outputs

    def test_update_returns_dict(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler, inputs, outputs = self._run_prefill(tok, cfg)
        updated = handler.update_pytorch_inputs(inputs, outputs)
        assert hasattr(updated, "__getitem__") and hasattr(updated, "keys")

    def test_update_has_input_ids(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler, inputs, outputs = self._run_prefill(tok, cfg)
        updated = handler.update_pytorch_inputs(inputs, outputs)
        assert "input_ids" in updated

    def test_update_has_position_ids(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler, inputs, outputs = self._run_prefill(tok, cfg)
        updated = handler.update_pytorch_inputs(inputs, outputs)
        assert "position_ids" in updated

    def test_update_input_ids_is_single_token(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler, inputs, outputs = self._run_prefill(tok, cfg)
        updated = handler.update_pytorch_inputs(inputs, outputs)
        assert updated["input_ids"].shape == (1, 1), (
            f"Decode input_ids must be shape (1,1), got {updated['input_ids'].shape}"
        )

    def test_update_position_ids_advances(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        prompt_len = 8
        handler, inputs, outputs = self._run_prefill(tok, cfg, prompt_len=prompt_len)
        updated = handler.update_pytorch_inputs(inputs, outputs)
        decode_pos = updated["position_ids"].item()
        prefill_last_valid = inputs["position_ids"][inputs["position_ids"] >= 0].max().item()
        assert decode_pos > prefill_last_valid, (
            f"Decode position {decode_pos} must be > last prefill position {prefill_last_valid}"
        )

    def test_update_next_token_is_valid(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler, inputs, outputs = self._run_prefill(tok, cfg)
        updated = handler.update_pytorch_inputs(inputs, outputs)
        next_token = updated["input_ids"].item()
        assert 0 <= next_token < tok.vocab_size, (
            f"Next token {next_token} is not a valid token ID (vocab_size={tok.vocab_size})"
        )


class TestPrepareOrtInputs:
    def test_returns_dict_like(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        ort_inputs = _make_handler(tok, cfg).prepare_ort_inputs()
        assert hasattr(ort_inputs, "__getitem__") and hasattr(ort_inputs, "keys")

    def test_has_input_ids(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        ort_inputs = dict(_make_handler(tok, cfg).prepare_ort_inputs())
        assert "input_ids" in ort_inputs

    def test_has_position_ids(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        ort_inputs = dict(_make_handler(tok, cfg).prepare_ort_inputs())
        assert "position_ids" in ort_inputs

    def test_has_past_key_value_inputs(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        ort_inputs = dict(_make_handler(tok, cfg).prepare_ort_inputs())
        has_past = any("past_key" in k or "past_value" in k for k in ort_inputs.keys())
        assert has_past, f"No past_key/past_value inputs found: {list(ort_inputs.keys())}"

    def test_input_ids_are_numpy_int64(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        ort_inputs = dict(_make_handler(tok, cfg).prepare_ort_inputs())
        ids = ort_inputs["input_ids"]
        assert isinstance(ids, np.ndarray), f"input_ids must be numpy array, got {type(ids)}"
        assert ids.dtype == np.int64, f"input_ids must be int64, got {ids.dtype}"

    def test_position_ids_are_numpy_int64(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        ort_inputs = dict(_make_handler(tok, cfg).prepare_ort_inputs())
        pos = ort_inputs["position_ids"]
        assert isinstance(pos, np.ndarray)
        assert pos.dtype == np.int64

    def test_past_key_values_are_numpy_float32(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        ort_inputs = dict(_make_handler(tok, cfg).prepare_ort_inputs())
        for key, val in ort_inputs.items():
            if "past_key" in key or "past_value" in key:
                assert isinstance(val, np.ndarray)
                assert val.dtype == np.float32, f"{key} must be float32, got {val.dtype}"

    def test_past_key_values_are_zero_initialized(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        ort_inputs = dict(_make_handler(tok, cfg).prepare_ort_inputs())
        for key, val in ort_inputs.items():
            if "past_key" in key or "past_value" in key:
                assert np.all(val == 0), f"{key} must be zero-initialized for prefill"

    def test_past_key_values_ctx_len_dimension(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        ort_inputs = dict(_make_handler(tok, cfg, ctx_len=CTX_LEN).prepare_ort_inputs())
        for key, val in ort_inputs.items():
            if "past_key" in key or "past_value" in key:
                assert val.shape[2] == CTX_LEN, f"{key} ctx_len={val.shape[2]}, expected {CTX_LEN}"

    def test_correct_number_of_kv_cache_inputs(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        ort_inputs = dict(_make_handler(tok, cfg).prepare_ort_inputs())
        past_keys = [k for k in ort_inputs if "past_key" in k]
        past_values = [k for k in ort_inputs if "past_value" in k]
        assert len(past_keys) == cfg.n_layer
        assert len(past_values) == cfg.n_layer

    def test_pytorch_and_ort_inputs_have_same_keys(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler = _make_handler(tok, cfg)
        pt_inputs = handler.prepare_pytorch_inputs()
        ort_inputs = dict(handler.prepare_ort_inputs())
        assert "input_ids" in pt_inputs and "input_ids" in ort_inputs
        assert "position_ids" in pt_inputs and "position_ids" in ort_inputs


class TestUpdateOrtInputsOutputs:
    def _make_fake_ort_outputs(self, cfg, prompt_len=8):
        n_layers = cfg.n_layer
        n_heads = cfg.n_head
        head_dim = cfg.n_embd // n_heads
        outputs = {
            "logits": np.random.randn(1, prompt_len, cfg.vocab_size).astype(np.float32),
        }
        for i in range(n_layers):
            outputs[f"past_key.{i}_RetainedState"] = np.zeros(
                (1, n_heads, CTX_LEN, head_dim), dtype=np.float32
            )
            outputs[f"past_value.{i}_RetainedState"] = np.zeros(
                (1, n_heads, CTX_LEN, head_dim), dtype=np.float32
            )
        return outputs

    def test_update_ort_outputs_returns_dict(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler = _make_handler(tok, cfg)
        result = handler.update_ort_outputs(self._make_fake_ort_outputs(cfg))
        assert hasattr(result, "__getitem__") and hasattr(result, "keys")

    def test_update_ort_outputs_has_logits(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler = _make_handler(tok, cfg)
        result = handler.update_ort_outputs(self._make_fake_ort_outputs(cfg))
        assert "logits" in result

    def test_update_ort_inputs_returns_dict(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler = _make_handler(tok, cfg)
        ort_inputs = dict(handler.prepare_ort_inputs())
        processed = handler.update_ort_outputs(self._make_fake_ort_outputs(cfg))
        updated = handler.update_ort_inputs(ort_inputs, processed)
        assert hasattr(updated, "__getitem__") and hasattr(updated, "keys")

    def test_update_ort_inputs_has_input_ids(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler = _make_handler(tok, cfg)
        ort_inputs = dict(handler.prepare_ort_inputs())
        processed = handler.update_ort_outputs(self._make_fake_ort_outputs(cfg))
        updated = handler.update_ort_inputs(ort_inputs, processed)
        assert "input_ids" in updated

    def test_update_ort_inputs_has_position_ids(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler = _make_handler(tok, cfg)
        ort_inputs = dict(handler.prepare_ort_inputs())
        processed = handler.update_ort_outputs(self._make_fake_ort_outputs(cfg))
        updated = handler.update_ort_inputs(ort_inputs, processed)
        assert "position_ids" in updated

    def test_update_ort_inputs_input_ids_batch_size_is_1(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler = _make_handler(tok, cfg, prompt_len=8)
        ort_inputs = dict(handler.prepare_ort_inputs())
        processed = handler.update_ort_outputs(self._make_fake_ort_outputs(cfg, prompt_len=8))
        updated = handler.update_ort_inputs(ort_inputs, processed)
        assert updated["input_ids"].shape[0] == 1
        assert isinstance(updated["input_ids"], np.ndarray)

    def test_update_ort_inputs_position_ids_advances(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        prompt_len = 8
        handler = _make_handler(tok, cfg, prompt_len=prompt_len)
        ort_inputs = dict(handler.prepare_ort_inputs())
        processed = handler.update_ort_outputs(self._make_fake_ort_outputs(cfg, prompt_len=prompt_len))
        updated = handler.update_ort_inputs(ort_inputs, processed)
        decode_pos = updated["position_ids"].flatten()[0]
        prefill_last_valid = ort_inputs["position_ids"][ort_inputs["position_ids"] >= 0].max()
        assert decode_pos > prefill_last_valid, (
            f"Decode position {decode_pos} must be > last prefill position {prefill_last_valid}"
        )

    def test_update_ort_inputs_are_numpy_arrays(self):
        tok = _get_tokenizer()
        cfg = _make_tiny_gpt2_config(tok)
        handler = _make_handler(tok, cfg)
        ort_inputs = dict(handler.prepare_ort_inputs())
        processed = handler.update_ort_outputs(self._make_fake_ort_outputs(cfg))
        updated = handler.update_ort_inputs(ort_inputs, processed)
        for key, val in updated.items():
            assert isinstance(val, np.ndarray), f"ORT input '{key}' must be numpy array, got {type(val)}"
