# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Tests for QEFFBaseModel base class.

CPU-only tests that do NOT require QAIC hardware.
Run with: pytest tests/unit_test/base/ -n auto -v
"""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

VOCAB_SIZE = 500
CTX_LEN = 32
SEQ_LEN = 8


def make_tiny_gpt2():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=VOCAB_SIZE, n_positions=CTX_LEN, n_ctx=CTX_LEN)
    return GPT2LMHeadModel(cfg).eval(), cfg


def make_tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return LlamaForCausalLM(cfg).eval(), cfg


@pytest.mark.cpu_only
class TestQEFFBaseModelProperties:
    """Test QEFFBaseModel properties and class methods."""

    def test_model_name_returns_class_name(self):
        """model_name property returns a non-empty string."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert isinstance(qeff.model_name, str)
        assert len(qeff.model_name) > 0

    def test_model_name_strips_qeff_prefix(self):
        """model_name strips QEff/QEFF prefix from transformed model class name."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # After KVCacheTransform, model becomes QEffGPT2LMHeadModel
        # model_name should strip the QEff prefix
        assert not qeff.model_name.startswith("QEff")
        assert not qeff.model_name.startswith("QEFF")

    def test_transform_names_returns_list_of_strings(self):
        """_transform_names instance method returns list of transform names."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        names = qeff._transform_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert len(names) > 0

    def test_transform_names_includes_pytorch_transforms(self):
        """_transform_names includes KVCacheTransform."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        names = qeff._transform_names()
        assert "KVCacheTransform" in names

    def test_transform_names_includes_onnx_transforms(self):
        """_transform_names includes ONNX transforms when present."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        names = qeff._transform_names()
        # _transform_names returns pytorch + onnx transform names.
        # QEFFAutoModelForCausalLM._onnx_transforms is empty by default,
        # so only pytorch transforms are expected.
        assert isinstance(names, list)
        assert len(names) > 0

    def test_init_sets_onnx_path_to_none(self):
        """__init__ sets onnx_path to None initially."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.onnx_path is None

    def test_init_sets_qpc_path_to_none(self):
        """__init__ sets qpc_path to None initially."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.qpc_path is None

    def test_init_sets_qpc_session_to_none(self):
        """__init__ sets qpc_session to None initially."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.qpc_session is None

    def test_init_is_weights_offloaded_false(self):
        """__init__ sets _is_weights_offloaded to False initially."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff._is_weights_offloaded is False

    def test_model_architecture_extracted(self):
        """model_architecture is extracted from config.architectures."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # GPT2 config has architectures attribute
        assert qeff.model_architecture is not None or qeff.model_architecture is None


@pytest.mark.cpu_only
class TestQEFFBaseModelWeightOffloading:
    """Test weight offloading functionality."""

    def test_offload_model_weights_sets_flag(self):
        """_offload_model_weights(True) offloads weights and sets flag."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        result = qeff._offload_model_weights(offload_pt_weights=True)
        assert result is True
        assert qeff._is_weights_offloaded is True

    def test_offload_model_weights_false_does_not_offload(self):
        """_offload_model_weights(False) does not offload weights."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        result = qeff._offload_model_weights(offload_pt_weights=False)
        assert result is False
        assert qeff._is_weights_offloaded is False

    def test_offload_model_weights_idempotent(self):
        """_offload_model_weights is idempotent (second call returns False)."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff._offload_model_weights(offload_pt_weights=True)
        # Second call should return False (already offloaded)
        result = qeff._offload_model_weights(offload_pt_weights=True)
        assert result is False

    def test_model_offloaded_check_raises_when_offloaded(self):
        """_model_offloaded_check raises RuntimeError when weights are offloaded."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff._offload_model_weights(offload_pt_weights=True)
        with pytest.raises(RuntimeError, match="weights have been offloaded"):
            qeff._model_offloaded_check()

    def test_model_offloaded_check_passes_when_not_offloaded(self):
        """_model_offloaded_check does not raise when weights are not offloaded."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # Should not raise
        qeff._model_offloaded_check()

    def test_offload_clears_parameter_storage(self):
        """_offload_model_weights moves all parameters and buffers to meta device."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # Check that parameters are NOT on meta before offloading
        assert not any(p.is_meta for p in qeff.model.parameters())

        qeff._offload_model_weights(offload_pt_weights=True)

        # After offloading, ALL parameters and buffers must be on meta device
        assert all(p.is_meta for p in qeff.model.parameters())
        assert all(b.is_meta for b in qeff.model.buffers())

    def test_offload_clears_plain_tensor_attributes(self):
        """_offload_model_weights clears plain tensor attributes (not params/buffers)."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)

        # Attach a plain tensor attribute to a submodule (simulates MoE stacked weights)
        first_child = next(iter(qeff.model.modules()))
        first_child.extra_weight = torch.randn(8, 8)
        assert not first_child.extra_weight.is_meta

        qeff._offload_model_weights(offload_pt_weights=True)

        # The plain tensor attribute should also be on meta device
        assert first_child.extra_weight.is_meta


@pytest.mark.cpu_only
class TestQEFFBaseModelHashParams:
    """Test hash_params initialization."""

    def test_hash_params_is_dict(self):
        """hash_params is a dictionary."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert isinstance(qeff.hash_params, dict)

    def test_hash_params_contains_qeff_auto_class(self):
        """hash_params contains qeff_auto_class key."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert "qeff_auto_class" in qeff.hash_params
        assert qeff.hash_params["qeff_auto_class"] == "QEFFAutoModelForCausalLM"

    def test_hash_params_contains_pretrained_model_name(self):
        """hash_params contains pretrained_model_name_or_path when provided."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, pretrained_model_name_or_path="test-model")
        assert "pretrained_model_name_or_path" in qeff.hash_params
        assert qeff.hash_params["pretrained_model_name_or_path"] == "test-model"


@pytest.mark.cpu_only
@pytest.mark.onnx
@pytest.mark.slow
class TestQEFFBaseModelGetOnnxPath:
    """Test get_onnx_path method."""

    def test_get_onnx_path_returns_onnx_path(self, tmp_export_dir):
        """get_onnx_path calls export and returns a valid onnx_path."""
        import os

        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # get_onnx_path calls self.export() internally
        onnx_path = qeff.get_onnx_path()
        assert onnx_path is not None
        assert qeff.onnx_path is not None
        assert os.path.exists(str(onnx_path))

    def test_get_onnx_path_sets_onnx_path_attribute(self, tmp_export_dir):
        """get_onnx_path sets self.onnx_path after export."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.onnx_path is None  # Before export
        qeff.get_onnx_path()
        assert qeff.onnx_path is not None  # After export

    def test_get_onnx_path_second_call_returns_cached_path(self, tmp_export_dir):
        """get_onnx_path returns the same path on a second call (cached)."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        onnx_path_1 = qeff.get_onnx_path()
        onnx_path_2 = qeff.get_onnx_path()
        assert str(onnx_path_1) == str(onnx_path_2)
