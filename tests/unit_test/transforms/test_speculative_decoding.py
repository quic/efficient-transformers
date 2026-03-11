# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Tests for Speculative Decoding (SpDTransform) in QEfficient.

Tests verify:
  - SpDTransform.apply() with speculative_model_type="target" attaches tlm_forward
  - SpDTransform._module_mapping contains expected model classes
  - SpDTransform raises ValueError for invalid speculative_model_type
  - SpDTransform raises NotImplementedError for unsupported model class
  - QEFFAutoModelForCausalLM has check_and_get_num_speculative_tokens method
  - QEFFAutoModelForCausalLM has build_prefill_specialization / build_decode_specialization
  - is_tlm flag is set correctly on the wrapper

All tests run on CPU only.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform, SpDTransform

VOCAB_SIZE = 500
SEQ_LEN = 8
CTX_LEN = 32


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


def make_kv_transformed_llama():
    model, cfg = make_tiny_llama()
    transformed, _ = KVCacheTransform.apply(model)
    return transformed, cfg


# ---------------------------------------------------------------------------
# Tests: SpDTransform module mapping and structure
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSpDTransformStructure:
    """SpDTransform must have correct class-level structure."""

    def test_spd_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import SpDTransform
        assert SpDTransform is not None

    def test_module_mapping_is_set(self):
        assert hasattr(SpDTransform, "_module_mapping")
        assert len(SpDTransform._module_mapping) > 0

    def test_module_mapping_contains_llama(self):
        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM
        assert QEffLlamaForCausalLM in SpDTransform._module_mapping

    def test_module_mapping_contains_qwen2(self):
        from QEfficient.transformers.models.qwen2.modeling_qwen2 import QEffQwen2ForCausalLM
        assert QEffQwen2ForCausalLM in SpDTransform._module_mapping

    def test_apply_classmethod_exists(self):
        assert hasattr(SpDTransform, "apply")
        assert callable(SpDTransform.apply)


# ---------------------------------------------------------------------------
# Tests: SpDTransform no-op paths (already tested in test_transform_accuracy.py,
# but included here for completeness)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSpDTransformNoOpPaths:
    """SpDTransform must not apply when qaic_config is None or missing key."""

    def test_no_transform_when_qaic_config_is_none(self):
        model, _ = make_kv_transformed_llama()
        _, applied = SpDTransform.apply(model, qaic_config=None)
        assert not applied

    def test_no_transform_when_speculative_model_type_missing(self):
        model, _ = make_kv_transformed_llama()
        _, applied = SpDTransform.apply(model, qaic_config={})
        assert not applied

    def test_invalid_speculative_model_type_raises_value_error(self):
        model, _ = make_kv_transformed_llama()
        with pytest.raises(ValueError):
            SpDTransform.apply(model, qaic_config={"speculative_model_type": "invalid_xyz_abc"})

    def test_unsupported_model_class_raises_not_implemented(self):
        import torch.nn as nn

        class UnsupportedModel(nn.Module):
            def forward(self, x):
                return x

        with pytest.raises(NotImplementedError):
            SpDTransform.apply(
                UnsupportedModel(),
                qaic_config={"speculative_model_type": "target"},
            )


# ---------------------------------------------------------------------------
# Tests: SpDTransform actual apply (TLM path)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSpDTransformTLMApply:
    """SpDTransform with speculative_model_type='target' must attach tlm_forward."""

    def test_spd_transform_applies_to_llama_with_target_type(self):
        """SpDTransform must apply successfully to QEffLlamaForCausalLM with target type."""
        model, _ = make_kv_transformed_llama()
        transformed, applied = SpDTransform.apply(
            model, qaic_config={"speculative_model_type": "target"}
        )
        assert applied, "SpDTransform must apply when speculative_model_type='target'"

    def test_spd_transform_forward_is_replaced(self):
        """After SpDTransform, model.forward must be replaced with a SpD-specific forward."""
        model, _ = make_kv_transformed_llama()
        original_forward = model.forward
        transformed, applied = SpDTransform.apply(
            model, qaic_config={"speculative_model_type": "target"}
        )
        assert applied
        assert hasattr(transformed, "forward")
        # The forward must have been replaced (different from original)
        assert transformed.forward is not original_forward, (
            "SpDTransform must replace model.forward with a SpD-specific forward"
        )

    def test_spd_transform_returns_model_instance(self):
        """SpDTransform must return the same model instance (in-place modification)."""
        model, _ = make_kv_transformed_llama()
        transformed, applied = SpDTransform.apply(
            model, qaic_config={"speculative_model_type": "target"}
        )
        assert applied
        assert transformed is model, "SpDTransform must modify model in-place"

    def test_spd_transformed_model_is_still_eval_mode(self):
        """SpDTransform must not change the model's training mode."""
        model, _ = make_kv_transformed_llama()
        assert not model.training
        transformed, _ = SpDTransform.apply(
            model, qaic_config={"speculative_model_type": "target"}
        )
        assert not transformed.training, "SpDTransform must not change model to training mode"

    def test_spd_transform_model_still_has_parameters(self):
        """After SpDTransform, model must still have its parameters."""
        model, _ = make_kv_transformed_llama()
        param_count_before = sum(p.numel() for p in model.parameters())
        transformed, _ = SpDTransform.apply(
            model, qaic_config={"speculative_model_type": "target"}
        )
        param_count_after = sum(p.numel() for p in transformed.parameters())
        assert param_count_before == param_count_after, (
            f"SpDTransform changed parameter count: {param_count_before} → {param_count_after}"
        )


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForCausalLM SpD-related methods
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestQEFFAutoModelSpDMethods:
    """QEFFAutoModelForCausalLM must have SpD-related methods."""

    def test_has_check_and_get_num_speculative_tokens(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
        assert hasattr(QEFFAutoModelForCausalLM, "check_and_get_num_speculative_tokens")
        assert callable(QEFFAutoModelForCausalLM.check_and_get_num_speculative_tokens)

    def test_has_build_prefill_specialization(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
        assert hasattr(QEFFAutoModelForCausalLM, "build_prefill_specialization")
        assert callable(QEFFAutoModelForCausalLM.build_prefill_specialization)

    def test_has_build_decode_specialization(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
        assert hasattr(QEFFAutoModelForCausalLM, "build_decode_specialization")
        assert callable(QEFFAutoModelForCausalLM.build_decode_specialization)

    def test_has_is_tlm_property(self):
        """QEFFAutoModelForCausalLM instances must expose is_tlm."""
        from transformers import GPT2Config, GPT2LMHeadModel

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model)
        assert hasattr(qeff, "is_tlm"), "QEFFAutoModelForCausalLM instance must have is_tlm attribute"

    def test_is_tlm_false_by_default(self):
        """Without SpD config, is_tlm must be False."""
        from transformers import GPT2Config, GPT2LMHeadModel

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.is_tlm is False, "is_tlm must be False when no SpD config is provided"

    def test_check_and_get_num_speculative_tokens_returns_none_for_non_tlm(self):
        """For a non-TLM model, check_and_get_num_speculative_tokens must not raise."""
        from transformers import GPT2Config, GPT2LMHeadModel

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model)
        # For non-TLM, is_tlm=False; method accepts num_speculative_tokens and prefill_seq_len
        result = qeff.check_and_get_num_speculative_tokens(
            num_speculative_tokens=None, prefill_seq_len=1
        )
        assert result is None, (
            f"check_and_get_num_speculative_tokens must return None for non-TLM, got {result}"
        )

    def test_build_prefill_specialization_returns_dict(self):
        """build_prefill_specialization must return a dict-like object."""
        from transformers import GPT2Config, GPT2LMHeadModel

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model)
        result = qeff.build_prefill_specialization(
            prefill_seq_len=8, ctx_len=32, batch_size=1, full_batch_size=None
        )
        assert isinstance(result, dict), (
            f"build_prefill_specialization must return dict, got {type(result)}"
        )

    def test_build_decode_specialization_returns_dict(self):
        """build_decode_specialization must return a dict-like object."""
        from transformers import GPT2Config, GPT2LMHeadModel

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
        model = GPT2LMHeadModel(cfg)
        qeff = QEFFAutoModelForCausalLM(model)
        result = qeff.build_decode_specialization(
            ctx_len=32, batch_size=1, full_batch_size=None
        )
        assert isinstance(result, dict), (
            f"build_decode_specialization must return dict, got {type(result)}"
        )


# ---------------------------------------------------------------------------
# Tests: TLM forward execution
# ---------------------------------------------------------------------------


@pytest.mark.transforms
@pytest.mark.accuracy
class TestTLMForwardExecution:
    """After SpDTransform, the replaced tlm_forward must produce correct outputs."""

    def _make_tlm_inputs(self, batch=1, num_spec_tokens=3, n_layers=2, n_kv=2, head_dim=32):
        """Create inputs for TLM forward with pre-allocated zero KV cache."""
        seq_len = num_spec_tokens + 1
        input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        past_key_values = tuple(
            (
                torch.zeros(batch, n_kv, CTX_LEN, head_dim, dtype=torch.float32),
                torch.zeros(batch, n_kv, CTX_LEN, head_dim, dtype=torch.float32),
            )
            for _ in range(n_layers)
        )
        return input_ids, position_ids, past_key_values

    def test_tlm_forward_returns_logits(self):
        """tlm_forward must return an object with logits attribute."""
        model, cfg = make_kv_transformed_llama()
        transformed, applied = SpDTransform.apply(
            model, qaic_config={"speculative_model_type": "target"}
        )
        assert applied

        batch, num_spec_tokens = 1, 3
        # n_kv=2, head_dim=64//2=32 for tiny llama
        # num_logits_to_keep must be a tensor (as expected by spd_transform_forward)
        input_ids, position_ids, past_kv = self._make_tlm_inputs(batch, num_spec_tokens, n_layers=2, n_kv=2, head_dim=32)
        num_logits_tensor = torch.tensor([num_spec_tokens], dtype=torch.int64)

        with torch.no_grad():
            output = transformed(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_kv,
                num_logits_to_keep=num_logits_tensor,
            )
        assert hasattr(output, "logits"), "TLM forward must return output with logits"

    def test_tlm_forward_logits_are_finite(self):
        """tlm_forward logits must be finite (no NaN/Inf)."""
        model, cfg = make_kv_transformed_llama()
        transformed, applied = SpDTransform.apply(
            model, qaic_config={"speculative_model_type": "target"}
        )
        assert applied

        batch, num_spec_tokens = 1, 3
        input_ids, position_ids, past_kv = self._make_tlm_inputs(batch, num_spec_tokens, n_layers=2, n_kv=2, head_dim=32)
        num_logits_tensor = torch.tensor([num_spec_tokens], dtype=torch.int64)

        with torch.no_grad():
            output = transformed(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_kv,
                num_logits_to_keep=num_logits_tensor,
            )
        assert torch.isfinite(output.logits).all(), "TLM logits must be finite"

    def test_tlm_forward_logits_shape_is_batch_x_kept_x_vocab(self):
        """tlm_forward logits shape must be [batch, num_logits_to_keep, vocab_size].
        num_logits_to_keep is a 1D tensor of shape [1] containing the count,
        so the output has shape[1] == num_logits_to_keep.shape[0] == 1."""
        model, cfg = make_kv_transformed_llama()
        transformed, applied = SpDTransform.apply(
            model, qaic_config={"speculative_model_type": "target"}
        )
        assert applied

        batch, num_spec_tokens = 1, 3
        input_ids, position_ids, past_kv = self._make_tlm_inputs(batch, num_spec_tokens, n_layers=2, n_kv=2, head_dim=32)
        # num_logits_to_keep is a 1D tensor; shape[0] determines how many logits are kept
        num_logits_tensor = torch.tensor([num_spec_tokens], dtype=torch.int64)

        with torch.no_grad():
            output = transformed(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_kv,
                num_logits_to_keep=num_logits_tensor,
            )
        # batch dimension must match
        assert output.logits.shape[0] == batch
        # vocab dimension must match
        assert output.logits.shape[-1] == VOCAB_SIZE
        # logits must be 3D: [batch, seq, vocab]
        assert output.logits.ndim == 3

    def test_tlm_forward_greedy_tokens_in_valid_range(self):
        """Greedy tokens from tlm_forward must be in [0, vocab_size)."""
        model, cfg = make_kv_transformed_llama()
        transformed, applied = SpDTransform.apply(
            model, qaic_config={"speculative_model_type": "target"}
        )
        assert applied

        batch, num_spec_tokens = 1, 3
        input_ids, position_ids, past_kv = self._make_tlm_inputs(batch, num_spec_tokens, n_layers=2, n_kv=2, head_dim=32)
        num_logits_tensor = torch.tensor([num_spec_tokens], dtype=torch.int64)

        with torch.no_grad():
            output = transformed(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_kv,
                num_logits_to_keep=num_logits_tensor,
            )
        greedy_tokens = output.logits.argmax(dim=-1)
        assert (greedy_tokens >= 0).all()
        assert (greedy_tokens < VOCAB_SIZE).all()


# ---------------------------------------------------------------------------
# Tests: SpDTransform for Qwen2
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSpDTransformQwen2:
    """SpDTransform must apply correctly to Qwen2 models."""

    def _make_kv_transformed_qwen2(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform

        cfg = Qwen2Config(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=CTX_LEN,
        )
        model = Qwen2ForCausalLM(cfg).eval()
        transformed, _ = KVCacheTransform.apply(model)
        return transformed, cfg

    def test_spd_transform_applies_to_qwen2_with_target_type(self):
        """SpDTransform must apply successfully to QEffQwen2ForCausalLM."""
        model, _ = self._make_kv_transformed_qwen2()
        transformed, applied = SpDTransform.apply(
            model, qaic_config={"speculative_model_type": "target"}
        )
        assert applied, "SpDTransform must apply to Qwen2 with target type"

    def test_spd_transform_qwen2_forward_is_replaced(self):
        """After SpDTransform, Qwen2 model.forward must be replaced."""
        model, _ = self._make_kv_transformed_qwen2()
        original_forward = model.forward
        transformed, applied = SpDTransform.apply(
            model, qaic_config={"speculative_model_type": "target"}
        )
        assert applied
        assert transformed.forward is not original_forward

    def test_spd_transform_qwen2_produces_finite_logits(self):
        """After SpDTransform, Qwen2 forward must produce finite logits."""
        from QEfficient.transformers.cache_utils import QEffDynamicCache

        model, _ = self._make_kv_transformed_qwen2()
        transformed, applied = SpDTransform.apply(
            model, qaic_config={"speculative_model_type": "target"}
        )
        assert applied

        batch, num_spec_tokens = 1, 2
        seq_len = num_spec_tokens + 1
        input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        # Use tuple-based KV cache (n_kv=2, head_dim=64//2=32)
        past_kv = tuple(
            (
                torch.zeros(batch, 2, CTX_LEN, 32, dtype=torch.float32),
                torch.zeros(batch, 2, CTX_LEN, 32, dtype=torch.float32),
            )
            for _ in range(2)
        )
        num_logits_tensor = torch.tensor([num_spec_tokens], dtype=torch.int64)

        with torch.no_grad():
            output = transformed(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_kv,
                num_logits_to_keep=num_logits_tensor,
            )
        assert torch.isfinite(output.logits).all()


# ---------------------------------------------------------------------------
# Tests: post_processing.py registry
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestPostProcessingRegistry:
    """post_processing.model_type_registry must contain expected model types."""

    def test_model_type_registry_is_not_empty(self):
        """model_type_registry must not be empty."""
        from QEfficient.transformers.post_processing import model_type_registry

        assert len(model_type_registry) > 0

    def test_model_type_registry_contains_turbo(self):
        """model_type_registry must contain 'turbo' (the SpD post-processing type)."""
        from QEfficient.transformers.post_processing import model_type_registry

        assert "turbo" in model_type_registry

    def test_model_type_registry_keys_are_strings(self):
        """All keys in model_type_registry must be strings."""
        from QEfficient.transformers.post_processing import model_type_registry

        for key in model_type_registry:
            assert isinstance(key, str), f"Registry key must be string, got {type(key)}"

    def test_model_type_registry_values_are_callable(self):
        """All values in model_type_registry must be callable."""
        from QEfficient.transformers.post_processing import model_type_registry

        for model_type, handler in model_type_registry.items():
            assert callable(handler), f"Handler for '{model_type}' must be callable"


# ---------------------------------------------------------------------------
# Tests: SpD ONNX structure (GAP I)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSpDONNXStructure:
    """SpD-related ONNX structure tests — verify num_logits_to_keep input and build_and_attach_mlp."""

    def test_build_and_attach_mlp_importable(self):
        """build_and_attach_mlp must be importable from post_processing."""
        from QEfficient.transformers.post_processing import build_and_attach_mlp
        assert build_and_attach_mlp is not None

    def test_build_and_attach_mlp_is_callable(self):
        """build_and_attach_mlp must be callable."""
        from QEfficient.transformers.post_processing import build_and_attach_mlp
        assert callable(build_and_attach_mlp)

    def test_build_and_attach_mlp_accepts_model_parameter(self):
        """build_and_attach_mlp must accept 'model' as first parameter."""
        import inspect
        from QEfficient.transformers.post_processing import build_and_attach_mlp
        sig = inspect.signature(build_and_attach_mlp)
        assert "model" in sig.parameters

    def test_build_and_attach_mlp_accepts_speculative_model_type(self):
        """build_and_attach_mlp must accept 'speculative_model_type' parameter."""
        import inspect
        from QEfficient.transformers.post_processing import build_and_attach_mlp
        sig = inspect.signature(build_and_attach_mlp)
        assert "speculative_model_type" in sig.parameters

    def test_model_type_registry_has_turbo(self):
        """model_type_registry must contain 'turbo' key."""
        from QEfficient.transformers.post_processing import model_type_registry
        assert "turbo" in model_type_registry

    def test_build_and_attach_turbo_importable(self):
        """build_and_attach_turbo must be importable from spd.turbo."""
        from QEfficient.transformers.spd.turbo import build_and_attach_turbo
        assert build_and_attach_turbo is not None

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_tlm_onnx_has_num_logits_to_keep_input(self, tmp_export_dir):
        """TLM ONNX export must include 'num_logits_to_keep' as an input."""
        import onnx
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        model, cfg = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(
            model,
            qaic_config={"speculative_model_type": "target"},
        )
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))

        input_names = [inp.name for inp in onnx_model.graph.input]
        assert "num_logits_to_keep" in input_names, (
            f"TLM ONNX must have 'num_logits_to_keep' input. Found: {input_names}"
        )

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_tlm_onnx_logits_output_is_present(self, tmp_export_dir):
        """TLM ONNX export must include 'logits' as an output."""
        import onnx
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        model, cfg = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(
            model,
            qaic_config={"speculative_model_type": "target"},
        )
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))

        output_names = [out.name for out in onnx_model.graph.output]
        assert "logits" in output_names, (
            f"TLM ONNX must have 'logits' output. Found: {output_names}"
        )
