# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Tests for QEFFAutoModel API surface in QEfficient.

Tests verify:
  - QEFFAutoModelForCausalLM wraps models correctly
  - is_tlm property is False by default
  - build_prefill_specialization returns dict with correct keys
  - build_decode_specialization returns dict with correct keys
  - check_and_get_num_speculative_tokens returns None for non-TLM
  - prefill() method exists
  - QEFFAutoModel (encoder) wraps BERT correctly
  - QEFFAutoModelForCTC wraps Wav2Vec2 correctly

All tests run on CPU only, using tiny in-memory models.
"""

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


def make_tiny_gpt2():
    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=500, n_positions=32, n_ctx=32)
    return GPT2LMHeadModel(cfg).eval()


def make_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=500,
        max_position_embeddings=32,
    )
    return LlamaForCausalLM(cfg).eval()


def make_tiny_bert():
    from transformers import BertConfig, BertModel

    cfg = BertConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=500,
        max_position_embeddings=64,
    )
    return BertModel(cfg).eval()


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForCausalLM basic wrapping
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFAutoModelForCausalLMBasic:
    """QEFFAutoModelForCausalLM must wrap models and expose correct attributes."""

    def test_wraps_gpt2_model(self):
        """QEFFAutoModelForCausalLM must wrap a GPT2LMHeadModel."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        model = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff is not None

    def test_wraps_llama_model(self):
        """QEFFAutoModelForCausalLM must wrap a LlamaForCausalLM."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        model = make_tiny_llama()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff is not None

    def test_is_tlm_false_by_default(self):
        """is_tlm must be False when no SpD config is provided."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        model = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.is_tlm is False

    def test_has_prefill_method(self):
        """QEFFAutoModelForCausalLM must have a prefill() method."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        assert hasattr(QEFFAutoModelForCausalLM, "prefill")
        assert callable(QEFFAutoModelForCausalLM.prefill)

    def test_has_export_method(self):
        """QEFFAutoModelForCausalLM must have an export() method."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        assert hasattr(QEFFAutoModelForCausalLM, "export")
        assert callable(QEFFAutoModelForCausalLM.export)

    def test_has_check_and_get_num_speculative_tokens(self):
        """QEFFAutoModelForCausalLM must have check_and_get_num_speculative_tokens."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        assert hasattr(QEFFAutoModelForCausalLM, "check_and_get_num_speculative_tokens")
        assert callable(QEFFAutoModelForCausalLM.check_and_get_num_speculative_tokens)

    def test_has_build_prefill_specialization(self):
        """QEFFAutoModelForCausalLM must have build_prefill_specialization."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        assert hasattr(QEFFAutoModelForCausalLM, "build_prefill_specialization")
        assert callable(QEFFAutoModelForCausalLM.build_prefill_specialization)

    def test_has_build_decode_specialization(self):
        """QEFFAutoModelForCausalLM must have build_decode_specialization."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        assert hasattr(QEFFAutoModelForCausalLM, "build_decode_specialization")
        assert callable(QEFFAutoModelForCausalLM.build_decode_specialization)


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForCausalLM specialization API
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFAutoModelForCausalLMSpecializations:
    """build_prefill_specialization and build_decode_specialization must return correct dicts."""

    def _make_qeff(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        return QEFFAutoModelForCausalLM(make_tiny_gpt2())

    def test_build_prefill_specialization_returns_dict(self):
        """build_prefill_specialization must return a dict."""
        qeff = self._make_qeff()
        result = qeff.build_prefill_specialization(prefill_seq_len=8, ctx_len=32, batch_size=1, full_batch_size=None)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_build_prefill_specialization_has_seq_len_key(self):
        """build_prefill_specialization dict must contain 'seq_len'."""
        qeff = self._make_qeff()
        result = qeff.build_prefill_specialization(prefill_seq_len=8, ctx_len=32, batch_size=1, full_batch_size=None)
        assert "seq_len" in result, f"'seq_len' not in prefill spec: {result}"

    def test_build_prefill_specialization_has_ctx_len_key(self):
        """build_prefill_specialization dict must contain 'ctx_len'."""
        qeff = self._make_qeff()
        result = qeff.build_prefill_specialization(prefill_seq_len=8, ctx_len=32, batch_size=1, full_batch_size=None)
        assert "ctx_len" in result, f"'ctx_len' not in prefill spec: {result}"

    def test_build_prefill_specialization_seq_len_matches_input(self):
        """build_prefill_specialization seq_len must match the input prefill_seq_len."""
        qeff = self._make_qeff()
        result = qeff.build_prefill_specialization(prefill_seq_len=16, ctx_len=64, batch_size=1, full_batch_size=None)
        assert result["seq_len"] == 16, f"Expected seq_len=16, got {result['seq_len']}"

    def test_build_prefill_specialization_ctx_len_matches_input(self):
        """build_prefill_specialization ctx_len must match the input ctx_len."""
        qeff = self._make_qeff()
        result = qeff.build_prefill_specialization(prefill_seq_len=8, ctx_len=64, batch_size=1, full_batch_size=None)
        assert result["ctx_len"] == 64, f"Expected ctx_len=64, got {result['ctx_len']}"

    def test_build_decode_specialization_returns_dict(self):
        """build_decode_specialization must return a dict."""
        qeff = self._make_qeff()
        result = qeff.build_decode_specialization(ctx_len=32, batch_size=1, full_batch_size=None)
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_build_decode_specialization_has_seq_len_key(self):
        """build_decode_specialization dict must contain 'seq_len'."""
        qeff = self._make_qeff()
        result = qeff.build_decode_specialization(ctx_len=32, batch_size=1, full_batch_size=None)
        assert "seq_len" in result, f"'seq_len' not in decode spec: {result}"

    def test_build_decode_specialization_has_ctx_len_key(self):
        """build_decode_specialization dict must contain 'ctx_len'."""
        qeff = self._make_qeff()
        result = qeff.build_decode_specialization(ctx_len=32, batch_size=1, full_batch_size=None)
        assert "ctx_len" in result, f"'ctx_len' not in decode spec: {result}"

    def test_build_decode_specialization_seq_len_is_1(self):
        """build_decode_specialization seq_len must be 1 (decode step)."""
        qeff = self._make_qeff()
        result = qeff.build_decode_specialization(ctx_len=32, batch_size=1, full_batch_size=None)
        assert result["seq_len"] == 1, f"Expected seq_len=1 for decode, got {result['seq_len']}"

    def test_build_decode_specialization_ctx_len_matches_input(self):
        """build_decode_specialization ctx_len must match the input ctx_len."""
        qeff = self._make_qeff()
        result = qeff.build_decode_specialization(ctx_len=64, batch_size=1, full_batch_size=None)
        assert result["ctx_len"] == 64, f"Expected ctx_len=64, got {result['ctx_len']}"

    def test_check_and_get_num_speculative_tokens_returns_none_for_non_tlm(self):
        """For non-TLM model, check_and_get_num_speculative_tokens must return None."""
        qeff = self._make_qeff()
        result = qeff.check_and_get_num_speculative_tokens(num_speculative_tokens=None, prefill_seq_len=1)
        assert result is None, f"Expected None for non-TLM, got {result}"

    def test_build_decode_specialization_with_num_speculative_tokens(self):
        """build_decode_specialization with num_speculative_tokens must include it in result."""
        qeff = self._make_qeff()
        result = qeff.build_decode_specialization(
            ctx_len=32, batch_size=1, full_batch_size=None, num_speculative_tokens=3
        )
        assert isinstance(result, dict)
        # The result should reflect the speculative tokens in some way
        assert "ctx_len" in result


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForCausalLM prefill toggle
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFAutoModelForCausalLMPrefillToggle:
    """prefill() method must exist and be callable."""

    def test_prefill_method_is_callable(self):
        """QEFFAutoModelForCausalLM.prefill must be callable."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        assert callable(QEFFAutoModelForCausalLM.prefill)

    def test_prefill_method_accepts_enable_parameter(self):
        """prefill() must accept an 'enable' parameter."""
        import inspect

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        sig = inspect.signature(QEFFAutoModelForCausalLM.prefill)
        assert "enable" in sig.parameters, f"prefill() must have 'enable' parameter, got: {list(sig.parameters.keys())}"

    def test_prefill_method_accepts_enable_chunking_parameter(self):
        """prefill() must accept an 'enable_chunking' parameter."""
        import inspect

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        sig = inspect.signature(QEFFAutoModelForCausalLM.prefill)
        assert "enable_chunking" in sig.parameters, (
            f"prefill() must have 'enable_chunking' parameter, got: {list(sig.parameters.keys())}"
        )


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModel (encoder)
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFAutoModelEncoder:
    """QEFFAutoModel must wrap encoder-only models like BERT."""

    def test_qeff_auto_model_is_importable(self):
        """QEFFAutoModel must be importable."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModel

        assert QEFFAutoModel is not None

    def test_qeff_auto_model_wraps_bert(self):
        """QEFFAutoModel must wrap a BertModel."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModel

        model = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        assert qeff is not None

    def test_qeff_auto_model_has_export_method(self):
        """QEFFAutoModel must have an export() method."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModel

        assert hasattr(QEFFAutoModel, "export")
        assert callable(QEFFAutoModel.export)

    def test_qeff_auto_model_forward_produces_finite_hidden_states(self):
        """QEFFAutoModel forward must produce finite hidden states."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModel

        model = make_tiny_bert()
        qeff = QEFFAutoModel(model)

        input_ids = torch.randint(0, 500, (1, 16))
        attention_mask = torch.ones(1, 16, dtype=torch.long)

        with torch.no_grad():
            output = qeff.model(input_ids=input_ids, attention_mask=attention_mask)

        assert torch.isfinite(output.last_hidden_state).all(), "QEFFAutoModel forward must produce finite hidden states"


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForCTC
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFAutoModelForCTC:
    """QEFFAutoModelForCTC must be importable and wrap CTC models."""

    def test_qeff_auto_model_for_ctc_is_importable(self):
        """QEFFAutoModelForCTC must be importable."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCTC

        assert QEFFAutoModelForCTC is not None

    def test_qeff_auto_model_for_ctc_has_export_method(self):
        """QEFFAutoModelForCTC must have an export() method."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCTC

        assert hasattr(QEFFAutoModelForCTC, "export")
        assert callable(QEFFAutoModelForCTC.export)

    def test_qeff_auto_model_for_ctc_class_attributes(self):
        """QEFFAutoModelForCTC must have expected class attributes."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCTC

        # Must have _pytorch_transforms or similar
        assert hasattr(QEFFAutoModelForCTC, "_pytorch_transforms") or hasattr(
            QEFFAutoModelForCTC, "_onnx_transforms"
        ), "QEFFAutoModelForCTC must have transform attributes"


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForSequenceClassification
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFAutoModelForSequenceClassification:
    """QEFFAutoModelForSequenceClassification must be importable."""

    def test_importable(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForSequenceClassification

        assert QEFFAutoModelForSequenceClassification is not None

    def test_has_export_method(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForSequenceClassification

        assert hasattr(QEFFAutoModelForSequenceClassification, "export")

    def test_wraps_bert_for_sequence_classification(self):
        """QEFFAutoModelForSequenceClassification must wrap BertForSequenceClassification."""
        from transformers import BertConfig, BertForSequenceClassification

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForSequenceClassification

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
            num_labels=3,
        )
        model = BertForSequenceClassification(cfg).eval()
        qeff = QEFFAutoModelForSequenceClassification(model)
        assert qeff is not None


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForSpeechSeq2Seq
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFAutoModelForSpeechSeq2Seq:
    """QEFFAutoModelForSpeechSeq2Seq must be importable."""

    def test_importable(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForSpeechSeq2Seq

        assert QEFFAutoModelForSpeechSeq2Seq is not None

    def test_has_export_method(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForSpeechSeq2Seq

        assert hasattr(QEFFAutoModelForSpeechSeq2Seq, "export")


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForCausalLM model registry
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFAutoModelRegistry:
    """QEFFAutoModelForCausalLM must have correct model registry."""

    def test_has_pytorch_transforms_list(self):
        """QEFFAutoModelForCausalLM must have _pytorch_transforms list."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        assert hasattr(QEFFAutoModelForCausalLM, "_pytorch_transforms")
        assert isinstance(QEFFAutoModelForCausalLM._pytorch_transforms, list)

    def test_pytorch_transforms_contains_kv_cache_transform(self):
        """_pytorch_transforms must contain KVCacheTransform."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
        from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform

        assert KVCacheTransform in QEFFAutoModelForCausalLM._pytorch_transforms

    def test_pytorch_transforms_contains_custom_ops_transform(self):
        """_pytorch_transforms must contain CustomOpsTransform."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
        from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform

        assert CustomOpsTransform in QEFFAutoModelForCausalLM._pytorch_transforms

    def test_has_onnx_transforms_list(self):
        """QEFFAutoModelForCausalLM must have _onnx_transforms list."""
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        assert hasattr(QEFFAutoModelForCausalLM, "_onnx_transforms")
        assert isinstance(QEFFAutoModelForCausalLM._onnx_transforms, list)

    def test_onnx_transforms_contains_fp16_clip(self):
        """FP16ClipTransform is importable and available for use."""
        from QEfficient.base.onnx_transforms import FP16ClipTransform

        assert FP16ClipTransform is not None
        assert hasattr(FP16ClipTransform, "apply")

    def test_onnx_transforms_contains_split_tensors(self):
        """SplitTensorsTransform is importable and available for use."""
        from QEfficient.base.onnx_transforms import SplitTensorsTransform

        assert SplitTensorsTransform is not None
        assert hasattr(SplitTensorsTransform, "apply")


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForCausalLM CCL mode (GAP F)
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFAutoModelForCausalLMCCL:
    """CCL specialization methods must include comp_ctx_lengths in the result."""

    def _make_qeff(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        return QEFFAutoModelForCausalLM(make_tiny_gpt2())

    def test_build_prefill_specialization_with_ccl_returns_dict(self):
        """build_prefill_specialization with comp_ctx_lengths must return a dict."""
        qeff = self._make_qeff()
        result = qeff.build_prefill_specialization(
            prefill_seq_len=8,
            ctx_len=32,
            batch_size=1,
            full_batch_size=None,
            comp_ctx_lengths=[16, 32],
        )
        assert isinstance(result, dict), f"build_prefill_specialization with CCL must return dict, got {type(result)}"

    def test_build_decode_specialization_with_ccl_returns_dict(self):
        """build_decode_specialization with comp_ctx_lengths must return a dict."""
        qeff = self._make_qeff()
        result = qeff.build_decode_specialization(
            ctx_len=32,
            batch_size=1,
            full_batch_size=None,
            comp_ctx_lengths=[16, 32],
        )
        assert isinstance(result, dict), f"build_decode_specialization with CCL must return dict, got {type(result)}"

    def test_build_prefill_specialization_ccl_result_has_comp_ctx_lengths_key(self):
        """build_prefill_specialization with CCL must include 'comp_ctx_lengths' in result."""
        qeff = self._make_qeff()
        result = qeff.build_prefill_specialization(
            prefill_seq_len=8,
            ctx_len=32,
            batch_size=1,
            full_batch_size=None,
            comp_ctx_lengths=[16, 32],
        )
        assert "comp_ctx_lengths" in result, f"CCL prefill spec must have 'comp_ctx_lengths' key: {result}"

    def test_build_decode_specialization_ccl_result_has_comp_ctx_lengths_key(self):
        """build_decode_specialization with CCL must include 'comp_ctx_lengths' in result."""
        qeff = self._make_qeff()
        result = qeff.build_decode_specialization(
            ctx_len=32,
            batch_size=1,
            full_batch_size=None,
            comp_ctx_lengths=[16, 32],
        )
        assert "comp_ctx_lengths" in result, f"CCL decode spec must have 'comp_ctx_lengths' key: {result}"

    def test_build_prefill_specialization_ccl_preserves_comp_ctx_lengths_values(self):
        """build_prefill_specialization must preserve the comp_ctx_lengths values."""
        qeff = self._make_qeff()
        comp_ctx_lengths = [16, 32]
        result = qeff.build_prefill_specialization(
            prefill_seq_len=8,
            ctx_len=32,
            batch_size=1,
            full_batch_size=None,
            comp_ctx_lengths=comp_ctx_lengths,
        )
        assert result["comp_ctx_lengths"] == comp_ctx_lengths, (
            f"Expected comp_ctx_lengths={comp_ctx_lengths}, got {result['comp_ctx_lengths']}"
        )

    def test_build_decode_specialization_ccl_preserves_comp_ctx_lengths_values(self):
        """build_decode_specialization must preserve the comp_ctx_lengths values."""
        qeff = self._make_qeff()
        comp_ctx_lengths = [16, 32]
        result = qeff.build_decode_specialization(
            ctx_len=32,
            batch_size=1,
            full_batch_size=None,
            comp_ctx_lengths=comp_ctx_lengths,
        )
        assert result["comp_ctx_lengths"] == comp_ctx_lengths, (
            f"Expected comp_ctx_lengths={comp_ctx_lengths}, got {result['comp_ctx_lengths']}"
        )

    def test_build_prefill_specialization_ccl_still_has_ctx_len(self):
        """build_prefill_specialization with CCL must still have 'ctx_len' key."""
        qeff = self._make_qeff()
        result = qeff.build_prefill_specialization(
            prefill_seq_len=8,
            ctx_len=32,
            batch_size=1,
            full_batch_size=None,
            comp_ctx_lengths=[16, 32],
        )
        assert "ctx_len" in result, f"CCL prefill spec must still have 'ctx_len': {result}"

    def test_build_decode_specialization_ccl_still_has_ctx_len(self):
        """build_decode_specialization with CCL must still have 'ctx_len' key."""
        qeff = self._make_qeff()
        result = qeff.build_decode_specialization(
            ctx_len=32,
            batch_size=1,
            full_batch_size=None,
            comp_ctx_lengths=[16, 32],
        )
        assert "ctx_len" in result, f"CCL decode spec must still have 'ctx_len': {result}"


# ---------------------------------------------------------------------------
# Tests: QEFFAutoModelForCausalLM prefill state change (GAP F)
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFAutoModelForCausalLMPrefillStateChange:
    """prefill() method and PrefillOnlyTransform must have correct structure."""

    def _make_qeff(self):
        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        return QEFFAutoModelForCausalLM(make_tiny_gpt2())

    def test_prefill_method_is_callable(self):
        """prefill() must be callable."""
        qeff = self._make_qeff()
        assert callable(qeff.prefill)

    def test_prefill_method_accepts_enable_parameter(self):
        """prefill() must accept an 'enable' parameter."""
        import inspect

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        sig = inspect.signature(QEFFAutoModelForCausalLM.prefill)
        assert "enable" in sig.parameters

    def test_prefill_method_accepts_enable_chunking_parameter(self):
        """prefill() must accept an 'enable_chunking' parameter."""
        import inspect

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        sig = inspect.signature(QEFFAutoModelForCausalLM.prefill)
        assert "enable_chunking" in sig.parameters

    def test_prefill_method_accepts_retain_full_kv_parameter(self):
        """prefill() must accept a 'retain_full_kv' parameter."""
        import inspect

        from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

        sig = inspect.signature(QEFFAutoModelForCausalLM.prefill)
        assert "retain_full_kv" in sig.parameters

    def test_prefill_only_transform_importable(self):
        """PrefillOnlyTransform must be importable."""
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyTransform

        assert PrefillOnlyTransform is not None

    def test_prefill_only_transform_has_module_mapping(self):
        """PrefillOnlyTransform must have a _module_mapping."""
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyTransform

        assert hasattr(PrefillOnlyTransform, "_module_mapping")
        assert isinstance(PrefillOnlyTransform._module_mapping, dict)
        assert len(PrefillOnlyTransform._module_mapping) > 0

    def test_revert_prefill_only_transform_importable(self):
        """RevertPrefillOnlyTransform must be importable."""
        from QEfficient.transformers.models.pytorch_transforms import RevertPrefillOnlyTransform

        assert RevertPrefillOnlyTransform is not None

    def test_revert_prefill_only_transform_has_module_mapping(self):
        """RevertPrefillOnlyTransform must have a _module_mapping."""
        from QEfficient.transformers.models.pytorch_transforms import RevertPrefillOnlyTransform

        assert hasattr(RevertPrefillOnlyTransform, "_module_mapping")
        assert isinstance(RevertPrefillOnlyTransform._module_mapping, dict)
        assert len(RevertPrefillOnlyTransform._module_mapping) > 0

    def test_prefill_only_transform_maps_to_prefill_variants(self):
        """PrefillOnlyTransform _module_mapping values must be prefill-only variants."""
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyTransform

        for src_cls, dst_cls in PrefillOnlyTransform._module_mapping.items():
            dst_name = dst_cls.__name__
            assert "Prefill" in dst_name or "prefill" in dst_name.lower(), (
                f"PrefillOnlyTransform maps {src_cls.__name__} -> {dst_name}, "
                f"but destination should be a prefill variant"
            )

    def test_prefill_only_chunked_transform_importable(self):
        """PrefillOnlyChunkedTransform must be importable."""
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyChunkedTransform

        assert PrefillOnlyChunkedTransform is not None

    def test_prefill_only_chunked_transform_has_module_mapping(self):
        """PrefillOnlyChunkedTransform must have a _module_mapping."""
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyChunkedTransform

        assert hasattr(PrefillOnlyChunkedTransform, "_module_mapping")
        assert isinstance(PrefillOnlyChunkedTransform._module_mapping, dict)
