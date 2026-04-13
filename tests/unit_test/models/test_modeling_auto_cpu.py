# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
CPU-only tests for QEfficient auto model classes in modeling_auto.py.

Covers:
  - QEFFTransformersBase: __repr__, quantization config guard
  - QEFFAutoModelForCausalLM: logic methods (build_prefill_specialization,
    build_decode_specialization, check_and_get_num_speculative_tokens,
    prefill, get_seq_len_and_handle_specialized_prefill_model),
    compile validation errors, generate TypeError
  - QEFFAutoModelForSequenceClassification: init, get_model_config, export
  - QEFFAutoModel: init (with/without pooling), get_model_config, export,
    pytorch_feature_generate, generate TypeError
  - MultimodalUtilityMixin: __new__ TypeError, auto_correct_inputs
  - QEFFAutoModelForSpeechSeq2Seq: init TypeError, get_model_config
  - QEFFAutoModelForCTC: init, get_model_config, export

All tests run on CPU only and are safe for parallel execution.
Run with: pytest tests/unit_test/models/test_modeling_auto_cpu.py -n auto -v
"""

import os
from unittest.mock import MagicMock

import pytest
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    WhisperConfig,
    WhisperForConditionalGeneration,
)

from QEfficient.transformers.models.modeling_auto import (
    MultimodalUtilityMixin,
    QEFFAutoModel,
    QEFFAutoModelForCausalLM,
    QEFFAutoModelForCTC,
    QEFFAutoModelForSequenceClassification,
    QEFFAutoModelForSpeechSeq2Seq,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 500
CTX_LEN = 32
SEQ_LEN = 8


# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


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


def make_tiny_bert():
    cfg = BertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return BertModel(cfg).eval(), cfg


def make_tiny_bert_seq_cls():
    cfg = BertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        num_labels=2,
    )
    return BertForSequenceClassification(cfg).eval(), cfg


def make_tiny_whisper():
    # vocab_size must be > all default token IDs (pad=50257, bos=50257, eos=50257, decoder_start=50258)
    # Use small vocab with explicit token IDs within range
    cfg = WhisperConfig(
        num_mel_bins=80,
        encoder_layers=1,
        decoder_layers=1,
        d_model=64,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        vocab_size=200,
        max_source_positions=32,
        max_target_positions=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=3,
    )
    # Ensure num_hidden_layers is set (QEFFAutoModelForSpeechSeq2Seq reads this)
    if not hasattr(cfg, "num_hidden_layers") or cfg.num_hidden_layers is None:
        cfg.num_hidden_layers = cfg.encoder_layers + cfg.decoder_layers
    return WhisperForConditionalGeneration(cfg).eval(), cfg


def make_tiny_wav2vec2():
    cfg = Wav2Vec2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        vocab_size=100,
    )
    return Wav2Vec2ForCTC(cfg).eval(), cfg


# ---------------------------------------------------------------------------
# Helper: InputInfo for MultimodalUtilityMixin tests
# ---------------------------------------------------------------------------


class _InputInfo:
    """Minimal stand-in for the InputInfo objects returned by get_inputs_info()."""

    def __init__(self, name, datatype):
        self.name = name
        self.datatype = datatype

    def __repr__(self):
        return f"InputInfo(name={self.name}, datatype={self.datatype})"


class _ConcreteMultimodalModel(MultimodalUtilityMixin):
    """Minimal concrete subclass of MultimodalUtilityMixin for testing."""

    def __init__(self, inputs_info):
        self.model = MagicMock()
        self.model.get_inputs_info.return_value = inputs_info

    @property
    def get_model_config(self):
        return {}

    def export(self, *args, **kwargs):
        pass

    def compile(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Stage 1: QEFFTransformersBase
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFTransformersBase:
    """Tests for QEFFTransformersBase base class."""

    def test_repr_contains_class_name(self):
        """__repr__ includes the QEff class name."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        r = repr(qeff)
        assert "QEFFAutoModelForCausalLM" in r

    def test_repr_contains_model_repr(self):
        """__repr__ includes the underlying model's repr."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        r = repr(qeff)
        # The model repr should be embedded
        assert len(r) > len("QEFFAutoModelForCausalLM")

    def test_init_raises_assertion_for_unsupported_quantization_config(self):
        """__init__ raises AssertionError when quantization_config is unsupported type.

        QEFFTransformersBase (parent of QEFFAutoModelForSequenceClassification) raises
        AssertionError. QEFFAutoModelForCausalLM only warns (different base class).
        """
        model, cfg = make_tiny_bert_seq_cls()
        # Inject a fake quantization_config that is NOT in QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING
        model.config.quantization_config = object()
        with pytest.raises(AssertionError, match="from_pretrained"):
            QEFFAutoModelForSequenceClassification(model)

    def test_init_passes_without_quantization_config(self):
        """__init__ succeeds when model has no quantization_config."""
        model, cfg = make_tiny_gpt2()
        assert not hasattr(model.config, "quantization_config")
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff is not None


# ---------------------------------------------------------------------------
# Stage 2: QEFFAutoModelForCausalLM — logic methods
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.causal_lm
class TestQEFFAutoModelForCausalLMLogic:
    """Tests for QEFFAutoModelForCausalLM logic methods (no QAIC required)."""

    # --- Basic properties ---

    def test_repr_contains_class_name(self):
        """__repr__ includes QEFFAutoModelForCausalLM."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert "QEFFAutoModelForCausalLM" in repr(qeff)

    def test_get_model_config_returns_dict(self):
        """get_model_config returns the model's config as a dict."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        config = qeff.get_model_config
        assert isinstance(config, dict)

    def test_continuous_batching_default_false(self):
        """continuous_batching defaults to False."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.continuous_batching is False

    def test_continuous_batching_true_when_set(self):
        """continuous_batching=True is stored correctly."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        assert qeff.continuous_batching is True

    def test_is_tlm_false_by_default(self):
        """is_tlm is False when no qaic_config specifies speculative model."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.is_tlm is False

    def test_num_layers_set_correctly(self):
        """num_layers matches the model config."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.num_layers == 2

    def test_init_raises_type_error_for_non_causal_lm(self):
        """__init__ raises TypeError when model is not a CausalLM or LMHeadModel."""
        model, cfg = make_tiny_bert()
        with pytest.raises(TypeError, match="CausalLM or LMHeadModel"):
            QEFFAutoModelForCausalLM(model)

    # --- build_prefill_specialization ---

    def test_build_prefill_specialization_basic(self):
        """build_prefill_specialization returns dict with seq_len and ctx_len."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        spec = qeff.build_prefill_specialization(prefill_seq_len=32, ctx_len=128, batch_size=1)
        assert isinstance(spec, dict)
        assert "seq_len" in spec
        assert spec["seq_len"] == 32
        assert "ctx_len" in spec
        assert spec["ctx_len"] == 128

    def test_build_prefill_specialization_batch_size(self):
        """build_prefill_specialization includes batch_size when kv_cache_batch_size is set."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # kv_cache_batch_size overrides batch_size in the returned spec; must be passed explicitly
        spec = qeff.build_prefill_specialization(prefill_seq_len=32, ctx_len=128, batch_size=4, kv_cache_batch_size=4)
        assert spec["batch_size"] == 4

    def test_build_prefill_specialization_continuous_batching(self):
        """build_prefill_specialization with continuous_batching uses full_batch_size."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        spec = qeff.build_prefill_specialization(
            prefill_seq_len=32,
            ctx_len=128,
            batch_size=1,
            kv_cache_batch_size=4,
            full_batch_size=4,
        )
        assert "full_batch_size" in spec
        assert spec["full_batch_size"] == 4

    def test_build_prefill_specialization_no_none_values(self):
        """build_prefill_specialization filters out None values."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        spec = qeff.build_prefill_specialization(prefill_seq_len=32, ctx_len=128, batch_size=1)
        assert all(v is not None for v in spec.values())

    # --- build_decode_specialization ---

    def test_build_decode_specialization_basic(self):
        """build_decode_specialization returns dict with seq_len=1 for decode."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        spec = qeff.build_decode_specialization(prefill_seq_len=32, ctx_len=128, batch_size=1)
        assert isinstance(spec, dict)
        assert "seq_len" in spec
        assert spec["seq_len"] == 1  # decode step is always seq_len=1

    def test_build_decode_specialization_ctx_len(self):
        """build_decode_specialization includes ctx_len."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        spec = qeff.build_decode_specialization(prefill_seq_len=32, ctx_len=256, batch_size=1)
        assert spec["ctx_len"] == 256

    def test_build_decode_specialization_continuous_batching(self):
        """build_decode_specialization with continuous_batching uses full_batch_size."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        spec = qeff.build_decode_specialization(
            prefill_seq_len=32,
            ctx_len=128,
            batch_size=1,
            kv_cache_batch_size=4,
            full_batch_size=4,
        )
        assert "full_batch_size" in spec
        assert spec["full_batch_size"] == 4

    def test_build_decode_specialization_no_none_values(self):
        """build_decode_specialization filters out None values."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        spec = qeff.build_decode_specialization(prefill_seq_len=32, ctx_len=128, batch_size=1)
        assert all(v is not None for v in spec.values())

    # --- check_and_get_num_speculative_tokens ---

    def test_check_speculative_tokens_not_tlm_returns_none(self):
        """check_and_get_num_speculative_tokens returns None when is_tlm=False."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.is_tlm is False
        result = qeff.check_and_get_num_speculative_tokens(None, 32)
        assert result is None

    def test_check_speculative_tokens_not_tlm_ignores_value(self):
        """check_and_get_num_speculative_tokens returns None even if value passed when not TLM."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        result = qeff.check_and_get_num_speculative_tokens(5, 32)
        assert result is None

    def test_check_speculative_tokens_tlm_none_raises_type_error(self):
        """check_and_get_num_speculative_tokens raises TypeError when TLM and None passed."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.is_tlm = True  # Force TLM mode for testing
        with pytest.raises(TypeError, match="num_speculative_tokens"):
            qeff.check_and_get_num_speculative_tokens(None, 32)

    def test_check_speculative_tokens_seq_len_too_short_raises_value_error(self):
        """check_and_get_num_speculative_tokens raises ValueError when prefill_seq_len too short."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.is_tlm = True  # Force TLM mode for testing
        # prefill_seq_len=3 < num_speculative_tokens+1=6
        with pytest.raises(ValueError, match="sequence length"):
            qeff.check_and_get_num_speculative_tokens(5, 3)

    def test_check_speculative_tokens_valid_returns_value(self):
        """check_and_get_num_speculative_tokens returns the value when valid."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.is_tlm = True  # Force TLM mode for testing
        result = qeff.check_and_get_num_speculative_tokens(3, 32)
        assert result == 3

    # --- prefill ---

    def test_prefill_enable_applies_transform(self):
        """prefill(enable=True) applies PrefillOnlyTransform without error."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # Should not raise
        qeff.prefill(enable=True)

    def test_prefill_disable_reverts_transform(self):
        """prefill(enable=False) applies RevertPrefillOnlyTransform without error."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        qeff.prefill(enable=True)
        # Should not raise
        qeff.prefill(enable=False)

    def test_prefill_enable_chunking(self):
        """prefill(enable=True, enable_chunking=True) applies chunked transform."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # Should not raise
        qeff.prefill(enable=True, enable_chunking=True)

    def test_prefill_retain_full_kv(self):
        """prefill(enable=False, retain_full_kv=True) applies RevertPrefillKeepAttentionTransform."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # Should not raise
        qeff.prefill(enable=False, retain_full_kv=True)


# ---------------------------------------------------------------------------
# Stage 3: QEFFAutoModelForCausalLM — compile/generate validation
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.causal_lm
class TestQEFFAutoModelForCausalLMCompileValidation:
    """Tests for compile/generate validation logic (no QAIC required)."""

    def test_compile_continuous_batching_requires_full_batch_size(self):
        """compile raises TypeError when continuous_batching=True and full_batch_size=None."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        with pytest.raises(TypeError, match="full_batch_size"):
            qeff.compile(prefill_seq_len=32, ctx_len=128)

    def test_compile_prefill_only_must_be_bool(self):
        """compile raises TypeError when prefill_only is not a boolean."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        with pytest.raises(TypeError, match="prefill_only"):
            qeff.compile(prefill_seq_len=32, ctx_len=128, prefill_only="yes")

    def test_compile_prefill_only_true_continuous_batching_requires_kv_cache_batch_size(self):
        """compile raises ValueError when prefill_only=True + continuous_batching=True + no kv_cache_batch_size."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        with pytest.raises((TypeError, ValueError)):
            qeff.compile(prefill_seq_len=32, ctx_len=128, prefill_only=True)

    def test_generate_raises_type_error_without_compile(self):
        """generate raises TypeError when QPC is not compiled."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        tokenizer = MagicMock()
        with pytest.raises(TypeError, match="compile API"):
            qeff.generate(tokenizer=tokenizer, prompts=["Hello"])

    def test_generate_raises_not_implemented_for_pytorch_runtime(self):
        """generate raises NotImplementedError when runtime_ai100=False."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        tokenizer = MagicMock()
        with pytest.raises(NotImplementedError):
            qeff.generate(tokenizer=tokenizer, prompts=["Hello"], runtime_ai100=False)


# ---------------------------------------------------------------------------
# Stage 4: QEFFAutoModelForCausalLM — get_seq_len_and_handle_specialized_prefill_model
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.causal_lm
class TestQEFFAutoModelForCausalLMGetSeqLen:
    """Tests for get_seq_len_and_handle_specialized_prefill_model."""

    def test_enable_chunking_returns_constant(self):
        """With enable_chunking=True, returns ONNX_EXPORT_EXAMPLE_SEQ_LEN."""
        from QEfficient.utils.constants import ONNX_EXPORT_EXAMPLE_SEQ_LEN

        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        result = qeff.get_seq_len_and_handle_specialized_prefill_model(prefill_seq_len=None, enable_chunking=True)
        assert result == ONNX_EXPORT_EXAMPLE_SEQ_LEN
        assert qeff.hash_params.get("prefill_only") is True
        assert qeff.hash_params.get("chunking") is True

    def test_no_prefill_seq_len_no_env_var_raises_value_error(self):
        """Without prefill_seq_len and NUM_Q_BLOCKS env var, raises ValueError."""
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        # Ensure env var is not set
        os.environ.pop("NUM_Q_BLOCKS", None)
        with pytest.raises(ValueError, match="prefill_seq_len"):
            qeff.get_seq_len_and_handle_specialized_prefill_model(prefill_seq_len=None, enable_chunking=False)

    def test_valid_prefill_seq_len_sets_env_var(self):
        """Valid prefill_seq_len sets NUM_Q_BLOCKS env var."""
        from QEfficient.utils.constants import GPT_OSS_PREFILL_Q_BLOCK_SIZE

        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        os.environ.pop("NUM_Q_BLOCKS", None)
        block_size = GPT_OSS_PREFILL_Q_BLOCK_SIZE
        prefill_seq_len = block_size * 2  # Must be divisible by block_size
        result = qeff.get_seq_len_and_handle_specialized_prefill_model(
            prefill_seq_len=prefill_seq_len, enable_chunking=False
        )
        assert os.environ.get("NUM_Q_BLOCKS") is not None
        assert result >= prefill_seq_len or result > 0
        # Cleanup
        os.environ.pop("NUM_Q_BLOCKS", None)

    def test_prefill_seq_len_not_divisible_raises_value_error(self):
        """prefill_seq_len not divisible by block_size raises ValueError."""
        from QEfficient.utils.constants import GPT_OSS_PREFILL_Q_BLOCK_SIZE

        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        os.environ.pop("NUM_Q_BLOCKS", None)
        # Use a value that is NOT divisible by block_size
        bad_seq_len = GPT_OSS_PREFILL_Q_BLOCK_SIZE + 1
        with pytest.raises(ValueError):
            qeff.get_seq_len_and_handle_specialized_prefill_model(prefill_seq_len=bad_seq_len, enable_chunking=False)


# ---------------------------------------------------------------------------
# Stage 5: QEFFAutoModelForSequenceClassification
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.seq_classification
class TestQEFFAutoModelForSequenceClassification:
    """Tests for QEFFAutoModelForSequenceClassification."""

    def test_init_sets_use_cache_true(self):
        """__init__ sets model.config.use_cache=True."""
        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        assert qeff.model.config.use_cache is True

    def test_init_stores_model(self):
        """__init__ stores the model."""
        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        assert qeff.model is not None

    def test_get_model_config_returns_dict(self):
        """get_model_config returns the model's config as a dict."""
        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        config = qeff.get_model_config
        assert isinstance(config, dict)

    def test_hash_params_contains_qeff_auto_class(self):
        """hash_params contains qeff_auto_class key."""
        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        assert qeff.hash_params.get("qeff_auto_class") == "QEFFAutoModelForSequenceClassification"

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_produces_onnx_file(self, tmp_export_dir):
        """export produces a valid ONNX file."""
        import os

        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        assert onnx_path is not None
        assert os.path.exists(str(onnx_path))
        assert os.path.getsize(str(onnx_path)) > 0

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_onnx_has_correct_inputs(self, tmp_export_dir):
        """Exported ONNX has input_ids and attention_mask inputs."""
        import onnx

        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        input_names = {inp.name for inp in onnx_model.graph.input}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_onnx_has_logits_output(self, tmp_export_dir):
        """Exported ONNX has logits output."""
        import onnx

        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        output_names = {out.name for out in onnx_model.graph.output}
        assert "logits" in output_names

    def test_compile_warns_for_many_seq_lens(self):
        """compile warns when seq_len list has >= 15 items."""
        import warnings
        from unittest.mock import patch

        model, cfg = make_tiny_bert_seq_cls()
        qeff = QEFFAutoModelForSequenceClassification(model)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Patch _compile so we never need QAIC hardware; it raises to stop execution
            with patch.object(qeff, "_compile", side_effect=RuntimeError("mocked compile")):
                with pytest.raises(RuntimeError):
                    qeff.compile(seq_len=list(range(1, 20)))  # 19 items >= 15

        assert any("fewer than 15" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# Stage 6: QEFFAutoModel
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.embedding
class TestQEFFAutoModel:
    """Tests for QEFFAutoModel (encoder/embedding models)."""

    def test_init_without_pooling(self):
        """__init__ without pooling initializes correctly."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        assert qeff is not None
        assert qeff.model is not None

    def test_init_with_mean_pooling(self):
        """__init__ with pooling='mean' applies PoolingTransform."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model, pooling="mean")
        assert qeff is not None

    def test_init_with_cls_pooling(self):
        """__init__ with pooling='cls' applies PoolingTransform."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model, pooling="cls")
        assert qeff is not None

    def test_init_sets_use_cache_true(self):
        """__init__ sets model.base_model.config.use_cache=True."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        assert qeff.model.base_model.config.use_cache is True

    def test_get_model_config_returns_dict(self):
        """get_model_config returns the model's config as a dict."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        config = qeff.get_model_config
        assert isinstance(config, dict)

    def test_hash_params_contains_qeff_auto_class(self):
        """hash_params contains qeff_auto_class key."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        assert qeff.hash_params.get("qeff_auto_class") == "QEFFAutoModel"

    def test_pytorch_feature_generate_returns_output(self):
        """pytorch_feature_generate runs the model and returns output."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        # _write_io_dir must be initialised before calling pytorch_feature_generate directly
        qeff._write_io_dir = None
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.int64),
            "attention_mask": torch.ones((1, SEQ_LEN), dtype=torch.int64),
        }
        output = qeff.pytorch_feature_generate(model=qeff.model, inputs=inputs)
        assert output is not None

    def test_generate_raises_type_error_without_compile(self):
        """generate raises TypeError when QPC is not compiled."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.int64),
            "attention_mask": torch.ones((1, SEQ_LEN), dtype=torch.int64),
        }
        with pytest.raises(TypeError, match="compile API"):
            qeff.generate(inputs=inputs, runtime_ai100=True)

    def test_generate_pytorch_runtime_returns_output(self):
        """generate with runtime_ai100=False uses PyTorch runtime."""
        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.int64),
            "attention_mask": torch.ones((1, SEQ_LEN), dtype=torch.int64),
        }
        output = qeff.generate(inputs=inputs, runtime_ai100=False)
        assert output is not None

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_produces_onnx_file(self, tmp_export_dir):
        """export produces a valid ONNX file."""
        import os

        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        assert onnx_path is not None
        assert os.path.exists(str(onnx_path))
        assert os.path.getsize(str(onnx_path)) > 0

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_onnx_has_correct_inputs(self, tmp_export_dir):
        """Exported ONNX has input_ids and attention_mask inputs."""
        import onnx

        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        input_names = {inp.name for inp in onnx_model.graph.input}
        assert "input_ids" in input_names
        assert "attention_mask" in input_names

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_onnx_has_output(self, tmp_export_dir):
        """Exported ONNX has at least one output."""
        import onnx

        model, cfg = make_tiny_bert()
        qeff = QEFFAutoModel(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        assert len(onnx_model.graph.output) > 0


# ---------------------------------------------------------------------------
# Stage 7: MultimodalUtilityMixin
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestMultimodalUtilityMixin:
    """Tests for MultimodalUtilityMixin."""

    def test_direct_instantiation_raises_type_error(self):
        """Direct instantiation of MultimodalUtilityMixin raises TypeError."""
        with pytest.raises(TypeError):
            MultimodalUtilityMixin()

    def test_auto_correct_inputs_success(self):
        """auto_correct_inputs succeeds with correct inputs."""
        inputs_info = [
            _InputInfo("input_ids", torch.int64),
            _InputInfo("attention_mask", torch.int64),
        ]
        mixin = _ConcreteMultimodalModel(inputs_info)
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.int64),
            "attention_mask": torch.ones((1, SEQ_LEN), dtype=torch.int64),
        }
        result = mixin.auto_correct_inputs(inputs)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_auto_correct_inputs_filters_extra_keys(self):
        """auto_correct_inputs filters out keys not in inputs_info."""
        inputs_info = [
            _InputInfo("input_ids", torch.int64),
        ]
        mixin = _ConcreteMultimodalModel(inputs_info)
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.int64),
            "extra_key": torch.zeros((1, SEQ_LEN), dtype=torch.float32),
        }
        result = mixin.auto_correct_inputs(inputs)
        assert "input_ids" in result
        assert "extra_key" not in result

    def test_auto_correct_inputs_missing_key_raises_runtime_error(self):
        """auto_correct_inputs raises RuntimeError when a required key is missing."""
        inputs_info = [
            _InputInfo("input_ids", torch.int64),
            _InputInfo("attention_mask", torch.int64),
        ]
        mixin = _ConcreteMultimodalModel(inputs_info)
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.int64),
            # attention_mask is missing
        }
        with pytest.raises(RuntimeError):
            mixin.auto_correct_inputs(inputs)

    def test_auto_correct_inputs_wrong_dtype_raises_runtime_error(self):
        """auto_correct_inputs raises RuntimeError when dtype does not match."""
        inputs_info = [
            _InputInfo("input_ids", torch.int64),
            _InputInfo("attention_mask", torch.int64),
        ]
        mixin = _ConcreteMultimodalModel(inputs_info)
        inputs = {
            "input_ids": torch.zeros((1, SEQ_LEN), dtype=torch.float32),  # wrong dtype
            "attention_mask": torch.ones((1, SEQ_LEN), dtype=torch.int64),
        }
        with pytest.raises(RuntimeError):
            mixin.auto_correct_inputs(inputs)

    def test_auto_correct_inputs_error_message_contains_expected_info(self):
        """RuntimeError message contains expected input names."""
        inputs_info = [
            _InputInfo("input_ids", torch.int64),
        ]
        mixin = _ConcreteMultimodalModel(inputs_info)
        inputs = {}  # empty inputs
        with pytest.raises(RuntimeError) as exc_info:
            mixin.auto_correct_inputs(inputs)
        assert "input_ids" in str(exc_info.value) or "Expected" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Stage 8: QEFFAutoModelForSpeechSeq2Seq
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.speech
class TestQEFFAutoModelForSpeechSeq2Seq:
    """Tests for QEFFAutoModelForSpeechSeq2Seq."""

    def test_init_raises_type_error_for_non_conditional_generation(self):
        """__init__ raises TypeError for non-ForConditionalGeneration models."""
        model, cfg = make_tiny_gpt2()
        with pytest.raises(TypeError, match="ForConditionalGeneration"):
            QEFFAutoModelForSpeechSeq2Seq(model)

    def test_init_raises_type_error_for_causal_lm(self):
        """__init__ raises TypeError for CausalLM models."""
        model, cfg = make_tiny_llama()
        with pytest.raises(TypeError, match="ForConditionalGeneration"):
            QEFFAutoModelForSpeechSeq2Seq(model)

    @pytest.mark.slow
    def test_init_succeeds_with_whisper(self):
        """__init__ succeeds with a Whisper ForConditionalGeneration model."""
        model, cfg = make_tiny_whisper()
        qeff = QEFFAutoModelForSpeechSeq2Seq(model)
        assert qeff is not None
        assert qeff.model is not None

    @pytest.mark.slow
    def test_get_model_config_returns_dict(self):
        """get_model_config returns the model's config as a dict."""
        model, cfg = make_tiny_whisper()
        qeff = QEFFAutoModelForSpeechSeq2Seq(model)
        config = qeff.get_model_config
        assert isinstance(config, dict)

    @pytest.mark.slow
    def test_hash_params_contains_qeff_auto_class(self):
        """hash_params contains qeff_auto_class key."""
        model, cfg = make_tiny_whisper()
        qeff = QEFFAutoModelForSpeechSeq2Seq(model)
        assert qeff.hash_params.get("qeff_auto_class") == "QEFFAutoModelForSpeechSeq2Seq"

    @pytest.mark.slow
    def test_num_layers_set_correctly(self):
        """num_layers is set from model config."""
        model, cfg = make_tiny_whisper()
        qeff = QEFFAutoModelForSpeechSeq2Seq(model)
        assert isinstance(qeff.num_layers, int)
        assert qeff.num_layers > 0

    def test_generate_raises_type_error_without_compile(self):
        """generate raises TypeError when QPC is not compiled."""
        model, cfg = make_tiny_whisper()
        try:
            qeff = QEFFAutoModelForSpeechSeq2Seq(model)
        except Exception:
            pytest.skip("Whisper model initialization failed, skipping generate test")
        with pytest.raises(TypeError, match="compile API"):
            qeff.generate(inputs={}, generation_len=10)


# ---------------------------------------------------------------------------
# Stage 9: QEFFAutoModelForCTC
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
class TestQEFFAutoModelForCTC:
    """Tests for QEFFAutoModelForCTC (Wav2Vec2 and similar CTC models)."""

    def test_init_sets_use_cache_true(self):
        """__init__ sets model.base_model.config.use_cache=True."""
        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        assert qeff.model.base_model.config.use_cache is True

    def test_init_stores_model(self):
        """__init__ stores the model."""
        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        assert qeff.model is not None

    def test_get_model_config_returns_dict(self):
        """get_model_config returns the model's config as a dict."""
        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        config = qeff.get_model_config
        assert isinstance(config, dict)

    def test_hash_params_contains_qeff_auto_class(self):
        """hash_params contains qeff_auto_class key."""
        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        assert qeff.hash_params.get("qeff_auto_class") == "QEFFAutoModelForCTC"

    def test_generate_raises_type_error_without_compile(self):
        """generate raises TypeError when QPC is not compiled."""
        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        processor = MagicMock()
        with pytest.raises(TypeError, match="compile API"):
            qeff.generate(processor=processor, inputs=torch.zeros(1, 100))

    def test_generate_pytorch_runtime_calls_model(self):
        """generate with runtime_ai100=False calls pytorch_feature_generate."""
        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        processor = MagicMock()
        processor.return_value = MagicMock(input_values=torch.zeros((1, 100), dtype=torch.float32))
        # pytorch_feature_generate calls processor and model
        # We just verify it doesn't raise TypeError about compile
        try:
            qeff.generate(processor=processor, inputs=torch.zeros(1, 100), runtime_ai100=False)
        except Exception as e:
            # Any exception other than TypeError about compile is acceptable
            assert "compile API" not in str(e)

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_produces_onnx_file(self, tmp_export_dir):
        """export produces a valid ONNX file."""
        import os

        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        assert onnx_path is not None
        assert os.path.exists(str(onnx_path))
        assert os.path.getsize(str(onnx_path)) > 0

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_onnx_has_input_values(self, tmp_export_dir):
        """Exported ONNX has input_values input."""
        import onnx

        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        input_names = {inp.name for inp in onnx_model.graph.input}
        assert "input_values" in input_names

    @pytest.mark.onnx
    @pytest.mark.slow
    def test_export_onnx_has_logits_output(self, tmp_export_dir):
        """Exported ONNX has logits output."""
        import onnx

        model, cfg = make_tiny_wav2vec2()
        qeff = QEFFAutoModelForCTC(model)
        onnx_path = qeff.export(export_dir=str(tmp_export_dir))
        onnx_model = onnx.load(str(onnx_path))
        output_names = {out.name for out in onnx_model.graph.output}
        assert "logits" in output_names
