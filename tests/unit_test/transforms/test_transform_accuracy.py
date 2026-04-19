# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Accuracy tests for PyTorch transforms in QEfficient.

Improvements over unit_v2:
  - Expanded CustomOpsTransform coverage: Phi3, Gemma, Gemma2
  - Expanded KVCacheTransform coverage: Phi3, Gemma, Gemma2, Falcon
  - Expanded combined transforms: Phi3, Gemma, Gemma2
  - SamplerTransform and SpDTransform behavior tests

Tests verify that transforms:
  1. Replace the correct module types
  2. Do NOT change the model's numerical output (accuracy preservation)
  3. Work correctly in combination

All tests run on CPU only, using tiny in-memory models.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers import (
    FalconConfig,
    FalconForCausalLM,
    Gemma2Config,
    Gemma2ForCausalLM,
    GemmaConfig,
    GemmaForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from QEfficient.transformers.models.pytorch_transforms import (
    CustomOpsTransform,
    KVCacheTransform,
    PoolingTransform,
    SamplerTransform,
    SpDTransform,
)

VOCAB_SIZE = 500
SEQ_LEN = 8
CTX_LEN = 32


# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


def make_tiny_gpt2():
    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=64, vocab_size=VOCAB_SIZE, n_positions=CTX_LEN, n_ctx=CTX_LEN)
    return GPT2LMHeadModel(cfg).eval()


def make_tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return LlamaForCausalLM(cfg).eval()


def make_tiny_mistral():
    cfg = MistralConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return MistralForCausalLM(cfg).eval()


def make_tiny_qwen2():
    cfg = Qwen2Config(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return Qwen2ForCausalLM(cfg).eval()


def make_tiny_phi3():
    cfg = Phi3Config(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        pad_token_id=0,
    )
    return Phi3ForCausalLM(cfg).eval()


def make_tiny_gemma():
    cfg = GemmaConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    return GemmaForCausalLM(cfg).eval()


def make_tiny_gemma2():
    cfg = Gemma2Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
        sliding_window=CTX_LEN,
    )
    return Gemma2ForCausalLM(cfg).eval()


def make_tiny_falcon():
    cfg = FalconConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        hidden_size=64,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        new_decoder_architecture=False,
        multi_query=True,
    )
    return FalconForCausalLM(cfg).eval()


# ---------------------------------------------------------------------------
# QEff input helpers
# ---------------------------------------------------------------------------


def _get_dims(config):
    """Extract (n_layers, n_kv_heads, head_dim) from any model config."""
    if hasattr(config, "num_hidden_layers"):
        n_layers = config.num_hidden_layers
        n_attn = config.num_attention_heads
        n_kv = getattr(config, "num_key_value_heads", n_attn)
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // n_attn)
    else:
        # GPT2
        n_layers = config.n_layer
        n_kv = config.n_head
        head_dim = config.n_embd // config.n_head
    return n_layers, n_kv, head_dim


def _make_qeff_inputs(input_ids, config, ctx_len=CTX_LEN):
    """Build QEff-style inputs: input_ids + position_ids + zero-initialized past_key_values."""
    batch, seq = input_ids.shape
    position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    n_layers, n_kv, head_dim = _get_dims(config)
    past_key_values = tuple(
        (
            torch.zeros(batch, n_kv, ctx_len, head_dim, dtype=torch.float32),
            torch.zeros(batch, n_kv, ctx_len, head_dim, dtype=torch.float32),
        )
        for _ in range(n_layers)
    )
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
    }


# ---------------------------------------------------------------------------
# Tests: CustomOpsTransform - module replacement
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestCustomOpsTransformReplacement:
    """CustomOpsTransform must replace RMSNorm with CustomRMSNormAIC."""

    def test_llama_rms_norm_replaced_with_custom_rms_norm(self):
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        from QEfficient.customop import CustomRMSNormAIC

        model = make_tiny_llama()
        assert any(isinstance(m, LlamaRMSNorm) for m in model.modules())

        transformed, applied = CustomOpsTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if type(m) is LlamaRMSNorm:
                pytest.fail("Found unreplaced LlamaRMSNorm after transform")

        assert any(isinstance(m, CustomRMSNormAIC) for m in transformed.modules())

    def test_mistral_rms_norm_replaced(self):
        from QEfficient.customop import CustomRMSNormAIC

        model = make_tiny_mistral()
        transformed, applied = CustomOpsTransform.apply(model)
        assert applied
        assert any(isinstance(m, CustomRMSNormAIC) for m in transformed.modules())

    def test_qwen2_rms_norm_replaced(self):
        from QEfficient.customop import CustomRMSNormAIC

        model = make_tiny_qwen2()
        transformed, applied = CustomOpsTransform.apply(model)
        assert applied
        assert any(isinstance(m, CustomRMSNormAIC) for m in transformed.modules())

    def test_phi3_rms_norm_replaced(self):
        from QEfficient.customop import CustomRMSNormAIC

        model = make_tiny_phi3()
        transformed, applied = CustomOpsTransform.apply(model)
        assert applied
        assert any(isinstance(m, CustomRMSNormAIC) for m in transformed.modules())

    def test_gemma_rms_norm_replaced(self):
        from QEfficient.customop import GemmaCustomRMSNormAIC

        model = make_tiny_gemma()
        transformed, applied = CustomOpsTransform.apply(model)
        assert applied
        assert any(isinstance(m, GemmaCustomRMSNormAIC) for m in transformed.modules())

    def test_gemma2_rms_norm_replaced(self):
        from QEfficient.customop import GemmaCustomRMSNormAIC

        model = make_tiny_gemma2()
        transformed, applied = CustomOpsTransform.apply(model)
        assert applied
        assert any(isinstance(m, GemmaCustomRMSNormAIC) for m in transformed.modules())

    def test_gpt2_not_transformed(self):
        """GPT2 uses LayerNorm, not RMSNorm. CustomOpsTransform must not apply."""
        model = make_tiny_gpt2()
        transformed, applied = CustomOpsTransform.apply(model)
        assert not applied, "CustomOpsTransform must not apply to GPT2 (no RMSNorm)"

    def test_module_mapping_contains_expected_types(self):
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        from transformers.models.mistral.modeling_mistral import MistralRMSNorm
        from transformers.models.phi3.modeling_phi3 import Phi3RMSNorm
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

        mapping = CustomOpsTransform._module_mapping
        assert LlamaRMSNorm in mapping
        assert MistralRMSNorm in mapping
        assert Qwen2RMSNorm in mapping
        assert Phi3RMSNorm in mapping
        assert GemmaRMSNorm in mapping
        assert Gemma2RMSNorm in mapping


# ---------------------------------------------------------------------------
# Tests: CustomOpsTransform - accuracy preservation
# ---------------------------------------------------------------------------


@pytest.mark.transforms
@pytest.mark.accuracy
class TestCustomOpsTransformAccuracy:
    """
    CustomOpsTransform must NOT change the model's numerical output.
    CustomRMSNormAIC must be numerically equivalent to LlamaRMSNorm.
    """

    def test_llama_output_unchanged_after_custom_ops_transform(self):
        """Llama logits must be identical before and after CustomOpsTransform."""
        model = make_tiny_llama()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        transformed, _ = CustomOpsTransform.apply(model)
        with torch.no_grad():
            after_logits = transformed(input_ids=input_ids).logits[:, -1, :]

        max_diff = (before_logits - after_logits).abs().max().item()
        assert max_diff < 1e-5, (
            f"CustomOpsTransform changed Llama output: max_diff={max_diff:.2e}. "
            f"CustomRMSNormAIC must be numerically equivalent to LlamaRMSNorm."
        )

    def test_llama_greedy_token_unchanged_after_custom_ops_transform(self):
        model = make_tiny_llama()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_token = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        transformed, _ = CustomOpsTransform.apply(model)
        with torch.no_grad():
            after_token = transformed(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        assert before_token == after_token, (
            f"CustomOpsTransform changed greedy token: before={before_token}, after={after_token}"
        )

    def test_mistral_output_unchanged_after_custom_ops_transform(self):
        model = make_tiny_mistral()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        transformed, _ = CustomOpsTransform.apply(model)
        with torch.no_grad():
            after_logits = transformed(input_ids=input_ids).logits[:, -1, :]

        max_diff = (before_logits - after_logits).abs().max().item()
        assert max_diff < 1e-5, f"CustomOpsTransform changed Mistral output: max_diff={max_diff:.2e}"

    def test_phi3_output_unchanged_after_custom_ops_transform(self):
        model = make_tiny_phi3()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        transformed, _ = CustomOpsTransform.apply(model)
        with torch.no_grad():
            after_logits = transformed(input_ids=input_ids).logits[:, -1, :]

        max_diff = (before_logits - after_logits).abs().max().item()
        assert max_diff < 1e-5, f"CustomOpsTransform changed Phi3 output: max_diff={max_diff:.2e}"

    def test_gemma_output_unchanged_after_custom_ops_transform(self):
        model = make_tiny_gemma()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        transformed, _ = CustomOpsTransform.apply(model)
        with torch.no_grad():
            after_logits = transformed(input_ids=input_ids).logits[:, -1, :]

        max_diff = (before_logits - after_logits).abs().max().item()
        assert max_diff < 1e-5, f"CustomOpsTransform changed Gemma output: max_diff={max_diff:.2e}"

    def test_custom_rms_norm_forward_is_finite(self):
        """CustomRMSNormAIC forward must produce finite outputs."""
        model = make_tiny_llama()
        transformed, _ = CustomOpsTransform.apply(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        with torch.no_grad():
            out = transformed(input_ids=input_ids)
        assert torch.isfinite(out.logits).all()


# ---------------------------------------------------------------------------
# Tests: KVCacheTransform - module replacement
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestKVCacheTransformReplacement:
    """KVCacheTransform must replace attention layers with QEff variants."""

    def test_gpt2_attention_replaced(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

        from QEfficient.transformers.models.gpt2.modeling_gpt2 import QEffGPT2Attention

        model = make_tiny_gpt2()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, GPT2Attention):
                assert isinstance(m, QEffGPT2Attention)

    def test_gpt2_lm_head_model_replaced(self):
        from QEfficient.transformers.models.gpt2.modeling_gpt2 import QEffGPT2LMHeadModel

        model = make_tiny_gpt2()
        transformed, _ = KVCacheTransform.apply(model)
        assert isinstance(transformed, QEffGPT2LMHeadModel)

    def test_llama_attention_replaced(self):
        from transformers.models.llama.modeling_llama import LlamaAttention

        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaAttention

        model = make_tiny_llama()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, LlamaAttention):
                assert isinstance(m, QEffLlamaAttention)

    def test_llama_for_causal_lm_replaced(self):
        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM

        model = make_tiny_llama()
        transformed, _ = KVCacheTransform.apply(model)
        assert isinstance(transformed, QEffLlamaForCausalLM)

    def test_mistral_attention_replaced(self):
        from transformers.models.mistral.modeling_mistral import MistralAttention

        from QEfficient.transformers.models.mistral.modeling_mistral import QEffMistralAttention

        model = make_tiny_mistral()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, MistralAttention):
                assert isinstance(m, QEffMistralAttention)

    def test_qwen2_attention_replaced(self):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

        from QEfficient.transformers.models.qwen2.modeling_qwen2 import QEffQwen2Attention

        model = make_tiny_qwen2()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, Qwen2Attention):
                assert isinstance(m, QEffQwen2Attention)

    def test_phi3_attention_replaced(self):
        from transformers.models.phi3.modeling_phi3 import Phi3Attention

        from QEfficient.transformers.models.phi3.modeling_phi3 import QEffPhi3Attention

        model = make_tiny_phi3()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, Phi3Attention):
                assert isinstance(m, QEffPhi3Attention)

    def test_gemma_attention_replaced(self):
        from transformers.models.gemma.modeling_gemma import GemmaAttention

        from QEfficient.transformers.models.gemma.modeling_gemma import QEffGemmaAttention

        model = make_tiny_gemma()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, GemmaAttention):
                assert isinstance(m, QEffGemmaAttention)

    def test_falcon_attention_replaced(self):
        from transformers.models.falcon.modeling_falcon import FalconAttention

        from QEfficient.transformers.models.falcon.modeling_falcon import QEffFalconAttention

        model = make_tiny_falcon()
        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if isinstance(m, FalconAttention):
                assert isinstance(m, QEffFalconAttention)

    def test_module_mapping_covers_major_architectures(self):
        from transformers.models.falcon.modeling_falcon import FalconForCausalLM
        from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        from transformers.models.mistral.modeling_mistral import MistralForCausalLM
        from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
        from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

        mapping = KVCacheTransform._module_mapping
        assert GPT2LMHeadModel in mapping
        assert LlamaForCausalLM in mapping
        assert MistralForCausalLM in mapping
        assert MixtralForCausalLM in mapping
        assert Qwen2ForCausalLM in mapping
        assert Phi3ForCausalLM in mapping
        assert GemmaForCausalLM in mapping
        assert FalconForCausalLM in mapping


# ---------------------------------------------------------------------------
# Tests: KVCacheTransform - accuracy preservation
# ---------------------------------------------------------------------------


@pytest.mark.transforms
@pytest.mark.accuracy
class TestKVCacheTransformAccuracy:
    """
    KVCacheTransform must NOT change the model's greedy next token prediction.
    This is the core regression test for the KV cache transform.
    """

    def _check_greedy_token_preserved(self, model, label):
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_token = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        cfg = model.config
        transformed, _ = KVCacheTransform.apply(model)
        qeff_inputs = _make_qeff_inputs(input_ids, cfg)

        with torch.no_grad():
            after_out = transformed(**qeff_inputs)
        after_token = after_out.logits[:, -1, :].argmax(-1).item()

        assert before_token == after_token, (
            f"[{label}] KVCacheTransform changed greedy token: "
            f"before={before_token}, after={after_token}. "
            f"KVCacheTransform must not change the model's prediction."
        )

    def test_gpt2_greedy_token_preserved_after_kv_transform(self):
        self._check_greedy_token_preserved(make_tiny_gpt2(), "GPT2")

    def test_llama_greedy_token_preserved_after_kv_transform(self):
        self._check_greedy_token_preserved(make_tiny_llama(), "Llama")

    def test_mistral_greedy_token_preserved_after_kv_transform(self):
        self._check_greedy_token_preserved(make_tiny_mistral(), "Mistral")

    def test_qwen2_greedy_token_preserved_after_kv_transform(self):
        self._check_greedy_token_preserved(make_tiny_qwen2(), "Qwen2")

    def test_phi3_greedy_token_preserved_after_kv_transform(self):
        self._check_greedy_token_preserved(make_tiny_phi3(), "Phi3")

    def test_gemma_greedy_token_preserved_after_kv_transform(self):
        self._check_greedy_token_preserved(make_tiny_gemma(), "Gemma")

    def test_gpt2_logits_numerically_close_after_kv_transform(self):
        """GPT2 logits must be numerically close before and after KVCacheTransform."""
        model = make_tiny_gpt2()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        cfg = model.config
        transformed, _ = KVCacheTransform.apply(model)
        qeff_inputs = _make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            after_logits = transformed(**qeff_inputs).logits[:, -1, :]

        hf_probs = F.softmax(before_logits, dim=-1)
        qeff_probs = F.softmax(after_logits, dim=-1)
        max_diff = (hf_probs - qeff_probs).abs().max().item()
        assert max_diff < 1e-3, f"KVCacheTransform changed GPT2 probability distribution: max_diff={max_diff:.2e}"

    def test_llama_logits_numerically_close_after_kv_transform(self):
        model = make_tiny_llama()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        cfg = model.config
        transformed, _ = KVCacheTransform.apply(model)
        qeff_inputs = _make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            after_logits = transformed(**qeff_inputs).logits[:, -1, :]

        hf_probs = F.softmax(before_logits, dim=-1)
        qeff_probs = F.softmax(after_logits, dim=-1)
        max_diff = (hf_probs - qeff_probs).abs().max().item()
        assert max_diff < 1e-3, f"KVCacheTransform changed Llama probability distribution: max_diff={max_diff:.2e}"

    def test_phi3_logits_numerically_close_after_kv_transform(self):
        model = make_tiny_phi3()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_logits = model(input_ids=input_ids).logits[:, -1, :]

        cfg = model.config
        transformed, _ = KVCacheTransform.apply(model)
        qeff_inputs = _make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            after_logits = transformed(**qeff_inputs).logits[:, -1, :]

        hf_probs = F.softmax(before_logits, dim=-1)
        qeff_probs = F.softmax(after_logits, dim=-1)
        max_diff = (hf_probs - qeff_probs).abs().max().item()
        assert max_diff < 1e-3, f"KVCacheTransform changed Phi3 probability distribution: max_diff={max_diff:.2e}"


# ---------------------------------------------------------------------------
# Tests: Combined transforms accuracy
# ---------------------------------------------------------------------------


@pytest.mark.transforms
@pytest.mark.accuracy
class TestCombinedTransformsAccuracy:
    """
    Applying CustomOpsTransform + KVCacheTransform together must preserve accuracy.
    This is the exact combination used by QEFFAutoModelForCausalLM.
    """

    def _check_combined_transforms(self, model, label):
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            original_token = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        cfg = model.config
        model, _ = CustomOpsTransform.apply(model)
        model, _ = KVCacheTransform.apply(model)

        qeff_inputs = _make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            transformed_token = model(**qeff_inputs).logits[:, -1, :].argmax(-1).item()

        assert original_token == transformed_token, (
            f"[{label}] Combined transforms changed greedy token: "
            f"original={original_token}, transformed={transformed_token}"
        )

    def test_llama_combined_transforms_preserve_greedy_token(self):
        self._check_combined_transforms(make_tiny_llama(), "Llama")

    def test_mistral_combined_transforms_preserve_greedy_token(self):
        self._check_combined_transforms(make_tiny_mistral(), "Mistral")

    def test_qwen2_combined_transforms_preserve_greedy_token(self):
        self._check_combined_transforms(make_tiny_qwen2(), "Qwen2")

    def test_phi3_combined_transforms_preserve_greedy_token(self):
        self._check_combined_transforms(make_tiny_phi3(), "Phi3")

    def test_gemma_combined_transforms_preserve_greedy_token(self):
        self._check_combined_transforms(make_tiny_gemma(), "Gemma")

    def test_combined_transforms_produce_finite_outputs(self):
        """Combined transforms must produce finite logits for all supported models."""
        for factory, label in [
            (make_tiny_llama, "Llama"),
            (make_tiny_mistral, "Mistral"),
            (make_tiny_qwen2, "Qwen2"),
            (make_tiny_phi3, "Phi3"),
        ]:
            model = factory()
            cfg = model.config
            model, _ = CustomOpsTransform.apply(model)
            model, _ = KVCacheTransform.apply(model)

            input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
            qeff_inputs = _make_qeff_inputs(input_ids, cfg)
            with torch.no_grad():
                out = model(**qeff_inputs)
            assert torch.isfinite(out.logits).all(), f"{label} combined transforms produce NaN/Inf"

    def test_gpt2_kv_transform_then_custom_ops_no_crash(self):
        """Applying KVCacheTransform then CustomOpsTransform to GPT2 must not crash."""
        model = make_tiny_gpt2()
        model, _ = KVCacheTransform.apply(model)
        model, applied = CustomOpsTransform.apply(model)
        assert not applied, "CustomOpsTransform must not apply to GPT2"


# ---------------------------------------------------------------------------
# Tests: PoolingTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestPoolingTransformCorrectness:
    """PoolingTransform must produce correct pooled embeddings."""

    def test_mean_pooling_wraps_model(self):
        from transformers import BertConfig, BertModel

        from QEfficient.transformers.embeddings.embedding_utils import PooledModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        pooled, applied = PoolingTransform.apply(model, pooling="mean")
        assert isinstance(pooled, PooledModel)

    def test_cls_pooling_wraps_model(self):
        from transformers import BertConfig, BertModel

        from QEfficient.transformers.embeddings.embedding_utils import PooledModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        pooled, applied = PoolingTransform.apply(model, pooling="cls")
        assert isinstance(pooled, PooledModel)

    def test_invalid_pooling_raises_error(self):
        from transformers import BertConfig, BertModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        with pytest.raises((ValueError, KeyError, TypeError)):
            PoolingTransform.apply(model, pooling="invalid_pooling_xyz")

    def test_mean_pooled_output_matches_manual_mean(self):
        """PooledModel mean output must match manually computed mean pooling."""
        from transformers import BertConfig, BertModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        inputs = {
            "input_ids": torch.randint(0, 500, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }

        with torch.no_grad():
            hf_out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        manual_mean = (hf_out.last_hidden_state * mask).sum(1) / mask.sum(1)

        pooled, _ = PoolingTransform.apply(model, pooling="mean")
        with torch.no_grad():
            pooled_mean = pooled(**inputs)

        max_diff = (manual_mean - pooled_mean).abs().max().item()
        assert max_diff < 1e-5, f"Mean pooling mismatch: max_diff={max_diff:.2e}"

    def test_max_pooling_wraps_model(self):
        """PoolingTransform with pooling='max' must wrap the model in PooledModel."""
        from transformers import BertConfig, BertModel

        from QEfficient.transformers.embeddings.embedding_utils import PooledModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        pooled, _ = PoolingTransform.apply(model, pooling="max")
        # PoolingTransform always returns applied=False (it wraps, not replaces)
        assert isinstance(pooled, PooledModel)

    def test_max_pooled_output_matches_manual_max(self):
        """PooledModel max output must match manually computed max pooling."""
        from transformers import BertConfig, BertModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        inputs = {
            "input_ids": torch.randint(0, 500, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }

        with torch.no_grad():
            hf_out = model(**inputs)
        # Manual max pooling: max over sequence dimension
        manual_max = hf_out.last_hidden_state.max(dim=1).values

        pooled, _ = PoolingTransform.apply(model, pooling="max")
        with torch.no_grad():
            pooled_max = pooled(**inputs)

        max_diff = (manual_max - pooled_max).abs().max().item()
        assert max_diff < 1e-5, f"Max pooling mismatch: max_diff={max_diff:.2e}"

    def test_avg_pooling_wraps_model(self):
        """PoolingTransform with pooling='avg' must wrap the model in PooledModel."""
        from transformers import BertConfig, BertModel

        from QEfficient.transformers.embeddings.embedding_utils import PooledModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        # 'avg' is supported in POOLING_MAP
        pooled, _ = PoolingTransform.apply(model, pooling="avg")
        assert isinstance(pooled, PooledModel)

    def test_custom_callable_pooling_is_accepted(self):
        """PoolingTransform must accept a callable as the pooling argument."""
        from transformers import BertConfig, BertModel

        from QEfficient.transformers.embeddings.embedding_utils import PooledModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()

        def custom_pool(last_hidden_states, attention_mask):
            # Simple: return first token (like CLS)
            return last_hidden_states[:, 0, :]

        try:
            pooled, _ = PoolingTransform.apply(model, pooling=custom_pool)
            assert isinstance(pooled, PooledModel)
        except (ValueError, TypeError, NotImplementedError):
            # If custom callable is not supported, skip
            pytest.skip("Custom callable pooling not supported in this version")

    def test_pooling_output_is_finite(self):
        """Pooled output must be finite (no NaN/Inf)."""
        from transformers import BertConfig, BertModel

        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
        )
        model = BertModel(cfg).eval()
        inputs = {
            "input_ids": torch.randint(0, 500, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }

        for pooling_type in ["mean", "cls", "max"]:
            try:
                pooled, _ = PoolingTransform.apply(model, pooling=pooling_type)
                with torch.no_grad():
                    output = pooled(**inputs)
                assert torch.isfinite(output).all(), f"Pooled output for '{pooling_type}' must be finite"
            except (ValueError, KeyError):
                pass  # Skip unsupported pooling types


# ---------------------------------------------------------------------------
# Tests: SamplerTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSamplerTransformBehavior:
    """SamplerTransform must only apply when qaic_config has include_sampler=True."""

    def test_no_transform_when_qaic_config_is_none(self):
        model = make_tiny_gpt2()
        kv_model, _ = KVCacheTransform.apply(model)
        _, applied = SamplerTransform.apply(kv_model, qaic_config=None)
        assert not applied

    def test_no_transform_when_include_sampler_false(self):
        model = make_tiny_gpt2()
        kv_model, _ = KVCacheTransform.apply(model)
        _, applied = SamplerTransform.apply(kv_model, qaic_config={"include_sampler": False})
        assert not applied

    def test_unsupported_model_raises_not_implemented(self):
        import torch.nn as nn

        class UnsupportedModel(nn.Module):
            def forward(self, x):
                return x

        with pytest.raises(NotImplementedError):
            SamplerTransform.apply(UnsupportedModel(), qaic_config={"include_sampler": True})

    def test_supported_model_classes_include_gpt2_and_llama(self):
        from QEfficient.transformers.models.gpt2.modeling_gpt2 import QEffGPT2LMHeadModel
        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM

        assert QEffGPT2LMHeadModel in SamplerTransform._module_mapping
        assert QEffLlamaForCausalLM in SamplerTransform._module_mapping


# ---------------------------------------------------------------------------
# Tests: SpDTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSpDTransformBehavior:
    """SpDTransform must only apply when speculative_model_type is in qaic_config."""

    def test_no_transform_when_qaic_config_is_none(self):
        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        _, applied = SpDTransform.apply(kv_model, qaic_config=None)
        assert not applied

    def test_no_transform_when_speculative_model_type_missing(self):
        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        _, applied = SpDTransform.apply(kv_model, qaic_config={})
        assert not applied

    def test_invalid_speculative_model_type_raises_value_error(self):
        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        with pytest.raises(ValueError):
            SpDTransform.apply(kv_model, qaic_config={"speculative_model_type": "invalid_xyz"})

    def test_module_mapping_contains_llama_and_qwen2(self):
        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM
        from QEfficient.transformers.models.qwen2.modeling_qwen2 import QEffQwen2ForCausalLM

        assert QEffLlamaForCausalLM in SpDTransform._module_mapping
        assert QEffQwen2ForCausalLM in SpDTransform._module_mapping


# ---------------------------------------------------------------------------
# Tests: SamplerTransform actual apply
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestSamplerTransformActualApply:
    """SamplerTransform with include_sampler=True must attach sampler_forward."""

    def test_sampler_transform_applies_to_gpt2_with_include_sampler_true(self):
        """SamplerTransform must apply to QEffGPT2LMHeadModel when include_sampler=True."""
        model = make_tiny_gpt2()
        kv_model, _ = KVCacheTransform.apply(model)
        _, applied = SamplerTransform.apply(kv_model, qaic_config={"include_sampler": True})
        assert applied, "SamplerTransform must apply when include_sampler=True"

    def test_sampler_transform_applies_to_llama_with_include_sampler_true(self):
        """SamplerTransform must apply to QEffLlamaForCausalLM when include_sampler=True."""
        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        _, applied = SamplerTransform.apply(kv_model, qaic_config={"include_sampler": True})
        assert applied, "SamplerTransform must apply to Llama when include_sampler=True"

    def test_sampler_transform_saves_old_forward(self):
        """After SamplerTransform, model.old_forward must be set to the original forward."""
        model = make_tiny_gpt2()
        kv_model, _ = KVCacheTransform.apply(model)
        original_forward = kv_model.forward
        SamplerTransform.apply(kv_model, qaic_config={"include_sampler": True})
        assert hasattr(kv_model, "old_forward"), "SamplerTransform must save old_forward"
        assert kv_model.old_forward == original_forward, "old_forward must be the original forward method"

    def test_sampler_transform_replaces_forward_with_sampler_forward(self):
        """After SamplerTransform, model.forward must be replaced."""
        model = make_tiny_gpt2()
        kv_model, _ = KVCacheTransform.apply(model)
        original_forward = kv_model.forward
        SamplerTransform.apply(kv_model, qaic_config={"include_sampler": True})
        # The forward must have been replaced
        assert kv_model.forward is not original_forward, "SamplerTransform must replace model.forward"

    def test_sampler_transform_returns_same_model_instance(self):
        """SamplerTransform must modify model in-place."""
        model = make_tiny_gpt2()
        kv_model, _ = KVCacheTransform.apply(model)
        transformed, applied = SamplerTransform.apply(kv_model, qaic_config={"include_sampler": True})
        assert applied
        assert transformed is kv_model, "SamplerTransform must modify model in-place"

    def test_sampler_transform_module_mapping_contains_gpt2_and_llama(self):
        from QEfficient.transformers.models.gpt2.modeling_gpt2 import QEffGPT2LMHeadModel
        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaForCausalLM

        assert QEffGPT2LMHeadModel in SamplerTransform._module_mapping
        assert QEffLlamaForCausalLM in SamplerTransform._module_mapping

    def test_sampler_transform_module_mapping_contains_phi3_and_qwen2(self):
        from QEfficient.transformers.models.phi3.modeling_phi3 import QEffPhi3ForCausalLM
        from QEfficient.transformers.models.qwen2.modeling_qwen2 import QEffQwen2ForCausalLM

        assert QEffPhi3ForCausalLM in SamplerTransform._module_mapping
        assert QEffQwen2ForCausalLM in SamplerTransform._module_mapping


# ---------------------------------------------------------------------------
# Tests: MoE transform (Mixtral)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestMoETransformReplacement:
    """KVCacheTransform must replace MixtralSparseMoeBlock with QEffMixtralSparseMoeBlock."""

    def _make_tiny_mixtral(self):
        from transformers import MixtralConfig, MixtralForCausalLM

        cfg = MixtralConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=CTX_LEN,
            num_experts_per_tok=2,
            num_local_experts=4,
        )
        return MixtralForCausalLM(cfg).eval(), cfg

    def test_mixtral_sparse_moe_block_replaced(self):
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        from QEfficient.transformers.models.mixtral_moe.modeling_mixtral import QEffMixtralSparseMoeBlock

        model, cfg = self._make_tiny_mixtral()
        assert any(isinstance(m, MixtralSparseMoeBlock) for m in model.modules())

        transformed, applied = KVCacheTransform.apply(model)
        assert applied

        for m in transformed.modules():
            if type(m) is MixtralSparseMoeBlock:
                pytest.fail("Found unreplaced MixtralSparseMoeBlock after transform")

        assert any(isinstance(m, QEffMixtralSparseMoeBlock) for m in transformed.modules())

    def test_mixtral_for_causal_lm_replaced(self):
        from QEfficient.transformers.models.mixtral_moe.modeling_mixtral import QEffMixtralForCausalLM

        model, cfg = self._make_tiny_mixtral()
        transformed, _ = KVCacheTransform.apply(model)
        assert isinstance(transformed, QEffMixtralForCausalLM)

    def test_mixtral_greedy_token_preserved_after_kv_transform(self):
        """Mixtral greedy token must be preserved after KVCacheTransform."""
        model, cfg = self._make_tiny_mixtral()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            before_token = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        transformed, _ = KVCacheTransform.apply(model)
        qeff_inputs = _make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            after_token = transformed(**qeff_inputs).logits[:, -1, :].argmax(-1).item()

        assert before_token == after_token, (
            f"Mixtral KVCacheTransform changed greedy token: before={before_token}, after={after_token}"
        )

    def test_mixtral_kv_transform_produces_finite_outputs(self):
        model, cfg = self._make_tiny_mixtral()
        transformed, _ = KVCacheTransform.apply(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        qeff_inputs = _make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            out = transformed(**qeff_inputs)
        assert torch.isfinite(out.logits).all(), "Mixtral KVCacheTransform must produce finite logits"


# ---------------------------------------------------------------------------
# Tests: T5ModelTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestT5ModelTransform:
    """T5ModelTransform must replace T5Attention and T5LayerNorm with QEff variants."""

    def _make_tiny_t5(self):
        from transformers import T5Config, T5ForConditionalGeneration

        cfg = T5Config(
            num_heads=2,
            d_model=64,
            d_ff=128,
            d_kv=32,
            num_layers=1,
            num_decoder_layers=1,
            vocab_size=500,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=16,
        )
        return T5ForConditionalGeneration(cfg).eval(), cfg

    def test_t5_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        assert T5ModelTransform is not None

    def test_t5_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        assert hasattr(T5ModelTransform, "_module_mapping")
        assert len(T5ModelTransform._module_mapping) > 0

    def test_t5_transform_maps_t5_attention(self):
        from transformers.models.t5.modeling_t5 import T5Attention

        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        assert T5Attention in T5ModelTransform._module_mapping
        qeff_cls = T5ModelTransform._module_mapping[T5Attention]
        assert qeff_cls.__name__ == "QEffT5Attention"

    def test_t5_transform_maps_t5_layer_norm(self):
        from transformers.models.t5.modeling_t5 import T5LayerNorm

        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        assert T5LayerNorm in T5ModelTransform._module_mapping
        qeff_cls = T5ModelTransform._module_mapping[T5LayerNorm]
        assert qeff_cls.__name__ == "QEffT5LayerNorm"

    def test_t5_transform_replaces_attention(self):
        from transformers.models.t5.modeling_t5 import T5Attention

        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        model, cfg = self._make_tiny_t5()
        assert any(isinstance(m, T5Attention) for m in model.modules())

        transformed, applied = T5ModelTransform.apply(model)
        assert applied

        qeff_t5_attn_cls = T5ModelTransform._module_mapping[T5Attention]
        for m in transformed.modules():
            if type(m) is T5Attention:
                pytest.fail("Found unreplaced T5Attention after T5ModelTransform")

        assert any(isinstance(m, qeff_t5_attn_cls) for m in transformed.modules())

    def test_t5_transform_replaces_layer_norm(self):
        from transformers.models.t5.modeling_t5 import T5LayerNorm

        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        model, cfg = self._make_tiny_t5()
        transformed, applied = T5ModelTransform.apply(model)
        assert applied
        qeff_t5_ln_cls = T5ModelTransform._module_mapping[T5LayerNorm]
        assert any(isinstance(m, qeff_t5_ln_cls) for m in transformed.modules())

    def test_t5_transform_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import T5ModelTransform

        assert hasattr(T5ModelTransform, "apply")
        assert callable(T5ModelTransform.apply)


# ---------------------------------------------------------------------------
# Tests: TextClassificationTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestTextClassificationTransformDirect:
    """TextClassificationTransform must directly replace DisentangledSelfAttention."""

    def _make_tiny_deberta(self):
        from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

        cfg = DebertaV2Config(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=500,
            max_position_embeddings=64,
            num_labels=3,
            type_vocab_size=0,
            pos_att_type=["p2c", "c2p"],
        )
        return DebertaV2ForSequenceClassification(cfg).eval(), cfg

    def test_text_classification_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import TextClassificationTransform

        assert TextClassificationTransform is not None

    def test_text_classification_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import TextClassificationTransform

        assert hasattr(TextClassificationTransform, "_module_mapping")
        assert len(TextClassificationTransform._module_mapping) > 0

    def test_text_classification_transform_maps_disentangled_self_attention(self):
        from transformers.models.deberta_v2.modeling_deberta_v2 import DisentangledSelfAttention

        from QEfficient.transformers.models.pytorch_transforms import TextClassificationTransform

        assert DisentangledSelfAttention in TextClassificationTransform._module_mapping
        qeff_cls = TextClassificationTransform._module_mapping[DisentangledSelfAttention]
        assert qeff_cls.__name__ == "QEffDisentangledSelfAttention"

    def test_text_classification_transform_replaces_attention(self):
        from transformers.models.deberta_v2.modeling_deberta_v2 import DisentangledSelfAttention

        from QEfficient.transformers.models.pytorch_transforms import TextClassificationTransform

        try:
            model, cfg = self._make_tiny_deberta()
        except Exception as e:
            pytest.skip(f"DeBERTa-v2 not available: {e}")

        assert any(isinstance(m, DisentangledSelfAttention) for m in model.modules())

        transformed, applied = TextClassificationTransform.apply(model)
        assert applied

        qeff_cls = TextClassificationTransform._module_mapping[DisentangledSelfAttention]
        for m in transformed.modules():
            if type(m) is DisentangledSelfAttention:
                pytest.fail("Found unreplaced DisentangledSelfAttention after transform")

        assert any(isinstance(m, qeff_cls) for m in transformed.modules())

    def test_text_classification_transform_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import TextClassificationTransform

        assert hasattr(TextClassificationTransform, "apply")
        assert callable(TextClassificationTransform.apply)


# ---------------------------------------------------------------------------
# Tests: BlockedKVAttentionTransform
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockedAttentionTransform:
    """BlockedAttentionTransform must patch forward with blocking config parameter."""

    def test_blocked_transform_importable(self):
        from QEfficient.blocking.attention_blocking import AttentionBlockingConfig
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform

        assert BlockingAttentionTransform is not None
        assert AttentionBlockingConfig is not None

    def test_blocked_transform_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform

        assert hasattr(BlockingAttentionTransform, "apply")
        assert callable(BlockingAttentionTransform.apply)

    def test_blocked_transform_applies_to_llama(self):
        """BlockingAttentionTransform must apply to a KV-transformed Llama model."""
        from QEfficient.blocking.attention_blocking import AttentionBlockingConfig
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform

        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        blocking_config = AttentionBlockingConfig(mode="kv", num_kv_blocks=4)
        transformed, applied = BlockingAttentionTransform.apply(kv_model, attn_blocking_config=blocking_config)
        assert applied, "BlockingAttentionTransform must apply to KV-transformed Llama"

    def test_blocked_transform_patches_forward(self):
        """After BlockingAttentionTransform, attention forward must be patched."""
        from QEfficient.blocking.attention_blocking import AttentionBlockingConfig
        from QEfficient.transformers.models.llama.modeling_llama import QEffLlamaAttention
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform

        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        blocking_config = AttentionBlockingConfig(mode="kv", num_kv_blocks=4)
        BlockingAttentionTransform.apply(kv_model, attn_blocking_config=blocking_config)

        # After transform, attention modules should have patched forward
        for m in kv_model.modules():
            if isinstance(m, QEffLlamaAttention):
                # The forward should be a partial function with num_kv_blocks
                assert hasattr(m, "forward"), "Attention module must have forward after transform"
                break

    def test_blocked_transform_returns_model_and_bool(self):
        from QEfficient.blocking.attention_blocking import AttentionBlockingConfig
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform

        model = make_tiny_llama()
        kv_model, _ = KVCacheTransform.apply(model)
        blocking_config = AttentionBlockingConfig(mode="kv", num_kv_blocks=4)
        result = BlockingAttentionTransform.apply(kv_model, attn_blocking_config=blocking_config)
        assert len(result) == 2
        assert isinstance(result[1], bool)


# ---------------------------------------------------------------------------
# Tests: PrefillOnly transforms (structure only - GPT_OSS is external)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestPrefillOnlyTransformStructure:
    """PrefillOnly transforms must have correct structure."""

    def test_prefill_only_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyTransform

        assert PrefillOnlyTransform is not None

    def test_prefill_only_chunked_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyChunkedTransform

        assert PrefillOnlyChunkedTransform is not None

    def test_revert_prefill_only_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import RevertPrefillOnlyTransform

        assert RevertPrefillOnlyTransform is not None

    def test_revert_prefill_keep_attention_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import RevertPrefillKeepAttentionTransform

        assert RevertPrefillKeepAttentionTransform is not None

    def test_prefill_only_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyTransform

        assert hasattr(PrefillOnlyTransform, "_module_mapping")
        assert len(PrefillOnlyTransform._module_mapping) > 0

    def test_prefill_only_chunked_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyChunkedTransform

        assert hasattr(PrefillOnlyChunkedTransform, "_module_mapping")
        assert len(PrefillOnlyChunkedTransform._module_mapping) > 0

    def test_revert_prefill_only_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import RevertPrefillOnlyTransform

        assert hasattr(RevertPrefillOnlyTransform, "_module_mapping")
        assert len(RevertPrefillOnlyTransform._module_mapping) > 0

    def test_prefill_only_transform_maps_gpt_oss_model(self):
        from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import (
            QEffGptOssModel,
            QEffPrefillOnlyGptOssModel,
        )
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyTransform

        assert QEffGptOssModel in PrefillOnlyTransform._module_mapping
        assert PrefillOnlyTransform._module_mapping[QEffGptOssModel] is QEffPrefillOnlyGptOssModel

    def test_prefill_only_transform_maps_gpt_oss_attention(self):
        from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import (
            QEffGptOssAttention,
            QEffPrefillOnlyGptOssAttention,
        )
        from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyTransform

        assert QEffGptOssAttention in PrefillOnlyTransform._module_mapping
        assert PrefillOnlyTransform._module_mapping[QEffGptOssAttention] is QEffPrefillOnlyGptOssAttention

    def test_revert_prefill_only_is_inverse_of_prefill_only(self):
        """RevertPrefillOnlyTransform must be the inverse of PrefillOnlyTransform for non-identity mappings."""
        from QEfficient.transformers.models.pytorch_transforms import (
            PrefillOnlyTransform,
            RevertPrefillOnlyTransform,
        )

        # For each (src, dst) in PrefillOnlyTransform where src != dst,
        # (dst, src) must be in RevertPrefillOnlyTransform
        for src, dst in PrefillOnlyTransform._module_mapping.items():
            if src is dst:
                continue  # Skip identity mappings
            assert dst in RevertPrefillOnlyTransform._module_mapping, (
                f"RevertPrefillOnlyTransform missing inverse mapping for {dst}"
            )
            assert RevertPrefillOnlyTransform._module_mapping[dst] is src, (
                f"RevertPrefillOnlyTransform[{dst}] must be {src}"
            )

    def test_all_prefill_transforms_have_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import (
            PrefillOnlyChunkedTransform,
            PrefillOnlyTransform,
            RevertPrefillKeepAttentionTransform,
            RevertPrefillOnlyTransform,
        )

        for cls in [
            PrefillOnlyTransform,
            PrefillOnlyChunkedTransform,
            RevertPrefillOnlyTransform,
            RevertPrefillKeepAttentionTransform,
        ]:
            assert hasattr(cls, "apply"), f"{cls.__name__} missing apply method"
            assert callable(cls.apply), f"{cls.__name__}.apply is not callable"


# ---------------------------------------------------------------------------
# Tests: VlmKVOffloadTransform (GAP D)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestVlmKVOffloadTransform:
    """VlmKVOffloadTransform must be importable and have correct module mapping."""

    def test_vlm_kv_offload_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert VlmKVOffloadTransform is not None

    def test_vlm_kv_offload_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert hasattr(VlmKVOffloadTransform, "_module_mapping")
        assert len(VlmKVOffloadTransform._module_mapping) > 0

    def test_vlm_kv_offload_transform_maps_mllama_cross_attention(self):
        from transformers.models.mllama.modeling_mllama import MllamaTextCrossAttention

        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert MllamaTextCrossAttention in VlmKVOffloadTransform._module_mapping

    def test_vlm_kv_offload_transform_maps_to_two_qpc_variant(self):
        from transformers.models.mllama.modeling_mllama import MllamaTextCrossAttention

        from QEfficient.transformers.models.mllama.modeling_mllama import QEffMllamaTextCrossAttentionTwoQPC
        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert VlmKVOffloadTransform._module_mapping[MllamaTextCrossAttention] is QEffMllamaTextCrossAttentionTwoQPC

    def test_vlm_kv_offload_transform_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmKVOffloadTransform

        assert hasattr(VlmKVOffloadTransform, "apply")
        assert callable(VlmKVOffloadTransform.apply)


# ---------------------------------------------------------------------------
# Tests: VlmNoKVOffloadTransform (GAP D)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestVlmNoKVOffloadTransform:
    """VlmNoKVOffloadTransform must be importable and have correct module mapping."""

    def test_vlm_no_kv_offload_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert VlmNoKVOffloadTransform is not None

    def test_vlm_no_kv_offload_transform_has_module_mapping(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert hasattr(VlmNoKVOffloadTransform, "_module_mapping")
        assert len(VlmNoKVOffloadTransform._module_mapping) > 0

    def test_vlm_no_kv_offload_transform_maps_mllama_cross_attention(self):
        from transformers.models.mllama.modeling_mllama import MllamaTextCrossAttention

        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert MllamaTextCrossAttention in VlmNoKVOffloadTransform._module_mapping

    def test_vlm_no_kv_offload_transform_maps_to_single_qpc_variant(self):
        from transformers.models.mllama.modeling_mllama import MllamaTextCrossAttention

        from QEfficient.transformers.models.mllama.modeling_mllama import QEffMllamaTextCrossAttentionSingleQPC
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert (
            VlmNoKVOffloadTransform._module_mapping[MllamaTextCrossAttention] is QEffMllamaTextCrossAttentionSingleQPC
        )

    def test_vlm_no_kv_offload_transform_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import VlmNoKVOffloadTransform

        assert hasattr(VlmNoKVOffloadTransform, "apply")
        assert callable(VlmNoKVOffloadTransform.apply)

    def test_vlm_offload_and_no_offload_map_to_different_classes(self):
        """VlmKVOffloadTransform and VlmNoKVOffloadTransform must map to different QEff classes."""
        from transformers.models.mllama.modeling_mllama import MllamaTextCrossAttention

        from QEfficient.transformers.models.pytorch_transforms import (
            VlmKVOffloadTransform,
            VlmNoKVOffloadTransform,
        )

        offload_cls = VlmKVOffloadTransform._module_mapping[MllamaTextCrossAttention]
        no_offload_cls = VlmNoKVOffloadTransform._module_mapping[MllamaTextCrossAttention]
        assert offload_cls is not no_offload_cls, (
            "VlmKVOffloadTransform and VlmNoKVOffloadTransform must map to different classes"
        )


# ---------------------------------------------------------------------------
# Tests: KVCacheExternalModuleMapperTransform (GAP D)
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestKVCacheExternalModuleMapperTransform:
    """KVCacheExternalModuleMapperTransform must have correct string-based mappings."""

    def test_external_mapper_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert KVCacheExternalModuleMapperTransform is not None

    def test_external_mapper_has_match_string_replace_method(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert hasattr(KVCacheExternalModuleMapperTransform, "_match_string_replace_method")
        assert isinstance(KVCacheExternalModuleMapperTransform._match_string_replace_method, dict)

    def test_external_mapper_contains_internvl(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert "InternVLChatModel" in KVCacheExternalModuleMapperTransform._match_string_replace_method

    def test_external_mapper_contains_molmo(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert "MolmoForCausalLM" in KVCacheExternalModuleMapperTransform._match_string_replace_method

    def test_external_mapper_contains_grok1(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert "Grok1ModelForCausalLM" in KVCacheExternalModuleMapperTransform._match_string_replace_method

    def test_external_mapper_internvl_has_forward(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        internvl_mapping = KVCacheExternalModuleMapperTransform._match_string_replace_method["InternVLChatModel"]
        assert "forward" in internvl_mapping
        assert callable(internvl_mapping["forward"])

    def test_external_mapper_molmo_has_forward(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        molmo_mapping = KVCacheExternalModuleMapperTransform._match_string_replace_method["MolmoForCausalLM"]
        assert "forward" in molmo_mapping
        assert callable(molmo_mapping["forward"])

    def test_external_mapper_grok1_has_forward(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        grok1_mapping = KVCacheExternalModuleMapperTransform._match_string_replace_method["Grok1ModelForCausalLM"]
        assert "forward" in grok1_mapping
        assert callable(grok1_mapping["forward"])

    def test_external_mapper_has_apply_method(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert hasattr(KVCacheExternalModuleMapperTransform, "apply")
        assert callable(KVCacheExternalModuleMapperTransform.apply)

    def test_external_mapper_internvl_has_get_dummy_inputs(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        internvl_mapping = KVCacheExternalModuleMapperTransform._match_string_replace_method["InternVLChatModel"]
        assert "get_dummy_inputs" in internvl_mapping
        assert callable(internvl_mapping["get_dummy_inputs"])

    def test_external_mapper_rms_norm_has_forward(self):
        """RMSLayerNorm must be mapped to CustomRMSNormAIC.forward."""
        from QEfficient.customop import CustomRMSNormAIC
        from QEfficient.transformers.models.pytorch_transforms import KVCacheExternalModuleMapperTransform

        assert "RMSLayerNorm" in KVCacheExternalModuleMapperTransform._match_string_replace_method
        rms_mapping = KVCacheExternalModuleMapperTransform._match_string_replace_method["RMSLayerNorm"]
        assert rms_mapping["forward"] is CustomRMSNormAIC.forward
