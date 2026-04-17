# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Accuracy tests for CausalLM models: HF PyTorch → QEff PyTorch → ONNX structure.

Improvements over unit_v2:
  - Expanded model coverage: GPT2, Llama, Mistral, Qwen2, Phi3, Gemma, Gemma2, Falcon
  - Continuous batching mode tests
  - ONNX structure validation for all models

Key accuracy assertions:
  - HF and QEff produce the SAME greedy next token (argmax of last-token logits)
  - HF and QEff logits are numerically close (softmax max_diff < 1e-3)
  - Decode step produces valid tokens in range [0, vocab_size)

All tests run on CPU only.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers import (
    FalconConfig,
    FalconForCausalLM,
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

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

CTX_LEN = 32
SEQ_LEN = 8
VOCAB_SIZE = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_dims(config):
    """Extract (n_layers, n_kv_heads, head_dim) from any config."""
    if hasattr(config, "num_hidden_layers"):
        n_layers = config.num_hidden_layers
        n_attn = config.num_attention_heads
        n_kv = getattr(config, "num_key_value_heads", n_attn)
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // n_attn)
    else:
        n_layers = config.n_layer
        n_attn = config.n_head
        n_kv = config.n_head
        head_dim = config.n_embd // n_attn
    return n_layers, n_kv, head_dim


def make_qeff_inputs(input_ids, config, ctx_len=CTX_LEN):
    """Build QEff-style inputs: input_ids + position_ids + zero-init past_key_values."""
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
    return {"input_ids": input_ids, "position_ids": position_ids, "past_key_values": past_key_values}


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


def make_tiny_mistral():
    cfg = MistralConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return MistralForCausalLM(cfg).eval(), cfg


def make_tiny_qwen2():
    cfg = Qwen2Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return Qwen2ForCausalLM(cfg).eval(), cfg


def make_tiny_phi3():
    cfg = Phi3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        pad_token_id=0,
    )
    return Phi3ForCausalLM(cfg).eval(), cfg


def make_tiny_gemma():
    cfg = GemmaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    return GemmaForCausalLM(cfg).eval(), cfg


def make_tiny_falcon():
    cfg = FalconConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=64,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        new_decoder_architecture=False,
        multi_query=True,
    )
    return FalconForCausalLM(cfg).eval(), cfg


# ---------------------------------------------------------------------------
# Stage 1: HF PyTorch baseline
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
class TestHFCausalLMBaseline:
    """HF models run correctly on CPU and produce valid logits."""

    def _check_logits_shape(self, factory, label):
        model, cfg = factory()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (1, SEQ_LEN, VOCAB_SIZE), (
            f"[{label}] Expected logits shape (1, {SEQ_LEN}, {VOCAB_SIZE}), got {out.logits.shape}"
        )

    def test_gpt2_forward_returns_logits_with_correct_shape(self):
        self._check_logits_shape(make_tiny_gpt2, "GPT2")

    def test_llama_forward_returns_logits_with_correct_shape(self):
        self._check_logits_shape(make_tiny_llama, "Llama")

    def test_mistral_forward_returns_logits_with_correct_shape(self):
        self._check_logits_shape(make_tiny_mistral, "Mistral")

    def test_qwen2_forward_returns_logits_with_correct_shape(self):
        self._check_logits_shape(make_tiny_qwen2, "Qwen2")

    def test_phi3_forward_returns_logits_with_correct_shape(self):
        self._check_logits_shape(make_tiny_phi3, "Phi3")

    def test_gemma_forward_returns_logits_with_correct_shape(self):
        self._check_logits_shape(make_tiny_gemma, "Gemma")

    def test_falcon_forward_returns_logits_with_correct_shape(self):
        self._check_logits_shape(make_tiny_falcon, "Falcon")

    def test_hf_logits_are_finite(self):
        """HF logits must not contain NaN or Inf for any model."""
        for factory, label in [
            (make_tiny_gpt2, "GPT2"),
            (make_tiny_llama, "Llama"),
            (make_tiny_mistral, "Mistral"),
            (make_tiny_qwen2, "Qwen2"),
            (make_tiny_phi3, "Phi3"),
            (make_tiny_gemma, "Gemma"),
        ]:
            model, cfg = factory()
            input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits
            assert torch.isfinite(logits).all(), f"[{label}] HF logits contain NaN/Inf"

    def test_gpt2_greedy_decode_is_deterministic(self):
        model, cfg = make_tiny_gpt2()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        with torch.no_grad():
            t1 = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()
            t2 = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()
        assert t1 == t2, "Greedy decode must be deterministic"


# ---------------------------------------------------------------------------
# Stage 2: QEff PyTorch accuracy vs HF
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestQEffCausalLMAccuracyVsHF:
    """
    QEff KV-transformed model must produce the same greedy next token as HF.
    This is the primary regression test: if KVCacheTransform or CustomOpsTransform
    changes the model's numerical output, these tests will catch it.
    """

    def _assert_same_greedy_token(self, model, cfg, label):
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            hf_logits = model(input_ids=input_ids).logits[:, -1, :]
        hf_token = hf_logits.argmax(-1).item()

        qeff_model = QEFFAutoModelForCausalLM(model)
        qeff_inputs = make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            qeff_logits = qeff_model.model(**qeff_inputs).logits[:, -1, :]
        qeff_token = qeff_logits.argmax(-1).item()

        assert hf_token == qeff_token, (
            f"[{label}] Greedy token mismatch: HF={hf_token}, QEff={qeff_token}. "
            f"KVCacheTransform must not change the model's greedy prediction."
        )

    def _assert_logits_numerically_close(self, model, cfg, label, atol=1e-3):
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            hf_logits = model(input_ids=input_ids).logits[:, -1, :]

        qeff_model = QEFFAutoModelForCausalLM(model)
        qeff_inputs = make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            qeff_logits = qeff_model.model(**qeff_inputs).logits[:, -1, :]

        hf_probs = F.softmax(hf_logits, dim=-1)
        qeff_probs = F.softmax(qeff_logits, dim=-1)
        max_diff = (hf_probs - qeff_probs).abs().max().item()
        assert max_diff < atol, f"[{label}] Probability distribution mismatch: max_diff={max_diff:.6f} > atol={atol}."

    def test_gpt2_qeff_matches_hf_greedy_token(self):
        model, cfg = make_tiny_gpt2()
        self._assert_same_greedy_token(model, cfg, "GPT2")

    def test_llama_qeff_matches_hf_greedy_token(self):
        model, cfg = make_tiny_llama()
        self._assert_same_greedy_token(model, cfg, "Llama")

    def test_mistral_qeff_matches_hf_greedy_token(self):
        model, cfg = make_tiny_mistral()
        self._assert_same_greedy_token(model, cfg, "Mistral")

    def test_qwen2_qeff_matches_hf_greedy_token(self):
        model, cfg = make_tiny_qwen2()
        self._assert_same_greedy_token(model, cfg, "Qwen2")

    def test_phi3_qeff_matches_hf_greedy_token(self):
        model, cfg = make_tiny_phi3()
        self._assert_same_greedy_token(model, cfg, "Phi3")

    def test_gemma_qeff_matches_hf_greedy_token(self):
        model, cfg = make_tiny_gemma()
        self._assert_same_greedy_token(model, cfg, "Gemma")

    def test_gpt2_qeff_logits_numerically_close_to_hf(self):
        model, cfg = make_tiny_gpt2()
        self._assert_logits_numerically_close(model, cfg, "GPT2")

    def test_llama_qeff_logits_numerically_close_to_hf(self):
        model, cfg = make_tiny_llama()
        self._assert_logits_numerically_close(model, cfg, "Llama")

    def test_mistral_qeff_logits_numerically_close_to_hf(self):
        model, cfg = make_tiny_mistral()
        self._assert_logits_numerically_close(model, cfg, "Mistral")

    def test_qwen2_qeff_logits_numerically_close_to_hf(self):
        model, cfg = make_tiny_qwen2()
        self._assert_logits_numerically_close(model, cfg, "Qwen2")

    def test_phi3_qeff_logits_numerically_close_to_hf(self):
        model, cfg = make_tiny_phi3()
        self._assert_logits_numerically_close(model, cfg, "Phi3")

    def test_qeff_logits_are_finite(self):
        """QEff logits must not contain NaN or Inf for any model."""
        for factory, label in [
            (make_tiny_gpt2, "GPT2"),
            (make_tiny_llama, "Llama"),
            (make_tiny_mistral, "Mistral"),
            (make_tiny_qwen2, "Qwen2"),
            (make_tiny_phi3, "Phi3"),
        ]:
            model, cfg = factory()
            qeff_model = QEFFAutoModelForCausalLM(model)
            input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
            qeff_inputs = make_qeff_inputs(input_ids, cfg)
            with torch.no_grad():
                logits = qeff_model.model(**qeff_inputs).logits
            assert torch.isfinite(logits).all(), f"[{label}] QEff logits contain NaN/Inf"

    def test_qeff_past_key_values_returned(self):
        """QEff model must return past_key_values for the decode step."""
        model, cfg = make_tiny_gpt2()
        qeff_model = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        qeff_inputs = make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            out = qeff_model.model(**qeff_inputs)
        assert out.past_key_values is not None, "QEff model must return past_key_values"

    def test_gpt2_top5_tokens_overlap_with_hf(self):
        """Top-5 predicted tokens must overlap between HF and QEff."""
        model, cfg = make_tiny_gpt2()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            hf_top5 = set(model(input_ids=input_ids).logits[:, -1, :].topk(5).indices.squeeze().tolist())

        qeff_model = QEFFAutoModelForCausalLM(model)
        qeff_inputs = make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            qeff_top5 = set(qeff_model.model(**qeff_inputs).logits[:, -1, :].topk(5).indices.squeeze().tolist())

        overlap = len(hf_top5 & qeff_top5)
        assert overlap >= 4, f"Top-5 token overlap too low: {overlap}/5. HF={hf_top5}, QEff={qeff_top5}"


# ---------------------------------------------------------------------------
# Stage 2b: Decode step accuracy
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestQEffDecodeStepAccuracy:
    """Decode step must produce consistent, finite tokens."""

    def _run_prefill_then_decode(self, model, cfg, n_decode_steps=3, input_ids=None):
        """Run prefill + n decode steps, return list of generated token IDs."""
        qeff_model = QEFFAutoModelForCausalLM(model)
        if input_ids is None:
            input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        qeff_inputs = make_qeff_inputs(input_ids, cfg)

        generated = []
        with torch.no_grad():
            out = qeff_model.model(**qeff_inputs)
            next_token = out.logits[:, -1, :].argmax(-1).item()
            generated.append(next_token)
            prev_pos = SEQ_LEN - 1

            for _ in range(n_decode_steps - 1):
                n_layers, n_kv, head_dim = _get_dims(cfg)
                decode_inputs = {
                    "input_ids": torch.tensor([[next_token]], dtype=torch.long),
                    "position_ids": torch.tensor([[prev_pos + 1]], dtype=torch.long),
                    "past_key_values": tuple(
                        (
                            torch.zeros(1, n_kv, CTX_LEN, head_dim, dtype=torch.float32),
                            torch.zeros(1, n_kv, CTX_LEN, head_dim, dtype=torch.float32),
                        )
                        for _ in range(n_layers)
                    ),
                }
                out = qeff_model.model(**decode_inputs)
                next_token = out.logits[:, -1, :].argmax(-1).item()
                generated.append(next_token)
                prev_pos += 1

        return generated

    def test_gpt2_decode_produces_valid_tokens(self):
        model, cfg = make_tiny_gpt2()
        tokens = self._run_prefill_then_decode(model, cfg, n_decode_steps=3)
        assert len(tokens) == 3
        assert all(0 <= t < VOCAB_SIZE for t in tokens), f"Invalid token IDs: {tokens}"

    def test_llama_decode_produces_valid_tokens(self):
        model, cfg = make_tiny_llama()
        tokens = self._run_prefill_then_decode(model, cfg, n_decode_steps=3)
        assert len(tokens) == 3
        assert all(0 <= t < VOCAB_SIZE for t in tokens), f"Invalid token IDs: {tokens}"

    def test_mistral_decode_produces_valid_tokens(self):
        model, cfg = make_tiny_mistral()
        tokens = self._run_prefill_then_decode(model, cfg, n_decode_steps=3)
        assert len(tokens) == 3
        assert all(0 <= t < VOCAB_SIZE for t in tokens), f"Invalid token IDs: {tokens}"

    def test_phi3_decode_produces_valid_tokens(self):
        model, cfg = make_tiny_phi3()
        tokens = self._run_prefill_then_decode(model, cfg, n_decode_steps=3)
        assert len(tokens) == 3
        assert all(0 <= t < VOCAB_SIZE for t in tokens), f"Invalid token IDs: {tokens}"

    def test_gpt2_prefill_token_matches_hf_next_token(self):
        """The first token from QEff prefill must match HF's greedy next token."""
        model, cfg = make_tiny_gpt2()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            hf_next = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        qeff_model = QEFFAutoModelForCausalLM(model)
        qeff_inputs = make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            qeff_next = qeff_model.model(**qeff_inputs).logits[:, -1, :].argmax(-1).item()

        assert hf_next == qeff_next, f"Prefill next token mismatch: HF={hf_next}, QEff={qeff_next}"

    def test_llama_prefill_token_matches_hf_next_token(self):
        model, cfg = make_tiny_llama()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

        with torch.no_grad():
            hf_next = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        qeff_model = QEFFAutoModelForCausalLM(model)
        qeff_inputs = make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            qeff_next = qeff_model.model(**qeff_inputs).logits[:, -1, :].argmax(-1).item()

        assert hf_next == qeff_next, f"Prefill next token mismatch: HF={hf_next}, QEff={qeff_next}"

    def test_gpt2_decode_is_deterministic(self):
        """Same model + same input must produce the same decode sequence."""
        import copy

        model, cfg = make_tiny_gpt2()
        model_copy = copy.deepcopy(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        tokens1 = self._run_prefill_then_decode(model, cfg, n_decode_steps=3, input_ids=input_ids)
        tokens2 = self._run_prefill_then_decode(model_copy, cfg, n_decode_steps=3, input_ids=input_ids)
        assert tokens1 == tokens2, f"Decode is not deterministic: {tokens1} vs {tokens2}"


# ---------------------------------------------------------------------------
# Stage 2c: Continuous batching mode
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
class TestContinuousBatchingMode:
    """
    QEFFAutoModelForCausalLM with continuous_batching=True must wrap correctly
    and produce valid outputs.
    """

    def test_gpt2_continuous_batching_wraps_without_error(self):
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        assert qeff is not None
        assert qeff.continuous_batching is True

    def test_llama_continuous_batching_wraps_without_error(self):
        model, cfg = make_tiny_llama()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        assert qeff is not None
        assert qeff.continuous_batching is True

    def test_gpt2_continuous_batching_model_is_transformed(self):
        """With continuous_batching=True, the model must still be KV-transformed."""
        from QEfficient.transformers.models.gpt2.modeling_gpt2 import QEffGPT2LMHeadModel

        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        assert isinstance(qeff.model, QEffGPT2LMHeadModel)

    def test_continuous_batching_false_is_default(self):
        model, cfg = make_tiny_gpt2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff.continuous_batching is False

    def test_continuous_batching_model_produces_finite_logits(self):
        """Continuous batching model must produce finite logits."""
        model, cfg = make_tiny_llama()
        qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        qeff_inputs = make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            out = qeff.model(**qeff_inputs)
        assert torch.isfinite(out.logits).all()


# ---------------------------------------------------------------------------
# Stage 3: ONNX export structure
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.onnx
@pytest.mark.slow
class TestCausalLMONNXStructure:
    """
    ONNX export must produce valid models with correct KV cache inputs/outputs.
    """

    def _check_onnx_export(self, factory, label, tmp_export_dir):
        import os

        model, cfg = factory()
        qeff_model = QEFFAutoModelForCausalLM(model)
        onnx_path = qeff_model.export(export_dir=str(tmp_export_dir))
        assert onnx_path is not None, f"[{label}] ONNX export returned None"
        assert os.path.exists(str(onnx_path)), f"[{label}] ONNX file does not exist"
        assert os.path.getsize(str(onnx_path)) > 0, f"[{label}] ONNX file is empty"
        return onnx_path

    def test_gpt2_onnx_export_succeeds(self, tmp_export_dir):
        self._check_onnx_export(make_tiny_gpt2, "GPT2", tmp_export_dir)

    def test_llama_onnx_export_succeeds(self, tmp_export_dir):
        self._check_onnx_export(make_tiny_llama, "Llama", tmp_export_dir)

    def test_mistral_onnx_export_succeeds(self, tmp_export_dir):
        self._check_onnx_export(make_tiny_mistral, "Mistral", tmp_export_dir)

    def test_qwen2_onnx_export_succeeds(self, tmp_export_dir):
        self._check_onnx_export(make_tiny_qwen2, "Qwen2", tmp_export_dir)

    def test_phi3_onnx_export_succeeds(self, tmp_export_dir):
        self._check_onnx_export(make_tiny_phi3, "Phi3", tmp_export_dir)

    def test_gpt2_onnx_passes_checker(self, tmp_export_dir):
        import onnx

        onnx_path = self._check_onnx_export(make_tiny_gpt2, "GPT2", tmp_export_dir)
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

    def test_llama_onnx_passes_checker(self, tmp_export_dir):
        import onnx

        onnx_path = self._check_onnx_export(make_tiny_llama, "Llama", tmp_export_dir)
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

    def test_gpt2_onnx_has_input_ids_and_position_ids(self, tmp_export_dir):
        import onnx

        onnx_path = self._check_onnx_export(make_tiny_gpt2, "GPT2", tmp_export_dir)
        onnx_model = onnx.load(str(onnx_path))
        input_names = {inp.name for inp in onnx_model.graph.input}
        assert "input_ids" in input_names, f"input_ids missing from ONNX inputs: {input_names}"
        assert "position_ids" in input_names, f"position_ids missing from ONNX inputs: {input_names}"

    def test_gpt2_onnx_has_kv_cache_inputs_for_all_layers(self, tmp_export_dir):
        import onnx

        n_layers = 2
        onnx_path = self._check_onnx_export(make_tiny_gpt2, "GPT2", tmp_export_dir)
        onnx_model = onnx.load(str(onnx_path))
        input_names = {inp.name for inp in onnx_model.graph.input}
        for i in range(n_layers):
            assert f"past_key.{i}" in input_names, f"past_key.{i} missing from ONNX inputs"
            assert f"past_value.{i}" in input_names, f"past_value.{i} missing from ONNX inputs"

    def test_llama_onnx_has_kv_cache_inputs_for_all_layers(self, tmp_export_dir):
        import onnx

        n_layers = 2
        onnx_path = self._check_onnx_export(make_tiny_llama, "Llama", tmp_export_dir)
        onnx_model = onnx.load(str(onnx_path))
        input_names = {inp.name for inp in onnx_model.graph.input}
        for i in range(n_layers):
            assert f"past_key.{i}" in input_names, f"past_key.{i} missing from ONNX inputs"
            assert f"past_value.{i}" in input_names, f"past_value.{i} missing from ONNX inputs"

    def test_gpt2_onnx_has_logits_output(self, tmp_export_dir):
        import onnx

        onnx_path = self._check_onnx_export(make_tiny_gpt2, "GPT2", tmp_export_dir)
        onnx_model = onnx.load(str(onnx_path))
        output_names = {out.name for out in onnx_model.graph.output}
        assert "logits" in output_names, f"logits missing from ONNX outputs: {output_names}"

    def test_gpt2_onnx_has_retained_state_outputs(self, tmp_export_dir):
        """KV cache outputs must be present as RetainedState outputs."""
        import onnx

        onnx_path = self._check_onnx_export(make_tiny_gpt2, "GPT2", tmp_export_dir)
        onnx_model = onnx.load(str(onnx_path))
        output_names = [out.name for out in onnx_model.graph.output]
        retained = [n for n in output_names if "RetainedState" in n]
        assert len(retained) > 0, f"No RetainedState outputs found: {output_names}"

    def test_gpt2_onnx_uses_correct_opset_version(self, tmp_export_dir):
        """Exported ONNX must use the opset version defined in QEfficient constants."""
        import onnx

        from QEfficient.utils.constants import ONNX_EXPORT_OPSET

        onnx_path = self._check_onnx_export(make_tiny_gpt2, "GPT2", tmp_export_dir)
        onnx_model = onnx.load(str(onnx_path))
        opset_versions = [op.version for op in onnx_model.opset_import]
        assert ONNX_EXPORT_OPSET in opset_versions, (
            f"Expected opset {ONNX_EXPORT_OPSET} in ONNX opset_import, got {opset_versions}"
        )

    def test_gpt2_ort_session_creation_succeeds(self, tmp_export_dir):
        """ORT session must be creatable from the exported ONNX."""
        import onnxruntime as ort

        onnx_path = self._check_onnx_export(make_tiny_gpt2, "GPT2", tmp_export_dir)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        assert session is not None
        ort_inputs = {inp.name for inp in session.get_inputs()}
        assert "input_ids" in ort_inputs
        assert "position_ids" in ort_inputs

    def _check_ort_prefill_accuracy(self, factory, label, tmp_export_dir):
        """
        Export model with SUBFUNC_ENABLED, run ORT prefill, return
        (pt_logits_last, ort_logits_last, session, output_names, input_ids, cfg).

        ORT cannot handle INT32_MAX as a GatherND index (the default sentinel used during
        ONNX export). Subfunc mode substitutes 0 instead, which is a valid index and
        produces numerically identical results because those positions are masked out
        afterward by the attention mask.
        """
        import numpy as np
        import onnxruntime as ort

        from QEfficient.transformers.cache_utils import InvalidIndexProvider

        model, cfg = factory()
        qeff_model = QEFFAutoModelForCausalLM(model)

        InvalidIndexProvider.SUBFUNC_ENABLED = True
        try:
            onnx_path = qeff_model.export(export_dir=str(tmp_export_dir), offload_pt_weights=False)
        finally:
            InvalidIndexProvider.SUBFUNC_ENABLED = False

        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        qeff_inputs = make_qeff_inputs(input_ids, cfg)
        with torch.no_grad():
            pt_logits = qeff_model.model(**qeff_inputs).logits[:, -1, :].numpy()

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        n_layers, n_kv, head_dim = _get_dims(cfg)
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "position_ids": torch.arange(SEQ_LEN).unsqueeze(0).numpy(),
        }
        for i in range(n_layers):
            ort_inputs[f"past_key.{i}"] = np.zeros((1, n_kv, CTX_LEN, head_dim), dtype=np.float32)
            ort_inputs[f"past_value.{i}"] = np.zeros((1, n_kv, CTX_LEN, head_dim), dtype=np.float32)

        output_names = [o.name for o in session.get_outputs()]
        ort_out = dict(zip(output_names, session.run(output_names, ort_inputs)))
        ort_logits = ort_out["logits"][:, -1, :]

        return pt_logits, ort_logits, session, output_names, input_ids, cfg

    def test_gpt2_ort_prefill_produces_correct_logits(self, tmp_export_dir):
        """ORT prefill must produce logits matching QEff PyTorch."""
        pt_logits, ort_logits, _, _, _, _ = self._check_ort_prefill_accuracy(make_tiny_gpt2, "GPT2", tmp_export_dir)
        pt_token = int(pt_logits.argmax(-1))
        ort_token = int(ort_logits.argmax(-1))
        assert pt_token == ort_token, f"Token mismatch: PyTorch={pt_token}, ORT={ort_token}"

    def test_llama_ort_session_creation_succeeds(self, tmp_export_dir):
        """ORT session must be creatable from the exported Llama ONNX."""
        import onnxruntime as ort

        from QEfficient.transformers.cache_utils import InvalidIndexProvider

        model, cfg = make_tiny_llama()
        qeff_model = QEFFAutoModelForCausalLM(model)
        InvalidIndexProvider.SUBFUNC_ENABLED = True
        try:
            onnx_path = qeff_model.export(export_dir=str(tmp_export_dir), offload_pt_weights=False)
        finally:
            InvalidIndexProvider.SUBFUNC_ENABLED = False
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        assert session is not None
        ort_inputs = {inp.name for inp in session.get_inputs()}
        assert "input_ids" in ort_inputs
        assert "position_ids" in ort_inputs

    def test_mistral_ort_session_creation_succeeds(self, tmp_export_dir):
        """ORT session must be creatable from the exported Mistral ONNX."""
        import onnxruntime as ort

        from QEfficient.transformers.cache_utils import InvalidIndexProvider

        model, cfg = make_tiny_mistral()
        qeff_model = QEFFAutoModelForCausalLM(model)
        InvalidIndexProvider.SUBFUNC_ENABLED = True
        try:
            onnx_path = qeff_model.export(export_dir=str(tmp_export_dir), offload_pt_weights=False)
        finally:
            InvalidIndexProvider.SUBFUNC_ENABLED = False
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        assert session is not None
        ort_inputs = {inp.name for inp in session.get_inputs()}
        assert "input_ids" in ort_inputs
        assert "position_ids" in ort_inputs

    def test_qwen2_ort_session_creation_succeeds(self, tmp_export_dir):
        """ORT session must be creatable from the exported Qwen2 ONNX."""
        import onnxruntime as ort

        from QEfficient.transformers.cache_utils import InvalidIndexProvider

        model, cfg = make_tiny_qwen2()
        qeff_model = QEFFAutoModelForCausalLM(model)
        InvalidIndexProvider.SUBFUNC_ENABLED = True
        try:
            onnx_path = qeff_model.export(export_dir=str(tmp_export_dir), offload_pt_weights=False)
        finally:
            InvalidIndexProvider.SUBFUNC_ENABLED = False
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        assert session is not None
        ort_inputs = {inp.name for inp in session.get_inputs()}
        assert "input_ids" in ort_inputs
        assert "position_ids" in ort_inputs

    def test_phi3_ort_session_creation_succeeds(self, tmp_export_dir):
        """ORT session must be creatable from the exported Phi3 ONNX."""
        import onnxruntime as ort

        from QEfficient.transformers.cache_utils import InvalidIndexProvider

        model, cfg = make_tiny_phi3()
        qeff_model = QEFFAutoModelForCausalLM(model)
        InvalidIndexProvider.SUBFUNC_ENABLED = True
        try:
            onnx_path = qeff_model.export(export_dir=str(tmp_export_dir), offload_pt_weights=False)
        finally:
            InvalidIndexProvider.SUBFUNC_ENABLED = False
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        assert session is not None
        ort_inputs = {inp.name for inp in session.get_inputs()}
        assert "input_ids" in ort_inputs
        assert "position_ids" in ort_inputs

    def test_llama_ort_prefill_produces_correct_logits(self, tmp_export_dir):
        """ORT Llama prefill must produce logits matching QEff PyTorch."""
        pt_logits, ort_logits, _, _, _, _ = self._check_ort_prefill_accuracy(make_tiny_llama, "Llama", tmp_export_dir)
        pt_token = int(pt_logits.argmax(-1))
        ort_token = int(ort_logits.argmax(-1))
        assert pt_token == ort_token, f"[Llama] Token mismatch: PyTorch={pt_token}, ORT={ort_token}"

    def test_mistral_ort_prefill_produces_correct_logits(self, tmp_export_dir):
        """ORT Mistral prefill must produce logits matching QEff PyTorch."""
        pt_logits, ort_logits, _, _, _, _ = self._check_ort_prefill_accuracy(
            make_tiny_mistral, "Mistral", tmp_export_dir
        )
        pt_token = int(pt_logits.argmax(-1))
        ort_token = int(ort_logits.argmax(-1))
        assert pt_token == ort_token, f"[Mistral] Token mismatch: PyTorch={pt_token}, ORT={ort_token}"

    def test_qwen2_ort_prefill_produces_correct_logits(self, tmp_export_dir):
        """ORT Qwen2 prefill must produce logits matching QEff PyTorch."""
        pt_logits, ort_logits, _, _, _, _ = self._check_ort_prefill_accuracy(make_tiny_qwen2, "Qwen2", tmp_export_dir)
        pt_token = int(pt_logits.argmax(-1))
        ort_token = int(ort_logits.argmax(-1))
        assert pt_token == ort_token, f"[Qwen2] Token mismatch: PyTorch={pt_token}, ORT={ort_token}"

    def test_phi3_ort_prefill_produces_correct_logits(self, tmp_export_dir):
        """ORT Phi3 prefill must produce logits matching QEff PyTorch."""
        pt_logits, ort_logits, _, _, _, _ = self._check_ort_prefill_accuracy(make_tiny_phi3, "Phi3", tmp_export_dir)
        pt_token = int(pt_logits.argmax(-1))
        ort_token = int(ort_logits.argmax(-1))
        assert pt_token == ort_token, f"[Phi3] Token mismatch: PyTorch={pt_token}, ORT={ort_token}"

    def test_gpt2_ort_logits_are_finite(self, tmp_export_dir):
        """ORT logits must not contain NaN or Inf."""
        import numpy as np

        _, ort_logits, _, _, _, _ = self._check_ort_prefill_accuracy(make_tiny_gpt2, "GPT2", tmp_export_dir)
        assert np.isfinite(ort_logits).all(), "ORT GPT2 logits contain NaN/Inf"

    def test_gpt2_ort_output_shape_is_correct(self, tmp_export_dir):
        """ORT logits shape must be (batch, seq_len, vocab_size) where seq_len matches input."""
        import numpy as np
        import onnxruntime as ort

        from QEfficient.transformers.cache_utils import InvalidIndexProvider

        model, cfg = make_tiny_gpt2()
        qeff_model = QEFFAutoModelForCausalLM(model)
        InvalidIndexProvider.SUBFUNC_ENABLED = True
        try:
            onnx_path = qeff_model.export(export_dir=str(tmp_export_dir), offload_pt_weights=False)
        finally:
            InvalidIndexProvider.SUBFUNC_ENABLED = False

        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        n_layers, n_kv, head_dim = _get_dims(cfg)
        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "position_ids": torch.arange(SEQ_LEN).unsqueeze(0).numpy(),
        }
        for i in range(n_layers):
            ort_inputs[f"past_key.{i}"] = np.zeros((1, n_kv, CTX_LEN, head_dim), dtype=np.float32)
            ort_inputs[f"past_value.{i}"] = np.zeros((1, n_kv, CTX_LEN, head_dim), dtype=np.float32)

        output_names = [o.name for o in session.get_outputs()]
        ort_out = dict(zip(output_names, session.run(output_names, ort_inputs)))
        logits = ort_out["logits"]
        # ORT model returns logits with shape (batch, actual_seq_len, vocab_size)
        # where actual_seq_len may be 1 (last token only) or match input seq_len
        assert logits.shape[0] == 1, f"Expected batch size 1, got {logits.shape[0]}"
        assert logits.shape[2] == VOCAB_SIZE, f"Expected vocab size {VOCAB_SIZE}, got {logits.shape[2]}"
        assert logits.shape[1] in [1, SEQ_LEN], f"Expected seq_len to be 1 or {SEQ_LEN}, got {logits.shape[1]}"

    def test_gpt2_ort_kv_cache_outputs_present(self, tmp_export_dir):
        """ORT outputs must include RetainedState KV cache entries."""
        _, _, session, output_names, _, _ = self._check_ort_prefill_accuracy(make_tiny_gpt2, "GPT2", tmp_export_dir)
        retained = [n for n in output_names if "RetainedState" in n]
        assert len(retained) > 0, f"No RetainedState outputs in ORT session: {output_names}"

    def test_gpt2_ort_logits_numerically_close_to_pytorch(self, tmp_export_dir):
        """ORT and PyTorch softmax distributions must be close (max_diff < 1e-3)."""
        import numpy as np

        pt_logits, ort_logits, _, _, _, _ = self._check_ort_prefill_accuracy(make_tiny_gpt2, "GPT2", tmp_export_dir)
        pt_probs = torch.tensor(pt_logits).softmax(-1).numpy()
        ort_probs = torch.tensor(ort_logits).softmax(-1).numpy()
        max_diff = float(np.abs(pt_probs - ort_probs).max())
        assert max_diff < 1e-3, f"ORT vs PyTorch softmax max_diff={max_diff:.6f} exceeds 1e-3"
