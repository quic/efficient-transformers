# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""

Gemma2 is architecturally distinct from all other tested models:
  1. Uses QEffHybridCache (not QEffDynamicCache) — completely different cache class
  2. QEffGemma2ForCausalLM.forward() uses:
       logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
       hidden_states = outputs[0][arange, logit_index]
     → returns logits of shape (batch, 1, vocab), NOT (batch, seq, vocab)
  3. Has final_logit_softcapping (tanh-based logit capping)
  4. Has sliding-window attention layers interleaved with full-context layers

A bug in any of these paths would be invisible to the existing test suite.

Tests verify:
  - HF Gemma2 baseline: correct logit shape, finite outputs
  - QEff Gemma2 wraps correctly (QEffGemma2ForCausalLM class is used)
  - QEff Gemma2 returns (batch, 1, vocab) shaped logits
  - QEff Gemma2 prefill token matches HF greedy token
  - QEff Gemma2 logits are numerically close to HF (softmax max_diff < 1e-3)
  - QEff Gemma2 cache is non-zero after prefill (CtxScatterFunc ran)
  - QEff Gemma2 prefill → decode handoff with REAL cache
  - QEff Gemma2 decode produces valid, finite, deterministic tokens
  - QEff Gemma2 real cache differs from zero cache (cache influences output)

All tests run on CPU only.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers import Gemma2Config, Gemma2ForCausalLM

from QEfficient.transformers.models.gemma2.modeling_gemma2 import QEffGemma2ForCausalLM
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

CTX_LEN = 32
PREFILL_LEN = 8
VOCAB_SIZE = 500


# ---------------------------------------------------------------------------
# Tiny Gemma2 factory
# ---------------------------------------------------------------------------


def make_tiny_gemma2():
    """
    Minimal Gemma2 config that exercises both sliding and non-sliding layers.
    sliding_window_pattern=2 → layers 0,2 are sliding; layers 1,3 are non-sliding.
    Softcapping disabled so HF and QEff logits are directly comparable.
    """
    cfg = Gemma2Config(
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
        sliding_window=8,
        sliding_window_pattern=2,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
    )
    return Gemma2ForCausalLM(cfg).eval(), cfg


def _zero_kv_cache(config, ctx_len=CTX_LEN):
    """Build a zero-initialised past_key_values tuple for Gemma2."""
    n_layers = config.num_hidden_layers
    n_kv = config.num_key_value_heads
    head_dim = config.head_dim
    return tuple(
        (
            torch.zeros(1, n_kv, ctx_len, head_dim, dtype=torch.float32),
            torch.zeros(1, n_kv, ctx_len, head_dim, dtype=torch.float32),
        )
        for _ in range(n_layers)
    )


def _prefill_inputs(input_ids, config, ctx_len=CTX_LEN):
    """Build QEff-style prefill inputs for Gemma2."""
    seq = input_ids.shape[1]
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "past_key_values": _zero_kv_cache(config, ctx_len),
    }


def _decode_inputs(next_token, decode_position, past_key_values):
    """Build a single-token decode input using the REAL past_key_values."""
    return {
        "input_ids": torch.tensor([[next_token]], dtype=torch.long),
        "position_ids": torch.tensor([[decode_position]], dtype=torch.long),
        "past_key_values": past_key_values,
    }


def _extract_next_token(logits):
    """
    Extract greedy next token. QEffGemma2ForCausalLM returns (batch, 1, vocab),
    so logits[0, -1, :] works for both (batch, seq, vocab) and (batch, 1, vocab).
    """
    return logits[0, -1, :].argmax(-1).item()


# ---------------------------------------------------------------------------
# Tests: HF Gemma2 baseline
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
class TestHFGemma2Baseline:
    """HF Gemma2 model runs correctly on CPU and produces valid logits."""

    def test_forward_returns_logits_with_correct_shape(self):
        model, cfg = make_tiny_gemma2()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (1, PREFILL_LEN, VOCAB_SIZE), (
            f"Expected (1, {PREFILL_LEN}, {VOCAB_SIZE}), got {out.logits.shape}"
        )

    def test_logits_are_finite(self):
        model, cfg = make_tiny_gemma2()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert torch.isfinite(out.logits).all()

    def test_greedy_token_is_in_valid_range(self):
        model, cfg = make_tiny_gemma2()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            token = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()
        assert 0 <= token < VOCAB_SIZE

    def test_greedy_decode_is_deterministic(self):
        model, cfg = make_tiny_gemma2()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            t1 = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()
            t2 = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()
        assert t1 == t2


# ---------------------------------------------------------------------------
# Tests: QEff Gemma2 architecture
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
class TestQEffGemma2Architecture:
    """QEff Gemma2 must use QEffGemma2ForCausalLM after KVCacheTransform."""

    def test_qeff_wraps_without_error(self):
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert qeff is not None
        assert hasattr(qeff, "model")

    def test_qeff_model_class_is_qeff_gemma2(self):
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert isinstance(qeff.model, QEffGemma2ForCausalLM), f"Expected QEffGemma2ForCausalLM, got {type(qeff.model)}"

    def test_qeff_model_is_eval_mode(self):
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        assert not qeff.model.training

    def test_qeff_model_has_same_parameter_count_as_hf(self):
        model, cfg = make_tiny_gemma2()
        hf_params = sum(p.numel() for p in model.parameters())
        qeff = QEFFAutoModelForCausalLM(model)
        qeff_params = sum(p.numel() for p in qeff.model.parameters())
        assert hf_params == qeff_params, f"Parameter count changed: HF={hf_params}, QEff={qeff_params}"


# ---------------------------------------------------------------------------
# Tests: QEff Gemma2 logit shape (argmax-based extraction)
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestQEffGemma2LogitShape:
    """
    QEffGemma2ForCausalLM uses position_ids.argmax to extract a single logit
    per batch item, returning (batch, 1, vocab) — not (batch, seq, vocab).
    This is a unique property that must be explicitly tested.
    """

    def test_prefill_logits_shape_is_batch_1_vocab(self):
        """
        QEff Gemma2 prefill must return logits of shape (1, 1, VOCAB_SIZE),
        not (1, PREFILL_LEN, VOCAB_SIZE).
        """
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            out = qeff.model(**_prefill_inputs(input_ids, cfg))
        assert out.logits.shape == (1, 1, VOCAB_SIZE), (
            f"QEffGemma2 prefill logits shape: expected (1, 1, {VOCAB_SIZE}), "
            f"got {out.logits.shape}. "
            f"QEffGemma2ForCausalLM uses position_ids.argmax to extract a single logit."
        )

    def test_decode_logits_shape_is_batch_1_vocab(self):
        """QEff Gemma2 decode must also return (1, 1, VOCAB_SIZE)."""
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            prefill_out = qeff.model(**_prefill_inputs(input_ids, cfg))
        prefill_token = _extract_next_token(prefill_out.logits)
        with torch.no_grad():
            decode_out = qeff.model(**_decode_inputs(prefill_token, PREFILL_LEN, prefill_out.past_key_values))
        assert decode_out.logits.shape == (1, 1, VOCAB_SIZE), (
            f"QEffGemma2 decode logits shape: expected (1, 1, {VOCAB_SIZE}), got {decode_out.logits.shape}"
        )

    def test_prefill_logits_are_finite(self):
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            out = qeff.model(**_prefill_inputs(input_ids, cfg))
        assert torch.isfinite(out.logits).all()


# ---------------------------------------------------------------------------
# Tests: QEff Gemma2 accuracy vs HF
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestQEffGemma2AccuracyVsHF:
    """
    QEff Gemma2 must produce the same greedy next token as HF and
    numerically close logits.
    """

    def test_prefill_token_matches_hf(self):
        """QEff Gemma2 prefill greedy token must match HF greedy token."""
        model, cfg = make_tiny_gemma2()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            hf_token = model(input_ids=input_ids).logits[:, -1, :].argmax(-1).item()

        qeff = QEFFAutoModelForCausalLM(model)
        with torch.no_grad():
            qeff_out = qeff.model(**_prefill_inputs(input_ids, cfg))
        qeff_token = _extract_next_token(qeff_out.logits)

        assert hf_token == qeff_token, (
            f"Gemma2 prefill token mismatch: HF={hf_token}, QEff={qeff_token}. "
            f"KVCacheTransform must not change the greedy prediction."
        )

    def test_prefill_logits_numerically_close_to_hf(self):
        """QEff Gemma2 softmax probabilities must be close to HF (max_diff < 1e-3)."""
        model, cfg = make_tiny_gemma2()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            hf_logits = model(input_ids=input_ids).logits[:, -1, :]

        qeff = QEFFAutoModelForCausalLM(model)
        with torch.no_grad():
            qeff_out = qeff.model(**_prefill_inputs(input_ids, cfg))
        # qeff_out.logits is (1, 1, vocab) — squeeze to (1, vocab)
        qeff_logits = qeff_out.logits[:, -1, :]

        hf_probs = F.softmax(hf_logits, dim=-1)
        qeff_probs = F.softmax(qeff_logits, dim=-1)
        max_diff = (hf_probs - qeff_probs).abs().max().item()
        assert max_diff < 1e-3, f"Gemma2 probability distribution mismatch: max_diff={max_diff:.6f} > 1e-3"

    def test_top5_tokens_overlap_with_hf(self):
        """Top-5 predicted tokens must overlap between HF and QEff."""
        model, cfg = make_tiny_gemma2()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            hf_top5 = set(model(input_ids=input_ids).logits[:, -1, :].topk(5).indices.squeeze().tolist())

        qeff = QEFFAutoModelForCausalLM(model)
        with torch.no_grad():
            qeff_out = qeff.model(**_prefill_inputs(input_ids, cfg))
        qeff_top5 = set(qeff_out.logits[:, -1, :].topk(5).indices.squeeze().tolist())

        overlap = len(hf_top5 & qeff_top5)
        assert overlap >= 4, f"Gemma2 top-5 token overlap too low: {overlap}/5. HF={hf_top5}, QEff={qeff_top5}"


# ---------------------------------------------------------------------------
# Tests: QEff Gemma2 KV cache is written during prefill
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestQEffGemma2CacheWritten:
    """
    After Gemma2 prefill, the KV cache must contain non-zero values.
    Gemma2 uses QEffHybridCache — a completely different cache class from
    QEffDynamicCache. A zero cache means the scatter never ran.
    """

    def test_past_key_values_not_none_after_prefill(self):
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            out = qeff.model(**_prefill_inputs(input_ids, cfg))
        assert out.past_key_values is not None, "Gemma2 past_key_values is None after prefill"

    def test_cache_is_non_zero_after_prefill(self):
        """
        Gemma2 uses QEffHybridCache which stores tensors in key_cache/value_cache lists.
        At least one position in the prefill range must be non-zero.
        """
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            out = qeff.model(**_prefill_inputs(input_ids, cfg))

        pkv = out.past_key_values

        # QEffHybridCache stores in key_cache list
        if hasattr(pkv, "key_cache") and len(pkv.key_cache) > 0:
            layer0_keys = pkv.key_cache[0]
        elif hasattr(pkv, "layers") and len(pkv.layers) > 0:
            layer0_keys = pkv.layers[0].keys
        elif isinstance(pkv, (list, tuple)) and len(pkv) > 0:
            layer0_keys = pkv[0][0]
        else:
            pytest.skip(f"Unrecognised past_key_values type: {type(pkv)}")
            return

        assert layer0_keys is not None, "Layer-0 keys are None after Gemma2 prefill"
        prefill_slice = layer0_keys[0, :, :PREFILL_LEN, :]
        assert not torch.all(prefill_slice == 0.0), (
            "Gemma2 KV cache is all-zeros after prefill — CtxScatterFunc never ran"
        )

    def test_cache_has_correct_number_of_layers(self):
        """past_key_values must have one entry per transformer layer."""
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            out = qeff.model(**_prefill_inputs(input_ids, cfg))

        pkv = out.past_key_values
        if hasattr(pkv, "key_cache"):
            n_cached = len(pkv.key_cache)
        elif hasattr(pkv, "layers"):
            n_cached = len(pkv.layers)
        elif isinstance(pkv, (list, tuple)):
            n_cached = len(pkv)
        else:
            pytest.skip(f"Unrecognised past_key_values type: {type(pkv)}")
            return

        assert n_cached == cfg.num_hidden_layers, f"Expected {cfg.num_hidden_layers} cached layers, got {n_cached}"


# ---------------------------------------------------------------------------
# Tests: QEff Gemma2 prefill → decode handoff with REAL cache
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestQEffGemma2PrefillDecodeHandoff:
    """
    Gemma2 prefill → decode handoff with the REAL cache.
    This is the critical path that was completely untested.
    """

    def test_decode_with_real_cache_produces_valid_token(self):
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            prefill_out = qeff.model(**_prefill_inputs(input_ids, cfg))
        prefill_token = _extract_next_token(prefill_out.logits)

        with torch.no_grad():
            decode_out = qeff.model(**_decode_inputs(prefill_token, PREFILL_LEN, prefill_out.past_key_values))

        dec_token = _extract_next_token(decode_out.logits)
        assert 0 <= dec_token < VOCAB_SIZE, f"Gemma2 decode token {dec_token} out of range [0, {VOCAB_SIZE})"

    def test_decode_with_real_cache_returns_finite_logits(self):
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            prefill_out = qeff.model(**_prefill_inputs(input_ids, cfg))
        prefill_token = _extract_next_token(prefill_out.logits)

        with torch.no_grad():
            decode_out = qeff.model(**_decode_inputs(prefill_token, PREFILL_LEN, prefill_out.past_key_values))

        assert torch.isfinite(decode_out.logits).all(), "Gemma2 decode logits contain NaN/Inf after real-cache handoff"

    def test_three_decode_steps_all_valid(self):
        """Three consecutive decode steps with real cache must all produce valid tokens."""
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            prefill_out = qeff.model(**_prefill_inputs(input_ids, cfg))

        token = _extract_next_token(prefill_out.logits)
        current_past = prefill_out.past_key_values
        decode_pos = PREFILL_LEN
        decode_tokens = []

        for step in range(3):
            with torch.no_grad():
                out = qeff.model(**_decode_inputs(token, decode_pos, current_past))
            token = _extract_next_token(out.logits)
            decode_tokens.append(token)
            current_past = out.past_key_values
            decode_pos += 1

        assert len(decode_tokens) == 3
        for i, tok in enumerate(decode_tokens):
            assert 0 <= tok < VOCAB_SIZE, f"Gemma2 decode step {i}: token {tok} out of range"

    def test_three_decode_steps_all_finite(self):
        """All decode logits must be finite."""
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            prefill_out = qeff.model(**_prefill_inputs(input_ids, cfg))

        token = _extract_next_token(prefill_out.logits)
        current_past = prefill_out.past_key_values
        decode_pos = PREFILL_LEN

        for step in range(3):
            with torch.no_grad():
                out = qeff.model(**_decode_inputs(token, decode_pos, current_past))
            assert torch.isfinite(out.logits).all(), f"Gemma2 decode step {step}: logits contain NaN/Inf"
            token = _extract_next_token(out.logits)
            current_past = out.past_key_values
            decode_pos += 1

    def test_decode_is_deterministic(self):
        """Same model + same input must produce the same decode sequence."""
        import copy

        model, cfg = make_tiny_gemma2()
        model_copy = copy.deepcopy(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        def _run(m):
            qeff = QEFFAutoModelForCausalLM(m)
            with torch.no_grad():
                prefill_out = qeff.model(**_prefill_inputs(input_ids, cfg))
            token = _extract_next_token(prefill_out.logits)
            current_past = prefill_out.past_key_values
            tokens = []
            for pos in range(PREFILL_LEN, PREFILL_LEN + 3):
                with torch.no_grad():
                    out = qeff.model(**_decode_inputs(token, pos, current_past))
                token = _extract_next_token(out.logits)
                tokens.append(token)
                current_past = out.past_key_values
            return tokens

        tokens1 = _run(model)
        tokens2 = _run(model_copy)
        assert tokens1 == tokens2, f"Gemma2 decode is not deterministic: {tokens1} vs {tokens2}"

    def test_real_cache_differs_from_zero_cache(self):
        """
        The decode token using the REAL prefill cache must differ from the
        decode token using a ZERO cache for at least one seed.
        """
        model, cfg = make_tiny_gemma2()
        found_difference = False

        for seed in range(8):
            torch.manual_seed(seed)
            qeff = QEFFAutoModelForCausalLM(model)
            input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

            with torch.no_grad():
                prefill_out = qeff.model(**_prefill_inputs(input_ids, cfg))
            prefill_token = _extract_next_token(prefill_out.logits)
            real_cache = prefill_out.past_key_values

            # Decode with REAL cache
            with torch.no_grad():
                out_real = qeff.model(**_decode_inputs(prefill_token, PREFILL_LEN, real_cache))
            real_token = _extract_next_token(out_real.logits)

            # Decode with ZERO cache
            with torch.no_grad():
                out_zero = qeff.model(**_decode_inputs(prefill_token, PREFILL_LEN, _zero_kv_cache(cfg)))
            zero_token = _extract_next_token(out_zero.logits)

            if real_token != zero_token:
                found_difference = True
                break

        assert found_difference, (
            "Gemma2 real-cache decode always produced the same token as zero-cache "
            "decode across 8 seeds. The KV cache may not be influencing output."
        )

    def test_decode_position_advances_strictly(self):
        """Each decode step must use a strictly increasing position_id."""
        model, cfg = make_tiny_gemma2()
        qeff = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        with torch.no_grad():
            prefill_out = qeff.model(**_prefill_inputs(input_ids, cfg))

        token = _extract_next_token(prefill_out.logits)
        current_past = prefill_out.past_key_values
        positions_used = [PREFILL_LEN - 1]

        for step in range(4):
            next_pos = positions_used[-1] + 1
            decode_in = _decode_inputs(token, next_pos, current_past)
            assert decode_in["position_ids"].item() == next_pos
            positions_used.append(next_pos)

            with torch.no_grad():
                out = qeff.model(**decode_in)
            token = _extract_next_token(out.logits)
            current_past = out.past_key_values

        for i in range(1, len(positions_used)):
            assert positions_used[i] > positions_used[i - 1], (
                f"Gemma2 positions not strictly increasing: {positions_used}"
            )
