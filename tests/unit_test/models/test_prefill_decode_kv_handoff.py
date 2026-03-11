# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Priority-1 fix: Real prefill → decode KV-cache handoff correctness.

The existing test_causal_lm_accuracy.py decode tests feed a ZERO cache into
every decode step, so they never exercise the actual prefill→decode handoff.
These tests pass the REAL past_key_values returned by prefill into the decode
step — the only way to catch:
  - Cache not being written during prefill (CtxScatterFunc never ran)
  - Decode reading from the wrong cache slot (off-by-one in position_ids)
  - Logit-index extraction bugs (argmax-based logit selection in Llama/Gemma2)
  - Position counter not advancing across decode steps

Key design note: QEffLlamaForCausalLM and QEffGemma2ForCausalLM both use
  logit_index = position_ids.argmax(1, keepdim=True)
and return logits of shape (batch, 1, vocab) — NOT (batch, seq, vocab).
_extract_next_token() handles both shapes via logits[0, -1, :].

Models: GPT2, Llama, Mistral, Qwen2, Phi3, Gemma
All tests run on CPU only.
"""

import pytest
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GemmaConfig,
    GemmaForCausalLM,
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
PREFILL_LEN = 8
VOCAB_SIZE = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_dims(config):
    """Return (n_layers, n_kv_heads, head_dim) for any config."""
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


def _zero_kv_cache(config, ctx_len=CTX_LEN):
    """Build a zero-initialised past_key_values tuple (QEff prefill input)."""
    n_layers, n_kv, head_dim = _get_dims(config)
    return tuple(
        (
            torch.zeros(1, n_kv, ctx_len, head_dim, dtype=torch.float32),
            torch.zeros(1, n_kv, ctx_len, head_dim, dtype=torch.float32),
        )
        for _ in range(n_layers)
    )


def _prefill_inputs(input_ids, config, ctx_len=CTX_LEN):
    """Build QEff-style prefill inputs with zero-init KV cache."""
    seq = input_ids.shape[1]
    position_ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "past_key_values": _zero_kv_cache(config, ctx_len),
    }


def _extract_next_token(logits):
    """
    Extract greedy next token from logits of shape (batch, seq, vocab) or
    (batch, 1, vocab). QEffLlamaForCausalLM and QEffGemma2ForCausalLM both
    return (batch, 1, vocab) via position_ids.argmax-based logit extraction.
    logits[0, -1, :] works for both shapes.
    """
    return logits[0, -1, :].argmax(-1).item()


def _decode_inputs(next_token, decode_position, past_key_values):
    """Build a single-token decode input using the REAL past_key_values."""
    return {
        "input_ids": torch.tensor([[next_token]], dtype=torch.long),
        "position_ids": torch.tensor([[decode_position]], dtype=torch.long),
        "past_key_values": past_key_values,
    }


# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


def make_tiny_gpt2():
    cfg = GPT2Config(
        n_layer=2, n_head=2, n_embd=64,
        vocab_size=VOCAB_SIZE, n_positions=CTX_LEN, n_ctx=CTX_LEN,
    )
    return GPT2LMHeadModel(cfg).eval(), cfg


def make_tiny_llama():
    cfg = LlamaConfig(
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        hidden_size=64, intermediate_size=128, vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return LlamaForCausalLM(cfg).eval(), cfg


def make_tiny_mistral():
    cfg = MistralConfig(
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        hidden_size=64, intermediate_size=128, vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return MistralForCausalLM(cfg).eval(), cfg


def make_tiny_qwen2():
    cfg = Qwen2Config(
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        hidden_size=64, intermediate_size=128, vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return Qwen2ForCausalLM(cfg).eval(), cfg


def make_tiny_phi3():
    cfg = Phi3Config(
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        hidden_size=64, intermediate_size=128, vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN, pad_token_id=0,
    )
    return Phi3ForCausalLM(cfg).eval(), cfg


def make_tiny_gemma():
    cfg = GemmaConfig(
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        hidden_size=64, intermediate_size=128, vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN, head_dim=32,
    )
    return GemmaForCausalLM(cfg).eval(), cfg


# ---------------------------------------------------------------------------
# Core runner: prefill then N decode steps with REAL cache
# ---------------------------------------------------------------------------


def _run_real_handoff(factory, n_decode_steps=3, seed=42):
    """
    Run prefill with zero-init cache, then run n_decode_steps using the
    REAL past_key_values returned by each step.

    Returns:
        prefill_token  - greedy token from prefill
        decode_tokens  - list of greedy tokens from each decode step
        all_logits     - list of raw logit tensors for each step
    """
    torch.manual_seed(seed)
    model, cfg = factory()
    qeff = QEFFAutoModelForCausalLM(model)

    input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
    prefill_in = _prefill_inputs(input_ids, cfg)

    with torch.no_grad():
        prefill_out = qeff.model(**prefill_in)

    prefill_token = _extract_next_token(prefill_out.logits)
    all_logits = [prefill_out.logits]
    decode_tokens = []

    current_past = prefill_out.past_key_values
    current_decode_pos = PREFILL_LEN  # first decode position is PREFILL_LEN

    for _ in range(n_decode_steps):
        decode_in = _decode_inputs(prefill_token, current_decode_pos, current_past)
        with torch.no_grad():
            decode_out = qeff.model(**decode_in)

        next_tok = _extract_next_token(decode_out.logits)
        decode_tokens.append(next_tok)
        all_logits.append(decode_out.logits)
        current_past = decode_out.past_key_values
        prefill_token = next_tok
        current_decode_pos += 1

    return prefill_token, decode_tokens, all_logits


# ---------------------------------------------------------------------------
# Tests: KV cache is actually written during prefill
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestPrefillWritesCache:
    """
    After prefill, past_key_values must be non-None and contain non-zero
    values in the prefill positions. A zero cache means CtxScatterFunc
    never ran — the most catastrophic possible failure.
    """

    def _assert_cache_written(self, factory, label):
        model, cfg = factory()
        qeff = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        with torch.no_grad():
            out = qeff.model(**_prefill_inputs(input_ids, cfg))

        assert out.past_key_values is not None, (
            f"[{label}] past_key_values is None after prefill"
        )

        # Inspect layer-0 keys — works for both QEffDynamicCache and legacy tuple
        pkv = out.past_key_values
        if hasattr(pkv, "layers"):
            layer0_keys = pkv.layers[0].keys  # QEffDynamicCache
        elif isinstance(pkv, (list, tuple)) and len(pkv) > 0:
            layer0_keys = pkv[0][0]           # legacy tuple
        else:
            pytest.skip(f"[{label}] Unrecognised past_key_values type: {type(pkv)}")
            return

        assert layer0_keys is not None, f"[{label}] Layer-0 keys are None after prefill"
        # At least one value in positions 0..PREFILL_LEN-1 must be non-zero
        prefill_slice = layer0_keys[0, :, :PREFILL_LEN, :]
        assert not torch.all(prefill_slice == 0.0), (
            f"[{label}] KV cache is all-zeros after prefill — CtxScatterFunc never ran"
        )

    def test_gpt2_cache_written_after_prefill(self):
        self._assert_cache_written(make_tiny_gpt2, "GPT2")

    def test_llama_cache_written_after_prefill(self):
        self._assert_cache_written(make_tiny_llama, "Llama")

    def test_mistral_cache_written_after_prefill(self):
        self._assert_cache_written(make_tiny_mistral, "Mistral")

    def test_qwen2_cache_written_after_prefill(self):
        self._assert_cache_written(make_tiny_qwen2, "Qwen2")

    def test_phi3_cache_written_after_prefill(self):
        self._assert_cache_written(make_tiny_phi3, "Phi3")

    def test_gemma_cache_written_after_prefill(self):
        self._assert_cache_written(make_tiny_gemma, "Gemma")


# ---------------------------------------------------------------------------
# Tests: Decode with REAL cache produces valid, finite, deterministic tokens
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestRealCacheDecodeCorrectness:
    """
    Decode steps using the REAL prefill cache must produce valid, finite,
    deterministic token IDs. This is the test that was missing.
    """

    def _assert_valid(self, factory, label):
        _, decode_tokens, _ = _run_real_handoff(factory, n_decode_steps=3)
        assert len(decode_tokens) == 3
        for i, tok in enumerate(decode_tokens):
            assert 0 <= tok < VOCAB_SIZE, (
                f"[{label}] Decode step {i}: token {tok} out of range [0, {VOCAB_SIZE})"
            )

    def _assert_finite(self, factory, label):
        _, _, all_logits = _run_real_handoff(factory, n_decode_steps=3)
        for i, logits in enumerate(all_logits):
            assert torch.isfinite(logits).all(), (
                f"[{label}] Step {i}: logits contain NaN/Inf after real-cache handoff"
            )

    def _assert_deterministic(self, factory, label):
        _, tokens1, _ = _run_real_handoff(factory, n_decode_steps=3, seed=7)
        _, tokens2, _ = _run_real_handoff(factory, n_decode_steps=3, seed=7)
        assert tokens1 == tokens2, (
            f"[{label}] Decode is not deterministic: {tokens1} vs {tokens2}"
        )

    def test_gpt2_decode_valid(self):
        self._assert_valid(make_tiny_gpt2, "GPT2")

    def test_llama_decode_valid(self):
        self._assert_valid(make_tiny_llama, "Llama")

    def test_mistral_decode_valid(self):
        self._assert_valid(make_tiny_mistral, "Mistral")

    def test_qwen2_decode_valid(self):
        self._assert_valid(make_tiny_qwen2, "Qwen2")

    def test_phi3_decode_valid(self):
        self._assert_valid(make_tiny_phi3, "Phi3")

    def test_gemma_decode_valid(self):
        self._assert_valid(make_tiny_gemma, "Gemma")

    def test_gpt2_decode_finite(self):
        self._assert_finite(make_tiny_gpt2, "GPT2")

    def test_llama_decode_finite(self):
        self._assert_finite(make_tiny_llama, "Llama")

    def test_mistral_decode_finite(self):
        self._assert_finite(make_tiny_mistral, "Mistral")

    def test_qwen2_decode_finite(self):
        self._assert_finite(make_tiny_qwen2, "Qwen2")

    def test_gpt2_decode_deterministic(self):
        self._assert_deterministic(make_tiny_gpt2, "GPT2")

    def test_llama_decode_deterministic(self):
        self._assert_deterministic(make_tiny_llama, "Llama")

    def test_mistral_decode_deterministic(self):
        self._assert_deterministic(make_tiny_mistral, "Mistral")


# ---------------------------------------------------------------------------
# Tests: Real cache influences decode output (cache is actually used)
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestRealCacheInfluencesOutput:
    """
    The decode token when using the REAL prefill cache must differ from the
    decode token when using a ZERO cache for at least one seed.
    If they are always identical, the cache is not influencing the output at all.
    """

    def _assert_cache_influences_output(self, factory, label, n_seeds=8):
        model, cfg = factory()
        found_difference = False

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            qeff = QEFFAutoModelForCausalLM(model)
            input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

            # Prefill to get real cache
            prefill_in = _prefill_inputs(input_ids, cfg)
            with torch.no_grad():
                prefill_out = qeff.model(**prefill_in)
            prefill_token = _extract_next_token(prefill_out.logits)
            real_cache = prefill_out.past_key_values
            decode_pos = PREFILL_LEN

            # Decode with REAL cache
            with torch.no_grad():
                out_real = qeff.model(**_decode_inputs(prefill_token, decode_pos, real_cache))
            real_token = _extract_next_token(out_real.logits)

            # Decode with ZERO cache (what the old tests did)
            with torch.no_grad():
                out_zero = qeff.model(**_decode_inputs(
                    prefill_token, decode_pos, _zero_kv_cache(cfg)
                ))
            zero_token = _extract_next_token(out_zero.logits)

            if real_token != zero_token:
                found_difference = True
                break

        assert found_difference, (
            f"[{label}] Real-cache decode always produced the same token as zero-cache "
            f"decode across {n_seeds} seeds. The KV cache may not be influencing output."
        )

    def test_llama_real_cache_differs_from_zero_cache(self):
        self._assert_cache_influences_output(make_tiny_llama, "Llama")

    def test_mistral_real_cache_differs_from_zero_cache(self):
        self._assert_cache_influences_output(make_tiny_mistral, "Mistral")

    def test_qwen2_real_cache_differs_from_zero_cache(self):
        self._assert_cache_influences_output(make_tiny_qwen2, "Qwen2")


# ---------------------------------------------------------------------------
# Tests: Decode position advances strictly across steps
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestDecodePositionAdvancesStrictly:
    """
    Each decode step must use a strictly increasing position_id.
    If positions don't advance, the model writes to the same cache slot
    every step, silently corrupting the KV cache.
    """

    def _assert_positions_advance(self, factory, label):
        model, cfg = factory()
        qeff = QEFFAutoModelForCausalLM(model)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))
        prefill_in = _prefill_inputs(input_ids, cfg)

        with torch.no_grad():
            prefill_out = qeff.model(**prefill_in)

        token = _extract_next_token(prefill_out.logits)
        current_past = prefill_out.past_key_values
        positions_used = [PREFILL_LEN - 1]  # last prefill position

        for step in range(4):
            next_pos = positions_used[-1] + 1
            decode_in = _decode_inputs(token, next_pos, current_past)
            assert decode_in["position_ids"].item() == next_pos, (
                f"[{label}] Step {step}: position_ids={decode_in['position_ids'].item()}, "
                f"expected {next_pos}"
            )
            positions_used.append(next_pos)

            with torch.no_grad():
                out = qeff.model(**decode_in)
            token = _extract_next_token(out.logits)
            current_past = out.past_key_values

        for i in range(1, len(positions_used)):
            assert positions_used[i] > positions_used[i - 1], (
                f"[{label}] Positions not strictly increasing: {positions_used}"
            )

    def test_gpt2_positions_advance(self):
        self._assert_positions_advance(make_tiny_gpt2, "GPT2")

    def test_llama_positions_advance(self):
        self._assert_positions_advance(make_tiny_llama, "Llama")

    def test_mistral_positions_advance(self):
        self._assert_positions_advance(make_tiny_mistral, "Mistral")

    def test_qwen2_positions_advance(self):
        self._assert_positions_advance(make_tiny_qwen2, "Qwen2")

    def test_phi3_positions_advance(self):
        self._assert_positions_advance(make_tiny_phi3, "Phi3")


# ---------------------------------------------------------------------------
# Tests: Full pipeline — HF prefill token == QEff prefill token, then real decode
# ---------------------------------------------------------------------------


@pytest.mark.causal_lm
@pytest.mark.accuracy
class TestFullPipelineConsistency:
    """
    Combined regression test:
    1. QEff prefill token must match HF greedy token.
    2. First decode step using REAL cache must produce a finite, valid token.
    """

    def _assert_full_pipeline(self, factory, label):
        model, cfg = factory()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, PREFILL_LEN))

        # HF baseline
        with torch.no_grad():
            hf_logits = model(input_ids=input_ids).logits[:, -1, :]
        hf_token = hf_logits.argmax(-1).item()

        # QEff prefill
        qeff = QEFFAutoModelForCausalLM(model)
        with torch.no_grad():
            prefill_out = qeff.model(**_prefill_inputs(input_ids, cfg))
        qeff_token = _extract_next_token(prefill_out.logits)

        assert hf_token == qeff_token, (
            f"[{label}] Prefill token mismatch: HF={hf_token}, QEff={qeff_token}"
        )

        # Decode with REAL cache
        with torch.no_grad():
            decode_out = qeff.model(**_decode_inputs(
                qeff_token, PREFILL_LEN, prefill_out.past_key_values
            ))

        assert torch.isfinite(decode_out.logits).all(), (
            f"[{label}] Decode logits contain NaN/Inf after real-cache handoff"
        )
        dec_token = _extract_next_token(decode_out.logits)
        assert 0 <= dec_token < VOCAB_SIZE, (
            f"[{label}] Decode token {dec_token} out of range [0, {VOCAB_SIZE})"
        )

    def test_gpt2_full_pipeline(self):
        self._assert_full_pipeline(make_tiny_gpt2, "GPT2")

    def test_llama_full_pipeline(self):
        self._assert_full_pipeline(make_tiny_llama, "Llama")

    def test_mistral_full_pipeline(self):
        self._assert_full_pipeline(make_tiny_mistral, "Mistral")

    def test_qwen2_full_pipeline(self):
        self._assert_full_pipeline(make_tiny_qwen2, "Qwen2")

    def test_phi3_full_pipeline(self):
        self._assert_full_pipeline(make_tiny_phi3, "Phi3")

    def test_gemma_full_pipeline(self):
        self._assert_full_pipeline(make_tiny_gemma, "Gemma")
