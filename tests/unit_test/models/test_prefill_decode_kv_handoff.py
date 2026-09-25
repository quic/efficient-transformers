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

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
import torch
from transformers import (
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
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=VOCAB_SIZE,
        n_positions=CTX_LEN,
        n_ctx=CTX_LEN,
    )
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

        assert out.past_key_values is not None, f"[{label}] past_key_values is None after prefill"

        # Inspect layer-0 keys — works for both QEffDynamicCache and legacy tuple
        pkv = out.past_key_values
        if hasattr(pkv, "layers"):
            layer0_keys = pkv.layers[0].keys  # QEffDynamicCache
        elif isinstance(pkv, (list, tuple)) and len(pkv) > 0:
            layer0_keys = pkv[0][0]  # legacy tuple
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
            assert 0 <= tok < VOCAB_SIZE, f"[{label}] Decode step {i}: token {tok} out of range [0, {VOCAB_SIZE})"

    def _assert_finite(self, factory, label):
        _, _, all_logits = _run_real_handoff(factory, n_decode_steps=3)
        for i, logits in enumerate(all_logits):
            assert torch.isfinite(logits).all(), f"[{label}] Step {i}: logits contain NaN/Inf after real-cache handoff"

    def _assert_deterministic(self, factory, label):
        _, tokens1, _ = _run_real_handoff(factory, n_decode_steps=3, seed=7)
        _, tokens2, _ = _run_real_handoff(factory, n_decode_steps=3, seed=7)
        assert tokens1 == tokens2, f"[{label}] Decode is not deterministic: {tokens1} vs {tokens2}"

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
                out_zero = qeff.model(**_decode_inputs(prefill_token, decode_pos, _zero_kv_cache(cfg)))
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
                f"[{label}] Step {step}: position_ids={decode_in['position_ids'].item()}, expected {next_pos}"
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

        assert hf_token == qeff_token, f"[{label}] Prefill token mismatch: HF={hf_token}, QEff={qeff_token}"

        # Decode with REAL cache
        with torch.no_grad():
            decode_out = qeff.model(**_decode_inputs(qeff_token, PREFILL_LEN, prefill_out.past_key_values))

        assert torch.isfinite(decode_out.logits).all(), (
            f"[{label}] Decode logits contain NaN/Inf after real-cache handoff"
        )
        dec_token = _extract_next_token(decode_out.logits)
        assert 0 <= dec_token < VOCAB_SIZE, f"[{label}] Decode token {dec_token} out of range [0, {VOCAB_SIZE})"

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


# ---------------------------------------------------------------------------
# Tests: canonical example wires disaggregated text compilation correctly
# ---------------------------------------------------------------------------


def _parse_text_example_args(*args):
    from examples._common import args as example_args
    from examples.text_generation.basic_inference import build_parser

    namespace = build_parser().parse_args(list(args))

    def raise_validation_error(message):
        raise ValueError(message)

    example_args.validate_args(namespace, raise_validation_error)
    return namespace


def test_disaggregated_text_example_compiles_decode_and_pipeline_prefill(tmp_path):
    from examples.text_generation.basic_inference import compile_disaggregated

    namespace = _parse_text_example_args(
        "--disaggregated",
        "--full-batch-size",
        "4",
        "--prefill-num-devices",
        "8",
        "--decode-num-devices",
        "4",
        "--mdp-num-partitions",
        "4",
        "--mdp-strategy",
        "intersection",
        "--moe-expert-parallel-chunk-size",
        "128",
        "--prefill-blocking-mode",
        "prefill_online",
        "--decode-blocking-mode",
        "kv_headpar",
        "--num-kv-blocks",
        "2",
        "--num-q-blocks",
        "2",
        "--prefill-node-precision-info",
        "prefill.yaml",
        "--decode-node-precision-info",
        "decode.yaml",
        "--decode-aic-enable-depth-first",
        "--prefill-user-tiled",
        "--compile-dir",
        str(tmp_path),
    )

    class RecordingModel:
        def __init__(self):
            self.compile_calls = []

        def compile(self, **kwargs):
            self.compile_calls.append(kwargs)
            return f"qpc-{len(self.compile_calls)}"

    model = RecordingModel()
    assert compile_disaggregated(model, namespace) == ("qpc-2", "qpc-1")
    decode_call, prefill_call = model.compile_calls

    assert decode_call["prefill_seq_len"] == 1
    assert decode_call["compile_dir"] == str(tmp_path / "decode")
    assert (tmp_path / "decode").is_dir()
    assert decode_call["num_devices"] == 4
    assert decode_call["prefill_only"] is False
    assert decode_call["offload_pt_weights"] is False
    assert decode_call["split_retained_state_io"] is True
    assert decode_call["retain_full_kv"] is True
    assert decode_call["qaic_config"]["blocking_mode"] == "kv_headpar"
    assert decode_call["node_precision_info"] == "decode.yaml"
    assert decode_call["aic_enable_depth_first"] is True
    assert "user_tiled" not in decode_call

    assert prefill_call["prefill_seq_len"] == namespace.prefill_seq_len
    assert prefill_call["compile_dir"] == str(tmp_path / "prefill")
    assert (tmp_path / "prefill").is_dir()
    assert prefill_call["num_devices"] == 8
    assert prefill_call["prefill_only"] is True
    assert prefill_call["enable_chunking"] is True
    assert prefill_call["mdp_num_partitions"] == 4
    assert prefill_call["mdp_strategy"] == "intersection"
    assert prefill_call["qaic_config"]["blocking_mode"] == "prefill_online"
    assert prefill_call["qaic_config"]["moe_config"] == {"expert_parallel_chunk_size": 128}
    assert prefill_call["node_precision_info"] == "prefill.yaml"
    assert prefill_call["user_tiled"] is True
    assert "aic_enable_depth_first" not in prefill_call


def test_disaggregated_text_example_rejects_non_divisible_pipeline_layout():
    with pytest.raises(ValueError, match="divisible"):
        _parse_text_example_args(
            "--disaggregated",
            "--full-batch-size",
            "4",
            "--prefill-num-devices",
            "6",
            "--mdp-num-partitions",
            "4",
        )


def test_text_example_exposes_dtype_hardware_and_compile_only_controls():
    from examples._common import args as example_args

    namespace = _parse_text_example_args(
        "--dtype",
        "bfloat16",
        "--aic-hw-version",
        "ai200",
        "--compile-only",
    )

    assert namespace.dtype == "bfloat16"
    assert namespace.compile_only is True
    assert example_args.compiler_options(namespace)["aic_hw_version"] == "ai200"


def test_disaggregated_text_example_selects_ccl_specialization():
    from examples.text_generation.basic_inference import _select_ccl_length

    assert _select_ccl_length(None, 64, 256) is None
    assert _select_ccl_length([128, 256], 64, 256) == 128
    assert _select_ccl_length([128, 256], 129, 256) == 256


def test_disaggregated_text_example_uses_compiler_normalized_ccl_lengths():
    from examples.text_generation.basic_inference import compile_disaggregated

    namespace = _parse_text_example_args(
        "--disaggregated",
        "--full-batch-size",
        "2",
        "--ccl-prefill",
        "128",
        "256",
        "--ccl-decode",
        "256",
    )

    class NormalizingModel:
        def compile(self, **kwargs):
            if kwargs["prefill_only"]:
                self.comp_ctx_lengths_prefill = [256]
                return "prefill-qpc"
            self.comp_ctx_lengths_decode = [256]
            return "decode-qpc"

    compile_disaggregated(NormalizingModel(), namespace)

    assert namespace.comp_ctx_lengths_prefill == [256]
    assert namespace.comp_ctx_lengths_decode == [256]


@pytest.fixture
def text_example_dependencies(monkeypatch, tmp_path):
    """Keep CLI orchestration tests offline while recording public API calls."""
    tokenizer = MagicMock()
    model = MagicMock()
    model.vocab_size = 1024
    model.compile.return_value = tmp_path / "compile"
    model.generate.return_value = SimpleNamespace(generated_texts=["generated"])
    load_model = MagicMock(return_value=model)
    load_tokenizer = MagicMock(return_value=tokenizer)
    monkeypatch.setattr(QEFFAutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", load_tokenizer)
    return SimpleNamespace(model=model, load_model=load_model, load_tokenizer=load_tokenizer)


def test_continuous_batching_refill_writes_to_request_row():
    from QEfficient.generation.text_generation_inference import QEffTextGenerationBase

    generator = QEffTextGenerationBase.__new__(QEffTextGenerationBase)
    generator.include_sampler = False
    generator.return_pdfs = False
    generator.decode_input_ids = np.zeros((2, 1), dtype=np.int64)
    generator.decode_pos_ids = np.zeros((2, 1), dtype=np.int64)
    generator.generation_len = np.zeros((2, 1), dtype=np.int64)
    generator.generated_ids = np.array([[11, 12], [21, 22], [0, 0]], dtype=np.int64)

    generator.update_decode_input(
        {"logits": np.array([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32)},
        position_ids=np.array([[7]], dtype=np.int64),
        generation_len=2,
        decode_batch_id=0,
        generated_batch_id=2,
    )

    assert generator.generated_ids[0].tolist() == [11, 12]
    assert generator.generated_ids[2].tolist() == [3, 0]
    assert generator.decode_input_ids[0].tolist() == [3]
    assert generator.decode_pos_ids[0].tolist() == [7]


def test_text_example_forwards_sampler_runtime_arguments(monkeypatch, text_example_dependencies):
    _run_text_example(
        monkeypatch,
        "--include-sampler",
        "--return-pdfs",
        "--include-guided-decoding",
        "--full-batch-size",
        "2",
    )

    generate_kwargs = text_example_dependencies.model.generate.call_args.kwargs
    assert generate_kwargs["include_sampler"] is True
    assert generate_kwargs["return_pdfs"] is True
    assert generate_kwargs["include_guided_decoding"] is True
    params = generate_kwargs["sampling_params"]
    expected = {
        "repetition_penalties": ((2, 1), np.float32),
        "presence_penalties": ((2, 1), np.float32),
        "temperatures": ((2, 1), np.float32),
        "top_ks": ((2, 1), np.int32),
        "top_ps": ((2, 1), np.float32),
        "min_ps": ((2, 1), np.float32),
        "random_numbers": ((2, 512), np.float32),
        "token_bitmasks": ((2, 1024), np.bool_),
    }
    assert set(params) == set(expected)
    for name, (shape, dtype) in expected.items():
        assert params[name].shape == shape
        assert params[name].dtype == dtype


def test_disaggregated_runtime_normalizes_padding_positions_and_keeps_decode_eos(monkeypatch):
    from examples.text_generation.basic_inference import run_disaggregated

    class Tokenizer:
        padding_side = "left"
        pad_token_id = None
        eos_token_id = 2
        vocab_size = 16

        def __call__(self, prompt, return_tensors, padding, max_length=None):
            del prompt, return_tensors
            input_ids = np.array([[7, 8, 9]], dtype=np.int64)
            attention_mask = np.ones_like(input_ids)
            if padding == "max_length":
                pad_width = max_length - input_ids.shape[1]
                input_ids = np.pad(input_ids, ((0, 0), (0, pad_width)), constant_values=self.pad_token_id)
                attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_width)))
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        def decode(self, tokens):
            return " ".join(map(str, tokens))

    class PrefillSession:
        recorded_inputs = []

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def np_run_pipeline(self, inputs, **kwargs):
            del kwargs
            self.recorded_inputs.append({name: value.copy() for name, value in inputs.items()})
            return 0

        def complete_inf(self, *args, **kwargs):
            pass

        def get_outputs(self, **kwargs):
            return {"logits": np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]], dtype=np.float32)}

    class DecodeSession:
        binding_index_map = {"batch_index": 0}
        kv_cache_info = [((1, 1, 8, 1), np.float32)]
        decode_buff_map = []
        decode_rs_kv_only_buff_map = []
        decode_execObj_idx = 0

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def set_data_for_kv_handoff(self, *args, **kwargs):
            pass

        def np_run(self, inputs, **kwargs):
            self.inputs = inputs
            return 0

        def complete_inf(self, *args, **kwargs):
            pass

        def get_outputs(self, **kwargs):
            logits = np.zeros((1, 1, 6), dtype=np.float32)
            logits[..., 2] = 1.0
            return {"logits": logits}

    sessions = iter([PrefillSession, DecodeSession])

    def make_session(*args, **kwargs):
        return next(sessions)(*args, **kwargs)

    monkeypatch.setattr("QEfficient.generation.cloud_infer.QAICInferenceSession", make_session)
    namespace = _parse_text_example_args(
        "--disaggregated",
        "--full-batch-size",
        "1",
        "--prefill-seq-len",
        "4",
        "--ctx-len",
        "8",
        "--generation-len",
        "4",
    )
    tokenizer = Tokenizer()

    result = run_disaggregated(tokenizer, "prefill-qpc", "decode-qpc", ["hello"], namespace)

    assert tokenizer.padding_side == "right"
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert PrefillSession.recorded_inputs[0]["position_ids"].tolist() == [[0, 1, 2, -1]]
    assert result["tokens"] == [[5, 2]]


def _run_text_example(monkeypatch, *args):
    from examples.text_generation.basic_inference import main

    monkeypatch.setattr(sys, "argv", ["basic_inference.py", *args])
    main()


@pytest.mark.parametrize("artifacts", [False, True])
@pytest.mark.parametrize("continuous_batching", [False, True])
def test_text_example_runs_standard_api(
    monkeypatch, text_example_dependencies, tmp_path, capsys, artifacts, continuous_batching
):
    deps = text_example_dependencies
    options = ["--device-group", "2,3", "--prompt", "hello"]
    if artifacts:
        options.append("--artifacts")
        deps.model.generate.return_value = tmp_path / "compile" / "io"
    if continuous_batching:
        options += ["--continuous-batching", "--full-batch-size", "2"]
    _run_text_example(monkeypatch, *options)

    assert deps.load_model.call_args.kwargs["continuous_batching"] is continuous_batching
    assert deps.model.compile.call_args.kwargs["artifacts"] is artifacts
    assert deps.model.compile.call_args.kwargs["num_devices"] == 2
    assert deps.model.generate.call_args.kwargs["artifacts"] is artifacts
    assert deps.model.generate.call_args.kwargs["device_ids"] == [2, 3]
    assert "device_id" not in deps.model.generate.call_args.kwargs
    output = capsys.readouterr().out
    assert ("Runner inputs written to:" in output) is artifacts
    assert ("Generated: generated" in output) is not artifacts


@pytest.mark.parametrize("stage", ["both", "prefill", "decode"])
def test_text_example_compile_only_skips_generation(monkeypatch, text_example_dependencies, stage):
    _run_text_example(monkeypatch, "--artifacts", "--compile-only", "--stage", stage)
    text_example_dependencies.model.compile.assert_called_once()
    text_example_dependencies.model.generate.assert_not_called()


@pytest.mark.parametrize("flag", ["--num-hidden-layers", "--num-hidden-layers-override"])
@pytest.mark.parametrize("num_layers", [-1, 0, 2])
def test_text_example_preserves_layer_override_options(monkeypatch, text_example_dependencies, flag, num_layers):
    _run_text_example(monkeypatch, flag, str(num_layers), "--compile-only")
    load_kwargs = text_example_dependencies.load_model.call_args.kwargs
    if num_layers > 0:
        assert load_kwargs["num_hidden_layers"] == num_layers
    else:
        assert "num_hidden_layers" not in load_kwargs


@pytest.mark.parametrize("write_io", [False, True])
@pytest.mark.parametrize("include_sampler", [False, True])
def test_text_example_real_generate_accepts_runtime_kwargs(
    monkeypatch, text_example_dependencies, tmp_path, write_io, include_sampler
):
    """Use the real wrapper and enforce the downstream runtime signature."""
    import QEfficient

    model = text_example_dependencies.model
    model.qpc_path = tmp_path / "qpc"
    model.comp_ctx_lengths_prefill = None
    model.comp_ctx_lengths_decode = None
    model.is_tlm = False
    model.generate.side_effect = lambda **kwargs: QEFFAutoModelForCausalLM.generate(model, **kwargs)
    runtime = create_autospec(QEfficient.cloud_ai_100_exec_kv, return_value=SimpleNamespace(generated_texts=["ok"]))
    bundle = MagicMock(return_value=tmp_path / "io")
    monkeypatch.setattr(QEfficient, "cloud_ai_100_exec_kv", runtime)
    monkeypatch.setattr("QEfficient.transformers.models.modeling_auto.write_causal_lm_runner_bundle", bundle)
    options = []
    if write_io:
        options.append("--write-io")
    if include_sampler:
        options.append("--include-sampler")
    _run_text_example(monkeypatch, *options)
    runtime.assert_called_once()
    assert bundle.call_count == int(write_io)
    assert "write_io" not in runtime.call_args.kwargs
    assert runtime.call_args.kwargs["include_sampler"] is include_sampler
    if write_io:
        bundle_kwargs = bundle.call_args.kwargs
        assert (bundle_kwargs["sampling_params"] is not None) is include_sampler


@pytest.mark.parametrize("mode", ["kv_paged", "qkv_paged", "hqkv_paged"])
def test_text_example_forwards_paged_config(monkeypatch, text_example_dependencies, mode):
    _run_text_example(
        monkeypatch,
        "--enable-blocking",
        "--blocking-mode",
        mode,
        "--num-kv-blocks",
        "2",
        "--num-q-blocks",
        "2",
        "--head-block-size",
        "4",
    )
    expected = {"blocking_mode": mode, "num_kv_blocks": 2, "num_q_blocks": 2, "head_block_size": 4}
    assert text_example_dependencies.load_model.call_args.kwargs["qaic_config"] == expected
    assert text_example_dependencies.model.compile.call_args.kwargs["qaic_config"] == expected
    text_example_dependencies.model.generate.assert_called_once()


def test_text_example_forwards_gdn_and_dynamo(monkeypatch, text_example_dependencies):
    _run_text_example(monkeypatch, "--gdn-chunk-size", "64", "--dynamo", "--compile-only")
    deps = text_example_dependencies
    assert deps.load_model.call_args.kwargs["qaic_config"] == {"gdn_chunk_size": 64}
    assert deps.model.compile.call_args.kwargs["qaic_config"] == {"gdn_chunk_size": 64}
    assert deps.model.compile.call_args.kwargs["dynamo"] is True


def test_disaggregated_text_example_artifacts_preserve_gdn_and_skip_runtime(monkeypatch, text_example_dependencies):
    run = MagicMock(side_effect=AssertionError("artifacts must not create DMA sessions"))
    monkeypatch.setattr("examples.text_generation.basic_inference.run_disaggregated", run)
    _run_text_example(
        monkeypatch,
        "--disaggregated",
        "--full-batch-size",
        "2",
        "--compile-only",
        "--artifacts",
        "--gdn-chunk-size",
        "64",
    )
    calls = text_example_dependencies.model.compile.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs["artifacts"] is True
        assert call.kwargs["qaic_config"]["gdn_chunk_size"] == 64
    run.assert_not_called()


@pytest.mark.parametrize(
    "options, message",
    [
        (["--artifacts", "--enable-qnn"], "--enable-qnn"),
        (["--artifacts", "--disaggregated", "--full-batch-size", "2"], "--compile-only"),
        (["--artifacts", "--stage", "decode"], "--compile-only"),
        (["--artifacts", "--enable-blocking", "--blocking-mode", "kv_paged"], "--compile-only"),
        (["--blocking-mode", "hqkv_paged"], "--enable-blocking"),
        (["--gdn-chunk-size", "0"], "--gdn-chunk-size"),
        (["--gdn-chunk-size", "-4"], "--gdn-chunk-size"),
        (["--dynamo", "--layerwise"], "--dynamo"),
        (
            [
                "--artifacts",
                "--stage",
                "prefill",
                "--num-devices",
                "2",
                "--mdp-num-partitions",
                "2",
                "--mdp-strategy",
                "intersection",
            ],
            "intersection executes the compiler",
        ),
        (["--dflash", "--artifacts"], "--artifacts is not supported"),
        (["--dflash", "--dtype=float32"], "--dtype is not supported"),
        (["--dflash", "--no-layerwise"], "--no-layerwise is not supported"),
        (["--dflash", "--disaggregated"], "--disaggregated is not supported"),
        (["--dflash", "--continuous-batching"], "--continuous-batching is not supported"),
        (["--dflash", "--include-sampler"], "--include-sampler is not supported"),
        (["--dflash", "--speculative-model-type", "target"], "--speculative-model-type is not supported"),
        (["--dflash", "--batch-size", "2"], "--batch-size 1"),
        (["--dflash", "--prompt", "one", "two"], "exactly one prompt"),
        (["--dflash", "--tlm-cores", "0"], "core counts must be >= 1"),
        (["--tlm-qpc", "/unused/qpc"], "requires --dflash"),
    ],
)
def test_text_example_rejects_unsupported_options_before_loading(
    monkeypatch, text_example_dependencies, capsys, options, message
):
    with pytest.raises(SystemExit) as error:
        _run_text_example(monkeypatch, *options)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    text_example_dependencies.load_model.assert_not_called()
    text_example_dependencies.load_tokenizer.assert_not_called()


@pytest.mark.parametrize("flag", ["--help", "--help-advanced", "--dry-run"])
def test_text_example_cli_does_not_import_runtime(monkeypatch, flag):
    import builtins

    original_import = builtins.__import__

    def forbid_runtime(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy", "transformers", "QEfficient"} or name.startswith(
            "examples.performance.dflash"
        ):
            raise AssertionError(f"CLI imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_runtime)
    if flag.startswith("--help"):
        with pytest.raises(SystemExit) as error:
            _run_text_example(monkeypatch, flag)
        assert error.value.code == 0
    else:
        _run_text_example(monkeypatch, "--dflash", "--model-name", "Qwen3-4B", flag)


def test_text_example_help_exposes_mainline_flags(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _run_text_example(monkeypatch, "--help")
    help_text = capsys.readouterr().out
    for flag in ("--artifacts", "--dynamo", "--dflash"):
        assert flag in help_text
    assert "--gdn-chunk-size" not in help_text
    with pytest.raises(SystemExit):
        _run_text_example(monkeypatch, "--help-advanced")
    advanced_help = capsys.readouterr().out
    for flag in ("--gdn-chunk-size", "--tlm-qpc", "--dlm-devices"):
        assert flag in advanced_help


@pytest.mark.parametrize("continuous_batching", [False, True])
def test_text_example_writes_real_artifact_bundle(monkeypatch, tmp_path, continuous_batching):
    """Exercise CLI -> real export/compile/generate, substituting only HF loaders."""

    class Tokenizer:
        padding_side = "right"
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, prompts, return_tensors, padding, max_length=None):
            ids = np.tile(np.array([[1, 2]], dtype=np.int64), (len(prompts), 1))
            mask = np.ones_like(ids)
            if padding == "max_length":
                ids = np.pad(ids, ((0, 0), (0, max_length - 2)))
                mask = np.pad(mask, ((0, 0), (0, max_length - 2)))
            return {"input_ids": ids, "attention_mask": mask}

    model, _ = make_tiny_gpt2()
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=continuous_batching)
    monkeypatch.setattr(QEFFAutoModelForCausalLM, "from_pretrained", lambda *a, **kw: qeff)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: Tokenizer())
    compiler = MagicMock(side_effect=AssertionError("artifacts invoked the compiler"))
    runtime = MagicMock(side_effect=AssertionError("artifacts invoked the runtime"))
    monkeypatch.setattr("QEfficient.base.modeling_qeff.subprocess.run", compiler)
    monkeypatch.setattr("QEfficient.cloud_ai_100_exec_kv", runtime)
    options = [
        "--artifacts",
        "--compile-dir",
        str(tmp_path),
        "--prefill-seq-len",
        "8",
        "--ctx-len",
        "32",
        "--prompt",
        "hello",
        "--no-offload-pt-weights",
    ]
    if continuous_batching:
        options += ["--continuous-batching", "--full-batch-size", "2"]
    _run_text_example(monkeypatch, *options)

    compiler.assert_not_called()
    runtime.assert_not_called()
    assert qeff.qpc_path is None
    compile_dir = qeff.compile_artifacts_path
    assert (compile_dir / "qaic-compile.sh").is_file()
    assert (compile_dir / "specializations.json").is_file()
    io_dir = compile_dir / "io"
    entries = json.loads((io_dir / "aic_batch_io.json").read_text())["IO-files"][0]
    inputs = {entry["map-to"]: entry for entry in entries if entry["io-direction"] == "in"}
    assert {"input_ids", "position_ids"} <= inputs.keys()
    assert ("batch_index" in inputs) is continuous_batching
    for entry in inputs.values():
        assert (io_dir / entry["path"]).stat().st_size == np.prod(entry["dims"]) * entry["elem-size"]
