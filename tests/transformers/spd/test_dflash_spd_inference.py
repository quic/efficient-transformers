# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
DFlash speculative-decoding tests.

Includes a host-side regression test for generation-length enforcement and an
end-to-end QAIC test that builds a tiny from-scratch TLM/DLM pair, compiles both
to real QPCs, and drives them through the SPD decode loop.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.generation.dflash_generation import (
    SpecDecodingMetrics,
    run_spd_inference_gemma4,
    run_spd_inference_single,
)
from QEfficient.utils.constants import ONNX_EXPORT_EXAMPLE_SEQ_LEN

VOCAB_SIZE = 64
HIDDEN_SIZE = 64
BLOCK_SIZE = 4  # DFlash decode block size; also the DLM's prefill_seq_len
# TLM prompt-chunk size for prefill. Must differ from BLOCK_SIZE: the TLM's decode
# specialization seq_len is always overridden to BLOCK_SIZE (see
# build_decode_specialization's dflash_tlm branch), so a prefill spec with the same
# seq_len would be indistinguishable from it and the AIC compiler would reject the
# network ("No input that uniquely identifies specialization"). A multiple of
# BLOCK_SIZE keeps the prefill loop's sub-block splitting exact (no remainder).
PROMPT_CHUNK_SIZE = 2 * BLOCK_SIZE
CTX_LEN = 32
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
MASK_TOKEN_ID = 2


class _FakeTokenizer:
    """Minimal stand-in implementing only what run_spd_inference_single calls:
    __call__(prompts, return_tensors="np", padding=...) and pad/eos token ids.
    Deterministically maps each prompt string to a short fixed-length id sequence,
    keeping everything offline and within the tiny vocab."""

    pad_token_id = PAD_TOKEN_ID
    eos_token_id = EOS_TOKEN_ID

    def __call__(self, prompts, return_tensors="np", padding=True, max_length=None):
        assert return_tensors == "np"
        raw_ids = [[3 + (i % (VOCAB_SIZE - 3)) for i in range(1, len(p) + 1)] for p in prompts]
        length = max_length if max_length is not None else max(len(ids) for ids in raw_ids)
        input_ids = np.full((len(prompts), length), self.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((len(prompts), length), dtype=np.int64)
        for i, ids in enumerate(raw_ids):
            n = min(len(ids), length)
            input_ids[i, :n] = ids[:n]
            attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeTLMSession:
    def __init__(self):
        self.run_count = 0

    def set_buffers(self, _buffers):
        pass

    def run(self, inputs):
        self.run_count += 1
        seq_len = inputs["input_ids"].shape[1]
        hidden_states = np.zeros((1, seq_len, HIDDEN_SIZE), dtype=np.float32)
        if self.run_count == 1:
            logits = np.zeros((1, seq_len), dtype=np.int32)
            logits[:, 1] = 5
        else:
            logits = np.array([[6, 7, 8, 9]], dtype=np.int32)
        return {"logits": logits, "hidden_states": hidden_states}


class _FakeDLMSession:
    def set_buffers(self, _buffers):
        pass

    def run(self, _inputs):
        logits = np.zeros((1, BLOCK_SIZE, VOCAB_SIZE), dtype=np.float32)
        token_ids = np.array([5, 6, 7, 8])
        logits[0, np.arange(BLOCK_SIZE), token_ids] = 1
        return {"logits": logits}


class _FakeGemmaTLMSession:
    """Tracks static vision-buffer handling in the Gemma4 SPD path."""

    class _Binding:
        def __init__(self, name, index, dims, dtype="fp32"):
            self.name = name
            self.index = index
            self.dims = dims
            self.type = dtype

    def __init__(self):
        self.input_names = ["input_ids", "position_ids", "vision_embeds"]
        self.output_names = ["logits", "hidden_states"]
        self.bindings = [
            self._Binding("input_ids", 0, [1, BLOCK_SIZE]),
            self._Binding("position_ids", 1, [1, BLOCK_SIZE]),
            self._Binding("vision_embeds", 2, [1, 2, HIDDEN_SIZE]),
            self._Binding("logits", 3, [1, BLOCK_SIZE, VOCAB_SIZE]),
            self._Binding("hidden_states", 4, [1, BLOCK_SIZE, HIDDEN_SIZE]),
        ]
        self.binding_index_map = {binding.name: binding.index for binding in self.bindings}
        self.allowed_shapes = [
            [
                (8, [1, BLOCK_SIZE]),
                (8, [1, BLOCK_SIZE]),
                (4, [1, 2, HIDDEN_SIZE]),
                (4, [1, BLOCK_SIZE, VOCAB_SIZE]),
                (4, [1, BLOCK_SIZE, HIDDEN_SIZE]),
            ]
        ]
        self.aic_to_np_dtype_mapping = {"fp32": np.dtype(np.float32)}
        self.vision_buffer_sets = 0
        self.run_vision_feeds = 0
        self.skipped_buffers = []

    def set_buffers(self, buffers):
        if "vision_embeds" in buffers:
            self.vision_buffer_sets += 1

    def skip_buffers(self, buffers):
        self.skipped_buffers.extend(buffers)

    def run(self, inputs):
        self.run_vision_feeds += "vision_embeds" in inputs
        seq_len = inputs["input_ids"].shape[1]
        logits = np.zeros((1, seq_len, VOCAB_SIZE), dtype=np.float32)
        token_ids = np.resize(np.array([5, 6, 7, 8]), seq_len)
        logits[0, np.arange(seq_len), token_ids] = 1
        return {
            "logits": logits,
            "hidden_states": np.zeros((1, seq_len, HIDDEN_SIZE), dtype=np.float32),
        }


def test_dflash_acceptance_rate_uses_accepted_tokens():
    metrics = SpecDecodingMetrics(block_size=BLOCK_SIZE)
    metrics.total_accepted_tokens = 6
    metrics.total_generated_tokens = 9
    metrics.num_total_iters = 3

    assert metrics.acceptance_rate() == 2.0


def test_dflash_generation_len_caps_partial_block():
    metrics = run_spd_inference_single(
        prompt_text="hi",
        tokenizer=_FakeTokenizer(),
        dlm_session=_FakeDLMSession(),
        tlm_session=_FakeTLMSession(),
        mask_token_id=MASK_TOKEN_ID,
        vocab_size=VOCAB_SIZE,
        prompt_chunk_size=PROMPT_CHUNK_SIZE,
        ctx_len=CTX_LEN,
        block_size=BLOCK_SIZE,
        max_iterations=5,
        hidden_size=HIDDEN_SIZE,
        generation_len=1,
    )

    assert metrics.generated_ids == [6]
    assert metrics.generated_sources == ["dlm"]
    assert metrics.total_generated_tokens == 1
    assert metrics.total_accepted_tokens == 1
    assert metrics.num_total_iters == 1


def test_dflash_rejects_padded_prompt_longer_than_context():
    with pytest.raises(ValueError, match=r"ctx_len \(7\) must be greater than or equal to padded_len \(8\)"):
        run_spd_inference_single(
            prompt_text="hi",
            tokenizer=_FakeTokenizer(),
            dlm_session=_FakeDLMSession(),
            tlm_session=_FakeTLMSession(),
            mask_token_id=MASK_TOKEN_ID,
            vocab_size=VOCAB_SIZE,
            prompt_chunk_size=PROMPT_CHUNK_SIZE,
            ctx_len=PROMPT_CHUNK_SIZE - 1,
            block_size=BLOCK_SIZE,
            max_iterations=5,
            hidden_size=HIDDEN_SIZE,
            generation_len=1,
        )


def _make_tiny_qwen3_config():
    return Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        # Draft export traces context and noise at consecutive position ranges.
        max_position_embeddings=max(CTX_LEN, 2 * ONNX_EXPORT_EXAMPLE_SEQ_LEN),
        head_dim=32,
    )


def test_gemma4_binds_vision_embeddings_once_and_skips_them_for_decode():
    tlm_session = _FakeGemmaTLMSession()
    run_spd_inference_gemma4(
        prompt_text="ignored when input_ids are provided",
        tokenizer=_FakeTokenizer(),
        dlm_session=_FakeDLMSession(),
        tlm_session=tlm_session,
        mask_token_id=MASK_TOKEN_ID,
        vocab_size=VOCAB_SIZE,
        prompt_chunk_size=BLOCK_SIZE,
        ctx_len=CTX_LEN,
        block_size=BLOCK_SIZE,
        max_iterations=1,
        hidden_size=HIDDEN_SIZE,
        generation_len=1,
        input_ids=np.array([[3, 4]], dtype=np.int64),
        vision_embeds=np.zeros((1, 2, HIDDEN_SIZE), dtype=np.float32),
    )

    assert tlm_session.vision_buffer_sets == 1
    assert tlm_session.run_vision_feeds == 0
    assert tlm_session.skipped_buffers == ["vision_embeds"]


@pytest.mark.on_qaic
def test_dflash_spd_inference(manual_cleanup):
    tlm_model = QEFFAutoModelForCausalLM(
        Qwen3ForCausalLM(_make_tiny_qwen3_config()).eval(), qaic_config={"target_layer_ids": [1]}
    )
    dlm_model = QEFFAutoModelForCausalLM(
        Qwen3ForCausalLM(_make_tiny_qwen3_config()).eval(), qaic_config={"dflash_dlm": True}
    )

    tlm_qpc = tlm_model.compile(
        prefill_seq_len=PROMPT_CHUNK_SIZE,
        ctx_len=CTX_LEN,
        num_cores=2,
        num_devices=1,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        dflash_block_size=BLOCK_SIZE,
    )
    dlm_qpc = dlm_model.compile(
        prefill_seq_len=BLOCK_SIZE,
        ctx_len=CTX_LEN,
        num_cores=2,
        num_devices=1,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        prefill_only=True,
    )

    tlm_session = QAICInferenceSession(tlm_qpc)
    dlm_session = QAICInferenceSession(dlm_qpc)
    tlm_session.skip_buffers(
        {
            x
            for x in tlm_session.input_names + tlm_session.output_names
            if x.startswith("past_") or x.endswith("_RetainedState")
        }
    )
    dlm_session.skip_buffers(
        {
            x
            for x in dlm_session.input_names + dlm_session.output_names
            if x.startswith("past_") or x.endswith("_RetainedState")
        }
    )

    metrics = run_spd_inference_single(
        prompt_text="hi",
        tokenizer=_FakeTokenizer(),
        dlm_session=dlm_session,
        tlm_session=tlm_session,
        mask_token_id=MASK_TOKEN_ID,
        vocab_size=VOCAB_SIZE,
        prompt_chunk_size=PROMPT_CHUNK_SIZE,
        ctx_len=CTX_LEN,
        block_size=BLOCK_SIZE,
        max_iterations=5,
        hidden_size=HIDDEN_SIZE,
        generation_len=8,
    )

    assert metrics.num_total_iters > 0
    assert metrics.total_generated_tokens > 0
    assert len(metrics.generated_ids) == len(metrics.generated_sources)
    assert all(source in ("dlm", "tlm") for source in metrics.generated_sources)
    assert os.path.isfile(os.path.join(os.path.dirname(tlm_qpc), "qconfig.json"))
    assert os.path.isfile(os.path.join(os.path.dirname(dlm_qpc), "qconfig.json"))

    manual_cleanup([tlm_model.onnx_path, dlm_model.onnx_path])


@pytest.fixture
def dflash_text_example(monkeypatch):
    from examples.performance.dflash import basic_inference_text as example

    class Tokenizer(_FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def decode(self, tokens, **kwargs):
            return " ".join(str(token) for token in tokens)

    tokenizer = Tokenizer()
    draft = _FakeDLMSession()
    target = _FakeTLMSession()
    target.binding_index_map = {"input_ids": 0}
    target.bindings = [SimpleNamespace(dims=(1, PROMPT_CHUNK_SIZE))]
    target.allowed_shapes = [[(np.int64, (1, size))] for size in (PROMPT_CHUNK_SIZE, BLOCK_SIZE)]
    config = SimpleNamespace(
        dflash_config={"mask_token_id": MASK_TOKEN_ID},
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=VOCAB_SIZE,
    )
    monkeypatch.setattr(example, "compile_tlm_qpc", MagicMock(return_value="compiled-tlm"))
    monkeypatch.setattr(example, "compile_dlm_qpc", MagicMock(return_value="compiled-dlm"))
    monkeypatch.setattr(example, "load_spd_sessions", MagicMock(return_value=(draft, target)))
    monkeypatch.setattr(example.AutoConfig, "from_pretrained", MagicMock(return_value=config))
    monkeypatch.setattr(example.AutoTokenizer, "from_pretrained", MagicMock(return_value=tokenizer))
    return example


def _dflash_text_options():
    return dict(
        model_name="Qwen/Qwen3-4B",
        prompt="hi",
        ctx_len=CTX_LEN,
        prefill_seq_len=PROMPT_CHUNK_SIZE,
        generation_len=5,
        iteration=5,
        tlm_devices=[0],
        dlm_devices=[1],
    )


@pytest.mark.parametrize("reuse_target, reuse_draft", [(False, False), (True, False), (False, True), (True, True)])
def test_dflash_text_example_compile_only_and_independent_qpc_reuse(
    dflash_text_example, tmp_path, reuse_target, reuse_draft
):
    example = dflash_text_example
    options = _dflash_text_options()
    options.update(compile_only=True, compile_dir=str(tmp_path), tlm_cores=4, dlm_cores=6)
    if reuse_target:
        options["tlm_qpc"] = "existing-tlm"
    if reuse_draft:
        options["dlm_qpc"] = "existing-dlm"
    assert example.run_text_dflash(**options) is None

    assert example.compile_tlm_qpc.call_count == int(not reuse_target)
    assert example.compile_dlm_qpc.call_count == int(not reuse_draft)
    for stage, compile_model, reused, cores in (
        ("tlm", example.compile_tlm_qpc, reuse_target, 4),
        ("dlm", example.compile_dlm_qpc, reuse_draft, 6),
    ):
        if not reused:
            assert compile_model.call_args.kwargs["compile_dir"] == str(tmp_path / stage)
            assert compile_model.call_args.kwargs["num_cores"] == cores
            assert compile_model.call_args.kwargs["num_devices"] == 1
            assert (tmp_path / stage).is_dir()
    example.load_spd_sessions.assert_not_called()
    example.AutoTokenizer.from_pretrained.assert_not_called()
    example.AutoConfig.from_pretrained.assert_not_called()


def test_dflash_text_example_uses_existing_runtime_and_generation_limit(dflash_text_example):
    example = dflash_text_example
    options = _dflash_text_options()
    options.update(tlm_qpc="existing-tlm", dlm_qpc="existing-dlm")
    result = example.run_text_dflash(**options)

    reference = run_spd_inference_single(
        prompt_text="hi",
        tokenizer=_FakeTokenizer(),
        dlm_session=_FakeDLMSession(),
        tlm_session=_FakeTLMSession(),
        mask_token_id=MASK_TOKEN_ID,
        vocab_size=VOCAB_SIZE,
        prompt_chunk_size=PROMPT_CHUNK_SIZE,
        ctx_len=CTX_LEN,
        block_size=BLOCK_SIZE,
        max_iterations=5,
        hidden_size=HIDDEN_SIZE,
        generation_len=5,
    )
    assert result.generated_ids == reference.generated_ids
    assert result.generated_sources == reference.generated_sources
    assert result.total_generated_tokens == 5
    example.load_spd_sessions.assert_called_once_with("existing-tlm", "existing-dlm", [0], [1])


def test_dflash_text_example_rejects_missing_decode_specialization(dflash_text_example):
    target = dflash_text_example.load_spd_sessions.return_value[1]
    target.allowed_shapes = [[(np.int64, (1, PROMPT_CHUNK_SIZE))]]
    with pytest.raises(ValueError, match="no decode specialization"):
        dflash_text_example.run_text_dflash(**_dflash_text_options())


@pytest.mark.parametrize("model_name", ["unknown-model", "Qwen3-VL-32B-Instruct"])
def test_dflash_text_example_rejects_unsupported_model_before_compile(dflash_text_example, model_name):
    import argparse

    options = _dflash_text_options()
    options["model_name"] = model_name
    with pytest.raises((ValueError, argparse.ArgumentTypeError)):
        dflash_text_example.run_text_dflash(**options)
    dflash_text_example.compile_tlm_qpc.assert_not_called()
    dflash_text_example.compile_dlm_qpc.assert_not_called()


def test_canonical_dflash_routes_to_shared_runner(monkeypatch, dflash_text_example, tmp_path):
    from examples.text_generation.basic_inference import main

    run = MagicMock()
    monkeypatch.setattr(dflash_text_example, "run_text_dflash", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basic_inference.py",
            "--dflash",
            "--model-name",
            "Qwen3-4B",
            "--prompt",
            "hi",
            "--device-group",
            "2,3",
            "--dlm-devices",
            "1",
            "--tlm-cores",
            "4",
            "--dlm-cores",
            "6",
            "--tlm-qpc",
            "existing-tlm",
            "--tlm-hf-path",
            "local-target",
            "--compile-only",
            "--compile-dir",
            str(tmp_path),
            "--ctx-len",
            "256",
            "--generation-len",
            "12",
            "--iteration",
            "7",
            "--format-prompt",
            "--category",
            "math",
        ],
    )
    main()
    run.assert_called_once()
    kwargs = run.call_args.kwargs
    assert kwargs["tlm_devices"] == [2, 3]
    assert kwargs["dlm_devices"] == [1]
    assert kwargs["tlm_cores"] == 4
    assert kwargs["dlm_cores"] == 6
    assert kwargs["tlm_qpc"] == "existing-tlm"
    assert kwargs["tlm_hf_path"] == "local-target"
    assert kwargs["compile_dir"] == str(tmp_path)
    assert kwargs["compile_only"] is True
    assert kwargs["ctx_len"] == 256
    assert kwargs["generation_len"] == 12
    assert kwargs["iteration"] == 7
    assert kwargs["format_prompt"] is True
    assert kwargs["category"] == "math"


def test_standalone_dflash_cli_preserves_existing_options(monkeypatch, dflash_text_example):
    run = MagicMock()
    monkeypatch.setattr(dflash_text_example, "run_text_dflash", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "basic_inference_text.py",
            "--model_name",
            "Qwen3-4B",
            "--prompt",
            "hi",
            "--tlm_qpc",
            "existing-tlm",
            "--dlm_qpc",
            "existing-dlm",
            "--tlm_devices",
            "2",
            "--dlm_devices",
            "3",
        ],
    )
    dflash_text_example.main()
    assert run.call_args.kwargs["tlm_qpc"] == "existing-tlm"
    assert run.call_args.kwargs["dlm_qpc"] == "existing-dlm"
    assert run.call_args.kwargs["tlm_devices"] == [2]
    assert run.call_args.kwargs["dlm_devices"] == [3]
    assert run.call_args.kwargs["ctx_len"] == 4096
    assert run.call_args.kwargs["iteration"] == 300
