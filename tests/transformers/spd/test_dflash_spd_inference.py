# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Full end-to-end DFlash speculative-decoding test on real QAIC hardware.

Builds a tiny from-scratch TLM/DLM pair (fully random weights, no checkpoint
downloads), compiles both to real QPCs, and drives them through
``QEfficient.generation.dflash_generation.run_spd_inference_single`` -- the
actual SPD decode loop used by the DFlash example scripts -- via two real
``QAICInferenceSession``s. This validates the mechanism end to end (prefill,
draft/target exchange, accept/reject bookkeeping); it does not assert
numerical accuracy, since the weights are random.
"""

import os

import numpy as np
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.generation.dflash_generation import run_spd_inference_single

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


def _make_tiny_qwen3_config():
    return Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )


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
