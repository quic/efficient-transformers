# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

configs = [
    pytest.param(
        [0],  # device_group
        2,  # num_speculative_tokens
        32,  # prefill_seq_len
        128,  # ctx_len
        1,  # prefill_bsz
        8,  # full_batch_size
        "JackFram/llama-68m",  # model_name
        True,  # continuous_batching
        id="CB llama",
    ),
    pytest.param(
        [0],  # device_group
        2,  # num_speculative_tokens
        32,  # prefill_seq_len
        128,  # ctx_len
        1,  # prefill_bsz
        None,  # full_batch_size
        "JackFram/llama-68m",  # model_name
        False,  # continuous_batching
        id="non-CB llama",
    ),
]


@pytest.mark.parametrize(
    "device_group,num_speculative_tokens,prefill_seq_len,ctx_len,prefill_bsz,full_batch_size,model_name,continuous_batching",
    configs,
)
def test_llama_tlm_logit_dims(
    device_group: List[int],
    num_speculative_tokens: int,
    prefill_seq_len: int,
    ctx_len: int,
    prefill_bsz: int,
    full_batch_size: Optional[int],
    model_name: str,
    continuous_batching: bool,
):
    # get vocab size
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)

    # export and compile tlm model
    qeff_model = AutoModelForCausalLM.from_pretrained(
        model_name, continuous_batching=continuous_batching, num_speculative_tokens=num_speculative_tokens
    )
    qpc_path: str = qeff_model.compile(
        num_devices=len(device_group),
        num_cores=14,
        batch_size=prefill_bsz,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        mxfp6_matmul=True,
        full_batch_size=full_batch_size,
    )

    # init qaic session
    session = QAICInferenceSession(qpc_path, device_ids=device_group)
    # skip inputs/outputs buffers
    session.skip_buffers(set([x for x in session.input_names if x.startswith("past_")]))
    session.skip_buffers(set([x for x in session.output_names if x.endswith("_RetainedState")]))
    # prefill dummy inputs
    prefill_inputs = dict(
        input_ids=np.zeros((prefill_bsz, prefill_seq_len), dtype=np.int64),
        position_ids=np.arange(prefill_seq_len, dtype=np.int64).reshape(-1, 1).repeat(prefill_bsz, 1).transpose(),
        num_logits_to_keep=torch.arange(num_speculative_tokens + 1).view(num_speculative_tokens + 1, 1).numpy(),
    )
    # decode dummy inputs
    num_logits_to_keep = num_speculative_tokens + 1
    decode_bsz = full_batch_size if full_batch_size is not None else prefill_bsz
    decode_inputs = dict(
        input_ids=np.zeros((decode_bsz, num_logits_to_keep), dtype=np.int64),
        position_ids=np.full((decode_bsz, num_logits_to_keep), -1, dtype=np.int64),
    )
    if full_batch_size is not None:
        prefill_inputs["batch_index"] = np.arange(prefill_bsz, dtype=np.int64).reshape(prefill_bsz, 1)
        decode_inputs["batch_index"] = np.arange(decode_bsz, dtype=np.int64).reshape(-1, 1)
    # create dummy logits
    prefill_logits = dict(logits=np.random.randn(prefill_bsz, num_logits_to_keep, vocab_size).astype(np.float32))
    decode_logits = dict(logits=np.random.randn(decode_bsz, num_logits_to_keep, vocab_size).astype(np.float32))
    # get prefill/decode logits
    session.set_buffers(prefill_logits)
    prefill_outputs = session.run(prefill_inputs)
    session.set_buffers(decode_logits)
    decode_outputs = session.run(decode_inputs)

    # assert expected logit dims
    assert prefill_logits["logits"].shape == prefill_outputs["logits"].shape
    assert decode_logits["logits"].shape == decode_outputs["logits"].shape


@pytest.mark.parametrize(
    "device_group,num_speculative_tokens,prefill_seq_len,ctx_len,prefill_bsz,full_batch_size,model_name,continuous_batching",
    configs,
)
def test_llama_dlm_logit_dims(
    device_group: List[int],
    num_speculative_tokens: int,
    prefill_seq_len: int,
    ctx_len: int,
    prefill_bsz: int,
    full_batch_size: Optional[int],
    model_name: str,
    continuous_batching: bool,
):
    # get vocab size
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)

    # export and compile tlm model
    qeff_model = AutoModelForCausalLM.from_pretrained(model_name, continuous_batching=continuous_batching, is_dlm=True)
    qpc_path: str = qeff_model.compile(
        num_devices=len(device_group),
        num_cores=16,
        batch_size=prefill_bsz,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        mxfp6_matmul=True,
        full_batch_size=full_batch_size,
    )

    # init qaic session
    session = QAICInferenceSession(qpc_path, device_ids=device_group)
    # skip inputs/outputs buffers
    session.skip_buffers(set([x for x in session.input_names if x.startswith("past_")]))
    session.skip_buffers(set([x for x in session.output_names if x.endswith("_RetainedState")]))
    # prefill dummy inputs
    prefill_inputs = dict(
        input_ids=np.zeros((prefill_bsz, prefill_seq_len), dtype=np.int64),
        position_ids=np.arange(prefill_seq_len, dtype=np.int64).reshape(-1, 1).repeat(prefill_bsz, 1).transpose(),
        batch_index=np.arange(prefill_bsz, dtype=np.int64).reshape(-1, 1),
    )
    # decode-1 dummy inputs
    decode_bsz = full_batch_size if full_batch_size is not None else prefill_bsz
    decode1_inputs = dict(
        input_ids=np.zeros((decode_bsz, 1), dtype=np.int64),
        position_ids=np.full((decode_bsz, 1), -1, dtype=np.int64),
        batch_index=np.arange(decode_bsz, dtype=np.int64).reshape(-1, 1),
    )
    # decode-2 dummy inputs
    decode2_inputs = dict(
        input_ids=np.zeros((decode_bsz, 2), dtype=np.int64),
        position_ids=np.full((decode_bsz, 2), -1, dtype=np.int64),
        batch_index=np.arange(decode_bsz, dtype=np.int64).reshape(-1, 1),
    )
    # create dummy logits
    prefill_logits = dict(logits=np.random.randn(prefill_bsz, 1, vocab_size).astype(np.float32))
    decode_logits = dict(logits=np.random.randn(decode_bsz, 1, vocab_size).astype(np.float32))
    # get prefill/decode logits
    session.set_buffers(prefill_logits)
    prefill_outputs = session.run(prefill_inputs)
    session.set_buffers(decode_logits)
    decode1_outputs = session.run(decode1_inputs)
    decode2_outputs = session.run(decode2_inputs)

    # assert expected logit dims
    assert prefill_logits["logits"].shape == prefill_outputs["logits"].shape
    assert decode_logits["logits"].shape == decode1_outputs["logits"].shape
    assert decode_logits["logits"].shape == decode2_outputs["logits"].shape
