# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from transformers import AutoTokenizer
from typing import List
import numpy as np
import pytest

configs = [
    pytest.param(
        # device_group, num_speculative_tokens, prompt_len, ctx_len, prefill_bsz, full_batch_size, model_name, id
        [0], 5, 32, 128, 1, 8, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", id="llama"
    ),
]

@pytest.mark.parametrize("device_group,num_speculative_tokens,prompt_len,ctx_len,prefill_bsz,full_batch_size,model_name", configs)
def test_llama_tlm_logit_dims(
    device_group: List[int],
    num_speculative_tokens: int,
    prompt_len: int,
    ctx_len: int,
    prefill_bsz: int,
    full_batch_size: int,
    model_name: str
):

    # get vocab size
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)

    # export_and_compile tlm model
    qeff_model = AutoModelForCausalLM.from_pretrained(model_name, num_speculative_tokens=num_speculative_tokens)
    qpc_path: str = qeff_model.export_and_compile(
        num_cores=16,
        device_group=device_group,
        batch_size=prefill_bsz,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        mxfp6=True,
        mxint8=True,
        full_batch_size=full_batch_size
    )

    # init qaic session 
    session = QAICInferenceSession(qpc_path, device_ids=device_group)
    # skip inputs/outputs buffers
    session.skip_buffers(
        set([x for x in session.input_names if x.startswith("past_")])
    )
    session.skip_buffers(
        set([x for x in session.output_names if x.endswith("_RetainedState")])
    )
    # prefill dummy inputs
    prefill_inputs = dict(
        input_ids = np.zeros((prefill_bsz, prompt_len), dtype=np.int64),
        position_ids = np.arange(prompt_len, dtype=np.int64).reshape(-1,1).repeat(prefill_bsz,1).transpose(),
        batch_index= np.arange(prefill_bsz, dtype=np.int64).reshape(prefill_bsz,1)
    )
    # decode dummy inputs
    decode_inputs = dict(
        input_ids = np.zeros((full_batch_size, num_speculative_tokens+1), dtype=np.int64),
        position_ids = np.full((full_batch_size, num_speculative_tokens+1), -1, dtype=np.int64),
        batch_index=np.arange(full_batch_size, dtype=np.int64).reshape(-1,1)
    )
    # create dummy logits
    prefill_logits = dict(logits=np.random.randn(prefill_bsz, prompt_len, vocab_size).astype(np.float32))
    decode_logits = dict(logits=np.random.randn(full_batch_size, num_speculative_tokens+1, vocab_size).astype(np.float32))
    # get prefill/decode logits
    session.set_buffers(prefill_logits)
    prefill_outputs = session.run(prefill_inputs)
    session.set_buffers(decode_logits)
    decode_outputs = session.run(decode_inputs)


    # assert expected logit dims
    assert prefill_logits["logits"].shape == prefill_outputs["logits"].shape
    assert decode_logits["logits"].shape == decode_outputs["logits"].shape


@pytest.mark.parametrize("device_group,num_speculative_tokens,prompt_len,ctx_len,prefill_bsz,full_batch_size,model_name", configs)
def test_llama_dlm_logit_dims(
    device_group: List[int],
    num_speculative_tokens: int,
    prompt_len: int,
    ctx_len: int,
    prefill_bsz: int,
    full_batch_size: int,
    model_name: str
):

    # get vocab size
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)

    # export_and_compile tlm model
    qeff_model = AutoModelForCausalLM.from_pretrained(model_name, is_dlm=True)
    qpc_path: str = qeff_model.export_and_compile(
        num_cores=16,
        device_group=device_group,
        batch_size=prefill_bsz,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        mxfp6=True,
        mxint8=True,
        full_batch_size=full_batch_size
    )

    # init qaic session 
    session = QAICInferenceSession(qpc_path, device_ids=device_group)
    # skip inputs/outputs buffers
    session.skip_buffers(
        set([x for x in session.input_names if x.startswith("past_")])
    )
    session.skip_buffers(
        set([x for x in session.output_names if x.endswith("_RetainedState")])
    )
    # prefill dummy inputs
    prefill_inputs = dict(
        input_ids = np.zeros((prefill_bsz, prompt_len), dtype=np.int64),
        position_ids = np.arange(prompt_len, dtype=np.int64).reshape(-1,1).repeat(prefill_bsz,1).transpose(),
        batch_index=np.arange(prefill_bsz, dtype=np.int64).reshape(-1,1)
    )
    # decode-1 dummy inputs
    decode1_inputs = dict(
        input_ids = np.zeros((full_batch_size, 1), dtype=np.int64),
        position_ids = np.full((full_batch_size, 1), -1, dtype=np.int64),
        batch_index=np.arange(full_batch_size, dtype=np.int64).reshape(-1,1)
    )
    # decode-2 dummy inputs
    decode2_inputs = dict(
        input_ids = np.zeros((full_batch_size, 2), dtype=np.int64),
        position_ids = np.full((full_batch_size, 2), -1, dtype=np.int64),
        batch_index=np.arange(full_batch_size, dtype=np.int64).reshape(-1,1)
    )
    # create dummy logits
    prefill_logits = dict(logits=np.random.randn(prefill_bsz, 1, vocab_size).astype(np.float32))
    decode_logits = dict(logits=np.random.randn(full_batch_size, 1, vocab_size).astype(np.float32))
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