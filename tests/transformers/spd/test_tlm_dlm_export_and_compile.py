# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional

import numpy as np
import pytest
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.device_utils import get_available_device_id

configs = [
    pytest.param(
        2,  # num_speculative_tokens
        32,  # prefill_seq_len
        128,  # ctx_len
        1,  # prefill_bsz
        8,  # full_batch_size
        "JackFram/llama-68m",  # model_name
        id="CB llama",
    ),
    pytest.param(
        2,  # num_speculative_tokens
        32,  # prefill_seq_len
        128,  # ctx_len
        1,  # prefill_bsz
        None,  # full_batch_size
        "JackFram/llama-68m",  # model_name
        id="non-CB llama",
    ),
]


@pytest.mark.parametrize(
    "num_speculative_tokens,prefill_seq_len,ctx_len,prefill_bsz,full_batch_size,model_name",
    configs,
)
def test_llama_tlm_logit_dims(
    num_speculative_tokens: int,
    prefill_seq_len: int,
    ctx_len: int,
    prefill_bsz: int,
    full_batch_size: Optional[int],
    model_name: str,
):
    device_group = get_available_device_id()
    if not device_group:
        pytest.skip("No available devices to run model on Cloud AI 100")
    # get vocab size
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)

    # export and compile tlm model
    continuous_batching = full_batch_size is not None
    qeff_model = AutoModelForCausalLM.from_pretrained(
        model_name, continuous_batching=continuous_batching, is_tlm=True
    )
    qpc_path: str = qeff_model.compile(
        num_devices=len(device_group),
        num_cores=16,
        batch_size=prefill_bsz,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        mxfp6_matmul=True,
        full_batch_size=full_batch_size,
        num_speculative_tokens=num_speculative_tokens,
    )

    # init qaic session
    session = QAICInferenceSession(qpc_path, device_ids=device_group)
    # skip inputs/outputs buffers
    session.skip_buffers(set([x for x in session.input_names if x.startswith("past_")]))
    session.skip_buffers(set([x for x in session.output_names if x.endswith("_RetainedState")]))
    # prefill dummy inputs
    num_logits_to_keep = num_speculative_tokens + 1
    prefill_inputs = dict(
        input_ids=np.zeros((prefill_bsz, prefill_seq_len), dtype=np.int64),
        position_ids=np.arange(prefill_seq_len, dtype=np.int64).reshape(-1, 1).repeat(prefill_bsz, 1).transpose(),
        num_logits_to_keep=np.arange(1).reshape(1, 1),
    )
    # decode dummy inputs
    decode_bsz = full_batch_size if full_batch_size is not None else prefill_bsz
    decode_inputs = dict(
        input_ids=np.zeros((decode_bsz, num_logits_to_keep), dtype=np.int64),
        position_ids=np.full((decode_bsz, num_logits_to_keep), -1, dtype=np.int64),
        num_logits_to_keep=np.arange(num_logits_to_keep).reshape(num_logits_to_keep, 1),
    )
    if full_batch_size is not None:
        prefill_inputs["batch_index"] = np.arange(prefill_bsz, dtype=np.int64).reshape(prefill_bsz, 1)
        decode_inputs["batch_index"] = np.arange(decode_bsz, dtype=np.int64).reshape(-1, 1)
    # create dummy logits
    prefill_logits = dict(logits=np.random.randn(prefill_bsz, 1, vocab_size).astype(np.float32))
    decode_logits = dict(logits=np.random.randn(decode_bsz, num_logits_to_keep, vocab_size).astype(np.float32))
    # get prefill/decode logits
    session.set_buffers(prefill_logits)
    prefill_outputs = session.run(prefill_inputs)
    session.set_buffers(decode_logits)
    decode_outputs = session.run(decode_inputs)

    # assert expected logit dims
    assert prefill_logits["logits"].shape == prefill_outputs["logits"].shape
    assert decode_logits["logits"].shape == decode_outputs["logits"].shape
