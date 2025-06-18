# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List

import pytest

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.constants import Constants

configs = [
    pytest.param(
        "meta-llama/Llama-3.1-8B",  # model
        Constants.INPUT_STR,  # prompts
        32,  # prefill_seq_len
        256,  # ctx_len
        4,  # full_batch_size
        1,  # num_devices
        16,  # num_cores
        1,  # spec_length
        id="Llama-3.1-8B_32_256_4_1_16_1",
    ),
    pytest.param(
        "meta-llama/Llama-3.1-8B",  # model
        Constants.INPUT_STR,  # prompts
        32,  # prefill_seq_len
        256,  # ctx_len
        4,  # full_batch_size
        4,  # num_devices
        16,  # num_cores
        1,  # spec_length
        id="Llama-3.1-8B_32_256_4_4_16_1",
    ),
]


@pytest.mark.on_qaic
@pytest.mark.parametrize(
    "model, prompts, prefill_seq_len, ctx_len, full_batch_size, num_devices, num_cores, spec_length",
    configs,
)
def test_sampler_transform(
    model: str,
    prompts: List[str],
    prefill_seq_len: int,
    ctx_len: int,
    full_batch_size: int,
    num_devices: int,
    num_cores: int,
    spec_length: int,
):
    # Export and compile QEfficient models
    qaic_config = {
        "include_sampler": True,
        "return_pdfs": False,
        "max_top_k_ids": 512,
    }
    model_w_sampler = AutoModelForCausalLM.from_pretrained(model, continuous_batching=True, qaic_config=qaic_config)
    model_wo_sampler = AutoModelForCausalLM.from_pretrained(model, continuous_batching=True, qaic_config=None)
    model_w_sampler_qpc_path: str = model_w_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=num_devices,
        num_cores=num_cores,
        num_speculative_tokens=spec_length - 1,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_wo_sampler_qpc_path: str = model_wo_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=num_devices,
        num_cores=num_cores,
        num_speculative_tokens=spec_length - 1,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )

    # Init qaic session
    model_w_sampler_session = QAICInferenceSession(model_w_sampler_qpc_path)
    model_wo_sampler_session = QAICInferenceSession(model_wo_sampler_qpc_path)

    # Skip inputs/outputs buffers
    model_w_sampler_session.skip_buffers(set([x for x in model_w_sampler_session.input_names if x.startswith("past_")]))
    model_w_sampler_session.skip_buffers(
        set([x for x in model_w_sampler_session.output_names if x.endswith("_RetainedState")])
    )
    model_wo_sampler_session.skip_buffers(
        set([x for x in model_wo_sampler_session.input_names if x.startswith("past_")])
    )
    model_wo_sampler_session.skip_buffers(
        set([x for x in model_wo_sampler_session.output_names if x.endswith("_RetainedState")])
    )

    # Validate sampler inputs
    sampler_inputs = [
        "last_accepted_output_tokens",
        "repetition_penalties",
        "presence_penalties",
        "temperatures",
        "top_ks",
        "top_ps",
        "min_ps",
        "random_numbers",
    ]
    for input_name in sampler_inputs:
        assert (
            input_name in model_w_sampler_session.input_names
        ), f"Sampler input {input_name} not found in QPC compiled with Sampler"
        assert (
            input_name not in model_wo_sampler_session.input_names
        ), f"Sampler input {input_name} found in QPC compiled without Sampler"
