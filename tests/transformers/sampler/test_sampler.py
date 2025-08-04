# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List

import numpy as np
import pytest

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants

configs = [
    pytest.param(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # model
        Constants.INPUT_STR * 4,  # prompts
        32,  # prefill_seq_len
        256,  # ctx_len
        20,  # generation_len
        4,  # full_batch_size
        1,  # spec_length
    ),
]


@pytest.mark.on_qaic
@pytest.mark.parametrize(
    "model, prompts, prefill_seq_len, ctx_len, generation_len, full_batch_size, spec_length",
    configs,
)
def test_sampler_transform(
    model: str,
    prompts: List[str],
    prefill_seq_len: int,
    ctx_len: int,
    generation_len: int,
    full_batch_size: int,
    spec_length: int,
):
    """
    Test if `SamplerTransform` adds nodes at the output of a `QEffForCausalLM model` to enable the
    sampling of next tokens at the device (instead of the host) and returns the
    next tokens and/or probability distributions.
    """
    # Export and compile QEfficient models
    qaic_config = {
        "include_sampler": True,
        "return_pdfs": False,
        "max_top_k_ids": 512,
    }
    model_w_sampler = QEFFAutoModelForCausalLM.from_pretrained(model, continuous_batching=True, qaic_config=qaic_config)
    model_wo_sampler = QEFFAutoModelForCausalLM.from_pretrained(model, continuous_batching=True, qaic_config=None)
    model_w_sampler_qpc_path: str = model_w_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length - 1,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_wo_sampler_qpc_path: str = model_wo_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
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
    sampler_inputs = Constants.SAMPLER_INPUTS
    for input_name in sampler_inputs:
        assert (
            input_name in model_w_sampler_session.input_names
        ), f"Sampler input {input_name} not found in QPC compiled with On Device Sampler"
        assert (
            input_name not in model_wo_sampler_session.input_names
        ), f"Sampler input {input_name} found in QPC compiled without On Device Sampler"


@pytest.mark.on_qaic
@pytest.mark.parametrize(
    "model, prompts, prefill_seq_len, ctx_len, generation_len, full_batch_size, spec_length",
    configs,
)
def test_greedy_sampling(
    model: str,
    prompts: List[str],
    prefill_seq_len: int,
    ctx_len: int,
    generation_len: int,
    full_batch_size: int,
    spec_length: int,
):
    """
    Test greedy sampling with QPC compiled with and without On Device Sampling.
    """
    # Export and compile QEfficient models
    qaic_config = {
        "include_sampler": True,
        "return_pdfs": False,
        "max_top_k_ids": 512,
    }
    model_w_sampler = QEFFAutoModelForCausalLM.from_pretrained(model, continuous_batching=True, qaic_config=qaic_config)
    model_wo_sampler = QEFFAutoModelForCausalLM.from_pretrained(model, continuous_batching=True, qaic_config=None)
    model_w_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length - 1,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_wo_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length - 1,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )

    # Generate texts from prompts
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model)
    model_w_sampler_exec_info = model_w_sampler.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=None,
        generation_len=generation_len,
        include_sampler=True,
        return_pdfs=False,
        sampling_params={
            "repetition_penalties": np.array(1.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "presence_penalties": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            # "frequency_penalties": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "temperatures": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "top_ks": np.array(512, dtype=np.int32).repeat(full_batch_size).reshape(-1, 1),
            "top_ps": np.array(1.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "min_ps": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "random_numbers": np.array(0.0, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        },
    )
    model_wo_sampler_exec_info = model_wo_sampler.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=None,
        generation_len=generation_len,
        include_sampler=False,
        return_pdfs=False,
        sampling_params=None,
    )

    # Compare generated texts and ids
    assert (
        model_w_sampler_exec_info.generated_texts == model_wo_sampler_exec_info.generated_texts
    ), "Generated texts do not match"
    assert (
        model_w_sampler_exec_info.generated_ids == model_wo_sampler_exec_info.generated_ids
    ).all(), "Generated ids do not match"


@pytest.mark.on_qaic
@pytest.mark.parametrize(
    "model, prompts, prefill_seq_len, ctx_len, generation_len, full_batch_size, spec_length",
    configs,
)
def test_random_sampling(
    model: str,
    prompts: List[str],
    prefill_seq_len: int,
    ctx_len: int,
    generation_len: int,
    full_batch_size: int,
    spec_length: int,
):
    """
    Test random sampling with QPC compiled with and without On Device Sampling.
    """
    # Export and compile QEfficient models
    qaic_config = {
        "include_sampler": True,
        "return_pdfs": False,
        "max_top_k_ids": 512,
    }
    model_w_sampler = QEFFAutoModelForCausalLM.from_pretrained(model, continuous_batching=True, qaic_config=qaic_config)
    model_wo_sampler = QEFFAutoModelForCausalLM.from_pretrained(model, continuous_batching=True, qaic_config=None)
    model_w_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length - 1,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_wo_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=1,
        num_cores=16,
        num_speculative_tokens=spec_length - 1,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )

    # Generate texts from prompts
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model)
    model_w_sampler_exec_info = model_w_sampler.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=None,
        generation_len=generation_len,
        include_sampler=True,
        return_pdfs=False,
        sampling_params={
            "repetition_penalties": np.array(20.2, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "presence_penalties": np.array(10.5, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            # "frequency_penalties": np.array(0.5, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "temperatures": np.array(100.1, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "top_ks": np.array(54720, dtype=np.int32).repeat(full_batch_size).reshape(-1, 1),
            "top_ps": np.array(0.89, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "min_ps": np.array(0.6, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "random_numbers": np.array(0.26, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        },
    )
    model_wo_sampler_exec_info = model_wo_sampler.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=None,
        generation_len=generation_len,
        include_sampler=False,
        return_pdfs=False,
        sampling_params=None,
    )

    # Compare generated texts
    golden_texts = {
        "w_sampler": "Raymond and my favorite color, alongside reds or purples (I can’t have either as",
        "wo_sampler": "John Smith and I am a software engineer. I have been working in the industry for the past ",
    }
    golden_ids = {
        "w_sampler": [
            [
                21380,
                322,
                590,
                25448,
                2927,
                29892,
                19963,
                2654,
                29879,
                470,
                3708,
                2701,
                313,
                29902,
                508,
                30010,
                29873,
                505,
                2845,
                408,
            ]
        ],
        "wo_sampler": [
            [
                2259,
                7075,
                322,
                306,
                626,
                263,
                7047,
                22055,
                29889,
                306,
                505,
                1063,
                1985,
                297,
                278,
                13661,
                363,
                278,
                4940,
                29871,
            ]
        ],
    }
    for i in range(full_batch_size):
        assert (
            tokenizer.decode(model_w_sampler_exec_info.generated_ids[i][:generation_len]) == golden_texts["w_sampler"]
        ), "Sampler generated texts does not match"
        assert (
            model_w_sampler_exec_info.generated_ids[i][:generation_len] == golden_ids["w_sampler"]
        ).all(), "Sampler generated ids do not match"
        assert (
            tokenizer.decode(model_wo_sampler_exec_info.generated_ids[i][:generation_len]) == golden_texts["wo_sampler"]
        ), "Without sampler generated texts does not match"
        assert (
            model_wo_sampler_exec_info.generated_ids[i][:generation_len] == golden_ids["wo_sampler"]
        ).all(), "Without sampler generated ids do not match"
