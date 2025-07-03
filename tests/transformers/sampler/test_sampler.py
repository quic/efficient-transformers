# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List

import numpy as np
import pytest

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants

configs = [
    pytest.param(
        "meta-llama/Llama-3.1-8B",  # model
        Constants.INPUT_STR,  # prompts
        32,  # prefill_seq_len
        256,  # ctx_len
        20,  # generation_len
        4,  # full_batch_size
        1,  # num_devices
        [0],  # device_group
        16,  # num_cores
        1,  # spec_length
        1.9,  # repetition_penalty
        0.8,  # presence_penalty
        0.67,  # temperature
        54720,  # top_k
        0.89,  # top_p
        0.6,  # min_p
        0.26,  # random_number
        id="Llama-3.1-8B_32_256_4_1_16_1",
    ),
    # pytest.param(
    #     "meta-llama/Llama-3.1-8B",
    #     Constants.INPUT_STR,
    #     32,
    #     256,
    #     20,
    #     4,
    #     4,
    #     [0, 1, 2, 3],
    #     16,
    #     1,
    #     1.9,
    #     0.8,
    #     0.67,
    #     54720,
    #     0.89,
    #     0.6,
    #     0.26,
    #     id="Llama-3.1-8B_32_256_4_4_16_1",
    # ),
]


@pytest.mark.on_qaic
@pytest.mark.parametrize(
    "model, prompts, prefill_seq_len, ctx_len, generation_len, full_batch_size, num_devices, device_group, num_cores, spec_length, repetition_penalty, presence_penalty, temperature, top_k, top_p, min_p, random_number",
    configs,
)
def test_sampler_transform(
    model: str,
    prompts: List[str],
    prefill_seq_len: int,
    ctx_len: int,
    generation_len: int,
    full_batch_size: int,
    num_devices: int,
    device_group: List[int],
    num_cores: int,
    spec_length: int,
    repetition_penalty: float,
    presence_penalty: float,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    random_number: float,
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


@pytest.mark.on_qaic
@pytest.mark.parametrize(
    "model, prompts, prefill_seq_len, ctx_len, generation_len, full_batch_size, num_devices, device_group, num_cores, spec_length, repetition_penalty, presence_penalty, temperature, top_k, top_p, min_p, random_number",
    configs,
)
def test_greedy_sampling(
    model: str,
    prompts: List[str],
    prefill_seq_len: int,
    ctx_len: int,
    generation_len: int,
    full_batch_size: int,
    num_devices: int,
    device_group: List[int],
    num_cores: int,
    spec_length: int,
    repetition_penalty: float,
    presence_penalty: float,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    random_number: float,
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
    model_w_sampler = AutoModelForCausalLM.from_pretrained(model, continuous_batching=True, qaic_config=qaic_config)
    model_wo_sampler = AutoModelForCausalLM.from_pretrained(model, continuous_batching=True, qaic_config=None)
    model_w_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=num_devices,
        num_cores=num_cores,
        num_speculative_tokens=spec_length - 1,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_wo_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=num_devices,
        num_cores=num_cores,
        num_speculative_tokens=spec_length - 1,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )

    # Generate texts from prompts
    model_w_sampler_exec_info = model_w_sampler.generate(
        tokenizer=load_hf_tokenizer(pretrained_model_name_or_path=model),
        prompts=prompts,
        device_id=device_group,
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
        tokenizer=load_hf_tokenizer(pretrained_model_name_or_path=model),
        prompts=prompts,
        device_id=device_group,
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
    ), "Generated ids do not match"


@pytest.mark.on_qaic
@pytest.mark.parametrize(
    "model, prompts, prefill_seq_len, ctx_len, generation_len, full_batch_size, num_devices, device_group, num_cores, spec_length, repetition_penalty, presence_penalty, temperature, top_k, top_p, min_p, random_number",
    configs,
)
def test_random_sampling(
    model: str,
    prompts: List[str],
    prefill_seq_len: int,
    ctx_len: int,
    generation_len: int,
    full_batch_size: int,
    num_devices: int,
    device_group: List[int],
    num_cores: int,
    spec_length: int,
    repetition_penalty: float,
    presence_penalty: float,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    random_number: float,
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
    model_w_sampler = AutoModelForCausalLM.from_pretrained(model, continuous_batching=True, qaic_config=qaic_config)
    model_wo_sampler = AutoModelForCausalLM.from_pretrained(model, continuous_batching=True, qaic_config=None)
    model_w_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=num_devices,
        num_cores=num_cores,
        num_speculative_tokens=spec_length - 1,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )
    model_wo_sampler.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
        num_devices=num_devices,
        num_cores=num_cores,
        num_speculative_tokens=spec_length - 1,
        mxint8_kv_cache=True,
        mxfp6_matmul=True,
    )

    # Generate texts from prompts
    model_w_sampler_exec_info = model_w_sampler.generate(
        tokenizer=load_hf_tokenizer(pretrained_model_name_or_path=model),
        prompts=prompts,
        device_id=device_group,
        generation_len=generation_len,
        include_sampler=True,
        return_pdfs=False,
        sampling_params={
            "repetition_penalties": np.array(repetition_penalty, dtype=np.float32)
            .repeat(full_batch_size)
            .reshape(-1, 1),
            "presence_penalties": np.array(presence_penalty, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            # "frequency_penalties": np.array(frequency_penalty, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "temperatures": np.array(temperature, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "top_ks": np.array(top_k, dtype=np.int32).repeat(full_batch_size).reshape(-1, 1),
            "top_ps": np.array(top_p, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "min_ps": np.array(min_p, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
            "random_numbers": np.array(random_number, dtype=np.float32).repeat(full_batch_size).reshape(-1, 1),
        },
    )
    model_wo_sampler_exec_info = model_wo_sampler.generate(
        tokenizer=load_hf_tokenizer(pretrained_model_name_or_path=model),
        prompts=prompts,
        device_id=device_group,
        generation_len=generation_len,
        include_sampler=False,
        return_pdfs=False,
        sampling_params=None,
    )

    # Compare generated texts
    golden_texts = {
        "w_sampler": [""] * full_batch_size,
        "wo_sampler": [""] * full_batch_size,
    }
    assert (
        model_w_sampler_exec_info.generated_texts == golden_texts["w_sampler"]
    ), "Sampler generated texts do not match"
    assert (
        model_wo_sampler_exec_info.generated_texts == golden_texts["wo_sampler"]
    ), "Without sampler generated texts do not match"
