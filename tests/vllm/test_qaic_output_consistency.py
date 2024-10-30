# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Technologies, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import gc
import random

import pytest
from vllm import LLM, SamplingParams

test_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

seq_len = 128
ctx_len = 256
decode_bsz = 4
dtype = "mxfp6"
kv_dtype = "mxint8"
device_group = [0]


@pytest.mark.parametrize("model_name", test_models)
def test_output_consistency(model_name):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=None)

    qllm = LLM(
        model=model_name,
        device_group=device_group,
        max_num_seqs=decode_bsz,
        max_model_len=ctx_len,
        max_seq_len_to_capture=seq_len,
        quantization=dtype,
        kv_cache_dtype=kv_dtype,
        device="qaic",
    )
    prompt = ["My name is"]

    output = qllm.generate(prompt * 5, sampling_params)

    check_output = []
    for i, op in enumerate(output):
        check_output.append(op.outputs[0].text)

    assert len(set(check_output)) == 1, "Outputs from different slots for same prompt does not match!!"

    del qllm
    gc.collect()


@pytest.mark.parametrize("model_name", test_models)
def test_generate(model_name):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=None)

    qllm = LLM(
        model=model_name,
        device_group=device_group,
        max_num_seqs=decode_bsz,
        max_model_len=ctx_len,
        max_seq_len_to_capture=seq_len,
        quantization=dtype,
        kv_cache_dtype=kv_dtype,
        device="qaic",
    )
    outputDict = dict()
    prompt = [
        "My name is",
        "How to eat mangosteen?",
        "How many people died in World War II",
        "Hello ",
        "Who is the president of United States",
        "Who is the president of India",
        "When it snowfalls in San Diego",
        "In which country yamana river flows",
        "How many people died in World War II",
        "Thy youth is proud livery, so gazed on now",
        "Will be a tattered weed, of small worth held:" "Then being asked where all thy beauty lies",
        "Where all the treasure of thy lusty days",
        "To say, within thine own deep-sunken eyes",
        "Where is Statue of Liberty located?",
    ]

    for p in prompt:
        outputDict[p] = []

    for _ in range(5):
        random.shuffle(prompt)
        output = qllm.generate(prompt, sampling_params)
        for i, op in enumerate(output):
            generated_text = op.outputs[0].text
            outputDict[prompt[i]].append(str(prompt[i] + generated_text))

    for key in outputDict.keys():
        assert len(set(outputDict[key])) == 1, "Outputs from different slots for same prompt does not match!!"

    del qllm
    gc.collect()


@pytest.mark.parametrize("model_name", test_models)
def test_generated_tokens(model_name):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=None)

    qllm = LLM(
        model=model_name,
        device_group=device_group,
        max_num_seqs=decode_bsz,
        max_model_len=ctx_len,
        max_seq_len_to_capture=seq_len,
        quantization=dtype,
        kv_cache_dtype=kv_dtype,
        device="qaic",
    )
    prompt = [
        "My name is",
        "How to eat mangosteen?",
        "How many people died in World War II",
        "Hello ",
        "Who is the president of United States",
        "Who is the president of India",
        "When it snowfalls in San Diego",
        "In which country yamana river flows",
        "How many people died in World War II",
        "Thy youth is proud livery, so gazed on now",
        "Will be a tattered weed, of small worth held:" "Then being asked where all thy beauty lies",
        "Where all the treasure of thy lusty days",
        "To say, within thine own deep-sunken eyes",
        "Where is Statue of Liberty located?",
    ]

    output = qllm.generate(prompt, sampling_params)

    for i, op in enumerate(output):
        continue

    assert len(prompt) == i + 1, "Number of Generated Tokens do not match the number of valid inputs!!"

    del qllm
    gc.collect()
