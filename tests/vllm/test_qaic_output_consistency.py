# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Technologies, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest
import gc
import random

from vllm import LLM, SamplingParams

from argparse import ArgumentParser
import subprocess, time

import math

@pytest.fixture(scope="session")
def model_name(pytestconfig):
    return pytestconfig.getoption("model_name")

@pytest.fixture(scope="session")
def seq_len(pytestconfig):
    return pytestconfig.getoption("seq_len")

@pytest.fixture(scope="session")
def ctx_len(pytestconfig):
    return pytestconfig.getoption("ctx_len")

@pytest.fixture(scope="session")
def decode_bsz(pytestconfig):
    return pytestconfig.getoption("decode_bsz")

@pytest.fixture(scope="session")
def dtype(pytestconfig):
    return pytestconfig.getoption("dtype")

@pytest.fixture(scope="session")
def kv_dtype(pytestconfig):
    return pytestconfig.getoption("kv_dtype")

@pytest.fixture(scope="session")
def dataset(pytestconfig):
    return pytestconfig.getoption("dataset")

model_name = None
seq_len = None
ctx_len = None
decode_bsz = None
dtype = None
kv_dtype = None
dataset = None
device_group = None
sampling_params = None
qllm = None

@pytest.fixture(autouse=True, scope="session")
def init(pytestconfig):

    global model_name
    global seq_len
    global ctx_len
    global decode_bsz
    global dtype
    global kv_dtype
    global dataset
    global device_group
    global sampling_params
    global qllm

    model_name = pytestconfig.getoption('model_name')
    seq_len = pytestconfig.getoption('seq_len')
    ctx_len = pytestconfig.getoption('ctx_len')
    decode_bsz = pytestconfig.getoption('decode_bsz')
    dtype = pytestconfig.getoption('dtype')
    kv_dtype = pytestconfig.getoption('kv_dtype')
    dataset = pytestconfig.getoption('dataset')
    device_group_num = pytestconfig.getoption('device_group')
    if device_group_num == 1:
        device_group = [0]
    elif device_group_num == 4:
        device_group = [0, 1, 2, 3]
    elif device_group_num == 8:
        device_group = [0, 1, 2, 3, 4, 5, 6, 7]

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


def test_output_consistency():
    prompt = ["My name is"]

    output = qllm.generate(prompt * 5, sampling_params)

    check_output = []
    for i, op in enumerate(output):
        check_output.append(op.outputs[0].text)

    #print(check_output)
    assert len(set(check_output)) == 1, "Outputs from different slots for same prompt does not match!!"

def test_generate():
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
    
def test_generated_tokens():
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
