# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import random

import pytest
from vllm import LLM, SamplingParams

# Model to test
test_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

# Constants for configuration
SEQ_LEN = 128
CTX_LEN = 256
DECOE_BSZ = 4
DTYPE = "mxfp6"
KV_DTYPE = "mxint8"
DEVICE_GROUP = [0]


@pytest.mark.parametrize("model_name", test_models)
def test_output_consistency(model_name):
    """This pytest function is used to check the consistency of vLLM.
       1) Single prompt test to check if the output generated in 5 different
          runs yields the same results
       2) Multiple prompt check to test if multiple prompts yield same results
          if run in different slots.

    Parameters
    ----------
    model_name : string
        Huggingface model card name.
    """
    sampling_params = SamplingParams(temperature=0.0, max_tokens=None)

    # Creating LLM Object
    qllm = LLM(
        model=model_name,
        device_group=DEVICE_GROUP,
        max_num_seqs=DECOE_BSZ,
        max_model_len=CTX_LEN,
        max_seq_len_to_capture=SEQ_LEN,
        quantization=DTYPE,
        kv_cache_dtype=KV_DTYPE,
        device="qaic",
    )

    # Single prompt test
    prompt1 = ["My name is"]

    output1 = qllm.generate(prompt1 * 5, sampling_params)

    check_output1 = []
    for i, op in enumerate(output1):
        check_output1.append(op.outputs[0].text)

    # Multiple prompt test
    outputDict = dict()
    prompt2 = [
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

    for p in prompt2:
        outputDict[p] = []

    for _ in range(5):
        random.shuffle(prompt2)
        output2 = qllm.generate(prompt2, sampling_params)
        for i, op in enumerate(output2):
            generated_text = op.outputs[0].text
            outputDict[prompt2[i]].append(str(prompt2[i] + generated_text))

    # Assertion to check the consistency of single prompt.
    assert len(set(check_output1)) == 1, "Outputs from different slots for same prompt does not match!!"

    # Assertion to check multiple prompts.
    for key in outputDict.keys():
        assert len(set(outputDict[key])) == 1, "Outputs from different slots for same prompt does not match!!"

    # Assertion to check if any prompts are missed.
    assert len(prompt2) == len(output2), "Number of Generated Tokens do not match the number of valid inputs!!"