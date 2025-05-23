# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
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


@pytest.mark.vllm
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
        max_num_seqs=DECOE_BSZ,
        max_model_len=CTX_LEN,
        max_seq_len_to_capture=SEQ_LEN,
        quantization=DTYPE,
        kv_cache_dtype=KV_DTYPE,
        device="qaic",
    )

    # Single prompt test
    single_prompt = ["My name is"]

    single_prompt_output = qllm.generate(single_prompt * 5, sampling_params)

    check_output = []
    for i, op in enumerate(single_prompt_output):
        check_output.append(op.outputs[0].text)

        # Assertion to check the consistency of single prompt.
    assert len(set(check_output)) == 1, "Outputs from different slots for same prompt does not match!!"

    # Multiple prompt test
    outputDict = dict()
    multiple_prompt = [
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
        "Will be a tattered weed, of small worth held:Then being asked where all thy beauty lies",
        "Where all the treasure of thy lusty days",
        "To say, within thine own deep-sunken eyes",
        "Where is Statue of Liberty located?",
    ]

    for p in multiple_prompt:
        outputDict[p] = []

    for _ in range(5):
        random.shuffle(multiple_prompt)
        multiple_prompt_output = qllm.generate(multiple_prompt, sampling_params)
        for i, op in enumerate(multiple_prompt_output):
            generated_text = op.outputs[0].text
            outputDict[multiple_prompt[i]].append(str(multiple_prompt[i] + generated_text))

    # Assertion to check multiple prompts.
    for key in outputDict.keys():
        assert len(set(outputDict[key])) == 1, "Outputs from different slots for same prompt does not match!!"

    # Assertion to check if any prompts are missed.
    assert len(multiple_prompt) == len(multiple_prompt_output), (
        "Number of Generated Tokens do not match the number of valid inputs!!"
    )
