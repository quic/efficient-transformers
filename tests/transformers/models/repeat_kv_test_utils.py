# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

REPEAT_KV_TEST_MODELS = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "ibm-granite/granite-3.1-1b-a400m-base",
    "Qwen/Qwen2-0.5B",
    "bigcode/starcoder2-3b",
    "meta-llama/Llama-3.2-1B",
    "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",
    "TheBloke/Llama-2-7B-GPTQ",
    "neuralmagic/Llama-3.2-3B-Instruct-FP8",
    "ibm-granite/granite-3.1-2b-instruct",
    "llava-hf/llava-1.5-7b-hf",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "allenai/Molmo-7B-D-0924",
    "OpenGVLab/InternVL2_5-1B",
    "Qwen/Qwen3.5-0.8B",
}


def get_text_config(config):
    if hasattr(config, "text_config"):
        return config.text_config
    if hasattr(config, "llm_config"):
        return config.llm_config
    return config
