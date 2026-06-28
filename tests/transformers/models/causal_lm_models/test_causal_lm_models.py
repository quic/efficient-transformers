# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest

from QEfficient.utils.test_utils import ModelConfig

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
)

causal_lm_models_dict = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "allenai/OLMo-2-0425-1B": "hf-internal-testing/tiny-random-Olmo2ForCausalLM",
    "Salesforce/codegen-350M-mono": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "ibm-granite/granite-3.1-1b-a400m-base": "hf-internal-testing/tiny-random-GraniteMoeForCausalLM",
    "microsoft/Phi-3-mini-4k-instruct": "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM",
    "tiiuae/falcon-7b": "yujiepan/falcon-tiny-random",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM",
    "Qwen/Qwen2-0.5B": "peft-internal-testing/tiny-dummy-qwen2",
    "bigcode/starcoder2-3b": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
    "Felladrin/Minueza-32M-Base": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "wtang06/mpt-125m-c4": "hf-internal-testing/tiny-random-MptForCausalLM",
    "hakurei/gpt-j-random-tinier": "hf-internal-testing/tiny-random-GPTJForCausalLM",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "yujiepan/mixtral-tiny-random",
    "meta-llama/Llama-3.2-1B": "tiny-random/llama-3",
    "unsloth/gemma-2b": "trl-internal-testing/tiny-GemmaForCausalLM",
    "unsloth/gemma-2-2b": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    "ibm-granite/granite-20b-code-base": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "ibm-granite/granite-3.1-2b-instruct": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "ibm-granite/granite-guardian-3.1-2b": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ": "optimum-intel-internal-testing/tiny-mixtral-AWQ-4bit",
    "TheBloke/Llama-2-7B-GPTQ": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "neuralmagic/Llama-3.2-3B-Instruct-FP8": "nm-testing/Meta-Llama-3-8B-Instruct-FP8",
    # "neuralmagic/Qwen2-0.5B-Instruct-FP8": "nm-testing/Qwen2-0.5B-Instruct-FP8",
    "Snowflake/Llama-3.1-SwiftKV-8B-Instruct": "snowflake-internal-testing/tiny-Llama-3.1-SwiftKV-8B-Instruct",
}

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    test_models_causal = list(causal_lm_models_dict.values())
else:
    test_models_causal = list(causal_lm_models_dict.keys())


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_export_compile(model_name):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, export_compile_only=True)


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_export_compile_cb(model_name):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        export_compile_only=True,
    )


@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", test_models_causal)
def test_generate(model_name):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name)


@pytest.mark.qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", test_models_causal)
def test_generate_cb(model_name):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
    )
