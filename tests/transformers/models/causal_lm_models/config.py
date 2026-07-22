# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import torch

from QEfficient.utils.constants import Constants

causal_lm_models_dict = {
    # --- CodeGen ---
    "Salesforce/codegen-350M-mono": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    # # --- Falcon ---
    # "tiiuae/falcon-7b": "hf-internal-testing/tiny-random-FalconForCausalLM",
    # "tiiuae/falcon-40b": "hf-internal-testing/tiny-random-FalconForCausalLM",
    # # --- Gemma ---
    # "unsloth/gemma-2b": "optimum-intel-internal-testing/tiny-random-GemmaForCausalLM",
    # "google/codegemma-2b": "optimum-intel-internal-testing/tiny-random-GemmaForCausalLM",
    # "google/codegemma-7b": "optimum-intel-internal-testing/tiny-random-GemmaForCausalLM",
    # "google/gemma-2b": "optimum-intel-internal-testing/tiny-random-GemmaForCausalLM",
    # "google/gemma-7b": "optimum-intel-internal-testing/tiny-random-GemmaForCausalLM",
    # # --- Gemma2 ---
    # "unsloth/gemma-2-2b": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    # "google/gemma-2-2b": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    # "google/gemma-2-9b": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    # "google/gemma-2-27b": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    # # --- GLM-4.5 MoE ---
    # "zai-org/GLM-4.5": "trl-internal-testing/tiny-Glm4MoeForCausalLM",
    # # --- GPT-2 ---
    # "openai-community/gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    # # --- GPT-J ---
    # "EleutherAI/gpt-j-6b": "hf-internal-testing/tiny-random-GPTJForCausalLM",
    # # --- GPT-OSS ---
    # "openai/gpt-oss-20b": "trl-internal-testing/tiny-GptOssForCausalLM",
    # # --- Granite MoE ---
    # "ibm-granite/granite-3.1-1b-a400m-base": "hf-internal-testing/tiny-random-GraniteMoeForCausalLM",
    # # --- GPTBigCode ---
    # "bigcode/starcoder": "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    # "ibm-granite/granite-20b-code-base": "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    # "ibm-granite/granite-20b-code-base-8k": "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    # "ibm-granite/granite-20b-code-instruct-8k": "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    # # --- Granite dense ---
    # "ibm-granite/granite-3.1-2b-instruct": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    # "ibm-granite/granite-3.1-8b-instruct": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    # "ibm-granite/granite-guardian-3.1-2b": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    # "ibm-granite/granite-guardian-3.1-8b": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    # # --- Grok-1 ---
    # "hpcai-tech/grok-1": "hpcai-tech/grok-1",  # no tiny found
    # # --- Jais ---
    # "inceptionai/jais-adapted-7b": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "inceptionai/jais-adapted-13b-chat": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "inceptionai/jais-adapted-70b": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # # --- Llama ---
    # "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "codellama/CodeLlama-7b-hf": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "codellama/CodeLlama-13b-hf": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "codellama/CodeLlama-34b-hf": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # # "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "hf-internal-testing/tiny-random-LlamaForCausalLM", # can't open on HF due to QCOM compliance
    # "lmsys/vicuna-13b-delta-v0": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "lmsys/vicuna-13b-v1.3": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "lmsys/vicuna-13b-v1.5": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "meta-llama/Llama-2-7b-chat-hf": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "meta-llama/Llama-2-13b-chat-hf": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "meta-llama/Llama-2-70b-chat-hf": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "meta-llama/Meta-Llama-3-8B": "trl-internal-testing/tiny-LlamaForCausalLM-3",
    # "meta-llama/Meta-Llama-3-70B": "trl-internal-testing/tiny-LlamaForCausalLM-3",
    # "meta-llama/Llama-3.1-8B": "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
    # "meta-llama/Llama-3.1-70B": "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
    # "meta-llama/Llama-3.2-1B": "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
    # "meta-llama/Llama-3.2-3B": "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
    # "meta-llama/Llama-3.3-70B-Instruct": "llamafactory/tiny-random-Llama-3",
    # # --- Quantized Llama models ---
    # "TheBloke/Llama-2-7B-GPTQ": "TheBloke/Llama-2-7B-GPTQ",  # no tiny found
    # "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ": "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",  # no tiny
    # "neuralmagic/Llama-3.2-3B-Instruct-FP8": "neuralmagic/Llama-3.2-3B-Instruct-FP8",  # no tiny
    # # --- Llama SwiftKV ---
    # "Snowflake/Llama-3.1-SwiftKV-8B-Instruct": "Snowflake/Llama-3.1-SwiftKV-8B-Instruct",  # no tiny
    # # --- Mistral ---
    # "Felladrin/Minueza-32M-Base": "hf-internal-testing/tiny-random-MistralForCausalLM",
    # "mistralai/Mistral-7B-Instruct-v0.1": "hf-internal-testing/tiny-random-MistralForCausalLM",
    # "mistralai/Codestral-22B-v0.1": "hf-internal-testing/tiny-random-MistralForCausalLM",
    # # --- Mixtral MoE ---
    # "mistralai/Mixtral-8x7B-v0.1": "hf-internal-testing/tiny-random-MixtralForCausalLM",
    # # --- MPT ---
    # "wtang06/mpt-125m-c4": "hf-internal-testing/tiny-random-MptForCausalLM",
    # # --- OLMo2 ---
    # "allenai/OLMo-2-0425-1B": "hf-internal-testing/tiny-random-Olmo2ForCausalLM",
    # # --- Phi3 ---
    # "microsoft/Phi-3-mini-4k-instruct": "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM",
    # # --- Qwen2 ---
    # "Qwen/Qwen2-0.5B": "peft-internal-testing/tiny-dummy-qwen2",
    # "Qwen/Qwen2-1.5B-Instruct": "peft-internal-testing/tiny-dummy-qwen2",
    # # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", # can't open on HF due to QCOM compliance
    # "neuralmagic/Qwen2-0.5B-Instruct-FP8": "neuralmagic/Qwen2-0.5B-Instruct-FP8",  # no tiny
    # # --- Qwen3 MoE ---
    # "Qwen/Qwen3-30B-A3B-Instruct-2507": "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM",
    # # --- Starcoder ---
    # "bigcode/starcoder2-3b": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
    # "bigcode/starcoder2-15b": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
}


transform_params = {"torch_dtype": torch.float16}
export_params = {"use_onnx_subfunctions": True}
compile_params = {
    "num_devices": 1,
    "prefill_seq_len": 32,
    "ctx_len": 128,
    "num_speculative_tokens": None,
    "use_onnx_subfunctions": True,
    "mdp_num_partitions": None,
    "mdp_strategy": None,
    "prefill_only": None,
    "enable_qnn": False,
    "qnn_config": None,
    "retain_full_kv": None,
}
generate_params = {"prompt": Constants.INPUT_STR, "generation_len": 25}


QEFF_TEST_PROFILE = os.environ.get("QEFF_TEST_PROFILE", "").strip().lower()

if QEFF_TEST_PROFILE == "tiny_model":
    test_models_causal = set(causal_lm_models_dict.values())
elif QEFF_TEST_PROFILE == "full_model":
    test_models_causal = set(causal_lm_models_dict.keys())
    compile_params.update({"num_devices": 4})  # setting number of devices 4 for full models.
    compile_params["mxfp6_matmul"] = True
    compile_params["mxint8_kv_cache"] = True
