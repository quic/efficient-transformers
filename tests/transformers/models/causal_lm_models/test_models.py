# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os

import pytest
import torch

from QEfficient.utils.constants import Constants
from QEfficient.utils.test_utils import load_qeff_causal_lm_model

from .check_causal_models import check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100, prefix_caching_inference

causal_lm_models_dict = {
    # --- CodeGen ---
    "Salesforce/codegen-350M-mono": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    # --- Falcon ---
    "tiiuae/falcon-7b": "hf-internal-testing/tiny-random-FalconForCausalLM",
    "tiiuae/falcon-40b": "hf-internal-testing/tiny-random-FalconForCausalLM",
    # --- Gemma ---
    "unsloth/gemma-2b": "optimum-intel-internal-testing/tiny-random-GemmaForCausalLM",
    "google/codegemma-2b": "optimum-intel-internal-testing/tiny-random-GemmaForCausalLM",
    "google/codegemma-7b": "optimum-intel-internal-testing/tiny-random-GemmaForCausalLM",
    "google/gemma-2b": "optimum-intel-internal-testing/tiny-random-GemmaForCausalLM",
    "google/gemma-7b": "optimum-intel-internal-testing/tiny-random-GemmaForCausalLM",
    # --- Gemma2 ---
    "unsloth/gemma-2-2b": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    "google/gemma-2-2b": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    "google/gemma-2-9b": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    "google/gemma-2-27b": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    # --- GLM-4.5 MoE ---
    "zai-org/GLM-4.5": "trl-internal-testing/tiny-Glm4MoeForCausalLM",
    # --- GPT-2 ---
    "openai-community/gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    # --- GPT-J ---
    "EleutherAI/gpt-j-6b": "hf-internal-testing/tiny-random-GPTJForCausalLM",
    # --- GPT-OSS ---
    "openai/gpt-oss-20b": "trl-internal-testing/tiny-GptOssForCausalLM",
    # --- Granite MoE ---
    "ibm-granite/granite-3.1-1b-a400m-base": "hf-internal-testing/tiny-random-GraniteMoeForCausalLM",
    # --- GPTBigCode ---
    "bigcode/starcoder": "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    "ibm-granite/granite-20b-code-base": "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    "ibm-granite/granite-20b-code-base-8k": "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    "ibm-granite/granite-20b-code-instruct-8k": "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    # --- Granite dense ---
    "ibm-granite/granite-3.1-2b-instruct": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "ibm-granite/granite-3.1-8b-instruct": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "ibm-granite/granite-guardian-3.1-2b": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "ibm-granite/granite-guardian-3.1-8b": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    # --- Grok-1 ---
    "hpcai-tech/grok-1": "hpcai-tech/grok-1",  # no tiny found
    # --- Jais ---
    "inceptionai/jais-adapted-7b": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "inceptionai/jais-adapted-13b-chat": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "inceptionai/jais-adapted-70b": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # --- Llama ---
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "codellama/CodeLlama-7b-hf": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "codellama/CodeLlama-13b-hf": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "codellama/CodeLlama-34b-hf": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "hf-internal-testing/tiny-random-LlamaForCausalLM", # can't open on HF due to QCOM compliance
    "lmsys/vicuna-13b-delta-v0": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "lmsys/vicuna-13b-v1.3": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "lmsys/vicuna-13b-v1.5": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "meta-llama/Llama-2-7b-chat-hf": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "meta-llama/Llama-2-13b-chat-hf": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "meta-llama/Llama-2-70b-chat-hf": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "meta-llama/Meta-Llama-3-8B": "trl-internal-testing/tiny-LlamaForCausalLM-3",
    "meta-llama/Meta-Llama-3-70B": "trl-internal-testing/tiny-LlamaForCausalLM-3",
    "meta-llama/Llama-3.1-8B": "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
    "meta-llama/Llama-3.1-70B": "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
    "meta-llama/Llama-3.2-1B": "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
    "meta-llama/Llama-3.2-3B": "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
    "meta-llama/Llama-3.3-70B-Instruct": "llamafactory/tiny-random-Llama-3",
    # --- Quantized Llama models ---
    "TheBloke/Llama-2-7B-GPTQ": "TheBloke/Llama-2-7B-GPTQ",  # no tiny found
    "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ": "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",  # no tiny
    "neuralmagic/Llama-3.2-3B-Instruct-FP8": "neuralmagic/Llama-3.2-3B-Instruct-FP8",  # no tiny
    # --- Llama SwiftKV ---
    "Snowflake/Llama-3.1-SwiftKV-8B-Instruct": "Snowflake/Llama-3.1-SwiftKV-8B-Instruct",  # no tiny
    # --- Mistral ---
    "Felladrin/Minueza-32M-Base": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "mistralai/Mistral-7B-Instruct-v0.1": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "mistralai/Codestral-22B-v0.1": "hf-internal-testing/tiny-random-MistralForCausalLM",
    # --- Mixtral MoE ---
    "mistralai/Mixtral-8x7B-v0.1": "hf-internal-testing/tiny-random-MixtralForCausalLM",
    # --- MPT ---
    "wtang06/mpt-125m-c4": "hf-internal-testing/tiny-random-MptForCausalLM",
    # --- OLMo2 ---
    "allenai/OLMo-2-0425-1B": "hf-internal-testing/tiny-random-Olmo2ForCausalLM",
    # --- Phi3 ---
    "microsoft/Phi-3-mini-4k-instruct": "optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM",
    # --- Qwen2 ---
    "Qwen/Qwen2-0.5B": "peft-internal-testing/tiny-dummy-qwen2",
    "Qwen/Qwen2-1.5B-Instruct": "peft-internal-testing/tiny-dummy-qwen2",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", # can't open on HF due to QCOM compliance
    "neuralmagic/Qwen2-0.5B-Instruct-FP8": "neuralmagic/Qwen2-0.5B-Instruct-FP8",  # no tiny
    # --- Qwen3 MoE ---
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM",
    # --- Starcoder ---
    "bigcode/starcoder2-3b": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
    "bigcode/starcoder2-15b": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
}

QEFF_TEST_PROFILE = os.environ.get("QEFF_TEST_PROFILE", "").strip().lower()

if QEFF_TEST_PROFILE == "tiny_model":
    test_models_causal = set(causal_lm_models_dict.values())
else:
    test_models_causal = set(causal_lm_models_dict.keys())


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_export_compile_default(model_name):
    """
    Fp16 end to end run subfunction True by default.  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    transform_params = {"torch_dtype": torch.float16, "qaic_config": None}
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
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        continuous_batching=False,
        export_compile_only=True,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_export_compile_generate_default(model_name):
    """
    Fp16 end to end run subfunction True by default.  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    transform_params = {"torch_dtype": torch.float16, "qaic_config": None}
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
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        continuous_batching=False,
        export_compile_only=False,
    )


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_export_compile_default_cb(model_name):
    """
    Fp16 + Subfunction + CB  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    transform_params = {"torch_dtype": torch.float16, "qaic_config": None}
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
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_export_compile_generate_default_cb(model_name):
    """
    Fp16 + Subfunction + CB  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    transform_params = {"torch_dtype": torch.float16, "qaic_config": None}
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
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
    )


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_export_compile_speculative_cb(model_name):
    """
    Fp16 + Subfunction + speculation + CB  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    transform_params = {"torch_dtype": torch.float16, "qaic_config": None}
    export_params = {"use_onnx_subfunctions": True}
    compile_params = {
        "num_devices": 1,
        "prefill_seq_len": 32,
        "ctx_len": 128,
        "num_speculative_tokens": Constants.NUM_SPECULATIVE_TOKENS,
        "use_onnx_subfunctions": True,
        "mdp_num_partitions": None,
        "mdp_strategy": None,
        "prefill_only": None,
        "enable_qnn": False,
        "qnn_config": None,
        "retain_full_kv": None,
    }
    generate_params = {"prompt": Constants.INPUT_STR, "generation_len": 25}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_export_compile_generate_speculative_cb(model_name):
    """
    Fp16 + Subfunction + speculation + CB  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    transform_params = {"torch_dtype": torch.float16, "qaic_config": None}
    export_params = {"use_onnx_subfunctions": True}
    compile_params = {
        "num_devices": 1,
        "prefill_seq_len": 32,
        "ctx_len": 128,
        "num_speculative_tokens": Constants.NUM_SPECULATIVE_TOKENS,
        "use_onnx_subfunctions": True,
        "mdp_num_partitions": None,
        "mdp_strategy": None,
        "prefill_only": None,
        "enable_qnn": False,
        "qnn_config": None,
        "retain_full_kv": None,
    }
    generate_params = {"prompt": Constants.INPUT_STR, "generation_len": 25}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
    )


@pytest.mark.on_qaic
@pytest.mark.llm
@pytest.mark.parametrize("model_name", test_models_causal)
def test_prefix_caching(model_name):
    """
    Fp16 + Subfunction + CB with prefix caching  (end to end run+ output verification)
    The test should first generate output with some prefix+suffix1 or batch_id and then confirm that we are still able to execute of prefix+suffix2 on same batch id and getting correct output.
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    qeff_model = load_qeff_causal_lm_model(
        model_name=model_name,
        continuous_batching=True,
        torch_dtype=torch.float16,
        num_hidden_layers=2,
    )
    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=256,
        full_batch_size=2,
        kv_cache_batch_size=4,
        num_cores=16,
        use_onnx_subfunctions=True,
    )
    prefix_caching_inference(model_name=model_name, qpc_path=qeff_model.qpc_path)
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_export_compile_ccl_cb(model_name):
    """
    Fp16 + Subfunction + CB + CCL  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    qaic_config = {
        "ccl_enabled": True,
    }
    transform_params = {"torch_dtype": torch.float16, "qaic_config": qaic_config}
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
        "comp_ctx_lengths_prefill": [256, 500],
        "comp_ctx_lengths_decode": [512, 1024],
    }
    generate_params = {"prompt": Constants.INPUT_STR, "generation_len": 25}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_export_compile_generate_ccl_cb(model_name):
    """
    Fp16 + Subfunction + CB + CCL  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    qaic_config = {
        "ccl_enabled": True,
    }
    transform_params = {"torch_dtype": torch.float16, "qaic_config": qaic_config}
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
        "comp_ctx_lengths_prefill": [256, 500],
        "comp_ctx_lengths_decode": [512, 1024],
    }
    generate_params = {"prompt": Constants.INPUT_STR, "generation_len": 25}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
    )


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp32_export_fp16_compile_ccl_cb(model_name):
    """
    FP32 export + FP16 compilation + Subfunction + CB + CCL  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    qaic_config = {
        "ccl_enabled": True,
    }
    transform_params = {"torch_dtype": torch.float32, "qaic_config": qaic_config}
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
        "comp_ctx_lengths_prefill": [256, 500],
        "comp_ctx_lengths_decode": [512, 1024],
    }
    generate_params = {"prompt": Constants.INPUT_STR, "generation_len": 25}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp32_export_fp16_compile_generate_ccl_cb(model_name):
    """
    FP32 export + FP16 compilation + Subfunction + CB + CCL  (end to end run+ output verification)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    qaic_config = {
        "ccl_enabled": True,
    }
    transform_params = {"torch_dtype": torch.float32, "qaic_config": qaic_config}
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
        "comp_ctx_lengths_prefill": [256, 500],
        "comp_ctx_lengths_decode": [512, 1024],
    }
    generate_params = {"prompt": Constants.INPUT_STR, "generation_len": 25}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
    )


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_bf16_export_bf16_compile_ccl_cb(model_name):
    """
    BF16 export and BF16 compilation + Subfunction + CB + CCL (only till compilation)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    qaic_config = {
        "ccl_enabled": True,
    }
    transform_params = {"torch_dtype": torch.bfloat16, "qaic_config": qaic_config}
    export_params = {"use_onnx_subfunctions": True}
    compile_params = {
        "num_devices": 1,
        "num_cores": 4,
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
        "aic-hw-version": "ai200",
        "comp_ctx_lengths_prefill": [256, 500],
        "comp_ctx_lengths_decode": [512, 1024],
    }
    generate_params = {"prompt": Constants.INPUT_STR, "generation_len": 25}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
    )


@pytest.mark.skip(
    reason="BF16 export and BF16 compilation + Subfunction + CB + CCL (only till compilation) - not supported yet"
)
@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_bf16_export_bf16_compile_generate_ccl_cb(model_name):
    """
    BF16 export and BF16 compilation + Subfunction + CB + CCL (only till compilation)
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    qaic_config = {
        "ccl_enabled": True,
    }
    transform_params = {"torch_dtype": torch.bfloat16, "qaic_config": qaic_config}
    export_params = {"use_onnx_subfunctions": True}
    compile_params = {
        "num_devices": 1,
        "num_cores": 4,
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
        "aic-hw-version": "ai200",
        "comp_ctx_lengths_prefill": [256, 500],
        "comp_ctx_lengths_decode": [512, 1024],
    }
    generate_params = {"prompt": Constants.INPUT_STR, "generation_len": 25}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
    )


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp16_export_compile_blocking_CB(model_name):
    """
    Fp16 + Subfunction + CB + Blocking enabled
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2
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

    # head blocking only
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE)
    transform_params = {"torch_dtype": torch.float16, "qaic_config": qaic_config}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
    )

    # kv blocking only
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)
    transform_params = {"torch_dtype": torch.float16, "qaic_config": qaic_config}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
    )

    # q block only
    qaic_config = dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS)
    transform_params = {"torch_dtype": torch.float16, "qaic_config": qaic_config}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
    )

    # qkv blocking
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS)
    transform_params = {"torch_dtype": torch.float16, "qaic_config": qaic_config}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
    )

    # head qkv blocking
    qaic_config = dict(
        enable_blocking=True,
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    transform_params = {"torch_dtype": torch.float16, "qaic_config": qaic_config}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=True,
    )


@pytest.mark.llm
@pytest.mark.qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_fp16_export_compile_generate_blocking_CB(model_name):
    """
    Fp16 + Subfunction + CB + Blocking enabled
    """
    if causal_lm_models_dict.get(model_name, None) == model_name and QEFF_TEST_PROFILE == "tiny_model":
        pytest.skip("Skipping it is not a tiny model and will run in nightly tests.")

    HEAD_BLOCK_SIZE = 8
    NUM_KV_BLOCKS = 2
    NUM_Q_BLOCKS = 2
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

    # head blocking only
    qaic_config = dict(enable_blocking=True, head_block_size=HEAD_BLOCK_SIZE)
    transform_params = {"torch_dtype": torch.float16, "qaic_config": qaic_config}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
    )

    # kv blocking only
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS)
    transform_params = {"torch_dtype": torch.float16, "qaic_config": qaic_config}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
    )

    # q block only
    qaic_config = dict(enable_blocking=True, num_q_blocks=NUM_Q_BLOCKS)
    transform_params = {"torch_dtype": torch.float16, "qaic_config": qaic_config}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
    )

    # qkv blocking
    qaic_config = dict(enable_blocking=True, num_kv_blocks=NUM_KV_BLOCKS, num_q_blocks=NUM_Q_BLOCKS)
    transform_params = {"torch_dtype": torch.float16, "qaic_config": qaic_config}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
    )

    # head qkv blocking
    qaic_config = dict(
        enable_blocking=True,
        head_block_size=HEAD_BLOCK_SIZE,
        num_kv_blocks=NUM_KV_BLOCKS,
        num_q_blocks=NUM_Q_BLOCKS,
    )
    transform_params = {"torch_dtype": torch.float16, "qaic_config": qaic_config}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        transform_params=transform_params,
        export_params=export_params,
        compile_params=compile_params,
        generate_params=generate_params,
        export_compile_only=False,
    )
