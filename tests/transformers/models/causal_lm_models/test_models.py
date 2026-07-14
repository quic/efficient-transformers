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
    # "Felladrin/Minueza-32M-Base": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "wtang06/mpt-125m-c4": "hf-internal-testing/tiny-random-MptForCausalLM",
    "hakurei/gpt-j-random-tinier": "hf-internal-testing/tiny-random-GPTJForCausalLM",
    "mistralai/Mistral-7B-Instruct-v0.1": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "mistralai/Mixtral-8x7B-v0.1": "hf-internal-testing/tiny-random-MixtralForCausalLM",
    # "meta-llama/Llama-3.2-1B": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "unsloth/gemma-2b": "trl-internal-testing/tiny-GemmaForCausalLM",
    "unsloth/gemma-2-2b": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    "ibm-granite/granite-20b-code-base": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    # "ibm-granite/granite-3.1-2b-instruct": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    # "ibm-granite/granite-guardian-3.1-2b": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ": "optimum-intel-internal-testing/tiny-mixtral-AWQ-4bit",
    # "TheBloke/Llama-2-7B-GPTQ": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    # "neuralmagic/Llama-3.2-3B-Instruct-FP8": "nm-testing/Meta-Llama-3-8B-Instruct-FP8",
    # "neuralmagic/Qwen2-0.5B-Instruct-FP8": "nm-testing/Qwen2-0.5B-Instruct-FP8",
    # "Snowflake/Llama-3.1-SwiftKV-8B-Instruct": "snowflake-internal-testing/tiny-Llama-3.1-SwiftKV-8B-Instruct",
}

if os.environ.get("QEFF_TEST_PROFILE", "").strip().lower() == "tiny_model":
    test_models_causal = list(causal_lm_models_dict.values())
else:
    test_models_causal = list(causal_lm_models_dict.keys())


@pytest.mark.llm
@pytest.mark.non_qaic
@pytest.mark.parametrize("model_name", test_models_causal)
def test_export_compile_default(model_name):
    """
    Fp16 end to end run subfunction True by default.  (end to end run+ output verification)
    """
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
    qeff_model = load_qeff_causal_lm_model(
        model_name=model_name,
        continuous_batching=True,
        torch_dtype=torch.float16,
    )
    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=256,
        full_batch_size=2,
        kv_cache_batch_size=4,
        num_cores=16,
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
