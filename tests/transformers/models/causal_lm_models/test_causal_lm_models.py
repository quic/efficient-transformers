# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os

import pytest
import torch
from transformers import AutoConfig

from QEfficient.utils._utils import create_json
from QEfficient.utils.constants import QnnConstants
from QEfficient.utils.test_utils import ModelConfig

from .check_causal_models import (
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100,
    check_kv_repeat_causal_lm_pytorch_vs_ai100,
    get_custom_n_layers,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../../configs/causal_model_configs.json")
with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)
    causal_lm_models = config_data["causal_lm_models"]
    per_pr_causal_text_models = config_data["per_pr_causal_text_models"]
test_models_causal = [model["model_name"] for model in causal_lm_models]
model_config_dict = {model["model_name"]: model for model in causal_lm_models}
test_models_per_pr_causal = per_pr_causal_text_models

PER_PR_PROMPT_LEN = 1024
PER_PR_CTX_LEN = 2048
PER_PR_GENERATION_LEN = 8
PER_PR_CCL_PREFILL = None
PER_PR_CCL_DECODE = [2048]
PER_PR_BF16_COMPILER_ISSUE = (
    "BF16 compile-only is a known repo/compiler gap: with aic_hw_version=ai200 and num_cores=4, "
    "qaic-compile fails broadly with unsupported ONNX COMPLEX128 casts or missing libjit_convert_f_to_f kernels."
)


def _per_pr_id(model_config):
    return model_config["id"]


def _per_pr_dummy_config(model_config):
    config = AutoConfig.from_pretrained(model_config["model_name"], trust_remote_code=True)
    if config_attr := model_config.get("config_attr"):
        config = getattr(config, config_attr, config)
    elif model_config.get("use_text_config"):
        config = getattr(config, "text_config", config)

    for attr, value in model_config.get("config_overrides", {}).items():
        setattr(config, attr, value)

    for attr in ("max_position_embeddings", "n_positions", "n_ctx", "seq_length"):
        if hasattr(config, attr):
            setattr(config, attr, max(getattr(config, attr), PER_PR_CTX_LEN))

    if num_hidden_layers := model_config.get("num_hidden_layers"):
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(config, attr):
                setattr(config, attr, num_hidden_layers)

    if layer_types := model_config.get("layer_types"):
        config.layer_types = layer_types
        config.num_hidden_layers = len(layer_types)
    return config


def _no_cleanup(*args, **kwargs):
    """No-op cleanup used by the compile-warm phase to preserve the compiled QPC."""


def _run_per_pr_qwen_causal_text_case(
    model_config,
    manual_cleanup,
    *,
    torch_dtype=torch.float16,
    compile_only=False,
    retain_full_kv=False,
    qaic_config=None,
    comp_ctx_lengths_prefill=None,
    comp_ctx_lengths_decode=None,
    kv_cache_batch_size=None,
    num_cores=16,
    compile_options=None,
):
    # Two-phase shared-QEFF_HOME run: the per-test cleanup must be suppressed in BOTH phases,
    # because variants of one model share a content-addressed export dir (the QPCs nest inside
    # it). If left active, a finishing variant's cleanup rmtree's the shared dir and destroys
    # sibling variants' warm QPCs / in-progress compiles. The caller owns the shared QEFF_HOME
    # lifecycle (starts clean, cleans up when the whole two-phase run is done).
    #   - QEFF_PER_PR_COMPILE_WARM_ONLY (Phase A): also force compile-only, no device touched.
    #   - QEFF_PER_PR_SHARED_HOME (Phase B): real generate + assertions against the warm cache.
    if os.environ.get("QEFF_PER_PR_COMPILE_WARM_ONLY"):
        compile_only = True
        manual_cleanup = _no_cleanup
    elif os.environ.get("QEFF_PER_PR_SHARED_HOME"):
        manual_cleanup = _no_cleanup

    if model_config.get("known_export_or_compile_issue"):
        pytest.xfail(model_config["known_export_or_compile_issue"])
    if model_config.get("known_runtime_parity_issue") and not compile_only:
        pytest.xfail(model_config["known_runtime_parity_issue"])

    config = _per_pr_dummy_config(model_config)
    num_cores = model_config.get("num_cores", num_cores)
    compile_options = {**model_config.get("compile_options", {}), **(compile_options or {})}
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_config["model_name"],
        manual_cleanup=manual_cleanup,
        continuous_batching=True,
        prompt_len=PER_PR_PROMPT_LEN,
        ctx_len=PER_PR_CTX_LEN,
        torch_dtype=torch_dtype,
        generation_len=PER_PR_GENERATION_LEN,
        use_onnx_subfunctions=True,
        compile_only=compile_only,
        retain_full_kv=retain_full_kv,
        qaic_config=qaic_config,
        config=config,
        comp_ctx_lengths_prefill=comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=comp_ctx_lengths_decode,
        kv_cache_batch_size=kv_cache_batch_size,
        num_cores=num_cores,
        compile_options=compile_options,
        tokenizer_name=model_config.get("tokenizer_id"),
    )


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_full_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    if model_name in ModelConfig.FULL_MODEL_TESTS_TO_SKIP:
        pytest.skip(f"Skipping full model test for {model_name} due to resource constraints.")
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name, compare_results=True, manual_cleanup=manual_cleanup, num_devices=4
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_few_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name, n_layer=n_layer, manual_cleanup=manual_cleanup)


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("use_onnx_subfunctions", [False, True])
@pytest.mark.parametrize("model_name", test_models_causal)
def test_few_causal_lm_onnx_mdp_compile_only(model_name, use_onnx_subfunctions, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        manual_cleanup=manual_cleanup,
        compile_only=True,
        mdp_num_partitions=2,
        mdp_strategy="onnx",
        use_onnx_subfunctions=use_onnx_subfunctions,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_dummy_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, n_layer=n_layer, manual_cleanup=manual_cleanup)
    else:
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(model_name, config=hf_config, manual_cleanup=manual_cleanup)


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_check_kv_repeat_custom_causal_lm_pytorch_vs_ai100(model_name, manual_cleanup):
    """
    Test function to validate the PyTorch model and the Cloud AI 100 model with repeating original KV heads.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )
    if model_name in ModelConfig.REPEAT_KV_TEST_MODELS:
        if model_name in ModelConfig.QUANTIZED_MODELS:
            n_layer = get_custom_n_layers(model_name)
            check_kv_repeat_causal_lm_pytorch_vs_ai100(model_name, manual_cleanup=manual_cleanup, n_layer=n_layer)
        else:
            check_kv_repeat_causal_lm_pytorch_vs_ai100(model_name, manual_cleanup=manual_cleanup, config=hf_config)
    else:
        pytest.skip(f"Skipping {model_name} as it is not in REPEAT_KV_TEST_MODELS")


@pytest.mark.full_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_full_causal_lm_pytorch_vs_ort_vs_ai100_cb(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    if model_name in ModelConfig.FULL_MODEL_TESTS_TO_SKIP:
        pytest.skip(f"Skipping full model test for {model_name} due to resource constraints.")
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )


@pytest.mark.few_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_few_causal_lm_pytorch_vs_ort_vs_ai100_cb(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    n_layer = get_custom_n_layers(model_name)
    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        continuous_batching=True,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_dummy_causal_lm_pytorch_vs_ort_vs_ai100_cb(model_name, manual_cleanup):
    if model_name in ModelConfig.SKIPPED_MODELS:
        pytest.skip("Test skipped for this model due to issues in HF.")
    custom_config = model_config_dict[model_name]
    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_name in ModelConfig.EXTERNAL_MODELS,
        **custom_config.get("additional_params", {}),
    )
    if model_name in ModelConfig.QUANTIZED_MODELS:
        n_layer = get_custom_n_layers(model_name)
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name,
            n_layer=n_layer,
            continuous_batching=True,
            manual_cleanup=manual_cleanup,
        )
    else:
        check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
            model_name,
            config=hf_config,
            continuous_batching=True,
            manual_cleanup=manual_cleanup,
        )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_config", test_models_per_pr_causal, ids=_per_pr_id)
def test_per_pr_causal_fp16_subfunction_cb(model_config, manual_cleanup):
    _run_per_pr_qwen_causal_text_case(model_config, manual_cleanup)


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_config", test_models_per_pr_causal, ids=_per_pr_id)
def test_per_pr_causal_fp16_subfunction_cb_prefix_caching(model_config, manual_cleanup):
    _run_per_pr_qwen_causal_text_case(model_config, manual_cleanup, kv_cache_batch_size=8)


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_config", test_models_per_pr_causal, ids=_per_pr_id)
def test_per_pr_causal_fp16_subfunction_cb_ccl(model_config, manual_cleanup):
    if model_config.get("known_ccl_export_or_compile_issue"):
        pytest.xfail(model_config["known_ccl_export_or_compile_issue"])
    _run_per_pr_qwen_causal_text_case(
        model_config,
        manual_cleanup,
        comp_ctx_lengths_prefill=PER_PR_CCL_PREFILL,
        comp_ctx_lengths_decode=PER_PR_CCL_DECODE,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_config",
    [model for model in test_models_per_pr_causal if model["supports_blocking"]],
    ids=_per_pr_id,
)
def test_per_pr_causal_fp16_subfunction_cb_blocking(model_config, manual_cleanup):
    _run_per_pr_qwen_causal_text_case(
        model_config,
        manual_cleanup,
        qaic_config={"enable_blocking": True, "num_kv_blocks": 2},
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_config", test_models_per_pr_causal, ids=_per_pr_id)
def test_per_pr_causal_fp32_export_fp16_compile_subfunction_cb_ccl(model_config, manual_cleanup):
    if model_config.get("known_ccl_export_or_compile_issue"):
        pytest.xfail(model_config["known_ccl_export_or_compile_issue"])
    _run_per_pr_qwen_causal_text_case(
        model_config,
        manual_cleanup,
        torch_dtype=torch.float32,
        comp_ctx_lengths_prefill=PER_PR_CCL_PREFILL,
        comp_ctx_lengths_decode=PER_PR_CCL_DECODE,
    )


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_config", test_models_per_pr_causal, ids=_per_pr_id)
def test_per_pr_causal_bf16_subfunction_cb_ccl_compile_only(model_config, manual_cleanup):
    pytest.xfail(PER_PR_BF16_COMPILER_ISSUE)
    if model_config.get("known_ccl_export_or_compile_issue"):
        pytest.xfail(model_config["known_ccl_export_or_compile_issue"])
    try:
        _run_per_pr_qwen_causal_text_case(
            model_config,
            manual_cleanup,
            torch_dtype=torch.bfloat16,
            compile_only=True,
            comp_ctx_lengths_prefill=PER_PR_CCL_PREFILL,
            comp_ctx_lengths_decode=PER_PR_CCL_DECODE,
            num_cores=4,
            compile_options={"aic_hw_version": "ai200"},
        )
    except RuntimeError as exc:
        if model_config.get("known_bf16_compile_issue") and (
            "FoldRMSNorm" in str(exc) or "getHandle<float>" in str(exc)
        ):
            pytest.xfail(model_config["known_bf16_compile_issue"])
        raise


@pytest.mark.dummy_layers
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize(
    "model_config",
    [model for model in test_models_per_pr_causal if model["supports_disagg"]],
    ids=_per_pr_id,
)
def test_per_pr_causal_moe_disagg_fp16_subfunction_cb_ccl(model_config, manual_cleanup):
    _run_per_pr_qwen_causal_text_case(
        model_config,
        manual_cleanup,
        retain_full_kv=True,
        comp_ctx_lengths_prefill=PER_PR_CCL_PREFILL,
        comp_ctx_lengths_decode=PER_PR_CCL_DECODE,
    )


######################### QNN Tests #########################


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name", test_models_causal)
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_qnn(model_name, manual_cleanup):
    """
    QNN Setup
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """
    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)
    n_layer = get_custom_n_layers(model_name)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        n_layer=n_layer,
        enable_qnn=True,
        qnn_config=qnn_config_json_path,
        manual_cleanup=manual_cleanup,
    )


@pytest.mark.on_qaic
@pytest.mark.qnn
@pytest.mark.llm_model
def test_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100_pl1_qnn(manual_cleanup):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model for a prompt length of 1, both with and without continuous batching.
    """
    model_name = "gpt2"
    prompt_len = 1

    qnn_config_json_path = os.path.join(os.getcwd(), "qnn_config.json")
    create_json(qnn_config_json_path, QnnConstants.QNN_SAMPLE_CONFIG)

    check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
        model_name=model_name,
        prompt_len=prompt_len,
        enable_qnn=True,
        qnn_config=qnn_config_json_path,
        manual_cleanup=manual_cleanup,
        num_devices=4,
    )
