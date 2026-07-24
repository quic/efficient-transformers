# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import pytest
import torch
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText

# since layerwise is for the qwen 3.5 model, we only test with that model id.
MODEL_IDS = {
    # "qwen3_vl_moe": "tiny-random/qwen3-vl-moe",
    # "qwen3_moe": "tiny-random/qwen3-moe",
    "qwen3_5_moe": "tiny-random/qwen3.5-moe",
}

COMMON_LAYERWISE_CB_COMPILE_KWARGS = {
    "num_cores": 16,
    "num_devices": 1,
    "full_batch_size": 4,
    "aic_enable_depth_first": True,
    "split_retained_state_io": True,
    "use_onnx_subfunctions": True,
    "mos": 1,
    "layerwise": True,
    "layerwise_window_size": 1,
}

VLM_LAYERWISE_CB_COMPILE_KWARGS = {
    **COMMON_LAYERWISE_CB_COMPILE_KWARGS,
    "prefill_seq_len": 32,
    "ctx_len": 4096,
    "batch_size": 1,
    "height": 354,
    "width": 536,
    "skip_vision": True,
}

LAYERWISE_CB_COMPILE_OVERRIDES = {
    "qwen3_vl_moe": {
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
    },
    "qwen3_5_moe": {
        "mxfp6_matmul": True,
    },
    "qwen3_moe": {
        "num_cores": 8,
        "prefill_seq_len": 4,
        "ctx_len": 128,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "offload_pt_weights": False,
        "num_speculative_tokens": None,
        "enable_chunking": True,
    },
}


FULL_BATCH_SIZES = [4, 8]
PREFILL_SEQ_LEN = 1


def _build_cb_prompts(full_batch_size):
    prompt_pool = [
        "Can you describe the image in detail?",
        "What are the objects in the image?",
        "What is the main subject of the image?",
        "What colors are predominant in the image?",
    ]
    return [prompt_pool[i % len(prompt_pool)] for i in range(full_batch_size)]


def _build_compile_kwargs(model_type, layerwise=True):
    if model_type in {"qwen3_vl_moe", "qwen3_5_moe"}:
        compile_kwargs = {**VLM_LAYERWISE_CB_COMPILE_KWARGS, **LAYERWISE_CB_COMPILE_OVERRIDES[model_type]}
    else:
        compile_kwargs = {**COMMON_LAYERWISE_CB_COMPILE_KWARGS, **LAYERWISE_CB_COMPILE_OVERRIDES[model_type]}

    if not layerwise:
        compile_kwargs.pop("layerwise", None)
        compile_kwargs.pop("layerwise_window_size", None)
    return compile_kwargs


def _load_qeff_model(model_type, model_id, config, layerwise=True):
    if model_type == "qwen3_moe":
        return QEFFAutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            layerwise=layerwise,
            continuous_batching=True,
        )

    config.torch_dtype = "float32"
    return QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        torch_dtype=torch.float32,
        layerwise=layerwise,
        continuous_batching=True,
    )


def _compile_cb_qpc_skip_vision(qeff_model, layerwise, full_batch_size):
    compile_kwargs = {
        "batch_size": 1,
        "full_batch_size": full_batch_size,
        "ctx_len": 4096,
        "height": 354,
        "width": 536,
        "num_cores": 16,
        "num_devices": 1,
        "mos": 1,
        "mxfp6_matmul": False,
        "mxint8_kv_cache": False,
        "use_onnx_subfunctions": True,
        "prefill_seq_len": PREFILL_SEQ_LEN,
        "skip_vision": True,
    }
    if layerwise:
        compile_kwargs["layerwise"] = True
        compile_kwargs["layerwise_window_size"] = 1
    qeff_model.compile(**compile_kwargs)


@pytest.mark.on_qaic
@pytest.mark.regular
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    [
        pytest.param("qwen3_5_moe", MODEL_IDS["qwen3_5_moe"]),
    ],
)
@pytest.mark.parametrize("full_batch_size", FULL_BATCH_SIZES)
def test_qwen_layerwise_vs_non_layerwise_cb_tokens_match(manual_cleanup, model_type, model_id, full_batch_size):
    generation_len = 10
    prompts = _build_cb_prompts(full_batch_size)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, padding=True)
    messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}] for prompt in prompts]
    text_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    config_layerwise = AutoConfig.from_pretrained(model_id)
    qeff_model_layerwise = _load_qeff_model(model_type, model_id, config_layerwise, layerwise=True)
    _compile_cb_qpc_skip_vision(qeff_model_layerwise, layerwise=True, full_batch_size=full_batch_size)
    layerwise_inputs = qeff_model_layerwise.model.prepare_inputs_for_generation(
        inputs=text_inputs.copy(), prefill_seq_len=PREFILL_SEQ_LEN, batch_size=full_batch_size
    )

    layerwise_exec_info = qeff_model_layerwise.generate(
        inputs=layerwise_inputs,
        generation_len=generation_len,
    )
    layerwise_tokens = layerwise_exec_info.generated_ids[:, :generation_len]

    config_non_layerwise = AutoConfig.from_pretrained(model_id)
    qeff_model_non_layerwise = _load_qeff_model(model_type, model_id, config_non_layerwise, layerwise=False)
    _compile_cb_qpc_skip_vision(qeff_model_non_layerwise, layerwise=False, full_batch_size=full_batch_size)
    non_layerwise_inputs = qeff_model_non_layerwise.model.prepare_inputs_for_generation(
        inputs=text_inputs.copy(), prefill_seq_len=PREFILL_SEQ_LEN, batch_size=full_batch_size
    )

    non_layerwise_exec_info = qeff_model_non_layerwise.generate(
        inputs=non_layerwise_inputs,
        generation_len=generation_len,
    )
    non_layerwise_tokens = non_layerwise_exec_info.generated_ids[:, :generation_len]

    assert layerwise_tokens.shape[0] == full_batch_size
    assert non_layerwise_tokens.shape[0] == full_batch_size
    assert (layerwise_tokens == non_layerwise_tokens).all(), "Layerwise and non-layerwise tokens do not match"

    if qeff_model_layerwise.onnx_path is not None:
        manual_cleanup(qeff_model_layerwise.onnx_path)
    if qeff_model_non_layerwise.onnx_path is not None:
        manual_cleanup(qeff_model_non_layerwise.onnx_path)
