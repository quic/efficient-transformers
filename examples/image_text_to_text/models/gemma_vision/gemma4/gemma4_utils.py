# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import numpy as np

from QEfficient.base.onnx_transforms import FP16ClipTransform

CHAT_TEMPLATE = """
{%- for message in messages %}
    {%- if loop.index0 == 0 %}
        {{- bos_token }}
    {%- endif %}
    {{- '<|turn|>' + message['role'] + '\n' }}
    {%- if message['content'] is string %}
        {{- message['content'] }}
    {%- else %}
        {%- for content in message['content'] %}
            {%- if content['type'] == 'image' %}
                {{- image_token }}
            {%- elif content['type'] == 'text' %}
                {{- content['text'] }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- '<turn|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|turn|>assistant\n' }}
{%- endif %}
"""


def build_messages(system_prompt: str, user_prompt: str, use_image: bool):
    messages = []
    if system_prompt and not use_image:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )

    if use_image:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        )
    else:
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            }
        )

    return messages


def build_compile_kwargs(*, effective_prefill_seq_len: int, effective_ctx_len: int, skip_vision: bool, **kwargs):
    kwargs = {
        "prefill_seq_len": effective_prefill_seq_len,
        "ctx_len": effective_ctx_len,
        "num_cores": kwargs["NUM_CORES"],
        "num_devices": kwargs["NUM_DEVICES"],
        "mxfp6_matmul": kwargs["MXFP6_MATMUL"],
        "mxint8_kv_cache": kwargs["MXINT8_KV_CACHE"],
        "aic_enable_depth_first": kwargs["AIC_ENABLE_DEPTH_FIRST"],
        "mos": kwargs["MOS"],
        "use_onnx_subfunctions": kwargs["USE_ONNX_SUBFUNCTIONS"],
        "split_model_io": kwargs.get("split_model_io", True),
        "batch_size": kwargs.get("BATCH_SIZE", 1),
        "node_precision_info": kwargs.get("node_precision_info", None),
    }
    if skip_vision:
        kwargs["skip_vision"] = True
    if kwargs["node_precision_info"] is None:
        kwargs["node_precision_info"] = False
    return kwargs

    return kwargs


def remove_fp16clip_transform_if_disabled(model, effective_fp16clip: bool):
    """
    Remove FP16ClipTransform from ONNX transforms when FP16 clipping is disabled.
    """
    if not effective_fp16clip:
        # ---- language model
        if hasattr(model, "lang_model") and hasattr(model.lang_model, "_onnx_transforms"):
            model.lang_model._onnx_transforms = [
                t for t in model.lang_model._onnx_transforms if t is not FP16ClipTransform
            ]
        # ---- vision model (optional)
        if getattr(model, "vision_model", None) is not None:
            if hasattr(model.vision_model, "_onnx_transforms"):
                model.vision_model._onnx_transforms = [
                    t for t in model.vision_model._onnx_transforms if t is not FP16ClipTransform
                ]


def normalize_generated_ids(generated_ids):
    array = np.asarray(generated_ids)
    if array.dtype == object:
        array = np.asarray([np.asarray(row).reshape(-1) for row in generated_ids], dtype=np.int64)
    array = np.asarray(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array.astype(np.int64, copy=False)


def resolve_stop_token_ids(model, config, tokenizer):
    """
    Resolve Gemma stop-token IDs from generation config / model config / tokenizer.
    """
    stop_ids = []

    generation_config = getattr(getattr(model, "model", None), "generation_config", None)
    candidates = [
        getattr(generation_config, "eos_token_id", None),
        getattr(config, "eos_token_id", None),
        getattr(getattr(config, "text_config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple, set)):
            stop_ids.extend(int(token_id) for token_id in candidate if token_id is not None)
        else:
            stop_ids.append(int(candidate))

    # Keep order stable while removing duplicates.
    return list(dict.fromkeys(stop_ids))


def truncate_generated_ids_at_stop(generated_ids, stop_token_ids):
    """
    Trim each generated sequence at the first stop token (exclusive).
    """
    if not stop_token_ids:
        return [row.tolist() for row in generated_ids]

    stop_set = set(int(x) for x in stop_token_ids)
    truncated = []
    for row in generated_ids:
        stop_index = next((idx for idx, token_id in enumerate(row) if int(token_id) in stop_set), len(row))
        truncated.append(row[:stop_index].tolist())
    return truncated


def effective_lens(model, prefill_seq_len: int, ctx_len: int, prompt_len: int, generation_len: int, skip_vision: bool):
    effective_ctx_len = max(ctx_len, prompt_len + generation_len)
    if skip_vision:
        effective_prefill_seq_len = prefill_seq_len
    else:
        effective_prefill_seq_len = max(prefill_seq_len, prompt_len)
    return effective_prefill_seq_len, effective_ctx_len
