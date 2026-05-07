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


def resolve_npi_mode(enable_npi: bool, disable_npi: bool) -> str:
    if enable_npi and disable_npi:
        print("Warning: both ENABLE_NPI and DISABLE_NPI are set; defaulting to auto NPI mode.")
        return "auto"
    return "enabled" if enable_npi else "disabled" if disable_npi else "auto"


def build_compile_kwargs(
    *, effective_prefill_seq_len: int, effective_ctx_len: int, skip_vision: bool, npi_mode: str, **kwargs
):
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
        "vision_use_onnx_subfunctions": kwargs["VISION_USE_ONNX_SUBFUNCTIONS"],
        "lang_use_onnx_subfunctions": kwargs["LANG_USE_ONNX_SUBFUNCTIONS"],
    }

    if skip_vision:
        kwargs["skip_vision"] = True

    if npi_mode == "enabled":
        if skip_vision:
            pass
        else:
            kwargs["node_precision_info"] = True
    elif npi_mode == "disabled":
        kwargs["node_precision_info"] = False
        if not skip_vision:
            kwargs["vision_node_precision_info"] = False

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


def effective_lens(model, prefill_seq_len: int, ctx_len: int, prompt_len: int, generation_len: int, skip_vision: bool):
    effective_ctx_len = max(ctx_len, prompt_len + generation_len)
    if skip_vision:
        effective_prefill_seq_len = prefill_seq_len
    else:
        effective_prefill_seq_len = max(prefill_seq_len, prompt_len)
    return effective_prefill_seq_len, effective_ctx_len
