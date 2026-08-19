# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Token-level parity tests for the Gemma4-MoE disaggregated prefill/decode DMA path.

Run the regular HF/QAIC parity test with:
    pytest -m "on_qaic and disagg_dma" \
        tests/transformers/disaggregated/test_gemma4_moe_disagg_kv_share_w_hf_fp32.py

Run the nightly full-model HF/ORT/QAIC three-way parity test with:
    pytest -m "nightly_disagg" \
        tests/transformers/disaggregated/test_gemma4_moe_disagg_kv_share_w_hf_fp32.py

"""

import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.base.onnx_transforms import FP16ClipTransform
from QEfficient.generation.cloud_infer import QAICInferenceSession
from tests.transformers.disaggregated._disagg_ort_test_utils import (
    assert_three_way_tokens_match as _assert_three_way_tokens_match,
)
from tests.transformers.disaggregated._disagg_ort_test_utils import (
    ensure_session_inputs as _ensure_session_inputs,
)
from tests.transformers.disaggregated._disagg_ort_test_utils import (
    session_input_names as _session_input_names,
)
from tests.transformers.disaggregated._disagg_ort_test_utils import (
    session_output_names as _session_output_names,
)
from tests.transformers.disaggregated._disagg_ort_test_utils import (
    update_state_from_outputs as _update_state_from_outputs,
)
from tests.transformers.disaggregated._nightly_disagg_config import nightly_disagg_configs

MODEL_NAME = "tiny-random/gemma-4-moe"
SYSTEM_PROMPT = "You are a helpful assistant."
NUM_HIDDEN_LAYERS = 2
VISION_DEPTH = 2
MOE_PREFILL_PACKED_CHUNK_SIZE = 256
PREFILL_SEQ_LEN = 256
CTX_LEN = 4096
BATCH_SIZE = 1
GENERATION_LEN = 30
IMAGE_SIZE = (536, 354)
VISION_SIZE = 280
ORT_EXTRA_DIMS = {
    "vision_size": VISION_SIZE,
    "height": IMAGE_SIZE[1],
    "image_height": IMAGE_SIZE[1],
    "width": IMAGE_SIZE[0],
    "image_width": IMAGE_SIZE[0],
}
TEXT_PROMPT = "Can you describe this image in detail?"
# Set QEFF_GEMMA4_SKIP_VISION=1 to exercise only the language  path.
SKIP_VISION = os.environ.get("QEFF_GEMMA4_SKIP_VISION", "1").strip().lower() in {"1", "true", "yes"}

PREFILL_NUM_DEVICES = 2
DECODE_NUM_DEVICES = 2
PREFILL_MDP_PARTITIONS = 2


VISION_INPUT_KEYS = {
    "pixel_values",
    "image_position_ids",
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
}
VISION_FP16_KEYS = {"pixel_values", "image_masks"}
VISION_OUTPUTS = ("vision_embeds",)

# Minimal Gemma4 chat template (from the sample's gemma4_utils); only used if neither the
# processor nor the tokenizer ships one.
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


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _load_hf_model_from_pretrained(config, model_name: str = MODEL_NAME):
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=config.torch_dtype,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=config.torch_dtype,
        )
    model.eval()
    return model


def _apply_reduced_layer_config(config, num_lang_layers: int, num_vision_layers: int):
    """Truncate depth so the compile/run stays cheap.

    Mirrors ``gemma4_example._apply_reduced_layer_config``: slice the pretrained
    ``layer_types`` (rather than regenerate the sliding/full pattern) so the truncated
    config stays consistent with whatever pattern the checkpoint actually ships.
    """
    config.text_config.num_hidden_layers = num_lang_layers
    config.vision_config.num_hidden_layers = num_vision_layers

    if hasattr(config.text_config, "layer_types") and config.text_config.layer_types:
        config.text_config.layer_types = config.text_config.layer_types[:num_lang_layers]

    if hasattr(config.text_config, "num_kv_shared_layers"):
        # KV sharing to avoid invalid first_kv_shared_layer_idx=0 edge cases.
        config.text_config.num_kv_shared_layers = 0

    return config


def _build_config(dtype: str = "float32", full_model: bool = False, model_name: str = MODEL_NAME):
    """Load the real config; optionally truncate depth so the compile/run stays cheap."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)

    if not full_model:
        config = _apply_reduced_layer_config(
            config,
            num_lang_layers=NUM_HIDDEN_LAYERS,
            num_vision_layers=VISION_DEPTH,
        )

    text_config = config.text_config if hasattr(config, "text_config") else config
    text_config.dtype = dtype
    text_config.torch_dtype = getattr(torch, dtype)
    return config


def _remove_fp16clip_transform(qeff_model: QEFFAutoModelForImageTextToText):
    """Strip FP16ClipTransform from both sub-models so the fp32 export stays true fp32.

    Mirrors ``gemma4_utils.remove_fp16clip_transform_if_disabled`` for the disabled case;
    without it the fp32 QAIC path clips activations and drifts from the HF fp32 reference.
    """
    for sub in (getattr(qeff_model, "lang_model", None), getattr(qeff_model, "vision_model", None)):
        if sub is not None and hasattr(sub, "_onnx_transforms"):
            sub._onnx_transforms = [t for t in sub._onnx_transforms if t is not FP16ClipTransform]


def _active_sessions(*sessions: QAICInferenceSession | None) -> list[QAICInferenceSession]:
    return [session for session in sessions if session is not None]


def _prepare_messages(image: Image.Image) -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TEXT_PROMPT},
            ],
        }
    ]


def _prepare_text_only_messages() -> list:
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": TEXT_PROMPT}],
        }
    ]


def _resolve_chat_template(processor, tokenizer) -> str:
    return getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None) or CHAT_TEMPLATE


def _prepare_processor_inputs(processor, chat_template: str, messages: list) -> dict:
    return dict(
        processor.apply_chat_template(
            messages,
            chat_template=chat_template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    )


def _prepare_disagg_np_inputs(processor, common_inputs: dict, *, cast_vision_fp16: bool) -> tuple[dict, dict, int]:
    inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in common_inputs.items()
    }

    pad_token_id = processor.tokenizer.pad_token_id or 1
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN
    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"],
        (0, padded_len - input_ids_length),
        "constant",
        pad_token_id,
    )
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"],
        (0, padded_len - input_ids_length),
        "constant",
        0,
    )
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.nn.functional.pad(
            inputs["mm_token_type_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            0,
        )
    np_inputs = {name: np.array(value) for name, value in inputs.items()}

    if SKIP_VISION:
        vision_inputs = {}
    else:
        vision_inputs = {name: value for name, value in np_inputs.items() if name in VISION_INPUT_KEYS}
    vision_dtype = "float16" if cast_vision_fp16 else "float32"
    vision_inputs.update(
        {name: vision_inputs[name].astype(vision_dtype) for name in VISION_FP16_KEYS if name in vision_inputs}
    )

    lang_inputs = {name: value for name, value in np_inputs.items() if name not in vision_inputs}
    if "position_ids" in np_inputs:
        lang_inputs["position_ids"] = np_inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    else:
        lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)

    if "mm_token_type_ids" not in lang_inputs:
        lang_inputs["mm_token_type_ids"] = np.zeros((BATCH_SIZE, padded_len), dtype=np.int64)

    lang_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
    return vision_inputs, lang_inputs, num_chunks


def _patch_ort_model_for_session(path: Path) -> Path:
    """Patch QEfficient's compile ONNX in place so ORT can load fp32 cache helpers."""
    try:
        import onnx
        from onnx import TensorProto, helper
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("onnx is required for ORT parity test.") from exc

    model = onnx.load(str(path), load_external_data=False)
    changed = False
    int32_max = 2147483647

    def _make_const_i64_1d(name: str, values: list):
        return helper.make_node(
            "Constant",
            [],
            [name],
            value=helper.make_tensor(name, TensorProto.INT64, [len(values)], values),
        )

    all_graph_nodes = list(model.graph.node)

    def _formal_param_for(function, call_arg: str) -> str | None:
        for node in all_graph_nodes:
            if node.op_type == function.name:
                for idx, inp in enumerate(node.input):
                    if inp == call_arg and idx < len(function.input):
                        return function.input[idx]
        for caller in model.functions:
            for node in caller.node:
                if node.op_type == function.name:
                    for idx, inp in enumerate(node.input):
                        if inp == call_arg and idx < len(function.input):
                            return function.input[idx]
        return None

    for function in model.functions:
        function_changed = False
        new_nodes = []

        if function.name == "CustomRMSNorm":
            for node in function.node:
                if node.op_type == "Cast" and list(node.input) == ["weight"] and list(node.output) == ["weight_0"]:
                    node = helper.make_node(
                        "CastLike",
                        ["weight", "hidden_states"],
                        ["weight_0"],
                        name=node.name or "CastLike_weight",
                    )
                    function_changed = True
                if node.op_type == "Expand" and list(node.output) == ["epsilon_2"]:
                    node.output[0] = "epsilon_2_pre_cast"
                    new_nodes.append(node)
                    new_nodes.append(
                        helper.make_node(
                            "CastLike",
                            ["epsilon_2_pre_cast", "variance"],
                            ["epsilon_2"],
                            name="CastLike_epsilon",
                        )
                    )
                    function_changed = True
                    continue
                new_nodes.append(node)

        elif function.name.startswith("CtxScatter"):
            pos_param = _formal_param_for(function, "position_ids") or next(
                (param for param in function.input if "pos" in param.lower()), None
            )
            if pos_param is not None:
                new_nodes.extend(
                    [
                        helper.make_node("Shape", ["data"], ["_sc_data_shape"]),
                        _make_const_i64_1d("_sc_dim1_idx", [1]),
                        helper.make_node("Gather", ["_sc_data_shape", "_sc_dim1_idx"], ["_sc_ctx_dim"]),
                        _make_const_i64_1d("_sc_one", [1]),
                        helper.make_node("Sub", ["_sc_ctx_dim", "_sc_one"], ["_sc_last_i64"]),
                        helper.make_node("CastLike", ["_sc_last_i64", pos_param], ["_sc_last"]),
                        _make_const_i64_1d("_sc_inv_i64", [int32_max]),
                        helper.make_node("CastLike", ["_sc_inv_i64", pos_param], ["_sc_inv"]),
                        helper.make_node("Equal", [pos_param, "_sc_inv"], ["_sc_inv_mask"]),
                        helper.make_node("Where", ["_sc_inv_mask", "_sc_last", pos_param], ["_sc_safe_pos"]),
                        helper.make_node(
                            "Cast",
                            ["_sc_safe_pos"],
                            ["_sc_pos_i64"],
                            name="Cast_sc_pos_i64",
                            to=TensorProto.INT64,
                        ),
                    ]
                )
                function_changed = True

            for node in function.node:
                if pos_param is not None:
                    for idx, inp in enumerate(node.input):
                        if inp == pos_param:
                            node.input[idx] = "_sc_pos_i64"
                if node.op_type == "ScatterND":
                    indices_in = node.input[1]
                    if not indices_in.endswith("_i64"):
                        cast_out = indices_in + "_i64"
                        new_nodes.append(
                            helper.make_node(
                                "Cast",
                                [indices_in],
                                [cast_out],
                                name=f"Cast_{indices_in}_i64",
                                to=TensorProto.INT64,
                            )
                        )
                        node.input[1] = cast_out
                        function_changed = True
                if node.op_type == "Cast" and list(node.output) == ["batch_idx_3"]:
                    for attr in node.attribute:
                        if attr.name == "to":
                            attr.i = TensorProto.INT64
                    function_changed = True
                if node.op_type == "Expand" and list(node.output) == ["ctx_idx"]:
                    node.output[0] = "ctx_idx_pre_i64"
                    new_nodes.append(node)
                    new_nodes.append(
                        helper.make_node(
                            "Cast",
                            ["ctx_idx_pre_i64"],
                            ["ctx_idx"],
                            name="Cast_ctx_idx_i64",
                            to=TensorProto.INT64,
                        )
                    )
                    function_changed = True
                    continue
                new_nodes.append(node)

        elif function.name.startswith("CtxGather"):
            ctx_indices_param = next((param for param in function.input if "ctx_indices" in param.lower()), None)
            pos_param = next((param for param in function.input if "pos" in param.lower()), None)
            clamp_target = pos_param or ctx_indices_param
            if clamp_target is not None:
                new_nodes.extend(
                    [
                        _make_const_i64_1d("_gc_inv_i64", [int32_max]),
                        helper.make_node("CastLike", ["_gc_inv_i64", clamp_target], ["_gc_inv"]),
                        helper.make_node("Equal", [clamp_target, "_gc_inv"], ["_gc_inv_mask"]),
                        _make_const_i64_1d("_gc_zero_i64", [0]),
                        helper.make_node("CastLike", ["_gc_zero_i64", clamp_target], ["_gc_zero"]),
                        helper.make_node("Where", ["_gc_inv_mask", "_gc_zero", clamp_target], ["_gc_safe_target"]),
                        helper.make_node(
                            "Cast",
                            ["_gc_safe_target"],
                            ["_gc_target_i64"],
                            name="Cast_gc_target_i64",
                            to=TensorProto.INT64,
                        ),
                    ]
                )
                function_changed = True

            inserted_ctx_indices_cast = False
            for node in function.node:
                if clamp_target is not None:
                    for idx, inp in enumerate(node.input):
                        if inp == clamp_target:
                            node.input[idx] = "_gc_target_i64"
                if (
                    not inserted_ctx_indices_cast
                    and ctx_indices_param is not None
                    and ctx_indices_param != clamp_target
                    and node.op_type in {"Expand", "Unsqueeze", "GatherND"}
                    and ctx_indices_param in node.input
                ):
                    cast_name = f"{ctx_indices_param}_i64"
                    new_nodes.append(
                        helper.make_node(
                            "Cast",
                            [ctx_indices_param],
                            [cast_name],
                            name=f"Cast_{ctx_indices_param}_i64",
                            to=TensorProto.INT64,
                        )
                    )
                    inserted_ctx_indices_cast = True
                    function_changed = True
                    for idx, inp in enumerate(node.input):
                        if inp == ctx_indices_param:
                            node.input[idx] = cast_name
                new_nodes.append(node)

        else:
            continue

        if function_changed:
            del function.node[:]
            function.node.extend(new_nodes)
            changed = True

    if changed:
        onnx.save(model, str(path))
    return path


def _make_ort_session(path: Path):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("onnxruntime is required for ORT parity test.") from exc

    ort_path = _patch_ort_model_for_session(path)
    return ort.InferenceSession(str(ort_path), providers=["CPUExecutionProvider"])


def _run_ort_generation(onnx_paths: dict[str, Path], processor, common_inputs: dict) -> np.ndarray:
    """Run the disaggregated vision -> prefill -> decode loop with ORT using QPC ONNX graphs."""
    vision_session = None if SKIP_VISION else _make_ort_session(onnx_paths["vision"])
    prefill_session = _make_ort_session(onnx_paths["kv_share_prefill"])
    decode_session = _make_ort_session(onnx_paths["kv_share_decode"])

    if vision_session is not None:
        print(f"[ORT] vision_inputs  : {_session_input_names(vision_session)}")
    print(f"[ORT] prefill_inputs : {_session_input_names(prefill_session)}")
    print(f"[ORT] decode_inputs  : {_session_input_names(decode_session)}")

    vision_inputs, lang_inputs, num_chunks = _prepare_disagg_np_inputs(
        processor,
        common_inputs,
        cast_vision_fp16=False,
    )

    vision_outputs = {}
    if vision_session is not None:
        vision_input_names = _session_input_names(vision_session)
        vision_feed = {name: value for name, value in vision_inputs.items() if name in vision_input_names}
        vision_outputs = dict(zip(_session_output_names(vision_session), vision_session.run(None, vision_feed)))

    persistent = {name: vision_outputs[name] for name in VISION_OUTPUTS if name in vision_outputs}
    state: dict = {}
    chunk_inputs = dict(lang_inputs)
    prefill_outputs: dict = {}

    for chunk_idx in range(num_chunks):
        start = chunk_idx * PREFILL_SEQ_LEN
        end = (chunk_idx + 1) * PREFILL_SEQ_LEN
        provided = dict(persistent)
        provided["input_ids"] = lang_inputs["input_ids"][:, start:end]
        provided["position_ids"] = lang_inputs["position_ids"][..., start:end]
        provided["mm_token_type_ids"] = lang_inputs["mm_token_type_ids"][..., start:end]
        provided["image_idx"] = chunk_inputs.get("image_idx", np.array([[0]], dtype=np.int64))
        feed = _ensure_session_inputs(
            prefill_session, provided, state, PREFILL_SEQ_LEN, BATCH_SIZE, CTX_LEN, ORT_EXTRA_DIMS
        )
        prefill_outputs = dict(zip(_session_output_names(prefill_session), prefill_session.run(None, feed)))
        _update_state_from_outputs(state, prefill_outputs)
        if "image_idx_output" in prefill_outputs:
            chunk_inputs["image_idx"] = prefill_outputs["image_idx_output"]

    generated_ids = [_get_next_token_ids(prefill_outputs["logits"])]
    position_ids = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1
    decode_inputs = {
        "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
        "position_ids": position_ids,
        "mm_token_type_ids": np.zeros((BATCH_SIZE, 1), dtype=np.int64),
    }
    if "image_idx_output" in prefill_outputs:
        decode_inputs["image_idx"] = prefill_outputs["image_idx_output"]

    for _ in range(GENERATION_LEN - 1):
        provided = dict(persistent)
        provided.update(decode_inputs)
        feed = _ensure_session_inputs(decode_session, provided, state, 1, BATCH_SIZE, CTX_LEN, ORT_EXTRA_DIMS)
        decode_outputs = dict(zip(_session_output_names(decode_session), decode_session.run(None, feed)))
        _update_state_from_outputs(state, decode_outputs)
        generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))
        position_ids = position_ids + 1
        decode_inputs = {
            "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
            "position_ids": position_ids,
            "mm_token_type_ids": np.zeros((BATCH_SIZE, 1), dtype=np.int64),
        }
        if "image_idx_output" in decode_outputs:
            decode_inputs["image_idx"] = decode_outputs["image_idx_output"]

    return np.stack(generated_ids, axis=1).astype(np.int64)


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64)


def _run_hf_torch_fp32(model, inputs: dict) -> np.ndarray:
    model = model.to(dtype=torch.float32).eval()
    gen_inputs = {
        name: value.to(dtype=torch.float32) if torch.is_floating_point(value) else value
        for name, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **gen_inputs,
            max_new_tokens=GENERATION_LEN,
            min_new_tokens=GENERATION_LEN,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    prompt_len = gen_inputs["input_ids"].shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


def _run_disagg_kv_share_qaic_generation(
    processor,
    common_inputs: dict,
    vision_session: QAICInferenceSession | None,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
) -> np.ndarray:
    inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in common_inputs.items()
    }

    pad_token_id = processor.tokenizer.pad_token_id or 1
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN
    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"],
        (0, padded_len - input_ids_length),
        "constant",
        pad_token_id,
    )
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"],
        (0, padded_len - input_ids_length),
        "constant",
        0,
    )
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.nn.functional.pad(
            inputs["mm_token_type_ids"],
            (0, padded_len - input_ids_length),
            "constant",
            0,
        )
    inputs = {name: np.array(value) for name, value in inputs.items()}
    vision_outputs = {}
    if vision_session is not None:
        vision_inputs = {name: value for name, value in inputs.items() if name in VISION_INPUT_KEYS}
        vision_inputs.update(
            {name: vision_inputs[name].astype("float16") for name in VISION_FP16_KEYS if name in vision_inputs}
        )
        vision_outputs = vision_session.run(vision_inputs)
        vision_session.deactivate()
    else:
        vision_inputs = {}

    lang_inputs = {name: value for name, value in inputs.items() if name not in vision_inputs}
    if "position_ids" in inputs:
        lang_inputs["position_ids"] = inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    else:
        lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)

    if "mm_token_type_ids" not in lang_inputs:
        lang_inputs["mm_token_type_ids"] = np.zeros((BATCH_SIZE, padded_len), dtype=np.int64)

    lang_inputs["image_idx"] = np.array([[0]])

    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    vision_persist = {name: vision_outputs[name] for name in VISION_OUTPUTS if name in vision_outputs}
    prefill_session.set_persistent_inputs(vision_persist)
    decode_persist = {"mm_token_type_ids": np.zeros((BATCH_SIZE, 1), dtype=np.int64), **vision_persist}
    decode_session.set_persistent_inputs(
        {name: value for name, value in decode_persist.items() if name in decode_session.binding_index_map}
    )

    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]

    chunk_inputs = dict(lang_inputs)
    exec_idx = None
    for chunk_idx in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][
            :, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][
            ..., chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        chunk_inputs["mm_token_type_ids"] = lang_inputs["mm_token_type_ids"][
            ..., chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        last_chunk = chunk_idx == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=kv_caches if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)
        chunk_inputs["image_idx"] = prefill_session.get_outputs(index=exec_idx)["image_idx_output"]

    prefill_out = prefill_session.get_outputs(index=exec_idx)
    generated_ids = [_get_next_token_ids(prefill_out["logits"])]

    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
    position_ids = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1
    decode_inputs = {
        "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
        "position_ids": position_ids,
    }
    if decode_has_image_idx:
        decode_inputs["image_idx"] = prefill_out["image_idx_output"]

    for _ in range(GENERATION_LEN - 1):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        decode_outputs = decode_session.get_outputs(index=exec_idx)
        generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))
        position_ids = position_ids + 1
        decode_inputs = {
            "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
            "position_ids": position_ids,
        }
        if decode_has_image_idx:
            decode_inputs["image_idx"] = decode_outputs["image_idx_output"]

    return np.stack(generated_ids, axis=1)


@pytest.mark.nightly_disagg
@pytest.mark.parametrize("nightly_config", nightly_disagg_configs("gemma4_moe"))
def test_gemma4_moe_disagg_kv_share_qaic_vs_ort_vs_hf_fp32(manual_cleanup, nightly_config):
    """Three-way parity: HF fp32 == ORT on QPC ONNX == QAIC disagg DMA."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    torch.manual_seed(42)

    model_id = nightly_config["model_id"]
    hf_model = _load_hf_model_from_pretrained(
        _build_config(dtype="float32", full_model=True, model_name=model_id),
        model_name=model_id,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if SKIP_VISION:
        messages = _prepare_text_only_messages()
    else:
        messages = _prepare_messages(Image.new("RGB", IMAGE_SIZE, color=(127, 127, 127)))
    chat_template = _resolve_chat_template(processor, processor.tokenizer)
    common_inputs = _prepare_processor_inputs(processor, chat_template, messages)
    hf_tokens = _run_hf_torch_fp32(hf_model, common_inputs)

    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32
    qeff_model = QEFFAutoModelForImageTextToText(
        hf_model,
        attn_implementation="eager",
        kv_offload=True,
        config=hf_model.config,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    # Disable FP16 clipping so the fp32 export stays true fp32 and can match HF fp32 exactly.
    _remove_fp16clip_transform(qeff_model)

    sessions = []
    compiled_onnx_paths = {}
    try:
        vision_qpc_path = None
        if not SKIP_VISION:
            vision_qpc_path = _compile_vision(qeff_model, num_devices=nightly_config["vision_num_devices"])
            compiled_onnx_paths["vision"] = _assert_onnx_path(qeff_model.vision_model.onnx_path, "vision")

        prefill_qpc_path, decode_qpc_path, lang_onnx_paths = _compile_kv_share_lang(
            qeff_model,
            moe_prefill_packed_chunk_size=MOE_PREFILL_PACKED_CHUNK_SIZE,
            prefill_num_devices=nightly_config["prefill_num_devices"],
            decode_num_devices=nightly_config["decode_num_devices"],
            mdp_num_partitions=nightly_config["stages"],
        )
        compiled_onnx_paths.update(lang_onnx_paths)
        print(f"Disagg ONNX paths: {compiled_onnx_paths}")

        ort_tokens = _run_ort_generation(
            onnx_paths=compiled_onnx_paths,
            processor=processor,
            common_inputs=common_inputs,
        )

        vision_session = None if SKIP_VISION else QAICInferenceSession(vision_qpc_path)
        prefill_session = QAICInferenceSession(prefill_qpc_path, kv_dma_share=True)
        decode_session = QAICInferenceSession(decode_qpc_path, kv_dma_share=True)
        sessions.extend(_active_sessions(vision_session, prefill_session, decode_session))

        qaic_tokens = _run_disagg_kv_share_qaic_generation(
            processor=processor,
            common_inputs=common_inputs,
            vision_session=vision_session,
            prefill_session=prefill_session,
            decode_session=decode_session,
        )
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(compiled_onnx_paths.values()) or [
            getattr(qeff_model.vision_model, "onnx_path", None),
            getattr(qeff_model.lang_model, "onnx_path", None),
        ]
        manual_cleanup([path for path in cleanup_paths if path is not None])

    hf_text = processor.tokenizer.batch_decode(hf_tokens, skip_special_tokens=True)
    ort_text = processor.tokenizer.batch_decode(ort_tokens, skip_special_tokens=True)
    qaic_text = processor.tokenizer.batch_decode(qaic_tokens, skip_special_tokens=True)
    print(f"HF   tokens : {hf_tokens.tolist()}")
    print(f"ORT  tokens : {ort_tokens.tolist()}")
    print(f"QAIC tokens : {qaic_tokens.tolist()}")
    print(f"HF   text   : {hf_text}")
    print(f"ORT  text   : {ort_text}")
    print(f"QAIC text   : {qaic_text}")

    _assert_three_way_tokens_match(hf_tokens, ort_tokens, qaic_tokens, BATCH_SIZE, GENERATION_LEN)


@pytest.mark.on_qaic
@pytest.mark.disagg_dma
def test_gemma4_moe_disagg_kv_share_qaic_vs_hf_fp32(manual_cleanup):
    torch.manual_seed(42)

    hf_model = _load_hf_model_from_pretrained(_build_config(dtype="float32"))
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if SKIP_VISION:
        messages = _prepare_text_only_messages()
    else:
        messages = _prepare_messages(Image.new("RGB", IMAGE_SIZE, color=(127, 127, 127)))
    chat_template = _resolve_chat_template(processor, processor.tokenizer)
    common_inputs = _prepare_processor_inputs(processor, chat_template, messages)
    hf_tokens = _run_hf_torch_fp32(hf_model, common_inputs)

    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32
    qeff_model = QEFFAutoModelForImageTextToText(
        hf_model,
        attn_implementation="eager",
        kv_offload=True,
        config=hf_model.config,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    # Disable FP16 clipping so the fp32 export stays true fp32 and can match HF fp32 exactly.
    _remove_fp16clip_transform(qeff_model)

    sessions = []
    compiled_onnx_paths = {}
    try:
        vision_qpc_path = None
        if not SKIP_VISION:
            vision_qpc_path = _compile_vision(qeff_model)
            compiled_onnx_paths["vision"] = _assert_onnx_path(qeff_model.vision_model.onnx_path, "vision")

        prefill_qpc_path, decode_qpc_path, lang_onnx_paths = _compile_kv_share_lang(
            qeff_model, moe_prefill_packed_chunk_size=MOE_PREFILL_PACKED_CHUNK_SIZE
        )
        compiled_onnx_paths.update(lang_onnx_paths)
        print(f"Disagg ONNX paths: {compiled_onnx_paths}")

        vision_session = None if SKIP_VISION else QAICInferenceSession(vision_qpc_path)
        prefill_session = QAICInferenceSession(prefill_qpc_path, kv_dma_share=True)
        decode_session = QAICInferenceSession(decode_qpc_path, kv_dma_share=True)
        sessions.extend(_active_sessions(vision_session, prefill_session, decode_session))

        qaic_tokens = _run_disagg_kv_share_qaic_generation(
            processor=processor,
            common_inputs=common_inputs,
            vision_session=vision_session,
            prefill_session=prefill_session,
            decode_session=decode_session,
        )
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(compiled_onnx_paths.values()) or [
            getattr(qeff_model.vision_model, "onnx_path", None),
            getattr(qeff_model.lang_model, "onnx_path", None),
        ]
        manual_cleanup([path for path in cleanup_paths if path is not None])

    assert qaic_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert hf_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert np.issubdtype(qaic_tokens.dtype, np.integer)
    assert np.issubdtype(hf_tokens.dtype, np.integer)

    matches = hf_tokens == qaic_tokens
    num_matched = int(matches.all(axis=0).cumprod().sum())  # leading run matched across all rows
    hf_text = processor.tokenizer.batch_decode(hf_tokens, skip_special_tokens=True)
    qaic_text = processor.tokenizer.batch_decode(qaic_tokens, skip_special_tokens=True)
    print(f"HF Torch fp32 tokens   : {hf_tokens.tolist()}")
    print(f"Disagg QAIC DMA tokens : {qaic_tokens.tolist()}")
    print(f"HF Torch fp32 text     : {hf_text}")
    print(f"Disagg QAIC DMA text   : {qaic_text}")
    print(f"Matched leading tokens : {num_matched}/{GENERATION_LEN}")

    if not matches.all():
        first_mismatch = int(np.argmin(matches.all(axis=0)))
        raise AssertionError(
            "Tokens don't match for HF Torch fp32 output and disagg QAIC DMA output; "
            f"first mismatch at token index {first_mismatch} "
            f"(matched {num_matched}/{GENERATION_LEN} leading tokens): "
            f"HF={hf_tokens[:, first_mismatch].tolist()} vs QAIC={qaic_tokens[:, first_mismatch].tolist()}"
        )


def _build_qeff_model(hf_model) -> QEFFAutoModelForImageTextToText:
    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32
    qeff_model = QEFFAutoModelForImageTextToText(
        hf_model,
        attn_implementation="eager",
        kv_offload=True,
        config=hf_model.config,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    # Disable FP16 clipping so the fp32 export stays true fp32 (identical for both handoffs).
    _remove_fp16clip_transform(qeff_model)
    return qeff_model


def _compile_vision(qeff_model, num_devices: int = 1) -> str:
    vision_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=num_devices,
        mos=1,
        aic_enable_depth_first=True,
        skip_vision=False,
        split_model_io=True,
        skip_lang=True,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )
    return vision_qpc_path.get("vision_qpc_path")


def _compile_kv_share_lang(
    qeff_model,
    moe_prefill_packed_chunk_size: int | None = None,
    prefill_num_devices: int = PREFILL_NUM_DEVICES,
    decode_num_devices: int = DECODE_NUM_DEVICES,
    mdp_num_partitions: int = PREFILL_MDP_PARTITIONS,
) -> tuple[str, str, dict]:
    onnx_paths = {}
    decode_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        num_cores=16,
        num_devices=decode_num_devices,
        retain_full_kv=True,  # required for DMA slice writes into full KV
        split_retained_state_io=True,
        mos=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=True,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )
    onnx_paths["kv_share_decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "kv_share decode")

    prefill_compile_kwargs = {
        "batch_size": BATCH_SIZE,
        "prefill_seq_len": PREFILL_SEQ_LEN,
        "ctx_len": CTX_LEN,
        "num_cores": 16,
        "num_devices": prefill_num_devices,
        "retain_full_kv": True,
        "split_retained_state_io": True,
        "mos": 1,
        "mxfp6_matmul": False,
        "mxint8_kv_cache": False,
        "aic_enable_depth_first": True,
        "mdp_num_partitions": mdp_num_partitions,
        "prefill_only": True,
        "enable_chunking": True,
        "skip_vision": True,
        "use_onnx_subfunctions": True,
        "offload_pt_weights": False,
    }
    if moe_prefill_packed_chunk_size is not None:
        prefill_compile_kwargs["moe_prefill_packed_chunk_size"] = moe_prefill_packed_chunk_size
    prefill_qpc_path = qeff_model.compile(**prefill_compile_kwargs)
    onnx_paths["kv_share_prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "kv_share prefill")
    return prefill_qpc_path.get("lang_prefill_qpc_path"), decode_qpc_path.get("lang_decode_qpc_path"), onnx_paths


@pytest.mark.skip("for local checking only")
@pytest.mark.on_qaic
@pytest.mark.multimodal
def test_gemma4_moe_disagg_kv_share_kv_handoff_correctness(manual_cleanup):
    torch.manual_seed(42)

    hf_model = _load_hf_model_from_pretrained(_build_config(dtype="float32"))
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    messages = _prepare_text_only_messages()
    chat_template = _resolve_chat_template(processor, processor.tokenizer)
    common_inputs = _prepare_processor_inputs(processor, chat_template, messages)
    prompt_len = int(common_inputs["input_ids"].shape[1])

    qeff_model = _build_qeff_model(hf_model)

    sessions = []
    compiled_onnx_paths = {}
    try:
        prefill_qpc_path, decode_qpc_path, share_onnx = _compile_kv_share_lang(qeff_model)
        compiled_onnx_paths.update(share_onnx)
        print(f"Disagg ONNX paths: {compiled_onnx_paths}")

        prefill_session = QAICInferenceSession(prefill_qpc_path, kv_dma_share=True)
        decode_session = QAICInferenceSession(decode_qpc_path, kv_dma_share=True)
        sessions.extend([prefill_session, decode_session])

        inputs = {
            name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for name, value in common_inputs.items()
        }

        pad_token_id = processor.tokenizer.pad_token_id or 1
        input_ids_length = inputs["input_ids"].shape[1]
        num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)
        padded_len = num_chunks * PREFILL_SEQ_LEN
        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"], (0, padded_len - input_ids_length), "constant", pad_token_id
        )
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
        )
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.nn.functional.pad(
                inputs["mm_token_type_ids"], (0, padded_len - input_ids_length), "constant", 0
            )
        lang_inputs = {name: np.array(value) for name, value in inputs.items()}

        if "position_ids" in lang_inputs:
            lang_inputs.pop("attention_mask", None)
        else:
            lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)

        if "mm_token_type_ids" not in lang_inputs:
            lang_inputs["mm_token_type_ids"] = np.zeros((BATCH_SIZE, padded_len), dtype=np.int64)

        lang_inputs["image_idx"] = np.array([[0]])

        assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
        decode_has_image_idx = "image_idx" in decode_session.binding_index_map

        decode_persist = {"mm_token_type_ids": np.zeros((BATCH_SIZE, 1), dtype=np.int64)}
        decode_session.set_persistent_inputs(
            {name: value for name, value in decode_persist.items() if name in decode_session.binding_index_map}
        )

        kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
        assert all(np.all(kv == 0) for kv in kv_caches), "KV caches are not zero-initialised before prefill"

        # -------------------- Chunked prefill --------------------
        chunk_inputs = dict(lang_inputs)
        exec_idx = None
        for chunk_idx in range(num_chunks):
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][
                :, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
            ]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                ..., chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
            ]
            chunk_inputs["mm_token_type_ids"] = lang_inputs["mm_token_type_ids"][
                ..., chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
            ]
            last_chunk = chunk_idx == num_chunks - 1
            exec_idx = prefill_session.np_run_pipeline(
                chunk_inputs,
                last_chunk=last_chunk,
                kv_cache_buffers=kv_caches if last_chunk else None,
            )
            prefill_session.complete_inf(exec_idx, is_prefill=True)
            chunk_inputs["image_idx"] = prefill_session.get_outputs(index=exec_idx)["image_idx_output"]

        prefill_out = prefill_session.get_outputs(index=exec_idx)
        first_token = _get_next_token_ids(prefill_out["logits"])
        next_pos = int(np.max(lang_inputs["position_ids"])) + 1

        written = [kv[:, :, :prompt_len, :] for kv in kv_caches]
        assert all(np.any(w != 0) for w in written), (
            "KV caches are still zero after the last prefill chunk -- DMA handoff did not write them"
        )

        pre_decode_kv = [kv.copy() for kv in kv_caches]

        # -------------------- First decode step --------------------
        decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        decode_inputs = {
            "input_ids": first_token.reshape(BATCH_SIZE, 1),
            "position_ids": np.array([[next_pos]], dtype=np.int64),
        }
        if decode_has_image_idx:
            decode_inputs["image_idx"] = prefill_out["image_idx_output"]
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        decode_session.get_outputs(index=exec_idx)

        for kv_before, kv_after in zip(pre_decode_kv, kv_caches):
            prefix_before = kv_before[:, :, :prompt_len, :]
            prefix_after = kv_after[:, :, :prompt_len, :]
            assert np.array_equal(prefix_before, prefix_after), (
                "decode step overwrote the prefill-written KV prefix -- the last prefill "
                "chunk's KV no longer matches the input KV the first decode step read"
            )
            new_pos_after = kv_after[:, :, next_pos, :]
            assert np.any(new_pos_after != 0), (
                f"decode step did not write KV at the new position {next_pos} -- "
                "write-back side of the handoff is not wired"
            )
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(compiled_onnx_paths.values()) or [
            getattr(qeff_model.vision_model, "onnx_path", None),
            getattr(qeff_model.lang_model, "onnx_path", None),
        ]
        manual_cleanup([path for path in cleanup_paths if path is not None])
