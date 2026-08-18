# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Token-level parity tests for the Qwen3-VL-MoE disaggregated prefill/decode DMA path.

Run the regular HF/QAIC parity test with:
    pytest -m "on_qaic and disagg_dma" \
        tests/transformers/disaggregated/test_qwen3_vl_moe_disagg_kv_share_w_hf_fp32.py

Run the nightly full HF/ORT/QAIC three-way parity test with:
    pytest -m "nightly_disagg" \
        tests/transformers/disaggregated/test_qwen3_vl_moe_disagg_kv_share_w_hf_fp32.py
"""

import copy
import os
import time
from pathlib import Path

import numpy as np
import pytest

# import requests
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
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
from tests.transformers.disaggregated._disagg_ort_test_utils import (
    vision_feed_for_ort as _vision_feed_for_ort,
)

MODEL_NAME = "tiny-random/qwen3-vl-moe"
THREE_WAY_PARITY_MODEL_NAME = os.environ.get("QEFF_QWEN3_VL_MOE_THREE_WAY_MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct")
TINY_RANDOM_MODEL_NAMES = {"tiny-random/qwen3-vl-moe"}
PREFILL_SEQ_LEN = 64
CTX_LEN = 4096
BATCH_SIZE = 1
GENERATION_LEN = 40
# IMAGE_SIZE = (1600, 1056)
IMAGE_SIZE = (536, 354)
TEXT_PROMPT = "Describe this image."

VISION_INPUTS = {
    "pixel_values",
    "image_grid_thw",
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
}
VISION_FP16_INPUTS = {"pixel_values", "image_masks"}
VISION_OUTPUTS = ("vision_embeds", "deepstack_features")

NIGHTLY_FULL_MODEL_VISION_NUM_DEVICES = int(
    os.environ.get("QEFF_QWEN3_VL_MOE_NIGHTLY_FULL_MODEL_VISION_NUM_DEVICES", "2")
)
NIGHTLY_FULL_MODEL_PREFILL_NUM_DEVICES = int(
    os.environ.get("QEFF_QWEN3_VL_MOE_NIGHTLY_FULL_MODEL_PREFILL_NUM_DEVICES", "8")
)
NIGHTLY_FULL_MODEL_DECODE_NUM_DEVICES = int(
    os.environ.get("QEFF_QWEN3_VL_MOE_NIGHTLY_FULL_MODEL_DECODE_NUM_DEVICES", "4")
)
NIGHTLY_FULL_MODEL_STAGES = int(os.environ.get("QEFF_QWEN3_VL_MOE_NIGHTLY_FULL_MODEL_STAGES", "4"))


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _assert_distinct_onnx_paths(onnx_paths: dict[str, Path]):
    unique_paths = {str(path) for path in onnx_paths.values()}
    assert len(unique_paths) == len(onnx_paths), f"Expected distinct ONNX paths per compile, got: {onnx_paths}"


def _load_hf_model_from_pretrained(config, dtype: str = "float32", model_name: str = MODEL_NAME):
    from_pretrained_kwargs = {"config": config} if config is not None else {}
    torch_dtype = getattr(config, "torch_dtype", None) if config is not None else None
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    if torch_dtype is None:
        torch_dtype = getattr(torch, dtype)

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            **from_pretrained_kwargs,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            **from_pretrained_kwargs,
        )
    model.eval()
    return model


def _normalize_text_layer_types(config):
    text_config = config.text_config if hasattr(config, "text_config") else config
    layer_types = list(getattr(text_config, "layer_types", []) or [])
    num_layers = getattr(text_config, "num_hidden_layers", len(layer_types))
    if layer_types:
        if len(layer_types) < num_layers:
            layer_types.extend(["full_attention"] * (num_layers - len(layer_types)))
        text_config.layer_types = layer_types[:num_layers]
    return config


def _build_config(dtype: str = "float32", full_model: bool = False, model_name: str = MODEL_NAME):
    if model_name in TINY_RANDOM_MODEL_NAMES:
        return None

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)
    if not full_model:
        config.vision_config.depth = 9
        config.text_config.num_hidden_layers = 4
        config.vision_config.deepstack_visual_indexes = [8]
    return _normalize_text_layer_types(config)


def _prepare_messages(image: Image.Image) -> list:
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": TEXT_PROMPT},
                ],
            }
        ]
        for _ in range(BATCH_SIZE)
    ]


def _prepare_processor_inputs(processor: AutoProcessor, messages: list) -> dict:
    process_vision_info = pytest.importorskip("qwen_vl_utils").process_vision_info

    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    return dict(processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"))


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64)


def _run_hf_torch_fp32(
    model, processor: AutoProcessor, messages: list, processor_inputs: dict | None = None
) -> np.ndarray:
    model = model.to(dtype=torch.float32).eval()
    inputs = (
        _prepare_processor_inputs(processor, messages)
        if processor_inputs is None
        else {
            name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
            for name, value in processor_inputs.items()
        }
    )
    inputs = {
        name: value.to(dtype=torch.float32) if torch.is_floating_point(value) else value
        for name, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GENERATION_LEN,
            min_new_tokens=GENERATION_LEN,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Shared QEff model / input preparation
# ---------------------------------------------------------------------------


def _build_qeff_model(hf_model) -> QEFFAutoModelForImageTextToText:
    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32
    return QEFFAutoModelForImageTextToText(
        hf_model,
        kv_offload=True,
        config=hf_model.config,
        torch_dtype=torch.float32,
        layerwise=False,
    )


def _prepare_qeff_inputs(qeff_model, processor, common_inputs, image, cast_vision_fp16: bool = True):
    """Return (np_inputs, vision_inputs, lang_inputs, num_chunks, padded_len).

    Args:
        cast_vision_fp16: If True (default, for QAIC), cast VISION_FP16_INPUTS to float16.
                          Set False for ORT runs where the exported graph expects float32.
    """
    inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in common_inputs.items()
    }
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs,
        prefill_seq_len=PREFILL_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )
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
    np_inputs = {name: np.array(value) for name, value in inputs.items()}

    vision_inputs = {name: value for name, value in np_inputs.items() if name in VISION_INPUTS}
    if cast_vision_fp16:
        # QAIC path: cast pixel_values and image_masks to float16
        vision_inputs.update(
            {name: vision_inputs[name].astype("float16") for name in VISION_FP16_INPUTS if name in vision_inputs}
        )
    else:
        # ORT fp32 path: ensure float inputs stay float32
        vision_inputs.update(
            {name: vision_inputs[name].astype("float32") for name in VISION_FP16_INPUTS if name in vision_inputs}
        )

    lang_inputs = {name: value for name, value in np_inputs.items() if name not in vision_inputs}
    if "position_ids" in np_inputs:
        lang_inputs["position_ids"] = np_inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    else:
        lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)
    lang_inputs["image_idx"] = np.array([[0]], dtype=np.int64)

    return np_inputs, vision_inputs, lang_inputs, num_chunks, padded_len


# ---------------------------------------------------------------------------
# ORT helpers (ported from the reference debug script)
# ---------------------------------------------------------------------------


def _get_specializations_for_ort_export(qeff_model, image: Image.Image, prefill_seq_len: int):
    qeff_model.transform(
        ctx_len=CTX_LEN,
        seq_len=prefill_seq_len,
        batch_size=BATCH_SIZE,
        num_devices=1,
        qaic_config=None,
        aic_num_cores=16,
    )
    specializations, _ = qeff_model.model.get_specializations(
        batch_size=BATCH_SIZE,
        prefill_seq_len=prefill_seq_len,
        ctx_len=CTX_LEN,
        comp_ctx_lengths_prefill=None,
        comp_ctx_lengths_decode=None,
        img_size=None,
        kv_offload=True,
        continuous_batching=False,
        kv_cache_batch_size=BATCH_SIZE,
        full_batch_size=None,
        height=image.height,
        width=image.width,
        mos=1,
    )
    return specializations


def _export_qeff_onnx_graphs_for_ort(qeff_model, image: Image.Image) -> dict[str, Path]:
    """Export QEfficient ONNX graphs for ORT without invoking QAIC compile."""
    specializations = _get_specializations_for_ort_export(qeff_model, image, PREFILL_SEQ_LEN)
    qeff_model.export(
        use_onnx_subfunctions=True,
        skip_vision=False,
        skip_lang=True,
        prefill_only=False,
        enable_chunking=False,
        prefill_seq_len=PREFILL_SEQ_LEN,
        offload_pt_weights=False,
        specializations=specializations,
    )
    vision_path = _assert_onnx_path(qeff_model.vision_model.onnx_path, "vision")

    specializations = _get_specializations_for_ort_export(qeff_model, image, 1)
    qeff_model.export(
        use_onnx_subfunctions=True,
        skip_vision=True,
        skip_lang=False,
        prefill_only=False,
        enable_chunking=False,
        prefill_seq_len=1,
        offload_pt_weights=False,
        specializations=specializations,
    )
    decode_path = _assert_onnx_path(qeff_model.lang_model.onnx_path, "decode")

    specializations = _get_specializations_for_ort_export(qeff_model, image, PREFILL_SEQ_LEN)
    qeff_model.export(
        use_onnx_subfunctions=True,
        skip_vision=True,
        skip_lang=False,
        prefill_only=True,
        enable_chunking=True,
        prefill_seq_len=PREFILL_SEQ_LEN,
        offload_pt_weights=False,
        specializations=specializations,
    )
    prefill_path = _assert_onnx_path(qeff_model.lang_model.onnx_path, "prefill")

    qeff_onnx_paths = {"vision": vision_path, "prefill": prefill_path, "decode": decode_path}
    _assert_distinct_onnx_paths(qeff_onnx_paths)
    print(f"QEfficient ONNX paths for ORT: {qeff_onnx_paths}")
    return qeff_onnx_paths


def _patch_custom_rmsnorm_for_ort(path: Path) -> Path:
    """Patch exported ONNX local functions so ORT can run FP32 graphs.

    Fixes applied per function type:
      CustomRMSNorm    : CastLike weight and epsilon to match hidden_states dtype.
      CtxScatter*      : Cast ScatterND indices to int64; clamp int32.max sentinel
                         in position_ids to last valid cache slot.
      CtxGather*       : Cast ctx_indices to int64; clamp int32.max sentinel
                         in ctx_indices (or position_ids) to 0 (safe read).
    """
    try:
        import onnx
        from onnx import TensorProto, helper
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("onnx is required for ORT parity test.") from exc

    model = onnx.load(str(path), load_external_data=False)
    changed = False
    INT32_MAX = 2147483647

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
        fn_changed = False
        new_nodes = []

        # --------------------------------------------------------------
        # CustomRMSNorm
        # --------------------------------------------------------------
        if function.name == "CustomRMSNorm":
            for node in function.node:
                if node.op_type == "Cast" and list(node.input) == ["weight"] and list(node.output) == ["weight_0"]:
                    node = helper.make_node(
                        "CastLike",
                        ["weight", "hidden_states"],
                        ["weight_0"],
                        name=node.name or "CastLike_weight",
                    )
                    fn_changed = True
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
                    fn_changed = True
                    continue
                new_nodes.append(node)

        # --------------------------------------------------------------
        # CtxScatter3D / CtxScatter3DInt / CtxScatter
        # --------------------------------------------------------------
        elif function.name.startswith("CtxScatter"):
            pos_param = _formal_param_for(function, "position_ids") or next(
                (p for p in function.input if "pos" in p.lower()), None
            )
            print(f"[patch] {function.name}: pos_param={pos_param!r}  inputs={list(function.input)}")

            if pos_param is not None:
                new_nodes.extend(
                    [
                        helper.make_node("Shape", ["data"], ["_sc_data_shape"]),
                        _make_const_i64_1d("_sc_dim1_idx", [1]),
                        helper.make_node("Gather", ["_sc_data_shape", "_sc_dim1_idx"], ["_sc_ctx_dim"]),
                        _make_const_i64_1d("_sc_one", [1]),
                        helper.make_node("Sub", ["_sc_ctx_dim", "_sc_one"], ["_sc_last_i64"]),
                        helper.make_node("CastLike", ["_sc_last_i64", pos_param], ["_sc_last"]),
                        _make_const_i64_1d("_sc_inv_i64", [INT32_MAX]),
                        helper.make_node("CastLike", ["_sc_inv_i64", pos_param], ["_sc_inv"]),
                        helper.make_node("Equal", [pos_param, "_sc_inv"], ["_sc_inv_mask"]),
                        helper.make_node("Where", ["_sc_inv_mask", "_sc_last", pos_param], ["_sc_safe_pos"]),
                        helper.make_node(
                            "Cast", ["_sc_safe_pos"], ["_sc_pos_i64"], name="Cast_sc_pos_i64", to=TensorProto.INT64
                        ),
                    ]
                )
                fn_changed = True

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
                        fn_changed = True

                if node.op_type == "Cast" and list(node.output) == ["batch_idx_3"]:
                    for attr in node.attribute:
                        if attr.name == "to":
                            attr.i = TensorProto.INT64
                    fn_changed = True

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
                    fn_changed = True
                    continue

                new_nodes.append(node)

        # --------------------------------------------------------------
        # CtxGather3D / CtxGather
        # --------------------------------------------------------------
        elif function.name.startswith("CtxGather"):
            fn_inputs = list(function.input)
            has_pos_param = any("pos" in p.lower() for p in fn_inputs)
            pos_param = next((p for p in fn_inputs if "pos" in p.lower()), None)
            ctx_indices_param = next((p for p in fn_inputs if "ctx_indices" in p.lower()), None)
            # sentinel travels via ctx_indices when there is no position_ids input
            clamp_target = pos_param if has_pos_param else ctx_indices_param
            print(f"[patch] {function.name}: clamp_target={clamp_target!r}  inputs={fn_inputs}")

            if clamp_target is not None:
                new_nodes.extend(
                    [
                        _make_const_i64_1d("_gc_inv_i64", [INT32_MAX]),
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
                fn_changed = True

            inserted_ctx_indices_cast = False
            for node in function.node:
                # Remap clamp_target -> _gc_target_i64
                if clamp_target is not None:
                    for idx, inp in enumerate(node.input):
                        if inp == clamp_target:
                            node.input[idx] = "_gc_target_i64"

                # Cast ctx_indices_param to int64 if it's separate from clamp_target
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
                    fn_changed = True
                    for idx, inp in enumerate(node.input):
                        if inp == ctx_indices_param:
                            node.input[idx] = cast_name

                new_nodes.append(node)

        else:
            continue  # nothing to patch

        if fn_changed:
            del function.node[:]
            function.node.extend(new_nodes)
            changed = True

    if not changed:
        print(f"[patch] No changes needed for {path.name}")
        return path

    patched_path = path.with_name(f"{path.stem}.ort.onnx")
    onnx.save(model, str(patched_path))
    print(f"[patch] Saved patched ONNX: {patched_path}")
    return patched_path


def _make_ort_session(path: Path):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("onnxruntime is required for ORT parity test.") from exc

    ort_path = _patch_custom_rmsnorm_for_ort(path)
    if ort_path != path:
        print(f"ORT patched ONNX: {path} -> {ort_path}")
    return ort.InferenceSession(str(ort_path), providers=["CPUExecutionProvider"])


def _run_ort_generation(
    onnx_paths: dict[str, Path],
    vision_inputs: dict,
    lang_inputs: dict,
    num_chunks: int,
    processor: AutoProcessor,
) -> np.ndarray:
    """Run the disaggregated vision -> prefill -> decode loop with ORT (fp32)."""
    vision_session = _make_ort_session(onnx_paths["vision"])
    prefill_session = _make_ort_session(onnx_paths["prefill"])
    decode_session = _make_ort_session(onnx_paths["decode"])

    print(f"[ORT] vision_inputs  : {_session_input_names(vision_session)}", flush=True)
    print(f"[ORT] prefill_inputs : {_session_input_names(prefill_session)}", flush=True)
    print(f"[ORT] decode_inputs  : {_session_input_names(decode_session)}", flush=True)

    # --- Vision ---
    vision_feed = _vision_feed_for_ort(vision_inputs, vision_session)
    print(
        f"[ORT] vision run start: { {name: tuple(value.shape) for name, value in vision_feed.items()} }",
        flush=True,
    )
    start_time = time.perf_counter()
    vision_outputs = dict(zip(_session_output_names(vision_session), vision_session.run(None, vision_feed)))
    print(
        "[ORT] vision run done in "
        f"{time.perf_counter() - start_time:.2f}s: "
        f"{ {name: tuple(value.shape) for name, value in vision_outputs.items()} }",
        flush=True,
    )
    persistent = {k: vision_outputs[k] for k in VISION_OUTPUTS if k in vision_outputs}

    state: dict = {}
    chunk_inputs = dict(lang_inputs)
    prefill_outputs: dict = {}

    # --- Prefill ---
    print(f"[ORT] prefill start: num_chunks={num_chunks}, seq_len={PREFILL_SEQ_LEN}", flush=True)
    for chunk_idx in range(num_chunks):
        start = chunk_idx * PREFILL_SEQ_LEN
        end = (chunk_idx + 1) * PREFILL_SEQ_LEN
        provided = dict(persistent)
        provided["input_ids"] = lang_inputs["input_ids"][:, start:end]
        provided["position_ids"] = lang_inputs["position_ids"][..., start:end]
        provided["image_idx"] = chunk_inputs.get("image_idx", np.array([[0]], dtype=np.int64))

        feed = _ensure_session_inputs(prefill_session, provided, state, PREFILL_SEQ_LEN, BATCH_SIZE, CTX_LEN)
        print(
            f"[ORT] prefill chunk {chunk_idx + 1}/{num_chunks} start: image_idx={provided['image_idx'].tolist()}",
            flush=True,
        )
        start_time = time.perf_counter()
        prefill_outputs = dict(zip(_session_output_names(prefill_session), prefill_session.run(None, feed)))
        print(
            f"[ORT] prefill chunk {chunk_idx + 1}/{num_chunks} done in "
            f"{time.perf_counter() - start_time:.2f}s: "
            f"image_idx_output={prefill_outputs.get('image_idx_output', np.array([])).tolist()}",
            flush=True,
        )
        _update_state_from_outputs(state, prefill_outputs)
        if "image_idx_output" in prefill_outputs:
            chunk_inputs["image_idx"] = prefill_outputs["image_idx_output"]

    first_token = _get_next_token_ids(prefill_outputs["logits"])
    print(f"[ORT] first token from prefill: {first_token.tolist()}", flush=True)
    tokens = [first_token]

    num_pos_sections = lang_inputs["position_ids"].shape[0]
    phys_pos = int(lang_inputs["position_ids"][0].max()) + 1
    mrope_pos = int(lang_inputs["position_ids"][1:].max()) + 1 if num_pos_sections > 1 else phys_pos

    def _decode_position_ids(next_phys: int, next_mrope: int) -> np.ndarray:
        pos = np.empty((num_pos_sections, BATCH_SIZE, 1), dtype=np.int64)
        pos[0] = next_phys
        if num_pos_sections > 1:
            pos[1:] = next_mrope
        return pos

    # --- Decode ---
    decode_inputs: dict = {
        "input_ids": first_token.reshape(BATCH_SIZE, 1),
        "position_ids": _decode_position_ids(phys_pos, mrope_pos),
    }
    if "image_idx_output" in prefill_outputs:
        decode_inputs["image_idx"] = prefill_outputs["image_idx_output"]

    print(f"[ORT] decode start: steps={GENERATION_LEN - 1}", flush=True)
    for step_idx in range(GENERATION_LEN - 1):
        provided = dict(persistent)
        provided.update(decode_inputs)
        feed = _ensure_session_inputs(decode_session, provided, state, 1, BATCH_SIZE, CTX_LEN)
        print(
            f"[ORT] decode step {step_idx + 1}/{GENERATION_LEN - 1} start: "
            f"input_ids={decode_inputs['input_ids'].reshape(-1).tolist()}",
            flush=True,
        )
        start_time = time.perf_counter()
        decode_outputs = dict(zip(_session_output_names(decode_session), decode_session.run(None, feed)))
        _update_state_from_outputs(state, decode_outputs)

        tok = _get_next_token_ids(decode_outputs["logits"])
        print(
            f"[ORT] decode step {step_idx + 1}/{GENERATION_LEN - 1} done in "
            f"{time.perf_counter() - start_time:.2f}s: token={tok.tolist()}",
            flush=True,
        )
        tokens.append(tok)
        phys_pos += 1
        mrope_pos += 1
        decode_inputs = {
            "input_ids": tok.reshape(BATCH_SIZE, 1),
            "position_ids": _decode_position_ids(phys_pos, mrope_pos),
        }
        if "image_idx_output" in decode_outputs:
            decode_inputs["image_idx"] = decode_outputs["image_idx_output"]

    token_array = np.stack(tokens, axis=1).astype(np.int64)
    print(f"[ORT] generation done: tokens={token_array.tolist()}", flush=True)
    return token_array


def _run_disagg_kv_share_qaic_generation(
    qeff_model: QEFFAutoModelForImageTextToText,
    processor: AutoProcessor,
    common_inputs: dict,
    vision_session: QAICInferenceSession,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
) -> np.ndarray:
    inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in common_inputs.items()
    }
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs,
        prefill_seq_len=PREFILL_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

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
    inputs = {name: np.array(value) for name, value in inputs.items()}

    # ---- Vision (producer): run the vision QPC once; its outputs feed every lang step ----
    vision_inputs = {name: value for name, value in inputs.items() if name in VISION_INPUTS}
    vision_inputs.update(
        {name: vision_inputs[name].astype("float16") for name in VISION_FP16_INPUTS if name in vision_inputs}
    )
    vision_outputs = vision_session.run(vision_inputs)
    vision_session.deactivate()

    lang_inputs = {name: value for name, value in inputs.items() if name not in vision_inputs}
    if "position_ids" in inputs:
        lang_inputs["position_ids"] = inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    else:
        lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)

    lang_inputs["image_idx"] = np.array([[0]])

    # image_idx must be a compiled prefill input binding; the KV-share path silently drops
    # unknown input names (warn + skip), so assert it up front. The decode QPC may not bind
    # it, so treat it as optional there and only wire it when present.
    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    # vision_embeds and deepstack_features are constant across every prefill chunk and decode
    # step, so register them once as persistent inputs instead of re-supplying them per step.
    vision_persist = {name: vision_outputs[name] for name in VISION_OUTPUTS if name in vision_outputs}
    prefill_session.set_persistent_inputs(vision_persist)
    decode_session.set_persistent_inputs(
        {name: value for name, value in vision_persist.items() if name in decode_session.binding_index_map}
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
    num_pos_sections = lang_inputs["position_ids"].shape[0]
    phys_pos = int(lang_inputs["position_ids"][0].max()) + 1
    mrope_pos = int(lang_inputs["position_ids"][1:].max()) + 1 if num_pos_sections > 1 else phys_pos

    def _decode_position_ids(next_phys: int, next_mrope: int) -> np.ndarray:
        pos = np.empty((num_pos_sections, BATCH_SIZE, 1), dtype=np.int64)
        pos[0] = next_phys
        if num_pos_sections > 1:
            pos[1:] = next_mrope
        return pos

    decode_inputs = {
        "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
        "position_ids": _decode_position_ids(phys_pos, mrope_pos),
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
        phys_pos += 1
        mrope_pos += 1
        decode_inputs = {
            "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
            "position_ids": _decode_position_ids(phys_pos, mrope_pos),
        }
        if decode_has_image_idx:
            decode_inputs["image_idx"] = decode_outputs["image_idx_output"]

    return np.stack(generated_ids, axis=1)


def _compile_disagg_qpcs(qeff_model, image: Image.Image, compiled_onnx_paths: dict[str, Path]) -> dict[str, str]:
    """Compile QEfficient QPCs and record the ONNX path used for each compile."""
    vision_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=image.height,
        width=image.width,
        num_cores=16,
        num_devices=NIGHTLY_FULL_MODEL_VISION_NUM_DEVICES,
        mos=1,
        aic_enable_depth_first=True,
        skip_vision=False,
        split_model_io=True,
        skip_lang=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=False,
    )
    compiled_onnx_paths["vision"] = _assert_onnx_path(qeff_model.vision_model.onnx_path, "vision")

    decode_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        height=image.height,
        width=image.width,
        num_cores=16,
        num_devices=NIGHTLY_FULL_MODEL_DECODE_NUM_DEVICES,
        retain_full_kv=True,
        split_retained_state_io=True,
        mos=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=True,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=False,
    )
    compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "decode")

    prefill_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=image.height,
        width=image.width,
        num_cores=16,
        num_devices=NIGHTLY_FULL_MODEL_PREFILL_NUM_DEVICES,
        retain_full_kv=True,
        split_retained_state_io=True,
        mos=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=False,
        mdp_num_partitions=NIGHTLY_FULL_MODEL_STAGES,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=False,
    )
    compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "prefill")

    _assert_distinct_onnx_paths(compiled_onnx_paths)
    print(f"Disagg ONNX paths: {compiled_onnx_paths}")
    return {
        "vision": vision_qpc_path.get("vision_qpc_path"),
        "prefill": prefill_qpc_path.get("lang_prefill_qpc_path"),
        "decode": decode_qpc_path.get("lang_decode_qpc_path"),
    }


def _create_disagg_sessions(qpc_paths: dict[str, str], sessions: list):
    vision_session = QAICInferenceSession(qpc_paths["vision"])
    prefill_session = QAICInferenceSession(qpc_paths["prefill"], kv_dma_share=True)
    decode_session = QAICInferenceSession(qpc_paths["decode"], kv_dma_share=True)
    sessions.extend([vision_session, prefill_session, decode_session])
    return vision_session, prefill_session, decode_session


@pytest.mark.nightly_disagg
def test_qwen3_vl_moe_disagg_kv_share_qaic_vs_ort_fp32(manual_cleanup):
    """Three-way parity: HF fp32 == ORT on QEfficient ONNX == QAIC disagg DMA."""
    pytest.importorskip("qwen_vl_utils")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    torch.manual_seed(42)
    hf_model = _load_hf_model_from_pretrained(
        _build_config(dtype="float32", full_model=True, model_name=THREE_WAY_PARITY_MODEL_NAME),
        model_name=THREE_WAY_PARITY_MODEL_NAME,
    )
    processor = AutoProcessor.from_pretrained(THREE_WAY_PARITY_MODEL_NAME, trust_remote_code=True)

    image = Image.new("RGB", IMAGE_SIZE, color=(127, 127, 127))
    messages = _prepare_messages(image)
    common_inputs = _prepare_processor_inputs(processor, messages)
    hf_tokens = _run_hf_torch_fp32(hf_model, processor, messages, common_inputs)

    qeff_model = _build_qeff_model(hf_model)
    sessions = []
    qeff_onnx_paths = {}
    compiled_onnx_paths = {}
    try:
        qeff_onnx_paths = _export_qeff_onnx_graphs_for_ort(qeff_model, image)
        _, vision_inputs, lang_inputs, num_chunks, _ = _prepare_qeff_inputs(
            qeff_model,
            processor,
            common_inputs,
            image,
            cast_vision_fp16=False,
        )
        ort_tokens = _run_ort_generation(qeff_onnx_paths, vision_inputs, lang_inputs, num_chunks, processor)

        qpc_paths = _compile_disagg_qpcs(qeff_model, image, compiled_onnx_paths)
        vision_session, prefill_session, decode_session = _create_disagg_sessions(qpc_paths, sessions)
        qaic_tokens = _run_disagg_kv_share_qaic_generation(
            qeff_model=qeff_model,
            processor=processor,
            common_inputs=common_inputs,
            vision_session=vision_session,
            prefill_session=prefill_session,
            decode_session=decode_session,
        )
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(qeff_onnx_paths.values()) + list(compiled_onnx_paths.values()) or [
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
@pytest.mark.multimodal
@pytest.mark.disagg_dma
def test_qwen3_vl_moe_disagg_kv_share_qaic_vs_hf_fp32(manual_cleanup):
    pytest.importorskip("qwen_vl_utils")
    torch.manual_seed(42)

    hf_model = _load_hf_model_from_pretrained(_build_config(dtype="float32"))
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # image_url = "https://thumbs.dreamstime.com/z/story-2962090.jpg"

    # image = Image.open(requests.get(image_url, stream=True).raw)
    image = Image.new("RGB", IMAGE_SIZE, color=(127, 127, 127))

    messages = _prepare_messages(image)
    common_inputs = _prepare_processor_inputs(processor, messages)
    hf_tokens = _run_hf_torch_fp32(hf_model, processor, messages, common_inputs)
    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32
    qeff_model = _build_qeff_model(hf_model)

    sessions = []
    compiled_onnx_paths = {}
    try:
        vision_qpc_path = qeff_model.compile(
            batch_size=BATCH_SIZE,
            prefill_seq_len=PREFILL_SEQ_LEN,
            ctx_len=CTX_LEN,
            height=image.height,
            width=image.width,
            num_cores=16,
            num_devices=1,
            mos=1,
            aic_enable_depth_first=True,
            skip_vision=False,
            split_model_io=True,
            skip_lang=True,
            use_onnx_subfunctions=True,
            layerwise=False,
            offload_pt_weights=False,
        )
        compiled_onnx_paths["vision"] = _assert_onnx_path(qeff_model.vision_model.onnx_path, "vision")

        decode_qpc_path = qeff_model.compile(
            batch_size=BATCH_SIZE,
            prefill_seq_len=1,
            ctx_len=CTX_LEN,
            height=image.height,
            width=image.width,
            num_cores=16,
            num_devices=1,
            retain_full_kv=True,
            split_retained_state_io=True,
            mos=1,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=True,
            prefill_only=False,
            skip_vision=True,
            use_onnx_subfunctions=True,
            layerwise=False,
            offload_pt_weights=False,
        )
        compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "decode")

        prefill_qpc_path = qeff_model.compile(
            batch_size=BATCH_SIZE,
            prefill_seq_len=PREFILL_SEQ_LEN,
            ctx_len=CTX_LEN,
            height=image.height,
            width=image.width,
            num_cores=16,
            num_devices=2,
            retain_full_kv=True,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            split_retained_state_io=True,
            mos=1,
            aic_enable_depth_first=True,
            mdp_num_partitions=2,
            prefill_only=True,
            enable_chunking=True,
            skip_vision=True,
            use_onnx_subfunctions=True,
            layerwise=False,
        )
        compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "prefill")
        _assert_distinct_onnx_paths(compiled_onnx_paths)
        print(f"Disagg ONNX paths: {compiled_onnx_paths}")

        vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))
        prefill_session = QAICInferenceSession(prefill_qpc_path.get("lang_prefill_qpc_path"), kv_dma_share=True)
        decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"), kv_dma_share=True)
        sessions.extend([vision_session, prefill_session, decode_session])

        qaic_tokens = _run_disagg_kv_share_qaic_generation(
            qeff_model=qeff_model,
            processor=processor,
            common_inputs=common_inputs,
            vision_session=vision_session,
            prefill_session=prefill_session,
            decode_session=decode_session,
        )
    finally:
        for session in sessions:
            session.deactivate()
        # cleanup_paths = list(compiled_onnx_paths.values()) or [
        #     getattr(qeff_model.vision_model, "onnx_path", None),
        #     getattr(qeff_model.lang_model, "onnx_path", None),
        # ]
        # manual_cleanup([path for path in cleanup_paths if path is not None])

    assert qaic_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert hf_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert np.issubdtype(qaic_tokens.dtype, np.integer)
    assert np.issubdtype(hf_tokens.dtype, np.integer)

    matches = hf_tokens == qaic_tokens
    num_matched = int(matches.all(axis=0).cumprod().sum())  # leading run matched across all rows
    print(f"HF Torch fp32 tokens   : {hf_tokens.tolist()}")
    print("HF output token:")
    print(processor.tokenizer.batch_decode(hf_tokens))
    print("QAIC ouptut tokens     : ")
    print(processor.tokenizer.batch_decode(qaic_tokens))
    print(f"Disagg QAIC DMA tokens : {qaic_tokens.tolist()}")
    print(f"Matched leading tokens : {num_matched}/{GENERATION_LEN}")

    if not matches.all():
        first_mismatch = int(np.argmin(matches.all(axis=0)))
        raise AssertionError(
            "Tokens don't match for HF Torch fp32 output and disagg QAIC DMA output; "
            f"first mismatch at token index {first_mismatch} "
            f"(matched {num_matched}/{GENERATION_LEN} leading tokens): "
            f"HF={hf_tokens[:, first_mismatch].tolist()} vs QAIC={qaic_tokens[:, first_mismatch].tolist()}"
        )
