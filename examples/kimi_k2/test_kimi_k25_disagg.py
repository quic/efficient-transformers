# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from export_kimi_k25_vision import (
    LOADED_EXPERT_IDS,
    NUM_EXPERTS_PER_TOKEN,
    _ensure_torch_fx_import_compatibility,
    _load_layer_subset_model,
    _patch_deepseek_init_weights_compat,
    _patch_kimi_tie_weights_compat,
    _prepare_config,
)
from test_kimi_k25 import (
    _clone_inputs,
    _decode_tokens,
    _greedy_generate_hf,
    _has_qaic_runtime_access,
    _prepare_inputs,
    _resolve_model_path,
    _set_deterministic,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

PREFILL_SEQ_LEN = 32
CTX_LEN = 1024
BATCH_SIZE = 1
GENERATION_LEN = 4
NUM_VISION_LAYERS = 2
NUM_TEXT_LAYERS = 2


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _assert_distinct_onnx_paths(onnx_paths: dict[str, Path]):
    unique_paths = {str(path) for path in onnx_paths.values()}
    assert len(unique_paths) == len(onnx_paths), f"Expected distinct ONNX paths per compile, got: {onnx_paths}"


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _session_input_names(session: QAICInferenceSession) -> set[str]:
    input_names = set(session.input_names)
    input_names.update(name.rsplit("/", 1)[-1] for name in session.input_names)
    return input_names


def _cast_for_session(session: QAICInferenceSession, name: str, value: np.ndarray) -> np.ndarray:
    binding_index = session.binding_index_map.get(name)
    if binding_index is None:
        return value
    dtype = session.aic_to_np_dtype_mapping[session.bindings[binding_index].type]
    return value.astype(dtype, copy=False)


def _filter_session_inputs(session: QAICInferenceSession, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_names = _session_input_names(session)
    return {name: _cast_for_session(session, name, value) for name, value in inputs.items() if name in input_names}


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64).reshape(BATCH_SIZE, 1)


def _update_retained_states(target_inputs: dict[str, np.ndarray], source_outputs: dict[str, np.ndarray]):
    for output_name, value in source_outputs.items():
        output_basename = output_name.rsplit("/", 1)[-1]
        if not output_basename.endswith("_RetainedState"):
            continue
        target_inputs[output_basename.removesuffix("_RetainedState")] = value


def _load_kimi_subset_model():
    _set_deterministic(1234)
    _ensure_torch_fx_import_compatibility()
    model_path = _resolve_model_path()
    config = _prepare_config(model_path)
    kimi_cls = get_class_from_dynamic_module("modeling_kimi_k25.KimiK25ForConditionalGeneration", str(model_path))
    _patch_kimi_tie_weights_compat(kimi_cls)
    _patch_deepseek_init_weights_compat(kimi_cls)

    model, tokenizer, processor = _load_layer_subset_model(
        model_path=model_path,
        kimi_cls=kimi_cls,
        config=config,
        num_vision_layers=NUM_VISION_LAYERS,
        num_text_layers=NUM_TEXT_LAYERS,
        loaded_expert_ids=LOADED_EXPERT_IDS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOKEN,
        dtype=torch.float32,
    )
    model.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
    return model.eval().to("cpu"), tokenizer, processor


def _get_image_compile_dims(model, inputs: dict[str, torch.Tensor]) -> dict[str, int]:
    grid_thws = inputs["grid_thws"].to(torch.long)
    h = int(grid_thws[0, 1].item())
    w = int(grid_thws[0, 2].item())
    num_patches = int(inputs["pixel_values"].shape[0])
    merge_height, merge_width = model.vision_tower.merge_kernel_size
    num_images = max(num_patches // (h * w), 1)
    num_image_tokens = num_images * (h // merge_height) * (w // merge_width)
    return {"num_patches": num_patches, "h": h, "w": w, "num_image_tokens": num_image_tokens}


def _compile_disagg_qpcs(qeff_model: QEFFAutoModelForImageTextToText, compile_dims: dict[str, int]):
    common_compile_kwargs = {
        "batch_size": BATCH_SIZE,
        "ctx_len": CTX_LEN,
        "num_devices": 1,
        "num_cores": 16,
        "mxfp6_matmul": False,
        "split_model_io": True,
        "mos": 1,
        "aic_enable_depth_first": True,
        "use_onnx_subfunctions": True,
        "layerwise": False,
        **compile_dims,
    }

    compiled_onnx_paths = {}
    vision_qpc_path = qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        skip_vision=False,
        skip_lang=True,
        **common_compile_kwargs,
    )
    compiled_onnx_paths["vision"] = _assert_onnx_path(qeff_model.vision_model.onnx_path, "vision")

    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        retain_full_kv=True,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        skip_lang=False,
        **common_compile_kwargs,
    )
    compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "prefill")

    decode_qpc_path = qeff_model.compile(
        prefill_seq_len=1,
        prefill_only=False,
        skip_vision=True,
        skip_lang=False,
        **common_compile_kwargs,
    )
    compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "decode")
    _assert_distinct_onnx_paths(compiled_onnx_paths)

    return vision_qpc_path, prefill_qpc_path, decode_qpc_path, compiled_onnx_paths


def _run_disagg_qaic_generation(
    common_inputs: dict[str, torch.Tensor],
    vision_session: QAICInferenceSession,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
) -> np.ndarray:
    inputs = {name: _numpy(value) for name, value in _clone_inputs(common_inputs).items()}
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN

    inputs["input_ids"] = np.pad(
        inputs["input_ids"],
        ((0, 0), (0, padded_len - input_ids_length)),
        constant_values=1,
    )
    inputs["attention_mask"] = np.pad(
        inputs["attention_mask"],
        ((0, 0), (0, padded_len - input_ids_length)),
        constant_values=0,
    )

    grid_thws = inputs["grid_thws"].astype(np.int64)
    h = int(grid_thws[0, 1])
    w = int(grid_thws[0, 2])
    vision_inputs = {
        "pixel_values": inputs["pixel_values"],
        "h_shape": np.ones((h,), dtype=np.int64),
        "w_shape": np.ones((w,), dtype=np.int64),
    }
    vision_outputs = vision_session.run(_filter_session_inputs(vision_session, vision_inputs))
    vision_session.deactivate()

    image_embeds = vision_outputs.get("image_embeds")
    assert image_embeds is not None, f"Vision QPC did not return image_embeds. Outputs: {vision_outputs.keys()}"

    lang_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "position_ids": np.where(inputs["attention_mask"] > 0, np.arange(padded_len), -1).astype(np.int64),
        "image_embeds": image_embeds,
        "image_idx": np.zeros((BATCH_SIZE, 1), dtype=np.int64),
    }

    prefill_session.set_buffers(vision_outputs)
    chunk_inputs = lang_inputs.copy()
    prefill_outputs = None
    for chunk_idx in range(num_chunks):
        start = chunk_idx * PREFILL_SEQ_LEN
        end = (chunk_idx + 1) * PREFILL_SEQ_LEN
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, start:end]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][:, start:end]
        prefill_outputs = prefill_session.run(_filter_session_inputs(prefill_session, chunk_inputs))
        _update_retained_states(chunk_inputs, prefill_outputs)
        if "image_idx_output" in prefill_outputs:
            chunk_inputs["image_idx"] = prefill_outputs["image_idx_output"].astype(np.int64)

    prefill_session.deactivate()
    assert prefill_outputs is not None, "QAIC prefill did not execute."

    generated_ids = [_get_next_token_ids(prefill_outputs["logits"])]
    decode_inputs = {
        "input_ids": generated_ids[-1],
        "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True).astype(np.int64) + 1,
        "image_embeds": chunk_inputs.get("image_embeds", image_embeds),
        "image_idx": chunk_inputs.get("image_idx", np.zeros((BATCH_SIZE, 1), dtype=np.int64)),
    }
    _update_retained_states(decode_inputs, prefill_outputs)

    for _ in range(1, GENERATION_LEN):
        decode_outputs = decode_session.run(_filter_session_inputs(decode_session, decode_inputs))
        generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))
        decode_inputs["input_ids"] = generated_ids[-1]
        decode_inputs["position_ids"] = decode_inputs["position_ids"] + 1
        if "image_idx_output" in decode_outputs:
            decode_inputs["image_idx"] = decode_outputs["image_idx_output"].astype(np.int64)
        _update_retained_states(decode_inputs, decode_outputs)

    return np.concatenate(generated_ids, axis=1)


@pytest.mark.on_qaic
@pytest.mark.multimodal
def test_kimi_k25_disagg_qaic_vs_hf_fp32():
    if not _has_qaic_runtime_access():
        pytest.skip("QAIC runtime is not available.")

    model, tokenizer, processor = _load_kimi_subset_model()
    inputs = _prepare_inputs(processor)
    inputs = {name: (value.to("cpu") if torch.is_tensor(value) else value) for name, value in inputs.items()}
    hf_tokens = _greedy_generate_hf(copy.deepcopy(model), _clone_inputs(inputs), max_new_tokens=GENERATION_LEN)

    qeff_model = QEFFAutoModelForImageTextToText(
        model,
        kv_offload=True,
        config=model.config,
        torch_dtype=torch.float32,
        layerwise=False,
    )

    compile_dims = _get_image_compile_dims(qeff_model.model, inputs)
    vision_qpc_path, prefill_qpc_path, decode_qpc_path, compiled_onnx_paths = _compile_disagg_qpcs(
        qeff_model,
        compile_dims,
    )
    print(f"Kimi-K2.5 disagg ONNX paths: {compiled_onnx_paths}")

    sessions = []
    try:
        vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))
        prefill_session = QAICInferenceSession(prefill_qpc_path.get("lang_prefill_qpc_path"))
        decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"))
        sessions.extend([vision_session, prefill_session, decode_session])
        qaic_tokens = _run_disagg_qaic_generation(
            common_inputs=inputs,
            vision_session=vision_session,
            prefill_session=prefill_session,
            decode_session=decode_session,
        )
    finally:
        for session in sessions:
            session.deactivate()

    print("HF:", _decode_tokens(tokenizer, hf_tokens), "\n", hf_tokens)
    print("Disagg QAIC:", _decode_tokens(tokenizer, torch.as_tensor(qaic_tokens)), "\n", qaic_tokens)

    assert qaic_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert hf_tokens.shape == (BATCH_SIZE, GENERATION_LEN)
    assert np.issubdtype(qaic_tokens.dtype, np.integer)
    assert torch.equal(hf_tokens, torch.as_tensor(qaic_tokens)), "HF and disagg QAIC tokens do not match"
