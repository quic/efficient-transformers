# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""Shared helpers for disaggregated HF/ORT/QAIC parity tests."""
from pathlib import Path

import numpy as np


def assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def assert_distinct_onnx_paths(onnx_paths: dict[str, Path]) -> None:
    unique_paths = {str(path) for path in onnx_paths.values()}
    assert len(unique_paths) == len(onnx_paths), f"Expected distinct ONNX paths per compile, got: {onnx_paths}"


def get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64)


def session_input_names(session) -> list:
    return [item.name for item in session.get_inputs()]


def session_output_names(session) -> list:
    return [item.name for item in session.get_outputs()]


def session_input_rank(session, name: str):
    for item in session.get_inputs():
        if item.name == name:
            return len(item.shape)
    return None


def qeff_rank4_image_grid_thw(grid) -> np.ndarray:
    arr = np.asarray(grid)
    if arr.ndim != 2 or arr.shape[-1] != 3:
        return arr
    if arr.shape[0] != 1:
        raise NotImplementedError(f"Rank-4 grid conversion supports one image, got {arr.shape}")
    t, h, w = (int(v) for v in arr[0])
    return np.zeros((arr.shape[0], t, h, w), dtype=arr.dtype)


def vision_feed_for_ort(vision_inputs: dict, vision_session) -> dict:
    feed = {k: v for k, v in vision_inputs.items() if k in session_input_names(vision_session)}
    if "image_grid_thw" in feed and session_input_rank(vision_session, "image_grid_thw") == 4:
        feed["image_grid_thw"] = qeff_rank4_image_grid_thw(feed["image_grid_thw"])
    return feed


def dtype_for_ort(type_name: str) -> np.dtype:
    if "float16" in type_name:
        return np.float16
    if "float" in type_name:
        return np.float32
    if "int64" in type_name:
        return np.int64
    if "int32" in type_name:
        return np.int32
    return np.float32


def resolve_ort_dim(dim, seq_len: int, batch_size: int, ctx_len: int, extra_dims: dict | None = None) -> int:
    if isinstance(dim, int):
        return dim
    if extra_dims and dim in extra_dims:
        return extra_dims[dim]
    if dim in {"batch_size", "batch", "full_batch_size", "full_batch", "vision_batch_size"}:
        return batch_size
    if dim in {"seq_len", "sequence_length"}:
        return seq_len
    if dim in {"ctx_len", "context_length", "past_sequence_length", "sliding_window"}:
        return ctx_len
    if dim in {"num_logits_to_keep"}:
        return 1
    raise ValueError(f"Cannot resolve dynamic ONNX dim {dim!r}")


def empty_input_from_meta(
    meta,
    seq_len: int,
    batch_size: int,
    ctx_len: int,
    extra_dims: dict | None = None,
) -> np.ndarray:
    shape = tuple(resolve_ort_dim(dim, seq_len, batch_size, ctx_len, extra_dims) for dim in meta.shape)
    return np.zeros(shape, dtype=dtype_for_ort(meta.type))


def ensure_session_inputs(
    session,
    provided: dict,
    state: dict,
    seq_len: int,
    batch_size: int,
    ctx_len: int,
    extra_dims: dict | None = None,
) -> dict:
    merged = {}
    for meta in session.get_inputs():
        if meta.name in provided:
            merged[meta.name] = provided[meta.name]
        elif meta.name in state:
            merged[meta.name] = state[meta.name]
        else:
            merged[meta.name] = empty_input_from_meta(meta, seq_len, batch_size, ctx_len, extra_dims)
    return merged


def update_state_from_outputs(state: dict, outputs: dict) -> None:
    for name, value in outputs.items():
        if name.endswith("_RetainedState"):
            state[name[: -len("_RetainedState")]] = value


def assert_three_way_tokens_match(
    hf_tokens: np.ndarray,
    ort_tokens: np.ndarray,
    qaic_tokens: np.ndarray,
    batch_size: int,
    generation_len: int,
) -> None:
    comparisons = (
        ("HF fp32", hf_tokens, "ORT fp32", ort_tokens),
        ("ORT fp32", ort_tokens, "QAIC disagg DMA", qaic_tokens),
        ("HF fp32", hf_tokens, "QAIC disagg DMA", qaic_tokens),
    )
    failures = []
    for label_a, tokens_a, label_b, tokens_b in comparisons:
        assert tokens_a.shape == (batch_size, generation_len), f"{label_a} shape mismatch: {tokens_a.shape}"
        assert tokens_b.shape == (batch_size, generation_len), f"{label_b} shape mismatch: {tokens_b.shape}"
        assert np.issubdtype(tokens_a.dtype, np.integer), f"{label_a} tokens are not integer dtype"
        assert np.issubdtype(tokens_b.dtype, np.integer), f"{label_b} tokens are not integer dtype"

        matching_steps = (tokens_a == tokens_b).all(axis=0)
        num_matched = int(matching_steps.cumprod().sum())
        print(f"{label_a} vs {label_b} matched leading tokens : {num_matched}/{generation_len}")
        if not matching_steps.all():
            first_mismatch = int(np.flatnonzero(~matching_steps)[0])
            failures.append(
                f"{label_a} vs {label_b}: first mismatch at token index {first_mismatch} "
                f"(matched {num_matched}/{generation_len} leading tokens): "
                f"{label_a}={tokens_a[:, first_mismatch].tolist()} vs "
                f"{label_b}={tokens_b[:, first_mismatch].tolist()}"
            )

    if failures:
        raise AssertionError("Three-way parity mismatch:\n" + "\n".join(failures))
