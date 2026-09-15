# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Utilities for writing replayable qaic-runner IO bundles.

This module prepares one host-side inference invocation from exported ONNX
metadata and compile artifacts. It serializes input tensors as raw files and
writes an ``aic_batch_io.json`` descriptor that external harnesses can replay
with ``qaic-runner``. The public helpers cover CausalLM, single-QPC VLM, and
dual-QPC VLM artifacts mode generation flows.
"""

import json
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

import numpy as np
import onnx
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.generation.generation_helpers import (
    _add_cross_qpc_placeholders,
    _add_specialization_control_inputs,
    _apply_input_shapes,
    _component_prefill_symbols,
    _concat_input_batches,
    _cross_qpc_output_shapes,
    _custom_io_item_sizes,
    _custom_io_precisions,
    _execution_batch_size,
    _filter_graph_inputs,
    _first_execution_batch,
    _get_compile_dir,
    _prepare_vlm_execution_inputs,
    _proxy_logits_width,
    _required_host_input_names,
    _resolve_output_shape,
    _slice_vlm_prefill_inputs,
    _specialization_symbols,
    build_prefill_inputs,
    load_prefill_specialization,
    prepare_tokenizer,
    slice_prefill_inputs,
)
from QEfficient.utils import get_padding_shape_from_config
from QEfficient.utils.logging_utils import logger

__all__ = [
    "_add_cross_qpc_placeholders",
    "_add_specialization_control_inputs",
    "_apply_input_shapes",
    "_component_prefill_symbols",
    "_concat_input_batches",
    "_cross_qpc_output_shapes",
    "_custom_io_item_sizes",
    "_custom_io_precisions",
    "_execution_batch_size",
    "_filter_graph_inputs",
    "_first_execution_batch",
    "_get_compile_dir",
    "_prepare_vlm_execution_inputs",
    "_proxy_logits_width",
    "_required_host_input_names",
    "_resolve_output_shape",
    "_slice_vlm_prefill_inputs",
    "_specialization_symbols",
    "load_prefill_specialization",
    "write_causal_lm_runner_bundle",
    "write_dual_qpc_vlm_runner_bundle",
    "write_runner_io_bundle",
    "write_single_qpc_vlm_runner_bundle",
]


def write_runner_io_bundle(
    *,
    onnx_path: Union[str, Path],
    compile_dir: Union[str, Path],
    specialization: Mapping[str, int],
    host_inputs: Mapping[str, np.ndarray],
    input_shape_overrides: Optional[Mapping[str, Sequence[int]]] = None,
    output_shape_overrides: Optional[Mapping[str, Sequence[int]]] = None,
    fallback_logits_width: Optional[int] = None,
) -> Path:
    """Write raw inputs and ``aic_batch_io.json`` for one qaic-runner invocation."""
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Exported ONNX not found at {onnx_path}.")

    compile_dir = Path(compile_dir)
    io_dir = compile_dir / "io"
    data_dir = io_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(onnx_path), load_external_data=False)
    required_inputs = _required_host_input_names(model)
    missing_inputs = required_inputs - host_inputs.keys()
    if missing_inputs:
        raise ValueError(f"Missing qaic-runner host inputs: {sorted(missing_inputs)}")
    graph_input_names = {graph_input.name for graph_input in model.graph.input}
    unexpected_inputs = host_inputs.keys() - graph_input_names
    if unexpected_inputs:
        raise ValueError(f"Inputs are not present in the exported ONNX graph: {sorted(unexpected_inputs)}")

    io_entries = []
    numpy_inputs = {name: np.asarray(value) for name, value in host_inputs.items()}
    for name, value in numpy_inputs.items():
        value.tofile(data_dir / f"{name}.raw")
        io_entries.append(
            {
                "path": f"data/{name}.raw",
                "io-direction": "in",
                "elem-size": int(value.itemsize),
                "map-to": name,
                "dims": list(value.shape),
            }
        )

    shapes = {name: value.shape for name, value in numpy_inputs.items()}
    shapes.update(input_shape_overrides or {})
    _apply_input_shapes(model, shapes)
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception as error:
        logger.warning(f"ONNX shape inference failed while creating runner inputs: {error}")

    symbols = {name: int(value) for name, value in specialization.items() if str(value).lstrip("-").isdigit()}
    symbols["seq_len"] = 1
    batch_source = numpy_inputs.get("input_ids")
    fallback_batch_size = int(batch_source.shape[0]) if batch_source is not None else None
    custom_item_sizes = _custom_io_item_sizes(compile_dir)
    for output in model.graph.output:
        if output.name.endswith(("_RetainedState", "_InternalRetainedState")):
            continue
        dtype = onnx.helper.tensor_dtype_to_np_dtype(output.type.tensor_type.elem_type)
        output_shape = (output_shape_overrides or {}).get(output.name)
        io_entries.append(
            {
                "path": f"data/{output.name}.raw",
                "io-direction": "out",
                "elem-size": custom_item_sizes.get(output.name, int(np.dtype(dtype).itemsize)),
                "map-to": output.name,
                "dims": list(output_shape)
                if output_shape is not None
                else _resolve_output_shape(output, symbols, fallback_batch_size, fallback_logits_width),
            }
        )

    (io_dir / "aic_batch_io.json").write_text(json.dumps({"IO-files": [io_entries]}, indent=2))
    return io_dir


def write_causal_lm_runner_bundle(
    *,
    model,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prompts: List[str],
    sampling_params: Optional[Mapping[str, np.ndarray]] = None,
) -> Path:
    """Prepare and write the first CausalLM prefill invocation."""
    if not prompts:
        raise ValueError("`prompts` must contain at least one prompt.")

    specialization = load_prefill_specialization(_get_compile_dir(model))
    prefill_seq_len = int(specialization["seq_len"])
    batch_prompts = _first_execution_batch(prompts, _execution_batch_size(specialization), "prompts")
    prepare_tokenizer(tokenizer)
    prefill_inputs, _, _ = build_prefill_inputs(tokenizer, batch_prompts, prefill_seq_len)
    host_inputs = slice_prefill_inputs(prefill_inputs, 0, prefill_seq_len)

    full_batch_size = specialization.get("full_batch_size")
    _add_specialization_control_inputs(model.onnx_path, host_inputs, specialization, sampling_params=sampling_params)

    shape_overrides = {}
    context_length = specialization.get("ctx_len")
    if context_length is not None:
        cache_batch_size = int(full_batch_size or specialization["batch_size"])
        cache_shape = get_padding_shape_from_config(model.model.config, cache_batch_size, int(context_length))
        graph = onnx.load(str(model.onnx_path), load_external_data=False).graph
        shape_overrides = {
            graph_input.name: cache_shape
            for graph_input in graph.input
            if graph_input.name.startswith(("past_key.", "past_value.")) and graph_input.name not in host_inputs
        }

    return write_runner_io_bundle(
        onnx_path=model.onnx_path,
        compile_dir=_get_compile_dir(model),
        specialization=specialization,
        host_inputs=host_inputs,
        input_shape_overrides=shape_overrides,
        fallback_logits_width=_proxy_logits_width(model),
    )


def write_single_qpc_vlm_runner_bundle(*, model, processor, images: List[str], prompts: List[str]) -> Path:
    """Prepare and write the first fused vision-language prefill invocation."""
    if processor is None or not images or not prompts:
        raise ValueError("`processor`, `images`, and `prompts` are required in artifacts mode.")

    from QEfficient.generation.embedding_handler import VisionHandler

    specialization = load_prefill_specialization(_get_compile_dir(model))
    prefill_seq_len = int(specialization["seq_len"])
    batch_size = _execution_batch_size(specialization)
    handler = VisionHandler(
        qeff_model=model,
        vision_session=None,
        processor=processor,
        tokenizer=getattr(processor, "tokenizer", None),
    )
    vision_inputs, host_inputs = _prepare_vlm_execution_inputs(handler, images, prompts, prefill_seq_len, batch_size)
    host_inputs.update(vision_inputs)
    _add_specialization_control_inputs(model.onnx_path, host_inputs, specialization)
    host_inputs = _filter_graph_inputs(model.onnx_path, host_inputs)
    return write_runner_io_bundle(
        onnx_path=model.onnx_path,
        compile_dir=_get_compile_dir(model),
        specialization=specialization,
        host_inputs=host_inputs,
        fallback_logits_width=_proxy_logits_width(model),
    )


def write_dual_qpc_vlm_runner_bundle(
    *,
    model,
    processor,
    images: List[str],
    prompts: List[str],
    skip_vision: bool,
    skip_lang: bool,
) -> Path:
    """Prepare one isolated vision or language invocation for a dual-QPC VLM."""
    if skip_vision == skip_lang:
        raise ValueError(
            "Artifacts mode dual-QPC generation requires exactly one of `skip_vision=True` or `skip_lang=True`; "
            "use the same component selection passed to compile()."
        )
    if processor is None or not images or not prompts:
        raise ValueError("`processor`, `images`, and `prompts` are required in artifacts mode.")

    from QEfficient.generation.embedding_handler import VisionHandler

    active_model = model.vision_model if skip_lang else model.lang_model
    if not getattr(active_model, "compile_artifacts_path", None) and not active_model.qpc_path:
        raise TypeError("Compile the active ImageTextToText component before generating runner inputs.")
    specialization = load_prefill_specialization(_get_compile_dir(active_model))
    prefill_seq_len = max(int(specialization.get("seq_len", 1)), 1)
    batch_size = _execution_batch_size(specialization)
    handler = VisionHandler(
        qeff_model=model,
        vision_session=None,
        processor=processor,
        tokenizer=getattr(processor, "tokenizer", None),
    )
    vision_inputs, lang_inputs = _prepare_vlm_execution_inputs(handler, images, prompts, prefill_seq_len, batch_size)
    if skip_lang:
        # Some processors leave model-specific vision metadata in the language group.
        # The active ONNX graph is the source of truth for the replay invocation.
        host_inputs = _filter_graph_inputs(active_model.onnx_path, vision_inputs, lang_inputs)
        shape_overrides = {}
        output_shape_overrides = _cross_qpc_output_shapes(model, specialization)
    else:
        host_inputs = lang_inputs
        _add_specialization_control_inputs(active_model.onnx_path, host_inputs, specialization)
        host_inputs = _filter_graph_inputs(active_model.onnx_path, host_inputs)
        _add_cross_qpc_placeholders(active_model, host_inputs, specialization)
        shape_overrides = {}
        output_shape_overrides = {}

    return write_runner_io_bundle(
        onnx_path=active_model.onnx_path,
        compile_dir=_get_compile_dir(active_model),
        specialization=specialization,
        host_inputs=host_inputs,
        input_shape_overrides=shape_overrides,
        output_shape_overrides=output_shape_overrides,
        fallback_logits_width=_proxy_logits_width(active_model),
    )
