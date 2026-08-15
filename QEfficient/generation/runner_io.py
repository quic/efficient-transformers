# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import onnx
import yaml
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.generation.input_preparation import build_prefill_inputs, prepare_tokenizer, slice_prefill_inputs
from QEfficient.utils import get_padding_shape_from_config
from QEfficient.utils.logging_utils import logger

_PRECISION_ITEM_SIZES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "int8": 1,
    "mxint8": 1,
}


def _get_compile_dir(model) -> Path:
    qpc_path = getattr(model, "qpc_path", None)
    if qpc_path is not None:
        return Path(qpc_path).parent
    compile_dir = getattr(model, "compile_artifacts_path", None)
    if compile_dir is not None:
        return Path(compile_dir)
    raise TypeError("Compile the model or generate compile artifacts before writing runner inputs.")


def load_prefill_specialization(compile_dir: Union[str, Path]) -> Dict[str, int]:
    """Return the first prefill specialization from a compile workspace."""
    specializations_path = Path(compile_dir) / "specializations.json"
    if not specializations_path.is_file():
        raise FileNotFoundError(f"specializations.json not found at {specializations_path}.")
    specializations = json.loads(specializations_path.read_text())["specializations"]
    if not specializations:
        raise ValueError(f"No specializations found in {specializations_path}.")
    for specialization in specializations:
        symbols = specialization.get("symbols", specialization)
        if int(symbols.get("seq_len", 0)) > 1:
            return symbols
    return specializations[0].get("symbols", specializations[0])


def _custom_io_precisions(compile_dir: Path) -> Dict[str, str]:
    custom_io_path = compile_dir / "custom_io.yaml"
    if not custom_io_path.is_file():
        return {}

    entries = yaml.safe_load(custom_io_path.read_text()) or []
    return {entry["IOName"]: entry["Precision"] for entry in entries if entry.get("IOName") and entry.get("Precision")}


def _custom_io_item_sizes(compile_dir: Path) -> Dict[str, int]:
    return {
        name: _PRECISION_ITEM_SIZES[precision]
        for name, precision in _custom_io_precisions(compile_dir).items()
        if precision in _PRECISION_ITEM_SIZES
    }


def _apply_input_shapes(model: onnx.ModelProto, shape_overrides: Mapping[str, Sequence[int]]) -> None:
    for graph_input in model.graph.input:
        shape = shape_overrides.get(graph_input.name)
        if shape is None or len(graph_input.type.tensor_type.shape.dim) != len(shape):
            continue
        for dimension, value in zip(graph_input.type.tensor_type.shape.dim, shape):
            dimension.ClearField("dim_param")
            dimension.dim_value = int(value)


def _resolve_output_shape(
    output: onnx.ValueInfoProto,
    symbols: Mapping[str, int],
    fallback_batch_size: Optional[int],
    fallback_logits_width: Optional[int] = None,
) -> List[int]:
    shape = []
    for axis, dimension in enumerate(output.type.tensor_type.shape.dim):
        if dimension.HasField("dim_value") and dimension.dim_value > 0:
            shape.append(int(dimension.dim_value))
        elif dimension.dim_param in symbols:
            shape.append(int(symbols[dimension.dim_param]))
        elif axis == 0 and fallback_batch_size is not None:
            shape.append(fallback_batch_size)
        elif output.name == "logits" and axis == 1 and "seq_len" in symbols:
            shape.append(int(symbols["seq_len"]))
        elif output.name == "logits" and axis == len(output.type.tensor_type.shape.dim) - 1 and fallback_logits_width:
            shape.append(fallback_logits_width)
        else:
            raise RuntimeError(f"Cannot resolve dimension {axis} ('{dimension.dim_param}') of output '{output.name}'.")
    return shape


def _required_host_input_names(model: onnx.ModelProto) -> set[str]:
    retained_inputs = set()
    for output in model.graph.output:
        for suffix in ("_InternalRetainedState", "_RetainedState"):
            if output.name.endswith(suffix):
                retained_inputs.add(output.name[: -len(suffix)])
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    return {
        graph_input.name
        for graph_input in model.graph.input
        if graph_input.name not in retained_inputs and graph_input.name not in initializer_names
    }


def _add_specialization_control_inputs(
    onnx_path: Union[str, Path],
    host_inputs: Dict[str, np.ndarray],
    specialization: Mapping[str, int],
    sampling_params: Optional[Mapping[str, np.ndarray]] = None,
) -> None:
    input_names = _required_host_input_names(onnx.load(str(onnx_path), load_external_data=False))
    batch_size = int(specialization.get("batch_size", 1))
    if "batch_index" in input_names and "batch_index" not in host_inputs:
        host_inputs["batch_index"] = np.zeros((batch_size, 1), dtype=np.int64)
    if "comp_ctx_lengths" in input_names and "comp_ctx_lengths" not in host_inputs:
        host_inputs["comp_ctx_lengths"] = np.zeros(int(specialization["comp_ctx_lengths"]), dtype=np.int64)
    if "num_logits_to_keep" in input_names and "num_logits_to_keep" not in host_inputs:
        host_inputs["num_logits_to_keep"] = np.zeros((batch_size, 1), dtype=np.int64)
    if "lora_ids" in input_names and "lora_ids" not in host_inputs:
        host_inputs["lora_ids"] = np.zeros((batch_size, 1), dtype=np.int64)
    if "last_accepted_output_tokens" in input_names and "input_ids" in host_inputs:
        host_inputs["last_accepted_output_tokens"] = host_inputs["input_ids"].copy()
    for name, value in (sampling_params or {}).items():
        if name in input_names:
            host_inputs[name] = np.asarray(value)


def _proxy_logits_width(model) -> Optional[int]:
    if not getattr(model, "_enable_proxy", False):
        return None

    config = model.model.config
    candidates = [config, getattr(config, "text_config", None), getattr(config, "language_config", None)]
    for candidate in filter(None, candidates):
        for attribute in ("hidden_size", "n_embd", "d_model"):
            if (hidden_size := getattr(candidate, attribute, None)) is not None:
                return int(hidden_size)
    raise AttributeError("Proxy model configuration does not expose its hidden size.")


def write_runner_io_bundle(
    *,
    onnx_path: Union[str, Path],
    compile_dir: Union[str, Path],
    specialization: Mapping[str, int],
    host_inputs: Mapping[str, np.ndarray],
    input_shape_overrides: Optional[Mapping[str, Sequence[int]]] = None,
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
    unexpected_inputs = host_inputs.keys() - required_inputs
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
        io_entries.append(
            {
                "path": f"data/{output.name}.raw",
                "io-direction": "out",
                "elem-size": custom_item_sizes.get(output.name, int(np.dtype(dtype).itemsize)),
                "map-to": output.name,
                "dims": _resolve_output_shape(output, symbols, fallback_batch_size, fallback_logits_width),
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
    prepare_tokenizer(tokenizer)
    prefill_inputs, _, _ = build_prefill_inputs(tokenizer, prompts[0], prefill_seq_len)
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


def _slice_vlm_prefill_inputs(lang_inputs: Mapping[str, np.ndarray], prefill_seq_len: int) -> Dict[str, np.ndarray]:
    host_inputs = {}
    for name, value in lang_inputs.items():
        value = np.asarray(value)
        if name in {"input_ids", "position_ids", "mm_token_type_ids", "token_type_ids"}:
            host_inputs[name] = value[..., :prefill_seq_len]
        elif name == "cross_attention_mask":
            host_inputs[name] = value[:, :prefill_seq_len, ...]
        elif name in {"image_idx", "batch_index"}:
            host_inputs[name] = value
    return host_inputs


def _filter_graph_inputs(onnx_path: Union[str, Path], inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    graph_input_names = {graph_input.name for graph_input in graph.input}
    return {name: np.asarray(value) for name, value in inputs.items() if name in graph_input_names}


def write_single_qpc_vlm_runner_bundle(*, model, processor, images: List[str], prompts: List[str]) -> Path:
    """Prepare and write the first fused vision-language prefill invocation."""
    if processor is None or not images or not prompts:
        raise ValueError("`processor`, `images`, and `prompts` are required in artifact-only mode.")

    from QEfficient.generation.embedding_handler import VisionHandler

    specialization = load_prefill_specialization(_get_compile_dir(model))
    prefill_seq_len = int(specialization["seq_len"])
    handler = VisionHandler(
        qeff_model=model,
        vision_session=None,
        processor=processor,
        tokenizer=getattr(processor, "tokenizer", None),
    )
    vision_inputs, lang_inputs, _ = handler.prepare_processor_inputs(images[0], prompts[0], prefill_seq_len)
    host_inputs = _slice_vlm_prefill_inputs(lang_inputs, prefill_seq_len)
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


def _add_cross_qpc_placeholders(model, host_inputs: Dict[str, np.ndarray], specialization: Mapping[str, int]) -> None:
    graph = onnx.load(str(model.onnx_path), load_external_data=False).graph
    symbols = {name: int(value) for name, value in specialization.items() if str(value).lstrip("-").isdigit()}
    custom_precisions = _custom_io_precisions(_get_compile_dir(model))
    precision_dtypes = {
        "bfloat16": np.uint16,
        "float16": np.float16,
        "float32": np.float32,
        "int8": np.int8,
        "mxint8": np.int8,
    }
    for graph_input in graph.input:
        if graph_input.name in host_inputs or graph_input.name.startswith("past_"):
            continue
        if not any(token in graph_input.name for token in ("vision_embeds", "deepstack_features")):
            continue
        shape = []
        for dimension in graph_input.type.tensor_type.shape.dim:
            if dimension.HasField("dim_value") and dimension.dim_value > 0:
                shape.append(int(dimension.dim_value))
            elif dimension.dim_param in symbols:
                shape.append(symbols[dimension.dim_param])
            else:
                raise RuntimeError(
                    f"Cannot resolve placeholder dimension {dimension.dim_param!r} for {graph_input.name!r}."
                )
        precision = custom_precisions.get(graph_input.name)
        dtype = precision_dtypes.get(
            precision, onnx.helper.tensor_dtype_to_np_dtype(graph_input.type.tensor_type.elem_type)
        )
        host_inputs[graph_input.name] = np.zeros(shape, dtype=dtype)
        logger.warning(
            f"Wrote a zero placeholder for {graph_input.name!r}; replace it with the vision QPC output before replay."
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
        raise ValueError("Artifact-only dual-QPC generation requires exactly one skipped component.")
    if processor is None or not images or not prompts:
        raise ValueError("`processor`, `images`, and `prompts` are required in artifact-only mode.")

    from QEfficient.generation.embedding_handler import VisionHandler

    active_model = model.vision_model if skip_lang else model.lang_model
    if not getattr(active_model, "compile_artifacts_path", None) and not active_model.qpc_path:
        raise TypeError("Compile the active ImageTextToText component before generating runner inputs.")
    specialization = load_prefill_specialization(_get_compile_dir(active_model))
    prefill_seq_len = max(int(specialization.get("seq_len", 1)), 1)
    handler = VisionHandler(
        qeff_model=model,
        vision_session=None,
        processor=processor,
        tokenizer=getattr(processor, "tokenizer", None),
    )
    vision_inputs, lang_inputs, _ = handler.prepare_processor_inputs(images[0], prompts[0], prefill_seq_len)
    if skip_lang:
        host_inputs = _filter_graph_inputs(active_model.onnx_path, vision_inputs)
        shape_overrides = {}
    else:
        host_inputs = _slice_vlm_prefill_inputs(lang_inputs, prefill_seq_len)
        _add_specialization_control_inputs(active_model.onnx_path, host_inputs, specialization)
        host_inputs = _filter_graph_inputs(active_model.onnx_path, host_inputs)
        _add_cross_qpc_placeholders(active_model, host_inputs, specialization)
        shape_overrides = {}

    return write_runner_io_bundle(
        onnx_path=active_model.onnx_path,
        compile_dir=_get_compile_dir(active_model),
        specialization=specialization,
        host_inputs=host_inputs,
        input_shape_overrides=shape_overrides,
        fallback_logits_width=_proxy_logits_width(active_model),
    )
