# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""QEff-owned MXFP6 preparation for weight-free ONNX export."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import onnx
import torch
from onnx import TensorProto, helper
from safetensors import safe_open

from QEfficient.utils.checkpoint_utils import atomic_save, read_weight_map, write_index

MXFP6_BLOCK_SIZE = 32
MXFP6_ONNX_OPSET = 28
MXFP6_QUANTIZER_VERSION = 2
MXFP6_SUBFUNCTION_TOPOLOGY_VERSION = 1
MXFP6_MAX_FINITE = 7.5
MXFP6_PREPARED_SENTINEL = ".mxfp6_prepared"
MXFP6_SCALE_SUFFIX = ".mxfp6_scale"
MXFP6_PACKED_SUFFIX = ".mxfp6_packed"

_SCALE_DTYPE_ALIASES = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "fp8": "e8m0",
    "e8m0": "e8m0",
    "float8e8m0": "e8m0",
    "float8_e8m0fnu": "e8m0",
}

_SCALE_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class Mxfp6Config:
    """Immutable normalized MXFP6 export settings."""

    enabled: bool = False
    scale_dtype: str = "float16"


@dataclass(frozen=True)
class _FlatMxfp6Target:
    insert_before_node: onnx.NodeProto


@dataclass(frozen=True)
class _FunctionMxfp6Target:
    function: onnx.FunctionProto
    call_node: onnx.NodeProto
    formal_index: int
    formal_name: str


_Mxfp6Target = Union[_FlatMxfp6Target, _FunctionMxfp6Target]


@dataclass(frozen=True)
class _Mxfp6Replacement:
    original_name: str
    logical_key: str
    value_info: onnx.ValueInfoProto
    target: _Mxfp6Target
    packed: torch.Tensor
    scale: torch.Tensor


def normalize_mxfp6_config(enabled: bool, scale_dtype: str = "float16") -> Mxfp6Config:
    """Normalize public MXFP6 options into the internal immutable config."""
    if not enabled:
        return Mxfp6Config()
    normalized = _SCALE_DTYPE_ALIASES.get(str(scale_dtype).lower())
    if normalized is None:
        raise ValueError(
            "`mxfp6_scale_dtype` must be one of: "
            "float16/fp16/half, float32/fp32/float, bfloat16/bf16, fp8/e8m0/float8e8m0."
        )
    return Mxfp6Config(enabled=True, scale_dtype=normalized)


def _tensorproto_enum(name: str) -> Optional[int]:
    return getattr(TensorProto, name, None)


def _dql_schema_supports_opset28() -> bool:
    try:
        schema = onnx.defs.get_schema("DequantizeLinear", 28, "")
    except Exception:
        return False
    return schema.since_version >= 28


def _dql_schema_supports_output_dtype() -> bool:
    try:
        schema = onnx.defs.get_schema("DequantizeLinear", 28, "")
    except Exception:
        return False
    return "output_dtype" in schema.attributes


def validate_mxfp6_capabilities(config: Mxfp6Config, feature_name: str = "mxfp6=True") -> None:
    """Validate Python and ONNX capabilities needed for QEff-owned MXFP6 export."""
    if not config.enabled:
        return
    if sys.version_info < (3, 10):
        raise AssertionError(f"{feature_name} requires Python >= 3.10")
    missing = []
    if _tensorproto_enum("FLOAT6E2M3") is None:
        missing.append("onnx.TensorProto.FLOAT6E2M3")
    if config.scale_dtype == "e8m0" and _tensorproto_enum("FLOAT8E8M0") is None:
        missing.append("onnx.TensorProto.FLOAT8E8M0")
    if not _dql_schema_supports_opset28():
        missing.append("DequantizeLinear schema >= 28")
    if missing:
        raise AssertionError(
            f"{feature_name} requires ONNX support for QEff-owned MXFP6 export, but this environment is missing: "
            + ", ".join(missing)
        )


def _fp6_positive_codebook() -> List[Tuple[int, float]]:
    values = [(0, 0.0)]
    for exp_bits in range(4):
        for mant in range(8):
            if exp_bits == 0 and mant == 0:
                continue
            if exp_bits == 0:
                value = mant / 8.0
            else:
                value = (1.0 + mant / 8.0) * (2.0 ** (exp_bits - 1))
            values.append(((exp_bits << 3) | mant, value))
    return values


_POSITIVE_CODEBOOK = _fp6_positive_codebook()
_POSITIVE_CODES = torch.tensor([code for code, _ in _POSITIVE_CODEBOOK], dtype=torch.uint8)
_POSITIVE_VALUES = torch.tensor([value for _, value in _POSITIVE_CODEBOOK], dtype=torch.float32)
_POSITIVE_MIDPOINTS = (_POSITIVE_VALUES[:-1] + _POSITIVE_VALUES[1:]) / 2
_LOWER_TIE_IS_ODD = (_POSITIVE_CODES[:-1].to(torch.int16) % 2) == 1


def pack_fp6_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack four 6-bit codes into three bytes, LSB-first."""
    flat = codes.detach().cpu().to(torch.uint8).flatten()
    if flat.numel() % 4 != 0:
        raise ValueError("FP6 code count must be divisible by 4")
    grouped = flat.reshape(-1, 4).to(torch.int32)
    packed0 = grouped[:, 0] | ((grouped[:, 1] & 0x03) << 6)
    packed1 = ((grouped[:, 1] >> 2) & 0x0F) | ((grouped[:, 2] & 0x0F) << 4)
    packed2 = ((grouped[:, 2] >> 4) & 0x03) | (grouped[:, 3] << 2)
    return torch.stack((packed0, packed1, packed2), dim=1).to(torch.uint8).flatten()


def unpack_fp6_codes(packed: torch.Tensor) -> torch.Tensor:
    """Unpack LSB-first ONNX FP6 bytes into 6-bit codes."""
    flat = packed.detach().cpu().to(torch.uint8).flatten()
    if flat.numel() % 3 != 0:
        raise ValueError("Packed FP6 byte count must be divisible by 3")
    grouped = flat.reshape(-1, 3).to(torch.int32)
    code0 = grouped[:, 0] & 0x3F
    code1 = ((grouped[:, 0] >> 6) | ((grouped[:, 1] & 0x0F) << 2)) & 0x3F
    code2 = ((grouped[:, 1] >> 4) | ((grouped[:, 2] & 0x03) << 4)) & 0x3F
    code3 = (grouped[:, 2] >> 2) & 0x3F
    return torch.stack((code0, code1, code2, code3), dim=1).to(torch.uint8).flatten()


def quantize_to_mxfp6(weight: torch.Tensor, scale_dtype: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a dense tensor to per-final-axis-block E2M3 codes.

    The E2M3 value table is defined locally from sign, two exponent bits,
    and three mantissa bits, then packed using ONNX's LSB-first FP6 byte
    layout. Runtime parity is gated on ONNX FLOAT6/DQL-28 support and is not
    inferred from this helper alone.
    """
    if not weight.is_floating_point():
        raise TypeError("MXFP6 quantization only supports floating-point weights")
    if weight.shape[-1] % MXFP6_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP6 requires the final weight dimension to be divisible by {MXFP6_BLOCK_SIZE}; got {tuple(weight.shape)}"
        )
    fp32 = weight.detach().cpu().to(torch.float32)
    if not torch.isfinite(fp32).all():
        raise ValueError("MXFP6 quantization does not support NaN or Inf weights")

    block_shape = (*fp32.shape[:-1], fp32.shape[-1] // MXFP6_BLOCK_SIZE, MXFP6_BLOCK_SIZE)
    blocks = fp32.reshape(block_shape)
    amax = blocks.abs().amax(dim=-1)
    scales = torch.where(
        amax == 0,
        torch.ones_like(amax),
        torch.pow(torch.tensor(2.0, dtype=torch.float32), torch.ceil(torch.log2(amax / MXFP6_MAX_FINITE))),
    )
    normalized = blocks / scales.unsqueeze(-1)
    abs_values = normalized.abs().clamp(max=MXFP6_MAX_FINITE).flatten()
    lower_indices = torch.bucketize(abs_values, _POSITIVE_MIDPOINTS, right=False)
    upper_indices = torch.bucketize(abs_values, _POSITIVE_MIDPOINTS, right=True)
    tie = lower_indices != upper_indices
    use_upper = tie & _LOWER_TIE_IS_ODD[lower_indices.clamp(max=_LOWER_TIE_IS_ODD.numel() - 1)]
    indices = torch.where(use_upper, upper_indices, lower_indices)
    quantized = _POSITIVE_CODES[indices]
    quantized = quantized | (normalized.flatten() < 0).to(torch.uint8) * 0x20

    packed_shape = (*fp32.shape[:-1], fp32.shape[-1] * 3 // 4)
    packed = pack_fp6_codes(quantized.flatten()).reshape(packed_shape)
    if scale_dtype == "e8m0":
        scales_out = _encode_e8m0_scales(scales)
    else:
        scales_out = scales.to(_SCALE_TORCH_DTYPES[scale_dtype])
    return packed, scales_out.contiguous()


def _encode_e8m0_scales(scales: torch.Tensor) -> torch.Tensor:
    """Encode positive power-of-two scales as E8M0 exponent bytes."""
    exponents = torch.round(torch.log2(scales.to(torch.float32))).to(torch.int32) + 127
    if torch.any((exponents < 0) | (exponents > 254)):
        raise ValueError("MXFP6 E8M0 scale exponent is out of encodable range")
    return exponents.to(torch.uint8)


def _graph_inputs_by_name(graph) -> Dict[str, onnx.ValueInfoProto]:
    return {value_info.name: value_info for value_info in graph.input}


def _node_consumers(graph) -> Dict[str, List[onnx.NodeProto]]:
    consumers: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    return consumers


def _is_final_axis_transpose(node: onnx.NodeProto) -> bool:
    if node.op_type != "Transpose":
        return False
    perm_attr = next((attr for attr in node.attribute if attr.name == "perm"), None)
    if perm_attr is None:
        return True
    perm = list(perm_attr.ints)
    return len(perm) >= 2 and perm[:-2] == list(range(len(perm) - 2)) and perm[-2:] == [len(perm) - 1, len(perm) - 2]


def _function_input_index(function: onnx.FunctionProto, call_node: onnx.NodeProto, input_name: str) -> Optional[int]:
    input_indices = [idx for idx, name in enumerate(call_node.input) if name == input_name]
    if len(input_indices) != 1:
        return None
    input_index = input_indices[0]
    if input_index >= len(function.input):
        return None
    return input_index


def _mxfp6_flat_insert_before_node(
    graph,
    input_name: str,
) -> Optional[onnx.NodeProto]:
    consumers = _node_consumers(graph)
    direct = consumers.get(input_name, [])
    if len(direct) != 1:
        return None
    node = direct[0]
    if node.op_type == "MatMul" and len(node.input) > 1 and node.input[1] == input_name:
        return node
    if not _is_final_axis_transpose(node) or not node.output:
        return None
    transpose_consumers = consumers.get(node.output[0], [])
    if (
        len(transpose_consumers) == 1
        and transpose_consumers[0].op_type == "MatMul"
        and len(transpose_consumers[0].input) > 1
        and transpose_consumers[0].input[1] == node.output[0]
    ):
        return node
    return None


def _mxfp6_target(
    graph,
    input_name: str,
    function_lookup: Mapping[Tuple[str, str], onnx.FunctionProto],
) -> Optional[_Mxfp6Target]:
    insert_before_node = _mxfp6_flat_insert_before_node(graph, input_name)
    if insert_before_node is not None:
        return _FlatMxfp6Target(insert_before_node=insert_before_node)

    consumers = _node_consumers(graph).get(input_name, [])
    if len(consumers) != 1:
        return None
    call_node = consumers[0]
    function = function_lookup.get((call_node.domain, call_node.op_type))
    if function is None:
        return None
    formal_index = _function_input_index(function, call_node, input_name)
    if formal_index is None:
        return None
    formal_name = function.input[formal_index]
    if _mxfp6_flat_insert_before_node(function, formal_name) is None:
        return None
    return _FunctionMxfp6Target(
        function=function,
        call_node=call_node,
        formal_index=formal_index,
        formal_name=formal_name,
    )


def _is_mxfp6_candidate_topology(
    graph,
    input_name: str,
    function_lookup: Optional[Mapping[Tuple[str, str], onnx.FunctionProto]] = None,
) -> bool:
    consumers = _node_consumers(graph).get(input_name, [])
    if len(consumers) != 1:
        return False
    node = consumers[0]
    if node.op_type in {"MatMul", "Transpose"}:
        return True
    if function_lookup is None:
        return False
    function = function_lookup.get((node.domain, node.op_type))
    if function is None:
        return False
    formal_index = _function_input_index(function, node, input_name)
    if formal_index is None:
        return False
    return _is_mxfp6_candidate_topology(function, function.input[formal_index])


def _is_lm_head_weight_name(name: str) -> bool:
    components = [component for component in name.replace("/", ".").split(".") if component]
    return len(components) >= 2 and components[-2:] == ["lm_head", "weight"]


def _is_lm_head_weight(spec_input) -> bool:
    return _is_lm_head_weight_name(spec_input.name) or _is_lm_head_weight_name(spec_input.location.key)


def _insert_before_node(graph, target_node: onnx.NodeProto, new_node: onnx.NodeProto) -> None:
    for idx, node in enumerate(graph.node):
        if node is target_node:
            graph.node.insert(idx, new_node)
            return
    graph.node.insert(0, new_node)


def _append_value_info(container, value_info: onnx.ValueInfoProto) -> None:
    names = {existing.name for existing in container.value_info}
    if value_info.name not in names:
        container.value_info.append(value_info)


def _remove_graph_input(graph, name: str) -> onnx.ValueInfoProto:
    for idx, value_info in enumerate(graph.input):
        if value_info.name == name:
            removed = graph.input[idx]
            del graph.input[idx]
            return removed
    raise ValueError(f"ONNX graph input '{name}' not found")


def _tensor_value_info(name: str, elem_type: int, shape: Sequence[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, list(shape))


def _set_default_opset(model: onnx.ModelProto, version: int) -> None:
    for opset in model.opset_import:
        if opset.domain == "":
            opset.version = max(opset.version, version)
            return
    model.opset_import.append(helper.make_opsetid("", version))


def _ensure_opset(model: onnx.ModelProto, domain: str, version: int) -> None:
    if any(opset.domain == domain for opset in model.opset_import):
        return
    model.opset_import.append(helper.make_opsetid(domain, version))


def _function_key(function: onnx.FunctionProto) -> Tuple[str, str]:
    return function.domain, function.name


def _group_function_replacements(
    replacements: Sequence[_Mxfp6Replacement],
) -> Dict[Tuple[str, str], Dict[int, List[_Mxfp6Replacement]]]:
    grouped: Dict[Tuple[str, str], Dict[int, List[_Mxfp6Replacement]]] = {}
    for replacement in replacements:
        if not isinstance(replacement.target, _FunctionMxfp6Target):
            continue
        function_group = grouped.setdefault(_function_key(replacement.target.function), {})
        function_group.setdefault(replacement.target.formal_index, []).append(replacement)
    return grouped


def _validate_function_replacements(
    model: onnx.ModelProto,
    replacements: Sequence[_Mxfp6Replacement],
) -> None:
    grouped = _group_function_replacements(replacements)
    replacement_by_name = {replacement.original_name: replacement for replacement in replacements}
    for (domain, name), replacements_by_formal in grouped.items():
        formal_indices = sorted(replacements_by_formal)
        calls = [node for node in model.graph.node if node.domain == domain and node.op_type == name]
        for call_node in calls:
            for formal_index in formal_indices:
                if formal_index >= len(call_node.input):
                    raise NotImplementedError(
                        "mxfp6=True cannot rewrite shared ONNX subfunctions when a call is missing a converted "
                        f"formal input: function '{domain}::{name}', formal index {formal_index}."
                    )
                actual_name = call_node.input[formal_index]
                replacement = replacement_by_name.get(actual_name)
                if replacement is None or not isinstance(replacement.target, _FunctionMxfp6Target):
                    raise NotImplementedError(
                        "mxfp6=True cannot partially rewrite shared ONNX subfunctions. "
                        f"Function '{domain}::{name}' call input '{actual_name}' at formal index {formal_index} "
                        "does not have a matching MXFP6 replacement."
                    )
                if _function_key(replacement.target.function) != (domain, name):
                    raise NotImplementedError(
                        f"mxfp6=True found an inconsistent ONNX subfunction replacement for input '{actual_name}'."
                    )
                if replacement.target.formal_index != formal_index:
                    raise NotImplementedError(
                        "mxfp6=True found an inconsistent formal input mapping for "
                        f"function '{domain}::{name}' input '{actual_name}'."
                    )

        reference_by_formal = {
            formal_index: replacements_for_formal[0]
            for formal_index, replacements_for_formal in replacements_by_formal.items()
        }
        for formal_index, reference in reference_by_formal.items():
            reference_packed_shape = list(reference.packed.shape)
            reference_scale_shape = list(reference.scale.shape)
            reference_logical_shape = list(_load_tensor_shape(reference.value_info))
            for replacement in replacements_by_formal[formal_index][1:]:
                if (
                    list(replacement.packed.shape) != reference_packed_shape
                    or list(replacement.scale.shape) != reference_scale_shape
                    or list(_load_tensor_shape(replacement.value_info)) != reference_logical_shape
                ):
                    raise NotImplementedError(
                        "mxfp6=True cannot rewrite a shared ONNX subfunction when converted actual inputs for "
                        f"formal index {formal_index} have different shapes."
                    )


def _load_tensor_shape(value_info: onnx.ValueInfoProto) -> List[int]:
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]


def _rewrite_function_replacements(
    model: onnx.ModelProto,
    replacements: Sequence[_Mxfp6Replacement],
    scale_dtype: str,
) -> None:
    grouped = _group_function_replacements(replacements)
    if not grouped:
        return
    replacement_by_name = {replacement.original_name: replacement for replacement in replacements}
    function_lookup = {_function_key(function): function for function in model.functions}
    scale_elem_type = _scale_tensorproto_dtype(scale_dtype)

    for function_key, replacements_by_formal in grouped.items():
        function = function_lookup[function_key]
        formal_indices = sorted(replacements_by_formal)
        for formal_index in formal_indices:
            replacement = replacements_by_formal[formal_index][0]
            formal_name = function.input[formal_index]
            if formal_name != replacement.target.formal_name:
                raise NotImplementedError(
                    "mxfp6=True found a changed ONNX FunctionProto signature while rewriting "
                    f"function '{function.domain}::{function.name}'."
                )
            logical_shape = list(replacement.packed.shape)
            logical_shape[-1] = logical_shape[-1] * 4 // 3
            packed_shape = list(replacement.packed.shape)
            scale_shape = list(replacement.scale.shape)
            packed_formal_name = formal_name + MXFP6_PACKED_SUFFIX
            scale_formal_name = formal_name + MXFP6_SCALE_SUFFIX
            unpacked_name = formal_name + ".mxfp6_unpacked"
            insert_before_node = _mxfp6_flat_insert_before_node(function, formal_name)
            if insert_before_node is None:
                raise NotImplementedError(
                    "mxfp6=True currently supports ONNX subfunction weights only as MatMul RHS, "
                    f"optionally through a sole final-axis Transpose. Unsupported formal '{formal_name}'."
                )

            function.input[formal_index] = packed_formal_name
            _append_value_info(function, _tensor_value_info(packed_formal_name, TensorProto.UINT8, packed_shape))
            _append_value_info(function, _tensor_value_info(scale_formal_name, scale_elem_type, scale_shape))
            _append_value_info(function, _tensor_value_info(unpacked_name, TensorProto.FLOAT6E2M3, logical_shape))
            _append_value_info(
                function,
                _tensor_value_info(formal_name, _output_tensorproto_dtype(replacement.value_info), logical_shape),
            )
            unpack_node = helper.make_node(
                "UnpackMxfp6",
                inputs=[packed_formal_name],
                outputs=[unpacked_name],
                name=formal_name + "_mxfp6_unpack",
                domain="com.qti.aisw.onnx",
            )
            dq_node = helper.make_node(
                "DequantizeLinear",
                inputs=[unpacked_name, scale_formal_name],
                outputs=[formal_name],
                name=formal_name + "_mxfp6_dq",
                axis=-1,
                block_size=MXFP6_BLOCK_SIZE,
                **(
                    {"output_dtype": _output_tensorproto_dtype(replacement.value_info)}
                    if _dql_schema_supports_output_dtype()
                    else {}
                ),
            )
            _insert_before_node(function, insert_before_node, unpack_node)
            _insert_before_node(function, insert_before_node, dq_node)

        for formal_index in formal_indices:
            function.input.append(function.input[formal_index].removesuffix(MXFP6_PACKED_SUFFIX) + MXFP6_SCALE_SUFFIX)

        for call_node in [
            node for node in model.graph.node if node.domain == function.domain and node.op_type == function.name
        ]:
            for formal_index in formal_indices:
                actual_name = call_node.input[formal_index]
                replacement = replacement_by_name[actual_name]
                call_node.input[formal_index] = replacement.original_name + MXFP6_PACKED_SUFFIX
            for formal_index in formal_indices:
                packed_actual_name = call_node.input[formal_index]
                call_node.input.append(packed_actual_name.removesuffix(MXFP6_PACKED_SUFFIX) + MXFP6_SCALE_SUFFIX)

        _set_default_opset(function, MXFP6_ONNX_OPSET)
        _ensure_opset(function, "com.qti.aisw.onnx", 1)


def _checkpoint_file_for_key(prepared_dir: Path, weight_map: Dict[str, str], key: str) -> Path:
    shard_name = weight_map.get(key)
    if shard_name is None:
        raise ValueError(f"Could not find checkpoint key '{key}' in prepared checkpoint index")
    return prepared_dir / shard_name


def _load_checkpoint_tensor(prepared_dir: Path, weight_map: Dict[str, str], key: str) -> torch.Tensor:
    with safe_open(str(_checkpoint_file_for_key(prepared_dir, weight_map, key)), framework="pt") as handle:
        return handle.get_tensor(key)


def _write_mxfp6_tensors(
    prepared_dir: Path,
    tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    weight_map: Dict[str, str],
) -> Dict[str, str]:
    by_shard: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, (packed, scale) in tensors.items():
        shard_name = weight_map[key]
        by_shard.setdefault(shard_name, {})[key + MXFP6_PACKED_SUFFIX] = packed
        by_shard[shard_name][key + MXFP6_SCALE_SUFFIX] = scale

    for shard_name, replacements in by_shard.items():
        existing = {}
        shard_path = prepared_dir / shard_name
        with safe_open(str(shard_path), framework="pt") as handle:
            for key in handle.keys():
                existing[key] = handle.get_tensor(key)
        for key, tensor in replacements.items():
            if key not in existing:
                existing[key] = tensor
        atomic_save(existing, shard_path)

    new_weight_map = dict(weight_map)
    for key in tensors:
        new_weight_map[key + MXFP6_PACKED_SUFFIX] = new_weight_map[key]
        new_weight_map[key + MXFP6_SCALE_SUFFIX] = new_weight_map[key]
    write_index(prepared_dir, new_weight_map)
    return new_weight_map


def finalize_mxfp6_export(
    onnx_path: Path, weight_spec_path: Path, prepared_model_ref: str, config: Optional[Mxfp6Config]
) -> None:
    """Rewrite saved weight-free ONNX and checkpoint files for QEff-owned MXFP6."""
    if config is None or not config.enabled:
        return
    from QEfficient.exporter.weight_free.weight_spec import (
        WeightSpecInput,
        WeightSpecLocation,
        load_weight_spec,
        save_weight_spec,
    )

    validate_mxfp6_capabilities(config)
    prepared_dir = Path(prepared_model_ref)
    sentinel = prepared_dir / MXFP6_PREPARED_SENTINEL
    spec = load_weight_spec(weight_spec_path)
    model = onnx.load(str(onnx_path), load_external_data=False)
    inputs_by_name = _graph_inputs_by_name(model.graph)
    function_lookup = {(function.domain, function.name): function for function in model.functions}
    weight_map = read_weight_map(prepared_dir)

    replacements = {}
    mxfp6_replacements = []
    updated_inputs = []
    for spec_input in spec.inputs:
        original_name = spec_input.name
        logical_key = spec_input.location.key
        value_info = inputs_by_name.get(original_name)
        if value_info is None:
            updated_inputs.append(spec_input)
            continue
        target = _mxfp6_target(model.graph, original_name, function_lookup)
        if target is None:
            if _is_mxfp6_candidate_topology(model.graph, original_name, function_lookup):
                raise NotImplementedError(
                    "mxfp6=True currently supports only dense weights consumed as MatMul RHS, "
                    f"optionally through a sole final-axis Transpose. Unsupported topology for ONNX input "
                    f"'{original_name}'."
                )
            updated_inputs.append(spec_input)
            continue
        if _is_lm_head_weight(spec_input):
            updated_inputs.append(spec_input)
            continue
        tensor = _load_checkpoint_tensor(prepared_dir, weight_map, logical_key)
        if tensor.ndim < 2:
            updated_inputs.append(spec_input)
            continue
        packed, scale = quantize_to_mxfp6(tensor, config.scale_dtype)
        replacements[logical_key] = (packed, scale)
        mxfp6_replacements.append(
            _Mxfp6Replacement(
                original_name=original_name,
                logical_key=logical_key,
                value_info=value_info,
                target=target,
                packed=packed,
                scale=scale,
            )
        )

        packed_input_name = original_name + MXFP6_PACKED_SUFFIX
        scale_input_name = original_name + MXFP6_SCALE_SUFFIX
        packed_key = logical_key + MXFP6_PACKED_SUFFIX
        scale_key = logical_key + MXFP6_SCALE_SUFFIX
        logical_shape = list(tensor.shape)
        packed_shape = list(packed.shape)
        scale_shape = list(scale.shape)
        updated_inputs.append(
            WeightSpecInput(
                name=packed_input_name,
                location=WeightSpecLocation(file=spec_input.location.file, key=packed_key),
                role="mxfp6_weight",
                metadata={
                    "source_dtype": _value_info_dtype_name(value_info),
                    "logical_dtype": "float6e2m3",
                    "storage_dtype": "uint8",
                    "packing": "onnx_lsb_first_6bit",
                    "logical_shape": logical_shape,
                    "packed_shape": packed_shape,
                    "block_size": MXFP6_BLOCK_SIZE,
                    "axis": -1,
                    "scale_input": scale_input_name,
                    "unpack_output": original_name + ".mxfp6_unpacked",
                },
            )
        )
        updated_inputs.append(
            WeightSpecInput(
                name=scale_input_name,
                location=WeightSpecLocation(file=spec_input.location.file, key=scale_key),
                role="mxfp6_scale",
                weight_input=packed_input_name,
                metadata={
                    "logical_dtype": config.scale_dtype,
                    "storage_dtype": "uint8" if config.scale_dtype == "e8m0" else config.scale_dtype,
                    "logical_shape": scale_shape,
                    "packed_shape": scale_shape,
                    "block_size": MXFP6_BLOCK_SIZE,
                    "axis": -1,
                },
            )
        )

    if not replacements:
        raise NotImplementedError(
            "mxfp6=True currently supports only dense weights consumed as MatMul RHS, "
            "optionally through a sole final-axis Transpose."
        )

    _validate_function_replacements(model, mxfp6_replacements)
    scale_elem_type = _scale_tensorproto_dtype(config.scale_dtype)
    for replacement in mxfp6_replacements:
        original_name = replacement.original_name
        packed_input_name = original_name + MXFP6_PACKED_SUFFIX
        scale_input_name = original_name + MXFP6_SCALE_SUFFIX
        unpacked_name = original_name + ".mxfp6_unpacked"
        logical_shape = list(replacement.packed.shape)
        logical_shape[-1] = logical_shape[-1] * 4 // 3
        packed_shape = list(replacement.packed.shape)
        scale_shape = list(replacement.scale.shape)
        _remove_graph_input(model.graph, original_name)
        model.graph.input.append(_tensor_value_info(packed_input_name, TensorProto.UINT8, packed_shape))
        model.graph.input.append(_tensor_value_info(scale_input_name, scale_elem_type, scale_shape))
        if isinstance(replacement.target, _FlatMxfp6Target):
            _append_value_info(model.graph, _tensor_value_info(unpacked_name, TensorProto.FLOAT6E2M3, logical_shape))
            unpack_node = helper.make_node(
                "UnpackMxfp6",
                inputs=[packed_input_name],
                outputs=[unpacked_name],
                name=original_name + "_mxfp6_unpack",
                domain="com.qti.aisw.onnx",
            )
            dq_node = helper.make_node(
                "DequantizeLinear",
                inputs=[unpacked_name, scale_input_name],
                outputs=[original_name],
                name=original_name + "_mxfp6_dq",
                axis=-1,
                block_size=MXFP6_BLOCK_SIZE,
                **(
                    {"output_dtype": _output_tensorproto_dtype(replacement.value_info)}
                    if _dql_schema_supports_output_dtype()
                    else {}
                ),
            )
            _insert_before_node(model.graph, replacement.target.insert_before_node, unpack_node)
            _insert_before_node(model.graph, replacement.target.insert_before_node, dq_node)

    _rewrite_function_replacements(model, mxfp6_replacements, config.scale_dtype)
    _write_mxfp6_tensors(prepared_dir, replacements, weight_map)
    spec.inputs = updated_inputs
    spec.version = 7
    _set_default_opset(model, 28)
    _ensure_opset(model, "com.qti.aisw.onnx", 1)
    onnx.save(model, str(onnx_path))
    save_weight_spec(weight_spec_path, spec)
    sentinel.write_text(
        json.dumps(
            {"quantizer_version": MXFP6_QUANTIZER_VERSION, "scale_dtype": config.scale_dtype},
            indent=2,
            sort_keys=True,
        )
    )


def _scale_tensorproto_dtype(scale_dtype: str) -> int:
    if scale_dtype == "float16":
        return TensorProto.FLOAT16
    if scale_dtype == "float32":
        return TensorProto.FLOAT
    if scale_dtype == "bfloat16":
        return TensorProto.BFLOAT16
    fp8 = _tensorproto_enum("FLOAT8E8M0")
    if fp8 is None:
        raise AssertionError("onnx.TensorProto.FLOAT8E8M0 is required for mxfp6_scale_dtype='e8m0'")
    return fp8


def _output_tensorproto_dtype(value_info: onnx.ValueInfoProto) -> int:
    elem_type = value_info.type.tensor_type.elem_type
    if elem_type:
        return elem_type
    return TensorProto.FLOAT16


def _value_info_dtype_name(value_info: onnx.ValueInfoProto) -> str:
    elem_type = value_info.type.tensor_type.elem_type
    return TensorProto.DataType.Name(elem_type) if elem_type else "UNKNOWN"
