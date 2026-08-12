# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import onnxscript
import torch
from onnx import ModelProto, TensorProto, helper
from onnxscript.onnx_types import UINT4

from QEfficient.customop.onnxscript_utils import qeff_custom_op
from QEfficient.customop.utils import select_interface
from QEfficient.utils import constants

ops = getattr(onnxscript, "opset" + str(constants.ONNX_LEGACY_EXPORT_OPSET))


@qeff_custom_op("com.qti.aisw.onnx", 1)
def CastToUInt4(weight_packed: onnxscript.UINT8) -> UINT4:
    """
    Unpack packed uint8 weights into uint4 values and cast output to UINT4.
    Supports N-D input: all leading dimensions are preserved; only the last
    dimension (in_features // 2) is doubled to (in_features).

    Input:  (..., in_features // 2) UINT8
            Each byte holds two nibbles: byte = (w_y << 4) | (w_x & 0x0F)
    Output: (..., in_features) UINT4, values in [0, 15]

    Operations:
      w_x          = weight_packed % 16          (lower nibble)
      w_y          = (weight_packed >> 4) % 16   (upper nibble)
      stacked      = concat([w_x, w_y], axis=-1) after unsqueeze
                     → (..., in//2, 2)
      leading_dims = shape[:-1]
      new_shape    = [...leading_dims, last_dim * 2]
      reshaped     = reshape(stacked, new_shape)
      output       = Cast(reshaped, to=UINT4)
    """
    sixteen = ops.CastLike(ops.Constant(value_ints=[16]), weight_packed)

    # Lower nibble: weight_packed & 0x0F  =  weight_packed % 16
    w_x = ops.Mod(weight_packed, sixteen)

    # Upper nibble: (weight_packed >> 4) & 0x0F
    shift = ops.CastLike(ops.Constant(value_ints=[4]), weight_packed)
    w_shifted = ops.BitShift(weight_packed, shift, direction="RIGHT")
    w_y = ops.Mod(w_shifted, sixteen)

    # Stack along a new last dim → (..., in_features//2, 2)
    w_x_unsq = ops.Unsqueeze(w_x, [-1])
    w_y_unsq = ops.Unsqueeze(w_y, [-1])
    stacked = ops.Concat(w_x_unsq, w_y_unsq, axis=-1)

    # N-D aware reshape: preserve all leading dims, double the last dim.
    # packed_shape = [d0, d1, ..., last_dim]
    packed_shape = ops.Shape(weight_packed)
    # All dims except the last: [d0, d1, ...]
    leading_dims = ops.Slice(packed_shape, starts=[0], ends=[-1], axes=[0])
    # Last dim only: [last_dim]
    last_dim = ops.Slice(packed_shape, starts=[-1], ends=[2147483647], axes=[0])
    # Double the last dim: [last_dim * 2]
    last_dim_doubled = ops.Mul(last_dim, ops.Constant(value_ints=[2]))
    # New shape: [d0, d1, ..., last_dim * 2]
    new_shape = ops.Concat(leading_dims, last_dim_doubled, axis=0)
    reshaped = ops.Reshape(stacked, new_shape)

    # Cast to UINT4 — data_type value is version-dependent (21 in ONNX 1.18, 23 in newer)
    return ops.Cast(reshaped, to=int(TensorProto.UINT4))


class CastToUInt4Func(torch.autograd.Function):
    """
    Custom op: unpacks packed uint8 → uint8 (values 0-15) in PyTorch.
    In ONNX the custom op subgraph includes a Cast → UINT4 as its last step.
    Supports N-D input: all leading dimensions are preserved.

    PyTorch forward  : packed uint8 (..., in//2) → uint8 (..., in), values [0, 15]
    ONNX symbolic    : emits CastToUInt4 node (com.qti.aisw.onnx)
                       The subgraph ends with Cast → UINT4.
    """

    @staticmethod
    def forward(weight_packed: torch.Tensor) -> torch.Tensor:
        w_x = weight_packed & 0x0F  # lower nibble, (..., in//2), range [0, 15]
        w_y = (weight_packed >> 4) & 0x0F  # upper nibble, (..., in//2), range [0, 15]
        # New shape: all leading dims unchanged, last dim doubled
        new_shape = list(weight_packed.shape[:-1]) + [weight_packed.shape[-1] * 2]
        return torch.stack(
            [w_x, w_y], dim=-1
        ).reshape(
            new_shape
        )  # Can't add a cast operation to uint4 here, as its not supported in pytorch; The ONNX export will handle the cast to IINT4 in the symbolic method.

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, weight_packed: torch.Value) -> torch.Value:
        output = g.onnxscript_op(CastToUInt4, weight_packed)
        return output


def cast_to_uint4(weight_packed: torch.Tensor) -> torch.Tensor:
    return select_interface(CastToUInt4Func.apply, torch.ops.qefficient.cast_to_uint4)(weight_packed)


def _find_value_info(model: ModelProto, value_name: str):
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if value_info.name == value_name:
            return value_info
    return None


def _tensor_shape_from_value_info(value_info):
    if value_info is None or not value_info.type.HasField("tensor_type"):
        return None

    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None

    shape = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return shape


def _cast_to_uint4_output_shape(model: ModelProto, input_name: str):
    shape = _tensor_shape_from_value_info(_find_value_info(model, input_name))
    if not shape:
        return None
    output_shape = list(shape)
    if isinstance(output_shape[-1], int):
        output_shape[-1] *= 2
    elif isinstance(output_shape[-1], str):
        output_shape[-1] = f"{output_shape[-1]}_x2"
    return output_shape


def _ensure_cast_to_uint4_value_info(model: ModelProto, node) -> bool:
    transformed = False
    graph_input_names = {value.name for value in model.graph.input}
    graph_output_names = {value.name for value in model.graph.output}
    typed_names = {value.name for value in list(model.graph.value_info) + list(model.graph.output)}

    for output_name in node.output:
        if output_name in graph_input_names:
            continue
        if output_name not in typed_names and output_name not in graph_output_names:
            output_shape = _cast_to_uint4_output_shape(model, node.input[0] if node.input else "")
            model.graph.value_info.append(helper.make_tensor_value_info(output_name, TensorProto.UINT4, output_shape))
            typed_names.add(output_name)
            transformed = True
    return transformed


def _unique_value_name(base_name: str, used_names: set[str]) -> str:
    suffix_index = 0
    new_name = base_name
    while new_name in used_names:
        suffix_index += 1
        new_name = f"{base_name}_{suffix_index}"
    used_names.add(new_name)
    return new_name


def _used_value_names(model: ModelProto) -> set[str]:
    used_names = {value.name for value in model.graph.input}
    used_names.update(value.name for value in model.graph.output)
    used_names.update(value.name for value in model.graph.value_info)
    for node in model.graph.node:
        used_names.update(name for name in node.input if name)
        used_names.update(name for name in node.output if name)
    return used_names


def _add_uint8_value_info_like(model: ModelProto, value_name: str, source_name: str) -> bool:
    if _find_value_info(model, value_name) is not None:
        return False
    source_value_info = _find_value_info(model, source_name)
    source_shape = _tensor_shape_from_value_info(source_value_info)
    model.graph.value_info.append(helper.make_tensor_value_info(value_name, TensorProto.UINT8, source_shape))
    return True


def _isolate_cast_to_uint4_graph_inputs(model: ModelProto) -> bool:
    graph_input_names = {value.name for value in model.graph.input}
    if not graph_input_names:
        return False

    used_names = _used_value_names(model)
    isolated_inputs = {}
    transformed = False
    rewritten_nodes = []
    for node in model.graph.node:
        if node.domain != "com.qti.aisw.onnx" or node.op_type != "CastToUInt4" or not node.input:
            rewritten_nodes.append(node)
            continue

        input_name = node.input[0]
        if input_name not in graph_input_names:
            rewritten_nodes.append(node)
            continue

        isolated_name = isolated_inputs.get(input_name)
        if isolated_name is None:
            isolated_name = _unique_value_name(f"{input_name}_packed_uint8", used_names)
            isolated_inputs[input_name] = isolated_name
            rewritten_nodes.append(
                helper.make_node("Identity", [input_name], [isolated_name], name=f"{isolated_name}_identity")
            )
            transformed = _add_uint8_value_info_like(model, isolated_name, input_name) or True

        node.input[0] = isolated_name
        rewritten_nodes.append(node)
        transformed = True

    if transformed:
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)
    return transformed


def _is_kimi_packed_int4_input(name: str) -> bool:
    return ".mlp.all_" in name and name.endswith(("_qweight", "_qzeros"))


def _const_int64_node(name: str, values: list[int]):
    return helper.make_node(
        "Constant", [], [name], value=helper.make_tensor(f"{name}_value", TensorProto.INT64, [len(values)], values)
    )


def _const_int32_node(name: str, values: list[int]):
    return helper.make_node(
        "Constant", [], [name], value=helper.make_tensor(f"{name}_value", TensorProto.INT32, [len(values)], values)
    )


def _make_int32_unpack_nodes(input_name: str, output_name: str, node_index: int, used_names: set[str]):
    prefix = f"{output_name}_int32_unpack_{node_index}"
    input_int32 = _unique_value_name(f"{prefix}_input_int32", used_names)
    sixteen = _unique_value_name(f"{prefix}_sixteen", used_names)
    lower = _unique_value_name(f"{prefix}_lower", used_names)
    upper = _unique_value_name(f"{prefix}_upper", used_names)
    axes_lower = _unique_value_name(f"{prefix}_axes_lower", used_names)
    axes_upper = _unique_value_name(f"{prefix}_axes_upper", used_names)
    lower_unsqueezed = _unique_value_name(f"{prefix}_lower_unsqueezed", used_names)
    upper_unsqueezed = _unique_value_name(f"{prefix}_upper_unsqueezed", used_names)
    stacked = _unique_value_name(f"{prefix}_stacked", used_names)
    packed_shape = _unique_value_name(f"{prefix}_packed_shape", used_names)
    starts_zero = _unique_value_name(f"{prefix}_starts_zero", used_names)
    ends_before_last = _unique_value_name(f"{prefix}_ends_before_last", used_names)
    axis_zero = _unique_value_name(f"{prefix}_axis_zero", used_names)
    leading_dims = _unique_value_name(f"{prefix}_leading_dims", used_names)
    starts_last = _unique_value_name(f"{prefix}_starts_last", used_names)
    ends_max = _unique_value_name(f"{prefix}_ends_max", used_names)
    axis_zero_last = _unique_value_name(f"{prefix}_axis_zero_last", used_names)
    last_dim = _unique_value_name(f"{prefix}_last_dim", used_names)
    two = _unique_value_name(f"{prefix}_two", used_names)
    last_dim_doubled = _unique_value_name(f"{prefix}_last_dim_doubled", used_names)
    new_shape = _unique_value_name(f"{prefix}_new_shape", used_names)
    return [
        helper.make_node("Cast", [input_name], [input_int32], to=TensorProto.INT32),
        _const_int32_node(sixteen, [16]),
        helper.make_node("Mod", [input_int32, sixteen], [lower]),
        helper.make_node("Div", [input_int32, sixteen], [upper]),
        _const_int64_node(axes_lower, [-1]),
        helper.make_node("Unsqueeze", [lower, axes_lower], [lower_unsqueezed]),
        _const_int64_node(axes_upper, [-1]),
        helper.make_node("Unsqueeze", [upper, axes_upper], [upper_unsqueezed]),
        helper.make_node("Concat", [lower_unsqueezed, upper_unsqueezed], [stacked], axis=-1),
        helper.make_node("Shape", [input_name], [packed_shape]),
        _const_int64_node(starts_zero, [0]),
        _const_int64_node(ends_before_last, [-1]),
        _const_int64_node(axis_zero, [0]),
        helper.make_node("Slice", [packed_shape, starts_zero, ends_before_last, axis_zero], [leading_dims]),
        _const_int64_node(starts_last, [-1]),
        _const_int64_node(ends_max, [2147483647]),
        _const_int64_node(axis_zero_last, [0]),
        helper.make_node("Slice", [packed_shape, starts_last, ends_max, axis_zero_last], [last_dim]),
        _const_int64_node(two, [2]),
        helper.make_node("Mul", [last_dim, two], [last_dim_doubled]),
        helper.make_node("Concat", [leading_dims, last_dim_doubled], [new_shape], axis=0),
        helper.make_node("Reshape", [stacked, new_shape], [output_name]),
    ]


def _expanded_blockwise_nodes(
    value_name: str,
    value_shape: list[int | str | None],
    target_shape: list[int | str | None],
    axis: int,
    block_size: int,
    prefix: str,
    used_names: set[str],
):
    axes = _unique_value_name(f"{prefix}_axes", used_names)
    unsqueezed = _unique_value_name(f"{prefix}_unsqueezed", used_names)
    expanded_shape_name = _unique_value_name(f"{prefix}_expanded_shape", used_names)
    expanded = _unique_value_name(f"{prefix}_expanded", used_names)
    final_shape_name = _unique_value_name(f"{prefix}_final_shape", used_names)
    reshaped = _unique_value_name(f"{prefix}_reshaped", used_names)
    expanded_shape = list(value_shape)
    expanded_shape.insert(axis + 1, block_size)
    if not all(isinstance(dim, int) for dim in expanded_shape + target_shape):
        return None, None
    nodes = [
        _const_int64_node(axes, [axis + 1]),
        helper.make_node("Unsqueeze", [value_name, axes], [unsqueezed]),
        _const_int64_node(expanded_shape_name, expanded_shape),
        helper.make_node("Expand", [unsqueezed, expanded_shape_name], [expanded]),
        _const_int64_node(final_shape_name, target_shape),
        helper.make_node("Reshape", [expanded, final_shape_name], [reshaped]),
    ]
    return nodes, reshaped


def _rewrite_kimi_int4_extdata_to_int32_dequant(model: ModelProto) -> bool:
    graph_inputs = {value.name: value for value in model.graph.input}
    kimi_inputs = {name for name in graph_inputs if _is_kimi_packed_int4_input(name)}
    if not kimi_inputs:
        return False

    transformed = False
    for name in kimi_inputs:
        tensor_type = graph_inputs[name].type.tensor_type
        if tensor_type.elem_type != TensorProto.UINT8:
            tensor_type.elem_type = TensorProto.UINT8
            transformed = True

    used_names = _used_value_names(model)
    cast_outputs = {}
    rewritten_nodes = []
    for node_index, node in enumerate(model.graph.node):
        if (
            node.domain == "com.qti.aisw.onnx"
            and node.op_type == "CastToUInt4"
            and node.input
            and node.input[0] in kimi_inputs
        ):
            rewritten_nodes.extend(_make_int32_unpack_nodes(node.input[0], node.output[0], node_index, used_names))
            cast_outputs[node.output[0]] = node.input[0]
            transformed = True
            continue
        rewritten_nodes.append(node)

    if not transformed:
        return False

    value_shapes = {
        value.name: _tensor_shape_from_value_info(value)
        for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    }
    for output_name, input_name in cast_outputs.items():
        value_shapes[output_name] = _cast_to_uint4_output_shape(model, input_name)

    final_nodes = []
    replaced_dequant_outputs = set()
    for node_index, node in enumerate(rewritten_nodes):
        if node.op_type != "DequantizeLinear" or len(node.input) < 3 or node.input[0] not in cast_outputs:
            final_nodes.append(node)
            continue

        x_name, scale_name, zero_name = node.input[:3]
        x_shape = value_shapes.get(x_name)
        scale_shape = value_shapes.get(scale_name)
        zero_shape = value_shapes.get(zero_name)
        if x_shape is None or scale_shape is None or zero_shape is None:
            final_nodes.append(node)
            continue

        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        axis = int(attrs.get("axis", 1))
        block_size = int(attrs.get("block_size", 1))
        prefix = f"{node.output[0]}_int32_dequant_{node_index}"
        scale_nodes, expanded_scale = _expanded_blockwise_nodes(
            scale_name, scale_shape, x_shape, axis, block_size, f"{prefix}_scale", used_names
        )
        zero_nodes, expanded_zero = _expanded_blockwise_nodes(
            zero_name, zero_shape, x_shape, axis, block_size, f"{prefix}_zero", used_names
        )
        if scale_nodes is None or zero_nodes is None:
            final_nodes.append(node)
            continue

        x_float = _unique_value_name(f"{prefix}_x_float", used_names)
        zero_float = _unique_value_name(f"{prefix}_zero_float", used_names)
        centered = _unique_value_name(f"{prefix}_centered", used_names)
        final_nodes.extend(scale_nodes)
        final_nodes.extend(zero_nodes)
        final_nodes.extend(
            [
                helper.make_node("Cast", [x_name], [x_float], to=TensorProto.FLOAT16),
                helper.make_node("Cast", [expanded_zero], [zero_float], to=TensorProto.FLOAT16),
                helper.make_node("Sub", [x_float, zero_float], [centered]),
                helper.make_node("Mul", [centered, expanded_scale], list(node.output)),
            ]
        )
        replaced_dequant_outputs.update(node.output)
        transformed = True

    del model.graph.node[:]
    model.graph.node.extend(final_nodes)

    for value in list(model.graph.value_info) + list(model.graph.output):
        if value.name in cast_outputs:
            value.type.tensor_type.elem_type = TensorProto.INT32
        elif value.name in replaced_dequant_outputs:
            value.type.tensor_type.elem_type = TensorProto.FLOAT16

    if not any(node.domain == "com.qti.aisw.onnx" and node.op_type == "CastToUInt4" for node in model.graph.node):
        kept_functions = [
            fn for fn in model.functions if not (fn.domain == "com.qti.aisw.onnx" and fn.name == "CastToUInt4")
        ]
        del model.functions[:]
        model.functions.extend(kept_functions)

    return transformed


def _rename_cast_to_uint4_outputs_shadowing_inputs(model: ModelProto) -> bool:
    graph_input_names = {value.name for value in model.graph.input}
    used_names = set(graph_input_names)
    used_names.update(value.name for value in model.graph.output)
    used_names.update(value.name for value in model.graph.value_info)
    for node in model.graph.node:
        used_names.update(name for name in node.input if name)
        used_names.update(name for name in node.output if name)

    pending_renames = {}
    transformed = False
    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name in pending_renames:
                node.input[index] = pending_renames[input_name]
                transformed = True

        if node.domain != "com.qti.aisw.onnx" or node.op_type != "CastToUInt4":
            continue

        for index, output_name in enumerate(node.output):
            if output_name not in graph_input_names:
                continue
            suffix_index = 0
            new_output_name = f"{output_name}_uint4"
            while new_output_name in used_names:
                suffix_index += 1
                new_output_name = f"{output_name}_uint4_{suffix_index}"
            node.output[index] = new_output_name
            used_names.add(new_output_name)
            pending_renames[output_name] = new_output_name
            transformed = True
    return transformed


def update_cast_to_uint4_output_types(model: ModelProto) -> bool:
    """Correct exported CastToUInt4 value-info to logical UINT4.

    PyTorch fake tensors cannot carry UINT4 dtype, so Dynamo annotates the
    custom-op node outputs as UINT8 even though the ONNXScript function returns
    UINT4. Keep external packed weight inputs as UINT8 and tag only the unpacked
    CastToUInt4 outputs as UINT4 so QAIC extdata dtype checks match the
    safetensors checkpoint storage while downstream dequantization still sees
    logical uint4 data.
    """
    kimi_rewrite_applied = _rewrite_kimi_int4_extdata_to_int32_dequant(model)
    transformed = kimi_rewrite_applied
    if not kimi_rewrite_applied:
        transformed = _isolate_cast_to_uint4_graph_inputs(model)
        transformed = _rename_cast_to_uint4_outputs_shadowing_inputs(model) or transformed

    cast_outputs = {
        output_name
        for node in model.graph.node
        if node.domain == "com.qti.aisw.onnx" and node.op_type == "CastToUInt4"
        for output_name in node.output
    }
    if not cast_outputs:
        return transformed

    for node in model.graph.node:
        if node.domain == "com.qti.aisw.onnx" and node.op_type == "CastToUInt4":
            transformed = _ensure_cast_to_uint4_value_info(model, node) or transformed

    for value in list(model.graph.value_info) + list(model.graph.output):
        if value.name not in cast_outputs:
            continue
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type != TensorProto.UINT4:
            tensor_type.elem_type = TensorProto.UINT4
            transformed = True
    return transformed


class DequantizeLinearFunc(torch.autograd.Function):
    """
    Emits a standard ONNX DequantizeLinear node (ai.onnx domain, not custom).

    Symmetric blockwise quantization — no zero_point:
      output = x * scale   (per block along the last axis)

    Supports N-D input:
      weight_unpacked : (..., in_features)   — quantized values
      scale           : (..., num_blocks)    — per-block scales
      block_size      : int                  — elements per block

    PyTorch forward  : expand blockwise scale along last dim, multiply
    ONNX symbolic    : DequantizeLinear(weight_unpacked, scale,
                                        axis=2, block_size=block_size)
                       axis=2 for 3D input (2, out_features, in_features).
                       No zero_point input (symmetric).
    """

    @staticmethod
    def forward(
        weight_unpacked: torch.Tensor, scale: torch.Tensor, zeros: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        # Expand per-block scale → per-element scale along last dim
        scale_expanded = scale.repeat_interleave(block_size, dim=-1)
        zeros_expanded = zeros.repeat_interleave(block_size, dim=-1)
        return (weight_unpacked.to(torch.int8) - zeros_expanded.to(torch.int8)) * scale_expanded

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph, weight_unpacked: torch.Value, scale: torch.Value, zeros: torch.Value, block_size: int
    ) -> torch.Value:
        # Standard DequantizeLinear: symmetric (no zero_point), blockwise.
        # Input is 3D: (2, out_features, in_features) → axis=2 (last dim).
        # DequantizeLinear natively supports batch dimensions.
        return g.op(
            "DequantizeLinear",
            weight_unpacked,
            scale,
            zeros,
            axis_i=2,
            block_size_i=block_size,
        )


def dequantize_linear(
    weight_unpacked: torch.Tensor, scale: torch.Tensor, zeros: torch.Tensor, block_size: int
) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        return torch.onnx.ops.symbolic(
            "::DequantizeLinear",
            (weight_unpacked, scale, zeros),
            {"axis": 2, "block_size": block_size},
            dtype=scale.dtype,
            shape=weight_unpacked.shape,
            version=18,
        )
    return DequantizeLinearFunc.apply(weight_unpacked, scale, zeros, block_size)
