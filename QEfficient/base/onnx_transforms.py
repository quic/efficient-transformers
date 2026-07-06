# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import numpy as np
import onnx
import torch
from onnx import ModelProto, TensorProto, external_data_helper, numpy_helper

from QEfficient.customop.ctx_scatter_gather import (
    CtxGather,
    CtxGather3D,
    CtxGatherBlockedKV,
    CtxGatherFunc,
    CtxGatherFunc3D,
    CtxGatherFunc3DGeneralized,
    CtxGatherFuncBlockedKV,
    CtxScatter,
    CtxScatter3D,
    CtxScatter3DInt,
    CtxScatterFunc,
    CtxScatterFunc3D,
    CtxScatterFunc3DGeneralized,
    CtxScatterFunc3DInt,
)
from QEfficient.customop.ctx_scatter_gather_cb import (
    CtxGatherBlockedKVCB,
    CtxGatherCB,
    CtxGatherCB3D,
    CtxGatherFuncBlockedKVCB,
    CtxGatherFuncCB,
    CtxGatherFuncCB3D,
    CtxScatterCB,
    CtxScatterCB3D,
    CtxScatterFuncCB,
    CtxScatterFuncCB3D,
)

# from QEfficient.customop.quantization_ops import CastToUInt4, CastToUInt4Func
from QEfficient.customop.rms_norm import CustomRMSNorm, CustomRMSNormFunc
from QEfficient.utils.constants import FILE_CHUNK_SIZE_DEFAULT, ONNX_EXPORT_OPSET, SIZE_THRESHOLD_DEFAULT

logger = logging.getLogger(__name__)


class BaseOnnxTransform:
    """Base class for ONNX graph modifications. Should NOT be instantiated."""

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Use the `apply` method directly.")

    @classmethod
    def apply(cls, model: ModelProto, **kwargs) -> Tuple[ModelProto, bool]:
        raise NotImplementedError("Use subclasses for ONNX transform")


class FP16ClipTransform(BaseOnnxTransform):
    """Clip FP32 tensors to FP16 range to avoid overflow during conversion."""

    @classmethod
    def apply(cls, tensor: TensorProto, onnx_base_dir: str, fp16_max: float, fp16_min: float) -> bool:
        nptensor = numpy_helper.to_array(tensor, onnx_base_dir)
        if nptensor.dtype == np.float32 and (np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)):
            neg_inf_mask = np.isinf(nptensor) & (nptensor < 0)
            clipped_tensor = np.clip(nptensor, fp16_min, fp16_max)

            if neg_inf_mask.any():
                clipped_tensor = np.where(neg_inf_mask, np.float32("-inf"), clipped_tensor)

            tensor.CopyFrom(numpy_helper.from_array(clipped_tensor, tensor.name))
            return True
        return False


class SplitTensorsTransform(BaseOnnxTransform):
    """Split large tensors into external data files for efficient storage."""

    @classmethod
    def apply(
        cls, tensor: TensorProto, model_name: str, file_num: int, mapping: Dict[str, Tuple[TensorProto, str]]
    ) -> None:
        file_name = f"{model_name}_{file_num}.onnx.data"
        mapping[tensor.name] = (tensor, file_name)


class CustomOpTransform(BaseOnnxTransform):
    """Register custom ONNX ops and append their function prototypes to the model."""

    _custom_ops: Dict[str, Tuple[Any, Any]] = {
        "CustomRMSNormFunc": (CustomRMSNormFunc, CustomRMSNorm),
        "CtxScatterFunc": (CtxScatterFunc, CtxScatter),
        "CtxScatterFunc3D": (CtxScatterFunc3D, CtxScatter3D),
        "CtxScatterFunc3DInt": (CtxScatterFunc3DInt, CtxScatter3DInt),
        "CtxScatterFunc3DGeneralized": (CtxScatterFunc3DGeneralized, CtxScatter3D),
        "CtxGatherFunc": (CtxGatherFunc, CtxGather),
        "CtxGatherFunc3D": (CtxGatherFunc3D, CtxGather3D),
        "CtxGatherFunc3DGeneralized": (CtxGatherFunc3DGeneralized, CtxGather3D),
        "CtxScatterFuncCB3D": (CtxScatterFuncCB3D, CtxScatterCB3D),
        "CtxGatherFuncCB3D": (CtxGatherFuncCB3D, CtxGatherCB3D),
        "CtxGatherFuncBlockedKV": (CtxGatherFuncBlockedKV, CtxGatherBlockedKV),
        "CtxGatherFuncBlockedKVCB": (CtxGatherFuncBlockedKVCB, CtxGatherBlockedKVCB),
        "CtxScatterFuncCB": (CtxScatterFuncCB, CtxScatterCB),
        "CtxGatherFuncCB": (CtxGatherFuncCB, CtxGatherCB),
        # "CastToUInt4": (CastToUInt4Func, CastToUInt4),
    }

    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        op_applied = False

        # Register with PyTorch ONNX exporter (for export time)
        for op_name, (func_class, _) in cls._custom_ops.items():
            if hasattr(func_class, "symbolic"):
                torch.onnx.register_custom_op_symbolic(f"::{op_name}", func_class.symbolic, ONNX_EXPORT_OPSET)

        used_op_types = {node.op_type for node in model.graph.node}
        for function_proto in model.functions:
            used_op_types.update(node.op_type for node in function_proto.node)

        # Add function prototypes to model
        existing = {f.name for f in model.functions}

        for func_name, onnxscript_func in cls._custom_ops.values():
            proto = onnxscript_func.to_function_proto()
            if proto.name not in used_op_types:
                continue
            if proto.name not in existing:
                model.functions.append(proto)
                op_applied = True
                cls._ensure_opset_imports(model, proto.domain, 1)
        cls._propagate_function_opset_imports(model)
        return op_applied

    @staticmethod
    def _ensure_opset_imports(container, domain: str, version: int) -> None:
        if any(opset.domain == domain for opset in container.opset_import):
            return
        container.opset_import.append(onnx.helper.make_opsetid(domain, version))

    @classmethod
    def _propagate_function_opset_imports(cls, model: ModelProto) -> None:
        for fn in model.functions:
            for node in fn.node:
                if node.domain:
                    cls._ensure_opset_imports(fn, node.domain, 1)
                    cls._ensure_opset_imports(model, node.domain, 1)


class RemovePrefix(BaseOnnxTransform):
    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        graph = model.graph
        renamed = False

        def strip_prefix(name: str) -> str:
            parts = name.rsplit("/", 1)
            return parts[1] if len(parts) == 2 else parts[0]

        input_names = []
        for i, inputs in enumerate(graph.input):
            original = inputs.name
            new = strip_prefix(original)
            if new != original:
                renamed = True
            inputs.name = new
            graph.input[i].name = new
            input_names.append(new)

        input_name_set = set(input_names)
        output_rename_map = {}

        # Rename model graph outputs and keep mapping so producer/consumer edges can be fixed.
        for out in graph.output:
            original = out.name
            new = strip_prefix(original)
            if new != original:
                out.name = new
                output_rename_map[original] = new
                renamed = True

        for node in graph.node:
            for i, out in enumerate(node.output):
                if out in output_rename_map and output_rename_map[out] != out:
                    node.output[i] = output_rename_map[out]
                    renamed = True

            new_inputs = []
            for s in node.input:
                # Keep node inputs in sync for renamed model outputs.
                if s in output_rename_map:
                    new_inputs.append(output_rename_map[s])
                    continue

                if s in input_name_set:
                    new_inputs.append(s)
                    continue

                replaced = s
                if "/" in s:
                    tail = s.rsplit("/", 1)[1]
                    if tail in input_name_set:
                        replaced = tail
                new_inputs.append(replaced)

            for idx in range(len(node.input)):
                if node.input[idx] != new_inputs[idx]:
                    node.input[idx] = new_inputs[idx]
                    renamed = True

        return renamed


class RenameFunctionOutputsTransform(BaseOnnxTransform):
    """Rename decoder function retained-state outputs to public graph names.

    When subfunction export emits ``*_InternalRetainedState``, this transform rewrites
    them to ``*_RetainedState`` while preserving any optional KV prefix infix
    (for example ``past_key.0_vllmKvCache``).
    """

    @classmethod
    def apply(cls, model: ModelProto, layer_idx=0) -> bool:
        graph = model.graph
        op_type_to_func = {f.name: f for f in model.functions}
        decoder_patterns = ["DecoderLayer", "Block", "Layer"]
        renamed = False
        model_out_map = {v.name: i for i, v in enumerate(graph.output)}

        for node in graph.node:
            if any(p in node.name or p in node.op_type for p in decoder_patterns):
                func = op_type_to_func.get(node.op_type)
                if not func:
                    continue
                for i, out_name in enumerate(func.output):
                    if "_InternalRetainedState" in out_name:
                        renamed = True
                        orig = node.output[i]
                        if orig.endswith("_InternalRetainedState"):
                            new = orig[: -len("_InternalRetainedState")] + "_RetainedState"
                        else:
                            base = out_name[: -len("_InternalRetainedState")]
                            new = orig
                            for token in (
                                "past_key.",
                                "past_value.",
                                "compressed_kv.",
                                "k_pe.",
                                "recurrent_state.",
                                "conv_state.",
                            ):
                                if not base.startswith(token):
                                    continue
                                tail = base[len(token) :]
                                _, _, infix = tail.partition("_")
                                infix = f"_{infix}" if infix else ""
                                new = f"{token}{layer_idx}{infix}_RetainedState"
                                break
                        node.output[i] = new
                        if orig in model_out_map:
                            graph.output[model_out_map[orig]].name = new
                layer_idx += 1
        return renamed


class PreserveNestedCacheRetainedStateTransform(BaseOnnxTransform):
    """Expose nested decoder cache side effects as explicit ONNX values."""

    _KV_INPUT_RE = re.compile(r"^past_(key|value)\.(\d+)(?:_RetainedState)?$")

    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        graph = model.graph
        produced_names = {name for node in graph.node for name in node.output}
        dangling_retained_outputs = {
            out.name for out in graph.output if out.name.endswith("_RetainedState") and out.name not in produced_names
        }
        if not dangling_retained_outputs:
            return False

        fn_by_name = {fn.name: fn for fn in model.functions}
        changed = False

        for node in graph.node:
            fn = fn_by_name.get(node.op_type)
            if fn is None:
                continue

            scatter_outputs = [
                fn_node.output[0] for fn_node in fn.node if fn_node.op_type == "CtxScatter" and fn_node.output
            ]
            if len(scatter_outputs) != 2:
                continue

            layer_idx = None
            kv_inputs = {}
            for inp_name in node.input:
                match = cls._KV_INPUT_RE.match(inp_name)
                if match is None:
                    continue
                kind, idx = match.groups()
                layer_idx = idx if layer_idx is None else layer_idx
                if layer_idx != idx:
                    kv_inputs = {}
                    break
                kv_inputs[kind] = inp_name

            if layer_idx is None or set(kv_inputs) != {"key", "value"}:
                continue

            for scatter_output in scatter_outputs:
                if scatter_output not in fn.output:
                    fn.output.append(scatter_output)
                    changed = True

            for kind in ("key", "value"):
                retained_input = kv_inputs[kind]
                plain_input = f"past_{kind}.{layer_idx}"
                if retained_input.endswith("_RetainedState"):
                    changed |= cls._rename_graph_input(graph, retained_input, plain_input)
                    for i, name in enumerate(node.input):
                        if name == retained_input:
                            node.input[i] = plain_input

            desired_outputs = [
                f"past_key.{layer_idx}_RetainedState",
                f"past_value.{layer_idx}_RetainedState",
            ]
            if node.output[-len(desired_outputs) :] != desired_outputs:
                missing_outputs = [name for name in desired_outputs if name not in node.output]
                if missing_outputs:
                    node.output.extend(missing_outputs)
                changed = True

        return changed

    @staticmethod
    def _rename_graph_input(graph: onnx.GraphProto, old_name: str, new_name: str) -> bool:
        changed = False
        for value in graph.input:
            if value.name == old_name:
                value.name = new_name
                changed = True
        for value in graph.value_info:
            if value.name == old_name:
                value.name = new_name
                changed = True
        for node in graph.node:
            for i, name in enumerate(node.input):
                if name == old_name:
                    node.input[i] = new_name
                    changed = True
        return changed


class RenameRepeatedSubgraphTransform(BaseOnnxTransform):
    """Rename dynamo repeated_subgraph function names to model-specific layer class names."""

    _REPEATED_SUBGRAPH_RE = re.compile(r"^repeated_subgraph(\d+)$")

    @classmethod
    def apply(cls, model: ModelProto, target_classnames: Optional[List[str]] = None, **kwargs) -> bool:
        target_classnames = [name for name in (target_classnames or []) if name]
        if not target_classnames:
            return False

        repeated_functions = []
        for fn in model.functions:
            match = cls._REPEATED_SUBGRAPH_RE.match(fn.name)
            if match:
                repeated_functions.append((int(match.group(1)), fn))

        if not repeated_functions:
            return False

        repeated_functions.sort(key=lambda item: item[0])
        old_to_new = {}
        used_names = {fn.name for fn in model.functions}

        for idx, (_, fn) in enumerate(repeated_functions):
            base_name = target_classnames[min(idx, len(target_classnames) - 1)]
            candidate = base_name
            suffix = 1
            while candidate in used_names and candidate != fn.name:
                candidate = f"{base_name}_{suffix}"
                suffix += 1
            used_names.discard(fn.name)
            used_names.add(candidate)
            old_to_new[fn.name] = candidate

        if not old_to_new:
            return False

        for fn in model.functions:
            if fn.name in old_to_new:
                fn.name = old_to_new[fn.name]

        def _rename_op_types(nodes: List[onnx.NodeProto]):
            for node in nodes:
                if node.op_type in old_to_new:
                    node.op_type = old_to_new[node.op_type]

        _rename_op_types(model.graph.node)
        for fn in model.functions:
            _rename_op_types(fn.node)

        return True


class AdapterWeightsToInputsTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model: onnx.ModelProto, *, adapter_name: str, **kwargs) -> Tuple[onnx.ModelProto, bool]:
        transformed = False
        removed_initializers = []

        # Find nodes with lora weights as inputs
        weight_suffix = f".{adapter_name}.weight"
        lora_weight_nodes = {
            inp: node for node in model.graph.node for inp in node.input if inp.endswith(weight_suffix)
        }

        for i, weight in enumerate(model.graph.initializer):
            if weight.name.endswith(weight_suffix):
                transformed = True

                # Create input/output for lora weights
                new_weight_name = weight.name[: -len(weight_suffix)] + ".weight"
                type_proto = onnx.helper.make_tensor_type_proto(weight.data_type, shape=list(weight.dims))
                inp = onnx.ValueInfoProto(name=new_weight_name, type=type_proto)
                out = onnx.ValueInfoProto(name=new_weight_name + "_RetainedState", type=type_proto)
                model.graph.input.append(inp)
                model.graph.output.append(out)

                # Create a node that connects input -> output
                node = onnx.helper.make_node("Identity", [inp.name], [out.name], new_weight_name + "_identity")
                model.graph.node.append(node)

                # Rename weight input
                lora_weight_node = lora_weight_nodes[weight.name]
                for j, inp in enumerate(lora_weight_node.input):
                    if inp == weight.name:
                        lora_weight_node.input[j] = new_weight_name

                # Remove weight initializers
                removed_initializers.append(i)

        if transformed:
            for i in sorted(removed_initializers, reverse=True):
                model.graph.initializer.pop(i)

        return model, transformed


class RewriteUnsupportedOpsTransform(BaseOnnxTransform):
    """Rewrite unsupported ops like SplitToSequence and CastLike into supported equivalents."""

    _SEQ_PRODUCER_OPS_INT64 = {"Shape", "Size", "NonZero"}
    _INT_TYPES = (TensorProto.INT64, TensorProto.INT32)

    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        type_map, const_map = cls._build_maps(model.graph)
        const_map.update(cls._infer_function_const_outputs(model))

        changed = cls._rewrite_container(model.graph, type_map, const_map)

        func_const_map = cls._infer_function_const_inputs(model, const_map)
        for fn in model.functions:
            changed |= cls._rewrite_container(fn, type_map, {**const_map, **func_const_map.get(fn.name, {})})

        # Second pass: rewrite aten_split / aten_getitem function-call patterns.
        # The dynamo exporter wraps SplitToSequence inside an "aten_split" function
        # and each SequenceAt inside an "aten_getitem" function.  The existing
        # _try_rewrite_split only handles co-located nodes; this handles the
        # indirected call-node pattern.
        changed |= cls._rewrite_split_getitem_calls(model)

        return changed

    @classmethod
    def _rewrite_split_getitem_calls(cls, model: ModelProto) -> bool:
        """
        Rewrite patterns where:
          - A call to "aten_split"    (wraps SplitToSequence internally)
          - N calls to "aten_getitem" (each wraps SequenceAt internally)
        appear together in any container (graph or function body).

        Each aten_getitem call takes the aten_split output as its first input and
        a Constant integer as its second input.  We replace the whole group with a
        single Split node and wire the outputs directly, then delete the
        (now-unused) aten_split and aten_getitem function definitions if no other
        call sites remain.
        """
        changed = False

        def _rewrite_in(container) -> bool:
            nodes = list(container.node)
            if not nodes:
                return False
            print("top of rewrite")
            output_to_node = {out: n for n in nodes for out in n.output}
            skip_nodes: Set[int] = set()
            replace_map: Dict[str, str] = {}
            new_nodes: List[onnx.NodeProto] = []
            local_changed = False

            for node in nodes:
                if id(node) in skip_nodes:
                    continue

                if node.op_type not in ("aten_split", "aten_split_with_sizes"):
                    new_nodes.append(node)
                    continue

                split_seq_out = node.output[0]

                # Collect all aten_getitem consumers of this split output
                getitem_consumers = [
                    n for n in nodes if n.op_type == "aten_getitem" and n.input and n.input[0] == split_seq_out
                ]
                other_consumers = [
                    n for n in nodes if n.input and split_seq_out in n.input and n.op_type != "aten_getitem"
                ]

                # Only rewrite if all consumers are aten_getitem
                if not getitem_consumers or other_consumers:
                    new_nodes.append(node)
                    continue

                # Resolve constant index for each getitem
                idx_map: Dict[int, int] = {}  # id(getitem_node) → index value
                for gi in getitem_consumers:
                    if len(gi.input) < 2:
                        break
                    idx_name = gi.input[1]
                    idx_val = cls._get_const_int(idx_name, output_to_node, {})
                    if idx_val is None:
                        break
                    idx_map[id(gi)] = idx_val
                else:
                    # All indices resolved — proceed with rewrite
                    indices = list(idx_map.values())
                    if set(indices) != set(range(max(indices) + 1)):
                        new_nodes.append(node)
                        continue

                    num_outputs = max(indices) + 1
                    split_input = node.input[0]  # the tensor being split
                    split_size_input = node.input[1] if len(node.input) > 1 else None
                    split_out_names = [f"{split_seq_out}_part_{i}" for i in range(num_outputs)]

                    # Build the Split node
                    split_node_inputs = [split_input]
                    split_node_name = f"{node.name}_split" if node.name else f"{split_seq_out}_split"

                    if split_size_input:
                        split_size_val = cls._get_const_int(split_size_input, output_to_node, {})
                        if split_size_val is not None:
                            const_name = f"{split_node_name}_sizes"
                            size_list = [split_size_val] * num_outputs
                            new_nodes.append(
                                onnx.helper.make_node(
                                    "Constant",
                                    [],
                                    [const_name],
                                    name=f"{const_name}_const",
                                    value=onnx.helper.make_tensor(
                                        const_name, TensorProto.INT64, [len(size_list)], size_list
                                    ),
                                )
                            )
                            split_node_inputs.append(const_name)
                        else:
                            split_node_inputs.append(split_size_input)

                    new_nodes.append(
                        onnx.helper.make_node(
                            "Split",
                            split_node_inputs,
                            split_out_names,
                            name=split_node_name,
                            axis=-1,
                        )
                    )

                    # Map each getitem output → the correct Split output slice
                    for gi in getitem_consumers:
                        replace_map[gi.output[0]] = split_out_names[idx_map[id(gi)]]
                        skip_nodes.add(id(gi))

                    skip_nodes.add(id(node))
                    local_changed = True
                    continue

                # Fallthrough: couldn't resolve all indices
                new_nodes.append(node)

            if local_changed:
                cls._apply_replacements(container, new_nodes, replace_map, skip_nodes)
                del container.node[:]
                container.node.extend([n for n in new_nodes if id(n) not in skip_nodes])

            return local_changed

        # Apply in graph and all function bodies
        changed |= _rewrite_in(model.graph)
        for fn in model.functions:
            changed |= _rewrite_in(fn)

        # Remove aten_split / aten_getitem function definitions if no call sites remain
        if changed:
            all_op_types: Set[str] = set()
            for node in model.graph.node:
                all_op_types.add(node.op_type)
            for fn in model.functions:
                for node in fn.node:
                    all_op_types.add(node.op_type)

            fns_to_keep = [fn for fn in model.functions if fn.name in all_op_types]
            if len(fns_to_keep) < len(model.functions):
                del model.functions[:]
                model.functions.extend(fns_to_keep)

        return changed

    @classmethod
    def _build_maps(cls, graph: onnx.GraphProto) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Build type and constant maps from graph."""
        type_map, const_map = {}, {}

        # Collect from value_info, inputs, outputs, initializers
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("elem_type"):
                type_map[vi.name] = vi.type.tensor_type.elem_type

        for init in graph.initializer:
            type_map[init.name] = init.data_type
            if init.data_type in cls._INT_TYPES:
                if init.data_location == TensorProto.EXTERNAL:
                    continue
                cls._extract_const_value(numpy_helper.to_array(init), const_map, init.name)

        # Collect from Constant nodes
        for node in graph.node:
            if node.op_type == "Constant" and node.output:
                cls._extract_constant_node(node, const_map)

        return type_map, const_map

    @classmethod
    def _extract_const_value(cls, arr: Any, const_map: Dict[str, Any], name: str) -> None:
        """Extract constant value from numpy array."""
        if arr.ndim == 0:
            const_map[name] = arr.item()
        elif arr.ndim == 1:
            const_map[name] = arr.tolist()

    @classmethod
    def _extract_constant_node(cls, node: onnx.NodeProto, const_map: Dict[str, Any]) -> None:
        """Extract constant from Constant node."""
        value_attr = next((a for a in node.attribute if a.name == "value"), None)
        if value_attr and value_attr.t.data_type in cls._INT_TYPES:
            if value_attr.t.data_location == TensorProto.EXTERNAL:
                return
            cls._extract_const_value(numpy_helper.to_array(value_attr.t), const_map, node.output[0])

    @classmethod
    def _rewrite_container(cls, container: Any, type_map: Dict[str, int], const_map: Dict[str, Any]) -> bool:
        """Rewrite nodes in container (graph or function)."""
        nodes = list(container.node)
        if not nodes:
            return False

        output_to_node = {out: node for node in nodes for out in node.output}
        replace_map, skip_nodes, new_nodes = {}, set(), []
        changed = False

        for node in nodes:
            if id(node) in skip_nodes:
                continue

            if node.op_type == "SplitToSequence":
                if cls._try_rewrite_split(node, nodes, output_to_node, const_map, new_nodes, replace_map, skip_nodes):
                    changed = True
                    continue

            if node.op_type == "CastLike":
                new_nodes.append(cls._create_cast_node(node, output_to_node, type_map))
                changed = True
                continue

            new_nodes.append(node)

        if replace_map:
            cls._apply_replacements(container, new_nodes, replace_map, skip_nodes)

        if changed:
            del container.node[:]
            container.node.extend([n for n in new_nodes if id(n) not in skip_nodes])

        return changed

    @classmethod
    def _try_rewrite_split(
        cls,
        node: onnx.NodeProto,
        nodes: List[onnx.NodeProto],
        output_to_node: Dict[str, onnx.NodeProto],
        const_map: Dict[str, Any],
        new_nodes: List[onnx.NodeProto],
        replace_map: Dict[str, str],
        skip_nodes: Set[int],
    ) -> bool:
        """Try to rewrite SplitToSequence + SequenceAt to Split."""
        split_out = node.output[0]
        consumers = [n for n in nodes if split_out in n.input and n.op_type == "SequenceAt"]

        if not consumers or len([n for n in nodes if split_out in n.input]) != len(consumers):
            new_nodes.append(node)
            return False

        # Resolve all indices
        seq_index_map = {}
        for seq_at in consumers:
            idx_val = cls._get_const_int(seq_at.input[1], output_to_node, const_map)
            if idx_val is None:
                new_nodes.append(node)
                return False
            seq_index_map[id(seq_at)] = idx_val

        # Validate contiguous indices
        indices = list(seq_index_map.values())
        if set(indices) != set(range(max(indices) + 1)):
            new_nodes.append(node)
            return False

        # Create Split node
        num_outputs = max(indices) + 1
        split_outputs = [f"{split_out}_part_{i}" for i in range(num_outputs)]
        split_node = cls._create_split_node(node, split_out, split_outputs, const_map, new_nodes, num_outputs)
        new_nodes.append(split_node)

        # Map outputs and mark for removal
        for seq_at in consumers:
            replace_map[seq_at.output[0]] = split_outputs[seq_index_map[id(seq_at)]]
            skip_nodes.add(id(seq_at))

        skip_nodes.add(id(node))
        return True

    @classmethod
    def _create_split_node(
        cls,
        node: onnx.NodeProto,
        split_out: str,
        split_outputs: List[str],
        const_map: Dict[str, Any],
        new_nodes: List[onnx.NodeProto],
        num_outputs: int,
    ) -> onnx.NodeProto:
        """Create Split node from SplitToSequence."""
        split_inputs = [node.input[0]]
        axis = next((a.i for a in node.attribute if a.name == "axis"), 0)

        # Handle split sizes
        if len(node.input) > 1 and node.input[1]:
            split_val = const_map.get(node.input[1])
            split_attr = (
                split_val
                if isinstance(split_val, list)
                else ([split_val] * num_outputs if isinstance(split_val, int) else None)
            )

            if split_attr:
                split_const_name = f"{node.name or split_out}_split_sizes"
                new_nodes.append(
                    onnx.helper.make_node(
                        "Constant",
                        [],
                        [split_const_name],
                        name=f"{split_const_name}_const",
                        value=onnx.helper.make_tensor(
                            split_const_name, TensorProto.INT64, [len(split_attr)], split_attr
                        ),
                    )
                )
                split_inputs.append(split_const_name)
            else:
                split_inputs.append(node.input[1])

        return onnx.helper.make_node(
            "Split", split_inputs, split_outputs, name=f"{node.name}_split" if node.name else "", axis=axis
        )

    @classmethod
    def _create_cast_node(
        cls, node: onnx.NodeProto, output_to_node: Dict[str, onnx.NodeProto], type_map: Dict[str, int]
    ) -> onnx.NodeProto:
        """Create Cast node from CastLike."""
        target_type = cls._resolve_type(node.input[1], output_to_node, type_map) or TensorProto.INT64
        return onnx.helper.make_node("Cast", [node.input[0]], list(node.output), name=node.name, to=target_type)

    @classmethod
    def _apply_replacements(
        cls, container: Any, new_nodes: List[onnx.NodeProto], replace_map: Dict[str, str], skip_nodes: Set[int]
    ) -> None:
        """Apply name replacements throughout container."""
        # Update node inputs
        for node in new_nodes:
            if id(node) not in skip_nodes:
                node.input[:] = [replace_map.get(inp, inp) for inp in node.input]

        # Update outputs
        if hasattr(container, "output"):
            if container.output and isinstance(container.output[0], str):
                container.output[:] = [replace_map.get(out, out) for out in container.output]
            else:
                for out in container.output:
                    out.name = replace_map.get(out.name, out.name)

        # Update value_info
        if hasattr(container, "value_info"):
            for vi in container.value_info:
                vi.name = replace_map.get(vi.name, vi.name)

    @classmethod
    def _get_const_int(
        cls, name: str, output_to_node: Dict[str, onnx.NodeProto], const_map: Dict[str, Any]
    ) -> Optional[int]:
        """Get constant integer value."""
        if name in const_map:
            val = const_map[name]
            return val if isinstance(val, int) else None

        node = output_to_node.get(name)
        if node and node.op_type == "Constant":
            value_attr = next((a for a in node.attribute if a.name == "value"), None)
            if value_attr and not value_attr.t.dims:
                return numpy_helper.to_array(value_attr.t).item()

        return None

    @classmethod
    def _resolve_type(
        cls, name: str, output_to_node: Dict[str, onnx.NodeProto], type_map: Dict[str, int]
    ) -> Optional[int]:
        """Resolve data type for a value."""
        if name in type_map:
            return type_map[name]

        node = output_to_node.get(name)
        if not node:
            return TensorProto.INT64 if any(t in name for t in ("sym_size", "size", "len", "int")) else None

        if node.op_type == "Constant":
            value_attr = next((a for a in node.attribute if a.name == "value"), None)
            return value_attr.t.data_type if value_attr else None

        if node.op_type == "Cast":
            to_attr = next((a for a in node.attribute if a.name == "to"), None)
            return int(to_attr.i) if to_attr else None

        return TensorProto.INT64 if node.op_type in cls._SEQ_PRODUCER_OPS_INT64 else None

    @classmethod
    def _infer_function_const_inputs(cls, model: ModelProto, const_map: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Infer constant function inputs from call sites."""
        func_const_map = {}

        for fn in model.functions:
            if not fn.input:
                continue

            per_input = {}
            for idx, fn_input in enumerate(fn.input):
                # Collect values from all call sites
                values = set()
                for node in model.graph.node:
                    if node.op_type != fn.name or idx >= len(node.input):
                        values.clear()
                        break

                    graph_input = node.input[idx]
                    if graph_input not in const_map:
                        values.clear()
                        break

                    val = const_map[graph_input]
                    values.add(tuple(val) if isinstance(val, list) else val)

                # Record if all call sites agree
                if len(values) == 1:
                    const_val = next(iter(values))
                    per_input[fn_input] = list(const_val) if isinstance(const_val, tuple) else const_val

            if per_input:
                func_const_map[fn.name] = per_input

        return func_const_map

    @classmethod
    def _infer_function_const_outputs(cls, model: ModelProto) -> Dict[str, Any]:
        """Infer constant outputs from functions."""
        # Build function output constants map
        func_output_consts = {}
        for fn in model.functions:
            output_map = {}
            for out_name in fn.output:
                prod = next((n for n in fn.node if out_name in n.output and n.op_type == "Constant"), None)
                if prod:
                    value_attr = next((a for a in prod.attribute if a.name == "value"), None)
                    if value_attr and value_attr.t.data_type in cls._INT_TYPES:
                        cls._extract_const_value(numpy_helper.to_array(value_attr.t), output_map, out_name)

            if output_map:
                func_output_consts[fn.name] = output_map

        # Propagate to call sites
        const_outputs = {}
        fn_map = {f.name: f for f in model.functions}

        for node in model.graph.node:
            if node.op_type in func_output_consts:
                fn = fn_map.get(node.op_type)
                if fn:
                    output_const_map = func_output_consts[node.op_type]
                    for idx, out_name in enumerate(node.output):
                        if idx < len(fn.output) and fn.output[idx] in output_const_map:
                            const_outputs[out_name] = output_const_map[fn.output[idx]]

        return const_outputs


class OnnxTransformPipeline(BaseOnnxTransform):
    """Pipeline to apply multiple ONNX transformations in sequence."""

    def __init__(self, transforms: List[Type[BaseOnnxTransform]]):
        self.transforms = transforms

    def apply(
        self,
        model: ModelProto,
        *,
        model_name: str = "",
        onnx_base_dir: Optional[str] = None,
        file_chunk_size: int = FILE_CHUNK_SIZE_DEFAULT,
        size_threshold: int = SIZE_THRESHOLD_DEFAULT,
        **kwargs,
    ) -> Tuple[ModelProto, bool]:
        if not self.transforms:
            return model, False

        # Same logic as before, but replace `transforms` with `self.transforms`
        mapping: Dict[str, Tuple[TensorProto, str]] = {}
        requested = set(self.transforms)
        applied = {t: False for t in requested}
        f16_applied = False
        do_fp16 = FP16ClipTransform in requested
        do_split = SplitTensorsTransform in requested
        fp16_min, fp16_max = np.finfo(np.float16).min, np.finfo(np.float16).max
        file_num_tracker = {"num": 0, "size": 0}
        if onnx_base_dir is not None:
            external_data_helper.load_external_data_for_model(model, onnx_base_dir)

        if do_fp16 or do_split:
            for tensor in external_data_helper._get_all_tensors(model):
                if do_fp16 and FP16ClipTransform.apply(tensor, onnx_base_dir, fp16_max, fp16_min):
                    f16_applied = True
                applied[FP16ClipTransform] = f16_applied

                if do_split and tensor.HasField("raw_data"):
                    tsize = len(tensor.raw_data)
                    if tsize > size_threshold:
                        if file_num_tracker["size"] + tsize > file_chunk_size:
                            file_num_tracker["num"] += 1
                            file_num_tracker["size"] = tsize
                        else:
                            file_num_tracker["size"] += tsize
                        applied[SplitTensorsTransform] = True
                        SplitTensorsTransform.apply(tensor, model_name, file_num_tracker["num"], mapping)

        def _set_external_data(tensor, file_name):
            external_data_helper.set_external_data(tensor, file_name)

        max_workers = min(32, (os.cpu_count() or 1) * 4)
        logger.info(f"Applying external data mapping with {max_workers} threads")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_set_external_data, tensor, file_name) for tensor, file_name in mapping.values()]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to set external data: {e}")

        # Non-looping transforms
        if CustomOpTransform in requested:
            applied[CustomOpTransform] = CustomOpTransform.apply(model)

        if RenameFunctionOutputsTransform in requested:
            applied[RenameFunctionOutputsTransform] = RenameFunctionOutputsTransform.apply(
                model, layer_idx=kwargs.get("layer_idx", 0)
            )

        if PreserveNestedCacheRetainedStateTransform in requested:
            applied[PreserveNestedCacheRetainedStateTransform] = PreserveNestedCacheRetainedStateTransform.apply(model)

        if RenameRepeatedSubgraphTransform in requested:
            applied[RenameRepeatedSubgraphTransform] = RenameRepeatedSubgraphTransform.apply(model, **kwargs)

        if AdapterWeightsToInputsTransform in requested:
            applied[AdapterWeightsToInputsTransform] = AdapterWeightsToInputsTransform.apply(model, **kwargs)

        # if RewriteUnsupportedOpsTransform in requested:
        #     applied[RewriteUnsupportedOpsTransform] = RewriteUnsupportedOpsTransform.apply(model)

        for t, done in applied.items():
            logger.info(f"Transform '{t.__name__}' applied={done}")

        return model, any(applied.values())
