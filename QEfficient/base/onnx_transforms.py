# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import copy
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import onnx
import torch
from onnx import ModelProto, TensorProto, external_data_helper, numpy_helper

from QEfficient.customop.ctx_scatter_gather import (
    CtxChunkScatterBatch,
    CtxChunkScatterBatchFunc,
    CtxGather,
    CtxGather3D,
    CtxGatherBlockedKV,
    CtxGatherBlockedKVBatch,
    CtxGatherFunc,
    CtxGatherFunc3D,
    CtxGatherFunc3DGeneralized,
    CtxGatherFuncBlockedKV,
    CtxGatherFuncBlockedKVBatch,
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
from QEfficient.customop.onnxscript_utils import get_onnxscript_func
from QEfficient.customop.quantization_ops import CastToUInt4, CastToUInt4Func
from QEfficient.customop.rms_norm import CustomRMSNorm, CustomRMSNormFunc
from QEfficient.utils import constants
from QEfficient.utils.constants import FILE_CHUNK_SIZE_DEFAULT, SIZE_THRESHOLD_DEFAULT

logger = logging.getLogger(__name__)


class BaseOnnxTransform:
    """Base class for ONNX graph modifications. Should NOT be instantiated."""

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Use the `apply` method directly.")

    @classmethod
    def apply(cls, model: ModelProto, **kwargs) -> bool:
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
        "CastToUInt4": (CastToUInt4Func, CastToUInt4),
        "CtxChunkScatterBatchFunc": (CtxChunkScatterBatchFunc, CtxChunkScatterBatch),
        "CtxGatherFuncBlockedKVBatch": (CtxGatherFuncBlockedKVBatch, CtxGatherBlockedKVBatch),
    }

    @classmethod
    def apply(cls, model: ModelProto, onnx_export_opset: int = constants.ONNX_LEGACY_EXPORT_OPSET) -> bool:
        op_applied = False

        # Register with PyTorch ONNX exporter (for export time)
        for op_name, (func_class, _) in cls._custom_ops.items():
            if hasattr(func_class, "symbolic"):
                torch.onnx.register_custom_op_symbolic(f"::{op_name}", func_class.symbolic, onnx_export_opset)

        used_op_types = {node.op_type for node in model.graph.node}
        for function_proto in model.functions:
            used_op_types.update(node.op_type for node in function_proto.node)

        # Add function prototypes to model
        existing = {f.name for f in model.functions}

        for func_name, onnxscript_func in cls._custom_ops.values():
            proto = get_onnxscript_func(onnxscript_func, onnx_export_opset).to_function_proto()
            if proto.name not in used_op_types:
                continue
            if proto.name not in existing:
                model.functions.append(proto)
                op_applied = True
                cls._ensure_opset_imports(model, proto.domain, 1)
        cls._propagate_function_opset_imports(model)
        return op_applied

    @staticmethod
    def _ensure_opset_imports(container: Union[ModelProto, onnx.FunctionProto], domain: str, version: int) -> None:
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
    """Expose nested decoder cache side effects as explicit ONNX values.

    Must run BEFORE RenameRepeatedSubgraphTransform: this transform looks up
    functions by the dynamo-assigned name (repeated_subgraphN) via fn_by_name.
    Renaming them first would break that lookup.
    """

    # Match past_key.N / past_value.N regardless of any suffix that follows
    # (plain, _RetainedState, or _<prefix>_RetainedState for kv_cache_prefix).
    _KV_INPUT_RE = re.compile(r"^past_(key|value)\.(\d+)")

    # All scatter op_type names that write back a KV cache tensor.
    # Keep in sync with CustomOpTransform._custom_ops and the dynamo
    # custom_translation_table in base/modeling_qeff.py.
    _SCATTER_OP_TYPES = frozenset(
        {
            "CtxScatter",
            "CtxScatterCB",
            "CtxScatter3D",
            "CtxScatter3DInt",
            "CtxScatterCB3D",
        }
    )

    @staticmethod
    def _scatter_sort_key(n) -> int:
        output_name = n.output[0] if n.output else ""
        if "key" in output_name:
            return 0
        if "value" in output_name:
            return 1
        return 2

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
        kv_rename_map: Dict[str, str] = {}

        for node in graph.node:
            fn = fn_by_name.get(node.op_type)
            if fn is None:
                continue

            # Collect scatter nodes that write back the KV cache.
            # Sort by first-input name so the key scatter reliably precedes the
            # value scatter: dynamo names function-body args generically (arg7_1
            # etc.), so we sort by the scatter output name instead — dynamo
            # preserves "key"/"value" in output tensor names even when input
            # argument names are opaque.
            scatter_nodes = [
                fn_node for fn_node in fn.node if fn_node.op_type in cls._SCATTER_OP_TYPES and fn_node.output
            ]
            if len(scatter_nodes) != 2:
                logger.debug(
                    "PreserveNestedCacheRetainedStateTransform: function '%s' has %d scatter node(s), expected 2 — skipping.",
                    node.op_type,
                    len(scatter_nodes),
                )
                continue

            scatter_nodes.sort(key=cls._scatter_sort_key)
            # Only the first two scatter outputs map to key / value respectively.
            scatter_outputs = [n.output[0] for n in scatter_nodes[:2]]

            # Identify layer index from KV inputs on this call node.
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

            desired_outputs = [
                f"past_key.{layer_idx}_RetainedState",
                f"past_value.{layer_idx}_RetainedState",
            ]
            # Skip layers whose retained-state outputs are not dangling —
            # either the graph is already correctly wired or a previous call
            # to this transform already fixed them.
            if not any(name in dangling_retained_outputs for name in desired_outputs):
                continue

            # Expose scatter outputs in the function's output list, rename KV
            # inputs and append retained-state output names to the call node —
            # all in one pass over the two key/value pairs.
            for kind, scatter_output, desired_output in zip(("key", "value"), scatter_outputs, desired_outputs):
                if scatter_output not in fn.output:
                    fn.output.append(scatter_output)
                    changed = True

                retained_input = kv_inputs[kind]
                plain_input = f"past_{kind}.{layer_idx}"
                if retained_input.endswith("_RetainedState"):
                    kv_rename_map[retained_input] = plain_input

                if desired_output not in node.output:
                    node.output.append(desired_output)
                    changed = True

        if kv_rename_map:
            changed |= cls._rename_graph_inputs_bulk(graph, kv_rename_map)
        return changed

    @staticmethod
    def _rename_graph_inputs_bulk(graph: onnx.GraphProto, rename_map: Dict[str, str]) -> bool:
        if not rename_map:
            return False
        changed = False
        for value in graph.input:
            if value.name in rename_map:
                value.name = rename_map[value.name]
                changed = True
        for value in graph.value_info:
            if value.name in rename_map:
                value.name = rename_map[value.name]
                changed = True
        for node in graph.node:
            new_inputs = [rename_map.get(n, n) for n in node.input]
            if new_inputs != list(node.input):
                node.input[:] = new_inputs
                changed = True
        return changed


class RenameRepeatedSubgraphTransform(BaseOnnxTransform):
    """Rename dynamo repeated_subgraph function names to model-specific layer class names.

    Must run AFTER PreserveNestedCacheRetainedStateTransform: that transform
    looks up functions by the dynamo-assigned name.  Renaming them first would
    break that lookup.
    """

    # Primary pattern emitted by torch.export repeated-subgraph canonicalization.
    # Extended to also match alternative patterns seen across PyTorch 2.x releases
    # so a dynamo-internal rename does not silently produce a no-op transform.
    _REPEATED_SUBGRAPH_PATTERNS = [
        re.compile(r"^repeated_subgraph(\d+)$"),  # torch >= 2.5 canonical name
        re.compile(r"^subgraph_(\d+)$"),  # alternative seen in some 2.x nightlies
        re.compile(r"^invoke_subgraph_(\d+)$"),  # earlier 2.x name
    ]

    @classmethod
    def _iter_all_nodes(cls, nodes):
        """Yield every NodeProto reachable from `nodes`, including nodes nested
        inside If/Loop subgraph attributes."""
        for node in nodes:
            yield node
            for attr in node.attribute:
                if attr.HasField("g"):
                    yield from cls._iter_all_nodes(attr.g.node)

    @staticmethod
    def _rename_op_types(nodes, old_to_new: Dict[str, str]) -> None:
        for node in RenameRepeatedSubgraphTransform._iter_all_nodes(nodes):
            if node.op_type in old_to_new:
                node.op_type = old_to_new[node.op_type]

    @classmethod
    def apply(cls, model: ModelProto, target_classnames: Optional[List[str]] = None, **kwargs) -> bool:
        target_classnames = [name for name in (target_classnames or []) if name]
        if not target_classnames:
            logger.warning(
                "RenameRepeatedSubgraphTransform: target_classnames is empty — transform is a no-op. "
                "Check that get_submodules_for_export() returns the decoder layer classes."
            )
            return False

        repeated_functions = []
        for fn in model.functions:
            for pattern in cls._REPEATED_SUBGRAPH_PATTERNS:
                match = pattern.match(fn.name)
                if match:
                    repeated_functions.append((int(match.group(1)), fn))
                    break

        if not repeated_functions:
            logger.warning(
                "RenameRepeatedSubgraphTransform: no repeated_subgraph functions found in the ONNX model. "
                "This may indicate that dynamo changed its internal subgraph naming convention. "
                "The transform is a no-op; function names remain as emitted by torch.export."
            )
            return False

        repeated_functions.sort(key=lambda item: item[0])
        old_to_new = {}
        used_names = {fn.name for fn in model.functions}

        for idx, (_, fn) in enumerate(repeated_functions):
            if idx >= len(target_classnames):
                logger.warning(
                    f"RenameRepeatedSubgraphTransform: more repeated subgraph functions ({len(repeated_functions)}) "
                    f"than target class names ({len(target_classnames)}). "
                    f"Function '{fn.name}' (index {idx}) will be assigned the last available name "
                    f"'{target_classnames[-1]}' with a numeric suffix — verify get_submodules_for_export() "
                    "returns all repeated block classes for this model."
                )
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

        cls._rename_op_types(model.graph.node, old_to_new)
        for fn in model.functions:
            cls._rename_op_types(fn.node, old_to_new)

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


class InlineTorchSubgraphFunctionsTransform(BaseOnnxTransform):
    """Inline torch-export local subgraph functions before QAIC compilation.

    QAIC currently rejects ONNX Loop nodes when they appear inside local
    FunctionProto bodies because function-local values are not accepted as
    compile-time constants for Loop control inputs. Dynamo subfunction export
    emits decoder layers and while_loop body/condition helpers in the
    ``pkg.torch.__subgraph__`` domain, so this transform inlines those generated
    torch subgraphs back into their callers. QEfficient custom op function
    prototypes are left untouched.
    """

    _TORCH_SUBGRAPH_DOMAIN = "pkg.torch.__subgraph__"

    @classmethod
    def apply(cls, model: ModelProto, **kwargs) -> bool:
        functions_by_key = {(function.domain, function.name): function for function in model.functions}
        function_keys_to_inline = {
            key for key in functions_by_key if key[0] == cls._TORCH_SUBGRAPH_DOMAIN
        }
        if not function_keys_to_inline:
            return False

        def scoped_name(name: str, value_map: Dict[str, str], prefix: str) -> str:
            if not name:
                return name
            return value_map.get(name, f"{prefix}/{name}")

        def collect_graph_local_names(graph) -> set[str]:
            local_names = {value.name for value in graph.input}
            local_names.update(value.name for value in graph.output)
            local_names.update(value.name for value in graph.value_info)
            local_names.update(tensor.name for tensor in graph.initializer)
            for node in graph.node:
                local_names.update(output for output in node.output if output)
            return local_names

        def clone_graph(graph, parent_map: Dict[str, str], parent_prefix: str, graph_prefix: str):
            cloned_graph = onnx.GraphProto()
            cloned_graph.CopyFrom(graph)
            cloned_graph.name = f"{graph_prefix}/{graph.name}" if graph.name else graph_prefix

            local_names = collect_graph_local_names(graph)
            local_map = dict(parent_map)
            for name in local_names:
                local_map[name] = f"{graph_prefix}/{name}"

            def map_graph_input(name: str) -> str:
                if not name:
                    return name
                if name in local_names:
                    return local_map[name]
                if name in parent_map:
                    return parent_map[name]
                return f"{parent_prefix}/{name}"

            def map_graph_local(name: str) -> str:
                return scoped_name(name, local_map, graph_prefix)

            for graph_input in cloned_graph.input:
                graph_input.name = map_graph_local(graph_input.name)
            for graph_output in cloned_graph.output:
                graph_output.name = map_graph_local(graph_output.name)
            for value_info in cloned_graph.value_info:
                value_info.name = map_graph_local(value_info.name)
            for initializer in cloned_graph.initializer:
                initializer.name = map_graph_local(initializer.name)
            for sparse_initializer in cloned_graph.sparse_initializer:
                if sparse_initializer.values.name:
                    sparse_initializer.values.name = map_graph_local(sparse_initializer.values.name)
                if sparse_initializer.indices.name:
                    sparse_initializer.indices.name = map_graph_local(sparse_initializer.indices.name)

            cloned_nodes = []
            for node in graph.node:
                cloned_node = clone_node(
                    node,
                    local_map,
                    graph_prefix,
                    input_mapper=map_graph_input,
                )
                cloned_nodes.extend(expand_node(cloned_node, graph_prefix))
            del cloned_graph.node[:]
            cloned_graph.node.extend(cloned_nodes)
            return cloned_graph

        def clone_attribute(attribute, value_map: Dict[str, str], parent_prefix: str, graph_prefix: str):
            cloned_attribute = onnx.AttributeProto()
            cloned_attribute.CopyFrom(attribute)
            if attribute.HasField("g"):
                cloned_attribute.g.CopyFrom(
                    clone_graph(attribute.g, value_map, parent_prefix, f"{graph_prefix}/{attribute.name}")
                )
            if attribute.graphs:
                del cloned_attribute.graphs[:]
                for graph_index, graph in enumerate(attribute.graphs):
                    cloned_attribute.graphs.append(
                        clone_graph(graph, value_map, parent_prefix, f"{graph_prefix}/{attribute.name}_{graph_index}")
                    )
            return cloned_attribute

        def clone_node(node, value_map: Dict[str, str], prefix: str, input_mapper=None):
            cloned_node = onnx.NodeProto()
            cloned_node.CopyFrom(node)
            original_name = cloned_node.name or cloned_node.op_type
            if cloned_node.name:
                cloned_node.name = f"{prefix}/{cloned_node.name}"
            if input_mapper is None:
                input_mapper = lambda input_name: scoped_name(input_name, value_map, prefix)
            cloned_node.input[:] = [input_mapper(input_name) for input_name in node.input]
            cloned_node.output[:] = [scoped_name(output_name, value_map, prefix) for output_name in node.output]
            attributes = list(node.attribute)
            del cloned_node.attribute[:]
            for attribute in attributes:
                cloned_node.attribute.append(clone_attribute(attribute, value_map, prefix, f"{prefix}/{original_name}"))
            return cloned_node

        def expand_node(node, prefix: str) -> List[onnx.NodeProto]:
            function = functions_by_key.get((node.domain, node.op_type))
            if function is None or (function.domain, function.name) not in function_keys_to_inline:
                return [node]

            value_map = {formal: actual for formal, actual in zip(function.input, node.input)}
            value_map.update({formal: actual for formal, actual in zip(function.output, node.output)})
            expanded_nodes = []
            for function_node in function.node:
                cloned_node = clone_node(function_node, value_map, prefix)
                expanded_nodes.extend(expand_node(cloned_node, prefix))
            return expanded_nodes

        expanded_graph_nodes = []
        inlined_call_count = 0
        for node_index, node in enumerate(model.graph.node):
            expanded_nodes = expand_node(node, node.name or f"node_{node_index}")
            if len(expanded_nodes) != 1 or expanded_nodes[0] is not node:
                inlined_call_count += 1
            expanded_graph_nodes.extend(expanded_nodes)

        if inlined_call_count == 0:
            return False

        del model.graph.node[:]
        model.graph.node.extend(expanded_graph_nodes)
        remaining_functions = [
            function for function in model.functions if (function.domain, function.name) not in function_keys_to_inline
        ]
        del model.functions[:]
        model.functions.extend(remaining_functions)
        return True


class StaticLoopInputsTransform(BaseOnnxTransform):
    """Rewrite exported ONNX Loop inputs to static constants for compiler unrolling.

    torch.while_loop can export a Loop with dynamic graph-produced control
    inputs. The current QAIC compiler expects those Loop control inputs to be
    compile-time constants. For fixed-CL KV blocking models, ``num_kv_blocks``
    is the static maximum trip count.
    """

    @classmethod
    def apply(cls, model: ModelProto, *, num_kv_blocks: Optional[int] = None, **kwargs) -> bool:
        if num_kv_blocks is None:
            return False

        transformed = False
        loop_index = 0

        def next_names() -> Tuple[str, str]:
            nonlocal loop_index
            trip_name = f"qeff_static_loop_trip_count_{loop_index}"
            cond_name = f"qeff_static_loop_cond_true_{loop_index}"
            loop_index += 1
            return trip_name, cond_name

        def make_trip_tensor(name: str) -> TensorProto:
            return onnx.helper.make_tensor(name, TensorProto.INT64, [], [int(num_kv_blocks)])

        def make_cond_tensor(name: str) -> TensorProto:
            return onnx.helper.make_tensor(name, TensorProto.BOOL, [], [True])

        def rewrite_loop_node(node, trip_name: str, cond_name: str) -> bool:
            if len(node.input) < 2:
                logger.warning(
                    "StaticLoopInputsTransform: Loop node '%s' has fewer than 2 inputs; skipping.", node.name
                )
                return False
            node.input[0] = trip_name
            node.input[1] = cond_name
            return True

        def rewrite_graph(graph) -> bool:
            graph_changed = False
            initializer_names = {initializer.name for initializer in graph.initializer}

            def add_initializer_once(name: str, tensor: TensorProto) -> None:
                if name not in initializer_names:
                    graph.initializer.append(tensor)
                    initializer_names.add(name)

            for node in graph.node:
                for attr in node.attribute:
                    if attr.HasField("g"):
                        graph_changed |= rewrite_graph(attr.g)
                    for nested_graph in attr.graphs:
                        graph_changed |= rewrite_graph(nested_graph)

                if node.op_type != "Loop":
                    continue
                trip_name, cond_name = next_names()
                add_initializer_once(trip_name, make_trip_tensor(trip_name))
                add_initializer_once(cond_name, make_cond_tensor(cond_name))
                graph_changed |= rewrite_loop_node(node, trip_name, cond_name)
            return graph_changed

        def rewrite_function(function) -> bool:
            function_changed = False
            constant_nodes = []
            for node in function.node:
                for attr in node.attribute:
                    if attr.HasField("g"):
                        function_changed |= rewrite_graph(attr.g)
                    for nested_graph in attr.graphs:
                        function_changed |= rewrite_graph(nested_graph)

                if node.op_type != "Loop":
                    continue
                trip_name, cond_name = next_names()
                constant_nodes.extend(
                    [
                        onnx.helper.make_node(
                            "Constant", [], [trip_name], value=make_trip_tensor(f"{trip_name}_value")
                        ),
                        onnx.helper.make_node(
                            "Constant", [], [cond_name], value=make_cond_tensor(f"{cond_name}_value")
                        ),
                    ]
                )
                function_changed |= rewrite_loop_node(node, trip_name, cond_name)

            if constant_nodes:
                existing_nodes = list(function.node)
                del function.node[:]
                function.node.extend(constant_nodes)
                function.node.extend(existing_nodes)
            return function_changed

        transformed |= rewrite_graph(model.graph)
        for function in model.functions:
            transformed |= rewrite_function(function)

        return transformed


class PruneFakeInitializersTransform(BaseOnnxTransform):
    """Remove initializers backed by FakeTensors from a dynamo onnx_program before serialisation.

    Operates on the torch.onnx.ONNXProgram object returned by dynamo export, not on a
    ModelProto, because FakeTensors cannot survive onnx.load round-trips.
    """

    @classmethod
    def apply(cls, onnx_program) -> bool:
        from torch._subclasses.fake_tensor import FakeTensor

        initializers = onnx_program.model.graph.initializers
        used_names = {name for node in onnx_program.model.graph for name in node.inputs}
        used_names.update(output.name for output in onnx_program.model.graph.outputs)

        pruned = False
        for name in list(initializers):
            const_value = getattr(initializers[name], "const_value", None)
            raw_value = getattr(const_value, "raw", None)
            if isinstance(raw_value, FakeTensor) and name not in used_names:
                del initializers[name]
                pruned = True
        return pruned


class RenameWsubNodesTransform(BaseOnnxTransform):
    """Name local-function operators from their semantic layer parameters."""

    _LAYER_PARAMETER_RE = re.compile(r"(?:^|\.)layers\.\d+\.(?P<role>.+)\.weight$")
    _WEIGHTED_OPS = {"MatMul", "Gemm", "CustomRMSNorm"}

    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        transformed = False
        for function in model.functions:
            function_inputs = set(function.input)
            for node in function.node:
                if node.op_type not in cls._WEIGHTED_OPS:
                    continue
                for input_name in node.input[1:]:
                    if input_name not in function_inputs:
                        continue
                    match = cls._LAYER_PARAMETER_RE.search(input_name)
                    if match is None:
                        continue
                    role = match.group("role").replace(".", "/")
                    semantic_name = f"{role}/{node.op_type}"
                    if node.name != semantic_name:
                        node.name = semantic_name
                        transformed = True
                    break
        return transformed


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
            applied[CustomOpTransform] = CustomOpTransform.apply(
                model, onnx_export_opset=kwargs.get("onnx_export_opset", constants.ONNX_LEGACY_EXPORT_OPSET)
            )

        if RenameFunctionOutputsTransform in requested:
            applied[RenameFunctionOutputsTransform] = RenameFunctionOutputsTransform.apply(
                model, layer_idx=kwargs.get("layer_idx", 0)
            )

        if RenameWsubNodesTransform in requested:
            applied[RenameWsubNodesTransform] = RenameWsubNodesTransform.apply(model)

        if PreserveNestedCacheRetainedStateTransform in requested:
            applied[PreserveNestedCacheRetainedStateTransform] = PreserveNestedCacheRetainedStateTransform.apply(model)

        if RenameRepeatedSubgraphTransform in requested:
            applied[RenameRepeatedSubgraphTransform] = RenameRepeatedSubgraphTransform.apply(model, **kwargs)

        if AdapterWeightsToInputsTransform in requested:
            applied[AdapterWeightsToInputsTransform] = AdapterWeightsToInputsTransform.apply(model, **kwargs)

        if InlineTorchSubgraphFunctionsTransform in requested:
            applied[InlineTorchSubgraphFunctionsTransform] = InlineTorchSubgraphFunctionsTransform.apply(model, **kwargs)

        if StaticLoopInputsTransform in requested:
            applied[StaticLoopInputsTransform] = StaticLoopInputsTransform.apply(model, **kwargs)

        for t, done in applied.items():
            logger.info(f"Transform '{t.__name__}' applied={done}")

        return model, any(applied.values())
