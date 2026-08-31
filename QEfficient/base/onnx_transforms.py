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
from typing import Any

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
from QEfficient.customop.fp8_dequantize import (
    FP8DequantizeBlockedFunc,
    FP8DequantizePerAxis,
    FP8DequantizePerAxisFunc,
    FP8DequantizePerTensor,
    FP8DequantizePerTensorFunc,
)

# from QEfficient.customop.quantization_ops import CastToUInt4, CastToUInt4Func
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
        cls, tensor: TensorProto, model_name: str, file_num: int, mapping: dict[str, tuple[TensorProto, str]]
    ) -> None:
        file_name = f"{model_name}_{file_num}.onnx.data"
        mapping[tensor.name] = (tensor, file_name)


class CustomOpTransform(BaseOnnxTransform):
    """Register custom ONNX ops and append their function prototypes to the model."""

    _custom_ops: dict[str, tuple[Any, Any]] = {
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
        "FP8DequantizePerTensorFunc": (FP8DequantizePerTensorFunc, FP8DequantizePerTensor),
        "FP8DequantizePerAxisFunc": (FP8DequantizePerAxisFunc, FP8DequantizePerAxis),
        # FP8DequantizeBlockedFunc maps to None: the blocked onnxscript function is
        # block-size-specific and is injected at export time via
        # _build_blocked_translation_table(model) in modeling_qeff.py.
        "FP8DequantizeBlockedFunc": (FP8DequantizeBlockedFunc, None),
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
            if onnxscript_func is None:
                continue
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
    def _ensure_opset_imports(container: ModelProto | onnx.FunctionProto, domain: str, version: int) -> None:
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
        kv_rename_map: dict[str, str] = {}

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
    def _rename_graph_inputs_bulk(graph: onnx.GraphProto, rename_map: dict[str, str]) -> bool:
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
    def _rename_op_types(nodes, old_to_new: dict[str, str]) -> None:
        for node in RenameRepeatedSubgraphTransform._iter_all_nodes(nodes):
            if node.op_type in old_to_new:
                node.op_type = old_to_new[node.op_type]

    @classmethod
    def apply(cls, model: ModelProto, target_classnames: list[str] | None = None, **kwargs) -> bool:
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

        # Fix pkg.torch.__subgraph__ domain so ORT can inline the functions.
        target_class_modules = kwargs.get("target_class_modules")
        cls._fix_subgraph_domains(model, old_to_new, target_class_modules)

        return True

    @classmethod
    def _fix_subgraph_domains(
        cls,
        model: ModelProto,
        old_to_new: dict[str, str],
        target_class_modules: dict[str, str] | None,
    ) -> None:
        """Change pkg.torch.__subgraph__ domain to the class's Python module path.

        torch.export emits repeated subgraph functions in domain 'pkg.torch.__subgraph__'.
        ORT cannot execute functions in that domain.  After renaming the function to its
        QEff class name we also update the domain to the class's module path so ORT can
        inline and execute the function body.
        """
        if not target_class_modules:
            return
        _TORCH_SUBGRAPH_DOMAIN = "pkg.torch.__subgraph__"
        # Build new_name -> module mapping (old_to_new maps old_name -> new_name).
        # Renaming can append numeric suffixes for collisions (e.g. FooLayer_1),
        # while target_class_modules is keyed by base class names (FooLayer).
        new_name_to_module = {}
        for new_name in old_to_new.values():
            module = target_class_modules.get(new_name)
            if module is None:
                base_name = re.sub(r"_\d+$", "", new_name)
                module = target_class_modules.get(base_name)
            if module is not None:
                new_name_to_module[new_name] = module
        for fn in model.functions:
            if fn.domain == _TORCH_SUBGRAPH_DOMAIN and fn.name in new_name_to_module:
                fn.domain = new_name_to_module[fn.name]
                if not any(op.domain == fn.domain for op in model.opset_import):
                    model.opset_import.append(onnx.helper.make_opsetid(fn.domain, 1))
        # Remove FP8 value_info entries from the main graph.  ORT uses graph-level value_info to validate call-site input types for local functions and rejects float8e4m3fn even when the function body handles it correctly.  Removing these annotations lets ORT infer the type from the initializer at runtime.
        _fp8_type = 17  # TensorProto.FLOAT8E4M3FN
        _fp8_vi_names = {vi.name for vi in model.graph.value_info if vi.type.tensor_type.elem_type == _fp8_type}
        if _fp8_vi_names:
            _kept_vi = [vi for vi in model.graph.value_info if vi.name not in _fp8_vi_names]
            del model.graph.value_info[:]
            model.graph.value_info.extend(_kept_vi)
        # Update call-site node domains in the graph and all functions
        for node in cls._iter_all_nodes(model.graph.node):
            if node.domain == _TORCH_SUBGRAPH_DOMAIN and node.op_type in new_name_to_module:
                node.domain = new_name_to_module[node.op_type]
        for fn in model.functions:
            for node in cls._iter_all_nodes(fn.node):
                if node.domain == _TORCH_SUBGRAPH_DOMAIN and node.op_type in new_name_to_module:
                    node.domain = new_name_to_module[node.op_type]

        # Drop pkg.torch.__subgraph__ import only when all usage has been remapped.
        # Otherwise keep (or restore) the import so ONNX remains valid.
        has_pkg_usage = any(fn.domain == _TORCH_SUBGRAPH_DOMAIN for fn in model.functions)
        if not has_pkg_usage:
            has_pkg_usage = any(node.domain == _TORCH_SUBGRAPH_DOMAIN for node in cls._iter_all_nodes(model.graph.node))
        if not has_pkg_usage:
            for fn in model.functions:
                if any(node.domain == _TORCH_SUBGRAPH_DOMAIN for node in cls._iter_all_nodes(fn.node)):
                    has_pkg_usage = True
                    break

        if has_pkg_usage:
            if not any(op.domain == _TORCH_SUBGRAPH_DOMAIN for op in model.opset_import):
                model.opset_import.append(onnx.helper.make_opsetid(_TORCH_SUBGRAPH_DOMAIN, 1))
            return

        kept = [op for op in model.opset_import if op.domain != _TORCH_SUBGRAPH_DOMAIN]
        del model.opset_import[:]
        model.opset_import.extend(kept)


class AdapterWeightsToInputsTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model: onnx.ModelProto, *, adapter_name: str, **kwargs) -> tuple[onnx.ModelProto, bool]:
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


class HoistFP8DequantFromSubfunctionTransform(BaseOnnxTransform):
    """Hoist FP8 dequantization nodes out of local subfunctions to the main graph.

    ORT 1.22.0 cannot execute FP8 tensors passed as inputs through local function
    boundaries.  This transform moves FP8DequantizeBlocked / FP8DequantizePerAxis /
    FP8DequantizePerTensor nodes from inside the decoder subfunction to the main
    graph.

    Strategy: for each FP8DequantizeX node inside the function:
    1. Create an equivalent node in the main graph using the actual initializer names.
    2. Replace the FP8 initializer input in the call node with the dequant output.
       The scale input is replaced with an empty string (unused after hoisting).
    3. Remove the dequant node from the function body.
    4. Update the function's value_info: change the FP8 arg type to FLOAT and
       add FLOAT type info for the dequant output name.

    Crucially, we do NOT change the number of function inputs or their positions —
    the function body uses positional arg names (arg0_1, arg1_1, ...) and any
    reordering would break all downstream nodes.
    """

    _FP8_DEQUANT_OPS = frozenset({"FP8DequantizeBlocked", "FP8DequantizePerAxis", "FP8DequantizePerTensor"})
    _FP8_TYPE = 17  # TensorProto.FLOAT8E4M3FN

    @classmethod
    def apply(cls, model: ModelProto) -> bool:
        fp8_init_names = {i.name for i in model.graph.initializer if i.data_type == cls._FP8_TYPE}
        if not fp8_init_names:
            return False

        changed = False
        for fn in model.functions:
            fp8_dequant_nodes = [n for n in fn.node if n.op_type in cls._FP8_DEQUANT_OPS]
            if not fp8_dequant_nodes:
                continue

            fp8_arg_names = {vi.name for vi in fn.value_info if vi.type.tensor_type.elem_type == cls._FP8_TYPE}
            if not fp8_arg_names:
                continue

            original_fn_inputs = list(fn.input)
            arg_to_idx = {arg_name: idx for idx, arg_name in enumerate(original_fn_inputs)}
            fp8_arg_to_dq_output = {}
            fn_changed = False

            for call_node in model.graph.node:
                if call_node.op_type != fn.name:
                    continue

                call_inputs = list(call_node.input)
                arg_to_actual = {
                    original_fn_inputs[i]: call_inputs[i] for i in range(min(len(original_fn_inputs), len(call_inputs)))
                }

                # Map: fp8_arg_idx -> actual_dq_output_name (in main graph)
                fp8_idx_to_actual_dq = {}
                scale_arg_indices = set()

                for dq_node in fp8_dequant_nodes:
                    fp8_arg = dq_node.input[0]
                    if fp8_arg not in fp8_arg_names:
                        continue

                    fp8_arg_idx = arg_to_idx.get(fp8_arg)
                    if fp8_arg_idx is None:
                        continue

                    actual_fp8_name = arg_to_actual.get(fp8_arg)
                    if actual_fp8_name not in fp8_init_names:
                        continue

                    scale_arg = dq_node.input[1] if len(dq_node.input) > 1 else None
                    actual_scale_name = arg_to_actual.get(scale_arg) if scale_arg else None

                    # Dequant output name inside the function body
                    dq_fn_output = dq_node.output[0]
                    # Hoisted dequant output name in the main graph
                    actual_dq_output = f"_hoisted_dequant_{actual_fp8_name}"

                    # Create the dequant node in the main graph
                    new_node_inputs = [actual_fp8_name]
                    if actual_scale_name:
                        new_node_inputs.append(actual_scale_name)
                    for extra_inp in dq_node.input[2:]:
                        new_node_inputs.append(arg_to_actual.get(extra_inp, extra_inp))

                    new_dq_node = onnx.helper.make_node(
                        dq_node.op_type,
                        inputs=new_node_inputs,
                        outputs=[actual_dq_output],
                        domain=dq_node.domain,
                        name=f"hoisted_{actual_fp8_name}",
                    )
                    for attr in dq_node.attribute:
                        new_dq_node.attribute.append(attr)

                    call_idx = list(model.graph.node).index(call_node)
                    model.graph.node.insert(call_idx, new_dq_node)

                    fp8_arg_to_dq_output[fp8_arg] = dq_fn_output
                    fp8_idx_to_actual_dq[fp8_arg_idx] = actual_dq_output
                    if scale_arg:
                        scale_arg_idx = arg_to_idx.get(scale_arg)
                        if scale_arg_idx is not None:
                            scale_arg_indices.add(scale_arg_idx)

                    changed = True
                    fn_changed = True

                if not fp8_idx_to_actual_dq:
                    continue

                # Replace FP8 and scale inputs in this call-site only.
                for fp8_arg_idx, actual_dq_output in fp8_idx_to_actual_dq.items():
                    call_node.input[fp8_arg_idx] = actual_dq_output
                for scale_idx in scale_arg_indices:
                    call_node.input[scale_idx] = ""

            if not fn_changed:
                continue

            # Remove dequant nodes from function body once after all call-sites are updated.
            for n in fp8_dequant_nodes:
                if n in fn.node:
                    fn.node.remove(n)

            # Rename function inputs from FP8 arg names to dequant output names.
            for fp8_arg, dq_fn_output in fp8_arg_to_dq_output.items():
                fp8_arg_idx = arg_to_idx.get(fp8_arg)
                if fp8_arg_idx is not None:
                    fn.input[fp8_arg_idx] = dq_fn_output

            # Update function value_info:
            # - Remove FP8 type annotations (those args now receive float)
            # - Add FLOAT type info for the dequant output names
            for vi in fn.value_info:
                if vi.name in fp8_arg_names:
                    vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
                    # Also rename the value_info entry to the dequant output name
                    vi.name = fp8_arg_to_dq_output.get(vi.name, vi.name)
            for dq_fn_output in fp8_arg_to_dq_output.values():
                # Only add if not already present (renamed above)
                if not any(vi.name == dq_fn_output for vi in fn.value_info):
                    float_vi = onnx.helper.make_tensor_value_info(dq_fn_output, onnx.TensorProto.FLOAT, None)
                    fn.value_info.append(float_vi)

        return changed


class OnnxTransformPipeline(BaseOnnxTransform):
    """Pipeline to apply multiple ONNX transformations in sequence."""

    def __init__(self, transforms: list[type[BaseOnnxTransform]]):
        self.transforms = transforms

    def apply(
        self,
        model: ModelProto,
        *,
        model_name: str = "",
        onnx_base_dir: str | None = None,
        file_chunk_size: int = FILE_CHUNK_SIZE_DEFAULT,
        size_threshold: int = SIZE_THRESHOLD_DEFAULT,
        **kwargs,
    ) -> tuple[ModelProto, bool]:
        if not self.transforms:
            return model, False

        # Same logic as before, but replace `transforms` with `self.transforms`
        mapping: dict[str, tuple[TensorProto, str]] = {}
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

        if HoistFP8DequantFromSubfunctionTransform in requested:
            applied[HoistFP8DequantFromSubfunctionTransform] = HoistFP8DequantFromSubfunctionTransform.apply(model)

        for t, done in applied.items():
            logger.info(f"Transform '{t.__name__}' applied={done}")

        return model, any(applied.values())
