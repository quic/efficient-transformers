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
from typing import Any, Dict, List, Optional, Tuple, Type, Union

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

        if PreserveNestedCacheRetainedStateTransform in requested:
            applied[PreserveNestedCacheRetainedStateTransform] = PreserveNestedCacheRetainedStateTransform.apply(model)

        if RenameRepeatedSubgraphTransform in requested:
            applied[RenameRepeatedSubgraphTransform] = RenameRepeatedSubgraphTransform.apply(model, **kwargs)

        if AdapterWeightsToInputsTransform in requested:
            applied[AdapterWeightsToInputsTransform] = AdapterWeightsToInputsTransform.apply(model, **kwargs)

        for t, done in applied.items():
            logger.info(f"Transform '{t.__name__}' applied={done}")

        return model, any(applied.values())


class RenameWsubTransform(BaseOnnxTransform):
    """Rename opaque w_sub ONNX weights and nodes to semantic HF names.

    Uses 256-byte fingerprint matching between ONNX external weight data and
    HF safetensors to build a rename map, then applies weight rename + node rename.

    Handles both individual external files (onnx__MatMul_NNN) and consolidated
    data files (model.onnx.data with offset+length metadata).
    """

    FINGERPRINT_BYTES = 256

    # Regex: model.layers.N.<role>.weight -> extract <role> for node rename
    _SEMANTIC_PARAM_RE = re.compile(
        r"model\.layers\.\d+\.((?:self_attn|mlp)[\w.]+?)(?:\.weight)?$"
    )
    _NORM_ROLE_RE = re.compile(
        r"model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight$"
    )

    @classmethod
    def apply(
        cls,
        model: ModelProto,
        *,
        hf_model_path: str,
        onnx_base_dir: str,
        **kwargs,
    ) -> Tuple[ModelProto, bool]:
        """Apply weight + node rename to a w_sub ONNX model.

        Args:
            model: ONNX ModelProto (loaded with load_external_data=False).
            hf_model_path: Path to HF model dir containing safetensors.
            onnx_base_dir: Directory containing ONNX external data files.

        Returns:
            (model, transformed) tuple.
        """
        if not hf_model_path or not os.path.isdir(hf_model_path):
            logger.warning(f"RenameWsubTransform: hf_model_path not found: {hf_model_path}")
            return model, False

        if not onnx_base_dir or not os.path.isdir(onnx_base_dir):
            logger.warning(f"RenameWsubTransform: onnx_base_dir not found: {onnx_base_dir}")
            return model, False

        # Step 1: Collect opaque initializer info
        opaque_infos = cls._get_opaque_initializer_info(model, onnx_base_dir)
        if not opaque_infos:
            logger.info("RenameWsubTransform: no opaque initializers found, skipping")
            return model, False

        # Step 2: Detect ONNX weight dtype
        onnx_dtype_code = opaque_infos[0]["data_type"]
        if onnx_dtype_code == onnx.TensorProto.FLOAT:
            target_np_dtype = np.float32
        elif onnx_dtype_code == onnx.TensorProto.FLOAT16:
            target_np_dtype = np.float16
        else:
            logger.warning(f"RenameWsubTransform: unsupported ONNX dtype code {onnx_dtype_code}")
            return model, False

        # Step 3: Build HF fingerprint map
        hf_fp_map = cls._build_hf_fingerprint_map(hf_model_path, target_np_dtype)
        if not hf_fp_map:
            logger.warning("RenameWsubTransform: no HF fingerprints built")
            return model, False

        # Step 4: Match ONNX fingerprints to HF
        rename_map: Dict[str, str] = {}
        for info in opaque_infos:
            onnx_fp = cls._read_onnx_fingerprint(info)
            if onnx_fp is None:
                continue
            hf_param = hf_fp_map.get(onnx_fp)
            if hf_param:
                clean_name = hf_param
                for prefix in ("language_model.",):
                    if clean_name.startswith(prefix):
                        clean_name = clean_name[len(prefix):]
                rename_map[info["name"]] = clean_name

        if not rename_map:
            logger.warning("RenameWsubTransform: no fingerprint matches found")
            return model, False

        logger.info(f"RenameWsubTransform: matched {len(rename_map)}/{len(opaque_infos)} weights")

        # Step 5: Apply weight rename + node rename
        weight_count, node_count = cls._apply_weight_and_node_rename(model, rename_map)
        logger.info(f"RenameWsubTransform: renamed {weight_count} weights, {node_count} nodes")

        # Step 6: Stamp metadata
        model.metadata_props.append(
            onnx.StringStringEntryProto(key="qeff_weight_names", value="semantic_hf")
        )
        if node_count > 0:
            model.metadata_props.append(
                onnx.StringStringEntryProto(key="qeff_node_names", value="semantic_role")
            )

        return model, True

    @classmethod
    def _get_opaque_initializer_info(cls, model: ModelProto, onnx_base_dir: str) -> List[Dict[str, Any]]:
        """Extract external data info for opaque initializers."""
        results = []
        for init in model.graph.initializer:
            if not ("onnx::MatMul_" in init.name or "onnx::Add_" in init.name):
                continue
            ext = {}
            for entry in init.external_data:
                ext[entry.key] = entry.value
            location = ext.get("location", "")
            offset = int(ext.get("offset", 0))
            length = int(ext.get("length", 0))
            if location:
                filepath = os.path.join(onnx_base_dir, location)
            else:
                bare_name = init.name.replace("::", "__")
                filepath = os.path.join(onnx_base_dir, bare_name)
                offset = 0
            results.append({
                "name": init.name,
                "filepath": filepath,
                "offset": offset,
                "length": length,
                "data_type": init.data_type,
            })
        return results

    @classmethod
    def _read_onnx_fingerprint(cls, info: Dict[str, Any]) -> Optional[bytes]:
        """Read first FINGERPRINT_BYTES from an ONNX external weight."""
        if not os.path.exists(info["filepath"]):
            return None
        read_len = min(cls.FINGERPRINT_BYTES, info["length"]) if info["length"] > 0 else cls.FINGERPRINT_BYTES
        try:
            with open(info["filepath"], "rb") as fh:
                fh.seek(info["offset"])
                data = fh.read(read_len)
            return data if len(data) == read_len else None
        except OSError:
            return None

    @classmethod
    def _build_hf_fingerprint_map(cls, hf_model_path: str, target_dtype) -> Dict[bytes, str]:
        """Build {fingerprint_bytes -> hf_param_name} from HF safetensors."""
        import json as _json
        import math as _math
        import struct as _struct

        fp_map: Dict[bytes, str] = {}
        index_path = os.path.join(hf_model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = _json.load(f)["weight_map"]
        else:
            single = os.path.join(hf_model_path, "model.safetensors")
            if not os.path.exists(single):
                return fp_map
            with open(single, "rb") as f:
                hdr_size = _struct.unpack("<Q", f.read(8))[0]
                hdr = _json.loads(f.read(hdr_size))
            weight_map = {k: "model.safetensors" for k in hdr if k != "__metadata__"}

        target_params = [
            k for k in weight_map
            if k.endswith(".weight")
            and "layers." in k
            and any(sub in k for sub in (
                "proj", "gate_proj", "up_proj", "down_proj", "router", "mlp.gate."
            ))
            and "mlp.experts." not in k
        ]

        for param_name in target_params:
            shard_path = os.path.join(hf_model_path, weight_map[param_name])
            if not os.path.exists(shard_path):
                continue
            fp = cls._read_hf_fingerprint(shard_path, param_name, target_dtype)
            if fp:
                fp_map[fp] = param_name

        return fp_map

    @classmethod
    def _read_hf_fingerprint(cls, shard_path: str, param_name: str, target_dtype) -> Optional[bytes]:
        """Read, transpose, dtype-convert a HF weight and return fingerprint bytes."""
        import json as _json
        import math as _math
        import struct as _struct

        target_bytes_per_elem = 4 if target_dtype == np.float32 else 2
        try:
            with open(shard_path, "rb") as f:
                hdr_size = _struct.unpack("<Q", f.read(8))[0]
                hdr = _json.loads(f.read(hdr_size))
                if param_name not in hdr:
                    return None
                info = hdr[param_name]
                hf_dtype = info["dtype"]
                shape = info["shape"]
                start = info["data_offsets"][0]
                if len(shape) < 2:
                    return None
                out_features, in_features = shape[0], shape[1]
                bytes_per_elem = {"BF16": 2, "F16": 2, "F32": 4}.get(hf_dtype, 0)
                if bytes_per_elem == 0:
                    return None
                n_rows = min(
                    out_features,
                    _math.ceil(cls.FINGERPRINT_BYTES / target_bytes_per_elem),
                )
                n_rows = max(n_rows, 1)
                read_bytes = n_rows * in_features * bytes_per_elem
                f.seek(8 + hdr_size + start)
                raw = f.read(read_bytes)
        except (OSError, KeyError, ValueError):
            return None

        if len(raw) < n_rows * in_features * bytes_per_elem:
            return None

        arr = np.frombuffer(raw, dtype=np.uint16 if hf_dtype in ("BF16", "F16") else np.float32)
        if hf_dtype == "BF16":
            fp32 = (arr.astype(np.uint32) << 16).view(np.float32)
        elif hf_dtype == "F16":
            fp32 = arr.view(np.float16).astype(np.float32)
        else:
            fp32 = arr.copy()

        fp32 = fp32.reshape(n_rows, in_features).T.copy()
        if target_dtype == np.float16:
            result = fp32.astype(np.float16)
        else:
            result = fp32
        return result.tobytes()[:cls.FINGERPRINT_BYTES]

    @classmethod
    def _get_node_role(cls, weight_name: str) -> Optional[str]:
        """Extract semantic role from a renamed weight name."""
        match = cls._SEMANTIC_PARAM_RE.match(weight_name)
        if match:
            return match.group(1).replace(".", "/")
        return None

    @classmethod
    def _apply_weight_and_node_rename(
        cls, model: ModelProto, rename_map: Dict[str, str]
    ) -> Tuple[int, int]:
        """Apply rename_map throughout the model and rename anchored nodes."""
        weight_count = 0
        node_count = 0

        # 1. Rename graph initializers
        for init in model.graph.initializer:
            if init.name in rename_map:
                init.name = rename_map[init.name]
                weight_count += 1

        # 2. Rename graph inputs
        for inp in model.graph.input:
            if inp.name in rename_map:
                inp.name = rename_map[inp.name]

        # 3. Rename main graph node inputs (call-site bindings)
        for node in model.graph.node:
            for i, inp_name in enumerate(node.input):
                if inp_name in rename_map:
                    node.input[i] = rename_map[inp_name]

        # 4. Rename function formals + internal node inputs
        for fn in model.functions:
            for i, inp_name in enumerate(fn.input):
                if inp_name in rename_map:
                    fn.input[i] = rename_map[inp_name]
            for node in fn.node:
                for i, inp_name in enumerate(node.input):
                    if inp_name in rename_map:
                        node.input[i] = rename_map[inp_name]

        # 5. Node rename: MatMul/Gemm nodes using renamed weight as input[1]
        for fn in model.functions:
            fn_input_set = set(fn.input)
            for node in fn.node:
                if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2:
                    weight_inp = node.input[1]
                    if weight_inp in fn_input_set:
                        role = cls._get_node_role(weight_inp)
                        if role:
                            node.name = role + "/" + node.op_type
                            node_count += 1

        # 6. Node rename: CustomRMSNorm nodes using layernorm weight
        for fn in model.functions:
            fn_input_set = set(fn.input)
            for node in fn.node:
                if node.op_type == "CustomRMSNorm" and len(node.input) >= 2:
                    weight_inp = node.input[1]
                    if weight_inp in fn_input_set:
                        match = cls._NORM_ROLE_RE.match(weight_inp)
                        if match:
                            node.name = match.group(1) + "/CustomRMSNorm"
                            node_count += 1

        return weight_count, node_count
