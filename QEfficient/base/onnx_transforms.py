# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from onnx import ModelProto, external_data_helper, numpy_helper


class OnnxTransform:
    """
    OnnxTransform is the base class for graph modifications on exported onnx.
    """

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Directly use the `apply` method.")

    @classmethod
    def apply(cls, model: ModelProto, **kwargs) -> Tuple[ModelProto, bool]:
        """
        Override this class to apply a transformation.
        :param model: The model's ONNX graph to transform
        :param kwargs: Parameters needed for specific transforms. All transforms should take **kwargs to ignore unneeded kwargs.

        :returns: ONNX graph after applying the transform
        :returns: Boolean indicating whether transform was applied
        """
        raise NotImplementedError("Use subclasses for ONNX transform")


class FP16ClipTransform(OnnxTransform):
    """
    Clips the tensor values to be in FP16 range, but preserves -inf values.
    """

    @classmethod
    def apply(cls, model: ModelProto, *, onnx_base_dir: Optional[str] = None, **kwargs) -> Tuple[ModelProto, bool]:
        """
        :param onnx_base_dir: Base directory to load tensors
        """
        finfo = np.finfo(np.float16)
        fp16_max = finfo.max
        fp16_min = finfo.min
        transformed = False

        for tensor in external_data_helper._get_all_tensors(model):
            nptensor = numpy_helper.to_array(tensor, onnx_base_dir)
            if nptensor.dtype == np.float32 and (np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)):
                neg_inf_mask = np.isinf(nptensor) & (nptensor < 0)
                clipped_tensor = np.clip(nptensor, fp16_min, fp16_max)

                # Restore -inf values
                if neg_inf_mask.any():
                    clipped_tensor = np.where(neg_inf_mask, np.float32("-inf"), clipped_tensor)

                new_tensor = numpy_helper.from_array(clipped_tensor, tensor.name)
                tensor.CopyFrom(new_tensor)
                transformed = True

        return model, transformed


class SplitTensorsTransform(OnnxTransform):
    """
    Split external tensors file
    """

    @classmethod
    def apply(
        cls,
        model: ModelProto,
        *,
        model_name: str,
        onnx_base_dir: Optional[str] = None,
        file_chunk_size: int = 10 * 2**30,  # 10 GiB
        size_threshold: int = 1024,
        **kwargs,
    ) -> Tuple[ModelProto, bool]:
        """
        :param model_name: Used for naming external files. i.e. {model_name}_0.onnx.data
        :param onnx_base_dir: Base directory to load tensors (if not already loaded).
        :param file_chunk_size: Chunk size to split external files into.
        :param size_threshold: Only tensors greater than this threshold (in bytes) will be saved externally.
        """
        file_num = 0
        current_file_size = 0
        transformed = False
        external_data_helper.load_external_data_for_model(model, onnx_base_dir)
        for tensor in external_data_helper._get_all_tensors(model):
            if tensor.HasField("raw_data") and ((tsize := len(tensor.raw_data)) > size_threshold):
                transformed = True
                current_file_size += tsize
                if current_file_size > file_chunk_size:
                    file_num += 1
                    current_file_size = tsize
                external_data_helper.set_external_data(tensor, f"{model_name}_{file_num}.onnx.data")
        return model, transformed


class CustomOpTransform(OnnxTransform):
    """
    Transform to register custom operations and add their function protos to the ONNX model.
    """

    # Registry of custom operations: op_name -> (func_class, onnxscript_func)
    _custom_ops: Dict[str, Tuple[Any, Any]] = {}

    @classmethod
    def register_custom_op(cls, op_name: str, func_class: Any, onnxscript_func: Any) -> None:
        """Register a custom operation."""
        cls._custom_ops[op_name] = (func_class, onnxscript_func)

    @classmethod
    def apply(cls, model: ModelProto, *, opset_version: int = 17, **kwargs) -> Tuple[ModelProto, bool]:
        """
        Apply custom op registration and add function protos to the model.

        :param model: The ONNX model to transform.
        :param opset_version: ONNX opset version for symbolic registration.
        :returns: (Transformed model, success flag).
        """
        transformed = False

        # Register all custom op symbolic functions with torch.onnx
        for op_name, (func_class, _) in cls._custom_ops.items():
            if hasattr(func_class, "symbolic"):
                torch.onnx.register_custom_op_symbolic(f"::{op_name}", func_class.symbolic, opset_version)

        # Gather function names and all nodes (graph + function nodes)
        func_names: Set[str] = {func.name for func in model.functions}
        all_nodes = list(model.graph.node)
        for func in model.functions:
            all_nodes.extend(func.node)

        # Collect used op types
        used_op_types: Set[str] = {node.op_type for node in all_nodes}

        # Precompute heuristic flags
        has_rmsnorm = any("RMSNorm" in op_type for op_type in used_op_types)
        has_ctx_ops = any(op_type in ["Gather", "GatherND", "Scatter", "ScatterND"] for op_type in used_op_types)

        # Get function protos for custom ops used in the model
        used_protos = cls._get_function_protos_for_model(used_op_types, has_rmsnorm, has_ctx_ops)

        # Append new function protos if not already present
        for proto in used_protos:
            if proto.name not in func_names:
                model.functions.append(proto)
                transformed = True

        return model, transformed

    @classmethod
    def _get_function_protos_for_model(cls, used_op_types: Set[str], has_rmsnorm: bool, has_ctx_ops: bool) -> List[Any]:
        """
        Get function protos for custom ops that are actually used in the model.

        :param used_op_types: Set of op types used in the model.
        :param has_rmsnorm: Flag indicating if RMSNorm-related ops are present.
        :param has_ctx_ops: Flag indicating if context-related ops are present.
        :returns: List of ONNX function protos.
        """
        used_protos: List[Any] = []
        for op_name, (_, onnxscript_func) in cls._custom_ops.items():
            if cls._is_custom_op_used(op_name, used_op_types, has_rmsnorm, has_ctx_ops):
                used_protos.append(onnxscript_func.to_function_proto())
        return used_protos

    @classmethod
    def _is_custom_op_used(cls, op_name: str, used_op_types: Set[str], has_rmsnorm: bool, has_ctx_ops: bool) -> bool:
        """
        Check if a custom op is used in the model.

        :param op_name: Name of the custom op.
        :param used_op_types: Set of op types used in the model.
        :param has_rmsnorm: Precomputed RMSNorm presence flag.
        :param has_ctx_ops: Precomputed context ops presence flag.
        :returns: True if the custom op is used, False otherwise.
        """
        if op_name in used_op_types:
            return True

        # Check for domain-specific ops
        if f"com.qti.aisw.onnx::{op_name.replace('Func', '')}" in used_op_types:
            return True

        # Heuristic checks
        if "RMSNorm" in op_name and has_rmsnorm:
            return True

        if "Ctx" in op_name and has_ctx_ops:
            return True

        return False


class RenameFunctionOutputsTransform(OnnxTransform):
    """
    Renames function outputs in decoder layers by removing 'Internal' from '_InternalRetainedState' patterns.
    """

    @classmethod
    def apply(cls, model: ModelProto, **kwargs) -> Tuple[ModelProto, bool]:
        """
        Rename function outputs in decoder layer nodes.

        :param model: The ONNX model to transform
        :returns: Transformed model and boolean indicating whether transform was applied
        """
        graph = model.graph
        op_type_to_func_map = {func.name: func for func in model.functions}
        decoder_layer_patterns = ["DecoderLayer", "Block", "Layer"]
        transformed = False

        # Create a dict mapping output name to its index for quick lookup
        model_graph_outputs_map = {val.name: idx for idx, val in enumerate(model.graph.output)}

        layer_index = 0
        for node in graph.node:
            if any(pattern in node.name or pattern in node.op_type for pattern in decoder_layer_patterns):
                func = op_type_to_func_map.get(node.op_type)
                if func is None:
                    continue

                for i, out_name in enumerate(func.output):
                    if "_InternalRetainedState" in out_name:
                        transformed = True
                        original_output_name = node.output[i]

                        # Generate new name based on key/value
                        if "key" in out_name:
                            new_name = f"past_key.{layer_index}_RetainedState"
                        elif "value" in out_name:
                            new_name = f"past_value.{layer_index}_RetainedState"
                        node.output[i] = new_name

                        # Update graph output name if it exists
                        if original_output_name in model_graph_outputs_map:
                            idx = model_graph_outputs_map[original_output_name]
                            model.graph.output[idx].name = new_name
                layer_index += 1
        return model, transformed
