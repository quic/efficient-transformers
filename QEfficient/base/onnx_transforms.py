# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import gc
import logging
import os
import warnings
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import torch
from onnx import ModelProto, TensorProto, external_data_helper, numpy_helper

from QEfficient.customop.ctx_scatter_gather import CtxGather, CtxGatherFunc, CtxScatter, CtxScatterFunc
from QEfficient.customop.rms_norm import CustomRMSNorm, CustomRMSNormFunc
from QEfficient.utils.constants import ONNX_TRANSFORM_MEMORY_CLEANUP_INTERVAL

logger = logging.getLogger(__name__)


class BaseOnnxTransform:
    """
    OnnxTransform is the base class for graph modifications on exported onnx.
    """

    _external_data_loaded_cache = {}  # Dict[int, bool]

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Use the `apply` method directly.")

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

    @classmethod
    def _check_external_data_loaded(cls, model: ModelProto) -> bool:
        """
        Check if external data is already loaded in the model.

        :param model: The ONNX model to check
        :returns: True if external data is already loaded, False otherwise
        """
        model_id = id(model)
        if model_id in cls._external_data_loaded_cache:
            return cls._external_data_loaded_cache[model_id]

        for tensor in external_data_helper._get_all_tensors(model):
            if len(tensor.external_data) > 0 and not tensor.HasField("raw_data"):
                cls._external_data_loaded_cache[model_id] = False
                return False

        cls._external_data_loaded_cache[model_id] = True
        return True

    @classmethod
    def _load_external_data(cls, model: ModelProto, onnx_base_dir: Optional[str] = None):
        """
        Performs a bulk load of external data if it's not already loaded.
        Updates the cache upon successful load.
        """
        model_id = id(model)
        if not cls._check_external_data_loaded(model):
            logger.info("External data not loaded. Performing bulk load.")
            external_data_helper.load_external_data_for_model(model, onnx_base_dir)
            cls._external_data_loaded_cache[model_id] = True
        else:
            logger.info("External data already loaded (or cached). Skipping bulk load.")

    @classmethod
    def _cleanup_memory(cls):
        """
        Force garbage collection to free up memory after tensor processing.
        """
        gc.collect()

    @classmethod
    def _cleanup_external_data_and_cache(cls, model: ModelProto):
        """
        Combines clearing external data from the model and its cache entry.
        """
        # Remove the loaded raw data from tensors
        for tensor in external_data_helper._get_all_tensors(model):
            if tensor.HasField("raw_data"):
                tensor.ClearField("raw_data")

        # Clear the cache entry for this model using its ID
        model_id = id(model)
        if model_id in cls._external_data_loaded_cache:
            del cls._external_data_loaded_cache[model_id]

        logger.info("External data and cache cleaned up.")


class OnnxTransform(BaseOnnxTransform):
    # Custom Ops Registry
    CUSTOM_OPS = {
        "CustomRMSNormFunc": (CustomRMSNormFunc, CustomRMSNorm),
        "CtxScatterFunc": (CtxScatterFunc, CtxScatter),
        "CtxGatherFunc": (CtxGatherFunc, CtxGather),
    }

    @classmethod
    def apply(
        cls,
        model: ModelProto,
        *,
        transforms: List[str],
        model_name: str = "",
        onnx_base_dir: Optional[str] = None,
        file_chunk_size: int = 10 * 2**30,
        size_threshold: int = 1024,
        opset_version: int = 17,
        **kwargs,
    ) -> Tuple[ModelProto, bool]:
        if len(transforms) == 0:
            warnings.warn("Transform list is empty. Skipping transformation.")
            return model, False

        try:
            cls._load_external_data(model, onnx_base_dir)
            tensors = external_data_helper._get_all_tensors(model)

            TensorInfo = namedtuple("TensorInfo", ["tensor", "tsize"])
            tensor_infos = [
                TensorInfo(tensor, len(tensor.raw_data) if tensor.HasField("raw_data") else 0) for tensor in tensors
            ]

            fp16_min, fp16_max = np.finfo(np.float16).min, np.finfo(np.float16).max
            file_num_tracker = {"num": 0, "size": 0}

            requested_transforms = set(transforms)
            applied_transforms = {name: False for name in requested_transforms}

            def process_tensor(index_info: Tuple[int, TensorInfo]) -> List[str]:
                idx, info = index_info
                tensor, tsize = info
                local_applied = []

                if "FP16ClipTransform" in requested_transforms:
                    if cls._clip_tensor(tensor, fp16_min, fp16_max):
                        local_applied.append("FP16ClipTransform")

                if "SplitTensorsTransform" in requested_transforms and tsize > size_threshold:
                    if file_num_tracker["size"] + tsize > file_chunk_size:
                        file_num_tracker["num"] += 1
                        file_num_tracker["size"] = tsize
                    else:
                        file_num_tracker["size"] += tsize

                    cls._split_tensor(tensor, model_name, file_num_tracker["num"])
                    local_applied.append("SplitTensorsTransform")

                if (idx + 1) % ONNX_TRANSFORM_MEMORY_CLEANUP_INTERVAL == 0:
                    cls._cleanup_memory()

                return local_applied

            with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
                results = list(executor.map(process_tensor, enumerate(tensor_infos)))

            for result in results:
                for transform_name in result:
                    applied_transforms[transform_name] = True

            if "CustomOpTransform" in requested_transforms:
                applied_transforms["CustomOpTransform"] = cls._custom_op_transform(model, opset_version)

            if "RenameFunctionOutputsTransform" in requested_transforms:
                applied_transforms["RenameFunctionOutputsTransform"] = cls._rename_function_outputs(model)

            for name in requested_transforms:
                if applied_transforms[name]:
                    logger.info(f"Transform '{name}' was applied.")
                else:
                    logger.warning(f"Transform '{name}' was requested but not applied.")

            return model, any(applied_transforms.values())

        finally:
            cls._cleanup_memory()

    @classmethod
    def _custom_op_transform(cls, model: ModelProto, opset_version) -> bool:
        op_applied = False
        for op_name, (func_class, _) in cls.CUSTOM_OPS.items():
            if hasattr(func_class, "symbolic"):
                torch.onnx.register_custom_op_symbolic(f"::{op_name}", func_class.symbolic, opset_version)

        existing_names = {func.name for func in model.functions}
        for _, onnxscript_func in cls.CUSTOM_OPS.values():
            proto = onnxscript_func.to_function_proto()
            if proto.name not in existing_names:
                model.functions.append(proto)
                op_applied = True

        return op_applied

    @classmethod
    def _rename_function_outputs(cls, model: ModelProto) -> bool:
        graph = model.graph
        op_type_to_func = {func.name: func for func in model.functions}
        decoder_patterns = ["DecoderLayer", "Block", "Layer"]
        renamed = False

        model_graph_outputs_map = {val.name: idx for idx, val in enumerate(model.graph.output)}

        layer_index = 0
        for node in graph.node:
            if any(p in node.name or p in node.op_type for p in decoder_patterns):
                func = op_type_to_func.get(node.op_type)
                if func is None:
                    continue

                for i, out_name in enumerate(func.output):
                    if "_InternalRetainedState" in out_name:
                        renamed = True
                        original = node.output[i]
                        if "key" in out_name:
                            new = f"past_key.{layer_index}_RetainedState"
                        elif "value" in out_name:
                            new = f"past_value.{layer_index}_RetainedState"
                        else:
                            continue
                        node.output[i] = new
                        if original in model_graph_outputs_map:
                            idx = model_graph_outputs_map[original]
                            model.graph.output[idx].name = new
                layer_index += 1

        return renamed

    @staticmethod
    def _clip_tensor(tensor, fp16_min, fp16_max) -> bool:
        if tensor.data_type != TensorProto.FLOAT:
            return False
        nptensor = numpy_helper.to_array(tensor)
        if np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min):
            neg_inf_mask = np.isinf(nptensor) & (nptensor < 0)
            clipped = np.clip(nptensor, fp16_min, fp16_max)
            if neg_inf_mask.any():
                clipped = np.where(neg_inf_mask, np.float32("-inf"), clipped)
            new_tensor = numpy_helper.from_array(clipped, tensor.name)
            tensor.CopyFrom(new_tensor)
            return True
        return False

    @staticmethod
    def _split_tensor(tensor, model_name: str, file_num: int) -> None:
        external_data_helper.set_external_data(tensor, f"{model_name}_{file_num}.onnx.data")
