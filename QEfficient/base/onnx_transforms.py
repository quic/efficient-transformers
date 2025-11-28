# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import gc
import logging
import warnings
from typing import List, Optional, Tuple, Type

import numpy as np
import torch
from onnx import ModelProto, TensorProto, external_data_helper, numpy_helper

from QEfficient.customop.ctx_scatter_gather import (
    CtxGather,
    CtxGatherFunc,
    CtxScatter,
    CtxScatterFunc,
)
from QEfficient.customop.rms_norm import CustomRMSNorm, CustomRMSNormFunc
from QEfficient.utils.constants import ONNX_TRANSFORM_MEMORY_CLEANUP_INTERVAL

logger = logging.getLogger(__name__)


class BaseOnnxTransform:
    """Base class for ONNX graph modifications. Should NOT be instantiated."""

    _external_data_loaded_cache = {}

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Use the `apply` method directly.")

    @classmethod
    def apply(cls, model: ModelProto, **kwargs) -> Tuple[ModelProto, bool]:
        raise NotImplementedError("Use subclasses for ONNX transform")

    @classmethod
    def _check_external_data_loaded(cls, model: ModelProto) -> bool:
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
        model_id = id(model)
        if not cls._check_external_data_loaded(model):
            logger.info("External data not loaded. Performing bulk load.")
            external_data_helper.load_external_data_for_model(model, onnx_base_dir)
            cls._external_data_loaded_cache[model_id] = True
        else:
            logger.info("External data already loaded (or cached). Skipping bulk load.")

    @classmethod
    def _cleanup_memory(cls):
        gc.collect()

    @classmethod
    def _cleanup_external_data_and_cache(cls, model: ModelProto):
        for tensor in external_data_helper._get_all_tensors(model):
            if tensor.HasField("raw_data"):
                tensor.ClearField("raw_data")
        model_id = id(model)
        if model_id in cls._external_data_loaded_cache:
            del cls._external_data_loaded_cache[model_id]
        logger.info("External data and cache cleaned up.")


class FP16ClipTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model, **kwargs):
        pass


class SplitTensorsTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model, **kwargs):
        pass


class CustomOpTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model, **kwargs):
        pass


class RenameFunctionOutputsTransform(BaseOnnxTransform):
    @classmethod
    def apply(cls, model, **kwargs):
        pass


class OnnxTransformPipeline(BaseOnnxTransform):
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
        transforms: List[Type[BaseOnnxTransform]],
        model_name: str = "",
        onnx_base_dir: Optional[str] = None,
        file_chunk_size: int = 10 * 2**30,
        size_threshold: int = 1024,
        opset_version: int = 17,
        **kwargs,
    ) -> Tuple[ModelProto, bool]:
        if not transforms:
            warnings.warn("Transform list is empty. Skipping transformation.")
            return model, False

        try:
            cls._load_external_data(model, onnx_base_dir)

            tensors = external_data_helper._get_all_tensors(model)

            requested = set(transforms)
            applied = {t: False for t in requested}
            do_fp16 = FP16ClipTransform in requested
            do_split = SplitTensorsTransform in requested
            fp16_min, fp16_max = np.finfo(np.float16).min, np.finfo(np.float16).max
            file_num_tracker = {"num": 0, "size": 0}

            # fp_16
            if do_fp16 or do_split:
                for idx, tensor in enumerate(tensors):
                    if do_fp16 and cls._clip_tensor(tensor, fp16_min, fp16_max):
                        applied[FP16ClipTransform] = True

                    # Split Tensor
                    if do_split and tensor.HasField("raw_data"):
                        tsize = len(tensor.raw_data)
                        if tsize > size_threshold:
                            if file_num_tracker["size"] + tsize > file_chunk_size:
                                file_num_tracker["num"] += 1
                                file_num_tracker["size"] = tsize
                            else:
                                file_num_tracker["size"] += tsize

                            cls._split_tensor(tensor, model_name, file_num_tracker["num"])
                            applied[SplitTensorsTransform] = True

                    if (idx + 1) % ONNX_TRANSFORM_MEMORY_CLEANUP_INTERVAL == 0:
                        cls._cleanup_memory()

            if CustomOpTransform in requested:
                applied[CustomOpTransform] = cls._custom_op_transform(model, opset_version)

            if RenameFunctionOutputsTransform in requested:
                applied[RenameFunctionOutputsTransform] = cls._rename_function_outputs(model)

            # Logging
            for t, done in applied.items():
                logger.info(f"Transform '{t.__name__}' applied={done}")

            return model, any(applied.values())

        finally:
            # pass
            cls._cleanup_memory()

    @classmethod
    def _custom_op_transform(cls, model: ModelProto, opset_version) -> bool:
        op_applied = False
        for op_name, (func_class, _) in cls.CUSTOM_OPS.items():
            if hasattr(func_class, "symbolic"):
                torch.onnx.register_custom_op_symbolic(f"::{op_name}", func_class.symbolic, opset_version)
        existing = {f.name for f in model.functions}
        for _, onnxscript_func in cls.CUSTOM_OPS.values():
            proto = onnxscript_func.to_function_proto()
            if proto.name not in existing:
                model.functions.append(proto)
                op_applied = True
        return op_applied

    @classmethod
    def _rename_function_outputs(cls, model: ModelProto) -> bool:
        graph = model.graph
        op_type_to_func = {f.name: f for f in model.functions}
        decoder_patterns = ["DecoderLayer", "Block", "Layer"]
        renamed = False
        model_out_map = {v.name: i for i, v in enumerate(graph.output)}
        layer_idx = 0
        for node in graph.node:
            if any(p in node.name or p in node.op_type for p in decoder_patterns):
                func = op_type_to_func.get(node.op_type)
                if func is None:
                    continue
                for i, out_name in enumerate(func.output):
                    if "_InternalRetainedState" in out_name:
                        renamed = True
                        orig = node.output[i]
                        if "key" in out_name:
                            new = f"past_key.{layer_idx}_RetainedState"
                        elif "value" in out_name:
                            new = f"past_value.{layer_idx}_RetainedState"
                        else:
                            continue
                        node.output[i] = new
                        if orig in model_out_map:
                            graph.output[model_out_map[orig]].name = new
                layer_idx += 1
        return renamed

    @staticmethod
    def _clip_tensor(tensor, fp16_min, fp16_max) -> bool:
        if tensor.data_type != TensorProto.FLOAT:
            return False
        arr = numpy_helper.to_array(tensor)
        if not (np.any(arr > fp16_max) or np.any(arr < fp16_min)):
            return False
        negmask = np.isinf(arr) & (arr < 0)
        clipped = np.clip(arr, fp16_min, fp16_max)
        if negmask.any():
            clipped = np.where(negmask, np.float32("-inf"), clipped)
        new_tensor = numpy_helper.from_array(clipped, tensor.name)
        tensor.CopyFrom(new_tensor)
        return True

    @staticmethod
    def _split_tensor(tensor, model_name: str, file_num: int) -> None:
        external_data_helper.set_external_data(tensor, f"{model_name}_{file_num}.onnx.data")
