# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import gc
import logging
from typing import Optional, Tuple

import numpy as np
from onnx import ModelProto, external_data_helper, numpy_helper

from QEfficient.utils.constants import ONNX_TRANSFROM_MEMORY_CLEANUP_INTERVAL

logger = logging.getLogger(__name__)


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

    @classmethod
    def _check_external_data_loaded(cls, model: ModelProto) -> bool:
        """
        Check if external data is already loaded in the model.

        :param model: The ONNX model to check
        :returns: True if external data is already loaded, False otherwise
        """
        for tensor in external_data_helper._get_all_tensors(model):
            # Check if tensor has external data but no raw data loaded
            if len(tensor.external_data) > 0 and not tensor.HasField("raw_data"):
                return False
        return True

    @classmethod
    def _cleanup_memory(cls):
        """
        Force garbage collection to free up memory after tensor processing.
        """
        gc.collect()


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

        processed_count = 0
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

                del neg_inf_mask, clipped_tensor, new_tensor

            del nptensor
            processed_count += 1

            if processed_count % ONNX_TRANSFROM_MEMORY_CLEANUP_INTERVAL == 0:
                cls._cleanup_memory()

        cls._cleanup_memory()
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

        # Check if external data is already loaded to avoid redundant loading
        external_data_already_loaded = cls._check_external_data_loaded(model)

        if not external_data_already_loaded:
            external_data_helper.load_external_data_for_model(model, onnx_base_dir)
        else:
            logger.info("External data already loaded, skipping redundant load operation")

        processed_count = 0
        for tensor in external_data_helper._get_all_tensors(model):
            if tensor.HasField("raw_data") and ((tsize := len(tensor.raw_data)) > size_threshold):
                transformed = True
                current_file_size += tsize
                if current_file_size > file_chunk_size:
                    file_num += 1
                    current_file_size = tsize
                external_data_helper.set_external_data(tensor, f"{model_name}_{file_num}.onnx.data")

            processed_count += 1
            if processed_count % ONNX_TRANSFROM_MEMORY_CLEANUP_INTERVAL == 0:
                cls._cleanup_memory()

        cls._cleanup_memory()

        return model, transformed
