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

from QEfficient.utils.constants import ONNX_TRANSFORM_MEMORY_CLEANUP_INTERVAL

logger = logging.getLogger(__name__)


class OnnxTransform:
    """
    OnnxTransform is the base class for graph modifications on exported onnx.
    """

    _external_data_loaded_cache = {}  # Dict[int, bool]

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
        # Use object ID as key instead of the object itself
        model_id = id(model)
        # Return cached result if available
        if model_id in cls._external_data_loaded_cache:
            return cls._external_data_loaded_cache[model_id]

        # Load the model if not already loaded
        for tensor in external_data_helper._get_all_tensors(model):
            # Check if tensor has external data but no raw data loaded
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
        try:
            # --- FIX: Ensure external data is loaded efficiently BEFORE processing ---
            cls._load_external_data(model, onnx_base_dir)

            finfo = np.finfo(np.float16)
            fp16_max = finfo.max
            fp16_min = finfo.min
            transformed = False

            processed_count = 0
            for tensor in external_data_helper._get_all_tensors(model):
                nptensor = numpy_helper.to_array(tensor)  # Removed onnx_base_dir as data is already loaded
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

                if processed_count % ONNX_TRANSFORM_MEMORY_CLEANUP_INTERVAL == 0:
                    cls._cleanup_memory()

            return model, transformed
        finally:
            # Ensure cleanup happens even if an exception occurs
            cls._cleanup_memory()


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
        try:
            file_num = 0
            current_file_size = 0
            transformed = False

            # --- Adjustment: The initial check and load will now use the new bulk loader ---
            # This will either use the cache (if FP16ClipTransform loaded it) or perform the bulk load itself.
            cls._load_external_data(model, onnx_base_dir)

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
                if processed_count % ONNX_TRANSFORM_MEMORY_CLEANUP_INTERVAL == 0:
                    cls._cleanup_memory()

            return model, transformed
        finally:
            # Ensure cleanup happens even if an exception occurs
            cls._cleanup_memory()
