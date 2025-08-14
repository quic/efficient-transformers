# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
import numpy as np
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from onnx import ModelProto, external_data_helper, numpy_helper



class OnnxTransform:
    """
    OnnxTransform is the base class for graph modifications on exported ONNX.
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
    @classmethod
    def apply(cls, model: ModelProto, *, onnx_base_dir: Optional[str] = None, **kwargs) -> Tuple[ModelProto, bool]:
        finfo = np.finfo(np.float16)
        fp16_max = finfo.max
        fp16_min = finfo.min

        def clip_tensor(tensor):
            nptensor = numpy_helper.to_array(tensor, onnx_base_dir)
            if nptensor.dtype == np.float32 and (np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)):
                clipped_tensor = np.clip(nptensor, fp16_min, fp16_max)
                new_tensor = numpy_helper.from_array(clipped_tensor, tensor.name)
                tensor.CopyFrom(new_tensor)
                return True
            return False

        tensors = external_data_helper._get_all_tensors(model)
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
            results = list(executor.map(clip_tensor, tensors))
        transformed = any(results)
        return model, transformed


class SplitTensorsTransform(OnnxTransform):
    @classmethod
    def apply(
        cls,
        model: ModelProto,
        *,
        model_name: str,
        onnx_base_dir: Optional[str] = None,
        file_chunk_size: int = 10 * 2**30,
        size_threshold: int = 1024,
        **kwargs,
    ) -> Tuple[ModelProto, bool]:
        external_data_helper.load_external_data_for_model(model, onnx_base_dir)
        tensors = external_data_helper._get_all_tensors(model)
        file_assignments = []
        file_num = 0
        current_file_size = 0
        transformed = False

        for tensor in tensors:
            if tensor.HasField("raw_data") and (tsize := len(tensor.raw_data)) > size_threshold:
                transformed = True
                if current_file_size + tsize > file_chunk_size:
                    file_num += 1
                    current_file_size = 0
                current_file_size += tsize
                file_assignments.append((tensor, f"{model_name}_{file_num}.onnx.data"))

        def process_tensor(args):
            tensor, file_name = args
            external_data_helper.set_external_data(tensor, file_name)

        with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
            list(executor.map(process_tensor, file_assignments))
        return model, transformed