# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Optional

import numpy as np
from onnx import ModelProto, external_data_helper, numpy_helper


class OnnxTransform:
    """
    OnnxTransform is the base class for graph modifications on exported onnx.
    """

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Directly use the `apply` method.")

    @classmethod
    def apply(cls, model: ModelProto) -> [ModelProto, bool]:
        """
        Override this class to apply a transformation.
        :param model: The model's ONNX graph to transform

        :returns: ONNX graph after applying the transform
        :returns: Boolean indicating whether tranform was applied
        """
        raise NotImplementedError("Use subclasses for ONNX transform")


class FP16Clip(OnnxTransform):
    """
    Clips the tensor values to be in FP16 range.
    """

    @classmethod
    def apply(cls, model: ModelProto, onnx_base_dir: Optional[str] = None) -> [ModelProto, bool]:
        finfo = np.finfo(np.float16)
        fp16_max = finfo.max
        fp16_min = finfo.min
        transformed = False
        for tensor in external_data_helper._get_all_tensors(model):
            nptensor = numpy_helper.to_array(tensor, onnx_base_dir)
            if nptensor.dtype == np.float32 and (np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)):
                nptensor = np.clip(nptensor, fp16_min, fp16_max)
                new_tensor = numpy_helper.from_array(nptensor, tensor.name)
                tensor.CopyFrom(new_tensor)
                transformed = True
        return model, transformed


class SplitWeights(OnnxTransform):
    pass


class LoraAdapters(OnnxTransform):
    pass
