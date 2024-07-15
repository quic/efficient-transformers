# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from onnx import ModelProto


class OnnxTransform:
    """
    OnnxTransform is the base class for graph modifications on exported onnx.
    """

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Directly use the `apply` method.")

    @classmethod
    def apply(cls, model: ModelProto) -> ModelProto:
        """
        Override this class to apply a transformation.
        :param model: The model's ONNX graph to transform

        :returns: ONNX graph after applying the transform
        """
        raise NotImplementedError("Use subclasses for ONNX transform")


class FP16Clip(OnnxTransform):
    pass


class SplitWeights(OnnxTransform):
    pass


class LoraAdapters(OnnxTransform):
    pass
