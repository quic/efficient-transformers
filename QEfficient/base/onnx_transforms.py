# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Optional, Tuple

import numpy as np
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
    Clips the tensor values to be in FP16 range.
    """

    @classmethod
    def apply(cls, model: ModelProto, *, onnx_base_dir: Optional[str] = None, **kwargs) -> Tuple[ModelProto, bool]:
        """
        :param onnx_base_dir: Base directory to load tensors (if not already loaded).
        """
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


class RemoveCrossAttentionIOTransform(OnnxTransform):
    """
    Removes the input and output names of cross-attention layers.
    """

    @classmethod
    def apply(cls, model: ModelProto, *, onnx_base_dir: Optional[str] = None, **kwargs) -> Tuple[ModelProto, bool]:
        """
        :param onnx_base_dir: Base directory to load tensors (if not already loaded).
        """
        layers_to_remove = [3, 8, 13, 18, 23, 28, 33, 38]
        names_to_remove = []
        for layer in layers_to_remove:
            names_to_remove.append(f"past_key.{layer}_RetainedState")
            names_to_remove.append(f"past_value.{layer}_RetainedState")
            names_to_remove.append(f"past_key.{layer}")
            names_to_remove.append(f"past_value.{layer}")

        graph = model.graph
        transformed = False

        # Remove outputs
        for name in names_to_remove:
            output_to_remove = None
            for output in graph.output:
                if output.name == name:
                    output_to_remove = output
                    break
            if output_to_remove:
                graph.output.remove(output_to_remove)
                transformed = True

        # # Remove inputs
        # for name in names_to_remove:
        #     input_to_remove = None
        #     for input in graph.input:
        #         if input.name == name:
        #             input_to_remove = input
        #             break
        #     if input_to_remove:
        #         graph.input.remove(input_to_remove)
        #         transformed = True

        return model, transformed
