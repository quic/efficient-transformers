# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
import warnings
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
from onnx import ModelProto, TensorProto, external_data_helper, numpy_helper


class OnnxTransform:
    """
    OnnxTransform is the base class for graph modifications on exported onnx.
    """

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


class ClipAndSplitTransform(OnnxTransform):
    @classmethod
    def apply(
        cls,
        model: ModelProto,
        *,
        model_name: str,
        onnx_base_dir: Optional[str] = None,
        apply_clip: bool = True,
        apply_split: bool = True,
        file_chunk_size: int = 10 * 2**30,
        size_threshold: int = 1024,
        **kwargs,
    ) -> Tuple[ModelProto, bool]:
        if not apply_clip and not apply_split:
            warnings.warn("Both apply_clip and apply_split are False. Skipping transformation.")
            return model, False

        external_data_helper.load_external_data_for_model(model, onnx_base_dir)
        tensors = external_data_helper._get_all_tensors(model)

        TensorInfo = namedtuple("TensorInfo", ["tensor", "tsize"])
        tensor_infos = [
            TensorInfo(tensor, len(tensor.raw_data) if tensor.HasField("raw_data") else 0) for tensor in tensors
        ]

        fp16_min, fp16_max = np.finfo(np.float16).min, np.finfo(np.float16).max
        file_num_tracker = {"num": 0, "size": 0}

        def process_tensor(info: TensorInfo) -> bool:
            tensor, tsize = info
            transformed_clip = False
            transformed_split = False

            if apply_clip:
                transformed_clip = cls._clip_tensor(tensor, onnx_base_dir, fp16_min, fp16_max)

            if apply_split and tsize > size_threshold:
                if file_num_tracker["size"] + tsize > file_chunk_size:
                    file_num_tracker["num"] += 1
                    file_num_tracker["size"] = tsize
                else:
                    file_num_tracker["size"] += tsize

                cls._split_tensor(tensor, model_name, file_num_tracker["num"])
                transformed_split = True

            if apply_clip and apply_split:
                return transformed_clip and transformed_split
            return transformed_clip or transformed_split

        with ThreadPoolExecutor(max_workers=os.cpu_count() * 4) as executor:
            transformed_flags = list(executor.map(process_tensor, tensor_infos))
        return model, any(transformed_flags)

    @staticmethod
    def _clip_tensor(tensor, onnx_base_dir, fp16_min, fp16_max) -> bool:
        if tensor.data_type != TensorProto.FLOAT:
            return False

        nptensor = numpy_helper.to_array(tensor, onnx_base_dir)
        if np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min):
            neg_inf_mask = np.isinf(nptensor) & (nptensor < 0)
            clipped_tensor = np.clip(nptensor, fp16_min, fp16_max)
            if neg_inf_mask.any():
                clipped_tensor = np.where(neg_inf_mask, np.float32("-inf"), clipped_tensor)
            new_tensor = numpy_helper.from_array(clipped_tensor, tensor.name)
            tensor.CopyFrom(new_tensor)
            return True
        return False

    @staticmethod
    def _split_tensor(tensor, model_name: str, file_num: int) -> None:
        external_data_helper.set_external_data(tensor, f"{model_name}_{file_num}.onnx.data")
