# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import List

from QEfficient.base.modeling_qeff import QEffBaseModel
from QEfficient.base.onnx_transforms import OnnxTransform
from QEfficient.base.pytorch_transforms import PytorchTransform
from QEfficient.peft.transforms import AdaptersOnnx, AdaptersPytorch
from QEfficient.transformers.transforms import CustomOps, KVCache


class QEffAutoPeftModelForCausalLM(QEffBaseModel):
    pytorch_transforms: List[PytorchTransform] = [CustomOps, KVCache, AdaptersPytorch]
    onnx_transforms: List[OnnxTransform] = [AdaptersOnnx]
