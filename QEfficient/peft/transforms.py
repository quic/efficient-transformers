# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from peft import PeftModelForCausalLM

from QEfficient.base.onnx_transforms import OnnxTransform
from QEfficient.base.pytorch_transforms import ModuleMapping
from QEfficient.peft.peft_model import QEffPeftModelForCausalLM


class AdaptersPytorch(ModuleMapping):
    _module_mapping = {PeftModelForCausalLM: QEffPeftModelForCausalLM}


class AdaptersOnnx(OnnxTransform):
    pass
