# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from peft import PeftModelForCausalLM

from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from QEfficient.peft.peft_model import QEffPeftModelForCausalLM


class PeftModelInputsTransform(ModuleMappingTransform):
    _module_mapping = {PeftModelForCausalLM: QEffPeftModelForCausalLM}
