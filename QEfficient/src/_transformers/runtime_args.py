# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from dataclasses import dataclass, field
from typing import List

from QEfficient.utils.constants import Constants


@dataclass
class QEFFAutoModelForCausalLMAI100RuntimeArgs:
    qpc_dir_path: str
    device_group: List[int]

    def __post_init__(self):
        assert os.path.isdir(self.qpc_dir_path), f"Please provide valid qpc_dir_path, got {self.qpc_dir_path}"


#FIXME: figure out if all these are really required -> update while adding execute feature
@dataclass
class QEFFAutoModelForCausalLMCPUORTRuntimeArgs:
    onnx_model_path: str
    ctx_len: int = field(default=Constants.CTX_LEN)
    prompt_len: int = field(default=Constants.PROMPT_LEN)

    def __post_init__(self):
        assert os.path.isfile(self.onnx_model_path), f"Please provide valid onnx_model_path, got {self.onnx_model_path}"
        assert self.ctx_len >= self.prompt_len, f"prompt_len can not be lesser than ctx_len; got ctx_len={self.ctx_len}, prompt_len={self.prompt_len}"