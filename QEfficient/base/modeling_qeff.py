# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""
** This file for holds the classes that handle main functions
1.load i.e. from_pretrained
2.execute
3.transform
4.export
5.compile
For different varities of Transformer Models

Representation of class inheritence followed keeping in line with transformers/diffusers repos ->

                                                                                            QEFFBaseModel
                                                 ________________________________________________|________________________________________________________________
                                                |                                                                                                                 |
                                            QEFFTransformersBase                                                                                           QEFFDiffusersBase
                                                |                                                                                                                 |
                                    ____________|________________________________________________________ ________________                       _________________|______________
                   _____           |                              |                                      |                |                     |                                |
                  |          QEFFAutoModel             QEFFAutoModelForCausalLM              QEFFAWQModelForCausalLM     ...                   ...                              ...
QEFFCommonLoader -|       [Provides way to          [Provides way to do 1-5 on                 [Supports 1-5 for
[Provides         |        do steps 1-5 on           transformers.AutoModelForCausalLM]         AWQ Models]
interface to      |_____   transformers.AutoModel]
Load any of
These models
by automatically
detecting the type
of the model]

** QEFFBASEModel is abstract base class that defines the basic structure of these classes.
** QEFFPipeline classes will stay at the same level as QEFFAutoModel in this hierarchy in future.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from onnxruntime import InferenceSession as ORTInferenceSession

from QEfficient.generation.cloud_infer import QAICInferenceSession


# Defining placeholder ENUM for execute function
class Runtime(Enum):
    CPU_ORT = "CPU_ORT"
    CPU_PT = "CPU_PT"
    AI_100 = "AI_100"


class QEFFBaseModel(ABC):
    """
    This class acts as parent class for all the varieties of model class (i.e. LLMs, SD, quantized etc.).
    Enforces certain methods to be implemented by child classes.

    All the child classes must provide way to load, transform(optimize), exoprt to ONNX etc. capabilities.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.onnx_path: Optional[str] = None
        self.ort_session: Optional[ORTInferenceSession] = None
        self.qpc_path: Optional[str] = None
        self.device_id: Optional[List[int]] = None
        self.qpc_session: Optional[QAICInferenceSession] = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        raise NotImplementedError("Must implement for child classes")

    def run_pytorch(self, inputs):
        raise NotImplementedError("Reached too far!!")

    def run_ort(self, inputs: Dict[str, np.ndarray], /, cache_session: bool = False):
        raise NotImplementedError("Reached too far!!")

    def run_cloud_ai_100(
        self,
    ):
        raise NotImplementedError("Reached too far!!")

    @abstractmethod
    def transform(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def export(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def compile(self, *args, **kwargs) -> Any:
        pass
