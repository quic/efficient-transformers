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
from typing import Any


# Defining placeholder ENUM for execute function
class Runtime(Enum):
    CPU_ORT = "CPU ONNX Runtime"
    CPU_PT = "CPU PyTorch Runtime"
    AI_100 = "AI_100"


class QEFFBaseModel(ABC):
    """
    This class acts as parent class for all the varieties of model class (i.e. LLMs, SD, quantized etc.).
    Enforces certain methods to be implemented by child classes.

    All the child classes must provide way to load, transform(optimize), exoprt to ONNX etc. capabilities.
    """

    def __init__(self) -> None:
        super().__init__()
        # Users can call generate or execute
        self.generate = self.execute
        self.__runtime = Runtime.CPU_PT
        self.ort_runtime_args = None
        self.cloud_ai_100_runtime_args = None

    @property
    def runtime(self) -> Runtime:
        return self.__runtime

    @runtime.setter
    def runtime(self, value: Runtime) -> None:
        if value == Runtime.CPU_ORT:
            assert self.ort_runtime_args is not None, "Please run export first"
        elif value == Runtime.AI_100:
            assert self.cloud_ai_100_runtime_args is not None, "Please run compile first"
        else:
            assert value == Runtime.CPU_PT, f"Expected runtime to be of type {Runtime.__class__} got {type(value)}"
        self.__runtime = value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        raise NotImplementedError("Must implement for child classes")

    @property
    def is_transformed(self) -> bool:
        raise NotImplementedError("Must implement for child classes")

    @abstractmethod
    def export_and_compile(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def transform(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def export(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def compile(self, *args, **kwargs) -> Any:
        pass
