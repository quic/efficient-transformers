# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import pytest
import torch
from torch import nn

from QEfficient.base.transforms import OnnxTransform, PytorchTransform


def test_pytorch_transform():
    with pytest.raises(ValueError):
        PytorchTransform()

    class TestTransform(PytorchTransform):
        _module_mapping = {nn.Linear: nn.Identity}

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.a = nn.Linear(32, 32)
            self.b = nn.ReLU()
            self.c = nn.Linear(32, 32)

        def forward(self, x):
            x = self.a(x)
            x = self.b(x)
            x = self.c(x)
            return x

    model = TestModel()
    x = torch.rand(1, 32)
    x = torch.where(x < 0, 0, x)
    y1 = model(x)
    assert torch.any(y1 != x)

    model = TestTransform.apply(model)
    y2 = model(x)
    assert torch.all(y2 == x)


def test_onnx_transform():
    with pytest.raises(ValueError):
        OnnxTransform()
