# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import pytest
import torch
from torch import nn

from QEfficient.base.pytorch_transforms import ModuleMappingTransform, ModuleMutatorTransform


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.a = nn.Linear(32, 64)
        self.b = nn.Linear(64, 32)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        return x


def test_module_mapping_transform():
    with pytest.raises(TypeError):
        ModuleMappingTransform()

    class TestTransform(ModuleMappingTransform):
        _module_mapping = {nn.Linear: nn.Identity}

    model = TestModel()
    x = torch.rand(1, 32)
    y1 = model(x)
    assert torch.any(y1 != x)

    model, transformed = TestTransform.apply(model)
    assert transformed
    y2 = model(x)
    assert torch.all(y2 == x)


def test_module_mutator_transform():
    with pytest.raises(TypeError):
        ModuleMutatorTransform()

    class TestTransform(ModuleMutatorTransform):
        _match_class = nn.Linear

        @classmethod
        def mutate(cls, original_module: nn.Module, parent_module: nn.Module):
            return nn.Identity()

    model = TestModel()
    prev_ids = [id(model.a), id(model.b)]
    x = torch.rand(1, 32)
    y1 = model(x)
    assert torch.any(y1 != x)
    model, transformed = TestTransform.apply(model)
    assert transformed
    assert not ([id(model.a), id(model.b)] == prev_ids)
    y2 = model(x)
    assert torch.all(y2 == x)
