# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from unittest import mock

import pytest
import torch
import torch.nn as nn

from QEfficient.finetune.experimental.core import model
from QEfficient.finetune.experimental.core.component_registry import ComponentFactory, registry
from QEfficient.finetune.experimental.core.model import BaseModel


class TestMockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


@registry.model("testcustom")
class TestCustomModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        print("init of custom class")

    def load_model(self) -> nn.Module:
        return TestMockModel()

    def load_tokenizer(self):
        return "dummy-tokenizer"


# BaseModel tests
def test_model_property_errors_if_not_created():
    m = TestCustomModel("dummy")
    with pytest.raises(RuntimeError):
        _ = m.model  # must call .create()


def test_create_builds_and_registers():
    m = ComponentFactory.create_model("testcustom", "dummy")
    # inner model exists and registered
    assert "_model" in m._modules
    assert isinstance(m.model, TestMockModel)
    # forward works
    out = m(torch.zeros(1, 2))
    assert out.shape == (1, 2)


def test_tokenizer_lazy_loading():
    m = ComponentFactory.create_model("testcustom", "dummy")
    assert m._tokenizer is None
    tok = m.tokenizer
    assert tok == "dummy-tokenizer"
    assert m._tokenizer == tok


def test_to_moves_inner_and_returns_self():
    m = ComponentFactory.create_model("testcustom", "dummy")
    with mock.patch.object(TestMockModel, "to", autospec=True) as mocked_to:
        ret = m.to("cpu:0")
    assert mocked_to.call_args[0][0] is m.model
    assert mocked_to.call_args[0][1] == "cpu:0"
    assert ret is m


def test_train_eval_sync_flags():
    m = ComponentFactory.create_model("testcustom", "dummy")
    m.eval()
    assert m.training is False
    assert m.model.training is False
    m.train()
    assert m.training is True
    assert m.model.training is True


def test_state_dict_contains_inner_params():
    m = ComponentFactory.create_model("testcustom", "dummy")
    sd = m.state_dict()
    # should contain params from TestMockModel.linear
    assert any("linear.weight" in k for k in sd)
    assert any("linear.bias" in k for k in sd)


# HFModel tests
def test_hfmodel_invalid_auto_class_raises():
    with pytest.raises(ValueError):
        ComponentFactory.create_model("hf", "hf-name", auto_class_name="AutoDoesNotExist")


def test_hfmodel_loads_auto_and_tokenizer(monkeypatch):
    # fake HF Auto class
    class FakeAuto(nn.Module):
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            inst = cls()
            inst.loaded = (name, kwargs)
            return inst

        def forward(self, x):
            return x

    fake_tok = mock.Mock()

    # Monkeypatch transformer classes used in HFModel
    monkeypatch.setattr(
        "QEfficient.finetune.experimental.core.model.transformers.AutoModelForCausalLM",
        FakeAuto,
        raising=False,
    )
    monkeypatch.setattr(
        model,
        "AutoTokenizer",
        mock.Mock(from_pretrained=mock.Mock(return_value=fake_tok)),
    )
    monkeypatch.setattr(
        "QEfficient.finetune.experimental.core.model.insert_pad_token",
        mock.Mock(),
        raising=False,
    )
    m = ComponentFactory.create_model("hf", "hf-name")
    assert isinstance(m.model, FakeAuto)

    # load tokenizer
    tok = m.load_tokenizer()

    assert hasattr(tok, "pad_token_id")
    assert m.model.loaded[0] == "hf-name"
