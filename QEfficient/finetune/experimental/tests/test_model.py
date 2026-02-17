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


# Config Loading Tests - Partial Loading of Meta Llama Model


def test_hfmodel_partial_loading_meta_llama_with_direct_config_params(monkeypatch):
    """
    Test partial loading of meta-llama model using direct config parameters.
    Loads meta-llama model with reduced layers (2 layers) for faster testing
    using direct config parameters (HuggingFace standard pattern).
    """

    # Mock model that respects num_hidden_layers parameter
    def create_partial_model(name, config=None, num_hidden_layers=None, **kwargs):
        model_instance = nn.Module()
        if config:
            model_instance.config = config
            n_layers = config.num_hidden_layers
        elif num_hidden_layers:
            model_instance.config = mock.Mock()
            model_instance.config.num_hidden_layers = num_hidden_layers
            model_instance.config.hidden_size = 4096
            n_layers = num_hidden_layers
        else:
            model_instance.config = mock.Mock()
            model_instance.config.num_hidden_layers = 32
            model_instance.config.hidden_size = 4096
            n_layers = 32

        model_instance.layers = nn.ModuleList(
            [nn.Linear(model_instance.config.hidden_size, model_instance.config.hidden_size) for _ in range(n_layers)]
        )
        return model_instance

    class MockAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, name, config=None, num_hidden_layers=None, **kwargs):
            return create_partial_model(name, config=config, num_hidden_layers=num_hidden_layers, **kwargs)

    monkeypatch.setattr(
        "QEfficient.finetune.experimental.core.model.transformers.AutoModelForCausalLM",
        MockAutoModelForCausalLM,
        raising=False,
    )
    monkeypatch.setattr(
        "QEfficient.finetune.experimental.core.model.AutoTokenizer",
        type("MockTokenizer", (), {"from_pretrained": lambda *args, **kwargs: mock.Mock(pad_token_id=0)}),
        raising=False,
    )
    monkeypatch.setattr(
        "QEfficient.finetune.experimental.core.model.insert_pad_token",
        lambda tok: None,
        raising=False,
    )

    # Load partial meta-llama model with direct config parameter (2 layers for testing)
    partial_model = ComponentFactory.create_model(
        "hf",
        "meta-llama/Llama-3.2-1B",
        num_hidden_layers=2,  # Load only 2 layers (partial loading)
    )

    # Verify partial model was loaded with reduced layers
    assert partial_model.model.config.num_hidden_layers == 2
    assert len(partial_model.model.layers) == 2
    assert partial_model.model.config.hidden_size == 4096


def test_hfmodel_partial_loading_meta_llama_for_fast_testing(monkeypatch):
    """
    Test partial loading of meta-llama model for fast testing.
    """

    # Mock model that respects num_hidden_layers parameter
    def create_partial_model(name, config=None, num_hidden_layers=None, **kwargs):
        model_instance = nn.Module()
        if config:
            model_instance.config = config
            n_layers = config.num_hidden_layers
        elif num_hidden_layers:
            model_instance.config = mock.Mock()
            model_instance.config.num_hidden_layers = num_hidden_layers
            model_instance.config.hidden_size = 4096
            n_layers = num_hidden_layers
        else:
            model_instance.config = mock.Mock()
            model_instance.config.num_hidden_layers = 32
            model_instance.config.hidden_size = 4096
            n_layers = 32

        model_instance.layers = nn.ModuleList(
            [nn.Linear(model_instance.config.hidden_size, model_instance.config.hidden_size) for _ in range(n_layers)]
        )
        # Track parameter count (fewer layers = fewer parameters)
        model_instance.param_count = sum(p.numel() for p in model_instance.layers.parameters())
        return model_instance

    class MockAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, name, config=None, num_hidden_layers=None, **kwargs):
            return create_partial_model(name, config=config, num_hidden_layers=num_hidden_layers, **kwargs)

    monkeypatch.setattr(
        "QEfficient.finetune.experimental.core.model.transformers.AutoModelForCausalLM",
        MockAutoModelForCausalLM,
        raising=False,
    )
    monkeypatch.setattr(
        "QEfficient.finetune.experimental.core.model.AutoTokenizer",
        type("MockTokenizer", (), {"from_pretrained": lambda *args, **kwargs: mock.Mock(pad_token_id=0)}),
        raising=False,
    )
    monkeypatch.setattr(
        "QEfficient.finetune.experimental.core.model.insert_pad_token",
        lambda tok: None,
        raising=False,
    )

    # Load partial meta-llama model (2 layers)
    test_model = ComponentFactory.create_model("hf", "meta-llama/Llama-3.2-1B", num_hidden_layers=2)

    # Verify partial model was loaded with reduced layers
    assert test_model.model.config.num_hidden_layers == 2
    assert len(test_model.model.layers) == 2

    # Verify model still works (can do forward pass)
    test_input = torch.randn(1, 10, test_model.model.config.hidden_size)
    output = test_model.model.layers[0](test_input)
    assert output.shape == test_input.shape

    # Verify we can test model functionality with partial model
    assert len(test_model.model.layers) == 2
    assert test_model.model.config.hidden_size == 4096  # Other config preserved
    assert test_model.model.param_count > 0  # Has parameters
