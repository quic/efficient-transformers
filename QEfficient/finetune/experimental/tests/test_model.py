import pytest
import torch
import torch.nn as nn
from unittest import mock

import transformers
from QEfficient.finetune.experimental.core import model
from QEfficient.finetune.experimental.core.model import BaseModel, HFModel


class TestMockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


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
    breakpoint()
    m = TestCustomModel.create("dummy")
    # inner model exists and registered
    assert "_model" in m._modules
    assert isinstance(m.model, TestMockModel)
    # forward works
    out = m(torch.zeros(1, 2))
    assert out.shape == (1, 2)


def test_tokenizer_lazy_loading():
    m = TestCustomModel.create("dummy")
    assert m._tokenizer is None
    tok = m.tokenizer
    assert tok == "dummy-tokenizer"
    assert m._tokenizer == tok


def test_to_moves_inner_and_returns_self():
    m = TestCustomModel.create("dummy")
    with mock.patch.object(TestMockModel, "to", autospec=True) as mocked_to:
        ret = m.to("cuda:0")
    mocked_to.assert_called_once_with(m.model, "cuda:0")
    assert ret is m


def test_train_eval_sync_flags():
    m = TestCustomModel.create("dummy")
    m.eval()
    assert m.training is False
    assert m.model.training is False
    m.train()
    assert m.training is True
    assert m.model.training is True


def test_resize_token_embeddings_and_get_input_embeddings_warn(monkeypatch):
    m = TestCustomModel.create("dummy")

    # resize_token_embeddings: underlying model lacks the method, should warn and not raise
    with mock.patch("QEfficient.finetune.experimental.core.model.logger.info") as mocked_log:
        m.resize_token_embeddings(10)
        mocked_log.assert_called_once()

    # get_input_embeddings: underlying model lacks method, should warn and return None
    with mock.patch("QEfficient.finetune.experimental.core.model.logger.info") as mocked_log:
        assert m.get_input_embeddings() is None
        mocked_log.assert_called_once()


def test_state_dict_contains_inner_params():
    m = TestCustomModel.create("dummy")
    sd = m.state_dict()
    # should contain params from TestMockModel.linear
    assert any("linear.weight" in k for k in sd)
    assert any("linear.bias" in k for k in sd)


# HFModel tests
def test_hfmodel_invalid_auto_class_raises():
    with pytest.raises(ValueError):
        HFModel.create("hf-name", auto_class_name="AutoDoesNotExist")


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

    m = HFModel.create("hf-name")
    assert isinstance(m.model, FakeAuto)

    # load tokenizer
    tok = m.load_tokenizer()

    # tokenizer was loaded and pad token inserted
    model.AutoTokenizer.from_pretrained.assert_called_once_with("hf-name")
    model.insert_pad_token.assert_called_once_with(fake_tok)
