import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

import QEfficient
from QEfficient.transformers.models import modeling_auto as ma


class _DummyInputInfo:
    def __init__(self, name, datatype):
        self.name = name
        self.datatype = datatype

    def __repr__(self):
        return f"{self.name}:{self.datatype}"


class _DummyMultimodal(ma.MultimodalUtilityMixin):
    def __init__(self, model):
        self.model = model


class _TinyHF:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        cfg = BertConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=128,
            max_position_embeddings=32,
        )
        return BertModel(cfg).eval()


class _TinyWrapper(ma.QEFFTransformersBase):
    _hf_auto_class = _TinyHF
    _pytorch_transforms = []
    _onnx_transforms = []

    @property
    def get_model_config(self):
        return self.model.config.__dict__

    def export(self, export_dir=None):
        return "dummy.onnx"

    def compile(self, *args, **kwargs):
        return "dummy.qpc"


def _fake_causal():
    obj = object.__new__(ma.QEFFAutoModelForCausalLM)
    obj.continuous_batching = False
    obj.ccl_enabled = False
    obj.comp_ctx_lengths_prefill = None
    obj.comp_ctx_lengths_decode = None
    obj.is_tlm = False
    obj.num_layers = 2
    obj.hash_params = {}
    obj.model = types.SimpleNamespace(qaic_config={})
    return obj


def test_multimodal_input_autocorrect_success_and_error():
    model = types.SimpleNamespace(
        get_inputs_info=lambda: [
            _DummyInputInfo("input_ids", torch.int64),
            _DummyInputInfo("pixel_values", torch.float32),
        ]
    )
    mm = _DummyMultimodal(model)

    inputs = {
        "input_ids": torch.zeros((1, 2), dtype=torch.int64),
        "pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        "ignored": torch.ones((1,), dtype=torch.float32),
    }
    corrected = mm.auto_correct_inputs(inputs)
    assert set(corrected.keys()) == {"input_ids", "pixel_values"}

    with pytest.raises(RuntimeError, match="Expected following input names"):
        mm.auto_correct_inputs({"input_ids": torch.zeros((1, 2), dtype=torch.int32)})


def test_multimodal_mixin_cannot_be_instantiated_directly():
    with pytest.raises(TypeError, match="only children"):
        ma.MultimodalUtilityMixin()


def test_transformers_base_from_pretrained_and_repr():
    wrapped = _TinyWrapper.from_pretrained(
        "unused",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        enable_proxy=True,
    )
    text = repr(wrapped)
    assert text.startswith("_TinyWrapper\n")
    assert isinstance(wrapped.get_model_config, dict)


def test_qeff_auto_model_compile_warns_and_forwards_specializations(monkeypatch):
    obj = object.__new__(ma.QEFFAutoModel)
    captured = {}

    def _fake_compile(**kwargs):
        captured.update(kwargs)
        return "/tmp/qpc"

    obj._compile = _fake_compile
    out = obj.compile(seq_len=list(range(20)), batch_size=3)
    assert out == "/tmp/qpc"
    assert len(captured["specializations"]) == 20
    assert captured["specializations"][0]["batch_size"] == 3


def test_qeff_auto_model_generate_and_pytorch_path():
    obj = object.__new__(ma.QEFFAutoModel)
    obj.qpc_path = Path("/tmp/fake.qpc")
    obj.onnx_path = "/tmp/fake.onnx"
    obj.cloud_ai_100_feature_generate = lambda **kwargs: {"ok": True}
    obj.pytorch_feature_generate = lambda **kwargs: {"pt": True}
    obj.model = object()

    assert obj.generate(inputs={"input_ids": torch.ones((1, 1), dtype=torch.int64)}, runtime_ai100=True) == {"ok": True}
    assert obj.generate(inputs={"input_ids": torch.ones((1, 1), dtype=torch.int64)}, runtime_ai100=False) == {"pt": True}


def test_qeff_auto_model_generate_requires_compiled_qpc():
    obj = object.__new__(ma.QEFFAutoModel)
    obj.qpc_path = None
    obj.onnx_path = "/tmp/fake.onnx"
    with pytest.raises(TypeError, match="compile API first"):
        obj.generate(inputs={"input_ids": torch.ones((1, 1), dtype=torch.int64)})


def test_cloud_ai_100_feature_generate_handles_retry_path(monkeypatch):
    obj = object.__new__(ma.QEFFAutoModel)
    obj.qpc_path = Path("/tmp/fake.qpc")
    obj.qpc_session = None
    obj._write_io_dir = None

    class _FakeBinding:
        def __init__(self, dims):
            self.dims = dims

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            self.bindings = [_FakeBinding([1, 8]), _FakeBinding([1, 8]), _FakeBinding([1, 8, 4])]
            self.allowed_shapes = [[None, [None, [1, 8]]]]
            self._n = 0

        def set_buffers(self, outputs):
            self.outputs = outputs

        def run(self, inputs):
            self._n += 1
            if self._n == 1:
                raise RuntimeError("first run fails")
            return {"output": np.zeros((1, 8, 4), dtype=np.float32)}

    monkeypatch.setattr(ma, "QAICInferenceSession", _FakeSession)

    out = obj.cloud_ai_100_feature_generate(
        inputs={"input_ids": torch.ones((1, 4), dtype=torch.int64), "attention_mask": torch.ones((1, 4), dtype=torch.int64)}
    )
    assert "output" in out
    assert out["output"].shape == (1, 8, 4)


def test_causal_prefill_transform_routing(monkeypatch):
    obj = _fake_causal()
    obj.model = "base"
    called = []

    monkeypatch.setattr(ma.PrefillOnlyChunkedTransform, "apply", lambda model: (called.append("chunked") or "m1", True))
    monkeypatch.setattr(ma.PrefillOnlyTransform, "apply", lambda model: (called.append("prefill") or "m2", True))
    monkeypatch.setattr(
        ma.RevertPrefillKeepAttentionTransform, "apply", lambda model: (called.append("revert_keep") or "m3", True)
    )
    monkeypatch.setattr(ma.RevertPrefillOnlyTransform, "apply", lambda model: (called.append("revert") or "m4", True))

    obj.prefill(enable=True, enable_chunking=True)
    obj.prefill(enable=True, enable_chunking=False)
    obj.prefill(enable=False, retain_full_kv=True)
    obj.prefill(enable=False, retain_full_kv=False)
    assert called == ["chunked", "prefill", "revert_keep", "revert"]


def test_get_seq_len_and_specialized_prefill_validation(monkeypatch):
    obj = _fake_causal()

    with pytest.raises(ValueError, match="prefill_seq_len"):
        obj.get_seq_len_and_handle_specialized_prefill_model(prefill_seq_len=7, enable_chunking=False)

    monkeypatch.setenv("NUM_Q_BLOCKS", "4")
    monkeypatch.setenv("NUM_FFN_BLOCKS", "6")
    with pytest.raises(ValueError, match="not divisible"):
        obj.get_seq_len_and_handle_specialized_prefill_model(prefill_seq_len=8, enable_chunking=False)

    monkeypatch.delenv("NUM_FFN_BLOCKS", raising=False)
    seq_len = obj.get_seq_len_and_handle_specialized_prefill_model(prefill_seq_len=8, enable_chunking=False)
    assert seq_len >= ma.constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
    assert obj.hash_params["NUM_Q_BLOCKS"] == 4


def test_causal_compile_with_overridden_specializations_and_custom_io():
    obj = _fake_causal()
    obj.build_prefill_specialization = lambda **kwargs: {"prefill": True}
    obj.build_decode_specialization = lambda **kwargs: {"decode": True}
    captured = {}

    def _fake_compile(**kwargs):
        captured.update(kwargs)
        return "/tmp/fake_qpc"

    obj._compile = _fake_compile
    out = obj.compile(
        prefill_seq_len=8,
        ctx_len=64,
        batch_size=2,
        specializations=[{"forced": "spec"}],
        mxint8_kv_cache=True,
    )
    assert out == "/tmp/fake_qpc"
    assert captured["specializations"] == [{"forced": "spec"}]
    assert captured["custom_io"]["past_key.0"] == "mxint8"
    assert captured["custom_io"]["past_value.1_RetainedState"] == "mxint8"


def test_causal_compile_ccl_invalid_string_raises():
    obj = _fake_causal()
    obj._compile = lambda **kwargs: "/tmp/unused"
    obj.build_prefill_specialization = lambda **kwargs: {"prefill": True}
    obj.build_decode_specialization = lambda **kwargs: {"decode": True}
    with pytest.raises(ValueError, match="Invalid format for comp_ctx_lengths"):
        obj.compile(comp_ctx_lengths_prefill="not-a-list", comp_ctx_lengths_decode="[1,2]")


def test_causal_compile_calls_tlm_validation_and_passes_value():
    obj = _fake_causal()
    obj.is_tlm = True
    obj.build_prefill_specialization = lambda **kwargs: {"prefill": True}
    obj.build_decode_specialization = lambda **kwargs: {"decode": True}
    captured = {}
    seen = {}

    def _fake_check(num_speculative_tokens, prefill_seq_len):
        seen["called"] = True
        return 2

    def _fake_compile(**kwargs):
        captured.update(kwargs)
        return "/tmp/fake_qpc"

    obj.check_and_get_num_speculative_tokens = _fake_check
    obj._compile = _fake_compile

    out = obj.compile(prefill_seq_len=8, ctx_len=64)
    assert out == "/tmp/fake_qpc"
    assert seen["called"] is True
    assert captured["num_speculative_tokens"] == 2


def test_causal_compile_sampler_plus_speculative_tokens_raises():
    obj = _fake_causal()
    obj.model.qaic_config = {"include_sampler": True}
    obj.build_prefill_specialization = lambda **kwargs: {"prefill": True}
    obj.build_decode_specialization = lambda **kwargs: {"decode": True}
    obj._compile = lambda **kwargs: "/tmp/fake_qpc"
    with pytest.raises(ValueError, match="sampler does not support"):
        obj.compile(prefill_seq_len=8, num_speculative_tokens=2)


def test_causal_generate_runtime_paths(monkeypatch):
    obj = _fake_causal()
    obj.onnx_path = "/tmp/fake.onnx"
    obj.qpc_path = Path("/tmp/fake.qpc")
    obj.comp_ctx_lengths_prefill = [8]
    obj.comp_ctx_lengths_decode = [16]

    monkeypatch.setattr(
        QEfficient,
        "cloud_ai_100_exec_kv",
        lambda **kwargs: {"generated": ["ok"], "generation_len": kwargs.get("generation_len")},
    )
    out = obj.generate(tokenizer=object(), prompts=["hi"], generation_len=4)
    assert out["generated"] == ["ok"]

    with pytest.raises(NotImplementedError, match="Only AI_100 runtime"):
        obj.generate(tokenizer=object(), prompts=["hi"], runtime_ai100=False)


def test_causal_generate_requires_compile():
    obj = _fake_causal()
    obj.onnx_path = "/tmp/fake.onnx"
    obj.qpc_path = None
    with pytest.raises(TypeError, match="compile API first"):
        obj.generate(tokenizer=object(), prompts=["hello"])
