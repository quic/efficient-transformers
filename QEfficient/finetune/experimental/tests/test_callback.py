# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from types import SimpleNamespace

import pytest
import torch
from transformers import TrainerCallback

from QEfficient.finetune.experimental.core import callbacks as callbacks_module
from QEfficient.finetune.experimental.core.callbacks import QAICOpByOpVerifierCallback, QAICProfilerCallback
from QEfficient.finetune.experimental.core.component_registry import ComponentFactory, registry


class ModelSummaryCallback(TrainerCallback):
    def __init__(self):
        pass


# Setup test data
CALLBACK_CONFIGS = [
    pytest.param(
        {
            "name": "early_stopping",
            "early_stopping_patience": 3,
            "early_stopping_threshold": 0.001,
        },
        id="early_stopping",
    ),
    pytest.param({"name": "tensorboard", "tb_writer": "SummaryWriter"}, id="tensorboard"),
    pytest.param(
        {
            "name": "model_summary",
            "max_depth": 1,
        },
        id="model_summary",
    ),
    pytest.param(
        {
            "name": "qaic_profiler_callback",
            "start_step": 0,
            "end_step": 1,
            "trace_dir": "/tmp/hw-trace",
            "device_ids": [0],
        },
        id="qaic_profiler",
    ),
]

REGISTRY_CALLBACK_CONFIGS = {
    "model_summary": {
        "name": "model_summary",
        "max_depth": 1,
        "callback_class": ModelSummaryCallback,
    },
}


@pytest.mark.parametrize("config", CALLBACK_CONFIGS)
def test_callbacks(config):
    """Test that registered callbacks that can be created with their configs."""
    # Create callbacks using the factory
    try:
        callback_inst = ComponentFactory.create_callback(**config)
    except ValueError as e:
        assert "Unknown callback" in str(e)
        return
    assert callback_inst is not None
    assert isinstance(callback_inst, TrainerCallback)


@pytest.mark.parametrize("callback_name,callback_class", REGISTRY_CALLBACK_CONFIGS.items())
def test_callbacks_registery(callback_name, callback_class):
    """Test that a callback registered correctly."""
    registry.callback(callback_name)(callback_class)
    callback = registry.get_callback(callback_name)
    assert callback is not None
    assert callback == callback_class


def test_qaic_profiler_uses_user_trace_dir():
    callback = QAICProfilerCallback(trace_dir="~/my_custom_hw_trace")
    assert callback.trace_dir == "~/my_custom_hw_trace"


def test_qaic_profiler_starts_with_trace_dir(monkeypatch):
    calls = []

    monkeypatch.setattr(callbacks_module, "get_local_rank", lambda: 0)
    monkeypatch.setattr(callbacks_module, "get_world_size", lambda: 1)

    def _mock_start(use_profiler, device_type, trace_dir=None):
        calls.append((use_profiler, device_type, trace_dir))

    monkeypatch.setattr(callbacks_module, "init_qaic_profiling", _mock_start)

    callback = QAICProfilerCallback(start_step=3, end_step=9, trace_dir="/tmp/hw-trace", device_ids=[2])
    state = SimpleNamespace(global_step=3)

    callback.on_step_begin(args=None, state=state, control=None)

    assert callback._profile_started is True
    assert calls == [(True, "qaic:2", "/tmp/hw-trace")]


def test_qaic_profiler_stops_once_at_end_step(monkeypatch):
    start_calls = []
    stop_calls = []

    monkeypatch.setattr(callbacks_module, "get_local_rank", lambda: 0)
    monkeypatch.setattr(callbacks_module, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        callbacks_module,
        "init_qaic_profiling",
        lambda use_profiler, device_type, trace_dir=None: start_calls.append((use_profiler, device_type, trace_dir)),
    )
    monkeypatch.setattr(
        callbacks_module,
        "stop_qaic_profiling",
        lambda use_profiler, device_type: stop_calls.append((use_profiler, device_type)),
    )

    callback = QAICProfilerCallback(start_step=1, end_step=2, trace_dir="/tmp/hw-trace", device_ids=[0])

    callback.on_step_begin(args=None, state=SimpleNamespace(global_step=1), control=None)
    callback.on_step_end(args=None, state=SimpleNamespace(global_step=2), control=None)
    callback.on_step_end(args=None, state=SimpleNamespace(global_step=3), control=None)

    assert len(start_calls) == 1
    assert stop_calls == [(True, "qaic:0")]
    assert callback._profile_started is False


def test_qaic_profiler_stops_on_train_end_when_not_stopped(monkeypatch):
    stop_calls = []

    monkeypatch.setattr(callbacks_module, "get_local_rank", lambda: 0)
    monkeypatch.setattr(callbacks_module, "get_world_size", lambda: 1)
    monkeypatch.setattr(callbacks_module, "init_qaic_profiling", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        callbacks_module,
        "stop_qaic_profiling",
        lambda use_profiler, device_type: stop_calls.append((use_profiler, device_type)),
    )

    callback = QAICProfilerCallback(start_step=0, end_step=100, device_ids=[4])
    callback.on_step_begin(args=None, state=SimpleNamespace(global_step=0), control=None)
    callback.on_train_end(args=None, state=SimpleNamespace(global_step=1), control=None)

    assert stop_calls == [(True, "qaic:4")]


def test_qaic_profiler_uses_local_rank_when_device_ids_not_set(monkeypatch):
    calls = []

    monkeypatch.setattr(callbacks_module, "get_local_rank", lambda: 3)
    monkeypatch.setattr(callbacks_module, "get_world_size", lambda: 8)
    monkeypatch.setattr(
        callbacks_module,
        "init_qaic_profiling",
        lambda use_profiler, device_type, trace_dir=None: calls.append((use_profiler, device_type, trace_dir)),
    )

    callback = QAICProfilerCallback(start_step=0, trace_dir="/tmp/hw-trace")
    callback.on_step_begin(args=None, state=SimpleNamespace(global_step=0), control=None)

    assert calls == [(True, "qaic:3", "/tmp/hw-trace")]


def test_qaic_profiler_maps_rank_to_device_id(monkeypatch):
    calls = []

    monkeypatch.setattr(callbacks_module, "get_local_rank", lambda: 1)
    monkeypatch.setattr(callbacks_module, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        callbacks_module,
        "init_qaic_profiling",
        lambda use_profiler, device_type, trace_dir=None: calls.append((use_profiler, device_type, trace_dir)),
    )

    callback = QAICProfilerCallback(start_step=5, trace_dir="/tmp/hw-trace", device_ids=[10, 11])
    callback.on_step_begin(args=None, state=SimpleNamespace(global_step=5), control=None)

    assert calls == [(True, "qaic:11", "/tmp/hw-trace")]


def test_qaic_profiler_invalid_step_range_raises():
    with pytest.raises(ValueError, match="end_step .* must be >= start_step"):
        QAICProfilerCallback(start_step=10, end_step=5)


def test_qaic_profiler_stops_only_started_devices(monkeypatch):
    start_calls = []
    stop_calls = []

    monkeypatch.setattr(callbacks_module, "get_local_rank", lambda: 0)
    monkeypatch.setattr(callbacks_module, "get_world_size", lambda: 1)

    def _mock_start(use_profiler, device_type, trace_dir=None):
        start_calls.append((use_profiler, device_type, trace_dir))
        if device_type == "qaic:1":
            raise RuntimeError("start failure")

    monkeypatch.setattr(callbacks_module, "init_qaic_profiling", _mock_start)
    monkeypatch.setattr(
        callbacks_module,
        "stop_qaic_profiling",
        lambda use_profiler, device_type: stop_calls.append((use_profiler, device_type)),
    )

    callback = QAICProfilerCallback(start_step=0, end_step=1, trace_dir="/tmp/hw-trace", device_ids=[0, 1])
    callback.on_step_begin(args=None, state=SimpleNamespace(global_step=0), control=None)
    callback.on_step_end(args=None, state=SimpleNamespace(global_step=1), control=None)

    assert start_calls == [(True, "qaic:0", "/tmp/hw-trace"), (True, "qaic:1", "/tmp/hw-trace")]
    assert stop_calls == [(True, "qaic:0")]


def test_qaic_profiler_resolves_rank_at_start_time(monkeypatch):
    calls = []
    rank_state = {"local_rank": 0}

    monkeypatch.setattr(callbacks_module, "get_local_rank", lambda: rank_state["local_rank"])
    monkeypatch.setattr(callbacks_module, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        callbacks_module,
        "init_qaic_profiling",
        lambda use_profiler, device_type, trace_dir=None: calls.append((use_profiler, device_type, trace_dir)),
    )

    callback = QAICProfilerCallback(start_step=0, trace_dir="/tmp/hw-trace", device_ids=[10, 11])
    rank_state["local_rank"] = 1
    callback.on_step_begin(args=None, state=SimpleNamespace(global_step=0), control=None)

    assert calls == [(True, "qaic:11", "/tmp/hw-trace")]


def test_qaic_op_by_op_verifier_on_step_end_without_initialized_ctx():
    callback = QAICOpByOpVerifierCallback(start_step=0, end_step=5, trace_dir="/tmp/op-trace")
    state = SimpleNamespace(global_step=1)

    # Should not raise when on_step_end is hit before any context is initialized.
    callback.on_step_end(args=None, state=state, control=None)


def test_qaic_op_by_op_verifier_default_trace_dir_is_under_output_dir(monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/train_out")
    callback = QAICOpByOpVerifierCallback(start_step=0, end_step=1)
    assert callback.trace_dir == os.path.abspath("/tmp/train_out/qaic_op_by_op_traces")


def test_qaic_op_by_op_verifier_relative_trace_dir_is_under_output_dir(monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/train_out")
    callback = QAICOpByOpVerifierCallback(start_step=0, end_step=1, trace_dir="./custom-op-trace")
    assert callback.trace_dir == os.path.abspath("/tmp/train_out/custom-op-trace")


def test_qaic_op_by_op_verifier_absolute_trace_dir_is_preserved(monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/train_out")
    callback = QAICOpByOpVerifierCallback(start_step=0, end_step=1, trace_dir="/var/tmp/op-trace")
    assert callback.trace_dir == os.path.abspath("/var/tmp/op-trace")


def test_qaic_op_by_op_verifier_casts_numeric_config(monkeypatch):
    captured = {}

    class _DummyCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _mock_get_op_verifier_ctx(**kwargs):
        captured.update(kwargs)
        return _DummyCtx()

    monkeypatch.setattr(callbacks_module, "get_op_verifier_ctx", _mock_get_op_verifier_ctx)

    callback = QAICOpByOpVerifierCallback(
        start_step="0",
        end_step="2",
        trace_dir="/tmp/op-trace",
        atol="0.1",
        rtol="1e-5",
    )
    callback.on_step_begin(args=None, state=SimpleNamespace(global_step=0), control=None)

    assert isinstance(captured["atol"], float)
    assert isinstance(captured["rtol"], float)
    assert captured["atol"] == 0.1
    assert captured["rtol"] == 1e-5


@pytest.mark.parametrize(
    "args,expected_dtype",
    [
        (SimpleNamespace(fp16=False, bf16=True), torch.bfloat16),
        (SimpleNamespace(fp16=False, bf16=False), torch.float32),
    ],
)
def test_qaic_op_by_op_verifier_uses_training_precision_for_ref_dtype(monkeypatch, args, expected_dtype):
    captured = {}

    class _DummyCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _mock_get_op_verifier_ctx(**kwargs):
        captured.update(kwargs)
        return _DummyCtx()

    monkeypatch.setattr(callbacks_module, "get_op_verifier_ctx", _mock_get_op_verifier_ctx)

    callback = QAICOpByOpVerifierCallback(start_step=0, end_step=2, trace_dir="/tmp/op-trace")
    callback.on_step_begin(args=args, state=SimpleNamespace(global_step=0), control=None)

    assert captured["ref_dtype"] == expected_dtype


def test_qaic_op_by_op_verifier_rejects_fp16_mode():
    callback = QAICOpByOpVerifierCallback(start_step=0, end_step=2, trace_dir="/tmp/op-trace")

    with pytest.raises(RuntimeError, match="not supported with fp16/GradScaler"):
        callback.on_step_begin(
            args=SimpleNamespace(fp16=True, bf16=False),
            state=SimpleNamespace(global_step=0),
            control=None,
        )


def test_qaic_op_by_op_verifier_exits_ctx_when_global_step_reaches_end_step(monkeypatch):
    events = []

    class _DummyCtx:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")
            return False

    monkeypatch.setattr(callbacks_module, "get_op_verifier_ctx", lambda **kwargs: _DummyCtx())

    callback = QAICOpByOpVerifierCallback(start_step=0, end_step=2, trace_dir="/tmp/op-trace")

    # Enter at step 1 (still inside [start_step, end_step) ).
    callback.on_step_begin(
        args=SimpleNamespace(fp16=False, bf16=False),
        state=SimpleNamespace(global_step=1),
        control=None,
    )
    # At on_step_end, HF Trainer may already report global_step == end_step.
    callback.on_step_end(args=None, state=SimpleNamespace(global_step=2), control=None)

    assert events == ["enter", "exit"]
    assert callback.op_verifier_ctx_step is None
