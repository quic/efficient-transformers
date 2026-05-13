# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from types import SimpleNamespace

import pytest
from transformers import TrainerCallback

from QEfficient.finetune.experimental.core import callbacks as callbacks_module
from QEfficient.finetune.experimental.core.callbacks import QAICProfilerCallback
from QEfficient.finetune.experimental.core.component_registry import ComponentFactory, registry


class ModelSummaryCallback(TrainerCallback):
    def __init__(self):
        pass


# Setup test data
CALLBACK_CONFIGS = {
    "early_stopping": {
        "name": "early_stopping",
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.001,
    },
    "tensorboard": {"name": "tensorboard", "tb_writer": "SummaryWriter"},
    "model_summary": {
        "name": "model_summary",
        "max_depth": 1,
    },
}

REGISTRY_CALLBACK_CONFIGS = {
    "model_summary": {
        "name": "model_summary",
        "max_depth": 1,
        "callback_class": ModelSummaryCallback,
    },
}


@pytest.mark.parametrize("callback_name", CALLBACK_CONFIGS.keys())
def test_callbacks(callback_name):
    """Test that registered callbacks that can be created with their configs."""
    # Create callbacks using the factory
    config = CALLBACK_CONFIGS[callback_name]
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
