# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import functools
import gc
import os
import resource
import sys
import threading
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

try:
    import psutil
except Exception:  # pragma: no cover - psutil is optional for profiling only.
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - keeps this utility import-safe.
    torch = None


_ENABLED_VALUES = {"1", "true", "yes", "on"}


def is_export_memory_profile_enabled() -> bool:
    return os.environ.get("QEFF_PROFILE_EXPORT_MEMORY", "").lower() in _ENABLED_VALUES


def _format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "n/a"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def format_bytes(num_bytes: Optional[int]) -> str:
    return _format_bytes(num_bytes)


def _proc_status_value(name: str) -> Optional[int]:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith(name + ":"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def _ru_maxrss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak
    return peak * 1024


@dataclass
class MemorySnapshot:
    label: str
    timestamp: float
    rss: Optional[int]
    uss: Optional[int]
    vms: Optional[int]
    hwm: Optional[int]
    ru_maxrss: Optional[int]
    py_current: Optional[int]
    py_peak: Optional[int]
    extra: Dict[str, object] = field(default_factory=dict)


class ExportMemoryProfiler:
    def __init__(self, name: str, enabled: Optional[bool] = None) -> None:
        self.name = name
        self.enabled = is_export_memory_profile_enabled() if enabled is None else enabled
        self._process = psutil.Process(os.getpid()) if self.enabled and psutil is not None else None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sampler: Optional[threading.Thread] = None
        self._sampling_interval = float(os.environ.get("QEFF_PROFILE_MEMORY_INTERVAL_SEC", "0.10"))
        self._current_stage = "idle"
        self._stage_peak_rss: Dict[str, int] = {}
        self._global_peak_rss = 0
        self._global_peak_hwm = 0
        self._global_peak_ru_maxrss = 0
        self._snapshots: list[MemorySnapshot] = []
        self._tracemalloc_started = False

    def __enter__(self) -> "ExportMemoryProfiler":
        if not self.enabled:
            return self
        if os.environ.get("QEFF_PROFILE_PYTHON_MEMORY", "").lower() in _ENABLED_VALUES and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_started = True
        self.snapshot("start", self._static_memory_summary())
        self._sampler = threading.Thread(target=self._sample_loop, name="qeff-export-memory-profiler", daemon=True)
        self._sampler.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.enabled:
            return
        self.snapshot("end", {"exception": exc_type.__name__ if exc_type else None})
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=1.0)
        self.report()
        if self._tracemalloc_started:
            tracemalloc.stop()

    @contextmanager
    def stage(self, label: str, extra: Optional[Dict[str, object]] = None):
        if not self.enabled:
            yield
            return
        previous_stage = self._current_stage
        self._current_stage = label
        gc.collect()
        start = time.perf_counter()
        start_snapshot = self.snapshot(label + ":begin", extra)
        try:
            yield
        finally:
            gc.collect()
            elapsed = time.perf_counter() - start
            end_snapshot = self.snapshot(label + ":end", {"elapsed_sec": f"{elapsed:.3f}"})
            peak = self._stage_peak_rss.get(label)
            delta = None
            hwm_delta = None
            ru_delta = None
            if start_snapshot.rss is not None and end_snapshot.rss is not None:
                delta = end_snapshot.rss - start_snapshot.rss
            if start_snapshot.hwm is not None and end_snapshot.hwm is not None:
                hwm_delta = end_snapshot.hwm - start_snapshot.hwm
            if start_snapshot.ru_maxrss is not None and end_snapshot.ru_maxrss is not None:
                ru_delta = end_snapshot.ru_maxrss - start_snapshot.ru_maxrss
            self._emit(
                "stage "
                f"{label}: elapsed={elapsed:.3f}s rss_delta={_format_bytes(delta)} "
                f"hwm_delta={_format_bytes(hwm_delta)} ru_maxrss_delta={_format_bytes(ru_delta)} "
                f"stage_peak_rss={_format_bytes(peak)}"
            )
            self._current_stage = previous_stage

    def snapshot(self, label: str, extra: Optional[Dict[str, object]] = None) -> MemorySnapshot:
        if not self.enabled:
            return MemorySnapshot(label, time.time(), None, None, None, None, None, None, None, extra or {})
        rss = uss = vms = None
        if self._process is not None:
            mem = self._process.memory_info()
            rss = mem.rss
            vms = mem.vms
            try:
                uss = self._process.memory_full_info().uss
            except Exception:
                uss = None
        else:
            rss = _proc_status_value("VmRSS")
            vms = _proc_status_value("VmSize")
        hwm = _proc_status_value("VmHWM")
        ru_maxrss = _ru_maxrss_bytes()
        py_current = py_peak = None
        if tracemalloc.is_tracing():
            py_current, py_peak = tracemalloc.get_traced_memory()
        snapshot = MemorySnapshot(label, time.time(), rss, uss, vms, hwm, ru_maxrss, py_current, py_peak, extra or {})
        with self._lock:
            self._snapshots.append(snapshot)
            if rss is not None:
                self._global_peak_rss = max(self._global_peak_rss, rss)
            if hwm is not None:
                self._global_peak_hwm = max(self._global_peak_hwm, hwm)
            if ru_maxrss is not None:
                self._global_peak_ru_maxrss = max(self._global_peak_ru_maxrss, ru_maxrss)
        self._emit_snapshot(snapshot)
        return snapshot

    def tensor_summary(self, label: str, tensors: Iterable[object]) -> None:
        if not self.enabled:
            return
        count = 0
        bytes_by_dtype: Dict[str, int] = {}
        devices: Dict[str, int] = {}
        for tensor in tensors:
            if torch is None or not isinstance(tensor, torch.Tensor):
                continue
            count += 1
            bytes_by_dtype[str(tensor.dtype)] = (
                bytes_by_dtype.get(str(tensor.dtype), 0) + tensor.numel() * tensor.element_size()
            )
            devices[str(tensor.device)] = devices.get(str(tensor.device), 0) + 1
        total_bytes = sum(bytes_by_dtype.values())
        self._emit(
            f"tensor_summary {label}: count={count} total={_format_bytes(total_bytes)} "
            f"by_dtype={{{', '.join(f'{k}: {_format_bytes(v)}' for k, v in sorted(bytes_by_dtype.items()))}}} "
            f"devices={devices}"
        )

    def report(self) -> None:
        if not self.enabled:
            return
        self._emit(
            f"report {self.name}: sampled_peak_rss={_format_bytes(self._global_peak_rss)} "
            f"peak_hwm={_format_bytes(self._global_peak_hwm)} "
            f"peak_ru_maxrss={_format_bytes(self._global_peak_ru_maxrss)}"
        )
        if self._stage_peak_rss:
            stages = ", ".join(f"{stage}={_format_bytes(value)}" for stage, value in self._stage_peak_rss.items())
            self._emit(f"report stage_peaks: {stages}")

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            rss = None
            if self._process is not None:
                try:
                    rss = self._process.memory_info().rss
                except Exception:
                    rss = None
            else:
                rss = _proc_status_value("VmRSS")
            if rss is not None:
                with self._lock:
                    self._global_peak_rss = max(self._global_peak_rss, rss)
                    self._stage_peak_rss[self._current_stage] = max(
                        self._stage_peak_rss.get(self._current_stage, 0), rss
                    )
            self._stop.wait(self._sampling_interval)

    def _emit_snapshot(self, snapshot: MemorySnapshot) -> None:
        extra = ""
        if snapshot.extra:
            extra = " " + " ".join(f"{key}={value}" for key, value in snapshot.extra.items())
        self._emit(
            f"snapshot {snapshot.label}: rss={_format_bytes(snapshot.rss)} uss={_format_bytes(snapshot.uss)} "
            f"vms={_format_bytes(snapshot.vms)} hwm={_format_bytes(snapshot.hwm)} "
            f"ru_maxrss={_format_bytes(snapshot.ru_maxrss)} py={_format_bytes(snapshot.py_current)}/"
            f"{_format_bytes(snapshot.py_peak)}{extra}"
        )

    def _emit(self, message: str) -> None:
        print(f"[QEFF-MEM] {message}", file=sys.stderr, flush=True)

    def _static_memory_summary(self) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "pid": os.getpid(),
            "psutil": psutil is not None,
            "sampling_interval_sec": self._sampling_interval,
        }
        if torch is not None:
            summary["torch"] = getattr(torch, "__version__", "unknown")
            summary["default_dtype"] = str(torch.get_default_dtype())
        return summary


def flatten_torch_tensors(value: object):
    if torch is not None and isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_torch_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_torch_tensors(item)


def onnx_initializer_summary(model) -> Dict[str, object]:
    try:
        from onnx import TensorProto
    except Exception:
        return {}
    dtype_sizes = {
        TensorProto.FLOAT: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.STRING: 0,
        TensorProto.BOOL: 1,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.UINT32: 4,
        TensorProto.UINT64: 8,
        TensorProto.COMPLEX64: 8,
        TensorProto.COMPLEX128: 16,
        TensorProto.BFLOAT16: 2,
    }
    bytes_by_type: Dict[str, int] = {}
    raw_bytes = 0
    external_count = 0
    initializer_count = 0
    for tensor in model.graph.initializer:
        initializer_count += 1
        numel = 1
        for dim in tensor.dims:
            numel *= dim
        estimated = numel * dtype_sizes.get(tensor.data_type, 0)
        dtype = TensorProto.DataType.Name(tensor.data_type)
        bytes_by_type[dtype] = bytes_by_type.get(dtype, 0) + estimated
        raw_bytes += len(tensor.raw_data)
        if tensor.data_location == TensorProto.EXTERNAL:
            external_count += 1
    return {
        "initializer_count": initializer_count,
        "initializer_estimated_bytes": _format_bytes(sum(bytes_by_type.values())),
        "initializer_raw_bytes_loaded": _format_bytes(raw_bytes),
        "initializer_external_count": external_count,
        "initializer_by_type": {key: _format_bytes(value) for key, value in sorted(bytes_by_type.items())},
    }


@contextmanager
def profile_torch_onnx_internals(profiler: ExportMemoryProfiler):
    if not profiler.enabled or torch is None:
        yield
        return

    try:
        import torch.onnx.utils as onnx_utils
    except Exception:
        yield
        return

    names = (
        "_trace_and_get_graph_from_model",
        "_create_jit_graph",
        "_optimize_graph",
        "_model_to_graph",
        "_export",
    )
    originals = {}

    def wrap(name, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with profiler.stage(f"torch.onnx.utils.{name}"):
                return fn(*args, **kwargs)

        return wrapped

    for name in names:
        fn = getattr(onnx_utils, name, None)
        if fn is None:
            continue
        originals[name] = fn
        setattr(onnx_utils, name, wrap(name, fn))
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(onnx_utils, name, fn)
