"""Layerwise prefill-only compile/profile for Qwen3-VL-30B-A3B with 2 text layers.

This example compiles only the language prefill QPC.  It skips vision and
decode compile, limits ``config.text_config.num_hidden_layers`` to 2, and adds
script-local ONNX export tracing timers for Qwen3-VL MoE modules.
"""

import collections
import contextlib
import functools
import os
import threading
import time

import torch
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
LANGUAGE_NUM_LAYERS = 2

PREFILL_SEQ_LEN = 128
CTX_LEN = 4096
BATCH_SIZE = 1
HEIGHT = 354
WIDTH = 536
NUM_CORES = 16
NUM_DEVICES = 1
PROFILE_SAMPLE_INTERVAL_SECONDS = 0.5
PROFILE_TOP_N = 20


class _TimingStats:
    def __init__(self):
        self.total_seconds = 0.0
        self.self_seconds = 0.0
        self.calls = 0
        self.max_seconds = 0.0
        self.max_self_seconds = 0.0

    def add(self, elapsed_seconds, child_seconds):
        self.calls += 1
        self.total_seconds += elapsed_seconds
        self.self_seconds += elapsed_seconds - child_seconds
        self.max_seconds = max(self.max_seconds, elapsed_seconds)
        self.max_self_seconds = max(self.max_self_seconds, elapsed_seconds - child_seconds)


class _ModuleTraceProfiler:
    def __init__(self):
        self.category_stats = collections.defaultdict(_TimingStats)
        self.module_stats = collections.defaultdict(_TimingStats)
        self.export_count = 0
        self._active = False
        self._local = threading.local()

    def reset(self):
        self.category_stats.clear()
        self.module_stats.clear()
        self.export_count = 0

    @contextlib.contextmanager
    def export_scope(self, output_path):
        self.export_count += 1
        start_time = time.perf_counter()
        previous_active = self._active
        self._active = True
        try:
            yield
        finally:
            self._active = previous_active
            elapsed_seconds = time.perf_counter() - start_time
            print(f"[onnx export profile] path={output_path} time={elapsed_seconds:.2f}s", flush=True)
            self.print_summary(f"after export #{self.export_count}")

    def wrap_forward(self, category, original_forward):
        @functools.wraps(original_forward)
        def wrapped_forward(module, *args, **kwargs):
            if not self._active:
                return original_forward(module, *args, **kwargs)

            label = self._module_label(category, module)
            stack = getattr(self._local, "stack", None)
            if stack is None:
                stack = []
                self._local.stack = stack

            frame = {"category": category, "label": label, "child_seconds": 0.0}
            stack.append(frame)
            start_time = time.perf_counter()
            try:
                return original_forward(module, *args, **kwargs)
            finally:
                elapsed_seconds = time.perf_counter() - start_time
                completed_frame = stack.pop()
                child_seconds = completed_frame["child_seconds"]
                self.category_stats[category].add(elapsed_seconds, child_seconds)
                self.module_stats[label].add(elapsed_seconds, child_seconds)
                if stack:
                    stack[-1]["child_seconds"] += elapsed_seconds

        return wrapped_forward

    def _module_label(self, category, module):
        class_name = module.__class__.__name__
        parts = [category, class_name]

        window = self._current_window()
        if window is not None:
            parts.append(f"window={window}")

        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None and hasattr(module, "self_attn"):
            layer_idx = getattr(module.self_attn, "layer_idx", None)
        if layer_idx is None:
            for frame in reversed(getattr(self._local, "stack", [])):
                for token in frame["label"].split("|"):
                    if token.startswith("layer_idx="):
                        layer_idx = token.split("=", 1)[1]
                        break
                if layer_idx is not None:
                    break
        if layer_idx is not None:
            parts.append(f"layer_idx={layer_idx}")

        return "|".join(str(part) for part in parts)

    @staticmethod
    def _current_window():
        try:
            from QEfficient.transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
                QEffQwen3VLMoeTextModel,
            )

            start = getattr(QEffQwen3VLMoeTextModel, "_start", None)
            end = getattr(QEffQwen3VLMoeTextModel, "_end", None)
            if start is not None and end is not None:
                return f"{start}-{end}"
        except Exception:
            return None
        return None

    def snapshot(self):
        return {
            "categories": dict(self.category_stats),
            "modules": dict(self.module_stats),
        }

    def print_summary(self, title):
        print(f"[module trace profile] {title}: categories by self time", flush=True)
        self._print_rows(self.category_stats, use_self_time=True, limit=PROFILE_TOP_N)
        print(f"[module trace profile] {title}: modules by self time", flush=True)
        self._print_rows(self.module_stats, use_self_time=True, limit=PROFILE_TOP_N)

    @staticmethod
    def _print_rows(stats_by_name, use_self_time, limit):
        rows = sorted(
            stats_by_name.items(),
            key=lambda item: item[1].self_seconds if use_self_time else item[1].total_seconds,
            reverse=True,
        )[:limit]
        if not rows:
            print("  <no module forward calls recorded>", flush=True)
            return

        for name, stats in rows:
            avg_seconds = stats.total_seconds / stats.calls if stats.calls else 0.0
            avg_self_seconds = stats.self_seconds / stats.calls if stats.calls else 0.0
            print(
                "  "
                f"{name}: calls={stats.calls}, "
                f"self={stats.self_seconds:.2f}s, total={stats.total_seconds:.2f}s, "
                f"avg_self={avg_self_seconds:.4f}s, avg_total={avg_seconds:.4f}s, "
                f"max_self={stats.max_self_seconds:.2f}s, max_total={stats.max_seconds:.2f}s",
                flush=True,
            )


@contextlib.contextmanager
def _profile_qeff_onnx_exports(force_reexport=True):
    profiler = _ModuleTraceProfiler()

    import QEfficient.transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe as qwen3_vl_moe
    import QEfficient.utils.export_utils as qeff_export_utils

    original_torch_onnx_export = torch.onnx.export
    original_create_export_hash = qeff_export_utils.create_export_hash
    patched_forwards = []

    module_classes = [
        ("decoder_wrapper", qwen3_vl_moe.QEffQwen3VLDecoderWrapper),
        ("text_model", qwen3_vl_moe.QEffQwen3VLMoeTextModel),
        ("decoder_layer", qwen3_vl_moe.QEffQwen3VLMoeTextDecoderLayer),
        ("attention", qwen3_vl_moe.QEffQwen3VLMoeTextAttention),
        ("moe", qwen3_vl_moe.QEffQwen3VLMoeTextSparseMoeBlock),
        ("moe_chunked", qwen3_vl_moe.QEffPrefillChunkedQwen3VLMoeTextSparseMoeBlock),
        ("router", qwen3_vl_moe.QEffQwen3VLMoeTextTopKRouter),
    ]

    for category, module_class in module_classes:
        original_forward = module_class.forward
        module_class.forward = profiler.wrap_forward(category, original_forward)
        patched_forwards.append((module_class, original_forward))

    profile_run_id = str(time.time_ns())

    def profiled_create_export_hash(*args, **kwargs):
        if force_reexport:
            kwargs = dict(kwargs)
            export_kwargs = dict(kwargs.get("export_kwargs") or {})
            export_kwargs["_qeff_profile_run_id"] = profile_run_id
            kwargs["export_kwargs"] = export_kwargs
        return original_create_export_hash(*args, **kwargs)

    def profiled_torch_onnx_export(*args, **kwargs):
        output_path = kwargs.get("f")
        if output_path is None and len(args) >= 3:
            output_path = args[2]
        with profiler.export_scope(output_path):
            return original_torch_onnx_export(*args, **kwargs)

    qeff_export_utils.create_export_hash = profiled_create_export_hash
    torch.onnx.export = profiled_torch_onnx_export
    try:
        yield profiler
    finally:
        torch.onnx.export = original_torch_onnx_export
        qeff_export_utils.create_export_hash = original_create_export_hash
        for module_class, original_forward in reversed(patched_forwards):
            module_class.forward = original_forward


def _read_rss_bytes(process_id):
    try:
        with open(f"/proc/{process_id}/statm", "r", encoding="utf-8") as statm_file:
            resident_pages = int(statm_file.readline().split()[1])
    except (
        FileNotFoundError,
        PermissionError,
        ProcessLookupError,
        IndexError,
        ValueError,
    ):
        return 0
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _get_child_process_map():
    child_processes = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue

        process_id = int(entry.name)
        try:
            with open(f"/proc/{process_id}/status", "r", encoding="utf-8") as status_file:
                for line in status_file:
                    if line.startswith("PPid:"):
                        parent_process_id = int(line.split()[1])
                        child_processes.setdefault(parent_process_id, []).append(process_id)
                        break
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            IndexError,
            ValueError,
        ):
            continue
    return child_processes


def _get_process_tree_rss_bytes():
    child_processes = _get_child_process_map()
    process_ids = [os.getpid()]
    total_rss_bytes = 0

    while process_ids:
        process_id = process_ids.pop()
        total_rss_bytes += _read_rss_bytes(process_id)
        process_ids.extend(child_processes.get(process_id, ()))

    return total_rss_bytes


def _get_current_process_rss_bytes():
    with open("/proc/self/statm", "r", encoding="utf-8") as statm_file:
        resident_pages = int(statm_file.readline().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


@contextlib.contextmanager
def _compile_profile(name):
    start_process_rss = _get_current_process_rss_bytes()
    start_tree_rss = _get_process_tree_rss_bytes()
    peak_tree_rss = start_tree_rss
    stop_event = threading.Event()

    def sample_rss():
        nonlocal peak_tree_rss
        while not stop_event.wait(PROFILE_SAMPLE_INTERVAL_SECONDS):
            peak_tree_rss = max(peak_tree_rss, _get_process_tree_rss_bytes())

    sampler = threading.Thread(target=sample_rss, daemon=True)
    start_time = time.perf_counter()
    sampler.start()
    try:
        yield
    finally:
        elapsed_seconds = time.perf_counter() - start_time
        end_process_rss = _get_current_process_rss_bytes()
        end_tree_rss = _get_process_tree_rss_bytes()
        peak_tree_rss = max(peak_tree_rss, end_tree_rss)
        stop_event.set()
        sampler.join(timeout=PROFILE_SAMPLE_INTERVAL_SECONDS)

        print(
            f"[compile profile] {name}: "
            f"time={elapsed_seconds:.2f}s, "
            f"process_rss_start={_format_bytes(start_process_rss)}, "
            f"process_rss_end={_format_bytes(end_process_rss)}, "
            f"process_rss_delta={_format_bytes(end_process_rss - start_process_rss)}, "
            f"tree_rss_start={_format_bytes(start_tree_rss)}, "
            f"tree_rss_end={_format_bytes(end_tree_rss)}, "
            f"tree_rss_peak={_format_bytes(peak_tree_rss)}, "
            f"tree_rss_delta={_format_bytes(end_tree_rss - start_tree_rss)}, "
            f"tree_rss_peak_delta={_format_bytes(peak_tree_rss - start_tree_rss)}"
        )


def main():
    force_reexport = os.environ.get("QEFF_PROFILE_FORCE_REEXPORT", "1") != "0"
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.dtype = "float16"
    config.text_config.dtype = "float16"
    config.text_config.num_hidden_layers = LANGUAGE_NUM_LAYERS

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        dtype=torch.float16,
        layerwise=True,
    )

    common_compile_kwargs = dict(
        batch_size=BATCH_SIZE,
        ctx_len=CTX_LEN,
        height=HEIGHT,
        width=WIDTH,
        num_cores=NUM_CORES,
        num_devices=NUM_DEVICES,
        mos=1,
        mxfp6_matmul=True,
        aic_enable_depth_first=True,
        split_model_io=True,
        use_onnx_subfunctions=True,
        layerwise=True,
    )

    print(
        "[profile config] "
        f"force_reexport={force_reexport}, "
        f"layers={LANGUAGE_NUM_LAYERS}, "
        f"prefill_seq_len={PREFILL_SEQ_LEN}, "
        "skip_vision=True, prefill_only=True",
        flush=True,
    )
    with _profile_qeff_onnx_exports(force_reexport=force_reexport) as profiler:
        with _compile_profile("prefill"):
            prefill_qpc_path = qeff_model.compile(
                **common_compile_kwargs,
                prefill_seq_len=PREFILL_SEQ_LEN,
                mxint8_kv_cache=True,
                retain_full_kv=True,
                prefill_only=True,
                enable_chunking=True,
                skip_vision=True,
                layerwise_window_size=1,
            )
        profiler.print_summary("final")
    print(f"Prefill QPC path: {prefill_qpc_path}")


if __name__ == "__main__":
    main()
