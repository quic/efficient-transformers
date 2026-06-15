# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Greedy probe for safely disabling legacy torch.onnx export passes.

This script compiles decode-only layerwise QPCs for ``tiny-random/qwen3-vl-moe``.
A pass is considered safe to disable only when qaic-compile produces a QPC with
``programqpc.bin``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "tiny-random/qwen3-vl-moe"
DEFAULT_MODE = "greedy"

CANDIDATE_PASSES = [
    "_jit_pass_constant_propagation",
    "_jit_pass_dce",
    "_jit_pass_cse",
    "_jit_pass_canonicalize_graph_fuser_ops",
    "_jit_pass_peephole",
    "_jit_pass_fuse_addmm",
    "_jit_pass_onnx_peephole",
    "_jit_pass_onnx_eval_peephole",
    "_jit_pass_onnx_constant_fold",
    "_jit_pass_dce_allow_deleting_nodes_with_side_effects",
    "_jit_pass_canonicalize",
    "_jit_pass_onnx_graph_shape_type_inference",
    "_jit_pass_onnx_set_dynamic_input_shape",
    "_jit_pass_onnx_deduplicate_initializers",
]

SAFE_PASSES = [
    "_jit_pass_constant_propagation",
    "_jit_pass_dce",
    "_jit_pass_cse",
    "_jit_pass_canonicalize_graph_fuser_ops",
    "_jit_pass_peephole",
    "_jit_pass_fuse_addmm",
    "_jit_pass_onnx_eval_peephole",
    "_jit_pass_onnx_constant_fold",
    "_jit_pass_dce_allow_deleting_nodes_with_side_effects",
    "_jit_pass_canonicalize",
    "_jit_pass_onnx_graph_shape_type_inference",
    "_jit_pass_onnx_deduplicate_initializers",
]

MAX_ERROR_CHARS = 1200


@dataclass
class AttemptResult:
    label: str
    disabled_passes: tuple[str, ...]
    success: bool
    elapsed_seconds: float
    export_seconds: float
    qpc_path: str | None
    error: str | None


def _noop(*args, **kwargs):
    return None


def _false_noop(*args, **kwargs):
    return False


def _identity_first_arg(*args, **kwargs):
    return args[0] if args else None


def _second_arg_or_empty_dict(*args, **kwargs):
    if len(args) >= 2:
        return args[1]
    return kwargs.get("params_dict", {})


def _same_params_dict(*args, **kwargs):
    if len(args) >= 2:
        return args[1]
    return kwargs.get("params_dict", {})


PASS_REPLACEMENTS = {
    "_jit_pass_constant_propagation": _noop,
    "_jit_pass_dce": _noop,
    "_jit_pass_cse": _false_noop,
    "_jit_pass_canonicalize_graph_fuser_ops": _noop,
    "_jit_pass_peephole": _noop,
    "_jit_pass_fuse_addmm": _noop,
    "_jit_pass_onnx_peephole": _noop,
    "_jit_pass_onnx_eval_peephole": _same_params_dict,
    "_jit_pass_onnx_constant_fold": _same_params_dict,
    "_jit_pass_dce_allow_deleting_nodes_with_side_effects": _noop,
    "_jit_pass_canonicalize": _identity_first_arg,
    "_jit_pass_onnx_graph_shape_type_inference": _noop,
    "_jit_pass_onnx_set_dynamic_input_shape": _noop,
    "_jit_pass_onnx_deduplicate_initializers": _second_arg_or_empty_dict,
}


class ExportTimer:
    def __init__(self):
        self.total_seconds = 0.0
        self.calls = 0

    @contextlib.contextmanager
    def patch_export(self):
        original_export = torch.onnx.export

        def timed_export(*args, **kwargs):
            output_path = kwargs.get("f")
            if output_path is None and len(args) >= 3:
                output_path = args[2]
            start_time = time.perf_counter()
            try:
                return original_export(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start_time
                self.total_seconds += elapsed
                self.calls += 1
                print(f"[probe export] path={output_path} time={elapsed:.2f}s", flush=True)

        torch.onnx.export = timed_export
        try:
            yield
        finally:
            torch.onnx.export = original_export


@contextlib.contextmanager
def disabled_onnx_passes(pass_names: Iterable[str]):
    originals = []
    for pass_name in pass_names:
        if pass_name not in PASS_REPLACEMENTS:
            raise KeyError(f"No replacement registered for {pass_name}")
        if not hasattr(torch._C, pass_name):
            raise AttributeError(f"torch._C has no pass named {pass_name}")
        originals.append((pass_name, getattr(torch._C, pass_name)))
        setattr(torch._C, pass_name, PASS_REPLACEMENTS[pass_name])
    try:
        yield
    finally:
        for pass_name, original in reversed(originals):
            setattr(torch._C, pass_name, original)


@contextlib.contextmanager
def unique_export_hash(label: str, disabled_passes: tuple[str, ...]):
    import QEfficient.utils.export_utils as qeff_export_utils

    original_create_export_hash = qeff_export_utils.create_export_hash
    run_id = str(time.time_ns())

    def profiled_create_export_hash(*args, **kwargs):
        kwargs = dict(kwargs)
        export_kwargs = dict(kwargs.get("export_kwargs") or {})
        export_kwargs["_qeff_pass_probe_label"] = label
        export_kwargs["_qeff_pass_probe_disabled"] = list(disabled_passes)
        export_kwargs["_qeff_pass_probe_run_id"] = run_id
        kwargs["export_kwargs"] = export_kwargs
        return original_create_export_hash(*args, **kwargs)

    qeff_export_utils.create_export_hash = profiled_create_export_hash
    try:
        yield
    finally:
        qeff_export_utils.create_export_hash = original_create_export_hash


def _iter_qpc_paths(value):
    if value is None:
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_qpc_paths(nested)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _iter_qpc_paths(nested)
        return
    yield Path(str(value))


def _valid_qpc_path(value) -> Path | None:
    for path in _iter_qpc_paths(value):
        if (path / "programqpc.bin").is_file():
            return path
    return None


def _truncate_error(error: BaseException) -> str:
    message = "".join(traceback.format_exception_only(type(error), error)).strip()
    if len(message) <= MAX_ERROR_CHARS:
        return message
    return message[:MAX_ERROR_CHARS] + "...<truncated>"


def _make_model():
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.dtype = "float16"
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config.dtype = "float16"
    return QEFFAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        dtype=torch.float16,
        layerwise=True,
    )


def _compile_kwargs(compile_dir: Path):
    return dict(
        compile_dir=str(compile_dir),
        batch_size=1,
        prefill_seq_len=1,
        ctx_len=128,
        num_cores=16,
        num_devices=1,
        height=56,
        width=56,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        skip_vision=True,
        split_retained_state_io=True,
        use_onnx_subfunctions=True,
        mos=1,
        layerwise=True,
        layerwise_window_size=1,
    )


def run_attempt(label: str, disabled_passes: Iterable[str], output_root: Path, clean: bool) -> AttemptResult:
    disabled_tuple = tuple(disabled_passes)
    attempt_root = output_root / label
    if clean and attempt_root.exists():
        shutil.rmtree(attempt_root)
    attempt_root.mkdir(parents=True, exist_ok=True)

    timer = ExportTimer()
    start_time = time.perf_counter()
    qpc_path = None
    error = None
    success = False
    print(
        f"[probe attempt] label={label} disabled={','.join(disabled_tuple) or '<none>'}",
        flush=True,
    )
    try:
        with unique_export_hash(label, disabled_tuple), disabled_onnx_passes(disabled_tuple), timer.patch_export():
            qeff_model = _make_model()
            result = qeff_model.compile(**_compile_kwargs(attempt_root))
        qpc = _valid_qpc_path(result)
        if qpc is None:
            raise RuntimeError(f"Compile returned no QPC with programqpc.bin: {result}")
        qpc_path = str(qpc)
        success = True
    except Exception as exc:  # noqa: BLE001 - experiment runner reports all failures.
        error = _truncate_error(exc)
    elapsed = time.perf_counter() - start_time
    print(
        "[probe result] "
        f"label={label} success={success} elapsed={elapsed:.2f}s "
        f"export={timer.total_seconds:.2f}s qpc={qpc_path or '<none>'}",
        flush=True,
    )
    if error:
        print(f"[probe error] label={label} {error}", flush=True)
    return AttemptResult(label, disabled_tuple, success, elapsed, timer.total_seconds, qpc_path, error)


def _print_summary(results: list[AttemptResult], safe_passes: list[str], unsafe_passes: list[str]) -> None:
    print("\n[probe summary] attempts", flush=True)
    for result in results:
        print(
            "[probe summary] "
            f"label={result.label} success={result.success} "
            f"elapsed={result.elapsed_seconds:.2f}s export={result.export_seconds:.2f}s "
            f"disabled={','.join(result.disabled_passes) or '<none>'} "
            f"qpc={result.qpc_path or '<none>'}",
            flush=True,
        )
    print(f"[probe summary] safe_disable_set={','.join(safe_passes) or '<none>'}", flush=True)
    print(f"[probe summary] unsafe_passes={','.join(unsafe_passes) or '<none>'}", flush=True)


def run_baseline(output_root: Path, clean: bool) -> int:
    result = run_attempt("baseline", (), output_root, clean)
    _print_summary([result], [], [])
    return 0 if result.success else 1


def run_individual(output_root: Path, clean: bool) -> int:
    results = []
    baseline = run_attempt("individual_baseline", (), output_root, clean)
    results.append(baseline)
    if not baseline.success:
        _print_summary(results, [], SAFE_PASSES)
        return 1

    baseline_export = baseline.export_seconds
    failed_passes = []
    for index, pass_name in enumerate(SAFE_PASSES, start=1):
        label = f"individual_{index:02d}_{pass_name.removeprefix('_jit_pass_')}"
        result = run_attempt(label, (pass_name,), output_root, clean)
        results.append(result)
        if not result.success:
            failed_passes.append(pass_name)
        delta = baseline_export - result.export_seconds
        pct = (delta / baseline_export * 100.0) if baseline_export else 0.0
        print(
            "[probe individual] "
            f"pass={pass_name} success={result.success} "
            f"baseline_export={baseline_export:.2f}s export={result.export_seconds:.2f}s "
            f"delta={delta:.2f}s pct={pct:.1f}%",
            flush=True,
        )

    _print_summary(results, SAFE_PASSES, failed_passes)
    return 0 if not failed_passes else 1


def run_greedy(output_root: Path, clean: bool) -> int:
    results = []
    baseline = run_attempt("baseline", (), output_root, clean)
    results.append(baseline)
    if not baseline.success:
        _print_summary(results, [], CANDIDATE_PASSES)
        return 1

    safe_passes: list[str] = []
    unsafe_passes: list[str] = []
    for pass_name in CANDIDATE_PASSES:
        candidate_passes = [*safe_passes, pass_name]
        label = f"try_{len(results):02d}_{pass_name.removeprefix('_jit_pass_')}"
        result = run_attempt(label, candidate_passes, output_root, clean)
        results.append(result)
        if result.success:
            safe_passes.append(pass_name)
            print(f"[probe decision] keep_disabled={pass_name}", flush=True)
        else:
            unsafe_passes.append(pass_name)
            print(f"[probe decision] restore={pass_name}", flush=True)

    if safe_passes:
        final_result = run_attempt("final_safe_set", safe_passes, output_root, clean)
        results.append(final_result)
        if not final_result.success:
            print("[probe decision] final safe-set verification failed", flush=True)
            _print_summary(results, safe_passes, unsafe_passes)
            return 1

    _print_summary(results, safe_passes, unsafe_passes)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("baseline", "greedy", "individual"),
        default=os.environ.get("QEFF_PASS_PROBE_MODE", DEFAULT_MODE),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("QEFF_PASS_PROBE_OUTPUT_ROOT", "/tmp/qeff_pass_probe_decode_tiny")),
    )
    parser.add_argument("--no-clean", action="store_true", help="Keep existing attempt directories before rerunning.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    clean = not args.no_clean
    print(
        f"[probe config] model={MODEL_ID} mode={args.mode} output_root={args.output_root} clean={clean}",
        flush=True,
    )
    if args.mode == "baseline":
        return run_baseline(args.output_root, clean)
    if args.mode == "individual":
        return run_individual(args.output_root, clean)
    return run_greedy(args.output_root, clean)


if __name__ == "__main__":
    raise SystemExit(main())
