# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Single-configuration metric runner for weight-free comparison.

Runs one mode (basic / cb / ccl / cb_ccl) with or without weight-free export,
profiles memory using QEffMemoryProfiler, and writes one JSON line to stdout.

Intended to be invoked by compare_weightfree.py — not for direct use.

Output JSON shape
-----------------
{
    "exit_code": 0,
    "export_peak_rss_mb": 27450.2,
    "export_duration_seconds": 71.4,
    "compile_peak_rss_mb": 29900.1,
    "compile_duration_seconds": 26.8,
    "onnx_size_gb": 0.932,
    "qpc_dir_size_gb": 13.473,
    "completions": ["generated text 1", ...],
    "error": null
}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# QEfficient is imported before memory_profiling so model wrappers and logging
# are initialized before the standalone runner profiler is loaded.
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM  # noqa: E402

# Memory profiler lives at <repo_root>/scripts/memory_profiling/
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from memory_profiling import QEffMemoryProfiler  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    "My name is",
    "Explain quantum computing",
    "Where is Taj Mahal located?",
    "Hey, are you conscious? Can you talk to me?",
]


def _op_peak_rss_mb(profiler: QEffMemoryProfiler, op_name: str) -> Optional[float]:
    """Peak RSS (MB) recorded during the named profiler operation."""
    ops = profiler.operations
    samples = profiler.samples
    if not samples:
        return None
    idx = next((i for i, (_, n) in enumerate(ops) if n == op_name), None)
    if idx is None:
        return None
    t_start = ops[idx][0]
    t_end = ops[idx + 1][0] if idx + 1 < len(ops) else samples[-1].timestamp
    vals = [s.rss_mb for s in samples if t_start <= s.timestamp < t_end]
    return round(max(vals), 4) if vals else None


def _dir_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024**3), 3)


def _export_dir_size_gb(onnx_path: Path) -> float:
    """Total size of the directory that holds the ONNX and its external data."""
    search = onnx_path.parent if onnx_path.is_file() else onnx_path
    total = sum(f.stat().st_size for f in search.rglob("*") if f.is_file())
    return round(total / (1024**3), 3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True)
    p.add_argument("--mode", default="basic", choices=["basic", "cb", "ccl", "cb_ccl"], help="Test mode.")
    p.add_argument("--weight_free", action="store_true", help="Use weight-free (meta-device) export.")
    p.add_argument("--ctx_len", type=int, default=128)
    p.add_argument("--prefill_seq_len", type=int, default=1)
    p.add_argument("--generation_len", type=int, default=100)
    p.add_argument("--num_devices", type=int, default=1)
    p.add_argument("--num_cores", type=int, default=16)
    p.add_argument("--full_batch_size", type=int, default=4, help="CB pool size (continuous batching modes only).")
    p.add_argument("--mxfp6_matmul", action="store_true")
    p.add_argument("--mxint8_kv_cache", action="store_true")
    p.add_argument(
        "--ccl_values",
        type=str,
        default=None,
        help="Comma-separated CCL context lengths, e.g. '64,128'. Defaults to [ctx_len//2, ctx_len].",
    )
    p.add_argument("--layers", type=int, default=None, help="Override num_hidden_layers for fast testing.")
    p.add_argument("--output_dir", required=True, help="Root directory for ONNX and QPC artifacts.")
    p.add_argument("--prompts", type=str, default=None, help="Pipe-separated prompts. Defaults to built-in set.")
    p.add_argument("--no_subfunctions", action="store_true", help="Disable ONNX subfunction extraction.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    result: dict = {
        "exit_code": 1,
        "transform_peak_rss_mb": None,
        "transform_duration_seconds": None,
        "export_peak_rss_mb": None,
        "export_duration_seconds": None,
        "compile_peak_rss_mb": None,
        "compile_duration_seconds": None,
        "onnx_size_gb": None,
        "qpc_dir_size_gb": None,
        "perf_ttft_seconds": None,
        "perf_decode_tokens_per_sec": None,
        "perf_total_tokens_per_sec": None,
        "perf_e2e_time_seconds": None,
        "completions": [],
        "error": None,
    }

    profiler = QEffMemoryProfiler(
        sampling_interval=0.05,
        verbose=False,
        track_child_processes=True,
    )
    profiler.start_monitoring()

    try:
        prompts = args.prompts.split("|") if args.prompts else DEFAULT_PROMPTS
        use_subfunctions = not args.no_subfunctions
        continuous_batching = args.mode in ("cb", "cb_ccl")
        ccl_mode = args.mode == "ccl"

        # CCL context lengths
        if ccl_mode:
            if args.ccl_values:
                ccl_list = [int(x) for x in args.ccl_values.split(",")]
            else:
                ccl_list = sorted({args.ctx_len // 2, args.ctx_len})
        else:
            ccl_list = None

        qaic_config: Optional[dict] = {"ccl_enabled": True} if ccl_mode else None

        output_dir = Path(args.output_dir)
        export_dir = output_dir / "onnx"
        compile_dir = output_dir / "qpc"
        export_dir.mkdir(parents=True, exist_ok=True)
        compile_dir.mkdir(parents=True, exist_ok=True)

        # ── Phase 1: Model loading ─────────────────────────────────────────
        profiler.mark_operation("Model Loading")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

        config = None
        try:
            config = AutoConfig.from_pretrained(args.model_name)
        except Exception:
            pass
        if config is None:
            config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

        config.dtype = torch.float32
        if not hasattr(config, "max_position_embeddings"):
            config.max_position_embeddings = getattr(config, "n_positions", 2048)
        if args.layers is not None:
            config.num_hidden_layers = args.layers

        if args.weight_free:
            with init_empty_weights():
                meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
            qeff_model = QEFFAutoModelForCausalLM(
                meta_model,
                continuous_batching=continuous_batching,
                qaic_config=qaic_config,
                pretrained_model_name_or_path=args.model_name,
            )
        else:
            qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
                args.model_name,
                attn_implementation="eager",
                config=config,
                continuous_batching=continuous_batching,
                qaic_config=qaic_config,
            )

        # ── Phase 2: Export (PyTorch → ONNX) ──────────────────────────────
        profiler.mark_operation("Model Export")
        t_export = time.perf_counter()

        onnx_path = Path(
            qeff_model.export(
                export_dir=str(export_dir),
                use_dynamo=True,
                use_weight_free_export=args.weight_free,
                use_onnx_subfunctions=use_subfunctions,
                offload_pt_weights=True,
            )
        )

        result["export_duration_seconds"] = round(time.perf_counter() - t_export, 3)
        result["onnx_size_gb"] = _export_dir_size_gb(onnx_path)

        # Read checkpoint-prep (transform) stats written by core.py into its
        # module-level variables. Only populated for weight-free export.
        if args.weight_free:
            from QEfficient.exporter.weight_free import core as _wf_core

            if getattr(_wf_core, "_checkpoint_prep_ran", False):
                result["transform_duration_seconds"] = round(_wf_core._last_prep_duration_seconds, 3)
                if _wf_core._last_prep_peak_rss_mb is not None:
                    result["transform_peak_rss_mb"] = round(_wf_core._last_prep_peak_rss_mb, 4)

        # ── Phase 3: Compile (ONNX → QPC) ─────────────────────────────────
        profiler.mark_operation("Model Compilation")
        t_compile = time.perf_counter()

        compile_kwargs: dict = dict(
            onnx_path=str(onnx_path),
            compile_dir=str(compile_dir),
            prefill_seq_len=args.prefill_seq_len,
            ctx_len=args.ctx_len,
            num_devices=args.num_devices,
            num_cores=args.num_cores,
            mxfp6_matmul=args.mxfp6_matmul,
            mxint8_kv_cache=args.mxint8_kv_cache,
            use_dynamo=True,
            use_onnx_subfunctions=use_subfunctions,
            use_weight_free_export=args.weight_free,
        )
        if continuous_batching:
            compile_kwargs["full_batch_size"] = args.full_batch_size
        if ccl_list is not None:
            compile_kwargs["comp_ctx_lengths_prefill"] = ccl_list
            compile_kwargs["comp_ctx_lengths_decode"] = ccl_list

        qpc_path = Path(qeff_model.compile(**compile_kwargs))

        result["compile_duration_seconds"] = round(time.perf_counter() - t_compile, 3)
        result["qpc_dir_size_gb"] = _dir_size_gb(qpc_path)

        # ── Phase 4: Inference ─────────────────────────────────────────────
        profiler.mark_operation("Inference")

        active_prompts = prompts if continuous_batching else prompts[:1]
        try:
            exec_info = qeff_model.generate(
                prompts=active_prompts,
                tokenizer=tokenizer,
                automation=True,
                generation_len=args.generation_len,
            )
            result["completions"] = list(exec_info.generated_texts)

            # Capture QPC inference performance metrics from exec_info
            pm = exec_info.perf_metrics
            bs = exec_info.batch_size
            result["perf_ttft_seconds"] = round(pm.prefill_time, 3)
            result["perf_decode_tokens_per_sec"] = round(pm.decode_perf * bs, 2)
            result["perf_total_tokens_per_sec"] = round(pm.total_perf * bs, 2)
            result["perf_e2e_time_seconds"] = round(pm.total_time, 3)

        except Exception as gen_exc:
            result["completions"] = []
            result["error"] = f"inference error: {gen_exc}"

        result["exit_code"] = 0

    except Exception as exc:
        result["error"] = str(exc)

    finally:
        profiler.stop_monitoring()
        result["export_peak_rss_mb"] = _op_peak_rss_mb(profiler, "Model Export")
        result["compile_peak_rss_mb"] = _op_peak_rss_mb(profiler, "Model Compilation")

    # Single JSON line — compare_weightfree.py reads this.
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
