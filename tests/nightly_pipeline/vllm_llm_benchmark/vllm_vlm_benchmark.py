#!/usr/bin/env python3
"""
Run CSV-driven vision-language model serving benchmarks through vLLM QAIC.

Each CSV row describes one VLM run with configurable disagg layout (disagg_mode:
ED, PD, EPD), specialization (specialization_mode: single/multi), blocking
(enable_blocking: true/false), and CCL (enable_ccl: true/false) settings.
Shared CSV parsing, command building, and subprocess orchestration live in
vllm_benchmark_common.py, shared with the LLM, embedding, and audio runners.
"""

from __future__ import annotations

from vllm_benchmark_common import build_arg_parser, run_benchmarks

LATEST_MODELS = {
    # "google/gemma-4-26B-A4B-it",
    "google/gemma-4-E2B-it",
    # "google/gemma-4-E4B-it",
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.5-27B",
    # "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    # "Qwen/Qwen3-VL-235B-A22B-Instruct",
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen2.5-VL-3B-Instruct",
}


def main() -> int:
    args = build_arg_parser("Run VLM vLLM QAIC benchmark CSV rows.").parse_args()
    return run_benchmarks(args, LATEST_MODELS)


if __name__ == "__main__":
    raise SystemExit(main())
