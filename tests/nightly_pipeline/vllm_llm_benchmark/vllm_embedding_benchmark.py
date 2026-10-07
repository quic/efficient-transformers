#!/usr/bin/env python3
"""
Run CSV-driven embedding-model serving benchmarks through vLLM QAIC.

Each CSV row describes one (model, pooling_method) run. Shared CSV parsing,
command building, and subprocess orchestration live in
vllm_benchmark_common.py, shared with the LLM and audio runners.
"""

from __future__ import annotations

from vllm_benchmark_common import build_arg_parser, run_benchmarks

LATEST_MODELS = {
    "BAAI/bge-base-en-v1.5",
}


def main() -> int:
    args = build_arg_parser("Run embedding vLLM QAIC benchmark CSV rows.").parse_args()
    return run_benchmarks(args, LATEST_MODELS)


if __name__ == "__main__":
    raise SystemExit(main())
