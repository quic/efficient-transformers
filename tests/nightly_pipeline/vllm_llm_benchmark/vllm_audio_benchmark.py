#!/usr/bin/env python3
"""
Run CSV-driven audio-model serving benchmarks through vLLM QAIC.

Each CSV row describes one whisper model run. Shared CSV parsing, command
building, and subprocess orchestration live in vllm_benchmark_common.py,
shared with the LLM and embedding runners.
"""

from __future__ import annotations

from vllm_benchmark_common import build_arg_parser, run_benchmarks

LATEST_MODELS = {
    "openai/whisper-tiny",
}


def main() -> int:
    args = build_arg_parser("Run audio vLLM QAIC benchmark CSV rows.").parse_args()
    return run_benchmarks(args, LATEST_MODELS)


if __name__ == "__main__":
    raise SystemExit(main())
