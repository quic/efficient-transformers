#!/usr/bin/env python3
"""
Run CSV-driven LLM serving benchmarks through vLLM QAIC.

The script supports the four initial LLM configurations used by the Jenkins
pipeline:

* default / CB + subfunction: python -m vllm.entrypoints.openai.api_server
* CCL enabled: python -m vllm.entrypoints.openai.api_server
* blocking: python -m vllm.entrypoints.openai.api_server
* disagg PD: python -m qaic_disagg

Each CSV row describes one model run. The runner launches the server, waits for
the ready marker, runs a benchmark client, parses the "Serving Benchmark Result"
block, and appends one or more rows to the requested output CSV.

Shared CSV parsing, command building, and subprocess orchestration live in
vllm_benchmark_common.py so the embedding/audio runners can reuse them.
"""

from __future__ import annotations

from vllm_benchmark_common import build_arg_parser, run_benchmarks

LATEST_MODELS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "openai/gpt-oss-20b",
    "meta-llama/Llama-3.3-70B-Instruct",
}


def main() -> int:
    args = build_arg_parser("Run LLM vLLM QAIC benchmark CSV rows.").parse_args()
    return run_benchmarks(args, LATEST_MODELS)


if __name__ == "__main__":
    raise SystemExit(main())
