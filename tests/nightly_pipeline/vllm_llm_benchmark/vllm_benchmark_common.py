#!/usr/bin/env python3
"""
Shared CSV-driven orchestration for vLLM QAIC serving benchmarks.

This module holds every piece of logic that does not vary by model domain
(LLM, embedding, audio): CSV parsing, command builders, server/client
subprocess management, benchmark-result parsing, and output-row writing.

Each domain-specific runner script (vllm_llm_benchmark.py,
vllm_embedding_benchmark.py, vllm_audio_benchmark.py) imports
`build_arg_parser` and `run_benchmarks` from here and supplies only its own
`LATEST_MODELS` curated set and CLI description.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


API_SERVER_READY_MARKERS = (
    "Application startup complete",
    "Uvicorn running on",
)
DISAGG_READY_MARKERS = ("Press Ctl-C once to shutdown all services.",)

SKIPPED_MODELS = {
    "zai-org/GLM-4.5",
    "hpcai-tech/grok-1",
}

BENCH_BLOCK_RE = re.compile(r"=+\s*Serving Benchmark Result\s*=+(.*?)=+", re.DOTALL)
BENCH_KV_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9 /().\-]*?):\s+([-+]?\d+(?:\.\d+)?)\s*$",
    re.MULTILINE,
)

BENCH_LABEL_TO_KEY = {
    "Successful requests": "successful_requests",
    "Failed requests": "failed_requests",
    "Maximum request concurrency": "max_request_concurrency",
    "Benchmark duration (s)": "benchmark_duration_s",
    "Total input tokens": "total_input_tokens",
    "Total generated tokens": "total_generated_tokens",
    "Request throughput (req/s)": "request_throughput_req_s",
    "Output token throughput (tok/s)": "output_token_throughput_tok_s",
    "Peak output token throughput (tok/s)": "peak_output_token_throughput_tok_s",
    "Peak concurrent requests": "peak_concurrent_requests",
    "Total token throughput (tok/s)": "total_token_throughput_tok_s",
    "Mean TTFT (ms)": "mean_ttft_ms",
    "Median TTFT (ms)": "median_ttft_ms",
    "P99 TTFT (ms)": "p99_ttft_ms",
    "Mean TPOT (ms)": "mean_tpot_ms",
    "Median TPOT (ms)": "median_tpot_ms",
    "P99 TPOT (ms)": "p99_tpot_ms",
    "Mean ITL (ms)": "mean_itl_ms",
    "Median ITL (ms)": "median_itl_ms",
    "P99 ITL (ms)": "p99_itl_ms",
}

INT_RESULT_KEYS = {
    "successful_requests",
    "failed_requests",
    "total_input_tokens",
    "total_generated_tokens",
}

OUTPUT_FIELDS = [
    "run_timestamp_utc",
    "config_name",
    "status",
    "error",
    "model",
    "server_type",
    "config_summary",
    "mode_type",
    "device_group",
    "encode_device_group",
    "prefill_device_group",
    "decode_device_group",
    "BS",
    "VBS",
    "PBS",
    "DBS",
    "PL",
    "GL",
    "CL",
    "num_prompts",
    "max_concurrency",
    "failed_requests",
    "benchmark_duration_s",
    "request_throughput_req_s",
    "output_token_throughput_tok_s",
    "total_token_throughput_tok_s",
    "mean_TTFT_ms",
    "P99_TTFT_ms",
    "mean_TPOT_ms",
    "P99_TPOT_ms",
    "mean_ITL_ms",
    "P99_ITL_ms",
    "decode_TPS",
    "vllm_qaic_branch",
    "qaic_disagg_branch",
    "qserve_branch",
    "qeff_branch",
    "qaic_sdk_version",
    "server_command",
    "client_command",
    "server_log",
    "client_log",
    "pooling_method",
]

PUBLISHED_FIELDS = [
    "model",
    "model_category",
    "config_name",
    "config_summary",
    "status",
    "mean_ttft_s",
    "mean_tpot_s",
    "mean_itl_s",
    "decode_TPS",
    "request_throughput_req_s",
    "vllm_qaic_branch",
    "qaic_disagg_branch",
    "qserve_branch",
    "qeff_branch",
    "qaic_sdk_version",
    "server_command",
    "client_command",
]


class BenchmarkError(RuntimeError):
    """Raised when a benchmark row cannot be completed."""


@dataclass
class ServerProcess:
    process: subprocess.Popen
    log_path: Path
    ready_event: threading.Event
    stream_thread: threading.Thread
    started_at: float
    ready_at: float | None = None

    @property
    def ready_time_s(self) -> float | None:
        if self.ready_at is None:
            return None
        return self.ready_at - self.started_at

    @property
    def returncode(self) -> int | None:
        return self.process.poll()


def value(row: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        raw = row.get(key)
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()
    return default


def parse_bool(raw: object, default: bool = False) -> bool:
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text == "":
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected boolean value, got {raw!r}")


def optional_int(row: dict, *keys: str, default: int | None = None) -> int | None:
    raw = value(row, *keys)
    if raw == "":
        return default
    return int(raw)


def add_value_arg(cmd: list[str], flag: str, row: dict, *keys: str) -> None:
    raw = value(row, *keys)
    if raw != "":
        cmd.extend([flag, raw])


def add_bool_arg(cmd: list[str], flag: str, row: dict, key: str) -> None:
    if parse_bool(row.get(key), default=False):
        cmd.append(flag)


def parse_json_cell(raw: str, field_name: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name}: invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"{field_name}: expected a JSON object")
    return parsed


def compact_json(data: dict) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=False)


def parse_device_group(raw: str) -> list[int]:
    values = []
    for item in re.split(r"[, ]+", raw.strip()):
        if item:
            values.append(int(item))
    return values


def device_count(raw: str) -> int:
    raw = (raw or "").strip()
    if not raw:
        return 0
    total = 0
    for token in raw.split():
        if ":" in token:
            parts = [p for p in token.split(":") if p]
            if len(parts) == 2:
                total += max(0, int(parts[1]) - int(parts[0]))
            elif len(parts) == 3:
                total += len(range(int(parts[0]), int(parts[1]), int(parts[2])))
            continue
        total += len([p for p in token.split(",") if p.strip()])
    return total


def split_space_tokens(raw: str) -> list[str]:
    return [token for token in (raw or "").split() if token]


def iter_ports(raw: str) -> Iterable[int]:
    for token in split_space_tokens(raw):
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                pieces = [int(p) for p in part.split(":") if p]
                if len(pieces) == 2:
                    yield from range(pieces[0], pieces[1] + 1)
                elif len(pieces) == 3:
                    yield from range(pieces[0], pieces[1] + 1, pieces[2])
                continue
            yield int(part)


def sanitize_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())
    return cleaned.strip("_").lower() or "unknown"


def command_to_shell_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def build_api_additional_config(row: dict) -> str:
    raw = value(row, "additional_config", "additional_config_json")
    if raw:
        # Validate early, but preserve a compact form in the command.
        return compact_json(parse_json_cell(raw, "additional_config"))

    additional_config: dict = {}
    device_group = value(row, "device_group")
    if device_group:
        additional_config["device_group"] = parse_device_group(device_group)

    override_config = parse_json_cell(
        value(row, "override_qaic_config"),
        "override_qaic_config",
    )

    for csv_key, json_key in [
        ("use_onnx_subfunctions", "use_onnx_subfunctions"),
        ("aic_enable_depth_first", "aic_enable_depth_first"),
        ("user_tiled", "user_tiled"),
        ("allow_mxint8_mdp_io", "allow_mxint8_mdp_io"),
        ("mxfp6_matmul", "mxfp6_matmul"),
        ("mxint8_kv_cache", "mxint8_kv_cache"),
    ]:
        raw_value = value(row, csv_key)
        if raw_value:
            override_config[json_key] = parse_bool(raw_value)

    qaic_config = parse_json_cell(value(row, "qaic_config"), "qaic_config")
    if value(row, "enable_blocking"):
        qaic_config["enable_blocking"] = parse_bool(row.get("enable_blocking"))
    if value(row, "blocking_mode"):
        qaic_config["blocking_mode"] = value(row, "blocking_mode")
    if value(row, "skip_kv"):
        qaic_config["skip_kv"] = parse_bool(row.get("skip_kv"))
    if qaic_config:
        override_config["qaic_config"] = qaic_config

    if override_config:
        additional_config["override_qaic_config"] = override_config

    if value(row, "ccl_enabled"):
        additional_config["ccl_enabled"] = parse_bool(row.get("ccl_enabled"))
    add_if_present(additional_config, "comp_ctx_lengths_prefill", row)
    add_if_present(additional_config, "comp_ctx_lengths_decode", row)

    return compact_json(additional_config) if additional_config else ""


def add_if_present(
    target: dict,
    key: str,
    row: dict,
    source_key: str | None = None,
) -> None:
    raw = value(row, source_key or key)
    if raw:
        target[key] = raw


def build_api_server_command(row: dict, args) -> list[str]:
    python_bin = value(row, "python_bin", default=args.python_bin)
    cmd = [python_bin, "-m", "vllm.entrypoints.openai.api_server"]

    add_bool_arg(cmd, "--enable-chunked-prefill", row, "enable_chunked_prefill")
    add_bool_arg(cmd, "--no-enable-chunked-prefill", row, "no_enable_chunked_prefill")

    add_value_arg(cmd, "--port", row, "port", "server_port")
    add_value_arg(cmd, "--max-model-len", row, "max_model_len", "CL")
    add_value_arg(cmd, "--kv-cache-dtype", row, "kv_cache_dtype")
    add_value_arg(cmd, "--host", row, "host")
    add_value_arg(
        cmd,
        "--long-prefill-token-threshold",
        row,
        "long_prefill_token_threshold",
    )
    add_value_arg(cmd, "--quantization", row, "quantization")
    add_value_arg(cmd, "--max-num-seqs", row, "max_num_seqs", "BS")
    add_value_arg(cmd, "--dtype", row, "dtype")
    add_value_arg(cmd, "--generation-config", row, "generation_config")
    add_value_arg(cmd, "--runner", row, "runner")
    add_value_arg(cmd, "--seed", row, "seed")
    add_value_arg(cmd, "--model", row, "model")
    add_bool_arg(cmd, "--trust-remote-code", row, "trust_remote_code")

    additional_config = build_api_additional_config(row)
    if additional_config:
        cmd.extend(["--additional-config", additional_config])

    pooler_config = value(row, "pooler_config")
    if pooler_config:
        parse_json_cell(pooler_config, "pooler_config")
        cmd.extend(["--pooler-config", compact_json(json.loads(pooler_config))])

    extra_args = value(row, "server_extra_args")
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    return cmd


def add_many_token_arg(cmd: list[str], flag: str, row: dict, key: str) -> None:
    raw = value(row, key)
    if raw:
        cmd.append(flag)
        cmd.extend(split_space_tokens(raw))


def build_disagg_server_command(row: dict, args) -> list[str]:
    python_bin = value(row, "python_bin", default=args.python_bin)
    cmd = [python_bin, "-m", "qaic_disagg"]

    add_value_arg(cmd, "--host", row, "host")
    add_value_arg(cmd, "--port", row, "port", "server_port")
    add_many_token_arg(cmd, "--encode-port", row, "encode_port")
    add_many_token_arg(cmd, "--encode-device-group", row, "encode_device_group")
    add_many_token_arg(cmd, "--decode-port", row, "decode_port")
    add_many_token_arg(cmd, "--decode-device-group", row, "decode_device_group")
    add_many_token_arg(cmd, "--prefill-port", row, "prefill_port")
    add_many_token_arg(cmd, "--prefill-device-group", row, "prefill_device_group")
    add_value_arg(cmd, "--model", row, "model")
    add_value_arg(cmd, "--encode-max-num-seqs", row, "encode_max_num_seqs", "VBS")
    add_value_arg(cmd, "--prefill-max-num-seqs", row, "prefill_max_num_seqs", "PBS")
    add_value_arg(cmd, "--decode-max-num-seqs", row, "decode_max_num_seqs", "DBS")
    add_value_arg(
        cmd,
        "--decode-long-prefill-token-threshold",
        row,
        "decode_long_prefill_token_threshold",
    )
    add_value_arg(
        cmd,
        "--prefill-long-prefill-token-threshold",
        row,
        "prefill_long_prefill_token_threshold",
    )
    add_value_arg(
        cmd,
        "--prefill-max-num-batched-tokens",
        row,
        "prefill_max_num_batched_tokens",
    )
    add_value_arg(cmd, "--max-model-len", row, "max_model_len", "CL")

    enable_blocking = parse_bool(value(row, "enable_blocking", default="true"))
    enable_ccl = parse_bool(value(row, "enable_ccl", default="true"))

    encode_override = value(row, "encode_override_qaic_config")
    if encode_override:
        parse_json_cell(encode_override, "encode_override_qaic_config")
        config = json.loads(encode_override)
        if not enable_blocking and "qaic_config" in config:
            config["qaic_config"].pop("enable_blocking", None)
            config["qaic_config"].pop("blocking_mode", None)
            config["qaic_config"].pop("num_kv_blocks", None)
        if not enable_ccl and "qaic_config" in config:
            config["qaic_config"].pop("enable_ccl", None)
        cmd.extend(
            [
                "--encode-override-qaic-config",
                compact_json(config),
            ]
        )

    prefill_override = value(row, "prefill_override_qaic_config")
    if prefill_override:
        parse_json_cell(prefill_override, "prefill_override_qaic_config")
        config = json.loads(prefill_override)
        if not enable_blocking and "qaic_config" in config:
            config["qaic_config"].pop("enable_blocking", None)
            config["qaic_config"].pop("blocking_mode", None)
            config["qaic_config"].pop("num_kv_blocks", None)
        if not enable_ccl and "qaic_config" in config:
            config["qaic_config"].pop("enable_ccl", None)
        cmd.extend(
            [
                "--prefill-override-qaic-config",
                compact_json(config),
            ]
        )

    decode_override = value(row, "decode_override_qaic_config")
    if decode_override:
        parse_json_cell(decode_override, "decode_override_qaic_config")
        config = json.loads(decode_override)
        if not enable_blocking and "qaic_config" in config:
            config["qaic_config"].pop("enable_blocking", None)
            config["qaic_config"].pop("blocking_mode", None)
            config["qaic_config"].pop("num_kv_blocks", None)
        if not enable_ccl and "qaic_config" in config:
            config["qaic_config"].pop("enable_ccl", None)
        cmd.extend(
            [
                "--decode-override-qaic-config",
                compact_json(config),
            ]
        )

    verbosity = optional_int(row, "verbosity", default=0) or 0
    if verbosity > 0:
        cmd.append("-" + ("v" * verbosity))

    add_value_arg(cmd, "--dtype", row, "dtype")
    add_value_arg(cmd, "--kv-cache-dtype", row, "kv_cache_dtype")
    add_value_arg(cmd, "--quantization", row, "quantization")
    add_value_arg(cmd, "--tool-call-parser", row, "tool_call_parser")
    add_bool_arg(cmd, "--enable-auto-tool-choice", row, "enable_auto_tool_choice")
    add_bool_arg(cmd, "--enable-log-outputs", row, "enable_log_outputs")
    add_value_arg(cmd, "--reasoning-parser", row, "reasoning_parser")
    # qaic_disagg accepts the vLLM generation config spelling, but current
    # production commands use --generation_config. Preserve that convention.
    add_value_arg(cmd, "--generation_config", row, "generation_config")
    # VLM production commands use the hyphenated vLLM spelling instead; kept as a
    # distinct column so it never collides with the underscore LLM-PD convention above.
    add_value_arg(cmd, "--generation-config", row, "vlm_generation_config")
    add_bool_arg(cmd, "--enable-log-requests", row, "enable_log_requests")
    add_value_arg(
        cmd,
        "--chat-template-content-format",
        row,
        "chat_template_content_format",
    )
    add_value_arg(cmd, "--chat-template", row, "chat_template")
    add_value_arg(cmd, "--kv-store-size", row, "kv_store_size")
    add_value_arg(cmd, "--kv-handOff-port", row, "kv_handoff_port")
    add_value_arg(cmd, "--scheduling-policy", row, "scheduling_policy")
    add_bool_arg(cmd, "--no-enable-prefix-caching", row, "no_enable_prefix_caching")
    add_value_arg(
        cmd,
        "--prefill-pipeline-parallel-size",
        row,
        "prefill_pipeline_parallel_size",
    )
    add_value_arg(cmd, "--proxy-worker", row, "proxy_worker")

    limit_mm_per_prompt = value(row, "limit_mm_per_prompt")
    if limit_mm_per_prompt:
        parse_json_cell(limit_mm_per_prompt, "limit_mm_per_prompt")
        cmd.extend(
            [
                "--limit-mm-per-prompt",
                compact_json(json.loads(limit_mm_per_prompt)),
            ]
        )

    extra_args = value(row, "server_extra_args")
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    return cmd


def resolve_client_type(row: dict, args) -> str:
    requested = value(row, "client_type", default="auto").lower()
    if requested == "auto":
        has_qserve_script = resolve_qserve_script(row, args, require_exists=False)
        return "qserve" if has_qserve_script else "vllm_bench"
    # If qserve is explicitly requested but not available, fallback to vllm_bench
    if requested == "qserve":
        has_qserve_script = resolve_qserve_script(row, args, require_exists=False)
        if not has_qserve_script:
            return "vllm_bench"
    return requested


def resolve_qserve_script(row: dict, args, require_exists: bool = True) -> str:
    candidates = [
        value(row, "client_script"),
        args.qserve_benchmark_script,
        os.environ.get("QSERVE_BENCHMARK_SCRIPT", ""),
    ]
    # Check dedicated QSERVE_DIR first (Jenkins clones qserve to a separate directory)
    qserve_dir = os.environ.get("QSERVE_DIR", "")
    if qserve_dir:
        candidates.append(str(Path(qserve_dir) / "qserve/benchmarks/benchmark_serving.py"))
    # Also check inside vllm-qaic (legacy fallback)
    vllm_dir = Path(args.vllm_qaic_dir) if args.vllm_qaic_dir else None
    if vllm_dir:
        candidates.append(str(vllm_dir / "qserve/qserve/benchmarks/benchmark_serving.py"))
    candidates.append("/home/qraniumtest/qserve/qserve/benchmarks/benchmark_serving.py")

    for candidate in candidates:
        if not candidate:
            continue
        try:
            if Path(candidate).is_file():
                return candidate
        except OSError:
            continue

    explicit = value(row, "client_script") or args.qserve_benchmark_script
    if explicit and not require_exists:
        return explicit
    if require_exists:
        raise BenchmarkError(
            "qserve benchmark_serving.py was requested but not found. "
            "Set --qserve-benchmark-script or use client_type=vllm_bench."
        )
    return ""


def add_hf_random_dataset_args(cmd: list[str], row: dict) -> None:
    tokenizer = value(row, "tokenizer")
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    random_range_ratio = value(row, "random_range_ratio")
    if random_range_ratio:
        cmd.extend(["--random-range-ratio", random_range_ratio])
    dataset_path = value(row, "dataset_path")
    if dataset_path:
        cmd.extend(["--dataset-path", dataset_path])
    hf_subset = value(row, "hf_subset")
    if hf_subset:
        cmd.extend(["--hf-subset", hf_subset])
    hf_split = value(row, "hf_split")
    if hf_split:
        cmd.extend(["--hf-split", hf_split])
    hf_output_len = value(row, "hf_output_len")
    if hf_output_len:
        cmd.extend(["--hf-output-len", hf_output_len])


def add_random_mm_args(cmd: list[str], row: dict) -> None:
    base_items = value(row, "random_mm_base_items_per_request")
    if base_items:
        cmd.extend(["--random-mm-base-items-per-request", base_items])
    range_ratio = value(row, "random_mm_num_mm_items_range_ratio")
    if range_ratio:
        cmd.extend(["--random-mm-num-mm-items-range-ratio", range_ratio])
    limit_mm_per_prompt = value(row, "random_mm_limit_mm_per_prompt")
    if limit_mm_per_prompt:
        parse_json_cell(limit_mm_per_prompt, "random_mm_limit_mm_per_prompt")
        cmd.extend(["--random-mm-limit-mm-per-prompt", compact_json(json.loads(limit_mm_per_prompt))])
    # --random-mm-bucket-config uses a Python-tuple-keyed literal (e.g.
    # '{(720,1080,1):1.0}'), which is not valid JSON - pass it through verbatim.
    bucket_config = value(row, "random_mm_bucket_config")
    if bucket_config:
        cmd.extend(["--random-mm-bucket-config", bucket_config])


def build_client_command(row: dict, args) -> list[str]:
    client_type = resolve_client_type(row, args)
    host = value(row, "host", default="127.0.0.1")
    port = value(row, "port", "server_port", default="8080")
    endpoint = value(row, "endpoint")
    backend = value(row, "backend", default="vllm")
    dataset_name = value(row, "dataset_name", "dataset-name", default="random")
    num_prompts = value(row, "num_prompts", "num-prompts", default="2")
    max_concurrency = value(row, "max_concurrency", "max-concurrency")
    seed = value(row, "seed")
    request_rate = value(row, "request_rate", "request-rate")
    input_len = value(row, "input_len", "random_input_len", "PL")
    output_len = value(row, "output_len", "random_output_len", "GL")

    if client_type == "qserve":
        script_path = resolve_qserve_script(row, args)
        python_bin = value(
            row,
            "client_python_bin",
            "python_bin",
            default=args.python_bin,
        )
        cmd = [
            python_bin,
            script_path,
            "--backend",
            backend,
            "--dataset-name",
            dataset_name,
            "--model",
            value(row, "model"),
            "--num-prompts",
            num_prompts,
            "--host",
            host,
            "--port",
            port,
        ]
        if max_concurrency:
            cmd.extend(["--max-concurrency", max_concurrency])
        if input_len:
            cmd.extend(["--random-input-len", input_len])
        if output_len:
            cmd.extend(["--random-output-len", output_len])
        if endpoint:
            cmd.extend(["--endpoint", endpoint])
        if seed:
            cmd.extend(["--seed", seed])
        add_hf_random_dataset_args(cmd, row)
        add_bool_arg(cmd, "--ignore-eos", row, "ignore_eos")
        add_bool_arg(cmd, "--trust-remote-code", row, "trust_remote_code")
        add_bool_arg(cmd, "--save-result", row, "save_result")
    elif client_type == "vllm_bench":
        base_url = value(row, "base_url") or f"http://{host}:{port}"
        cmd = [
            "vllm",
            "bench",
            "serve",
            "--backend",
            backend,
        ]
        if endpoint:
            cmd.extend(["--endpoint", endpoint])
        cmd.extend(
            [
                "--base-url",
                base_url,
                "--model",
                value(row, "model"),
                "--dataset-name",
                dataset_name,
                "--num-prompts",
                num_prompts,
            ]
        )
        if request_rate:
            cmd.extend(["--request-rate", request_rate])
        if max_concurrency:
            cmd.extend(["--max-concurrency", max_concurrency])

        length_style = value(row, "length_arg_style", default="random").lower()
        if length_style == "general":
            if input_len:
                cmd.extend(["--input-len", input_len])
            if output_len:
                cmd.extend(["--output-len", output_len])
        else:
            if input_len:
                cmd.extend(["--random-input-len", input_len])
            if output_len:
                cmd.extend(["--random-output-len", output_len])

        if seed:
            cmd.extend(["--seed", seed])
        add_hf_random_dataset_args(cmd, row)
        add_random_mm_args(cmd, row)
        add_bool_arg(cmd, "--ignore-eos", row, "ignore_eos")
        add_bool_arg(cmd, "--trust-remote-code", row, "trust_remote_code")
        temperature = value(row, "temperature")
        if temperature:
            cmd.extend(["--temperature", temperature])
        add_bool_arg(cmd, "--save-result", row, "save_result")
    else:
        raise BenchmarkError(f"unsupported client_type: {client_type}")

    extra_args = value(row, "client_extra_args")
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    return cmd


def build_server_command(row: dict, args) -> list[str]:
    server_type = value(row, "server_type", default="api_server").lower()
    if server_type == "api_server":
        return build_api_server_command(row, args)
    if server_type in {"qaic_disagg", "disagg"}:
        return build_disagg_server_command(row, args)
    raise BenchmarkError(f"unsupported server_type: {server_type}")


def server_ready_markers(row: dict) -> tuple[str, ...]:
    explicit = value(row, "server_ready_markers")
    if explicit:
        return tuple(marker.strip() for marker in explicit.split("|") if marker.strip())
    server_type = value(row, "server_type", default="api_server").lower()
    if server_type == "api_server":
        return API_SERVER_READY_MARKERS
    return DISAGG_READY_MARKERS


def server_ports(row: dict) -> list[int]:
    ports = []
    for key in ("port", "server_port", "prefill_port", "decode_port", "encode_port"):
        raw = value(row, key)
        if not raw:
            continue
        try:
            ports.extend(iter_ports(raw))
        except ValueError:
            continue
    return sorted(set(ports))


def kill_ports(ports: Iterable[int]) -> None:
    for port in ports:
        pids = []

        # Method 1: Try lsof (preferred, fast)
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            pids = [p.strip() for p in result.stdout.splitlines() if p.strip()]
            if pids:
                print(f"  Port {port}: found pid(s) {', '.join(pids)} (via lsof), killing...")
                subprocess.run(["kill", "-9", *pids], check=False, capture_output=True)
                time.sleep(0.5)
                continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Method 2: Try ss (common in containers)
        try:
            result = subprocess.run(
                ["ss", "-tlnp"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line:
                    match = re.search(r"pid=(\d+)", line)
                    if match:
                        pids.append(match.group(1))
            if pids:
                print(f"  Port {port}: found pid(s) {', '.join(pids)} (via ss), killing...")
                subprocess.run(["kill", "-9", *pids], check=False, capture_output=True)
                time.sleep(0.5)
                continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Method 3: Try fuser
        try:
            result = subprocess.run(
                ["fuser", f"{port}/tcp"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            pids = [p.strip() for p in result.stdout.split() if p.strip()]
            if pids:
                print(f"  Port {port}: found pid(s) {', '.join(pids)} (via fuser), killing...")
                subprocess.run(["kill", "-9", *pids], check=False, capture_output=True)
                time.sleep(0.5)
                continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Method 4: Try /proc/net/tcp (works in Linux containers without tools)
        try:
            with open("/proc/net/tcp", "r") as f:
                lines = f.readlines()
            port_hex = f"{port:X}"
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 10:
                    local_addr = parts[1]
                    if local_addr.endswith(f":{port_hex}"):
                        inode = parts[9]
                        for proc_dir in Path("/proc").glob("*/fd/*"):
                            try:
                                if proc_dir.is_symlink():
                                    target = str(proc_dir.resolve())
                                    if f"socket:[{inode}]" in target:
                                        pid = proc_dir.parent.parent.name
                                        if pid.isdigit() and pid not in pids:
                                            pids.append(pid)
                            except (OSError, ValueError):
                                pass
            if pids:
                print(f"  Port {port}: found pid(s) {', '.join(pids)} (via /proc), killing...")
                subprocess.run(["kill", "-9", *pids], check=False, capture_output=True)
                time.sleep(0.5)
                continue
        except (FileNotFoundError, OSError, ValueError):
            pass

        # Method 5: Try Python socket (universal, always available)
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            if result == 0:
                print(f"  Port {port}: in use but cannot determine PID (socket check)")
            else:
                print(f"  Port {port}: free (no processes)")
            continue
        except (OSError, ImportError):
            pass

        print(f"  Port {port}: unable to check (no tools available)")


def write_log_header(log_path: Path, command: list[str], title: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"{title}\n")
        log_file.write("=" * len(title) + "\n")
        timestamp = datetime.now(timezone.utc).isoformat()
        log_file.write(f"timestamp_utc: {timestamp}\n")
        log_file.write(f"command: {command_to_shell_string(command)}\n\n")


def launch_server(
    command: list[str],
    log_path: Path,
    markers: tuple[str, ...],
    cwd: str,
    timeout_s: int,
) -> ServerProcess:
    ports = server_ports_from_command(command)
    if ports:
        print(f"  Final port cleanup check before server launch for ports: {ports}")
        kill_ports(ports)
        time.sleep(0.5)  # Brief wait to ensure ports are released

    write_log_header(log_path, command, "Server Command")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=cwd,
        preexec_fn=os.setsid,  # noqa: PLW1509
    )

    ready_event = threading.Event()
    started_at = time.monotonic()
    server = ServerProcess(
        process=process,
        log_path=log_path,
        ready_event=ready_event,
        stream_thread=threading.Thread(),
        started_at=started_at,
    )

    def stream_output() -> None:
        assert process.stdout is not None
        with log_path.open("a", encoding="utf-8") as log_file:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
                if not ready_event.is_set() and any(marker in line for marker in markers):
                    server.ready_at = time.monotonic()
                    ready_event.set()

    stream_thread = threading.Thread(target=stream_output, daemon=True)
    server.stream_thread = stream_thread
    stream_thread.start()

    while True:
        if ready_event.is_set():
            return server
        if process.poll() is not None:
            stream_thread.join(timeout=5)
            raise BenchmarkError(f"server exited before ready marker; returncode={process.returncode}; log={log_path}")
        if timeout_s > 0 and time.monotonic() - started_at > timeout_s:
            raise BenchmarkError(f"server did not become ready within {timeout_s}s; log={log_path}")
        time.sleep(1)


def server_ports_from_command(command: list[str]) -> list[int]:
    ports = []
    for index, token in enumerate(command):
        if token in {
            "--port",
            "--prefill-port",
            "--decode-port",
            "--encode-port",
            "--kv-handOff-port",
        }:
            cursor = index + 1
            while cursor < len(command) and not command[cursor].startswith("--"):
                ports.extend(iter_ports(command[cursor]))
                cursor += 1
    return sorted(set(ports))


def terminate_server(server: ServerProcess, timeout_s: int) -> None:
    if server.process.poll() is not None:
        return
    print(f"  Terminating server pid {server.process.pid}")
    try:
        os.killpg(os.getpgid(server.process.pid), signal.SIGTERM)
        server.process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"  Server did not exit in {timeout_s}s; sending SIGKILL")
        os.killpg(os.getpgid(server.process.pid), signal.SIGKILL)
        server.process.wait(timeout=30)
    finally:
        server.stream_thread.join(timeout=5)


def run_client(command: list[str], log_path: Path, cwd: str) -> int:
    write_log_header(log_path, command, "Client Command")
    env = os.environ.copy()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=cwd,
    )
    assert process.stdout is not None
    with log_path.open("a", encoding="utf-8") as log_file:
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        process.wait()
        log_file.write(f"\nreturncode: {process.returncode}\n")
    return process.returncode


def parse_benchmark_output(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    content = log_path.read_text(encoding="utf-8", errors="replace")
    runs = []
    for run_number, block_match in enumerate(BENCH_BLOCK_RE.finditer(content), 1):
        run = {"run_number": run_number}
        for kv_match in BENCH_KV_RE.finditer(block_match.group(1)):
            label = kv_match.group(1).strip()
            key = BENCH_LABEL_TO_KEY.get(label)
            if key is None:
                continue
            raw = kv_match.group(2)
            run[key] = int(float(raw)) if key in INT_RESULT_KEYS else float(raw)
        for ms_key, sec_key in [
            ("mean_ttft_ms", "ttft_s"),
            ("median_ttft_ms", "median_ttft_s"),
            ("p99_ttft_ms", "p99_ttft_s"),
            ("mean_tpot_ms", "tpot_s"),
            ("median_tpot_ms", "median_tpot_s"),
            ("p99_tpot_ms", "p99_tpot_s"),
            ("mean_itl_ms", "itl_s"),
            ("median_itl_ms", "median_itl_s"),
            ("p99_itl_ms", "p99_itl_s"),
        ]:
            run[sec_key] = run.get(ms_key) / 1000.0 if run.get(ms_key) is not None else None
        runs.append(run)
    return runs


def rounded(value_: object) -> object:
    if isinstance(value_, (float, int)) and not isinstance(value_, bool):
        return round(value_, 2)
    return value_ if value_ is not None else ""


def build_config_summary(row: dict) -> str:
    server_type = value(row, "server_type", default="api_server").lower()
    if server_type in {"qaic_disagg", "disagg"}:
        vbs = value(row, "encode_max_num_seqs", "VBS")
        vbs_part = f"VBS:{vbs} / " if vbs else ""
        return (
            f"{vbs_part}"
            f"PBS:{value(row, 'prefill_max_num_seqs', 'PBS')} / "
            f"DBS:{value(row, 'decode_max_num_seqs', 'DBS')} / "
            f"PL:{value(row, 'input_len', 'random_input_len', 'PL')} / "
            f"GL:{value(row, 'output_len', 'random_output_len', 'GL')} / "
            f"CL:{value(row, 'max_model_len', 'CL')}"
        )
    return (
        f"BS:{value(row, 'max_num_seqs', 'BS')} / "
        f"PL:{value(row, 'input_len', 'random_input_len', 'PL')} / "
        f"GL:{value(row, 'output_len', 'random_output_len', 'GL')} / "
        f"CL:{value(row, 'max_model_len', 'CL')}"
    )


def build_mode_type(row: dict) -> str:
    server_type = value(row, "server_type", default="api_server").lower()
    if server_type in {"qaic_disagg", "disagg"}:
        es = device_count(value(row, "encode_device_group"))
        pp = value(row, "prefill_pipeline_parallel_size") or device_count(value(row, "prefill_device_group"))
        ts = device_count(value(row, "decode_device_group"))
        parts = []
        if es:
            parts.append(f"ES{es}")
        if pp:
            parts.append(f"PP{pp}")
        if ts:
            parts.append(f"TS{ts}")
        return "+".join(parts)
    ts = device_count(value(row, "device_group"))
    return f"TS{ts}" if ts else ""


def result_status(client_returncode: int | None, parsed_runs: list[dict]) -> str:
    if client_returncode is None:
        return "server_failed"
    if client_returncode != 0:
        return "client_failed_with_results" if parsed_runs else "client_failed"
    return "success" if parsed_runs else "no_results"


def get_qaic_sdk_version() -> str:
    try:
        result = subprocess.run(
            ["/opt/qti-aic/tools/qaic-version-util"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def make_output_row(
    row: dict,
    config_name: str,
    status: str,
    error: str,
    run: dict,
    server_log: Path,
    client_log: Path,
    server_cmd: list[str] | None = None,
    client_cmd: list[str] | None = None,
) -> dict:
    model = value(row, "model")
    max_concurrency = value(row, "max_concurrency", "max-concurrency")
    decode_bs = optional_int(row, "decode_max_num_seqs", "DBS", default=None)
    if decode_bs is None:
        decode_bs = optional_int(row, "max_num_seqs", "BS", default=None)
    mean_tpot_ms = run.get("mean_tpot_ms")
    decode_tps = ""
    if mean_tpot_ms and decode_bs:
        decode_tps = round((1000.0 / mean_tpot_ms) * decode_bs, 2)

    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_name": config_name,
        "status": status,
        "error": error,
        "model": model,
        "server_type": value(row, "server_type", default="api_server"),
        "config_summary": build_config_summary(row),
        "mode_type": build_mode_type(row),
        "device_group": value(row, "device_group"),
        "encode_device_group": value(row, "encode_device_group"),
        "prefill_device_group": value(row, "prefill_device_group"),
        "decode_device_group": value(row, "decode_device_group"),
        "BS": value(row, "max_num_seqs", "BS"),
        "VBS": value(row, "encode_max_num_seqs", "VBS"),
        "PBS": value(row, "prefill_max_num_seqs", "PBS"),
        "DBS": value(row, "decode_max_num_seqs", "DBS"),
        "PL": value(row, "input_len", "random_input_len", "PL"),
        "GL": value(row, "output_len", "random_output_len", "GL"),
        "CL": value(row, "max_model_len", "CL"),
        "num_prompts": value(row, "num_prompts", "num-prompts"),
        "max_concurrency": max_concurrency,
        "failed_requests": run.get("failed_requests", ""),
        "benchmark_duration_s": rounded(run.get("benchmark_duration_s", "")),
        "request_throughput_req_s": rounded(run.get("request_throughput_req_s", "")),
        "output_token_throughput_tok_s": rounded(run.get("output_token_throughput_tok_s", "")),
        "total_token_throughput_tok_s": rounded(run.get("total_token_throughput_tok_s", "")),
        "mean_TTFT_ms": rounded(run.get("mean_ttft_ms", "")),
        "P99_TTFT_ms": rounded(run.get("p99_ttft_ms", "")),
        "mean_TPOT_ms": rounded(run.get("mean_tpot_ms", "")),
        "P99_TPOT_ms": rounded(run.get("p99_tpot_ms", "")),
        "mean_ITL_ms": rounded(run.get("mean_itl_ms", "")),
        "P99_ITL_ms": rounded(run.get("p99_itl_ms", "")),
        "decode_TPS": decode_tps,
        "vllm_qaic_branch": os.environ.get("VLLM_QAIC_BRANCH", ""),
        "qaic_disagg_branch": os.environ.get("QAIC_DISAGG_BRANCH", ""),
        "qserve_branch": os.environ.get("QSERVE_BRANCH", ""),
        "qeff_branch": os.environ.get("QEFF_BRANCH", ""),
        "qaic_sdk_version": get_qaic_sdk_version(),
        "server_command": command_to_shell_string(server_cmd) if server_cmd else "",
        "client_command": command_to_shell_string(client_cmd) if client_cmd else "",
        "server_log": str(server_log),
        "client_log": str(client_log),
        "pooling_method": value(row, "pooling_method"),
    }


def append_output_rows(output_csv: Path, rows: list[dict]) -> None:
    if not rows:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def generate_published_csv(input_csv: Path, output_csv: Path) -> None:
    """Generate a simplified published CSV with only key fields for team distribution."""
    if not input_csv.exists():
        return
    with input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return

    # Convert milliseconds to seconds for latency metrics and add model_category
    for row in rows:
        if row.get("mean_TTFT_ms"):
            row["mean_ttft_s"] = str(round(float(row["mean_TTFT_ms"]) / 1000, 4))
        if row.get("mean_TPOT_ms"):
            row["mean_tpot_s"] = str(round(float(row["mean_TPOT_ms"]) / 1000, 4))
        if row.get("mean_ITL_ms"):
            row["mean_itl_s"] = str(round(float(row["mean_ITL_ms"]) / 1000, 4))

        # Determine model category based on config_name
        config_name = row.get("config_name", "").lower()
        if "embedding" in config_name:
            row["model_category"] = "Embedding"
        elif "audio" in config_name:
            row["model_category"] = "Audio"
        elif "vlm" in config_name:
            row["model_category"] = "VLM"
        else:
            row["model_category"] = "LLM"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PUBLISHED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_published_csvs(results_dir: Path, output_csv: Path) -> None:
    """Merge all published CSVs in results_dir into a single consolidated published CSV."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return

    all_rows = []
    published_csvs = sorted(results_dir.glob("*_results_published.csv"))

    for pub_csv in published_csvs:
        if not pub_csv.exists():
            continue
        with pub_csv.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            all_rows.extend(rows)

    if not all_rows:
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PUBLISHED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def read_csv_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = []
        for index, row in enumerate(csv.DictReader(f), 1):
            normalized = {str(k).strip(): (v or "").strip() for k, v in row.items()}
            normalized["_data_row"] = str(index)
            if parse_bool(normalized.get("enabled"), default=True):
                rows.append(normalized)
        return rows


def parse_rows_spec(row_spec: str, total_rows: int) -> list[int]:
    row_spec = (row_spec or "ALL").strip()
    if row_spec.upper() == "ALL":
        return list(range(1, total_rows + 1))
    selected = []
    for chunk in row_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            start, end = [int(part.strip()) for part in chunk.split(":", 1)]
            if start > end:
                raise ValueError(f"invalid row range {chunk!r}: start > end")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(chunk))
    invalid = [row for row in selected if row < 1 or row > total_rows]
    if invalid:
        raise ValueError(f"row(s) out of range for {total_rows} data rows: {invalid}")
    return selected


def run_one(row: dict, args, config_name: str, output_csv: Path) -> bool:
    row_number = int(row["_data_row"])
    model = value(row, "model")

    # Determine effective_config_name based on config_name
    if config_name == "audio":
        # Audio: always 'audio'
        effective_config_name = "audio"
    elif config_name == "embedding":
        # Embedding: use pooling_method
        effective_config_name = value(row, "pooling_method", default="unknown")
    elif config_name == "vlm":
        # VLM: specs/blocking/ccl/disagg_mode
        specs = value(row, "specialization_mode", default="single")
        blocking = "blocking" if parse_bool(row.get("enable_blocking"), default=False) else "non_blocking"
        ccl = "ccl" if parse_bool(row.get("enable_ccl"), default=False) else "non_ccl"
        disagg_mode = value(row, "disagg_mode", default="single")
        effective_config_name = f"{specs}/{blocking}/{ccl}/{disagg_mode}"
    else:
        # LLM: use config_name as-is (default, ccl, blocking, disagg_pd)
        effective_config_name = config_name

    run_id = f"{row_number:03d}_{sanitize_name(model)}_{sanitize_name(effective_config_name)}"
    log_dir = Path(args.results_dir) / "logs" / sanitize_name(config_name) / run_id
    server_log = log_dir / "server.log"
    client_log = log_dir / "client.log"

    server_cmd = build_server_command(row, args)
    client_cmd = build_client_command(row, args)

    # Pre-check and cleanup ports before starting server
    ports_to_use = server_ports_from_command(server_cmd)
    if ports_to_use:
        print()
        print("=" * 80)
        print(f"Pre-flight port check for ports: {ports_to_use}")
        print("=" * 80)
        kill_ports(ports_to_use)
        print("  Ports cleaned and ready for use")
        time.sleep(1)  # Wait for ports to fully release

    print()
    print("=" * 80)
    print(f"Config: {effective_config_name}  Row: {row_number}  Model: {model}")
    print(f"Server: {command_to_shell_string(server_cmd)}")
    print(f"Client: {command_to_shell_string(client_cmd)}")
    print("=" * 80)

    if args.dry_run:
        write_log_header(server_log, server_cmd, "Server Command Dry Run")
        write_log_header(client_log, client_cmd, "Client Command Dry Run")
        append_output_rows(
            output_csv,
            [
                make_output_row(
                    row=row,
                    config_name=effective_config_name,
                    status="dry_run",
                    error="",
                    run={},
                    server_log=server_log,
                    client_log=client_log,
                    server_cmd=server_cmd,
                    client_cmd=client_cmd,
                )
            ],
        )
        return True

    server = None
    client_returncode = None
    parsed_runs: list[dict] = []
    error = ""

    try:
        server = launch_server(
            server_cmd,
            server_log,
            server_ready_markers(row),
            cwd=args.base_dir,
            timeout_s=args.server_ready_timeout_s,
        )
        print(f"  Server ready after {server.ready_time_s:.2f}s")
        client_returncode = run_client(client_cmd, client_log, cwd=args.base_dir)
        parsed_runs = parse_benchmark_output(client_log)
        status = result_status(client_returncode, parsed_runs)
    except (BenchmarkError, OSError, subprocess.TimeoutExpired, ValueError) as exc:
        status = "error"
        error = str(exc)
        print(f"  ERROR: {error}")
    finally:
        if server is not None:
            terminate_server(server, timeout_s=args.server_stop_timeout_s)

        output_runs = parsed_runs or [{}]
        append_output_rows(
            output_csv,
            [
                make_output_row(
                    row=row,
                    config_name=effective_config_name,
                    status=status,
                    error=error,
                    run=run,
                    server_log=server_log,
                    client_log=client_log,
                    server_cmd=server_cmd,
                    client_cmd=client_cmd,
                )
                for run in output_runs
            ],
        )

    return status == "success"


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config-name",
        required=True,
        help="Logical config name for output rows.",
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV path.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--rows",
        default="ALL",
        help="Data row selector: ALL, 1, 1:4, or 1,3.",
    )
    parser.add_argument(
        "--latest-models-only",
        default="true",
        help="If true (default), restrict to the curated LATEST_MODELS list. SKIPPED_MODELS is always excluded.",
    )
    parser.add_argument(
        "--results-dir",
        default="vllm_llm_results",
        help="Log/result root.",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Subprocess working directory.",
    )
    parser.add_argument(
        "--python-bin",
        default="python3",
        help="Python binary for server/client modules.",
    )
    parser.add_argument(
        "--qserve-benchmark-script",
        default="",
        help="Path to qserve/qserve/benchmarks/benchmark_serving.py.",
    )
    parser.add_argument(
        "--vllm-qaic-dir",
        default=os.environ.get("VLLM_QAIC_DIR", ""),
        help="vllm-qaic checkout path used to auto-resolve qserve benchmark_serving.py.",
    )
    parser.add_argument(
        "--server-ready-timeout-s",
        type=int,
        default=21600,
        help="Seconds to wait for server readiness; 0 waits forever.",
    )
    parser.add_argument(
        "--server-stop-timeout-s",
        type=int,
        default=120,
        help="Seconds to wait for graceful server shutdown.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write commands/log stubs only.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after first failed row.",
    )
    parser.add_argument(
        "--disagg-mode",
        default="ALL",
        help="VLM only: filter rows by disagg_mode column (ED, PD, EPD, or ALL).",
    )
    parser.add_argument(
        "--specialization-mode",
        default="ALL",
        help="VLM only: filter rows by specialization_mode column (single, multi, or ALL).",
    )
    parser.add_argument(
        "--blocking-mode",
        default="ALL",
        help="VLM only: filter rows by blocking_mode column (blocking, non_blocking, or ALL).",
    )
    return parser


def row_matches_filter(row: dict, column: str, filter_value: str) -> bool:
    filter_value = (filter_value or "ALL").strip()
    if filter_value.upper() == "ALL":
        return True
    row_value = value(row, column)
    if not row_value:
        return True
    return row_value.lower() == filter_value.lower()


def run_benchmarks(args, latest_models: set[str]) -> int:
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    rows = read_csv_rows(input_csv)
    selected = set(parse_rows_spec(args.rows, len(rows)))
    latest_only = parse_bool(args.latest_models_only, default=True)
    disagg_mode = getattr(args, "disagg_mode", "ALL")
    specialization_mode = getattr(args, "specialization_mode", "ALL")
    blocking_mode = getattr(args, "blocking_mode", "ALL")

    if output_csv.exists():
        output_csv.unlink()

    failures = 0
    for row in rows:
        if int(row["_data_row"]) not in selected:
            continue
        model = value(row, "model")
        if model in SKIPPED_MODELS:
            print(f"Skipping row {row['_data_row']} ({model}): in SKIPPED_MODELS (>70B).")
            continue
        if latest_only and model not in latest_models:
            print(f"Skipping row {row['_data_row']} ({model}): not in LATEST_MODELS (latest-only mode).")
            continue
        if not row_matches_filter(row, "disagg_mode", disagg_mode):
            print(f"Skipping row {row['_data_row']} ({model}): disagg_mode does not match filter {disagg_mode!r}.")
            continue
        if not row_matches_filter(row, "specialization_mode", specialization_mode):
            print(
                f"Skipping row {row['_data_row']} ({model}): specialization_mode does not match "
                f"filter {specialization_mode!r}."
            )
            continue
        if not row_matches_filter(row, "blocking_mode", blocking_mode):
            print(f"Skipping row {row['_data_row']} ({model}): blocking_mode does not match filter {blocking_mode!r}.")
            continue
        ok = run_one(row, args, args.config_name, output_csv)
        if not ok:
            failures += 1
            if args.fail_fast:
                break

    print()
    print(f"Output CSV: {output_csv}")
    print("Note: Use merge_published_results.py to generate consolidated published CSV")

    if failures:
        print(f"Completed with {failures} failed row(s).")
        return 1
    print("Completed successfully.")
    return 0
