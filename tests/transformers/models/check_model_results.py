# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from datetime import datetime

import numpy as np


def parse_exec_info_metrics(exec_info_str):
    """
    Parse performance metrics from exec_info string.

    :exec_info_str: str - The exec_info string containing performance stats
    :return: dict - Dictionary containing parsed metrics
    """
    import re

    metrics = {
        "prefill_time_sec": None,
        "decode_throughput_tokens_per_sec": None,
        "total_throughput_tokens_per_sec": None,
        "e2e_inference_time_sec": None,
    }

    exec_info_text = str(exec_info_str)

    # Parse Average Prefill time (TTFT)
    if "Average Prefill time" in exec_info_text or "TTFT" in exec_info_text:
        match = re.search(r"Average Prefill time.*?is=\s*([\d.]+)\s*sec", exec_info_text)
        if match:
            metrics["prefill_time_sec"] = float(match.group(1))

    # Parse Decode throughput
    if "Decode" in exec_info_text:
        match = re.search(r"Decode\s+is=\s*([\d.]+)\s*tokens/sec", exec_info_text)
        if match:
            metrics["decode_throughput_tokens_per_sec"] = float(match.group(1))

    # Parse Total throughput
    if "Total is=" in exec_info_text:
        match = re.search(r"Total\s+is=\s*([\d.]+)\s*tokens/sec", exec_info_text)
        if match:
            metrics["total_throughput_tokens_per_sec"] = float(match.group(1))

    # Parse Total E2E inference time
    if "Total (E2E) inference time" in exec_info_text:
        match = re.search(r"Total \(E2E\) inference time\s+is=\s*([\d.]+)\s*sec", exec_info_text)
        if match:
            metrics["e2e_inference_time_sec"] = float(match.group(1))

    return metrics


def dump_and_compare_results(
    model_name,
    compile_params,
    json_file_path,
    cloud_ai_100_tokens,
    exec_info=None,
    pytorch_hf_tokens=None,
    pytorch_kv_tokens=None,
    ort_tokens=None,
):
    """
    Function to dump the test results to JSON file and compare the performance and output results with previous runs if available

    :model_name: str
    :pytorch_hf_tokens: list
    :pytorch_kv_tokens: list
    :ort_tokens: list
    :cloud_ai_100_tokens: list
    :exec_info: object
    :compile_params: dict
    :return None
    """

    current_logs_dir = os.environ.get("NIGHTLY_LOG_DIR")
    if current_logs_dir is None:
        current_logs_dir = os.path.expanduser("~/.cache/Nightly_Logs/build_tag")
    os.makedirs(current_logs_dir, exist_ok=True)
    # original_logs_dir = Path(current_logs_dir).parent
    original_logs_dir = current_logs_dir
    current_results_json_file_path = os.path.join(current_logs_dir, json_file_path)
    original_results_json_file_path = os.path.join(original_logs_dir, json_file_path)

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj

    exec_info_metrics = parse_exec_info_metrics(exec_info)

    test_data = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "compile_params": compile_params,
        "pytorch_hf_tokens": convert_to_serializable(pytorch_hf_tokens) if pytorch_hf_tokens is not None else None,
        "pytorch_kv_tokens": convert_to_serializable(pytorch_kv_tokens),
        "ort_tokens": convert_to_serializable(ort_tokens),
        "cloud_ai_100_tokens": convert_to_serializable(cloud_ai_100_tokens),
        "exec_info_metrics": exec_info_metrics,
        "exec_info_raw_string": str(exec_info),
    }

    # Load existing results if file exists
    all_results = {}
    if os.path.exists(current_results_json_file_path):
        with open(current_results_json_file_path, "r") as f:
            all_results = json.load(f)
            print(f"Loaded existing model results from {current_results_json_file_path}")
    else:
        with open(current_results_json_file_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        print(f"Created new results file at {current_results_json_file_path}")

    model_name_safe = model_name.replace("/", "_").replace("-", "_")
    all_results[model_name_safe] = test_data

    with open(current_results_json_file_path, "w") as f:
        json.dump(all_results, f, indent=4, default=str)
    print(f"Successfully saved test results to {current_results_json_file_path}")

    with open(original_results_json_file_path, "r") as f:
        previous_results = json.load(f)
        print(f"Loaded Previous model results from {original_results_json_file_path}")

    previous_data = previous_results[model_name_safe]

    # Compare performance metrics with 5% tolerance
    previous_metrics = previous_data.get("exec_info_metrics", {})
    current_metrics = exec_info_metrics

    for metric_name in [
        "prefill_time_sec",
        "decode_throughput_tokens_per_sec",
        "total_throughput_tokens_per_sec",
        "e2e_inference_time_sec",
    ]:
        prev_val = previous_metrics[metric_name]
        curr_val = current_metrics[metric_name]

        if prev_val is not None and curr_val is not None and prev_val != 0:
            percent_diff = abs((curr_val - prev_val) / prev_val) * 100
            assert percent_diff <= 5.0, (
                f"Performance metric {metric_name} exceeds 5% tolerance: "
                f"previous={prev_val}, current={curr_val}, diff={percent_diff:.2f}%"
            )
            print(f"✓ {metric_name}: {percent_diff:.2f}% difference (within 5% tolerance)")

    # Compare output tokens using Mean Absolute Deviation (MAD) with 10^-2 tolerance
    previous_tokens = previous_data.get("cloud_ai_100_tokens", None)

    if previous_tokens is not None and isinstance(previous_tokens, list):
        if previous_tokens and isinstance(previous_tokens[0], str):
            print("⊘ Output tokens: Skipping Tokens check (previous data contains strings)")
        else:
            prev_tokens_arr = np.array(previous_tokens, dtype=np.float32)
            curr_tokens_arr = np.array(cloud_ai_100_tokens, dtype=np.float32)

            mad = np.mean(np.abs(curr_tokens_arr - prev_tokens_arr))
            tolerance = 1e-2

            assert mad <= tolerance, f"Output tokens MAD exceeds 10^-2 tolerance: MAD={mad:.6f}, tolerance={tolerance}"
            print(f"✓ Output tokens MAD: {mad:.6f} (within 10^-2 tolerance)")
    return True
