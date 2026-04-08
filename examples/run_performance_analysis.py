# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import argparse
import json
import logging
import os
import torch
import warnings
from pathlib import Path
from typing import Optional, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
#from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils.constants import Constants


def _suppress_warnings():
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")
    logging.captureWarnings(True)
    for logger_name in ("torch", "torch.onnx", "onnx", "onnxruntime"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def evaluate_model_performance(
    model_name: str,
    prompt_len: int = Constants.PROMPT_LEN,
    ctx_len: int = Constants.CTX_LEN,
    batch_size: int = 1,
    prefill_only: bool = False,
    num_cores: int = 14,
    num_hidden_layers: int = 2,
    runner_num_iters: int = 10,
    profiling_type: str = "raw_device_stats",
    profiling_start_iter: int = 2,
    write_output_start_iter: Optional[int] = None,
    output_dir: str = None,
    enable_mla: Optional[bool] = False,
    mla_absorption_config: Optional[Dict[str, bool]] = False,
    num_devices: int = 1,
):
    _suppress_warnings()
    #replace_transformers_quantizers()

    model_path ="/home/huggingface_hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, num_hidden_layers=num_hidden_layers, trust_remote_code=True)
    model = QEFFAutoModelForCausalLM(model, num_kv_heads_repeat=num_devices)

    result = model.evaluate_performance(
        batch_size=batch_size,
        prefill_only=prefill_only,
        compile_kwargs={
            "prefill_seq_len": prompt_len,
            "ctx_len": ctx_len,
            "batch_size": batch_size,
            "num_cores": num_cores,
            "mxfp6_matmul": True,
            "aic_enable_depth_first": False,
            "mxint8_kv_cache": True,
            "enable_mla": enable_mla,
            "mla_absorption_config": mla_absorption_config,
            "num_devices": num_devices,
        },
        runner_num_iters=runner_num_iters,
        profiling_type=profiling_type,
        profiling_start_iter=profiling_start_iter,
        write_output_start_iter=write_output_start_iter,
        output_dir=output_dir,
    )

    report_path = Path(result["output_dir"]) / "performance_report.json"
    report_path.write_text(json.dumps(result, indent=2))
    print(f"Performance report written to: {report_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile and run qaic performance analysis for CausalLM")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--prompt-len", type=int, default=Constants.PROMPT_LEN, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=Constants.CTX_LEN, help="Context length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for compile and synthetic runner IO")
    parser.add_argument("--prefill-only", action="store_true", help="Run performance analysis only for prefill stage")
    parser.add_argument("--num-cores", type=int, default=14, help="Number of AI100 cores for compile")
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=2,
        help="Number of model layers to load for performance analysis",
    )
    parser.add_argument("--runner-num-iters", type=int, default=10, help="qaic-runner iterations")
    parser.add_argument(
        "--profiling-type",
        type=str,
        default="raw_device_stats",
        choices=["stats", "trace", "latency", "raw_device_stats"],
        help="qaic-runner profiling type",
    )
    parser.add_argument(
        "--profiling-start-iter",
        type=int,
        default=2,
        help="qaic-runner profiling start iteration",
    )
    parser.add_argument(
        "--write-output-start-iter",
        type=int,
        default=None,
        help="qaic-runner output write start iteration (must be >0 and < profiling-start-iter)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for performance artifacts")
    #parser.add_argument("--enable_mla", type=bool, default=False, help="enable_mla")
    #parser.add_argument("--mla_absorption_config", type=Dict[str, bool], default={"enable":False, "online":False}, help="mla_absorption_config")
    args = parser.parse_args()

    enable_mla = True
    mla_absorption_config = {"enable":True, "online":False}
    num_devices=4
    prompt_len=1
    ctx_len=16384
    
    
    evaluate_model_performance(
        model_name=args.model_name,
        prompt_len=args.prompt_len,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        prefill_only=args.prefill_only,
        num_cores=args.num_cores,
        num_hidden_layers=args.num_hidden_layers,
        runner_num_iters=args.runner_num_iters,
        profiling_type=args.profiling_type,
        profiling_start_iter=args.profiling_start_iter,
        write_output_start_iter=args.write_output_start_iter,
        output_dir=args.output_dir,
        enable_mla=enable_mla,
        mla_absorption_config=mla_absorption_config,
        num_devices=num_devices,
    )
