# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Single-entry SPD runner.

Given just a TLM model_name (short name from the supported table), this script:
  1. Looks up the matching DFlash DLM repo on Hugging Face.
  2. Reads hidden_size and block_size from the DLM config.
  3. Compiles TLM + DLM QPCs (if --tlm_qpc / --dlm_qpc are not supplied).
  4. Runs the SPD benchmark on the chosen dataset (default: humaneval).

Examples:
    # Compile + run with all defaults
    python benchmark.py --model_name Qwen3-4B

    # Reuse pre-compiled QPCs (no compilation step)
    python benchmark.py --model_name Qwen3-4B \\
        --tlm_qpc /path/to/tlm/qpc --dlm_qpc /path/to/dlm/qpc

    # Custom devices / cores / dataset
    python benchmark.py --model_name Llama-3.1-8B-Instruct \\
        --tlm_devices 0,1,2,3 --dlm_devices 4,5,6,7 \\
        --tlm_cores 8 --dlm_cores 8 --dataset gsm8k
"""

import argparse
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, THIS_DIR)

from utils import MODEL_MAP, compile_dlm_qpc, compile_tlm_qpc, resolve_model_name  # noqa: E402


def parse_device_list(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model_name",
        required=True,
        type=resolve_model_name,
        help="TLM name — either the short key (e.g. 'Qwen3-4B') or "
        "the full HF repo path (e.g. 'Qwen/Qwen3-4B'). "
        f"Supported: {', '.join(MODEL_MAP.keys())}",
    )
    p.add_argument("--tlm_hf_path", default=None, help="Override TLM HF repo (required if mapping has None).")

    # Optional pre-built QPCs (skip compilation)
    p.add_argument("--tlm_qpc", default=None, help="Pre-compiled TLM qpc dir (skip TLM compile).")
    p.add_argument("--dlm_qpc", default=None, help="Pre-compiled DLM qpc dir (skip DLM compile).")

    # Devices / cores
    p.add_argument(
        "--tlm_devices",
        type=parse_device_list,
        default=[0, 1, 2, 3],
        help="Comma-separated device IDs, e.g. '0,1,2,3' or '0'.",
    )
    p.add_argument(
        "--dlm_devices",
        type=parse_device_list,
        default=[0, 1, 2, 3],
        help="Comma-separated device IDs, e.g. '0,1,2,3' or '0'.",
    )
    p.add_argument("--tlm_cores", type=int, default=8)
    p.add_argument("--dlm_cores", type=int, default=8)

    # Compile / run knobs
    p.add_argument("--ctx_len", type=int, default=4096)
    p.add_argument("--prefill_seq_len", type=int, default=128)
    p.add_argument("--generation_len", type=int, default=1024)
    p.add_argument("--iteration", type=int, default=300)

    # Dataset / output
    p.add_argument("--dataset", default="humaneval", choices=["humaneval", "gsm8k", "math500"])
    p.add_argument("--num_samples", type=int, default=0, help="0 = all samples")
    p.add_argument("--output_dir", default=None, help="Default: ./results-<model_name>")
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


def main():
    args = parse_args()

    tlm_repo_default, dlm_repo = MODEL_MAP[args.model_name]
    tlm_repo = args.tlm_hf_path or tlm_repo_default
    if tlm_repo is None:
        raise SystemExit(f"No default TLM HF path for '{args.model_name}'. Pass --tlm_hf_path.")

    if args.tlm_qpc:
        print(f"[skip compile] using provided TLM qpc: {args.tlm_qpc}")
        tlm_qpc = args.tlm_qpc
    else:
        tlm_qpc = compile_tlm_qpc(
            tlm_repo,
            dlm_repo,
            prefill_seq_len=args.prefill_seq_len,
            ctx_len=args.ctx_len,
            num_cores=args.tlm_cores,
            num_devices=len(args.tlm_devices),
            hf_token=args.hf_token,
        )

    if args.dlm_qpc:
        print(f"[skip compile] using provided DLM qpc: {args.dlm_qpc}")
        dlm_qpc = args.dlm_qpc
    else:
        dlm_qpc = compile_dlm_qpc(
            tlm_repo,
            dlm_repo,
            ctx_len=args.ctx_len,
            num_cores=args.dlm_cores,
            num_devices=len(args.dlm_devices),
            hf_token=args.hf_token,
        )
    print(f"TLM qpc        : {tlm_qpc}")
    print(f"DLM qpc        : {dlm_qpc}")

    output_dir = args.output_dir or os.path.join(THIS_DIR, f"results-{args.model_name}")

    eval_script = os.path.join(THIS_DIR, "dflash_spd_benchmark.py")
    cmd = [
        sys.executable,
        eval_script,
        "--dataset",
        args.dataset,
        "--tlm_qpc",
        tlm_qpc,
        "--dlm_qpc",
        dlm_qpc,
        "--tlm_model_name",
        tlm_repo,
        "--dlm_model_name",
        dlm_repo,
        "--iteration",
        str(args.iteration),
        "--ctx_len",
        str(args.ctx_len),
        "--generation_len",
        str(args.generation_len),
        "--tlm_devices",
        *[str(d) for d in args.tlm_devices],
        "--dlm_devices",
        *[str(d) for d in args.dlm_devices],
        "--output_dir",
        output_dir,
    ]
    if args.hf_token:
        cmd += ["--hf_token", args.hf_token]
    if args.num_samples and args.num_samples > 0:
        cmd += ["--num_samples", str(args.num_samples)]

    print("\n>>> launching SPD eval:")
    print(" ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise SystemExit(f"SPD eval exited with rc={rc}")


if __name__ == "__main__":
    main()
