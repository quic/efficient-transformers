# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Single-entry SPD runner for the VLM (image+text) benchmark.

This is the multimodal counterpart of benchmark.py. Given a VLM model_name,
this script:
  1. Looks up the matching DFlash DLM repo on Hugging Face (MODEL_MAP).
  2. Reads target_layer_ids / block_size from the DLM config.
  3. Compiles the VLM TLM (vision encoder + language decoder) and the DLM QPCs
     (unless --tlm_qpc / --vision_qpc / --dlm_qpc are supplied).
  4. Runs dflash_spd_vision_benchmark.py over the MathVision dataset.

Because the QPC cache is keyed by model config + compile options, re-running with the
same knobs is a cache hit (no recompile); changing cores / devices / ctx_len / prefill
transparently rebuilds only what changed — so sweeping experiments is cheap.

Examples:
    # Compile + run with defaults (gemma-4-31B-it)
    python benchmark_vision.py --model_name gemma-4-31B-it \\
        --tlm_hf_path google/gemma-4-31B-it \\
        --tlm_devices 12,13,14,15 --dlm_devices 16,17,18,19 --vision_devices 20,21,22,23

    # Reuse pre-compiled QPCs (no compilation step)
    python benchmark_vision.py --model_name gemma-4-31B-it \\
        --tlm_qpc /path/to/lang/qpc --vision_qpc /path/to/vision/qpc --dlm_qpc /path/to/dlm/qpc \\
        --tlm_devices 12,13,14,15 --dlm_devices 16,17,18,19 --vision_devices 20,21,22,23

    # Custom cores / ctx / sample count
    python benchmark_vision.py --model_name gemma-4-31B-it --tlm_hf_path google/gemma-4-31B-it \\
        --tlm_cores 8 --ctx_len 2048 --split testmini --num_samples 20

Note: set QEFF_HOME to a filesystem with free space (compiles are large), e.g.
    export QEFF_HOME=/local/mnt/workspace/<user>/qeff_home
"""

import argparse
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, THIS_DIR)

from utils import MODEL_MAP, compile_gemma_vlm_dlm_qpc, compile_gemma_vlm_qpcs, resolve_model_name  # noqa: E402


def parse_device_list(s):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model_name",
        default="gemma-4-31B-it",
        type=resolve_model_name,
        help="VLM name — short key (e.g. 'gemma-4-31B-it') or full HF repo path. "
        f"Supported: {', '.join(MODEL_MAP.keys())}",
    )
    p.add_argument(
        "--tlm_hf_path",
        default="google/gemma-4-31B-it",
        help="TLM HF repo (MODEL_MAP's VLM entries have no default TLM path).",
    )

    # Optional pre-built QPCs (skip compilation)
    p.add_argument("--tlm_qpc", default=None, help="Pre-compiled language (TLM) qpc dir (skip TLM compile).")
    p.add_argument("--vision_qpc", default=None, help="Pre-compiled vision-encoder qpc dir (skip TLM compile).")
    p.add_argument("--dlm_qpc", default=None, help="Pre-compiled DLM qpc dir (skip DLM compile).")

    # Devices / cores (three towers: language, draft, vision)
    p.add_argument(
        "--tlm_devices",
        type=parse_device_list,
        default=[0, 1, 2, 3],
        help="Comma-separated device IDs for the language QPC, e.g. '12,13,14,15'.",
    )
    p.add_argument(
        "--dlm_devices",
        type=parse_device_list,
        default=[0, 1, 2, 3],
        help="Comma-separated device IDs for the DLM QPC, e.g. '16,17,18,19'.",
    )
    p.add_argument(
        "--vision_devices",
        type=parse_device_list,
        default=[0, 1, 2, 3],
        help="Comma-separated device IDs for the vision QPC. TLM+DLM fill their cards, so "
        "vision usually needs its own, e.g. '20,21,22,23'.",
    )
    p.add_argument("--tlm_cores", type=int, default=8, help="AIC cores for the language decoder.")
    p.add_argument("--dlm_cores", type=int, default=8, help="AIC cores for the DLM.")

    # Compile / run knobs
    p.add_argument("--ctx_len", type=int, default=2048)
    p.add_argument("--prefill_seq_len", type=int, default=128)
    p.add_argument("--generation_len", type=int, default=256)
    p.add_argument("--iteration", type=int, default=300)

    # Dataset / output
    p.add_argument("--split", default="testmini", help="MathVision split: testmini (304) or test (3040).")
    p.add_argument("--num_samples", type=int, default=20, help="Number of samples (0 = all).")
    p.add_argument(
        "--dataset",
        default=None,
        choices=["humaneval", "gsm8k", "math500"],
        help="Run a TEXT dataset through the language decoder only (no image, vision_embeds zero-bound) "
        "instead of the MathVision image dataset. Omit for the default MathVision run.",
    )
    p.add_argument("--output_dir", default=None, help="Default: ./results-mathvision-<model_name>")
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


def main():
    args = parse_args()

    _tlm_repo_default, dlm_repo = MODEL_MAP[args.model_name]
    tlm_repo = args.tlm_hf_path or _tlm_repo_default
    if tlm_repo is None:
        raise SystemExit(f"No default TLM HF path for '{args.model_name}'. Pass --tlm_hf_path.")

    # ── TLM (vision encoder + language decoder) ──────────────────────────────────
    if args.tlm_qpc and args.vision_qpc:
        print(f"[skip compile] using provided TLM lang qpc : {args.tlm_qpc}")
        print(f"[skip compile] using provided vision qpc   : {args.vision_qpc}")
        tlm_qpc, vision_qpc = args.tlm_qpc, args.vision_qpc
    else:
        if args.tlm_qpc or args.vision_qpc:
            print("[info] both --tlm_qpc and --vision_qpc are required to skip the VLM build; rebuilding both.")
        tlm_qpc, vision_qpc = compile_gemma_vlm_qpcs(
            tlm_repo,
            dlm_repo,
            prefill_seq_len=args.prefill_seq_len,
            ctx_len=args.ctx_len,
            num_cores=args.tlm_cores,
            num_devices=len(args.tlm_devices),
            hf_token=args.hf_token,
        )

    # ── DLM (draft) ──────────────────────────────────────────────────────────────
    if args.dlm_qpc:
        print(f"[skip compile] using provided DLM qpc      : {args.dlm_qpc}")
        dlm_qpc = args.dlm_qpc
    else:
        dlm_qpc = compile_gemma_vlm_dlm_qpc(
            tlm_repo,
            dlm_repo,
            ctx_len=args.ctx_len,
            num_cores=args.dlm_cores,
            num_devices=len(args.dlm_devices),
            hf_token=args.hf_token,
        )

    print(f"TLM lang qpc   : {tlm_qpc}")
    print(f"Vision qpc     : {vision_qpc}")
    print(f"DLM qpc        : {dlm_qpc}")

    if args.dataset:
        # Language-only text-dataset benchmark: drive the language decoder over a text
        # dataset with no image (vision_embeds zero-bound); the vision QPC is not loaded.
        output_dir = args.output_dir or os.path.join(THIS_DIR, f"results-text-{args.model_name}")
        eval_script = os.path.join(THIS_DIR, "dflash_spd_vision_text_benchmark.py")
        cmd = [
            sys.executable,
            eval_script,
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
            "--dataset",
            args.dataset,
            "--output_dir",
            output_dir,
        ]
    else:
        # Default: MathVision image dataset (vision encoder + language decoder).
        output_dir = args.output_dir or os.path.join(THIS_DIR, f"results-mathvision-{args.model_name}")
        eval_script = os.path.join(THIS_DIR, "dflash_spd_vision_benchmark.py")
        cmd = [
            sys.executable,
            eval_script,
            "--tlm_qpc",
            tlm_qpc,
            "--dlm_qpc",
            dlm_qpc,
            "--vision_qpc",
            vision_qpc,
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
            "--vision_devices",
            *[str(d) for d in args.vision_devices],
            "--split",
            args.split,
            "--output_dir",
            output_dir,
        ]
    if args.hf_token:
        cmd += ["--hf_token", args.hf_token]
    if args.num_samples and args.num_samples > 0:
        cmd += ["--num_samples", str(args.num_samples)]

    print("\n>>> launching vision SPD eval:")
    print(" ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise SystemExit(f"vision SPD eval exited with rc={rc}")


if __name__ == "__main__":
    main()
