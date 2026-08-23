# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Single-prompt SPD runner for the VLM (text or single image+text).

The multimodal counterpart of basic_inference.py. Given a VLM model_name, this
script:
  1. Looks up the matching DFlash DLM repo on Hugging Face (MODEL_MAP).
  2. Compiles the VLM TLM (vision encoder + language decoder) and the DLM QPCs
     (unless --tlm_qpc / --vision_qpc / --dlm_qpc are supplied).
  3. Runs dflash_spd_vision_single_prompt.py on a single text prompt, or on a single
     image+text prompt when --image is given.

Examples:
    # Text prompt through the VLM
    python basic_inference_vision.py --model_name gemma-4-31B-it --tlm_hf_path google/gemma-4-31B-it \\
        --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \\
        --prompt "Tell me about the Taj Mahal."

    # Single image + text prompt
    python basic_inference_vision.py --model_name gemma-4-31B-it --tlm_hf_path google/gemma-4-31B-it \\
        --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \\
        --image --image_url https://.../photo.jpg --image_prompt "Describe this image in detail."

    # Reuse pre-compiled QPCs (skip compilation)
    python basic_inference_vision.py --model_name gemma-4-31B-it \\
        --tlm_qpc /path/lang/qpc --vision_qpc /path/vision/qpc --dlm_qpc /path/dlm/qpc \\
        --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \\
        --prompt "Hello"

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

    # Prompt (text) or single image+text prompt
    p.add_argument("--prompt", default=None, help="Text prompt (used when --image is NOT set).")
    p.add_argument("--image", action="store_true", help="Run a single image+text prompt instead of text.")
    p.add_argument("--image_url", default=None, help="Image URL for --image mode.")
    p.add_argument("--image_prompt", default=None, help="Prompt text for --image mode.")

    # Optional pre-built QPCs (skip compilation)
    p.add_argument("--tlm_qpc", default=None, help="Pre-compiled language (TLM) qpc dir (skip TLM compile).")
    p.add_argument("--vision_qpc", default=None, help="Pre-compiled vision-encoder qpc dir (skip TLM compile).")
    p.add_argument("--dlm_qpc", default=None, help="Pre-compiled DLM qpc dir (skip DLM compile).")

    # Devices / cores (three towers: language, draft, vision)
    p.add_argument("--tlm_devices", type=parse_device_list, default=[0, 1, 2, 3], help="e.g. '40,41,42,43'.")
    p.add_argument("--dlm_devices", type=parse_device_list, default=[0, 1, 2, 3], help="e.g. '44,45,46,47'.")
    p.add_argument("--vision_devices", type=parse_device_list, default=[0, 1, 2, 3], help="e.g. '48,49,50,51'.")
    p.add_argument("--tlm_cores", type=int, default=8, help="AIC cores for the language decoder.")
    p.add_argument("--dlm_cores", type=int, default=8, help="AIC cores for the DLM.")

    # Compile / run knobs
    p.add_argument("--ctx_len", type=int, default=2048)
    p.add_argument("--prefill_seq_len", type=int, default=128)
    p.add_argument("--generation_len", type=int, default=256)
    p.add_argument("--iteration", type=int, default=300)
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


def main():
    args = parse_args()

    if not args.image and not args.prompt:
        raise SystemExit("Provide --prompt for a text prompt, or --image (+ --image_url/--image_prompt) for an image.")

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

    eval_script = os.path.join(THIS_DIR, "dflash_spd_vision_single_prompt.py")
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
    ]
    if args.image:
        cmd += ["--image"]
        if args.image_url:
            cmd += ["--image_url", args.image_url]
        if args.image_prompt:
            cmd += ["--image_prompt", args.image_prompt]
    else:
        cmd += ["--prompt", args.prompt]
    if args.hf_token:
        cmd += ["--hf_token", args.hf_token]

    print("\n>>> launching vision single-prompt SPD:")
    print(" ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise SystemExit(f"vision single-prompt SPD exited with rc={rc}")


if __name__ == "__main__":
    main()
