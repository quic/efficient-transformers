# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Single-prompt SPD runner for the VLM (text or single image+text).

The multimodal counterpart of basic_inference_text.py. Given a VLM model_name, this
script:
  1. Looks up the matching DFlash DLM repo on Hugging Face (MODEL_MAP).
  2. Compiles the VLM TLM (vision encoder + language decoder) and the DLM QPCs
     (unless --tlm_qpc / --vision_qpc / --dlm_qpc are supplied).
  3. Runs SPD inference in-process (QEfficient.generation.dflash_generation.run_vision_inference)
     on a single text prompt, or on a single image+text prompt when --image is given.

Examples:
    # Text prompt through the VLM
    python basic_inference_vision.py --model_name gemma-4-31B-it --tlm_hf_path google/gemma-4-31B-it \\
        --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \\
        --prompt "Tell me about the Taj Mahal."

    # Single image + text prompt
    python basic_inference_vision.py --model_name gemma-4-31B-it --tlm_hf_path google/gemma-4-31B-it \\
        --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \\
        --image --image_url https://.../photo.jpg --image_prompt "Describe this image in detail."

    # Reuse pre-built QPCs (skip compilation)
    python basic_inference_vision.py --model_name gemma-4-31B-it \\
        --tlm_qpc /path/lang/qpc --vision_qpc /path/vision/qpc --dlm_qpc /path/dlm/qpc \\
        --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \\
        --prompt "Hello"

    # qwen3-vl: text prompt (TLM/DLM default paths come from MODEL_MAP)
    python basic_inference_vision.py --model_name Qwen3-VL-32B-Instruct \\
        --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \\
        --prompt "Tell me about the Taj Mahal."

    # qwen3-vl: single image + text prompt
    python basic_inference_vision.py --model_name Qwen3-VL-32B-Instruct \\
        --tlm_devices 40,41,42,43 --dlm_devices 44,45,46,47 --vision_devices 48,49,50,51 \\
        --image --image_url https://.../photo.jpg --image_prompt "Describe this image in detail."

Note: set QEFF_HOME to a filesystem with free space (compiles are large), e.g.
    export QEFF_HOME=/local/mnt/workspace/<user>/qeff_home
"""

import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, THIS_DIR)

from rich.console import Console
from rich.markup import escape
from utils import (
    MODEL_MAP,
    compile_gemma_vlm_dlm_qpc,
    compile_gemma_vlm_qpcs,
    compile_qwen3vl_vlm_dlm_qpc,
    compile_qwen3vl_vlm_qpcs,
    resolve_model_name,
)

from QEfficient.generation.dflash_generation import run_vision_inference

console = Console()


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
        default=None,
        help="TLM HF repo. Falls back to MODEL_MAP's default when the entry has one "
        "(e.g. Qwen3-VL-32B-Instruct); required otherwise (e.g. gemma entries have none).",
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
    p.add_argument(
        "--height", type=int, default=None, help="Vision input height (qwen3-vl only; falls back to a model default)."
    )
    p.add_argument(
        "--width", type=int, default=None, help="Vision input width (qwen3-vl only; falls back to a model default)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.image and not args.prompt:
        raise SystemExit("Provide --prompt for a text prompt, or --image (+ --image_url/--image_prompt) for an image.")

    _tlm_repo_default, dlm_repo = MODEL_MAP[args.model_name]
    tlm_repo = args.tlm_hf_path or _tlm_repo_default
    if tlm_repo is None:
        raise SystemExit(f"No default TLM HF path for '{args.model_name}'. Pass --tlm_hf_path.")

    from transformers import AutoConfig

    tlm_config = AutoConfig.from_pretrained(tlm_repo, trust_remote_code=True, token=args.hf_token)
    is_qwen3vl = tlm_config.model_type == "qwen3_vl"

    # ── TLM (vision encoder + language decoder) ──────────────────────────────────
    if args.tlm_qpc and args.vision_qpc:
        print(f"[skip compile] using provided TLM lang qpc : {args.tlm_qpc}")
        print(f"[skip compile] using provided vision qpc   : {args.vision_qpc}")
        tlm_qpc, vision_qpc = args.tlm_qpc, args.vision_qpc
    else:
        if args.tlm_qpc or args.vision_qpc:
            print("[info] both --tlm_qpc and --vision_qpc are required to skip the VLM build; rebuilding both.")
        if is_qwen3vl:
            tlm_qpc, vision_qpc = compile_qwen3vl_vlm_qpcs(
                tlm_repo,
                dlm_repo,
                prefill_seq_len=args.prefill_seq_len,
                ctx_len=args.ctx_len,
                num_cores=args.tlm_cores,
                num_devices=len(args.tlm_devices),
                height=args.height,
                width=args.width,
                hf_token=args.hf_token,
            )
        else:
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
        dlm_compile_fn = compile_qwen3vl_vlm_dlm_qpc if is_qwen3vl else compile_gemma_vlm_dlm_qpc
        dlm_qpc = dlm_compile_fn(
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

    metrics, tokenizer, output_extra = run_vision_inference(
        tlm_qpc=tlm_qpc,
        dlm_qpc=dlm_qpc,
        vision_qpc=vision_qpc,
        tlm_model_name=tlm_repo,
        dlm_model_name=dlm_repo,
        tlm_devices=args.tlm_devices,
        dlm_devices=args.dlm_devices,
        vision_devices=args.vision_devices,
        prompt=args.prompt,
        image=args.image,
        image_url=args.image_url,
        image_prompt=args.image_prompt,
        height=args.height,
        width=args.width,
        iteration=args.iteration,
        ctx_len=args.ctx_len,
        generation_len=args.generation_len,
        hf_token=args.hf_token,
    )

    output_parts = ["Output: "]
    for tok_id, source in zip(metrics.generated_ids, metrics.generated_sources):
        text = escape(tokenizer.decode([tok_id], skip_special_tokens=True))
        if source == "dlm":
            output_parts.append(f"[blue]{text}[/blue]")
        else:
            output_parts.append(f"[white]{text}[/white]")
    console.print("".join(output_parts))

    ar = metrics.acceptance_rate()
    dlm_tps = metrics.dlm_tok_rate()
    tlm_tps = metrics.tlm_tok_rate()
    spd_tps = metrics.spd_tok_rate()

    w = 46
    print("\n" + "=" * w)
    print("  SPD Inference — Metrics")
    print("=" * w)
    print(f"  {'Acceptance Rate (tok/iter)':<30} {ar:>6.2f}")
    print(f"  {'DLM Throughput  (tok/s)':<30} {dlm_tps:>6.1f}")
    print(f"  {'TLM Throughput  (tok/s)':<30} {tlm_tps:>6.1f}")
    print(f"  {'SPD Decode Speed (tok/s)':<30} {spd_tps:>6.1f}")
    print(f"  {'Generated tokens':<30} {metrics.total_generated_tokens:>6}")
    print(f"  {'Iterations':<30} {metrics.num_total_iters:>6}")
    print(f"  {'Prefill time (s)':<30} {metrics.total_prefill_time:>6.3f}")
    if output_extra.get("vision_encode_time_s") is not None:
        # Standalone vision-encoder latency (pixel_values -> vision_embeds). This is a
        # one-shot per-image cost and is NOT included in the LM "Prefill time" above.
        print(f"  {'Vision Encode (s)':<30} {output_extra['vision_encode_time_s']:>6.3f}")
    print("=" * w + "\n")


if __name__ == "__main__":
    main()
