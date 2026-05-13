# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Single-entry SPD single-prompt runner.

Given a TLM model_name (short name OR full HF repo path) and a prompt, this
script:
  1. Looks up the matching DFlash DLM repo on Hugging Face.
  2. Reads hidden_size and block_size from the DLM config.
  3. Compiles TLM + DLM QPCs (only the side(s) not provided via
     --tlm_qpc / --dlm_qpc).
  4. Runs the SPD single-prompt inference script.

Examples:
    # Compile + run with all defaults
    python basic_inference.py --model_name Qwen3-4B \
        --prompt "Explain speculative decoding in two sentences."

    # Full HF path also accepted
    python basic_inference.py --model_name Qwen/Qwen3-4B \
        --prompt "Hello"

    # Reuse pre-compiled QPCs
    python basic_inference.py --model_name Qwen3-4B \
        --tlm_qpc /path/to/tlm/qpc --dlm_qpc /path/to/dlm/qpc \
        --prompt "What is 17 * 23?"
"""

import argparse
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, THIS_DIR)

from benchmark import MODEL_MAP, resolve_model_name  # noqa: E402  # reuse the alias table


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
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
    p.add_argument("--prompt", required=True, help="Input prompt text.")
    p.add_argument("--category", default="", help="Prompt category for formatting (math, coding, reasoning, …).")
    p.add_argument("--tlm_hf_path", default=None, help="Override TLM HF repo (required if mapping has None).")

    # Optional pre-built QPCs (skip compilation)
    p.add_argument("--tlm_qpc", default=None, help="Pre-compiled TLM qpc dir (skip TLM compile).")
    p.add_argument("--dlm_qpc", default=None, help="Pre-compiled DLM qpc dir (skip DLM compile).")

    # Devices / cores
    p.add_argument("--tlm_devices", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--dlm_devices", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--tlm_cores", type=int, default=8)
    p.add_argument("--dlm_cores", type=int, default=8)

    # Compile / run knobs
    p.add_argument("--ctx_len", type=int, default=4096)
    p.add_argument("--prefill_seq_len", type=int, default=128)
    p.add_argument("--generation_len", type=int, default=256)
    p.add_argument("--iteration", type=int, default=300)

    p.add_argument("--noise_embed_path", default=None, help="Defaults to noise_embedding/<model_name>_noise_embeds.npy")
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))

    # Internal modes used by self-spawned compile subprocesses
    p.add_argument("--_build", choices=["tlm", "dlm"], default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    tlm_repo_default, dlm_repo = MODEL_MAP[args.model_name]
    tlm_repo = args.tlm_hf_path or tlm_repo_default
    if tlm_repo is None:
        raise SystemExit(f"No default TLM HF path for '{args.model_name}'. Pass --tlm_hf_path.")

    # Sub-mode: spawned compile subprocess. Reuse benchmark.py's builders so we
    # don't duplicate the compile pipeline.
    if args._build is not None:
        from benchmark import _build_dlm, _build_tlm

        if args._build == "tlm":
            _build_tlm(args, tlm_repo, dlm_repo)
        else:
            _build_dlm(args, tlm_repo, dlm_repo)
        return

    # ── Resolve / discover hidden_size + block_size from DLM config ────────
    import transformers

    config = transformers.AutoConfig.from_pretrained(dlm_repo, token=args.hf_token, trust_remote_code=True)
    hidden_size = config.hidden_size
    block_size = getattr(config, "block_size", None)
    print(f"DLM repo       : {dlm_repo}")
    print(f"hidden_size    : {hidden_size}")
    print(f"block_size     : {block_size}")

    # ── Resolve QPC paths (compile only the side that wasn't pre-supplied) ─
    forwarded = [
        "--model_name",
        args.model_name,
        "--prompt",
        args.prompt,
        "--ctx_len",
        str(args.ctx_len),
        "--prefill_seq_len",
        str(args.prefill_seq_len),
        "--tlm_cores",
        str(args.tlm_cores),
        "--dlm_cores",
        str(args.dlm_cores),
        "--tlm_devices",
        *[str(d) for d in args.tlm_devices],
        "--dlm_devices",
        *[str(d) for d in args.dlm_devices],
    ]
    if args.tlm_hf_path:
        forwarded += ["--tlm_hf_path", args.tlm_hf_path]
    if args.hf_token:
        forwarded += ["--hf_token", args.hf_token]

    if args.tlm_qpc:
        print(f"[skip compile] using provided TLM qpc: {args.tlm_qpc}")
        tlm_qpc = args.tlm_qpc
    else:
        tlm_qpc = _spawn_compile("tlm", forwarded)

    if args.dlm_qpc:
        print(f"[skip compile] using provided DLM qpc: {args.dlm_qpc}")
        dlm_qpc = args.dlm_qpc
    else:
        dlm_qpc = _spawn_compile("dlm", forwarded)
    print(f"TLM qpc        : {tlm_qpc}")
    print(f"DLM qpc        : {dlm_qpc}")

    # ── Resolve noise embed path ───────────────────────────────────────────
    noise_embed = args.noise_embed_path or os.path.join(
        THIS_DIR, "noise_embedding", f"{args.model_name}_noise_embeds.npy"
    )
    if not os.path.exists(noise_embed):
        raise SystemExit(f"noise embedding not found: {noise_embed}\nPass --noise_embed_path explicitly.")

    # ── Run the existing single-prompt inference script ────────────────────
    eval_script = os.path.join(THIS_DIR, "dflash_spd_single_prompt.py")
    cmd = [
        sys.executable,
        eval_script,
        "--prompt",
        args.prompt,
        "--tlm_qpc",
        tlm_qpc,
        "--dlm_qpc",
        dlm_qpc,
        "--tlm_model_name",
        tlm_repo,
        "--dlm_model_name",
        dlm_repo,
        "--noise_embed_path",
        noise_embed,
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
    ]
    if args.hf_token:
        cmd += ["--hf_token", args.hf_token]
    if args.category:
        cmd += ["--category", args.category]

    print("\n>>> launching SPD single-prompt inference:")
    print(" ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise SystemExit(f"single-prompt inference exited with rc={rc}")


def _spawn_compile(mode, argv_template):
    """Run this same script with --_build {mode} in a fresh process and return
    the qpc path printed on the line starting with TLM_QPC= or DLM_QPC=."""
    cmd = [sys.executable, os.path.abspath(__file__), "--_build", mode] + argv_template
    print(f"\n>>> spawning compile subprocess: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"compile subprocess (--_build {mode}) failed (rc={proc.returncode})")

    tag = "TLM_QPC=" if mode == "tlm" else "DLM_QPC="
    qpc_line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(tag)), None)
    if qpc_line is None:
        raise SystemExit(f"could not find {tag} line in compile output")
    return qpc_line.split("=", 1)[1].strip()


if __name__ == "__main__":
    main()
