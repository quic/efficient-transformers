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
    python run_spd.py --model_name Qwen3-4B

    # Reuse pre-compiled QPCs (no compilation step)
    python run_spd.py --model_name Qwen3-4B \
        --tlm_qpc /path/to/tlm/qpc --dlm_qpc /path/to/dlm/qpc

    # Custom devices / cores / dataset
    python run_spd.py --model_name Llama-3.1-8B-Instruct \
        --tlm_devices 0 1 2 3 --dlm_devices 4 5 6 7 \
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


# ─────────────────────────────────────────────────────────────────────────────
# model_name (TLM short)  →  (TLM HF repo, DLM HF repo)
# DLM column comes verbatim from the user's supported list.
# TLM column is the standard HF repo when known; otherwise None and must be
# supplied via --tlm_hf_path on the command line.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_MAP = {
    "gemma-4-31B-it":          (None,                                "z-lab/gemma-4-31B-it-DFlash"),
    "gemma-4-26B-A4B-it":      (None,                                "z-lab/gemma-4-26B-A4B-it-DFlash"),
    "MiniMax-M2.7":            (None,                                "z-lab/MiniMax-M2.7-DFlash"),
    "MiniMax-M2.5":            (None,                                "z-lab/MiniMax-M2.5-DFlash"),
    "Kimi-K2.6":               (None,                                "z-lab/Kimi-K2.6-DFlash"),
    "Kimi-K2.5":               (None,                                "z-lab/Kimi-K2.5-DFlash"),
    "Qwen3.6-27B":             (None,                                "z-lab/Qwen3.6-27B-DFlash"),
    "Qwen3.6-35B-A3B":         (None,                                "z-lab/Qwen3.6-35B-A3B-DFlash"),
    "Qwen3.5-4B":              (None,                                "z-lab/Qwen3.5-4B-DFlash"),
    "Qwen3.5-9B":              (None,                                "z-lab/Qwen3.5-9B-DFlash"),
    "Qwen3.5-27B":             (None,                                "z-lab/Qwen3.5-27B-DFlash"),
    "Qwen3.5-35B-A3B":         (None,                                "z-lab/Qwen3.5-35B-A3B-DFlash"),
    "Qwen3.5-122B-A10B":       (None,                                "z-lab/Qwen3.5-122B-A10B-DFlash"),
    "gpt-oss-20b":             ("openai/gpt-oss-20b",                "z-lab/gpt-oss-20b-DFlash"),
    "gpt-oss-120b":            ("openai/gpt-oss-120b",               "z-lab/gpt-oss-120b-DFlash"),
    "Qwen3-Coder-Next":        (None,                                "z-lab/Qwen3-Coder-Next-DFlash"),
    "Qwen3-4B":                ("Qwen/Qwen3-4B",                     "z-lab/Qwen3-4B-DFlash-b16"),
    "Qwen3-8B":                ("Qwen/Qwen3-8B",                     "z-lab/Qwen3-8B-DFlash-b16"),
    "Qwen3-Coder-30B-A3B":     ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "z-lab/Qwen3-Coder-30B-A3B-DFlash"),
    "Llama-3.1-8B-Instruct":   ("meta-llama/Llama-3.1-8B-Instruct",  "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"),
}


# Build alias table: full HF repo path (e.g. "Qwen/Qwen3-4B") and basename
# (case-insensitive) → canonical short name. Lets users pass either form.
def _build_aliases(model_map):
    aliases = {}
    for short, (tlm_repo, _) in model_map.items():
        aliases[short.lower()] = short
        if tlm_repo:
            aliases[tlm_repo.lower()] = short
            aliases[tlm_repo.split("/", 1)[-1].lower()] = short
    return aliases


MODEL_ALIASES = _build_aliases(MODEL_MAP)


def resolve_model_name(name):
    """Map a user-supplied model name (short, full HF path, or basename) to
    the canonical short name used as a key in MODEL_MAP."""
    canonical = MODEL_ALIASES.get(name.lower())
    if canonical is None:
        raise argparse.ArgumentTypeError(
            f"unknown model_name '{name}'. Supported: "
            + ", ".join(sorted(MODEL_MAP.keys()))
        )
    return canonical


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True, type=resolve_model_name,
                   help="TLM name — either the short key (e.g. 'Qwen3-4B') or "
                        "the full HF repo path (e.g. 'Qwen/Qwen3-4B'). "
                        f"Supported: {', '.join(MODEL_MAP.keys())}")
    p.add_argument("--tlm_hf_path", default=None,
                   help="Override TLM HF repo (required if mapping has None).")

    # Optional pre-built QPCs (skip compilation)
    p.add_argument("--tlm_qpc", default=None, help="Pre-compiled TLM qpc dir (skip TLM compile).")
    p.add_argument("--dlm_qpc", default=None, help="Pre-compiled DLM qpc dir (skip DLM compile).")

    # Devices / cores
    p.add_argument("--tlm_devices", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--dlm_devices", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--tlm_cores", type=int, default=8)
    p.add_argument("--dlm_cores", type=int, default=8)

    # Compile / run knobs
    p.add_argument("--ctx_len",         type=int, default=4096)
    p.add_argument("--prefill_seq_len", type=int, default=128)
    p.add_argument("--generation_len",  type=int, default=1024)
    p.add_argument("--iteration",       type=int, default=300)

    # Dataset / output
    p.add_argument("--dataset", default="humaneval", choices=["humaneval", "gsm8k", "math500"])
    p.add_argument("--num_samples", type=int, default=0, help="0 = all samples")
    p.add_argument("--output_dir",  default=None, help="Default: ./results-<model_name>")
    p.add_argument("--noise_embed_path", default=None,
                   help="Defaults to noise_embedding/<model_name>_noise_embeds.npy")
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))

    # Internal modes used by self-spawned compile subprocesses
    p.add_argument("--_build", choices=["tlm", "dlm"], default=None,
                   help=argparse.SUPPRESS)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Compilation helpers — mirror make_models.py but parameterised
# ─────────────────────────────────────────────────────────────────────────────
def _read_dlm_meta(dlm_repo, hf_token):
    from utils import load_dflash_checkpoint
    state_dict, cfg = load_dflash_checkpoint(dlm_repo)
    target_layer_ids = cfg.get("dflash_config", {}).get("target_layer_ids", [])
    block_size = cfg.get("block_size", None)
    return state_dict, target_layer_ids, block_size


def _build_tlm(args, tlm_repo, dlm_repo):
    import torch
    from transformers import AutoModelForCausalLM
    from QEfficient import QEFFAutoModelForCausalLM
    from utils import build_tlm_model

    state_dict, target_layer_ids, block_size = _read_dlm_meta(dlm_repo, args.hf_token)
    tlm_target_ids = [i + 1 for i in target_layer_ids]

    print(f"[build_tlm] base={tlm_repo}  dlm={dlm_repo}  block_size={block_size}")
    base_model = AutoModelForCausalLM.from_pretrained(tlm_repo, torch_dtype=torch.float32, token=args.hf_token)
    build_tlm_model(base_model, state_dict, tlm_target_ids)

    tlm_qeff = QEFFAutoModelForCausalLM(base_model, qaic_config={"target_layer_ids": tlm_target_ids})
    qpc = tlm_qeff.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.tlm_cores,
        num_devices=len(args.tlm_devices),
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        dflash_block_size=block_size,
    )
    print(f"TLM_QPC={qpc}")
    return qpc


def _build_dlm(args, tlm_repo, dlm_repo):
    import torch
    from transformers import AutoModelForCausalLM
    from QEfficient import QEFFAutoModelForCausalLM
    from utils import build_dlm_model, extract_lm_head

    _, _, block_size = _read_dlm_meta(dlm_repo, args.hf_token)

    print(f"[build_dlm] dlm={dlm_repo}  block_size={block_size}")
    base_model = AutoModelForCausalLM.from_pretrained(tlm_repo, torch_dtype=torch.float32, token=args.hf_token)
    lm_head_w, lm_head_b = extract_lm_head(base_model)
    del base_model

    dlm_model = build_dlm_model(dlm_repo, lm_head_w, lm_head_b)
    dlm_qeff = QEFFAutoModelForCausalLM(dlm_model, qaic_config={"dflash_dlm": True})
    qpc = dlm_qeff.compile(
        prefill_seq_len=block_size,
        ctx_len=args.ctx_len,
        num_cores=args.dlm_cores,
        num_devices=len(args.dlm_devices),
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        mos=1,
        prefill_only=True,
    )
    print(f"DLM_QPC={qpc}")
    return qpc


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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    tlm_repo_default, dlm_repo = MODEL_MAP[args.model_name]
    tlm_repo = args.tlm_hf_path or tlm_repo_default
    if tlm_repo is None:
        raise SystemExit(
            f"No default TLM HF path for '{args.model_name}'. Pass --tlm_hf_path."
        )

    # ── Sub-mode: this process exists only to compile one model ─────────────
    if args._build == "tlm":
        _build_tlm(args, tlm_repo, dlm_repo)
        return
    if args._build == "dlm":
        _build_dlm(args, tlm_repo, dlm_repo)
        return

    # ── Resolve / discover hidden_size + block_size from DLM config ────────
    import transformers
    config = transformers.AutoConfig.from_pretrained(
        dlm_repo, token=args.hf_token, trust_remote_code=True
    )
    hidden_size = config.hidden_size
    block_size = getattr(config, "block_size", None)
    print(f"DLM repo       : {dlm_repo}")
    print(f"hidden_size    : {hidden_size}")
    print(f"block_size     : {block_size}")

    # ── Resolve QPC paths (compile only the side that wasn't pre-supplied) ─
    forwarded = [
        "--model_name", args.model_name,
        "--ctx_len", str(args.ctx_len),
        "--prefill_seq_len", str(args.prefill_seq_len),
        "--tlm_cores", str(args.tlm_cores),
        "--dlm_cores", str(args.dlm_cores),
        "--tlm_devices", *[str(d) for d in args.tlm_devices],
        "--dlm_devices", *[str(d) for d in args.dlm_devices],
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
        raise SystemExit(
            f"noise embedding not found: {noise_embed}\n"
            f"Pass --noise_embed_path explicitly."
        )

    output_dir = args.output_dir or os.path.join(THIS_DIR, f"results-{args.model_name}")

    # ── Run the existing SPD eval script ───────────────────────────────────
    eval_script = os.path.join(THIS_DIR, "dflash_spd_benchmark.py")
    cmd = [
        sys.executable, eval_script,
        "--dataset",          args.dataset,
        "--tlm_qpc",          tlm_qpc,
        "--dlm_qpc",          dlm_qpc,
        "--tlm_model_name",   tlm_repo,
        "--dlm_model_name",   dlm_repo,
        "--noise_embed_path", noise_embed,
        "--iteration",        str(args.iteration),
        "--ctx_len",          str(args.ctx_len),
        "--generation_len",   str(args.generation_len),
        "--tlm_devices",      *[str(d) for d in args.tlm_devices],
        "--dlm_devices",      *[str(d) for d in args.dlm_devices],
        "--output_dir",       output_dir,
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
