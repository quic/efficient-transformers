# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Build and compile TLM + DLM QPC packages for DFlash speculative decoding.

Usage:
    python make_models.py                  # build both (TLM + DLM) in separate subprocesses
    python make_models.py --mode tlm       # build TLM only (single process)
    python make_models.py --mode dlm       # build DLM only (single process)
    python make_models.py --mode both      # alias for default

The default 'both' mode launches one subprocess per model so that compiler
state from the TLM build cannot affect the DLM build (back-to-back compiles
in the same process have been observed to segfault).
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from transformers import AutoModelForCausalLM
from QEfficient import QEFFAutoModelForCausalLM

from utils import build_dlm_model, build_tlm_model, extract_lm_head, load_dflash_checkpoint

# ── Paths ─────────────────────────────────────────────────────────────────────
TLM_MODEL_PATH = "Qwen/Qwen3-4B"
# TLM_MODEL_PATH = "Qwen/Qwen3-8B"
# TLM_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"

DFLASH_MODEL_PATH = "z-lab/Qwen3-4B-DFlash-b16"
# DFLASH_MODEL_PATH = "z-lab/Qwen3-8B-DFlash-b16"
# DFLASH_MODEL_PATH = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"

# ── Compile options ───────────────────────────────────────────────────────────
COMPILE_KWARGS = dict(
    ctx_len=4096,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=4,
    mos=1,
    
)

TLM_NUM_CORES = 8
DLM_NUM_CORES = 16


def _load_dflash_meta():
    dflash_state_dict, cfg = load_dflash_checkpoint(DFLASH_MODEL_PATH)
    target_layer_ids = cfg.get("dflash_config", {}).get("target_layer_ids", [])
    mask_token_id = cfg.get("dflash_config", {}).get("mask_token_id", [])
    block_size = cfg.get("block_size", None)
    print(f"  target_layer_ids : {target_layer_ids}")
    print(f"  mask_token_id    : {mask_token_id}")
    print(f"  block_size       : {block_size}")
    return dflash_state_dict, target_layer_ids, block_size


def build_tlm():
    print(f"Loading DFlash checkpoint: {DFLASH_MODEL_PATH}")
    dflash_state_dict, target_layer_ids, block_size = _load_dflash_meta()

    print(f"\nLoading base model: {TLM_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(TLM_MODEL_PATH, torch_dtype=torch.float32)

    print("\n=== TLM ===")
    tlm_target_ids = [i + 1 for i in target_layer_ids]
    build_tlm_model(base_model, dflash_state_dict, tlm_target_ids)

    tlm_qeff = QEFFAutoModelForCausalLM(base_model, qaic_config={"target_layer_ids": tlm_target_ids})
    tlm_qpc_path = tlm_qeff.compile(
        prefill_seq_len=128,
        num_cores=TLM_NUM_CORES,
        dflash_block_size=block_size,
        **COMPILE_KWARGS,
    )
    print(f"tlm_qpc_path: {tlm_qpc_path}")
    return tlm_qpc_path


def build_dlm():
    print(f"Loading DFlash checkpoint: {DFLASH_MODEL_PATH}")
    _, _, block_size = _load_dflash_meta()

    print(f"\nLoading base model (for lm_head): {TLM_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(TLM_MODEL_PATH, torch_dtype=torch.float32)
    lm_head_weight, lm_head_bias = extract_lm_head(base_model)
    del base_model

    print("\n=== DLM ===")
    dlm_model = build_dlm_model(DFLASH_MODEL_PATH, lm_head_weight, lm_head_bias)

    dlm_qeff = QEFFAutoModelForCausalLM(dlm_model, qaic_config={"dflash_dlm": True})
    dlm_qpc_path = dlm_qeff.compile(
        prefill_seq_len=block_size,
        num_cores=DLM_NUM_CORES,
        prefill_only=True,
        **COMPILE_KWARGS,
        
    )
    print(f"dlm_qpc_path: {dlm_qpc_path}")
    return dlm_qpc_path


def _run_subprocess(mode: str):
    print(f"\n>>> Spawning subprocess: --mode {mode}")
    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--mode", mode],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"Subprocess for --mode {mode} exited with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["tlm", "dlm", "both"],
        default="both",
        help="Which model(s) to build. 'both' (default) runs TLM and DLM in separate subprocesses.",
    )
    args = parser.parse_args()

    if args.mode == "tlm":
        build_tlm()
    elif args.mode == "dlm":
        build_dlm()
    else:
        _run_subprocess("tlm")
        _run_subprocess("dlm")
        print("\n=== Done ===")


if __name__ == "__main__":
    main()
