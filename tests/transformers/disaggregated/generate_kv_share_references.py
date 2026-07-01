# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Reference generator for test_kv_share_disagg.py.

Run this script once on QAIC hardware to produce the golden token IDs that
the unit tests assert against:

    python generate_kv_share_references.py [--output kv_share_references.json]

The script runs the same _run_kv_share_disagg() pipeline used by the tests
and writes the generated token IDs to the reference JSON file.  Re-run
whenever the model config, compile flags, or generation length change.
"""

import argparse
import json
import os
import sys

from transformers import AutoTokenizer

# Allow running from the repo root or from the test directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from test_kv_share_disagg import _GENERATION_LEN, _KV_SHARE_CASES, _ref_key, _run_kv_share_disagg


def main():
    parser = argparse.ArgumentParser(description="Generate KV-share disagg reference tokens.")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "kv_share_references.json"),
        help="Path to write the reference JSON file.",
    )
    parser.add_argument(
        "--generation-len",
        type=int,
        default=_GENERATION_LEN,
        help=f"Number of tokens to generate per case (default: {_GENERATION_LEN}).",
    )
    args = parser.parse_args()

    refs: dict = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            refs = json.load(f)
        print(f"Loaded {len(refs)} existing reference(s) from {args.output}")

    for model_id, overrides, tokenizer_id, prompt_key, prompt in _KV_SHARE_CASES:
        key = _ref_key(model_id, prompt_key)
        print(f"\nGenerating reference for: {key}")
        print(f"  prompt : {prompt[:60]!r}{'...' if len(prompt) > 60 else ''}")

        tokens = _run_kv_share_disagg(
            model_id=model_id,
            overrides=overrides,
            tokenizer_id=tokenizer_id,
            prompt=prompt,
            generation_len=args.generation_len,
        )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  tokens : {tokens}")
        print(f"  text   : {text!r}")

        refs[key] = tokens

    with open(args.output, "w") as f:
        json.dump(refs, f, indent=2)
    print(f"\nWrote {len(refs)} reference(s) to {args.output}")


if __name__ == "__main__":
    main()
