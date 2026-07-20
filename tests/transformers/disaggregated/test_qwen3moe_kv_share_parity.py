# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""parity: KV-DMA-share disagg path vs HF generate.
python -m pytest     tests/transformers/disaggregated/test_qwen3moe_kv_share_parity.py     -k test_kv_share_matches_hf_generate_leading_tokens     -m on_qaic -s"""

import os

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from examples.disagg_serving.qwen3moe_disagg_mode_chunking_with_kv_share import run as run_kv_share
from examples.disagg_serving.qwen3moe_disagg_mode_with_chunking import run as run_baseline

# MODEL_ID = "yujiepan/qwen3-moe-tiny-random"
MODEL_ID = "Qwen/Qwen3-30B-A3B"
PROMPT = "Explain quantum computing in simple terms."
PREFILL_SEQ_LEN = 256
CTX_LEN = PREFILL_SEQ_LEN * 3
NUM_TOKEN_MATCH = 50
STAGES = 2
PREFILL_NUM_DEVICES = 2
DECODE_NUM_DEVICES = 1
HF_COMPARE_TOKENS = int(os.environ.get("QEFF_QWEN3MOE_HF_COMPARE_TOKENS", NUM_TOKEN_MATCH))
HF_MIN_LEADING_MATCH = int(os.environ.get("QEFF_QWEN3MOE_HF_MIN_MATCH", 20))
_raw_layers = os.environ.get("QEFF_QWEN3MOE_NUM_HIDDEN_LAYERS")
NUM_HIDDEN_LAYERS = int(_raw_layers) if _raw_layers and _raw_layers.strip() else None


@pytest.mark.on_qaic
def test_kv_share_matches_baseline_first_token_and_logits():
    base = run_baseline(
        MODEL_ID,
        PROMPT,
        PREFILL_SEQ_LEN,
        CTX_LEN,
        STAGES,
        PREFILL_NUM_DEVICES,
        DECODE_NUM_DEVICES,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
    )
    share = run_kv_share(
        MODEL_ID,
        PROMPT,
        PREFILL_SEQ_LEN,
        CTX_LEN,
        STAGES,
        PREFILL_NUM_DEVICES,
        DECODE_NUM_DEVICES,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
    )

    # First decode token must be identical (argmax over prefill logits).
    assert share["first_token"] == base["first_token"], (
        f"first token mismatch: kv_share={share['first_token']} baseline={base['first_token']}"
    )

    np.testing.assert_allclose(
        share["logits"],
        base["logits"],
        rtol=0,
        atol=0,
        err_msg="prefill logits diverged between kv_share and numpy-copy paths",
    )
    n = min(NUM_TOKEN_MATCH, len(base["tokens"]), len(share["tokens"]))
    assert share["tokens"][:n] == base["tokens"][:n], (
        f"decoded tokens diverged within first {n}:\n  kv_share={share['tokens'][:n]}\n  baseline={base['tokens'][:n]}"
    )


def _run_hf_greedy_reference(compare_tokens: int) -> list:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    config = None
    if NUM_HIDDEN_LAYERS is not None:
        config = AutoConfig.from_pretrained(MODEL_ID)
        config.num_hidden_layers = NUM_HIDDEN_LAYERS
    from_pretrained_kwargs = {"config": config} if config is not None else {}

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, attn_implementation="eager", torch_dtype=torch.float32, **from_pretrained_kwargs
    ).eval()

    inputs = tokenizer(PROMPT, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            max_new_tokens=compare_tokens,
            min_new_tokens=compare_tokens,
            do_sample=False,
            num_beams=1,
        )
    return sequences[0, prompt_len:].tolist()


@pytest.mark.on_qaic
def test_kv_share_matches_hf_generate_leading_tokens():
    compare_tokens = HF_COMPARE_TOKENS

    hf_tokens = _run_hf_greedy_reference(compare_tokens)
    share = run_kv_share(
        MODEL_ID,
        PROMPT,
        PREFILL_SEQ_LEN,
        CTX_LEN,
        STAGES,
        PREFILL_NUM_DEVICES,
        DECODE_NUM_DEVICES,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
    )
    qaic_tokens = share["tokens"]

    n = min(compare_tokens, len(hf_tokens), len(qaic_tokens))
    assert n > 0, "no tokens to compare"

    # Length of the leading run that matches token-for-token.
    matched = 0
    for hf_tok, qaic_tok in zip(hf_tokens[:n], qaic_tokens[:n]):
        if hf_tok != qaic_tok:
            break
        matched += 1

    # The first token depends only on prefill KV; a mismatch there is unambiguously a bug.
    assert qaic_tokens[0] == hf_tokens[0], f"first token mismatch: kv_share={qaic_tokens[0]} hf={hf_tokens[0]}"

    assert matched >= min(HF_MIN_LEADING_MATCH, n), (
        f"disagg DMA output diverged from HF generate after only {matched} tokens "
        f"(required >= {min(HF_MIN_LEADING_MATCH, n)} of {n}):\n"
        f"  hf   ={hf_tokens[:n]}\n  qaic ={qaic_tokens[:n]}"
    )
