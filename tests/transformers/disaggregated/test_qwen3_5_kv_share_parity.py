# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Parity: hybrid Qwen3.5-MoE KV-DMA-share disagg path vs. numpy-copy baseline."""

import numpy as np
import pytest

from examples.image_text_to_text.models.qwen3_5_moe.qwen3_5_disagg_mode import run as run_baseline
from examples.image_text_to_text.models.qwen3_5_moe.qwen3_5_disagg_mode_with_kv_share import run as run_kv_share

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
PROMPT = "Tell me about yourself."
PREFILL_SEQ_LEN = 64
CTX_LEN = 4096
GENERATION_LEN = 64
NUM_TOKEN_MATCH = 20
STAGES = 4
PREFILL_NUM_DEVICES = 8
DECODE_NUM_DEVICES = 2


@pytest.mark.on_qaic
@pytest.mark.llm_model
def test_qwen3_5_kv_share_matches_baseline_first_token_and_logits():
    base = run_baseline(
        MODEL_ID,
        PROMPT,
        PREFILL_SEQ_LEN,
        CTX_LEN,
        GENERATION_LEN,
        stages=STAGES,
        prefill_num_devices=PREFILL_NUM_DEVICES,
        decode_num_devices=DECODE_NUM_DEVICES,
    )
    share = run_kv_share(
        MODEL_ID,
        PROMPT,
        PREFILL_SEQ_LEN,
        CTX_LEN,
        GENERATION_LEN,
        stages=STAGES,
        prefill_num_devices=PREFILL_NUM_DEVICES,
        decode_num_devices=DECODE_NUM_DEVICES,
    )

    # First decode token must be identical (argmax over prefill logits).
    assert share["first_token"] == base["first_token"], (
        f"first token mismatch: kv_share={share['first_token']} baseline={base['first_token']}"
    )

    # Prefill logits must match: the two paths share the same QPC, so identical
    # DMA vs host-copy KV must produce bit-exact logits. (Relax atol to 1 only if the
    # mxint8_kv_cache requant path introduces a tolerated 1-ULP diff.)
    np.testing.assert_allclose(
        share["logits"],
        base["logits"],
        rtol=0,
        atol=0,
        err_msg="prefill logits diverged between kv_share and numpy-copy paths",
    )

    # First NUM_TOKEN_MATCH decoded tokens must match: each decode step re-wires the
    # DMA descriptor, so an incorrect per-step KV read/write-back surfaces here even
    # when the first token (which only depends on prefill KV) is correct.
    n = min(NUM_TOKEN_MATCH, len(base["tokens"]), len(share["tokens"]))
    assert share["tokens"][:n] == base["tokens"][:n], (
        f"decoded tokens diverged within first {n}:\n  kv_share={share['tokens'][:n]}\n  baseline={base['tokens'][:n]}"
    )
