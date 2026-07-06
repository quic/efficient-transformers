# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""parity: KV-DMA-share disagg path vs. numpy-copy baseline."""

import numpy as np
import pytest

from examples.disagg_serving.qwen3moe_disagg_mode_chunking_with_kv_share import run as run_kv_share
from examples.disagg_serving.qwen3moe_disagg_mode_with_chunking import run as run_baseline

MODEL_ID = "yujiepan/qwen3-moe-tiny-random"
PROMPT = "Explain quantum computing in simple terms."
PREFILL_SEQ_LEN = 256
CTX_LEN = PREFILL_SEQ_LEN * 3
NUM_TOKEN_MATCH = 100
STAGES = 2
PREFILL_NUM_DEVICES = 2
DECODE_NUM_DEVICES = 1


@pytest.mark.on_qaic
def test_kv_share_matches_baseline_first_token_and_logits():
    base = run_baseline(MODEL_ID, PROMPT, PREFILL_SEQ_LEN, CTX_LEN, STAGES, PREFILL_NUM_DEVICES, DECODE_NUM_DEVICES)
    share = run_kv_share(MODEL_ID, PROMPT, PREFILL_SEQ_LEN, CTX_LEN, STAGES, PREFILL_NUM_DEVICES, DECODE_NUM_DEVICES)

    # First decode token must be identical (argmax over prefill logits).
    assert share["first_token"] == base["first_token"], (
        f"first token mismatch: kv_share={share['first_token']} baseline={base['first_token']}"
    )

    # Prefill logits must match: the two paths share the same QPC, so identical
    # DMA vs host-copy KV must produce bit-exact logits.
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
