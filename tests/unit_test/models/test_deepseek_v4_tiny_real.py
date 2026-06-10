# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Real-weight parity test for ``deepseek_v4``.

Loads a layer-and-expert-trimmed slice of ``deepseek-ai/DeepSeek-V4-Pro`` with
real Hub weights via ``runners.real_weight_tiny.load_trimmed_real_weights``,
runs a CPU forward through HF and through the QEff PyTorch path, and asserts
logit-level parity (cosine >= 0.99, relative L2 <= 5e-2).

The dummy-weight ``test_deepseek_v4_tiny.py`` validates structure only;
this test catches numerical regressions in:
- RoPE undo direction on the V == K shared head.
- HyperConnection collapse/expand combine.
- MoE routing weight aggregation.

Skipped when ``HF_TOKEN`` is unset, when the model can't be downloaded,
or when the package layout has shifted under us.
"""

import copy
import os
import sys
from pathlib import Path

import pytest
import torch

# runners/ lives one level up from the fork; prepend so we can import
# the real-weight loader.
_RUNNERS_DIR = Path(__file__).resolve().parents[4] / "runners"
if str(_RUNNERS_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNNERS_DIR))

try:
    from real_weight_tiny import (  # type: ignore[import-not-found]
        assert_logit_parity,
        hf_token_available,
        load_trimmed_real_weights,
    )
except ImportError as e:
    pytest.skip(f"runners/real_weight_tiny.py not on path: {e}", allow_module_level=True)

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

MODEL_ID = "deepseek-ai/DeepSeek-V4-Pro"
TRIM_LAYERS = 2
TRIM_EXPERTS = 8


@pytest.fixture(scope="module")
def trimmed_model():
    if not hf_token_available():
        pytest.skip("HF_TOKEN not set — real-weight parity test skipped.")
    try:
        model, _cfg = load_trimmed_real_weights(
            MODEL_ID,
            num_hidden_layers=TRIM_LAYERS,
            n_routed_experts=TRIM_EXPERTS,
            additional_config_overrides={"num_experts_per_tok": 2},
            torch_dtype=torch.float32,
        )
    except Exception as e:
        pytest.skip(f"Could not load trimmed real weights for {MODEL_ID}: {type(e).__name__}: {e}")
    return model


def test_real_weight_logit_parity(trimmed_model):
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    hf_copy = copy.deepcopy(trimmed_model)
    qeff_copy = copy.deepcopy(trimmed_model)

    with torch.no_grad():
        hf_out = hf_copy(input_ids=input_ids).logits

    qeff = QEFFAutoModelForCausalLM(qeff_copy, continuous_batching=False, pretrained_model_name_or_path=MODEL_ID)
    qeff.transform(ctx_len=32, seq_len=8, batch_size=1, num_devices=1)
    with torch.no_grad():
        qeff_out = qeff.model(input_ids=input_ids).logits

    metrics = assert_logit_parity(hf_out, qeff_out, label="deepseek_v4 trimmed", min_cosine=0.99, max_relative_l2=5e-2)
    print(f"[parity] cosine_min={metrics['cosine_min']:.4f}  rel_l2={metrics['relative_l2']:.4e}")
