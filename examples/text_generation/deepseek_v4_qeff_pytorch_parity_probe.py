# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""DeepSeek-V4 HF -> QEff PyTorch parity probe using native Transformers support."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM

from QEfficient.transformers.models.deepseek_v4 import build_deepseek_v4_cache
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

DEFAULT_MODEL_ID = "silence09/DeepSeek-V4-Pro-Tiny"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 3, 4])
    args = parser.parse_args()

    model_hf = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=False,
    ).eval()
    qeff_model = QEFFAutoModelForCausalLM(model_hf)

    input_ids = torch.tensor([args.tokens], dtype=torch.long)
    with torch.no_grad():
        hf_logits = model_hf(
            input_ids=input_ids,
            past_key_values=build_deepseek_v4_cache(model_hf.config),
            use_cache=True,
        ).logits
        qeff_logits = qeff_model.model(
            input_ids=input_ids,
            past_key_values=build_deepseek_v4_cache(model_hf.config),
            use_cache=True,
        ).logits

    max_abs_diff = torch.max(torch.abs(hf_logits - qeff_logits)).item()
    print(f"model_id={args.model_id}")
    print(f"hf_logits_shape={tuple(hf_logits.shape)}")
    print(f"qeff_logits_shape={tuple(qeff_logits.shape)}")
    print(f"max_abs_diff={max_abs_diff}")
    if max_abs_diff != 0.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
