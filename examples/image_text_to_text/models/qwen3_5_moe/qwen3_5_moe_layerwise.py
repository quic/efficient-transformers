# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Layerwise prefill compile example for Qwen3.5-MoE.

The orchestration loop that previously lived in this script has been moved
behind the ``layerwise=True`` flag on ``.compile()`` / ``.export()``.

Note: ``layerwise=True`` is a provisional API and is scheduled for deprecation
once first-class multi-window export lands. Supported model types:
``qwen3_vl_moe``, ``qwen3_5_moe``, ``qwen3_moe``.
"""

import torch
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"


def main():
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.torch_dtype = "float32"

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        torch_dtype=torch.float32,
        layerwise=True,
    )

    qpc_path = qeff_model.compile(
        batch_size=1,
        prefill_seq_len=32,
        ctx_len=4096,
        num_cores=16,
        num_devices=1,
        height=354,
        width=536,
        mxfp6_matmul=False,
        aic_enable_depth_first=True,
        skip_vision=True,
        split_retained_state_io=True,
        use_onnx_subfunctions=True,
        prefill_only=True,
        mos=1,
        layerwise=True,
        layerwise_window_size=1,
    )
    print(f"Final QPC path: {qpc_path}")


if __name__ == "__main__":
    main()
