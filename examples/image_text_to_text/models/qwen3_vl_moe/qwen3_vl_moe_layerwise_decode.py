# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Layer-wise ONNX export + compile (decode) for qwen3_vl_moe (dual-QPC).

Same layer-wise language flow as the prefill script, but compiles the decode
specialization. The language decoder is exported one window of layers at a time
and merged into a single ONNX graph; the vision encoder is exported once.
"""

import torch
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
PREFILL_SEQ_LEN = 32
CTX_LEN = 4096
BATCH_SIZE = 1
NUM_CORES = 16
NUM_DEVICES = 1
HEIGHT = 354
WIDTH = 536
TEXT_WINDOW_SIZE = 1

# Optional: export only the first N (>1) text layers for quick validation.
# TOTAL_TEXT_LAYERS = 2
TOTAL_TEXT_LAYERS = None


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

    compile_kwargs = dict(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=NUM_CORES,
        num_devices=NUM_DEVICES,
        height=HEIGHT,
        width=WIDTH,
        mxfp6_matmul=True,
        aic_enable_depth_first=True,
        split_retained_state_io=True,
        use_onnx_subfunctions=True,
        mos=1,
        layerwise_window_size=TEXT_WINDOW_SIZE,
    )
    if TOTAL_TEXT_LAYERS is not None:
        compile_kwargs["total_layers"] = TOTAL_TEXT_LAYERS

    qpc_path = qeff_model.compile(**compile_kwargs)
    print(f"Layer-wise decode export + compile completed. QPC paths: {qpc_path}")


if __name__ == "__main__":
    main()
