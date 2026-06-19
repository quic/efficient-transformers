# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Layerwise prefill compile example for Qwen3-VL-MoE.

The orchestration loop that previously lived in this script has been moved
behind the ``layerwise=True`` flag on ``.compile()`` / ``.export()``.

Note: ``layerwise=True`` is a provisional API and is scheduled for deprecation
once first-class multi-window export lands. Supported model types:
``qwen3_vl_moe``, ``qwen3_5_moe``, ``qwen3_moe``.
"""

import torch
import transformers
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

# MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# MODEL_ID = "tiny-random/qwen3-vl-moe"


def main():
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.dtype = "float16"
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        dtype=torch.float16,
        layerwise=True,
    )
    batch_size = 1
    qpc_path = qeff_model.compile(
        batch_size=1,
        prefill_seq_len=1,
        ctx_len=4096,
        num_cores=16,
        num_devices=4,
        height=354,
        width=536,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=True,
        skip_vision=True,
        split_retained_state_io=True,
        use_onnx_subfunctions=True,
        mos=1,
        layerwise=True,
        layerwise_window_size=1,
    )
    print(f"Final QPC path: {qpc_path}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tell me about yourself."},
            ],
        },
    ]

    messages = [messages] * batch_size

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=128, batch_size=batch_size)
    output = qeff_model.generate(inputs=inputs, generation_len=100)
    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)


if __name__ == "__main__":
    main()
