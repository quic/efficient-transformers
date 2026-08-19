# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Decode compile example for Qwen3.5-MoE."""

import torch
import transformers
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
TORCH_DTYPE = torch.float16
RANDOM_SEED = 42


def main():
    # Tiny random checkpoints have missing params initialized from RNG.
    # Keep the seed fixed so parity checks are stable.
    torch.manual_seed(RANDOM_SEED)

    config = AutoConfig.from_pretrained(MODEL_ID)
    config.torch_dtype = TORCH_DTYPE
    # config.vision_config.depth = 4
    # config.text_config.num_hidden_layers = 4
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        dtype=TORCH_DTYPE,
    )

    qpc_path = qeff_model.compile(
        batch_size=1,
        prefill_seq_len=1,
        ctx_len=4096,
        num_cores=16,
        num_devices=4,
        height=354,
        width=536,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=True,
        skip_vision=True,
        split_retained_state_io=True,
        use_onnx_subfunctions=True,
        mos=1,
    )
    print(f"Final QPC path: {qpc_path}")

    batch_size = 1
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
