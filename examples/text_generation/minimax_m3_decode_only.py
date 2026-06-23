# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse

import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "MiniMaxAI/MiniMax-M3"


def main():
    parser = argparse.ArgumentParser(description="MiniMax-M3 VLM decode-only (PL=1) with API layerwise compile.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--num-devices", type=int, default=16)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--generation-len", type=int, default=32)
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--layerwise-window-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--skip-generate", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_id)
    if args.num_layers is not None:
        config.text_config.num_hidden_layers = args.num_layers

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        kv_offload=True,
        dtype=torch.float16,
        config=config,
        layerwise=True,
    )

    qpc_paths = qeff_model.compile(
        batch_size=1,
        prefill_seq_len=1,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        use_onnx_subfunctions=True,
        skip_vision=True,
        layerwise=True,
        layerwise_window_size=args.layerwise_window_size,
    )
    print(f"QPC paths: {qpc_paths}")
    if isinstance(qpc_paths, dict) and "lang_decode_qpc_path" in qpc_paths:
        qeff_model.lang_model.qpc_path = qpc_paths["lang_decode_qpc_path"]

    if args.skip_generate:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": args.prompt}],
            }
        ]
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs["vision_embeds"] = torch.zeros((1, 4, config.text_config.hidden_size), dtype=torch.float16)
    output = qeff_model.generate(inputs=inputs, generation_len=args.generation_len)

    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))


if __name__ == "__main__":
    main()
