# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Weight-free export and inference for Causal LM models on Cloud AI 100.

Weight-free export builds the ONNX graph on a meta-device model (no weights
in RAM). The QAIC compiler then loads the weights directly from the original
safetensors checkpoint at compile time, reducing peak memory during export.

Requires PyTorch >= 2.13. Install dependencies before running:
    pip install -r examples/dynamo/causal_lm/requirements.txt
"""

import argparse

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(
        description="Weight-free export and inference for Causal LM models on Cloud AI 100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--num-hidden-layers", type=int, default=-1, help="Override number of hidden layers")
    parser.add_argument("--prompt", type=str, default="My name is", help="Input prompt for generation")
    parser.add_argument("--prefill-seq-len", type=int, default=128, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context (KV-cache) length")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of new tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of AI 100 cores")
    parser.add_argument("--aic-hw-version", type=str, default="ai100", help="AIC hardware version")
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=None,
        help="Device IDs (comma-separated), e.g. [0,1]",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    config = AutoConfig.from_pretrained(args.model_name)
    config.dtype = torch.float16
    if args.num_hidden_layers > 0:
        config.num_hidden_layers = args.num_hidden_layers

    # Build model on meta device — no weights loaded into RAM.
    # The QAIC compiler loads weights from the checkpoint at compile time.
    with init_empty_weights():
        hf_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model = QEFFAutoModelForCausalLM(hf_model, pretrained_model_name_or_path=args.model_name)

    # Export (weight-free dynamo path) + compile to QPC
    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        aic_hw_version=args.aic_hw_version,
        num_devices=(2 if args.device_group is None else len(args.device_group)),
        use_weight_free_export=True,
        use_onnx_subfunctions=True,
        mxfp6_matmul=False,
    )
    print(f"Model compiled to: {qpc_path}")

    # Run inference on device
    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
        device_id=args.device_group,
        generation_len=args.generation_len,
    )

    print(f"\nPrompt   : {args.prompt}")
    print(f"Generated: {exec_info.generated_texts[0]}")


if __name__ == "__main__":
    main()
