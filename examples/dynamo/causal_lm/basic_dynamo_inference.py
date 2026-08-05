# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Dynamo-based export and inference for Causal LM models on Cloud AI 100.

Requires PyTorch >= 2.13. Install dependencies before running:
    pip install -r examples/dynamo/causal_lm/requirements.txt
"""

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


def load_qeff_model(model_name: str, num_hidden_layers: int, use_weight_free_export: bool, enable_proxy: bool):
    config = AutoConfig.from_pretrained(model_name)
    if num_hidden_layers > 0:
        config.num_hidden_layers = num_hidden_layers

    if not use_weight_free_export:
        return QEFFAutoModelForCausalLM.from_pretrained(model_name, config=config, enable_proxy=enable_proxy)

    config.dtype = torch.float16
    config.torch_dtype = torch.float16
    with torch.device("meta"):
        hf_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    return QEFFAutoModelForCausalLM(hf_model, pretrained_model_name_or_path=model_name, enable_proxy=enable_proxy)


def main():
    parser = argparse.ArgumentParser(
        description="Dynamo-based export and inference for Causal LM models on Cloud AI 100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--num-hidden-layers", type=int, default=-1, help="Override number of hidden layers")
    parser.add_argument("--prompt", type=str, default="My name is", help="Input prompt for generation")
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context (KV-cache) length")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of new tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of AI 100 cores")
    parser.add_argument("--aic-hw-version", type=str, default="ai100", help="AIC hardware version")
    parser.add_argument(
        "--use-weight-free-export",
        action="store_true",
        help="Build the model on meta tensors and load weights at compile time",
    )
    parser.add_argument(
        "--enable-proxy",
        action="store_true",
        help="Use proxy modules (QeffProxyEmbedding/QeffProxyLinear) in place of real weights during export",
    )
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=None,
        help="Device IDs (comma-separated), e.g. [0,1]",
    )
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = load_qeff_model(args.model_name, args.num_hidden_layers, args.use_weight_free_export, args.enable_proxy)

    # Export (via torch.export / dynamo) + compile to QPC
    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        aic_hw_version=args.aic_hw_version,
        num_devices=(2 if args.device_group is None else len(args.device_group)),
        dynamo=True,
        use_weight_free_export=args.use_weight_free_export,
        use_onnx_subfunctions=True,
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
