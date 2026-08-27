# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Qwen3.5-MoE CausalLM example that avoids loading model weights.

By default this script builds a tiny Qwen3.5-MoE text config locally, initializes
random weights with ``AutoModelForCausalLM.from_config()``, applies QEfficient
transforms, and runs a CPU forward smoke test. Use ``--use-hf-config`` when you
want to initialize random weights from a real Hugging Face config instead.
"""

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from QEfficient import QEFFAutoModelForCausalLM


def parse_device_group(device_group: str | None) -> list[int] | None:
    if device_group is None:
        return None
    return [int(device_id) for device_id in device_group.strip("[]").split(",") if device_id]


def tiny_qwen3_5_moe_text_config(prefill_seq_len: int) -> Qwen3_5MoeTextConfig:
    max_position_embeddings = max(prefill_seq_len, 32)
    return Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=max_position_embeddings,
        layer_types=["full_attention", "full_attention"],
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
        rope_parameters={"mrope_section": [1, 1, 1]},
        dtype="float32",
    )


def shrink_text_config_for_smoke(config, args: argparse.Namespace):
    if args.preserve_hf_size:
        return config

    config.vocab_size = args.smoke_vocab_size
    config.hidden_size = args.smoke_hidden_size
    config.num_attention_heads = args.smoke_num_attention_heads
    config.num_key_value_heads = args.smoke_num_key_value_heads
    config.head_dim = args.smoke_head_dim
    config.max_position_embeddings = max(args.prefill_seq_len, 32)
    config.moe_intermediate_size = args.smoke_moe_intermediate_size
    config.shared_expert_intermediate_size = args.smoke_shared_expert_intermediate_size
    config.num_experts = args.smoke_num_experts
    config.num_experts_per_tok = min(args.smoke_num_experts_per_tok, config.num_experts)
    rope_parameters = dict(getattr(config, "rope_parameters", {}) or {})
    rope_parameters.setdefault("rope_type", "default")
    rope_parameters["partial_rotary_factor"] = 1.0
    rope_parameters["mrope_section"] = [1, 1, 1]
    config.rope_parameters = rope_parameters
    return config


def load_config(args: argparse.Namespace):
    if not args.use_hf_config:
        return tiny_qwen3_5_moe_text_config(args.prefill_seq_len)

    config = AutoConfig.from_pretrained(args.model_name)
    if hasattr(config, "text_config"):
        config = config.text_config
    config.num_hidden_layers = args.num_hidden_layers
    if hasattr(config, "layer_types"):
        config.layer_types = config.layer_types[: args.num_hidden_layers]
    config.dtype = "float32"
    return shrink_text_config_for_smoke(config, args)


def qwen3_5_position_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    text_position_ids = torch.arange(seq_len, dtype=torch.long).reshape(1, 1, seq_len)
    return text_position_ids.expand(4, batch_size, seq_len)


def run_forward_smoke(model, args: argparse.Namespace) -> None:
    input_ids = torch.ones((args.batch_size, args.prefill_seq_len), dtype=torch.long)
    position_ids = qwen3_5_position_ids(args.batch_size, args.prefill_seq_len)

    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, position_ids=position_ids, use_cache=False)

    print(f"Forward smoke logits shape: {tuple(outputs.logits.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-MoE CausalLM config-only example")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3.5-35B-A3B",
        help="Hugging Face model ID used only with --use-hf-config or --generate",
    )
    parser.add_argument("--use-hf-config", action="store_true", help="Load config from Hugging Face without weights")
    parser.add_argument("--prompt", type=str, default="Explain quantum computing", help="Input prompt")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--prefill-seq-len", type=int, default=1, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=4096, help="Context length")
    parser.add_argument("--generation-len", type=int, default=None, help="Number of tokens to generate")
    parser.add_argument("--num-hidden-layers", type=int, default=4, help="Layer count when using --use-hf-config")
    parser.add_argument(
        "--preserve-hf-size",
        action="store_true",
        help="Keep HF config tensor dimensions; this can allocate very large random models",
    )
    parser.add_argument("--smoke-vocab-size", type=int, default=64, help="Vocab size for --use-hf-config smoke mode")
    parser.add_argument("--smoke-hidden-size", type=int, default=16, help="Hidden size for --use-hf-config smoke mode")
    parser.add_argument(
        "--smoke-moe-intermediate-size", type=int, default=16, help="MoE intermediate size for smoke mode"
    )
    parser.add_argument(
        "--smoke-shared-expert-intermediate-size",
        type=int,
        default=16,
        help="Shared expert intermediate size for smoke mode",
    )
    parser.add_argument("--smoke-num-experts", type=int, default=2, help="Number of experts for smoke mode")
    parser.add_argument("--smoke-num-experts-per-tok", type=int, default=1, help="Experts per token for smoke mode")
    parser.add_argument("--smoke-num-attention-heads", type=int, default=2, help="Attention heads for smoke mode")
    parser.add_argument("--smoke-num-key-value-heads", type=int, default=1, help="KV heads for smoke mode")
    parser.add_argument("--smoke-head-dim", type=int, default=8, help="Attention head dimension for smoke mode")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument("--num-devices", type=int, default=4, help="Number of devices")
    parser.add_argument("--device-group", type=parse_device_group, default=None, help="Device IDs, e.g. [0,1,2,3]")
    parser.add_argument("--compile", action="store_true", help="Compile the random-weight model")
    parser.add_argument("--generate", action="store_true", help="Run generate after compile; requires tokenizer")
    args = parser.parse_args()

    config = load_config(args)
    hf_model = AutoModelForCausalLM.from_config(config)
    model = QEFFAutoModelForCausalLM(hf_model)
    model.hash_params["qwen3_5_moe_config_only_example"] = "v2"

    run_forward_smoke(model, args)

    if not args.compile:
        return

    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        aic_enable_depth_first=False,
        mos=1,
        user_tiled=True,
        split_model_io=True,
        use_onnx_subfunctions=True,
    )
    print(f"Model compiled to: {qpc_path}")

    if args.generate:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        exec_info = model.generate(
            tokenizer=tokenizer,
            prompts=[args.prompt] * args.batch_size,
            device_id=args.device_group,
            generation_len=args.generation_len,
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {exec_info.generated_texts[0]}")


if __name__ == "__main__":
    main()
