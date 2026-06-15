# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Language-only Qwen3.5-MoE decode parity example.

This uses QEFFAutoModelForImageTextToText with skip_vision=True so only the
language decoder is exported and compiled. Defaults are the validated 397B/4L
FP16/no-MX, non-layerwise, PL=1 path and automatic gemma-like NPI generation.
"""

import argparse
import gc
import os

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from QEfficient import QEFFAutoModelForImageTextToText

# MODEL_ID = "tiny-random/qwen3.6-moe"
MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
TORCH_DTYPE = torch.float16
RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=os.getenv("MODEL_ID", MODEL_ID))
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--prefill-seq-len", type=int, default=1)
    parser.add_argument("--ctx-len", type=int, default=4)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--generation-len", type=int, default=3)
    parser.add_argument("--layerwise", action="store_true", help="Enable provisional layerwise compile path.")
    parser.add_argument(
        "--aic-npi-mode",
        choices=["off", "gemma_like"],
        default="gemma_like",
        help="For gemma_like, compile() auto-generates node_precision_info from the exported language ONNX.",
    )
    parser.add_argument("--compile-only", action="store_true", help="Compile and exit before AIC generation.")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HF PyTorch token generation.")
    parser.add_argument("--skip-qeff-pt", action="store_true", help="Skip transformed QEff PyTorch token generation.")
    return parser.parse_args()


def _apply_num_hidden_layers(config, num_hidden_layers):
    if num_hidden_layers is None:
        return
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = num_hidden_layers
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = list(config.layer_types)[:num_hidden_layers]
    if hasattr(config, "text_config"):
        if hasattr(config.text_config, "num_hidden_layers"):
            config.text_config.num_hidden_layers = num_hidden_layers
        if getattr(config.text_config, "layer_types", None) is not None:
            config.text_config.layer_types = list(config.text_config.layer_types)[:num_hidden_layers]


def _text_inputs(processor, prompt):
    inputs = processor(text=[prompt], return_tensors="pt")
    inputs.pop("token_type_ids", None)
    return inputs


def _generated_tail(output_ids, input_len):
    if isinstance(output_ids, torch.Tensor):
        output_ids = output_ids.detach().cpu().numpy()
    return np.asarray(output_ids)[:, input_len:]


@torch.no_grad()
def _run_pt_generate(label, model, processor, tokenizer, prompt, generation_len):
    inputs = _text_inputs(processor, prompt)
    input_len = inputs["input_ids"].shape[-1]
    output_ids = model.generate(**inputs, max_new_tokens=generation_len, do_sample=False)
    generated_ids = _generated_tail(output_ids, input_len)
    print(f"{label} generated ids: {generated_ids.tolist()}")
    print(f"{label} decoded: {tokenizer.batch_decode(generated_ids)}")
    return generated_ids


@torch.no_grad()
def _run_qeff_pt_generate(qeff_model, processor, tokenizer, prompt, prefill_seq_len, ctx_len, generation_len):
    inputs = _text_inputs(processor, prompt)
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs,
        prefill_seq_len=prefill_seq_len,
        batch_size=1,
    )
    inputs.pop("attention_mask", None)

    dummy_lang_inputs = qeff_model.model.get_dummy_inputs(kv_offload=True)["lang"]
    inputs["vision_embeds"] = dummy_lang_inputs["vision_embeds"]
    inputs["image_idx"] = dummy_lang_inputs["image_idx"]
    past_key_values = dummy_lang_inputs["past_key_values"]
    layer_types = getattr(qeff_model.model.config.text_config, "layer_types", [])
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type != "full_attention" or layer_idx >= len(past_key_values):
            continue
        layer_state = past_key_values[layer_idx]
        if len(layer_state) != 2:
            continue
        past_key_values[layer_idx] = [
            torch.zeros((*state.shape[:2], ctx_len, *state.shape[3:]), dtype=state.dtype, device=state.device)
            for state in layer_state
        ]
    inputs["past_key_values"] = past_key_values

    generated = []
    outputs = qeff_model.lang_model.model(**inputs)
    next_token = outputs[0].argmax(-1)
    if next_token.ndim == 2 and next_token.shape[1] > 1:
        next_token = next_token[:, -1:]
    generated.append(next_token.detach().cpu().numpy())

    decode_inputs = {
        "input_ids": next_token,
        "position_ids": inputs["position_ids"].amax(dim=-1, keepdim=True) + 1,
        "vision_embeds": outputs[1],
        "image_idx": outputs[2],
        "past_key_values": outputs[3],
    }

    for _ in range(1, generation_len):
        outputs = qeff_model.lang_model.model(**decode_inputs)
        next_token = outputs[0].argmax(-1)
        if next_token.ndim == 2 and next_token.shape[1] > 1:
            next_token = next_token[:, -1:]
        generated.append(next_token.detach().cpu().numpy())
        decode_inputs = {
            "input_ids": next_token,
            "position_ids": decode_inputs["position_ids"] + 1,
            "vision_embeds": outputs[1],
            "image_idx": outputs[2],
            "past_key_values": outputs[3],
        }

    generated_ids = np.concatenate(generated, axis=1)
    print(f"QEff PT generated ids: {generated_ids.tolist()}")
    print(f"QEff PT decoded: {tokenizer.batch_decode(generated_ids)}")
    return generated_ids


def _print_match(label, lhs, rhs):
    if lhs is None or rhs is None:
        return
    print(f"{label}: {bool(np.array_equal(np.asarray(lhs), np.asarray(rhs)))}")


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_CACHE", "/home/huggingface_hub")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/home/huggingface_hub")
    os.environ.setdefault("TMPDIR", os.path.abspath("build/qwen3_5_moe_compile_tmp"))
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

    # Required for the validated 397B/4L/4-device FP16/no-MX AIC parity path.
    os.environ["QEFF_QWEN35_MOE_TANH_SHARED_GATE"] = "1"
    os.environ["QEFF_QWEN35_MOE_TANH_LINEAR_BETA"] = "1"
    os.environ["QEFF_QWEN35_MOE_TANH_ATTN_GATE"] = "1"
    os.environ["QEFF_QWEN35_MOE_FORCE_RECURRENT_DECODE"] = "1"

    torch.manual_seed(RANDOM_SEED)

    config = AutoConfig.from_pretrained(args.model_id)
    config.torch_dtype = TORCH_DTYPE
    _apply_num_hidden_layers(config, args.num_hidden_layers)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)

    hf_ids = None
    if not args.skip_hf:
        hf_model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            attn_implementation="eager",
            config=config,
            dtype=TORCH_DTYPE,
        )
        hf_ids = _run_pt_generate("HF", hf_model, processor, tokenizer, args.prompt, args.generation_len)
        del hf_model
        gc.collect()

    torch.manual_seed(RANDOM_SEED)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        dtype=TORCH_DTYPE,
        layerwise=args.layerwise,
    )

    qeff_pt_ids = None
    if not args.skip_qeff_pt:
        qeff_pt_ids = _run_qeff_pt_generate(
            qeff_model,
            processor,
            tokenizer,
            args.prompt,
            args.prefill_seq_len,
            args.ctx_len,
            args.generation_len,
        )
        _print_match("HF == QEff PT", hf_ids, qeff_pt_ids)

    compile_kwargs = {}
    if args.aic_npi_mode == "gemma_like":
        compile_kwargs["node_precision_info"] = True

    qpc_paths = qeff_model.compile(
        batch_size=1,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        height=354,
        width=536,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=False,
        skip_vision=True,
        split_retained_state_io=True,
        use_onnx_subfunctions=True,
        mos=1,
        layerwise=args.layerwise,
        layerwise_window_size=1,
        **compile_kwargs,
    )
    print(f"Final QPC path(s): {qpc_paths}")

    if args.compile_only:
        return

    inputs = _text_inputs(processor, args.prompt)
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs,
        prefill_seq_len=args.prefill_seq_len,
        batch_size=1,
    )
    output = qeff_model.generate(
        inputs=inputs,
        generation_len=args.generation_len,
        device_ids=list(range(args.num_devices)),
    )
    aic_ids = np.asarray(output.generated_ids)[:, : args.generation_len]
    print(f"AIC generated ids: {aic_ids.tolist()}")
    print(f"AIC decoded: {tokenizer.batch_decode(aic_ids)}")
    _print_match("HF == AIC", hf_ids, aic_ids)
    _print_match("QEff PT == AIC", qeff_pt_ids, aic_ids)
    print(output)


if __name__ == "__main__":
    main()
