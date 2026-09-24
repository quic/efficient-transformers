# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Decode-only Qwen3-VL-MoE CB/subfunction regression reproducer.

This reproducer compiles only the four-layer language decoder with batch-fold
attention and runs decode from synthetic KV state. It does not compile or run
vision or prefill. Run it once with ``--no-subfunctions`` and once with the
default subfunctions enabled to compare ONNX signatures and decode throughput.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"
BS = 256
CTX_LEN = 10240
NUM_LAYERS = 4
NUM_DEVICES = 16
NUM_CORES = 16
NUM_KV_BLOCKS = 4
PACKED_CHUNK_SIZE = 128
CORES_PER_EXPERT = 2
VISION_SIZE = 187
NUM_FEATURE_LAYERS = 1
WARMUP_STEPS = 2
MEASURE_STEPS = 10


def _parse_device_ids(value: str) -> list[int]:
    return [int(device_id) for device_id in value.strip("[]").split(",") if device_id]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--bs", type=int, default=BS)
    parser.add_argument("--ctx-len", type=int, default=CTX_LEN)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--num-devices", type=int, default=NUM_DEVICES)
    parser.add_argument("--cores-per-expert", type=int, default=CORES_PER_EXPERT)
    parser.add_argument("--num-kv-blocks", type=int, default=NUM_KV_BLOCKS)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measure-steps", type=int, default=MEASURE_STEPS)
    parser.add_argument(
        "--device-ids",
        type=_parse_device_ids,
        default=None,
        help="QAIC device IDs, e.g. '[0,1,...,15]'; use '[16,...,31]' as fallback.",
    )
    parser.add_argument("--compile-dir", type=Path, default=None)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--no-subfunctions", action="store_true")
    return parser.parse_args()


def _decode_qaic_config(ctx_len: int, num_kv_blocks: int, cores_per_expert: int) -> dict:
    return {
        "blocking_mode": "kv_batch_fold",
        "num_kv_blocks": num_kv_blocks,
        "ctx_len": ctx_len,
        "moe_config": {
            "flavour": "expert_parallel",
            "tree_reduce": False,
            "cores_per_expert": cores_per_expert,
            "expert_parallel_chunk_size": PACKED_CHUNK_SIZE,
        },
    }


def _build_inputs(session: QAICInferenceSession, batch_size: int, num_layers: int) -> dict[str, np.ndarray]:
    inputs = {
        "input_ids": np.zeros((batch_size, 1), dtype=np.int64),
        "position_ids": np.zeros((4, batch_size, 1), dtype=np.int64),
        "vision_embeds": np.zeros((batch_size, VISION_SIZE, 4096), dtype=np.float16),
        "deepstack_features": np.zeros((NUM_FEATURE_LAYERS, batch_size, VISION_SIZE, 4096), dtype=np.float16),
        "image_idx": np.zeros((1, 1), dtype=np.int64),
        "batch_index": np.arange(batch_size, dtype=np.int64).reshape(batch_size, 1),
    }
    for layer_idx in range(num_layers):
        inputs[f"past_key.{layer_idx}"] = np.zeros(
            session.kv_cache_info[2 * layer_idx][0], dtype=session.kv_cache_info[2 * layer_idx][1]
        )
        inputs[f"past_value.{layer_idx}"] = np.zeros(
            session.kv_cache_info[2 * layer_idx + 1][0], dtype=session.kv_cache_info[2 * layer_idx + 1][1]
        )
    return inputs


def _print_decoder_function_signatures(onnx_path: str) -> None:
    model = onnx.load(onnx_path, load_external_data=False)
    decoder_functions = [function for function in model.functions if "DecoderLayer" in function.name]
    for function in decoder_functions:
        print(f"decoder function: {function.name} inputs={len(function.input)} outputs={len(function.output)}")


def main() -> None:
    args = _parse_args()
    if args.device_ids is None:
        args.device_ids = list(range(args.num_devices))
    if len(args.device_ids) != args.num_devices:
        raise ValueError(f"Expected {args.num_devices} device IDs, got {len(args.device_ids)}")

    config = AutoConfig.from_pretrained(args.model_id)
    config.dtype = "float16"
    config.torch_dtype = torch.float16
    config.text_config.num_hidden_layers = args.num_layers
    config.vision_config.depth = 9
    config.vision_config.deepstack_visual_indexes = [8]

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        kv_offload=True,
        continuous_batching=True,
        config=config,
        dtype=torch.float16,
        layerwise=False,
    )
    compile_kwargs = {
        "batch_size": args.bs,
        "full_batch_size": args.bs,
        "kv_cache_batch_size": args.bs,
        "prefill_seq_len": 1,
        "ctx_len": args.ctx_len,
        "height": 354,
        "width": 536,
        "num_cores": NUM_CORES,
        "num_devices": args.num_devices,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "split_model_io": True,
        "user_tiled": True,
        "prefill_only": False,
        "skip_vision": True,
        "use_onnx_subfunctions": not args.no_subfunctions,
        "layerwise": False,
        "offload_pt_weights": False,
        "qaic_config": _decode_qaic_config(args.ctx_len, args.num_kv_blocks, args.cores_per_expert),
    }
    if args.compile_dir is not None:
        compile_kwargs["compile_dir"] = str(args.compile_dir)

    print(f"compile config: {compile_kwargs}")
    compile_result = qeff_model.compile(**compile_kwargs)
    qpc_path = compile_result["lang_decode_qpc_path"]
    print(f"decode QPC: {qpc_path}")
    print(f"decode ONNX: {qeff_model.lang_model.onnx_path}")
    _print_decoder_function_signatures(qeff_model.lang_model.onnx_path)

    if args.compile_only:
        return

    session = QAICInferenceSession(qpc_path, device_ids=args.device_ids)
    decode_inputs = _build_inputs(session, args.bs, args.num_layers)
    for _ in range(args.warmup_steps):
        session.run(decode_inputs)

    start = time.perf_counter()
    for _ in range(args.measure_steps):
        outputs = session.run(decode_inputs)
        decode_inputs["input_ids"] = np.argmax(outputs["logits"], axis=-1).astype(np.int64)
        decode_inputs["position_ids"] = decode_inputs["position_ids"] + 1
        for layer_idx in range(args.num_layers):
            decode_inputs[f"past_key.{layer_idx}"] = outputs[f"past_key.{layer_idx}_RetainedState"]
            decode_inputs[f"past_value.{layer_idx}"] = outputs[f"past_value.{layer_idx}_RetainedState"]
    elapsed = time.perf_counter() - start
    steps_per_second = args.measure_steps / elapsed
    print(f"decode steps/s={steps_per_second:.4f}")
    print(f"aggregate tokens/s={args.bs * steps_per_second:.4f}")


if __name__ == "__main__":
    main()
