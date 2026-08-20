# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import copy
import importlib.util
from io import BytesIO
from pathlib import Path
from time import perf_counter

import numpy as np
import requests
import torch
from PIL import Image

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

# By default, this script loads a compact Kimi-K2.5 subset: 2 vision layers,
# 2 text layers, and only the first 4 routed experts.
LOAD_KIMI_UTILS_PATH = Path(__file__).resolve().parents[4] / "tests" / "utils" / "load_kimi_utils.py"
_load_kimi_spec = importlib.util.spec_from_file_location("load_kimi_utils", LOAD_KIMI_UTILS_PATH)
if _load_kimi_spec is None or _load_kimi_spec.loader is None:
    raise ImportError(f"Unable to load Kimi helpers from {LOAD_KIMI_UTILS_PATH}")
load_kimi_utils = importlib.util.module_from_spec(_load_kimi_spec)
_load_kimi_spec.loader.exec_module(load_kimi_utils)

LOADED_EXPERT_IDS = load_kimi_utils.LOADED_EXPERT_IDS
NUM_EXPERTS_PER_TOKEN = load_kimi_utils.NUM_EXPERTS_PER_TOKEN
NUM_TEXT_LAYERS = load_kimi_utils.NUM_TEXT_LAYERS
NUM_VISION_LAYERS = load_kimi_utils.NUM_VISION_LAYERS
load_kimi_k25_class = load_kimi_utils.load_kimi_k25_class
load_kimi_k25_layer_subset_model = load_kimi_utils.load_kimi_k25_layer_subset_model
prepare_config = load_kimi_utils.prepare_config
parse_expert_ids = load_kimi_utils.parse_expert_ids

PREFILL_SEQ_LEN = 512
CTX_LEN = 2048
BS = 1
GENERATION_LEN = 10
qaic_config = {"mla_absorption": {"cache_compressed": True, "absorption": False, "online": False}}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Kimi K2.5 vision disaggregated vision -> prefill -> decode flow.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Load the full model. By default, the script loads a small layer subset for faster startup.",
    )
    parser.add_argument("--num-vision-layers", type=int, default=NUM_VISION_LAYERS)
    parser.add_argument("--num-text-layers", type=int, default=NUM_TEXT_LAYERS)
    parser.add_argument("--expert-ids", type=parse_expert_ids, default=LOADED_EXPERT_IDS)
    parser.add_argument("--num-experts-per-token", type=int, default=NUM_EXPERTS_PER_TOKEN)
    parser.add_argument(
        "--image-url",
        type=str,
        default="https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Image height in pixels for Kimi-K2.5 vision compile. Defaults to the loaded image height.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Image width in pixels for Kimi-K2.5 vision compile. Defaults to the loaded image width.",
    )
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--prefill-seq-len", type=int, default=PREFILL_SEQ_LEN)
    parser.add_argument("--ctx-len", type=int, default=CTX_LEN)
    parser.add_argument("--generation-len", type=int, default=GENERATION_LEN)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--vision-num-devices", type=int, default=1)
    parser.add_argument("--lang-num-devices", type=int, default=1)
    parser.add_argument("--mxfp6-matmul", action="store_true")
    parser.add_argument("--mxint8-kv-cache", action="store_true")
    args = parser.parse_args()
    if (args.image_height is None) != (args.image_width is None):
        parser.error("--image-height and --image-width must be provided together.")
    return args


def _clone_inputs(inputs):
    return {key: (value.clone() if torch.is_tensor(value) else copy.deepcopy(value)) for key, value in inputs.items()}


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _session_input_names(session: QAICInferenceSession) -> set[str]:
    input_names = set(session.input_names)
    input_names.update(name.rsplit("/", 1)[-1] for name in session.input_names)
    return input_names


def _cast_for_session(session: QAICInferenceSession, name: str, value: np.ndarray) -> np.ndarray:
    binding_index = session.binding_index_map.get(name)
    if binding_index is None:
        return value
    dtype = session.aic_to_np_dtype_mapping[session.bindings[binding_index].type]
    return value.astype(dtype, copy=False)


def _filter_session_inputs(session: QAICInferenceSession, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_names = _session_input_names(session)
    return {name: _cast_for_session(session, name, value) for name, value in inputs.items() if name in input_names}


def _resolve_qpc_path(qpc_paths, key: str):
    if isinstance(qpc_paths, dict):
        qpc_path = qpc_paths.get(key)
        if qpc_path is None:
            raise KeyError(f"Missing {key!r} in compile output keys: {list(qpc_paths.keys())}")
        return qpc_path
    return qpc_paths


def _update_retained_states(target_inputs: dict[str, np.ndarray], source_outputs: dict[str, np.ndarray]):
    for output_name, value in source_outputs.items():
        output_basename = output_name.rsplit("/", 1)[-1]
        if output_basename.endswith("_RetainedState"):
            target_inputs[output_basename.removesuffix("_RetainedState")] = value


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64).reshape(BS, 1)


def _compile_disagg_qpcs(qeff_model: QEFFAutoModelForImageTextToText, args, image: Image.Image):
    qaic_config = {"mla_absorption": {"cache_compressed": True, "absorption": False, "online": False}}
    common_compile_kwargs = {
        "qaic_config": qaic_config,
        "batch_size": BS,
        "ctx_len": args.ctx_len,
        "image_height": image.height,
        "image_width": image.width,
        "num_cores": args.num_cores,
        "mxfp6_matmul": args.mxfp6_matmul,
        "mxint8_kv_cache": args.mxint8_kv_cache,
        "split_model_io": True,
        "mos": 1,
        "aic_enable_depth_first": True,
        "use_onnx_subfunctions": True,
        "layerwise": False,
    }

    print("Compiling vision QPC...")
    vision_qpc_path = qeff_model.compile(
        prefill_seq_len=args.prefill_seq_len,
        skip_vision=False,
        skip_lang=True,
        num_devices=args.vision_num_devices,
        **common_compile_kwargs,
    )

    print("Compiling prefill QPC...")
    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=args.prefill_seq_len,
        prefill_only=True,
        skip_vision=True,
        skip_lang=False,
        num_devices=args.lang_num_devices,
        **common_compile_kwargs,
    )

    print("Compiling decode QPC...")
    decode_qpc_path = qeff_model.compile(
        prefill_seq_len=1,
        prefill_only=False,
        skip_vision=True,
        skip_lang=False,
        num_devices=args.lang_num_devices,
        **common_compile_kwargs,
    )

    return vision_qpc_path, prefill_qpc_path, decode_qpc_path


def _run_disagg_generation(
    inputs: dict[str, torch.Tensor],
    vision_session: QAICInferenceSession,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
    *,
    prefill_seq_len: int,
    generation_len: int,
) -> np.ndarray:
    inputs = {name: _numpy(value) for name, value in _clone_inputs(inputs).items()}
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -prefill_seq_len)
    padded_len = num_chunks * prefill_seq_len

    inputs["input_ids"] = np.pad(
        inputs["input_ids"],
        ((0, 0), (0, padded_len - input_ids_length)),
        constant_values=1,
    )
    inputs["attention_mask"] = np.pad(
        inputs["attention_mask"],
        ((0, 0), (0, padded_len - input_ids_length)),
        constant_values=0,
    )

    grid_thws = inputs.pop("grid_thws").astype(np.int64)
    h = int(grid_thws[0, 1])
    w = int(grid_thws[0, 2])
    vision_inputs = {
        "pixel_values": inputs["pixel_values"],
        "h_shape": np.ones((h,), dtype=np.int64),
        "w_shape": np.ones((w,), dtype=np.int64),
    }

    vision_start = perf_counter()
    print("Running vision QPC...")
    vision_outputs = vision_session.run(_filter_session_inputs(vision_session, vision_inputs))
    vision_session.deactivate()
    vision_time = perf_counter() - vision_start

    vision_embeds = vision_outputs.get("vision_embeds")
    if vision_embeds is None:
        raise RuntimeError(f"Vision QPC did not return vision_embeds. Outputs: {vision_outputs.keys()}")

    lang_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "position_ids": np.where(inputs["attention_mask"] > 0, np.arange(padded_len), -1).astype(np.int64),
        "vision_embeds": vision_embeds,
        "image_idx": np.zeros((BS, 1), dtype=np.int64),
    }

    prefill_start = perf_counter()
    print("Running prefill QPC...")
    prefill_session.set_buffers(vision_outputs)
    chunk_inputs = lang_inputs.copy()
    prefill_outputs = None
    for chunk_idx in range(num_chunks):
        start = chunk_idx * prefill_seq_len
        end = (chunk_idx + 1) * prefill_seq_len
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, start:end]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][:, start:end]
        prefill_outputs = prefill_session.run(_filter_session_inputs(prefill_session, chunk_inputs))
        _update_retained_states(chunk_inputs, prefill_outputs)
        if "image_idx_output" in prefill_outputs:
            chunk_inputs["image_idx"] = prefill_outputs["image_idx_output"].astype(np.int64)

    prefill_session.deactivate()
    if prefill_outputs is None:
        raise RuntimeError("QAIC prefill did not execute.")
    prefill_time = perf_counter() - prefill_start + vision_time
    print(f"Prefill time, including vision: {prefill_time:.2f} secs")

    generated_ids = [_get_next_token_ids(prefill_outputs["logits"])]
    decode_inputs = {
        "input_ids": generated_ids[-1],
        "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True).astype(np.int64) + 1,
        "vision_embeds": chunk_inputs.get("vision_embeds", vision_embeds),
        "image_idx": chunk_inputs.get("image_idx", np.zeros((BS, 1), dtype=np.int64)),
    }
    _update_retained_states(decode_inputs, prefill_outputs)

    print("Running decode QPC...")
    decode_start = perf_counter()
    for _ in range(1, generation_len):
        decode_outputs = decode_session.run(_filter_session_inputs(decode_session, decode_inputs))
        generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))
        decode_inputs["input_ids"] = generated_ids[-1]
        decode_inputs["position_ids"] = decode_inputs["position_ids"] + 1
        if "image_idx_output" in decode_outputs:
            decode_inputs["image_idx"] = decode_outputs["image_idx_output"].astype(np.int64)
        _update_retained_states(decode_inputs, decode_outputs)

    decode_time = perf_counter() - decode_start
    if generation_len > 1:
        print(f"Decode tok/sec: {(generation_len - 1) / decode_time:.2f}")
    return np.concatenate(generated_ids, axis=1)


def _load_model(args):
    if args.full_model:
        config = prepare_config(args.model_path)
        kimi_cls = load_kimi_k25_class(args.model_path)
        model_kwargs = {
            "config": config,
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "torch_dtype": torch.float32,
        }
        model, tokenizer, processor = kimi_cls.from_pretrained(str(args.model_path), **model_kwargs)
    elif args.num_vision_layers is not None and args.num_text_layers is not None:
        model, tokenizer, processor = load_kimi_k25_layer_subset_model(
            model_path=args.model_path,
            num_vision_layers=args.num_vision_layers,
            num_text_layers=args.num_text_layers,
            loaded_expert_ids=args.expert_ids,
            num_experts_per_tok=args.num_experts_per_token,
            dtype=torch.float32,
        )
        print(
            "Loaded layer subset: "
            f"vision={model.config.vision_config.vt_num_hidden_layers}, "
            f"text={model.config.text_config.num_hidden_layers}, "
            f"experts={model.config.text_config.n_routed_experts}"
        )
    else:
        raise ValueError("Pass both --num-vision-layers and --num-text-layers to load a layer subset.")

    model.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
    return model.eval().to("cpu"), tokenizer, processor


def _prepare_inputs(processor, args):
    image = Image.open(BytesIO(requests.get(args.image_url, timeout=30).content)).convert("RGB")
    if args.image_height is not None:
        image = image.resize((args.image_width, args.image_height))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]
    inputs = processor(
        messages=messages,
        add_generation_prompt=True,
        tokenize=False,
        return_tensors="pt",
    )
    inputs = {name: (value.to("cpu") if torch.is_tensor(value) else value) for name, value in inputs.items()}
    return image, inputs


def main():
    args = parse_args()
    model, tokenizer, processor = _load_model(args)
    qeff_model = QEFFAutoModelForImageTextToText(
        model,
        kv_offload=True,
        config=model.config,
        torch_dtype=torch.float32,
        qaic_config=qaic_config,
        layerwise=False,
    )

    image, inputs = _prepare_inputs(processor, args)
    inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)

    vision_qpc_path, prefill_qpc_path, decode_qpc_path = _compile_disagg_qpcs(qeff_model, args, image)
    print(f"Vision QPC path: {vision_qpc_path}")
    print(f"Prefill QPC path: {prefill_qpc_path}")
    print(f"Decode QPC path: {decode_qpc_path}")

    sessions = []
    try:
        vision_session = QAICInferenceSession(_resolve_qpc_path(vision_qpc_path, "vision_qpc_path"))
        prefill_session = QAICInferenceSession(_resolve_qpc_path(prefill_qpc_path, "lang_prefill_qpc_path"))
        decode_session = QAICInferenceSession(_resolve_qpc_path(decode_qpc_path, "lang_decode_qpc_path"))
        sessions.extend([vision_session, prefill_session, decode_session])

        generated_ids = _run_disagg_generation(
            inputs,
            vision_session,
            prefill_session,
            decode_session,
            prefill_seq_len=args.prefill_seq_len,
            generation_len=args.generation_len,
        )
    finally:
        for session in sessions:
            session.deactivate()

    print(generated_ids)
    print(tokenizer.batch_decode(torch.as_tensor(generated_ids), skip_special_tokens=True))


if __name__ == "__main__":
    main()
