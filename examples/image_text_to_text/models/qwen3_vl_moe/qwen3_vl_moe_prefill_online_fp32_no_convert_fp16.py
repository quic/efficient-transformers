# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Compiler repro: blocked prefill_online attention exported in FP16 with explicit FP32 regions.

The model is loaded and exported in FP16. The online-softmax statistics and the output accumulator of
``prefill_online`` blocked attention are explicitly cast to FP32 in the exported ONNX. QEfficient normally
compiles FP16 models with ``-convert-to-fp16``, which makes the compiler rewrite the whole graph to FP16 and
discard those casts. By default this script skips that flag (``--convert-to-fp16`` restores it) so the
compiler has to honor the per-op precision in the ONNX.

Usage (4 layers, vision + language prefill, one image):
    HF_HUB_ENABLE_HF_TRANSFER=1 python qwen3_vl_moe_prefill_online_fp32_no_convert_fp16.py \
        --num-layers 4 --image-url https://picsum.photos/id/238/536/354 --dump-dir /path/to/dump
Use ``--no_blocking`` for the reference run without blocking.
"""

import argparse
import gc
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models import modeling_auto

ENCODE_CONFIG = {
    "prefill_seq_len": 10240,
    "ctx_len": 10240,
    "batch_size": 1,
    "full_batch_size": 1,
    "num_devices": 1,
    "num_cores": 16,
    "mxfp6_matmul": False,
    "mxint8_kv_cache": False,
    "aic_enable_depth_first": False,
    "mos": 1,
    "prefill_only": None,
    "height": 512,
    "width": 910,
    "use_onnx_subfunctions": True,
    "split_model_io": True,
    "mm_processor_kwargs": {
        "max_pixels": 16777216,
        "min_pixels": 4096,
    },
    "skip_lang": True,
    "skip_vision": False,
}

PREFILL_CONFIG = {
    "prefill_seq_len": 1024,
    "ctx_len": 10240,
    "batch_size": 1,
    "full_batch_size": 1,
    "num_devices": 1,
    "num_cores": 16,
    "mxfp6_matmul": True,
    "mxint8_kv_cache": True,
    "aic_enable_depth_first": False,
    "prefill_only": True,
    "vision_batch_size": 1,
    "height": 512,
    "width": 910,
    "split_model_io": True,
    "use_onnx_subfunctions": True,
    "enable_chunking": True,
    "retain_full_kv": True,
    "kv_cache_prefix": "vllmKvCache",
    "kv_cache_batch_size": 1,
    "layerwise": False,
    "user_tiled": True,
    "mdp_num_partitions": 1,
    "mdp_strategy": "intersection",
    "offload_pt_weights": True,
    "qaic_config": {
        "blocking_mode": "prefill_online",
        "num_kv_blocks": 4,
        "num_q_blocks": 8,
        # "n_rep_chunk": 4, #AssertionError: q_head_block_chunk (4) must divide NQH//num_cores (2). # for 30 B
        "n_rep_chunk": 2,
        "ctx_len": 10240,
        "skip_kv": True,
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": 256,
            "cores_per_expert": 1,
            "tree_reduce": False,
        },
    },
    "allow_mxint8_mdp_io": True,
    "mm_processor_kwargs": {
        "max_pixels": 16777216,
        "min_pixels": 4096,
    },
    "skip_vision": True,
    "vision_size": 1024,
}

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"  # "Qwen/Qwen3-VL-235B-A22B-Instruct"

DEFAULT_IMAGE_URLS = [
    "https://picsum.photos/id/238/536/354",
    "https://picsum.photos/id/237/536/354",
]
DEFAULT_PROMPT = "Describe the image in detail"
DEFAULT_NUM_LAYERS = 4

VISION_INPUT_KEYS = {
    "pixel_values",
    "image_masks",
    "image_input_idx",
    "image_grid_thw",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
}
VISION_FP16_KEYS = {"pixel_values", "image_masks"}
VISION_OUTPUT_KEYS = ("vision_embeds", "deepstack_features")


def _resize_for_vision_qpc(image):
    """Match the image shape used when the vision QPC was compiled."""
    target_height, target_width = smart_resize(
        height=ENCODE_CONFIG["height"],
        width=ENCODE_CONFIG["width"],
        factor=32,
        min_pixels=ENCODE_CONFIG["mm_processor_kwargs"]["min_pixels"],
        max_pixels=ENCODE_CONFIG["mm_processor_kwargs"]["max_pixels"],
    )
    if image.size != (target_width, target_height):
        image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
    return image


def _qpc_path(result, key):
    if isinstance(result, dict):
        path = result.get(key)
    else:
        path = result
    if not path:
        raise RuntimeError(f"Compilation did not return {key!r}: {result!r}")
    return str(path)


def _print_stats(name, value):
    array = np.asarray(value)
    finite = np.isfinite(array)
    finite_values = array[finite]
    minimum = finite_values.min() if finite_values.size else "n/a"
    maximum = finite_values.max() if finite_values.size else "n/a"
    print(
        f"{name}: shape={array.shape} dtype={array.dtype} "
        f"nan={np.isnan(array).sum()} +inf={np.isposinf(array).sum()} "
        f"-inf={np.isneginf(array).sum()} min={minimum} max={maximum}",
        flush=True,
    )


def _dump_array(dump_dir, name, value):
    if dump_dir is None:
        return
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    np.save(dump_dir / f"{name}.npy", np.asarray(value))


def _image_tag(image_url):
    """Return a filesystem-safe identifier derived from an image URL."""
    parsed = urlparse(image_url)
    path_parts = [part for part in parsed.path.split("/") if part]

    # Preserve the useful Picsum image ID for URLs such as /id/238/536/354.
    if "id" in path_parts:
        id_index = path_parts.index("id")
        if id_index + 1 < len(path_parts):
            candidate = path_parts[id_index + 1]
        else:
            candidate = parsed.netloc
    elif path_parts:
        candidate = Path(path_parts[-1]).stem
    else:
        candidate = parsed.netloc

    safe_tag = "".join(char if char.isalnum() or char in "._-" else "_" for char in candidate)
    return safe_tag.strip("._") or "image"


def _dump_vision_outputs(dump_dir, mode, image_url, vision_outputs):
    """Save each vision output with a filename derived from image_url."""
    if dump_dir is None:
        return

    dump_prefix = f"{mode}_image_{_image_tag(image_url)}"
    for name, value in vision_outputs.items():
        output_path = Path(dump_dir) / f"{dump_prefix}_{name}.npy"
        _dump_array(dump_dir, f"{dump_prefix}_{name}", value)
        print(f"Saved vision output {name}: {output_path}", flush=True)


def _build_model_config(model_id, num_layers):
    config = AutoConfig.from_pretrained(model_id)
    config.dtype = "float16"
    config.torch_dtype = torch.float16
    config.text_config.num_hidden_layers = num_layers
    return config


def _run_one_image(qeff_model, tokenizer, processor, vision_session, prefill_qpc, args, image_url):
    prefill_seq_len = PREFILL_CONFIG["prefill_seq_len"]
    mode = "no_blocking" if args.no_blocking else "blocking"
    image_tag = _image_tag(image_url)
    dump_prefix = f"{mode}_image_{image_tag}"
    print(f"\n=== Image: {image_url} ===", flush=True)
    image = Image.open(requests.get(image_url, stream=True, timeout=30).raw).convert("RGB")
    image = _resize_for_vision_qpc(image)
    prefill_session = QAICInferenceSession(
        prefill_qpc,
        kv_dma_share=True,
        full_batch_size=1,
        cluster_id="prefill",
    )
    try:
        assert "image_idx" in prefill_session.binding_index_map, "image_idx is missing from the prefill QPC"
        assert "batch_index" in prefill_session.binding_index_map, "batch_index is missing from the prefill QPC"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([messages])
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = qeff_model.model.prepare_inputs_for_generation(
            inputs=inputs,
            prefill_seq_len=prefill_seq_len,
            batch_size=1,
        )

        input_ids_length = inputs["input_ids"].shape[1]
        num_chunks = -(input_ids_length // -prefill_seq_len)
        padded_len = num_chunks * prefill_seq_len
        inputs["input_ids"] = torch.nn.functional.pad(
            inputs["input_ids"], (0, padded_len - input_ids_length), "constant", 1
        )
        inputs["attention_mask"] = torch.nn.functional.pad(
            inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
        )
        inputs = {name: np.asarray(value) for name, value in inputs.items()}

        vision_inputs = {
            name: value
            for name, value in inputs.items()
            if name in VISION_INPUT_KEYS and name in vision_session.input_names
        }
        for name in VISION_FP16_KEYS:
            if name in vision_inputs:
                vision_inputs[name] = vision_inputs[name].astype(np.float16)
        if not vision_inputs:
            raise RuntimeError("No processor inputs matched the vision QPC bindings")

        vision_outputs = vision_session.run(vision_inputs)
        print("Vision outputs:")
        for name in VISION_OUTPUT_KEYS:
            if name not in vision_outputs:
                raise RuntimeError(f"Missing vision output {name!r}; received {list(vision_outputs)}")
            _print_stats(name, vision_outputs[name])
        _dump_vision_outputs(args.dump_dir, mode, image_url, vision_outputs)

        prefill_input_names = set(prefill_session.input_names)
        lang_inputs = {
            name: value for name, value in inputs.items() if name not in vision_inputs and name in prefill_input_names
        }
        if "position_ids" in inputs:
            lang_inputs["position_ids"] = inputs["position_ids"]
            lang_inputs.pop("attention_mask", None)
        else:
            lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)
        lang_inputs["image_idx"] = np.array([[0]], dtype=np.int64)
        for name in VISION_OUTPUT_KEYS:
            if name in prefill_input_names:
                lang_inputs[name] = vision_outputs[name]

        kv_buffers = [np.zeros(shape, dtype=dtype) for shape, dtype in prefill_session.kv_cache_info]
        chunk_inputs = dict(lang_inputs)
        chunk_inputs["batch_index"] = np.array([[0]], dtype=np.int64)
        prefill_output = None
        last_valid_index = 0

        print(f"Running {num_chunks} prefill chunk(s)...", flush=True)
        lang_start = perf_counter()

        for chunk_idx in range(num_chunks):
            lower = chunk_idx * prefill_seq_len
            upper = (chunk_idx + 1) * prefill_seq_len
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, lower:upper]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., lower:upper]
            valid_positions = np.flatnonzero(chunk_inputs["position_ids"][0, 0] >= 0)
            last_valid_index = int(valid_positions[-1]) if valid_positions.size else 0
            last_chunk = chunk_idx == num_chunks - 1
            exec_idx = prefill_session.np_run_pipeline(
                chunk_inputs,
                last_chunk=last_chunk,
                kv_cache_buffers=kv_buffers if last_chunk else None,
            )
            prefill_session.complete_inf(exec_idx, is_prefill=True)
            prefill_output = prefill_session.get_outputs(index=exec_idx)
            if "image_idx_output" in prefill_output:
                chunk_inputs["image_idx"] = prefill_output["image_idx_output"]
            print(f"  chunk {chunk_idx + 1}/{num_chunks} complete", flush=True)
        prefill_time = perf_counter() - lang_start
        print(f">>>>>>>>>> Prefill time : {prefill_time:.2f} secs")
        logits = np.asarray(prefill_output["logits"])
        _dump_array(args.dump_dir, f"{dump_prefix}_logits", logits)
        _dump_array(args.dump_dir, f"{dump_prefix}_position_ids", inputs["position_ids"])
        print(f">>>>>>>>> logits : {logits}")
        _print_stats("prefill logits", logits)
        print(f"Prefill output keys: {sorted(prefill_output)}")

        rows = logits.reshape(-1, logits.shape[-1])
        selected_row = min(last_valid_index, rows.shape[0] - 1)
        selected_logits = rows[selected_row].astype(np.float32, copy=False)
        safe_logits = np.nan_to_num(selected_logits, nan=-np.inf)
        top_ids = np.argsort(safe_logits)[-10:][::-1]
        print(f"Selected logits row: {selected_row}")
        print("Top prefill token candidates:")
        for token_id in top_ids:
            token_id = int(token_id)
            print(
                f"  token_id={token_id} score={selected_logits[token_id]} "
                f"text={tokenizer.decode([token_id], skip_special_tokens=False)!r}"
            )
    finally:
        close = getattr(prefill_session, "close", None)
        if callable(close):
            close()
        del prefill_session
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help="number of language-model layers to compile; limited to 4 for this investigation",
    )
    parser.add_argument(
        "--image-url",
        action="append",
        dest="image_urls",
        help="Image URL; repeat this option for multiple independent runs.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--dump-dir",
        help="save vision outputs, position IDs, and full prefill logits as .npy files",
    )
    parser.add_argument(
        "--vision-qpc",
        help="reuse an existing vision QPC and compile only the language prefill stage",
    )
    parser.add_argument(
        "--prefill-qpc",
        help="reuse an existing language prefill QPC without exporting or compiling it",
    )
    parser.add_argument(
        "--no_blocking",
        action="store_true",
        help="remove qaic_config.blocking_mode from the prefill compile configuration",
    )
    parser.add_argument(
        "--no-skip-kv",
        action="store_true",
        help="set qaic_config.skip_kv=false for the prefill compile",
    )
    parser.add_argument(
        "--num-kv-blocks",
        type=int,
        help="override qaic_config.num_kv_blocks for blocking experiments",
    )
    parser.add_argument(
        "--num-q-blocks",
        type=int,
        help="override qaic_config.num_q_blocks for blocking experiments",
    )
    parser.add_argument(
        "--n-rep-chunk",
        type=int,
        help="override qaic_config.n_rep_chunk for blocking experiments",
    )
    parser.add_argument(
        "--convert-to-fp16",
        action="store_true",
        help="keep the default -convert-to-fp16 compiler flag (the repro skips it unless this is set)",
    )
    parser.add_argument(
        "--no-onnx-subfunctions",
        action="store_true",
        help="disable ONNX subfunction extraction for diagnostic QPCs",
    )
    args = parser.parse_args()
    image_urls = args.image_urls or DEFAULT_IMAGE_URLS

    if not args.convert_to_fp16:
        # QEfficient has no public switch for this flag: it is derived from the model dtype.
        modeling_auto._should_convert_to_fp16 = lambda target_dtype, compiler_options: False
        print("Compiler: -convert-to-fp16 is NOT passed; the ONNX per-op precision is honored")

    print(f"Prompt: {args.prompt}")
    print(f"Independent image runs: {len(image_urls)}")

    if not 1 <= args.num_layers <= DEFAULT_NUM_LAYERS:
        parser.error(f"--num-layers must be between 1 and {DEFAULT_NUM_LAYERS}")
    num_layers = args.num_layers
    print(f"Transformer layers: {num_layers}")
    config = _build_model_config(args.model_id, num_layers)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        kv_offload=True,
        continuous_batching=True,
        config=config,
        dtype=torch.float16,
        layerwise=False,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id)

    prefill_config = deepcopy(PREFILL_CONFIG)
    if args.no_blocking:
        removed_keys = (
            "blocking_mode",
            "num_kv_blocks",
            "num_q_blocks",
            "n_rep_chunk",
            "ctx_len",
        )
        for key in removed_keys:
            prefill_config["qaic_config"].pop(key, None)
        print(f"Prefill compile: removed qaic_config keys {removed_keys}; kept moe_config")
    if args.no_skip_kv:
        prefill_config["qaic_config"]["skip_kv"] = False
        print("Prefill compile: qaic_config.skip_kv=false")
    for option_name, config_name in (
        ("num_kv_blocks", "num_kv_blocks"),
        ("num_q_blocks", "num_q_blocks"),
        ("n_rep_chunk", "n_rep_chunk"),
    ):
        value = getattr(args, option_name)
        if value is not None:
            prefill_config["qaic_config"][config_name] = value
            print(f"Prefill compile: qaic_config.{config_name}={value}")
    encode_config = deepcopy(ENCODE_CONFIG)
    if args.no_onnx_subfunctions:
        encode_config["use_onnx_subfunctions"] = False
        prefill_config["use_onnx_subfunctions"] = False
        print("Diagnostic compile: use_onnx_subfunctions=false")
    print(f">>>>>>>>>>>> prefill_config :{prefill_config}")

    if args.vision_qpc:
        print(f"Reusing vision QPC: {args.vision_qpc}")
        vision_qpc = args.vision_qpc
    else:
        vision_result = qeff_model.compile(**encode_config)
        vision_qpc = _qpc_path(vision_result, "vision_qpc_path")
    if args.prefill_qpc:
        print(f"Reusing prefill QPC: {args.prefill_qpc}")
        prefill_qpc = args.prefill_qpc
    else:
        prefill_result = qeff_model.compile(**prefill_config)
        prefill_qpc = _qpc_path(prefill_result, "lang_prefill_qpc_path")
    print(f"Vision QPC: {vision_qpc}")
    print(f"Prefill QPC: {prefill_qpc}")

    vision_session = QAICInferenceSession(vision_qpc)
    for image_url in image_urls:
        _run_one_image(qeff_model, tokenizer, processor, vision_session, prefill_qpc, args, image_url)


if __name__ == "__main__":
    main()
