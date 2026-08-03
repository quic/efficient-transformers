# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Export, compile, and generate with a reduced Kimi K2.5 model slice.

The ONNX export step uses the Dynamo path directly on the dual-QPC component
wrappers. The exported ONNX files are then passed to the normal QEfficient
compile and generation APIs.
"""

import argparse
import importlib.util
import os
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image

from QEfficient import QEFFAutoModelForImageTextToText

LOAD_KIMI_UTILS_PATH = Path(__file__).resolve().parents[2] / "tests" / "utils" / "load_kimi_utils.py"
_load_kimi_spec = importlib.util.spec_from_file_location("load_kimi_utils", LOAD_KIMI_UTILS_PATH)
if _load_kimi_spec is None or _load_kimi_spec.loader is None:
    raise ImportError(f"Unable to load Kimi helpers from {LOAD_KIMI_UTILS_PATH}")
load_kimi_utils = importlib.util.module_from_spec(_load_kimi_spec)
_load_kimi_spec.loader.exec_module(load_kimi_utils)

LOADED_EXPERT_IDS = load_kimi_utils.LOADED_EXPERT_IDS
NUM_EXPERTS_PER_TOKEN = load_kimi_utils.NUM_EXPERTS_PER_TOKEN
NUM_TEXT_LAYERS = load_kimi_utils.NUM_TEXT_LAYERS
NUM_VISION_LAYERS = load_kimi_utils.NUM_VISION_LAYERS
parse_expert_ids = load_kimi_utils.parse_expert_ids
set_deterministic = load_kimi_utils.set_deterministic
load_kimi_k25_layer_subset_model = load_kimi_utils.load_kimi_k25_layer_subset_model

DEFAULT_EXPORT_DIR = Path.home() / ".cache" / "qeff" / "kimi_k25_dynamo_export"
DEFAULT_COMPILE_DIR = Path.home() / ".cache" / "qeff" / "kimi_k25_dynamo_compile"
DEFAULT_IMAGE_URL = "https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Kimi K2.5 ONNX with Dynamo, compile QPCs, and run generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default="/home/huggingface_hub/models--moonshotai--Kimi-K2.5/snapshots/4d01dfe0332d63057c186e0b262165819efb6611",
        help="Local Kimi-K2.5 snapshot path. Uses the local HF cache/default snapshot when omitted.",
    )
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--compile-dir", type=Path, default=DEFAULT_COMPILE_DIR)
    parser.add_argument("--vision-onnx-path", type=Path, default=None)
    parser.add_argument("--lang-onnx-path", type=Path, default=None)
    parser.add_argument("--vision-qpc-path", type=Path, default=None)
    parser.add_argument("--lang-qpc-path", type=Path, default=None)
    parser.add_argument("--component", choices=("lang", "vision", "both"), default="both")
    parser.add_argument("--num-vision-layers", type=int, default=NUM_VISION_LAYERS)
    parser.add_argument("--num-text-layers", type=int, default=NUM_TEXT_LAYERS)
    parser.add_argument("--expert-ids", type=parse_expert_ids, default=LOADED_EXPERT_IDS)
    parser.add_argument("--num-experts-per-token", type=int, default=NUM_EXPERTS_PER_TOKEN)
    parser.add_argument("--prefill-seq-len", type=int, default=2)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--image-url", type=str, default=DEFAULT_IMAGE_URL)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--generation-len", type=int, default=10)
    parser.add_argument(
        "--device-ids",
        type=lambda device_ids: [int(device_id) for device_id in device_ids.strip("[]").split(",")],
        default=[0],
        help="Device IDs for generation, e.g. [0] or [0,1].",
    )
    parser.add_argument("--mxfp6-matmul", action="store_true")
    parser.add_argument("--mxint8-kv-cache", action="store_true")
    parser.add_argument("--mos", type=int, default=1)
    parser.add_argument("--aic-enable-depth-first", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument(
        "--use-onnx-subfunctions",
        action="store_true",
        help="Enable repeated-subgraph ONNX functions during Dynamo export.",
    )
    parser.add_argument(
        "--keep-weights",
        action="store_true",
        help="Keep PyTorch weights resident after export instead of offloading them to meta tensors.",
    )
    parser.add_argument(
        "--use-weight-free-export",
        action="store_true",
        help="Export ONNX graphs without embedded weights and emit weight_spec.json metadata.",
    )
    args = parser.parse_args()
    if (args.image_height is None) != (args.image_width is None):
        parser.error("--image-height and --image-width must be provided together.")
    if not args.skip_generate and args.component != "both":
        parser.error("Generation requires --component both so both vision and language QPCs are available.")
    if args.skip_export and (args.vision_onnx_path is None or args.lang_onnx_path is None) and not args.skip_compile:
        parser.error("--skip-export requires --vision-onnx-path and --lang-onnx-path unless --skip-compile is set.")
    if args.skip_compile and not args.skip_generate and (args.vision_qpc_path is None or args.lang_qpc_path is None):
        parser.error("--skip-compile generation requires --vision-qpc-path and --lang-qpc-path.")
    return args


def validate_dynamo_torch_version():
    torch_version = torch.__version__
    major, minor = (int(part) for part in torch_version.split("+")[0].split(".")[:2])
    if (major, minor) < (2, 13):
        raise RuntimeError(
            f"Kimi K2.5 Dynamo export requires PyTorch >= 2.13, but {torch_version} is installed. "
            "Install the Dynamo requirements before running this example."
        )


def configure_qaic_tool_path():
    qaic_exec_path = Path("/opt/qti-aic/exec")
    if qaic_exec_path.exists():
        os.environ["PATH"] = f"{qaic_exec_path}{os.pathsep}{os.environ.get('PATH', '')}"


def build_qeff_model(args):
    model_path = args.model_path
    subset_model_path = None
    model_ref = model_path
    if args.use_weight_free_export:
        subset_model_path = args.export_dir.expanduser().resolve() / "kimi_k25_weightfree_subset"
        model_ref = subset_model_path
    model, tokenizer, processor = load_kimi_k25_layer_subset_model(
        model_path=model_path,
        num_vision_layers=args.num_vision_layers,
        num_text_layers=args.num_text_layers,
        loaded_expert_ids=args.expert_ids,
        num_experts_per_tok=args.num_experts_per_token,
        dtype=torch.float32,
        seed=args.seed,
        subset_model_path=subset_model_path,
    )
    model.eval()
    qaic_config = {"mla_absorption": {"cache_compressed": True, "absorption": False, "online": False}}
    qeff_model = QEFFAutoModelForImageTextToText(
        model, qaic_config=qaic_config, pretrained_model_name_or_path=str(model_ref)
    )
    qeff_model.model.eval()
    qeff_model.vision_model.model.eval()
    qeff_model.lang_model.model.eval()
    qeff_model.transform(
        ctx_len=args.ctx_len,
        seq_len=args.prefill_seq_len,
        bs=1,
        num_devices=args.num_devices,
        qaic_config=qaic_config,
        aic_num_cores=args.num_cores,
    )
    precompute_vision_rope_cache(qeff_model)
    return qeff_model, tokenizer, processor, qaic_config


def precompute_vision_rope_cache(qeff_model):
    rope_2d = qeff_model.vision_model.model.model.vision_tower.encoder.rope_2d
    rope_2d._ensure_precomputed_freqs(torch.device("cpu"))


def get_component_export_args(qeff_model, prefill_seq_len: int):
    inputs = qeff_model.model.get_dummy_inputs(
        kv_offload=True,
        continuous_batching=qeff_model.continuous_batching,
        prefill_seq_len=prefill_seq_len,
    )
    normalize_nested_cache_inputs_for_dynamo(inputs)
    dynamic_axes = qeff_model.model.get_onnx_dynamic_axes(
        kv_offload=True,
        continuous_batching=qeff_model.continuous_batching,
        comp_ctx_lengths=qeff_model.comp_ctx_lengths_decode,
    )
    output_names = qeff_model.model.get_output_names(kv_offload=True)
    add_static_dynamic_axes_for_dynamo(inputs, dynamic_axes)
    remove_vision_output_dynamic_axes_for_dynamo(dynamic_axes)
    freeze_batch_axes_for_dynamo(dynamic_axes)
    return inputs, output_names, dynamic_axes


def normalize_nested_cache_inputs_for_dynamo(inputs):
    for component_inputs in inputs.values():
        if "compressed_kvs" in component_inputs:
            component_inputs["compressed_kvs"] = [tuple(layer) for layer in component_inputs["compressed_kvs"]]
        if "past_key_values" in component_inputs:
            component_inputs["past_key_values"] = [list(layer) for layer in component_inputs["past_key_values"]]


def add_static_dynamic_axes_for_dynamo(inputs, dynamic_axes):
    nested_cache_inputs = {"past_key_values", "compressed_kvs"}
    for component_name, component_inputs in inputs.items():
        component_dynamic_axes = dynamic_axes[component_name]
        for input_name in component_inputs:
            if input_name not in nested_cache_inputs:
                component_dynamic_axes.setdefault(input_name, {})


def remove_vision_output_dynamic_axes_for_dynamo(dynamic_axes):
    dynamic_axes["vision"].pop("vision_embeds", None)


def freeze_batch_axes_for_dynamo(dynamic_axes):
    for component_dynamic_axes in dynamic_axes.values():
        for axes_map in component_dynamic_axes.values():
            for axis_idx, dim_name in tuple(axes_map.items()):
                if dim_name in {"batch_size", "full_batch_size"}:
                    axes_map.pop(axis_idx)


def export_vision(qeff_model, inputs, output_names, dynamic_axes, args):
    return qeff_model.vision_model._export(
        inputs["vision"],
        output_names=output_names["vision"],
        dynamic_axes=dynamic_axes["vision"],
        export_dir=args.export_dir,
        offload_pt_weights=False,
        dynamo=True,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        use_weight_free_export=args.use_weight_free_export,
    )


def export_language(qeff_model, inputs, output_names, dynamic_axes, args):
    qeff_model.lang_model.hash_params["prefill_only"] = False
    onnx_path = qeff_model.lang_model._export(
        inputs["lang"],
        output_names=output_names["lang"],
        dynamic_axes=dynamic_axes["lang"],
        export_dir=args.export_dir,
        offload_pt_weights=not args.keep_weights,
        dynamo=True,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        use_weight_free_export=args.use_weight_free_export,
    )
    return onnx_path


def export_components(qeff_model, inputs, output_names, dynamic_axes, args):
    exported_paths = {"vision": args.vision_onnx_path, "lang": args.lang_onnx_path}
    if args.skip_export:
        return {component_name: path for component_name, path in exported_paths.items() if path is not None}
    if args.component in {"vision", "both"}:
        exported_paths["vision"] = export_vision(qeff_model, inputs, output_names, dynamic_axes, args)
    if args.component in {"lang", "both"}:
        exported_paths["lang"] = export_language(qeff_model, inputs, output_names, dynamic_axes, args)
    return exported_paths


def load_generation_image(args):
    if args.image_path is not None:
        image = Image.open(args.image_path).convert("RGB")
    else:
        response = requests.get(args.image_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    if args.image_height is not None:
        image = image.resize((args.image_width, args.image_height))
    return image


def compile_components(qeff_model, exported_paths, image, qaic_config, args):
    if args.skip_compile:
        qpc_paths = {"vision_qpc_path": str(args.vision_qpc_path), "lang_qpc_path": str(args.lang_qpc_path)}
        qeff_model.vision_model.qpc_path = qpc_paths["vision_qpc_path"]
        qeff_model.lang_model.qpc_path = qpc_paths["lang_qpc_path"]
        qeff_model.qpc_paths = qpc_paths
        return qpc_paths

    qpc_paths = qeff_model.compile(
        vision_onnx_path=str(exported_paths.get("vision")),
        lang_onnx_path=str(exported_paths.get("lang")),
        compile_dir=str(args.compile_dir),
        qaic_config=qaic_config,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=args.mxfp6_matmul,
        mxint8_kv_cache=args.mxint8_kv_cache,
        skip_vision=args.component == "lang",
        skip_lang=args.component == "vision",
        aic_enable_depth_first=args.aic_enable_depth_first,
        mos=args.mos,
        image_height=image.height if image is not None else args.image_height,
        image_width=image.width if image is not None else args.image_width,
        use_weight_free_export=args.use_weight_free_export,
    )
    return qpc_paths


def build_generation_inputs(processor, qeff_model, image, args):
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
    inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)
    return inputs


def generate(qeff_model, tokenizer, processor, image, args):
    inputs = build_generation_inputs(processor, qeff_model, image, args)
    output = qeff_model.generate(
        inputs=inputs,
        device_ids=args.device_ids,
        generation_len=args.generation_len,
        image_height=image.height,
        image_width=image.width,
    )
    print(output.generated_ids)
    print(tokenizer.batch_decode(output.generated_ids))
    print(output)
    return output


def main():
    os.environ.setdefault("HF_HUB_CACHE", "/home/huggingface_hub")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    args = parse_args()
    validate_dynamo_torch_version()
    configure_qaic_tool_path()
    qeff_model, tokenizer, processor, qaic_config = build_qeff_model(args)
    inputs, output_names, dynamic_axes = get_component_export_args(qeff_model, args.prefill_seq_len)

    exported_paths = export_components(qeff_model, inputs, output_names, dynamic_axes, args)

    for component_name, onnx_path in exported_paths.items():
        print(f"{component_name} ONNX exported to: {onnx_path}")

    image = load_generation_image(args) if not args.skip_generate else None
    if not args.skip_compile or not args.skip_generate:
        qpc_paths = compile_components(qeff_model, exported_paths, image, qaic_config, args)
        print(f"QPC paths: {qpc_paths}")
    if not args.skip_generate:
        generate(qeff_model, tokenizer, processor, image, args)


if __name__ == "__main__":
    main()
