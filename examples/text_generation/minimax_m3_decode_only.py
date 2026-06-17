# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import functools
import os
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

import QEfficient
from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.base.modeling_qeff import QEFFBaseModel

MODEL_ID = "MiniMaxAI/MiniMax-M3"


def _resolve_text_config(config):
    return getattr(config, "text_config", config)


def _resolve_total_layers(config) -> int:
    total_layers = getattr(_resolve_text_config(config), "num_hidden_layers", None)
    if total_layers is None:
        raise ValueError("Could not resolve num_hidden_layers from config/text_config.")
    return int(total_layers)


def _build_windows(total_layers: int, window_size: int):
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}.")
    return [(start, min(start + window_size, total_layers)) for start in range(0, total_layers, window_size)]


def _layers_container(model):
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "language_model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(model, "layers", None),
    ]
    return next((layers for layers in candidates if layers is not None), None)


def _null_outside_window_layers(model):
    layers = _layers_container(model)
    if layers is None:
        return
    start = int(getattr(QEFFBaseModel, "_start", 0) or 0)
    end = int(getattr(QEFFBaseModel, "_end", 0) or 0)
    if end <= start:
        return
    for layer_idx in range(len(layers)):
        if layer_idx < start or layer_idx >= end:
            layers[layer_idx] = None


def _install_model_init_window_patch(model_cls):
    if model_cls is None or getattr(model_cls, "_qeff_minimax_window_patch_installed", False):
        return
    original_init = model_cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if os.environ.get("LAYERWISE_EXPORT", "False") == "True":
            _null_outside_window_layers(self)

    model_cls.__init__ = patched_init
    model_cls._qeff_minimax_window_patch_installed = True


def _install_shard_window_patch():
    if getattr(transformers.modeling_utils, "_qeff_minimax_window_shard_patch_installed", False):
        return
    original_get_checkpoint_shard_files = transformers.modeling_utils.get_checkpoint_shard_files

    @functools.wraps(original_get_checkpoint_shard_files)
    def patched_get_checkpoint_shard_files(*args, **kwargs):
        shard_files, metadata = original_get_checkpoint_shard_files(*args, **kwargs)
        if os.environ.get("LAYERWISE_EXPORT", "False") != "True":
            return shard_files, metadata

        weight_map = metadata.get("weight_map")
        if not weight_map:
            return shard_files, metadata

        start = int(getattr(QEFFBaseModel, "_start", 0) or 0)
        end = int(getattr(QEFFBaseModel, "_end", 0) or 0)
        if end <= start:
            return shard_files, metadata

        layer_prefixes = tuple(
            prefix
            for layer_idx in range(start, end)
            for prefix in (
                f"model.layers.{layer_idx}.",
                f"layers.{layer_idx}.",
                f"language_model.layers.{layer_idx}.",
                f"model.language_model.layers.{layer_idx}.",
            )
        )
        layer_owner_prefixes = (
            "model.layers.",
            "layers.",
            "language_model.layers.",
            "model.language_model.layers.",
        )
        filtered_weight_map = {}
        for checkpoint_key, shard_name in weight_map.items():
            if checkpoint_key.startswith(layer_owner_prefixes):
                if checkpoint_key.startswith(layer_prefixes):
                    filtered_weight_map[checkpoint_key] = shard_name
                continue
            filtered_weight_map[checkpoint_key] = shard_name

        if not filtered_weight_map:
            return shard_files, metadata

        shard_name_to_path = {path.split("/")[-1]: path for path in shard_files}
        filtered_shard_names = sorted(set(filtered_weight_map.values()))
        filtered_shard_files = [shard_name_to_path[name] for name in filtered_shard_names if name in shard_name_to_path]
        if not filtered_shard_files:
            return shard_files, metadata

        metadata["weight_map"] = filtered_weight_map
        metadata["all_checkpoint_keys"] = list(filtered_weight_map.keys())
        return filtered_shard_files, metadata

    transformers.modeling_utils.get_checkpoint_shard_files = patched_get_checkpoint_shard_files
    transformers.modeling_utils._qeff_minimax_window_shard_patch_installed = True


def _install_minimax_window_patches():
    _install_shard_window_patch()
    minimax_mod = getattr(getattr(transformers.models, "minimax_m3_vl", None), "modeling_minimax_m3_vl", None)
    if minimax_mod is None:
        return
    for class_name in ("MiniMaxM3VLForCausalLM", "MiniMaxM3SparseForConditionalGeneration"):
        _install_model_init_window_patch(getattr(minimax_mod, class_name, None))


def _set_layer_window(start: int, end: int, total_layers: int):
    QEFFBaseModel._start = start
    QEFFBaseModel._end = end
    QEFFBaseModel._total_layers = total_layers
    QEFFBaseModel._layerwise_active = True


def _reset_layer_window():
    QEFFBaseModel._start = 0
    QEFFBaseModel._end = 0
    QEFFBaseModel._total_layers = None
    QEFFBaseModel._layerwise_active = False


def _resolve_export_root(onnx_path: Path) -> Path:
    parts = list(onnx_path.parts)
    if "onnx_layerwise_tmp" in parts:
        return Path(*parts[: parts.index("onnx_layerwise_tmp")])
    return onnx_path.parent


def _compile_kwargs(args, qaic_config):
    return {
        "prefill_seq_len": 1,
        "ctx_len": args.ctx_len,
        "num_cores": args.num_cores,
        "mxfp6_matmul": args.mxfp6_matmul,
        "num_devices": args.num_devices,
        "use_onnx_subfunctions": True,
        "offload_pt_weights": False,
        "retain_full_kv": True,
        "qaic_config": qaic_config,
    }


def _load_qeff_model(model_id, qaic_config):
    return QEFFAutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.float16,
        qaic_config=qaic_config,
    )


def compile_layerwise(args, qaic_config):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    total_layers = _resolve_total_layers(config)
    windows = _build_windows(total_layers, args.layerwise_window_size)
    export_root = args.work_dir / "onnx"
    export_root.mkdir(parents=True, exist_ok=True)

    _install_minimax_window_patches()
    os.environ["LAYERWISE_EXPORT"] = "True"
    first_onnx_path = None
    qeff_model = None
    try:
        for start, end in windows:
            print(f"Exporting MiniMax-M3 layer window [{start}, {end})/{total_layers}")
            _set_layer_window(start, end, total_layers)
            qeff_model = _load_qeff_model(args.model_id, qaic_config)
            _null_outside_window_layers(qeff_model.model)
            onnx_path = Path(
                qeff_model.export(
                    export_dir=export_root,
                    prefill_seq_len=1,
                    use_onnx_subfunctions=True,
                    offload_pt_weights=True,
                )
            )
            if first_onnx_path is None:
                first_onnx_path = onnx_path
    finally:
        os.environ["LAYERWISE_EXPORT"] = "False"
        _reset_layer_window()

    if first_onnx_path is None or qeff_model is None:
        raise RuntimeError("Layerwise export did not produce any ONNX shard.")

    final_onnx_path = QEfficient.utils.layerwise_pipeline(str(_resolve_export_root(first_onnx_path)))
    print(f"Merged layerwise ONNX: {final_onnx_path}")
    return qeff_model.compile(
        onnx_path=final_onnx_path,
        compile_dir=args.work_dir / "compile",
        **_compile_kwargs(args, qaic_config),
    ), qeff_model


def compile_regular(args, qaic_config):
    qeff_model = _load_qeff_model(args.model_id, qaic_config)
    return qeff_model.compile(compile_dir=args.work_dir / "compile", **_compile_kwargs(args, qaic_config)), qeff_model


def main():
    parser = argparse.ArgumentParser(description="Compile and generate with MiniMax-M3 decode-only PL=1.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--generation-len", type=int, default=16)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--device-id", type=int, nargs="+", default=[0])
    parser.add_argument("--prompt", default="Once upon a time,")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/qeff-minimax-m3-decode-only"))
    parser.add_argument("--layerwise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layerwise-window-size", type=int, default=1)
    parser.add_argument("--blocking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-kv-blocks", type=int, default=2)
    parser.add_argument("--mxfp6-matmul", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    qaic_config = None
    if args.blocking:
        qaic_config = {"enable_blocking": True, "blocking_mode": "kv", "num_kv_blocks": args.num_kv_blocks}

    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.layerwise:
        qpc_path, qeff_model = compile_layerwise(args, qaic_config)
    else:
        qpc_path, qeff_model = compile_regular(args, qaic_config)
    print(f"QPC path: {qpc_path}")

    tokenizer_id = args.tokenizer_id or args.model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    qeff_model.generate(
        prompts=[args.prompt],
        tokenizer=tokenizer,
        device_id=args.device_id,
        generation_len=args.generation_len,
    )


if __name__ == "__main__":
    main()
