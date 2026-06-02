# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import functools
import os
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig
import QEfficient
from QEfficient import QEFFAutoModelForImageTextToText


MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
PREFILL_SEQ_LEN = 32
CTX_LEN = 4096
TEXT_WINDOW_SIZE = 1

# For quick local validation only (keep disabled for real export)
# TEST_TEXT_LAYERS = 4

# Export controls
BATCH_SIZE = 1
NUM_CORES = 16
NUM_DEVICES = 1
HEIGHT = 354
WIDTH = 536

def _ensure_pretrained_window_attrs():
    if not hasattr(transformers.modeling_utils.PreTrainedModel, "_start"):
        transformers.modeling_utils.PreTrainedModel._start = 0
    if not hasattr(transformers.modeling_utils.PreTrainedModel, "_end"):
        transformers.modeling_utils.PreTrainedModel._end = 0
    if not hasattr(transformers.modeling_utils.PreTrainedModel, "_total_layers"):
        transformers.modeling_utils.PreTrainedModel._total_layers = 0
    if not hasattr(transformers.modeling_utils.PreTrainedModel, "_text_start"):
        transformers.modeling_utils.PreTrainedModel._text_start = 0
    if not hasattr(transformers.modeling_utils.PreTrainedModel, "_text_end"):
        transformers.modeling_utils.PreTrainedModel._text_end = 0
    if not hasattr(transformers.modeling_utils.PreTrainedModel, "_text_total_layers"):
        transformers.modeling_utils.PreTrainedModel._text_total_layers = 0


def _build_layer_windows(total_layers: int, window_size: int):
    if total_layers <= 0:
        raise ValueError(f"Invalid total_layers={total_layers}. Expected: total_layers > 0.")
    if window_size <= 0:
        raise ValueError(f"Invalid window_size={window_size}. Expected: window_size > 0.")

    windows = []
    start = 0
    while start < total_layers:
        end = min(total_layers, start + window_size)
        windows.append((start, end))
        start = end
    return windows


def _get_text_layers_container(model):
    # VLM path first
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    # LLM-compatible fallbacks
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "layers"):
        return model.layers
    return None


def _null_outside_window_layers(model, apply_text: bool = True):
    if apply_text:
        text_start = int(
            getattr(
                transformers.modeling_utils.PreTrainedModel,
                "_text_start",
                getattr(transformers.modeling_utils.PreTrainedModel, "_start", 0),
            )
        )
        text_end = int(
            getattr(
                transformers.modeling_utils.PreTrainedModel,
                "_text_end",
                getattr(transformers.modeling_utils.PreTrainedModel, "_end", 0),
            )
        )
        text_layers = _get_text_layers_container(model)
        if text_layers is not None and text_end > text_start:
            for idx, _ in enumerate(text_layers):
                if idx < text_start or idx >= text_end:
                    text_layers[idx] = None

def _install_window_patch(model_cls):
    if getattr(model_cls, "_window_patch_installed", False):
        return

    original_init = model_cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _null_outside_window_layers(self, apply_text=True)

    model_cls.__init__ = patched_init
    model_cls._window_patch_installed = True


def _resolve_export_root(onnx_path: Path) -> Path:
    parts = list(onnx_path.parts)
    if "onnx_layerwise_tmp" in parts:
        marker_idx = parts.index("onnx_layerwise_tmp")
        return Path(*parts[:marker_idx])
    return onnx_path.parent


def _install_shard_window_patch():
    if getattr(transformers.modeling_utils, "_window_shard_patch_installed", False):
        return

    original_get_checkpoint_shard_files = transformers.modeling_utils.get_checkpoint_shard_files

    @functools.wraps(original_get_checkpoint_shard_files)
    def patched_get_checkpoint_shard_files(*args, **kwargs):
        shard_files, metadata = original_get_checkpoint_shard_files(*args, **kwargs)
        weight_map = metadata.get("weight_map")
        if not weight_map:
            return shard_files, metadata

        start = int(getattr(transformers.modeling_utils.PreTrainedModel, "_start", 0))
        end = int(getattr(transformers.modeling_utils.PreTrainedModel, "_end", 0))
        text_start = int(getattr(transformers.modeling_utils.PreTrainedModel, "_text_start", start))
        text_end = int(getattr(transformers.modeling_utils.PreTrainedModel, "_text_end", end))
        has_text_window = text_end > text_start
        if not has_text_window:
            return shard_files, metadata

        selected_text_prefixes = tuple(
            [f"model.layers.{layer_idx}." for layer_idx in range(text_start, text_end)]
            + [f"model.language_model.layers.{layer_idx}." for layer_idx in range(text_start, text_end)]
        )
        filtered_weight_map = {}
        for checkpoint_key, shard_name in weight_map.items():
            if checkpoint_key.startswith("model.layers.") or checkpoint_key.startswith("model.language_model.layers."):
                if not has_text_window or checkpoint_key.startswith(selected_text_prefixes):
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
    transformers.modeling_utils._window_shard_patch_installed = True


def _set_layer_windows(
    text_start: int,
    text_end: int,
    text_total_layers: int,
):
    transformers.modeling_utils.PreTrainedModel._start = text_start
    transformers.modeling_utils.PreTrainedModel._end = text_end
    transformers.modeling_utils.PreTrainedModel._total_layers = text_total_layers
    transformers.modeling_utils.PreTrainedModel._text_start = text_start
    transformers.modeling_utils.PreTrainedModel._text_end = text_end
    transformers.modeling_utils.PreTrainedModel._text_total_layers = text_total_layers

    # Qwen3-VL-MoE model code still checks QEffQwen3_5MoeTextModel window attrs
    # in a few places. Set both classes to keep layer-wise behavior consistent.
    qeff_vl_mod = QEfficient.transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe
    qeff_vl_mod.QEffQwen3VLMoeTextModel._start = text_start
    qeff_vl_mod.QEffQwen3VLMoeTextModel._end = text_end
    qeff_vl_mod.QEffQwen3VLMoeTextModel._total_layers = text_total_layers

    qeff_35_mod = getattr(QEfficient.transformers.models, "qwen3_5_moe", None)
    if qeff_35_mod is not None:
        qeff_35_text_model = getattr(qeff_35_mod.modeling_qwen3_5_moe, "QEffQwen3_5MoeTextModel", None)
        if qeff_35_text_model is not None:
            qeff_35_text_model._start = text_start
            qeff_35_text_model._end = text_end
            qeff_35_text_model._total_layers = text_total_layers

    QEfficient.base.modeling_qeff.QEFFBaseModel._start = text_start
    QEfficient.base.modeling_qeff.QEFFBaseModel._end = text_end
    QEfficient.base.modeling_qeff.QEFFBaseModel._total_layers = text_total_layers


def _stitch_layerwise_if_available(export_root: Path):
    # Some branches expose this helper; fall back gracefully when unavailable.
    pipeline_fn = getattr(QEfficient.utils, "layerwise_pipeline", None)
    if callable(pipeline_fn):
        return pipeline_fn(str(export_root))
    print(f"layerwise_pipeline() not found. Layer-wise ONNX shards kept under: {export_root / 'onnx_layerwise_tmp'}")
    return str(export_root / "onnx_layerwise_tmp")


def _new_qeff_model(model_id: str, config):
    return QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        torch_dtype=torch.float32,
    )


def main():
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.torch_dtype = "float32"
    config.vision_config.depth = 9
    # config.text_config.num_hidden_layers = 2
    config.vision_config.deepstack_visual_indexes = [8]

    # if TEST_TEXT_LAYERS:
    #     config.text_config.num_hidden_layers = TEST_TEXT_LAYERS

    text_config = getattr(config, "text_config", config)
    text_total_layers = 2 # getattr(text_config, "num_hidden_layers", None)
    if text_total_layers is None:
        raise ValueError("Could not resolve `num_hidden_layers` from config.text_config.")
    _ensure_pretrained_window_attrs()
    _install_shard_window_patch()

    hf_qwen_mod = transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe
    _install_window_patch(hf_qwen_mod.Qwen3VLMoeForConditionalGeneration)
    if hasattr(hf_qwen_mod, "Qwen3VLMoeForCausalLM"):
        _install_window_patch(hf_qwen_mod.Qwen3VLMoeForCausalLM)

    text_windows = _build_layer_windows(total_layers=text_total_layers, window_size=TEXT_WINDOW_SIZE)
    # Keep layerwise only on text path in this loop.
    num_windows = len(text_windows)
    first_onnx_path = None
    os.environ["LAYERWISE_EXPORT"] = "True"
    for window_idx in range(num_windows):
        text_start, text_end = text_windows[window_idx] if window_idx < len(text_windows) else (0, 0)
        skip_lang_for_window = window_idx >= len(text_windows)

        _set_layer_windows(
            text_start=text_start,
            text_end=text_end,
            text_total_layers=text_total_layers,
        )
        print(
            f"Exporting window {window_idx + 1}/{num_windows} "
            f"text=[{text_start},{text_end})/{text_total_layers} "
            f"skip_lang={skip_lang_for_window}"
        )

        qeff_model = _new_qeff_model(MODEL_ID, config)
        if hasattr(qeff_model, "model"):
            _null_outside_window_layers(
                qeff_model.model,
                apply_text=not skip_lang_for_window,
            )
        
        onnx_path = qeff_model.compile(
            batch_size=BATCH_SIZE,
            prefill_seq_len=PREFILL_SEQ_LEN,
            ctx_len=CTX_LEN,
            num_cores=NUM_CORES,
            num_devices=NUM_DEVICES,
            height=HEIGHT,
            width=WIDTH,
            mxfp6_matmul=True,
            aic_enable_depth_first=True,
            skip_vision=True,
            skip_lang=skip_lang_for_window,
            prefill_only=True,
            enable_chunking=True,
            split_retained_state_io=True,
            use_onnx_subfunctions=True,
            mos=1,
        )

        # import pdb; pdb.set_trace()
        if first_onnx_path is None:
            first_onnx_path = Path(str(onnx_path["lang_prefill_qpc_path"]))

    if first_onnx_path is None:
        raise RuntimeError("No ONNX path produced during layer-wise language export.")

    export_root = _resolve_export_root(first_onnx_path)
    final_artifact = _stitch_layerwise_if_available(export_root)
    print(f"Layer-wise language export completed. Final artifact/root: {final_artifact}")

    os.environ["LAYERWISE_EXPORT"] = "False"

if __name__ == "__main__":
    main()
