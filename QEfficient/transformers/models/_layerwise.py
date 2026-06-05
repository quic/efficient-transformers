# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Layer-wise export/compile orchestration for selected MoE architectures.

Provisional API. Scheduled for removal once first-class multi-window export lands.
Today only ``qwen3_vl_moe``, ``qwen3_5_moe`` and ``qwen3_moe`` carry the
windowing hooks (``_start``/``_end`` class attributes) that this driver relies
on; calling :func:`run_layerwise` for any other architecture will raise.
"""

from __future__ import annotations

import functools
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import transformers

import QEfficient

# Architectures whose modeling files declare _start/_end class attributes the
# layer-wise driver pokes. Keep this list narrow on purpose - adding a new
# architecture must come with the corresponding modeling-file hooks.
_LAYERWISE_SUPPORTED_MODEL_TYPES = frozenset(
    {
        "qwen3_vl_moe",
        "qwen3_vl_moe_text",
        "qwen3_5_moe",
        "qwen3_moe",
    }
)

_DEPRECATION_WARNED = False


def _maybe_warn_deprecation() -> None:
    global _DEPRECATION_WARNED
    if _DEPRECATION_WARNED:
        return
    warnings.warn(
        "layerwise=True is a provisional API and will be deprecated once "
        "first-class multi-window export lands. Use only for the supported "
        f"model types: {sorted(_LAYERWISE_SUPPORTED_MODEL_TYPES)}.",
        DeprecationWarning,
        stacklevel=3,
    )
    _DEPRECATION_WARNED = True


def _resolve_model_type(config) -> str:
    text_config = getattr(config, "text_config", None)
    if text_config is not None and getattr(text_config, "model_type", None):
        return text_config.model_type
    return getattr(config, "model_type", "") or ""


def assert_layerwise_supported(config) -> str:
    """Raise a clear error if the model architecture has no layerwise hooks."""
    top_type = getattr(config, "model_type", "") or ""
    text_type = _resolve_model_type(config)
    if top_type in _LAYERWISE_SUPPORTED_MODEL_TYPES or text_type in _LAYERWISE_SUPPORTED_MODEL_TYPES:
        return text_type or top_type
    raise NotImplementedError(
        "layerwise=True is only supported for model types: "
        f"{sorted(_LAYERWISE_SUPPORTED_MODEL_TYPES)}. Got model_type="
        f"'{top_type}' (text_config.model_type='{text_type}'). "
        "Run with layerwise=False (the default) for this model."
    )


# ---------------------------------------------------------------------------
# Internal helpers (lifted from the legacy example script)
# ---------------------------------------------------------------------------


def _ensure_pretrained_window_attrs() -> None:
    pt = transformers.modeling_utils.PreTrainedModel
    for attr in ("_start", "_end", "_total_layers", "_text_start", "_text_end", "_text_total_layers"):
        if not hasattr(pt, attr):
            setattr(pt, attr, 0)


def _build_layer_windows(total_layers: int, window_size: int) -> List[Tuple[int, int]]:
    if total_layers <= 0:
        raise ValueError(f"Invalid total_layers={total_layers}; expected > 0.")
    if window_size <= 0:
        raise ValueError(f"Invalid window_size={window_size}; expected > 0.")
    windows: List[Tuple[int, int]] = []
    start = 0
    while start < total_layers:
        end = min(total_layers, start + window_size)
        windows.append((start, end))
        start = end
    return windows


def _get_text_layers_container(model):
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "layers"):
        return model.layers
    return None


def _null_outside_window_layers(model, *, apply_text: bool = True) -> None:
    if not apply_text:
        return
    pt = transformers.modeling_utils.PreTrainedModel
    text_start = int(getattr(pt, "_text_start", getattr(pt, "_start", 0)))
    text_end = int(getattr(pt, "_text_end", getattr(pt, "_end", 0)))
    text_layers = _get_text_layers_container(model)
    if text_layers is not None and text_end > text_start:
        for idx, _ in enumerate(text_layers):
            if idx < text_start or idx >= text_end:
                text_layers[idx] = None


def _install_window_patch(model_cls) -> None:
    if getattr(model_cls, "_window_patch_installed", False):
        return
    original_init = model_cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _null_outside_window_layers(self, apply_text=True)

    model_cls.__init__ = patched_init
    model_cls._window_patch_installed = True


def _install_shard_window_patch() -> None:
    if getattr(transformers.modeling_utils, "_window_shard_patch_installed", False):
        return

    original_get_checkpoint_shard_files = transformers.modeling_utils.get_checkpoint_shard_files

    @functools.wraps(original_get_checkpoint_shard_files)
    def patched_get_checkpoint_shard_files(*args, **kwargs):
        shard_files, metadata = original_get_checkpoint_shard_files(*args, **kwargs)
        weight_map = metadata.get("weight_map")
        if not weight_map:
            return shard_files, metadata

        pt = transformers.modeling_utils.PreTrainedModel
        start = int(getattr(pt, "_start", 0))
        end = int(getattr(pt, "_end", 0))
        text_start = int(getattr(pt, "_text_start", start))
        text_end = int(getattr(pt, "_text_end", end))
        if text_end <= text_start:
            return shard_files, metadata

        selected_text_prefixes = tuple(
            [f"model.layers.{i}." for i in range(text_start, text_end)]
            + [f"model.language_model.layers.{i}." for i in range(text_start, text_end)]
        )
        filtered_weight_map: Dict[str, str] = {}
        for checkpoint_key, shard_name in weight_map.items():
            if checkpoint_key.startswith("model.layers.") or checkpoint_key.startswith("model.language_model.layers."):
                if checkpoint_key.startswith(selected_text_prefixes):
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


def _set_layer_windows(text_start: int, text_end: int, text_total_layers: int) -> None:
    pt = transformers.modeling_utils.PreTrainedModel
    pt._start = text_start
    pt._end = text_end
    pt._total_layers = text_total_layers
    pt._text_start = text_start
    pt._text_end = text_end
    pt._text_total_layers = text_total_layers

    qeff_vl_mod = getattr(QEfficient.transformers.models, "qwen3_vl_moe", None)
    if qeff_vl_mod is not None:
        cls = getattr(qeff_vl_mod.modeling_qwen3_vl_moe, "QEffQwen3VLMoeTextModel", None)
        if cls is not None:
            cls._start = text_start
            cls._end = text_end
            cls._total_layers = text_total_layers

    qeff_35_mod = getattr(QEfficient.transformers.models, "qwen3_5_moe", None)
    if qeff_35_mod is not None:
        cls = getattr(qeff_35_mod.modeling_qwen3_5_moe, "QEffQwen3_5MoeTextModel", None)
        if cls is not None:
            cls._start = text_start
            cls._end = text_end
            cls._total_layers = text_total_layers

    qeff_3_mod = getattr(QEfficient.transformers.models, "qwen3_moe", None)
    if qeff_3_mod is not None:
        cls = getattr(qeff_3_mod.modeling_qwen3_moe, "QEffQwen3MoeModel", None)
        if cls is not None:
            cls._start = text_start
            cls._end = text_end
            cls._total_layers = text_total_layers

    QEfficient.base.modeling_qeff.QEFFBaseModel._start = text_start
    QEfficient.base.modeling_qeff.QEFFBaseModel._end = text_end
    QEfficient.base.modeling_qeff.QEFFBaseModel._total_layers = text_total_layers


def _reset_layer_windows() -> None:
    _set_layer_windows(0, 0, 0)


def _resolve_export_root(onnx_path: Path) -> Path:
    parts = list(onnx_path.parts)
    if "onnx_layerwise_tmp" in parts:
        return Path(*parts[: parts.index("onnx_layerwise_tmp")])
    return onnx_path.parent


def _stitch_layerwise_if_available(export_root: Path) -> str:
    pipeline_fn = getattr(QEfficient.utils, "layerwise_pipeline", None)
    if callable(pipeline_fn):
        return pipeline_fn(str(export_root))
    return str(export_root / "onnx_layerwise_tmp")


def _install_window_patches_for(model_type: str) -> None:
    """Install the HF __init__/shard patches needed for the given model_type."""
    _install_shard_window_patch()
    if "qwen3_vl_moe" in model_type:
        hf_mod = transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe
        _install_window_patch(hf_mod.Qwen3VLMoeForConditionalGeneration)
        if hasattr(hf_mod, "Qwen3VLMoeForCausalLM"):
            _install_window_patch(hf_mod.Qwen3VLMoeForCausalLM)


@contextmanager
def _layerwise_export_env():
    """Toggle the layerwise-active flag on QEFFBaseModel for the inner block.

    No environment variables are touched - the flag is a pure in-process class
    attribute, which keeps the API self-contained and lets concurrent Python
    interpreters (e.g. test workers) operate independently.
    """
    base = QEfficient.base.modeling_qeff.QEFFBaseModel
    prev = getattr(base, "_layerwise_active", False)
    base._layerwise_active = True
    try:
        yield
    finally:
        base._layerwise_active = prev


def _resolve_text_total_layers(config) -> int:
    text_config = getattr(config, "text_config", config)
    total = getattr(text_config, "num_hidden_layers", None)
    if total is None:
        raise ValueError("Could not resolve `num_hidden_layers` from config.text_config / config.")
    return int(total)


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def run_layerwise(
    *,
    model_id: str,
    config,
    qeff_factory,
    compile_kwargs: Dict[str, Any],
    window_size: int = 1,
    final_compile: bool = True,
) -> Any:
    """Drive the per-window export loop and (optionally) the final stitched compile.

    Parameters
    ----------
    model_id : str
        HF id / path passed to ``qeff_factory(model_id, config)`` to (re)build a
        QEff model fresh per window.
    config : transformers.PretrainedConfig
        Already-mutated config (the caller is responsible for any vision tweaks
        like ``deepstack_visual_indexes``).
    qeff_factory : Callable[[str, PretrainedConfig], QEffModel]
        Factory invoked once per window to materialize a fresh QEff model that
        only loads the active window's weights.
    compile_kwargs : dict
        Forwarded verbatim to ``qeff_model.compile(...)`` per window. The driver
        injects ``skip_lang`` per-window and ``lang_onnx_path`` for the final
        stitched compile.
    window_size : int
        Number of text-decoder layers per window. ``1`` matches the legacy
        example.
    final_compile : bool
        When True (compile path), do the final QPC compile on the merged ONNX
        and return ``qpc_paths``. When False (export path), return the merged
        ONNX path only.

    Returns
    -------
    Either the qpc_paths dict (final_compile=True) or the merged ONNX path
    (final_compile=False).
    """
    _maybe_warn_deprecation()
    model_type = assert_layerwise_supported(config)

    text_total_layers = _resolve_text_total_layers(config)
    text_cfg = getattr(config, "text_config", config)
    text_cfg.num_hidden_layers = text_total_layers

    _ensure_pretrained_window_attrs()
    _install_window_patches_for(model_type)

    windows = _build_layer_windows(text_total_layers, window_size)
    first_onnx_path: Optional[Path] = None
    last_qeff_model = None

    with _layerwise_export_env():
        for window_idx, (text_start, text_end) in enumerate(windows):
            _set_layer_windows(text_start, text_end, text_total_layers)

            qeff_model = qeff_factory(model_id, config)
            last_qeff_model = qeff_model
            if hasattr(qeff_model, "model"):
                _null_outside_window_layers(qeff_model.model, apply_text=True)

            window_kwargs = dict(compile_kwargs)
            # skip_lang is a VLM-only kwarg; only inject when present in caller's kwargs.
            if "skip_lang" in window_kwargs:
                window_kwargs["skip_lang"] = False
            onnx_path = qeff_model.compile(**window_kwargs)
            if first_onnx_path is None:
                if isinstance(onnx_path, dict):
                    lang_key = next(
                        (
                            k
                            for k in (
                                "lang_decode_qpc_path",
                                "lang_prefill_qpc_path",
                                "lang_qpc_path",
                            )
                            if k in onnx_path
                        ),
                        None,
                    )
                    if lang_key is None:
                        raise RuntimeError(f"Layer-wise window produced no lang_*_qpc_path: keys={list(onnx_path)}")
                    lang_path = onnx_path[lang_key]
                else:
                    lang_path = onnx_path
                first_onnx_path = Path(str(lang_path))

    if first_onnx_path is None:
        raise RuntimeError("Layer-wise export produced no ONNX shards.")

    export_root = _resolve_export_root(first_onnx_path)
    final_artifact = _stitch_layerwise_if_available(export_root)

    _reset_layer_windows()

    if not final_compile:
        return final_artifact

    final_kwargs = dict(compile_kwargs)
    final_kwargs["lang_onnx_path"] = final_artifact
    final_kwargs.setdefault("skip_lang", False)
    return last_qeff_model.compile(**final_kwargs)
