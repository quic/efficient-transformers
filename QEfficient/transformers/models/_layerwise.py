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
windowing hooks that this driver relies on; calling :func:`run_layerwise` for
any other architecture will raise.
"""

from __future__ import annotations

import functools
import gc
import random
import shutil
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
import transformers

import QEfficient
from QEfficient.transformers.models.custom_loader import CustomLoader
from QEfficient.utils.logging_utils import logger

# Architectures whose modeling files declare layerwise window hooks. Keep this
# list narrow on purpose - adding a new
# architecture must come with the corresponding modeling-file hooks.
_LAYERWISE_SUPPORTED_MODEL_TYPES = frozenset(
    {
        "qwen3_vl_moe",
        "qwen3_vl_moe_text",
        "qwen3_5_moe",
        "qwen3_moe",
    }
)

# Hard cap on the RoPE base table rows we serialize per window. Chosen as a
# constant (not a function of ctx_len) on purpose: changing ctx_len at compile
# time should re-use the cached ONNX and only re-run the QPC compile. Any
# inference-time position id is bounded by ctx_len, and ctx_len in practice
# stays well under 32K for the supported MoE families today, so dropping
# the unreachable rows past 32K is lossless.
_LAYERWISE_ROPE_MAX_POSITIONS = 76800

_LAYERWISE_CONTEXT_ATTR = "_qeff_layerwise_context"

_DEPRECATION_WARNED = False


@dataclass
class LayerwiseContext:
    """Per-model layerwise state attached to QEff model instances."""

    model_id: str
    window_size: int
    model_type: str
    total_layers: int
    load_kwargs: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    active: bool = False
    start: int = 0
    end: int = 0
    force_full_init: bool = False

    def __post_init__(self) -> None:
        self.window_size = _validate_layerwise_window_size(self.window_size)
        self.total_layers = int(self.total_layers)

    @property
    def is_first_window(self) -> bool:
        return self.active and self.start == 0

    @property
    def is_last_window(self) -> bool:
        return self.active and self.end >= self.total_layers

    def build_windows(self) -> List[Tuple[int, int]]:
        return _build_layer_windows(self.total_layers, self.window_size)

    def set_window(self, start: int, end: int) -> None:
        self.start = int(start)
        self.end = int(end)
        self.active = True

    def reset_window(self) -> None:
        self.start = 0
        self.end = 0
        self.active = False
        self.force_full_init = False


def _validate_layerwise_window_size(window_size: int) -> int:
    if window_size is None:
        raise ValueError(
            "layerwise=True requires `layerwise_window_size` to be a positive integer. "
            "Got None. Pass an explicit value (for example, layerwise_window_size=1)."
        )
    if isinstance(window_size, bool) or not isinstance(window_size, Integral):
        raise TypeError(
            "layerwise=True requires `layerwise_window_size` to be a positive integer. "
            f"Got {type(window_size).__name__}: {window_size!r}."
        )
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError(
            f"layerwise=True requires `layerwise_window_size` to be a positive integer. Got {window_size}."
        )
    return window_size


def create_layerwise_context(
    *,
    model_id: str,
    config,
    window_size: int = 1,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> LayerwiseContext:
    """Build a validated layerwise context from a model id/config pair."""

    model_type = assert_layerwise_supported(config)
    return LayerwiseContext(
        model_id=model_id,
        window_size=window_size,
        model_type=model_type,
        total_layers=_resolve_text_total_layers(config),
        load_kwargs=dict(load_kwargs or {}),
    )


def attach_layerwise_context(module, context: LayerwiseContext, *, recursive: bool = True) -> None:
    """Attach layerwise context to a module, optionally to every child module."""

    if module is None:
        return
    if recursive and hasattr(module, "modules"):
        for submodule in module.modules():
            setattr(submodule, _LAYERWISE_CONTEXT_ATTR, context)
        return
    try:
        setattr(module, _LAYERWISE_CONTEXT_ATTR, context)
    except AttributeError:
        return


def _attach_context_to_qeff_model(qeff_model, context: LayerwiseContext) -> None:
    """Attach context to common QEff wrapper layouts."""

    if qeff_model is None:
        return
    if hasattr(qeff_model, "lang_model") and hasattr(qeff_model.lang_model, "model"):
        attach_layerwise_context(qeff_model.lang_model.model, context)
    elif hasattr(qeff_model, "model"):
        attach_layerwise_context(qeff_model.model, context)
    else:
        attach_layerwise_context(qeff_model, context)


def get_layerwise_context(module=None) -> Optional[LayerwiseContext]:
    """Return the instance-attached layerwise context, if any."""

    if isinstance(module, LayerwiseContext):
        return module if module.enabled else None
    context = getattr(module, _LAYERWISE_CONTEXT_ATTR, None) if module is not None else None
    if context is not None and context.enabled:
        return context
    return None


def has_layerwise_context(module) -> bool:
    context = getattr(module, _LAYERWISE_CONTEXT_ATTR, None) if module is not None else None
    return bool(context is not None and context.enabled)


@contextmanager
def suspend_layerwise_context(module):
    """Temporarily disable layerwise routing for a wrapper/module tree."""

    context = get_layerwise_context(module)
    if context is None:
        yield
        return

    previous_enabled = context.enabled
    previous_active = context.active
    context.enabled = False
    context.active = False
    try:
        yield
    finally:
        context.enabled = previous_enabled
        context.active = previous_active


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


def is_layerwise_active(module=None) -> bool:
    """True only while the layer-wise export driver is running.

    The driver flips this on inside :func:`_layerwise_export_env`. Outside that
    scope (the default, non-layerwise path) it is always False, which lets the
    modeling forwards short-circuit every window branch and behave exactly as
    they did before layer-wise support was added.
    """
    context = get_layerwise_context(module)
    if context is not None:
        return bool(context.enabled and context.active)
    return False


def resolve_layer_window(module_or_cls=None, total_layers: int = 0) -> Tuple[int, int]:
    """Return the ``[start, end)`` decoder-layer window to run this forward.

    When layer-wise export is inactive this always returns ``(0, total_layers)``.
    When the driver is active it honors the attached context window.
    """
    context = get_layerwise_context(module_or_cls)
    if context is not None:
        if not context.active:
            return 0, total_layers
        end = context.end if context.end else total_layers
        return int(context.start), int(end)
    return 0, total_layers


def is_last_layer_window(module_or_cls=None, total_layers: int = 0) -> bool:
    """True if this forward owns the final decoder window (applies final norm / lm_head).

    Always True on the default path; on the layer-wise path it is True only for
    the window whose end reaches the total layer count.
    """
    context = get_layerwise_context(module_or_cls)
    if context is not None:
        return True if not context.active else context.end >= total_layers
    return True


def get_layerwise_start(module=None) -> int:
    context = get_layerwise_context(module)
    if context is not None and context.active:
        return int(context.start)
    return 0


def get_layerwise_end(module=None) -> int:
    context = get_layerwise_context(module)
    if context is not None and context.active:
        return int(context.end)
    return 0


def get_layerwise_total_layers(module=None) -> int:
    context = get_layerwise_context(module)
    if context is not None:
        return int(context.total_layers)
    return 0


# ---------------------------------------------------------------------------
# Internal helpers (lifted from the legacy example script)
# ---------------------------------------------------------------------------


def _build_layer_windows(total_layers: int, window_size: int) -> List[Tuple[int, int]]:
    if total_layers <= 0:
        raise ValueError(f"Invalid total_layers={total_layers}; expected > 0.")
    window_size = _validate_layerwise_window_size(window_size)
    windows: List[Tuple[int, int]] = []
    start = 0
    while start < total_layers:
        end = min(total_layers, start + window_size)
        windows.append((start, end))
        start = end
    return windows


@functools.lru_cache(maxsize=32)
def _checkpoint_has_shard_index(model_id: str) -> bool:
    """Return True when checkpoint provides an index file for shard filtering."""
    hub_kwargs = {"_raise_exceptions_for_missing_entries": False}
    for index_filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            index_path = transformers.utils.hub.cached_file(model_id, index_filename, **hub_kwargs)
        except Exception:
            index_path = None
        if index_path is not None:
            return True
    return False


def _active_window_bounds(module=None, context: Optional[LayerwiseContext] = None) -> Tuple[int, int]:
    context = context or get_layerwise_context(module)
    if context is not None and context.active:
        return int(context.start), int(context.end)
    return 0, 0


def _find_language_model(model):
    """Locate the inner language_model that owns sin_cached / cos_cached / embed_tokens."""
    candidates = []
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        candidates.append(model.model.language_model)
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
    if hasattr(model, "model"):
        candidates.append(model.model)
    candidates.append(model)
    for cand in candidates:
        if any(hasattr(cand, attr) for attr in ("sin_cached", "cos_cached", "embed_tokens")):
            return cand
    return None


def _slim_for_window_export(qeff_model, *, ctx_len: Optional[int]) -> None:
    """Shrink top-level params that are unused (or oversized) for this window.

    Without this, every per-window ONNX shard re-bakes the full RoPE base table
    (``sin_cached``/``cos_cached`` of shape ``[max_position_embeddings, head_dim]``,
    typically tens of MB in fp32) plus the full vocab embedding, blowing each
    layer-window shard up by 1-2 orders of magnitude over the actual layer
    weight footprint. Each top-level param is touched in-place; the next
    window rebuilds the model from scratch via the factory so there is no
    leakage across windows.
    """
    import torch

    inner = qeff_model.model if hasattr(qeff_model, "model") else qeff_model
    lm = _find_language_model(inner)
    if lm is None:
        return

    text_start, text_end = _active_window_bounds(inner)
    context = get_layerwise_context(inner)
    if context is not None:
        text_total = int(context.total_layers)
    else:
        text_total = 0

    # 1) Truncate sin_cached / cos_cached to a fixed cap (32K rows) instead
    #    of ctx_len. A constant cap keeps the export hash invariant when the
    #    user changes ctx_len, so re-compiling at a different context length
    #    only re-runs the QPC compile - the cached layer-wise ONNX is reused.
    #    Inference-time position ids are bounded by ctx_len which is well
    #    under the cap for the supported MoE families today.
    del ctx_len  # signature retained for forward compat, value intentionally unused
    rope_cap = _LAYERWISE_ROPE_MAX_POSITIONS
    for attr in ("sin_cached", "cos_cached"):
        param = getattr(lm, attr, None)
        if param is None or not hasattr(param, "shape") or param.dim() < 2:
            continue
        cur_rows = int(param.shape[0])
        if cur_rows <= rope_cap:
            continue
        with torch.no_grad():
            truncated = param.detach()[:rope_cap].clone().contiguous()
        new_param = torch.nn.Parameter(truncated, requires_grad=False)
        setattr(lm, attr, new_param)

    # 2) Drop the vocab embedding for windows that don't run input-id lookup.
    #    The first window (text_start == 0) is the only one that calls
    #    ``get_input_embeddings()(input_ids)``; later windows take
    #    ``inputs_embeds`` directly so the embedding matrix is unreached but
    #    still serialized. Replace its weight with a tiny placeholder of the
    #    same dtype/device so module attributes stay valid.
    if text_start > 0 and hasattr(lm, "embed_tokens"):
        embed = getattr(lm, "embed_tokens", None)
        weight = getattr(embed, "weight", None)
        if weight is not None and weight.dim() == 2 and weight.shape[0] > 1:
            with torch.no_grad():
                tiny = torch.zeros((1, weight.shape[1]), dtype=weight.dtype, device=weight.device)
            embed.weight = torch.nn.Parameter(tiny, requires_grad=False)

    # 3) Drop the lm_head for windows that aren't the last one. Only the final
    #    window applies ``self.model.lm_head(hidden_states)``.
    outer = qeff_model.model if hasattr(qeff_model, "model") else qeff_model
    lm_head = getattr(outer, "lm_head", None)
    if (
        lm_head is not None
        and text_total > 0
        and text_end < text_total
        and hasattr(lm_head, "weight")
        and lm_head.weight is not None
        and lm_head.weight.dim() == 2
        and lm_head.weight.shape[0] > 1
    ):
        with torch.no_grad():
            tiny = torch.zeros(
                (1, lm_head.weight.shape[1]),
                dtype=lm_head.weight.dtype,
                device=lm_head.weight.device,
            )
        lm_head.weight = torch.nn.Parameter(tiny, requires_grad=False)
        if getattr(lm_head, "bias", None) is not None:
            lm_head.bias = torch.nn.Parameter(
                torch.zeros((1,), dtype=lm_head.bias.dtype, device=lm_head.bias.device),
                requires_grad=False,
            )


def _set_layer_windows(
    text_start: int,
    text_end: int,
    text_total_layers: int,
    context: LayerwiseContext,
) -> None:
    del text_total_layers
    if context is None:
        raise ValueError("_set_layer_windows requires a LayerwiseContext.")
    context.force_full_init = False
    if text_end > text_start:
        context.set_window(text_start, text_end)
    else:
        context.reset_window()


def _reset_layer_windows(context: LayerwiseContext) -> None:
    if context is None:
        raise ValueError("_reset_layer_windows requires a LayerwiseContext.")
    context.reset_window()


def _resolve_export_root(onnx_path: Path) -> Path:
    parts = list(onnx_path.parts)
    if "onnx_layerwise_tmp" in parts:
        return Path(*parts[: parts.index("onnx_layerwise_tmp")])
    if "final_data" in parts:
        return Path(*parts[: parts.index("final_data")])
    return onnx_path.parent


def _is_cached_merged(onnx_path: Path) -> bool:
    """Return True for layerwise-stitched ONNX files produced by the merge pipeline."""
    return onnx_path.name.startswith("merged_")


def _cached_merged_onnx(export_root: Path, total_layers: Optional[int] = None) -> Optional[Path]:
    """Return the complete cached merged ONNX, if it exists."""
    search_dirs = [export_root]
    cached_dir = export_root / "final_data"
    if cached_dir.is_dir():
        search_dirs.append(cached_dir)

    if total_layers is not None:
        for search_dir in search_dirs:
            expected = search_dir / f"merged_0-{total_layers}.onnx"
            if expected.is_file():
                return expected

    cached_merged = [path for search_dir in search_dirs for path in search_dir.glob("merged_0-*.onnx")]
    cached_merged = sorted(cached_merged)
    return cached_merged[-1] if cached_merged else None


def _cached_root_onnx(export_root: Path, model_name: Optional[str]) -> Optional[Path]:
    """Return a canonical non-merged ONNX in the hash root, if present."""
    if model_name is None:
        return None
    candidate = export_root / f"{model_name}.onnx"
    return candidate if candidate.is_file() else None


def _resolve_layerwise_model_name(qeff_model) -> Optional[str]:
    """Resolve the decoder model name used for canonical ONNX cache lookup."""
    lang_model = getattr(qeff_model, "lang_model", None)
    if lang_model is not None and hasattr(lang_model, "model_name"):
        return lang_model.model_name
    return getattr(qeff_model, "model_name", None)


def _relocate_merged_onnx_to_root(export_root: Path, merged_onnx: Path) -> Path:
    """Move a merged ONNX and its external tensor blobs into the hash root.

    The move keeps the layerwise cache layout aligned with regular export and
    avoids copying large external data files, which can temporarily double disk
    usage for large models.
    """
    if merged_onnx.parent == export_root:
        return merged_onnx

    model = onnx.load(str(merged_onnx), load_external_data=False)
    external_files = set()
    for tensor in model.graph.initializer:
        if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location":
                    external_files.add(entry.value)

    src_dir = merged_onnx.parent
    for rel_name in external_files:
        src = src_dir / rel_name
        dst = export_root / rel_name
        if src.exists() and src != dst and not dst.exists():
            shutil.move(str(src), str(dst))

    dst_onnx = export_root / merged_onnx.name
    if merged_onnx != dst_onnx:
        shutil.move(str(merged_onnx), str(dst_onnx))
    return dst_onnx


def _cleanup_layerwise_intermediates(export_root: Path, *, keep_onnx: Optional[Path] = None) -> None:
    """Remove transient layerwise export directories after final ONNX relocation."""
    for dirname in ("final_data", "onnx_layerwise_tmp"):
        path = export_root / dirname
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    for pattern in ("pref_*.onnx",):
        for path in export_root.glob(pattern):
            if keep_onnx is not None and path == keep_onnx:
                continue
            path.unlink(missing_ok=True)


def _stitch_layerwise_if_available(export_root: Path, total_layers: Optional[int] = None) -> str:
    # If a prior run already produced the complete merged ONNX, just return it.
    cached_merged = _cached_merged_onnx(export_root, total_layers)
    if cached_merged is not None:
        return str(cached_merged)
    pipeline_fn = getattr(QEfficient.utils, "layerwise_pipeline", None)
    if callable(pipeline_fn):
        return pipeline_fn(str(export_root), final_data_dir=".")
    return str(export_root / "onnx_layerwise_tmp")


def _cached_layerwise_onnx_path(qeff_model, compile_kwargs: Dict[str, Any]) -> Optional[Path]:
    """Return a cached merged ONNX path without exporting or loading weights."""
    probe_kwargs = dict(compile_kwargs)
    probe_kwargs["_layerwise_cache_probe"] = True
    cached = qeff_model.compile(**probe_kwargs)
    if isinstance(cached, dict):
        cached = next(
            (
                cached.get(key)
                for key in ("lang_decode_qpc_path", "lang_prefill_qpc_path", "lang_qpc_path")
                if cached.get(key) is not None
            ),
            None,
        )
    return Path(cached) if cached is not None else None


@contextmanager
def _layerwise_export_env(context: LayerwiseContext):
    """Preserve a LayerwiseContext's mutable export-window fields."""

    if context is None:
        raise ValueError("Layerwise export requires a LayerwiseContext.")
    prev_context_state = (
        context.active,
        context.start,
        context.end,
        context.force_full_init,
    )
    try:
        yield
    finally:
        context.active, context.start, context.end, context.force_full_init = prev_context_state


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
    probe_qeff_model=None,
    window_size: int = 1,
    final_compile: bool = True,
    context: Optional[LayerwiseContext] = None,
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
    probe_qeff_model : QEffModel, optional
        Existing wrapper used for cache probing before any per-window weights are
        loaded. In normal layerwise use this is the outer meta wrapper.
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
    model_type = assert_layerwise_supported(config)

    if context is None:
        _maybe_warn_deprecation()
        context = create_layerwise_context(
            model_id=model_id,
            config=config,
            window_size=window_size,
        )
    elif not context.enabled:
        raise ValueError("Layerwise context is disabled.")

    text_total_layers = int(context.total_layers)
    window_size = int(context.window_size)
    text_cfg = getattr(config, "text_config", config)
    text_cfg.num_hidden_layers = text_total_layers

    windows = context.build_windows()
    logger.info(
        "Layerwise export start: model_type=%s total_layers=%d window_size=%d windows=%d "
        "torch_threads=%d torch_interop_threads=%d",
        model_type,
        text_total_layers,
        window_size,
        len(windows),
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    )
    if torch.get_num_threads() <= 1:
        logger.warning(
            "Layerwise export is running with torch.get_num_threads()=%d; ONNX export may appear single-core. "
            "For faster export, increase PyTorch/OMP threads (for example, TORCH_NUM_THREADS or OMP_NUM_THREADS).",
            torch.get_num_threads(),
        )
    first_onnx_path: Optional[Path] = None
    last_qeff_model = None
    needs_deterministic_init = model_type in {"qwen3_5_moe", "qwen3_5_moe_text"} and not _checkpoint_has_shard_index(
        model_id
    )
    if needs_deterministic_init:
        init_rng_snapshot = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        }

        def _build_window_model():
            # Tiny/test checkpoints can miss some tensors and rely on random
            # initialization. Reusing one RNG snapshot per window preserves
            # layerwise-vs-default parity for those models.
            prev_python_state = random.getstate()
            prev_numpy_state = np.random.get_state()
            prev_torch_state = torch.random.get_rng_state()
            try:
                random.setstate(init_rng_snapshot["python"])
                np.random.set_state(init_rng_snapshot["numpy"])
                torch.random.set_rng_state(init_rng_snapshot["torch"])
                loader = CustomLoader(
                    model_id, layer_indices=range(context.start, context.end), load_kwargs=context.load_kwargs
                )
                qeff_model = loader.load_with(qeff_factory, model_id, config)
                _attach_context_to_qeff_model(qeff_model, context)
                return qeff_model
            finally:
                random.setstate(prev_python_state)
                np.random.set_state(prev_numpy_state)
                torch.random.set_rng_state(prev_torch_state)

    else:

        def _build_window_model():
            loader = CustomLoader(
                model_id, layer_indices=range(context.start, context.end), load_kwargs=context.load_kwargs
            )
            qeff_model = loader.load_with(qeff_factory, model_id, config)
            _attach_context_to_qeff_model(qeff_model, context)
            return qeff_model

    try:
        with _layerwise_export_env(context):
            _set_layer_windows(0, min(window_size, text_total_layers), text_total_layers, context=context)
            if needs_deterministic_init:
                context.force_full_init = True
            cached_probe = probe_qeff_model or _build_window_model()
            cached_onnx_path = _cached_layerwise_onnx_path(cached_probe, compile_kwargs)
            if cached_onnx_path is not None:
                logger.info("Layerwise cache hit: reusing merged ONNX at %s", cached_onnx_path)
                first_onnx_path = cached_onnx_path
                last_qeff_model = cached_probe

            for text_start, text_end in windows:
                if cached_onnx_path is not None:
                    break
                window_t0 = time.perf_counter()
                logger.info("Layerwise window export start: [%d, %d)", text_start, text_end)
                _set_layer_windows(text_start, text_end, text_total_layers, context=context)
                if needs_deterministic_init:
                    context.force_full_init = True

                qeff_model = _build_window_model()
                last_qeff_model = qeff_model
                _slim_for_window_export(qeff_model, ctx_len=compile_kwargs.get("ctx_len"))

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
                logger.info(
                    "Layerwise window export done: [%d, %d) in %.2fs",
                    text_start,
                    text_end,
                    time.perf_counter() - window_t0,
                )
                if hasattr(qeff_model, "model"):
                    CustomLoader.to_meta(qeff_model.model)
                del qeff_model
                gc.collect()
    finally:
        _reset_layer_windows(context=context)

    if first_onnx_path is None:
        raise RuntimeError("Layer-wise export produced no ONNX shards.")

    export_root = _resolve_export_root(first_onnx_path)
    model_name = _resolve_layerwise_model_name(last_qeff_model)
    root_cached_onnx = _cached_root_onnx(export_root, model_name)
    if root_cached_onnx is not None:
        canonical_onnx = root_cached_onnx
    elif cached_onnx_path is not None:
        cached_path = Path(cached_onnx_path)
        canonical_onnx = (
            _relocate_merged_onnx_to_root(export_root, cached_path) if _is_cached_merged(cached_path) else cached_path
        )
    else:
        final_artifact = _stitch_layerwise_if_available(export_root, text_total_layers)
        logger.info("Layerwise merged ONNX: %s", final_artifact)
        canonical_onnx = Path(final_artifact)
    if canonical_onnx.parent == export_root:
        _cleanup_layerwise_intermediates(export_root, keep_onnx=canonical_onnx)
    logger.info("Layerwise canonical ONNX: %s", canonical_onnx)

    if not final_compile:
        return str(canonical_onnx)

    final_kwargs = dict(compile_kwargs)
    if hasattr(last_qeff_model, "lang_model"):
        final_kwargs["lang_onnx_path"] = str(canonical_onnx)
        context_owner = getattr(last_qeff_model.lang_model, "model", last_qeff_model.lang_model)
    else:
        final_kwargs["onnx_path"] = str(canonical_onnx)
        context_owner = getattr(last_qeff_model, "model", last_qeff_model)
    final_kwargs.setdefault("skip_lang", False)
    with suspend_layerwise_context(context_owner):
        return last_qeff_model.compile(**final_kwargs)
