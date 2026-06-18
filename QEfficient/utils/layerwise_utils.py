# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Instance-scoped helpers for layer-wise export."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Integral
from typing import List, Optional, Tuple

import torch


_LAYERWISE_EXPORT_ACTIVE: ContextVar[bool] = ContextVar("qeff_layerwise_export_active", default=False)


@dataclass(frozen=True)
class LayerwiseContext:
    """Active decoder-layer window for a single layerwise export step."""

    start: int
    end: int
    total_layers: int
    active: bool = True
    force_full_init: bool = False


def get_layerwise_context(module) -> Optional[LayerwiseContext]:
    return getattr(module, "_qeff_layerwise_context", None)


def is_layerwise_active(module=None) -> bool:
    if module is None:
        return False
    context = get_layerwise_context(module)
    return bool(context and context.active)


def is_layerwise_export_active() -> bool:
    return bool(_LAYERWISE_EXPORT_ACTIVE.get())


@contextmanager
def layerwise_export_scope():
    token = _LAYERWISE_EXPORT_ACTIVE.set(True)
    try:
        yield
    finally:
        _LAYERWISE_EXPORT_ACTIVE.reset(token)


def resolve_layer_window(module, total_layers: int) -> Tuple[int, int]:
    context = get_layerwise_context(module)
    if not context or not context.active:
        return 0, total_layers
    end = context.end if context.end else total_layers
    return int(context.start), int(end)


def is_last_layer_window(module, total_layers: int) -> bool:
    _, end = resolve_layer_window(module, total_layers)
    return end >= total_layers


def build_layer_windows(total_layers: int, window_size: int) -> List[Tuple[int, int]]:
    if total_layers <= 0:
        raise ValueError(f"Invalid total_layers={total_layers}; expected > 0.")
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
    windows: List[Tuple[int, int]] = []
    start = 0
    while start < total_layers:
        end = min(total_layers, start + window_size)
        windows.append((start, end))
        start = end
    return windows


def resolve_text_total_layers(config) -> int:
    text_config = getattr(config, "text_config", config)
    total = getattr(text_config, "num_hidden_layers", None)
    if total is None:
        raise ValueError("Could not resolve `num_hidden_layers` from config.text_config / config.")
    return int(total)


def build_meta_model(hf_auto_class, pretrained_model_name_or_path: str, kwargs: dict):
    """Construct an HF model on meta using the effective Transformers config."""
    from transformers import AutoConfig

    config = kwargs.get("config", None)
    if config is None:
        config_kwargs = {
            k: kwargs[k] for k in ("trust_remote_code", "revision", "token", "subfolder", "cache_dir") if k in kwargs
        }
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
        if "num_hidden_layers" in kwargs:
            target_config = getattr(config, "text_config", config)
            target_config.num_hidden_layers = kwargs["num_hidden_layers"]
        kwargs["config"] = config
    torch_dtype = kwargs.get("torch_dtype", torch.float32)
    with torch.device("meta"):
        model = hf_auto_class.from_config(config, torch_dtype=torch_dtype)
    return model


def attach_layerwise_context(module, context: LayerwiseContext) -> None:
    if module is None:
        return
    setattr(module, "_qeff_layerwise_context", context)
    modules = module.modules() if hasattr(module, "modules") else []
    for child in modules:
        setattr(child, "_qeff_layerwise_context", context)


def clear_layerwise_context(module) -> None:
    if module is None:
        return
    if hasattr(module, "_qeff_layerwise_context"):
        delattr(module, "_qeff_layerwise_context")
    modules = module.modules() if hasattr(module, "modules") else []
    for child in modules:
        if hasattr(child, "_qeff_layerwise_context"):
            delattr(child, "_qeff_layerwise_context")


@contextmanager
def layerwise_context(module, start: int, end: int, total_layers: int, *, force_full_init: bool = False):
    context = LayerwiseContext(start=start, end=end, total_layers=total_layers, force_full_init=force_full_init)
    attach_layerwise_context(module, context)
    try:
        yield context
    finally:
        clear_layerwise_context(module)
