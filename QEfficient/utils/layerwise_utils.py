# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Helpers for the generic layer-wise ONNX export flow.

These utilities keep model-specific resolution logic (which submodule holds the
repeated decoder layers, what the checkpoint key prefix is, how window state is
threaded into the forward) out of the model wrapper classes.
"""

from typing import List, Tuple

import torch.nn as nn


def build_layer_windows(total_layers: int, window_size: int) -> List[Tuple[int, int]]:
    """Tile ``[0, total_layers)`` into descending ``(start, end)`` windows.

    Matches the tiling used by the original layer-wise example scripts: windows
    are produced from the last layer towards layer 0.
    """
    if total_layers <= 0:
        raise ValueError(f"Invalid total_layers={total_layers}. Expected total_layers > 0.")
    if window_size <= 0:
        raise ValueError(f"Invalid window_size={window_size}. Expected window_size > 0.")

    windows: List[Tuple[int, int]] = []
    end = total_layers
    while end > 0:
        start = max(0, end - window_size)
        windows.append((start, end))
        end = start
    return windows


def _find_layers_container(module: nn.Module):
    """Return the submodule that owns ``.layers`` (the repeated decoder stack).

    Tries the common nesting patterns used across CausalLM and VLM language
    models. Returns ``(owner_module, layer_prefix)`` where ``layer_prefix`` is
    the state-dict key prefix for the repeated layers, or ``(None, None)``.
    """
    candidates = [
        ("model.language_model", getattr(getattr(module, "model", None), "language_model", None)),
        ("language_model", getattr(module, "language_model", None)),
        ("model", getattr(module, "model", None)),
        ("", module),
    ]
    for prefix, owner in candidates:
        if owner is not None and hasattr(owner, "layers") and isinstance(owner.layers, nn.ModuleList):
            layer_prefix = f"{prefix}.layers." if prefix else "layers."
            return owner, layer_prefix
    return None, None


def resolve_text_model(model: nn.Module):
    """Resolve the text/decoder model that carries the repeated layers.

    Returns ``(text_model_module, layer_prefix)``. ``text_model_module`` is the
    module whose class holds the ``_start/_end/_total_layers`` window contract
    (i.e. the module that owns ``.layers``).
    """
    owner, layer_prefix = _find_layers_container(model)
    if owner is None:
        raise RuntimeError(
            "Could not locate the repeated decoder-layer container (`.layers`) on the model; "
            "layer-wise export is not supported for this architecture."
        )
    return owner, layer_prefix


def set_window_state(text_model: nn.Module, start: int, end: int, total_layers: int) -> None:
    """Set the layer-window state on the text-model class.

    The model ``forward`` reads ``_start/_end/_total_layers`` as class
    attributes, so these are set on ``type(text_model)`` (centralized here
    instead of monkeypatching from example scripts).

    The same window is mirrored onto :class:`QEFFBaseModel` because the shared
    ``_export_layerwise`` routine reads the window from there.
    """
    cls = type(text_model)
    cls._start = int(start)
    cls._end = int(end)
    cls._total_layers = int(total_layers)

    from QEfficient.base.modeling_qeff import QEFFBaseModel

    QEFFBaseModel._start = int(start)
    QEFFBaseModel._end = int(end)
    QEFFBaseModel._total_layers = int(total_layers)


def reset_window_state(text_model: nn.Module, total_layers: int) -> None:
    """Reset window state to cover the full model (``[0, total_layers)``)."""
    set_window_state(text_model, 0, total_layers, total_layers)


def build_meta_model(hf_auto_class, pretrained_model_name_or_path: str, **kwargs):
    """Instantiate a model on the ``meta`` device from its config only.

    Used for layer-wise export where weights are streamed per window instead of
    loaded up front. ``kwargs`` mirror those passed to ``from_pretrained``
    (e.g. ``torch_dtype``, ``attn_implementation``); only config-relevant ones
    are forwarded to ``from_config``.
    """
    import torch
    from transformers import AutoConfig

    config = kwargs.pop("config", None)
    if config is None:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=kwargs.get("trust_remote_code", False))

    torch_dtype = kwargs.get("torch_dtype", torch.float32)
    if torch_dtype is not None:
        config.torch_dtype = torch_dtype
    config.use_cache = True

    from_config_kwargs = {}
    if "attn_implementation" in kwargs:
        from_config_kwargs["attn_implementation"] = kwargs["attn_implementation"]
    if "trust_remote_code" in kwargs:
        from_config_kwargs["trust_remote_code"] = kwargs["trust_remote_code"]

    with torch.device("meta"):
        model = hf_auto_class.from_config(config, **from_config_kwargs)
    return model
