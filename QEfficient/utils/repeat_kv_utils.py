# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional, Sequence

import torch.nn as nn

from QEfficient.transformers.models.repeat_kv_projection_dispatch import duplicate_kv_projection_weights_dispatch
from QEfficient.utils.config_utils import resolve_attention_heads, resolve_hidden_size, resolve_kv_heads

TEXT_MODEL_CANDIDATE_PATHS = (
    ("language_model",),
    ("language_model", "model"),
    ("model", "language_model"),
    ("model", "language_model", "model"),
    ("model", "model", "language_model"),
    ("model", "model", "language_model", "model"),
    ("model",),
    ("model", "model"),
    ("transformer",),
    ("transformer", "model"),
    ("llm",),
    ("llm", "model"),
    ("backbone",),
)


def duplicate_kv_projection_weights(
    layer: nn.Module,
    orig_kv_heads: int,
    repeat: int,
    head_dim: int,
    hidden_size: int,
    layer_name: Optional[str] = None,
) -> None:
    """
    Duplicate KV projection weights for one projection layer to implement repeat-kv transform.
    """
    duplicate_kv_projection_weights_dispatch(
        layer=layer,
        orig_kv_heads=orig_kv_heads,
        repeat=repeat,
        head_dim=head_dim,
        hidden_size=hidden_size,
        layer_name=layer_name,
    )


def get_attention_module(block: nn.Module) -> nn.Module:
    for attr in ("cross_attn", "self_attn", "attention", "attn"):
        attn = getattr(block, attr, None)
        if attn is not None:
            return attn
    raise AttributeError(f"No attention module found in block type {block.__class__.__name__}")


def get_projection_layer(attn: nn.Module, names: Sequence[str]) -> nn.Module:
    for name in names:
        layer = getattr(attn, name, None)
        if layer is not None:
            return layer
    raise AttributeError(f"Missing projection layer in {attn.__class__.__name__}; expected one of {tuple(names)}")


def is_mla_attention(attn: nn.Module) -> bool:
    return hasattr(attn, "kv_a_proj_with_mqa") and hasattr(attn, "kv_lora_rank") and hasattr(attn, "qk_rope_head_dim")


def is_mla_model(text_model: nn.Module) -> bool:
    for block in getattr(text_model, "layers", []):
        try:
            attn = get_attention_module(block)
        except AttributeError:
            continue
        if is_mla_attention(attn):
            return True
    return False


def is_valid_text_model(candidate: nn.Module) -> bool:
    if candidate is None:
        return False
    cfg = getattr(candidate, "config", None)
    layers = getattr(candidate, "layers", None)
    attn_heads = resolve_attention_heads(cfg) if cfg is not None else None
    kv_heads = resolve_kv_heads(cfg) if cfg is not None else None
    hidden_size = resolve_hidden_size(cfg) if cfg is not None else None
    return (
        cfg is not None
        and layers is not None
        and attn_heads is not None
        and kv_heads is not None
        and hidden_size is not None
    )


def get_text_model(model: nn.Module) -> nn.Module:
    for path in TEXT_MODEL_CANDIDATE_PATHS:
        candidate = model
        valid_path = True
        for attr in path:
            if not hasattr(candidate, attr):
                valid_path = False
                break
            candidate = getattr(candidate, attr)
        if valid_path and is_valid_text_model(candidate):
            return candidate

    raise AttributeError(
        f"No suitable text model found in the provided model ({model.__class__.__name__}). "
        "Expected a module with `layers` and text `config` attributes."
    )
