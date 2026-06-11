# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional, Sequence

import torch
import torch.nn as nn

from QEfficient.customop.matmulnbits import QuantLinearORT, dequantize_blockwise_bits
from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ
from QEfficient.transformers.quantizers.quantizer_compressed_tensors import FP8DeQuantLinear
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
    layer_prefix = f"{layer_name}: " if layer_name else ""
    new_kv_heads = repeat * orig_kv_heads

    if isinstance(layer, WQLinear_GEMM):
        if layer.qweight.shape[1] % orig_kv_heads != 0:
            raise ValueError(
                f"{layer_prefix}Invalid AWQ qweight shape for RepeatKV: qweight.shape={tuple(layer.qweight.shape)}, "
                f"orig_kv_heads={orig_kv_heads}"
            )
        if layer.qzeros.shape[1] % orig_kv_heads != 0 or layer.scales.shape[1] % orig_kv_heads != 0:
            raise ValueError(
                f"{layer_prefix}Invalid AWQ qzeros/scales shape for RepeatKV: qzeros.shape={tuple(layer.qzeros.shape)}, "
                f"scales.shape={tuple(layer.scales.shape)}, orig_kv_heads={orig_kv_heads}"
            )

        layer.qweight.data = torch.repeat_interleave(
            layer.qweight.data.view(layer.qweight.shape[0], orig_kv_heads, -1), repeat, dim=1
        ).view(layer.qweight.shape[0], -1)
        layer.qzeros.data = torch.repeat_interleave(
            layer.qzeros.data.view(layer.qzeros.shape[0], orig_kv_heads, -1), repeat, dim=1
        ).view(layer.qzeros.shape[0], -1)
        layer.scales.data = torch.repeat_interleave(
            layer.scales.data.view(layer.scales.shape[0], orig_kv_heads, -1), repeat, dim=1
        ).view(layer.scales.shape[0], -1)
        layer.out_features = layer.out_features * repeat

    elif isinstance(layer, QuantLinearGPTQ):
        if layer.qweight.shape[1] % orig_kv_heads != 0:
            raise ValueError(
                f"{layer_prefix}Invalid GPTQ qweight shape for RepeatKV: qweight.shape={tuple(layer.qweight.shape)}, "
                f"orig_kv_heads={orig_kv_heads}"
            )
        if layer.qzeros.shape[1] % orig_kv_heads != 0 or layer.scales.shape[1] % orig_kv_heads != 0:
            raise ValueError(
                f"{layer_prefix}Invalid GPTQ qzeros/scales shape for RepeatKV: qzeros.shape={tuple(layer.qzeros.shape)}, "
                f"scales.shape={tuple(layer.scales.shape)}, orig_kv_heads={orig_kv_heads}"
            )

        layer.qweight.data = torch.repeat_interleave(
            layer.qweight.data.view(layer.qweight.shape[0], orig_kv_heads, -1), repeat, dim=1
        ).view(layer.qweight.shape[0], -1)
        layer.qzeros.data = torch.repeat_interleave(
            layer.qzeros.data.view(layer.qzeros.shape[0], orig_kv_heads, -1), repeat, dim=1
        ).view(layer.qzeros.shape[0], -1)
        layer.scales.data = torch.repeat_interleave(
            layer.scales.data.view(layer.scales.shape[0], orig_kv_heads, -1), repeat, dim=1
        ).view(layer.scales.shape[0], -1)
        layer.out_features = layer.out_features * repeat

    elif isinstance(layer, QuantLinearORT):
        float_weight, zeros_per_group, scales_per_group = dequantize_blockwise_bits(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            layer.bits,
            layer.group_size,
            layer.g_idx,
            layer.in_features,
            layer.out_features,
        )
        if float_weight.shape[0] % orig_kv_heads != 0:
            raise ValueError(
                f"{layer_prefix}Invalid QuantLinearORT weight shape for RepeatKV: "
                f"weight.shape={tuple(float_weight.shape)}, orig_kv_heads={orig_kv_heads}"
            )

        duplicated_weight = torch.repeat_interleave(
            float_weight.view(orig_kv_heads, -1, float_weight.shape[1]),
            repeat,
            dim=0,
        ).view(new_kv_heads * (float_weight.shape[0] // orig_kv_heads), float_weight.shape[1])

        duplicated_zeros = torch.repeat_interleave(
            zeros_per_group.view(orig_kv_heads, -1, zeros_per_group.shape[1]),
            repeat,
            dim=0,
        ).view(new_kv_heads * (zeros_per_group.shape[0] // orig_kv_heads), zeros_per_group.shape[1])
        duplicated_scales = torch.repeat_interleave(
            scales_per_group.view(orig_kv_heads, -1, scales_per_group.shape[1]),
            repeat,
            dim=0,
        ).view(new_kv_heads * (scales_per_group.shape[0] // orig_kv_heads), scales_per_group.shape[1])

        original_out_features = layer.out_features
        layer.out_features = original_out_features * repeat
        q_rows = layer.in_features // layer.group_size
        layer.qweight = torch.zeros(
            (layer.out_features, q_rows, layer.group_size // (8 // layer.bits)),
            dtype=layer.qweight.dtype,
            device=layer.qweight.device,
        )
        layer.qzeros = torch.zeros(
            (q_rows + (q_rows & 1)) * (layer.out_features // 8 * layer.bits),
            dtype=layer.qzeros.dtype,
            device=layer.qzeros.device,
        )
        layer.scales = torch.zeros(
            (q_rows * layer.out_features),
            dtype=layer.scales.dtype,
            device=layer.scales.device,
        )

        linear = nn.Linear(layer.in_features, layer.out_features, bias=False, dtype=duplicated_weight.dtype)
        linear.weight.data = duplicated_weight.to(linear.weight.dtype)
        layer.pack(
            linear,
            duplicated_scales.contiguous().to(layer.scales.dtype),
            duplicated_zeros.contiguous().to(torch.int32),
            layer.g_idx,
        )

    elif isinstance(layer, FP8DeQuantLinear):
        layer.weight.data = torch.repeat_interleave(
            layer.weight.data.view(orig_kv_heads, head_dim, hidden_size),
            repeat,
            dim=0,
        ).view(new_kv_heads * head_dim, hidden_size)
        layer.weight_scale.data = torch.repeat_interleave(
            layer.weight_scale.data.view(orig_kv_heads, head_dim), repeat, dim=0
        ).view(new_kv_heads * head_dim, -1)

    else:
        layer.weight.data = torch.repeat_interleave(
            layer.weight.data.view(orig_kv_heads, head_dim, hidden_size),
            repeat,
            dim=0,
        ).view(new_kv_heads * head_dim, hidden_size)

    if layer.bias is not None:
        layer.bias.data = torch.repeat_interleave(
            layer.bias.data.view(orig_kv_heads, head_dim),
            repeat,
            dim=0,
        ).view(new_kv_heads * head_dim)


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


def get_replication_root(model: nn.Module) -> nn.Module:
    candidate = getattr(model, "model", None)
    return candidate if isinstance(candidate, nn.Module) else model


def replication_targets(model: nn.Module, text_model: Optional[nn.Module] = None):
    targets = []
    root = get_replication_root(model)
    if root is not None:
        targets.append(root)
    if text_model is not None:
        targets.append(text_model)
        cfg = getattr(text_model, "config", None)
        if cfg is not None:
            targets.append(cfg)
    return targets


def is_replication_applied(model: nn.Module, text_model: Optional[nn.Module] = None) -> bool:
    return any(
        getattr(target, "_qeff_kv_replication_applied", False) for target in replication_targets(model, text_model)
    )
