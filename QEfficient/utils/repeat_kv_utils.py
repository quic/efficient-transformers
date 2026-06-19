# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from QEfficient.customop.matmulnbits import QuantLinearORT, dequantize_blockwise_bits
from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ
from QEfficient.transformers.quantizers.quantizer_compressed_tensors import FP8DeQuantLinear

ATTENTION_HEAD_CONFIG_KEYS = ("num_attention_heads", "n_head", "n_heads", "num_heads")
KV_HEAD_CONFIG_KEYS = ("num_key_value_heads", "n_kv_heads", "num_kv_heads", "effective_n_kv_heads")
HIDDEN_SIZE_CONFIG_KEYS = ("hidden_size", "n_embd", "d_model")

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


def get_first_config_value(config, names: Iterable[str], default=None, cast_int: bool = False):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value) if cast_int else value
    return default


def resolve_attention_heads(config) -> Optional[int]:
    return get_first_config_value(config, ATTENTION_HEAD_CONFIG_KEYS, cast_int=True)


def resolve_kv_heads(config) -> Optional[int]:
    value = get_first_config_value(config, KV_HEAD_CONFIG_KEYS, cast_int=True)
    if value is None:
        value = resolve_attention_heads(config)
    return value


def resolve_hidden_size(config) -> Optional[int]:
    return get_first_config_value(config, HIDDEN_SIZE_CONFIG_KEYS, cast_int=True)


def set_kv_head_aliases(config, value: int) -> None:
    setattr(config, "num_key_value_heads", value)
    for key in KV_HEAD_CONFIG_KEYS:
        if hasattr(config, key):
            setattr(config, key, value)


def calculate_num_replicate_kv_heads(num_devices: int, text_model_config) -> int:
    num_attention_heads = resolve_attention_heads(text_model_config)
    num_kv_heads = resolve_kv_heads(text_model_config)

    if num_attention_heads is None or num_kv_heads is None or num_attention_heads < 1 or num_kv_heads < 1:
        return 1

    num_devices = max(1, int(num_devices))
    max_repeat = max(1, num_attention_heads // num_kv_heads)

    for repeat in range(max_repeat, 0, -1):
        repeated_kv_heads = num_kv_heads * repeat
        if (repeated_kv_heads % num_devices == 0) and (num_attention_heads % repeated_kv_heads == 0):
            return repeat

    return 1


def _repeat_head_dim0(tensor: torch.Tensor, orig_kv_heads: int, repeat: int, *shape_tail: int) -> torch.Tensor:
    return torch.repeat_interleave(tensor.view(orig_kv_heads, *shape_tail), repeat, dim=0)


def _validate_packed_projection(layer: nn.Module, orig_kv_heads: int, layer_prefix: str) -> None:
    if layer.qweight.shape[1] % orig_kv_heads != 0:
        raise ValueError(
            f"{layer_prefix}Invalid packed qweight shape for RepeatKV: "
            f"qweight.shape={tuple(layer.qweight.shape)}, orig_kv_heads={orig_kv_heads}"
        )
    if layer.qzeros.shape[1] % orig_kv_heads != 0 or layer.scales.shape[1] % orig_kv_heads != 0:
        raise ValueError(
            f"{layer_prefix}Invalid packed qzeros/scales shape for RepeatKV: "
            f"qzeros.shape={tuple(layer.qzeros.shape)}, scales.shape={tuple(layer.scales.shape)}, "
            f"orig_kv_heads={orig_kv_heads}"
        )


def _duplicate_packed_projection(layer: nn.Module, orig_kv_heads: int, repeat: int, layer_prefix: str) -> None:
    _validate_packed_projection(layer, orig_kv_heads, layer_prefix)
    for attr_name in ("qweight", "qzeros", "scales"):
        tensor = getattr(layer, attr_name)
        repeated = torch.repeat_interleave(tensor.data.view(tensor.shape[0], orig_kv_heads, -1), repeat, dim=1)
        tensor.data = repeated.view(tensor.shape[0], -1)
    layer.out_features *= repeat


def _duplicate_quant_linear_ort(layer: QuantLinearORT, orig_kv_heads: int, repeat: int, layer_prefix: str) -> None:
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

    new_kv_heads = repeat * orig_kv_heads
    duplicated_weight = _repeat_head_dim0(
        float_weight,
        orig_kv_heads,
        repeat,
        float_weight.shape[0] // orig_kv_heads,
        float_weight.shape[1],
    ).view(new_kv_heads * (float_weight.shape[0] // orig_kv_heads), float_weight.shape[1])
    duplicated_zeros = _repeat_head_dim0(
        zeros_per_group,
        orig_kv_heads,
        repeat,
        zeros_per_group.shape[0] // orig_kv_heads,
        zeros_per_group.shape[1],
    ).view(new_kv_heads * (zeros_per_group.shape[0] // orig_kv_heads), zeros_per_group.shape[1])
    duplicated_scales = _repeat_head_dim0(
        scales_per_group,
        orig_kv_heads,
        repeat,
        scales_per_group.shape[0] // orig_kv_heads,
        scales_per_group.shape[1],
    ).view(new_kv_heads * (scales_per_group.shape[0] // orig_kv_heads), scales_per_group.shape[1])

    layer.out_features *= repeat
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
    layer.scales = torch.zeros(q_rows * layer.out_features, dtype=layer.scales.dtype, device=layer.scales.device)

    linear = nn.Linear(layer.in_features, layer.out_features, bias=False, dtype=duplicated_weight.dtype)
    linear.weight.data = duplicated_weight.to(linear.weight.dtype)
    layer.pack(
        linear,
        duplicated_scales.contiguous().to(layer.scales.dtype),
        duplicated_zeros.contiguous().to(torch.int32),
        layer.g_idx,
    )


def _duplicate_fp8_projection(
    layer: FP8DeQuantLinear, orig_kv_heads: int, repeat: int, head_dim: int, hidden_size: int
) -> None:
    new_kv_heads = repeat * orig_kv_heads
    layer.out_features *= repeat
    layer.weight.data = _repeat_head_dim0(layer.weight.data, orig_kv_heads, repeat, head_dim, hidden_size).view(
        new_kv_heads * head_dim, hidden_size
    )
    layer.weight_scale.data = _repeat_head_dim0(layer.weight_scale.data, orig_kv_heads, repeat, head_dim).view(
        new_kv_heads * head_dim, -1
    )


def _duplicate_dense_projection(
    layer: nn.Module, orig_kv_heads: int, repeat: int, head_dim: int, hidden_size: int
) -> None:
    new_kv_heads = repeat * orig_kv_heads
    if hasattr(layer, "out_features"):
        layer.out_features *= repeat
    layer.weight.data = _repeat_head_dim0(layer.weight.data, orig_kv_heads, repeat, head_dim, hidden_size).view(
        new_kv_heads * head_dim, hidden_size
    )


def duplicate_kv_projection_weights(
    layer: nn.Module,
    orig_kv_heads: int,
    repeat: int,
    head_dim: int,
    hidden_size: int,
    layer_name: Optional[str] = None,
) -> None:
    layer_prefix = f"{layer_name}: " if layer_name else ""

    if isinstance(layer, (WQLinear_GEMM, QuantLinearGPTQ)):
        _duplicate_packed_projection(layer, orig_kv_heads, repeat, layer_prefix)
    elif isinstance(layer, QuantLinearORT):
        _duplicate_quant_linear_ort(layer, orig_kv_heads, repeat, layer_prefix)
    elif isinstance(layer, FP8DeQuantLinear):
        _duplicate_fp8_projection(layer, orig_kv_heads, repeat, head_dim, hidden_size)
    elif hasattr(layer, "weight"):
        _duplicate_dense_projection(layer, orig_kv_heads, repeat, head_dim, hidden_size)
    else:
        raise TypeError(f"{layer_prefix}Unsupported projection layer for RepeatKV: {layer.__class__.__name__}")

    bias = getattr(layer, "bias", None)
    if bias is not None:
        new_kv_heads = repeat * orig_kv_heads
        bias.data = _repeat_head_dim0(bias.data, orig_kv_heads, repeat, head_dim).view(new_kv_heads * head_dim)


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
    if is_valid_text_model(model):
        return model

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
