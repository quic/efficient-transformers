# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn

from QEfficient.customop.matmulnbits import QuantLinearORT, dequantize_blockwise_bits
from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ
from QEfficient.transformers.quantizers.quantizer_compressed_tensors import FP8DeQuantLinear

DuplicateHandler = Callable[[nn.Module, int, int, int, int, str], None]


def _duplicate_awq(
    layer: WQLinear_GEMM,
    orig_kv_heads: int,
    repeat: int,
    head_dim: int,
    hidden_size: int,
    layer_prefix: str,
) -> None:
    del head_dim, hidden_size
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


def _duplicate_gptq(
    layer: QuantLinearGPTQ,
    orig_kv_heads: int,
    repeat: int,
    head_dim: int,
    hidden_size: int,
    layer_prefix: str,
) -> None:
    del head_dim, hidden_size
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


def _duplicate_quantlinear_ort(
    layer: QuantLinearORT,
    orig_kv_heads: int,
    repeat: int,
    head_dim: int,
    hidden_size: int,
    layer_prefix: str,
) -> None:
    del head_dim, hidden_size
    new_kv_heads = repeat * orig_kv_heads
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


def _duplicate_fp8(
    layer: FP8DeQuantLinear,
    orig_kv_heads: int,
    repeat: int,
    head_dim: int,
    hidden_size: int,
    layer_prefix: str,
) -> None:
    del layer_prefix
    new_kv_heads = repeat * orig_kv_heads
    layer.weight.data = torch.repeat_interleave(
        layer.weight.data.view(orig_kv_heads, head_dim, hidden_size),
        repeat,
        dim=0,
    ).view(new_kv_heads * head_dim, hidden_size)
    layer.weight_scale.data = torch.repeat_interleave(
        layer.weight_scale.data.view(orig_kv_heads, head_dim), repeat, dim=0
    ).view(new_kv_heads * head_dim, -1)


def _duplicate_default(
    layer: nn.Module,
    orig_kv_heads: int,
    repeat: int,
    head_dim: int,
    hidden_size: int,
    layer_prefix: str,
) -> None:
    del layer_prefix
    new_kv_heads = repeat * orig_kv_heads
    layer.weight.data = torch.repeat_interleave(
        layer.weight.data.view(orig_kv_heads, head_dim, hidden_size),
        repeat,
        dim=0,
    ).view(new_kv_heads * head_dim, hidden_size)


_KV_PROJECTION_DUPLICATE_HANDLERS: Dict[Type[nn.Module], DuplicateHandler] = {
    WQLinear_GEMM: _duplicate_awq,
    QuantLinearGPTQ: _duplicate_gptq,
    QuantLinearORT: _duplicate_quantlinear_ort,
    FP8DeQuantLinear: _duplicate_fp8,
}


def _resolve_duplicate_handler(layer: nn.Module) -> DuplicateHandler:
    layer_type = type(layer)
    if layer_type in _KV_PROJECTION_DUPLICATE_HANDLERS:
        return _KV_PROJECTION_DUPLICATE_HANDLERS[layer_type]
    for klass, handler in _KV_PROJECTION_DUPLICATE_HANDLERS.items():
        if isinstance(layer, klass):
            return handler
    return _duplicate_default


def duplicate_kv_projection_weights_dispatch(
    layer: nn.Module,
    orig_kv_heads: int,
    repeat: int,
    head_dim: int,
    hidden_size: int,
    layer_name: Optional[str] = None,
) -> None:
    layer_prefix = f"{layer_name}: " if layer_name else ""
    handler = _resolve_duplicate_handler(layer)
    handler(layer, orig_kv_heads, repeat, head_dim, hidden_size, layer_prefix)
    if layer.bias is not None:
        new_kv_heads = repeat * orig_kv_heads
        layer.bias.data = torch.repeat_interleave(
            layer.bias.data.view(orig_kv_heads, head_dim),
            repeat,
            dim=0,
        ).view(new_kv_heads * head_dim)
