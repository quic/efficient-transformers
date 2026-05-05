# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Utility helpers to suggest attention/FFN blocking configs for diffusers transformers and transformers

This module adapts the standalone configurator script into a clean, importable API
that can be fed model config + pipeline compile config to derive blocking settings.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode
from QEfficient.blocking.ffn_blocking import FFNBlockingConfig, FFNBlockingMode
from QEfficient.utils import get_attr_or_key, require_value
from QEfficient.utils.constants import DEFAULT_NUM_HEADS, FP16_BYTES, KV_LORA_RANK, ROPE_DIM, VTCM_SIZE_THRESHOLD


def _infer_head_dim(model_config: Any, num_heads: int) -> int:
    head_dim = get_attr_or_key(model_config, ("attention_head_dim", "head_dim", "head_dim_per_head"))
    if head_dim is not None:
        return int(head_dim)
    hidden_size = get_attr_or_key(model_config, ("hidden_size", "d_model", "model_dim", "attention_dim"))
    if hidden_size is None:
        raise ValueError("Missing head_dim or hidden_size to compute attention blocking configuration.")
    return int(hidden_size) // int(num_heads)


def _infer_data_bytes(compile_config: Dict[str, Any]) -> int:
    explicit = compile_config.get("data_bytes")
    if explicit is not None:
        return int(explicit)
    if compile_config.get("convert_to_fp16", False):
        return 2
    return 4


def _normalize_attention_mode(raw_mode: str) -> str:
    mode = raw_mode.lower()
    if "q" in mode and "kv" in mode:
        return "qkv"
    if "kv" in mode:
        return "kv"
    if "q" in mode:
        return "q"
    return ""


def _resolve_effective_blocking_mode(attention_cfg: Dict[str, Any], requested_mode: str) -> str:
    mode = _normalize_attention_mode(requested_mode)
    if mode == "":
        return ""
    num_q_blocks = attention_cfg.get("num_q_blocks") or 1
    num_kv_blocks = attention_cfg.get("num_kv_blocks") or 1
    head_block_size = (attention_cfg.get("head_block_size") or 1) if attention_cfg.get("head_blocking_enabled") else 1

    if head_block_size > 1 and num_q_blocks == 1 and num_kv_blocks == 1:
        return "h"
    if head_block_size > 1:
        return "hqkv"
    if num_q_blocks > 1 and num_kv_blocks > 1:
        return "qkv"
    if num_q_blocks > 1:
        return "q"
    if num_kv_blocks > 1:
        return "kv"
    return ""


def resolve_ffn_blocking_mode(ffn_cfg: Dict[str, Any], requested_mode: str) -> str:
    """
    Resolve effective FFN mode string ("", "t", "w", "tw") based on whether auto-config
    actually selected >1 blocks for tokens/weights.

    This mirrors `_resolve_effective_blocking_mode` for attention.
    """
    mode = (requested_mode or "").lower()
    if mode == "":
        return ""

    num_token_blocks = int(ffn_cfg.get("num_token_blocks") or 1)
    num_weight_blocks = int(ffn_cfg.get("num_weight_blocks") or 1)

    effective = ""
    if "t" in mode and num_token_blocks > 1:
        effective += "t"
    if "w" in mode and num_weight_blocks > 1:
        effective += "w"
    return effective


def _get_valid_num_blocks(config: Dict, requested_key: str) -> int:
    if config.get(requested_key) < 1:
        raise ValueError(f"Invalid value {requested_key} passed in qaic_config: {config.get(requested_key)}")
    return config.get(requested_key)


def block_candidates_generator(max_length: int) -> List[int]:
    block_list = []
    i = 1
    step = 1
    while i <= max_length:
        block_list.append(i)
        if i % (4 * step) == 0:
            step *= 2
        i += step
    return block_list


def matmul1_bytes(q_len: int, kv_block_size: int, num_heads: int = DEFAULT_NUM_HEADS) -> int:
    """Bytes for [1,num_heads,q,kv] x [1,1,kv,512] -> [1,num_heads,q,512] in fp16."""
    elems_a = num_heads * q_len * kv_block_size
    elems_b = kv_block_size * KV_LORA_RANK
    elems_out = num_heads * q_len * KV_LORA_RANK
    return FP16_BYTES * (elems_a + elems_b + elems_out)


def matmul2_bytes(q_len: int, kv_block_size: int, num_heads: int = DEFAULT_NUM_HEADS) -> int:
    """Bytes for [1,num_heads,q,576] x [1,1,576,kv] -> [1,num_heads,q,kv] in fp16."""
    elems_a = num_heads * q_len * (KV_LORA_RANK + ROPE_DIM)
    elems_b = 576 * kv_block_size
    elems_out = num_heads * q_len * kv_block_size
    return FP16_BYTES * (elems_a + elems_b + elems_out)


def max_kv_block_size(
    q_len: int,
    budget_bytes: int = VTCM_SIZE_THRESHOLD,
    num_heads: int = DEFAULT_NUM_HEADS,
) -> int:
    """Return the largest integer kv_block_size that satisfies both matmul budgets.

    Returns 0 if no positive kv_block_size can satisfy the constraints.
    """
    if q_len < 0:
        raise ValueError("q_len must be non-negative")
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")

    # Enforce strict inequality in bytes:
    # FP16_BYTES * elems < budget_bytes  =>  elems <= floor((budget_bytes - 1)/FP16_BYTES)
    max_elems = (budget_bytes - 1) // FP16_BYTES

    # Matmul1 elements:
    #   A_elems = num_heads*q_len*kv
    #   B_elems = kv*512
    #   C_elems = num_heads*q_len*512
    # Enforce A_elems + B_elems + C_elems <= max_elems
    c1_elems = num_heads * q_len * KV_LORA_RANK
    rem1 = max_elems - c1_elems
    den1 = num_heads * q_len + KV_LORA_RANK  # kv coefficient from A_elems + B_elems
    k1 = rem1 // den1 if rem1 >= 0 else -1

    # Matmul2 elements:
    #   A_elems = num_heads*q_len*576
    #   B_elems = 576*kv
    #   C_elems = num_heads*q_len*kv
    # Enforce A_elems + B_elems + C_elems <= max_elems
    a2_elems = num_heads * q_len * 576
    rem2 = max_elems - a2_elems
    den2 = num_heads * q_len + 576  # kv coefficient from B_elems + C_elems
    k2 = rem2 // den2 if rem2 >= 0 else -1

    kv = min(k1, k2)
    return max(0, kv)


def get_num_kv_blocks_for_mla(q_len, num_heads, ctx_len):
    """Compute the maximum kv_block_size under an fp16 memory budget.

    Constraints (bytes) per matmul:
    1) [1, num_heads, q_len, 576] x [1, 1, 576, kv] -> [1, num_heads, q_len, kv]
    2) [1, num_heads, q_len, kv] x [1, 1, kv, 512] -> [1, num_heads, q_len, 512]

    For each matmul, sum(input_a + input_b + output) must be < budget.
    The returned kv_block_size satisfies both constraints.
    """
    budget_bytes = VTCM_SIZE_THRESHOLD
    kv = max_kv_block_size(q_len, budget_bytes, num_heads)
    b1 = matmul1_bytes(q_len, kv, num_heads)
    b2 = matmul2_bytes(q_len, kv, num_heads)

    assert b1 < budget_bytes, "matmul1 is not under the budget"
    assert b2 < budget_bytes, "matmul2 is not under the budget"

    kv_block_size = ctx_len
    kv_block_size_list = block_candidates_generator(ctx_len)
    for i in range(len(kv_block_size_list) - 1):
        if kv_block_size_list[i] < kv < kv_block_size_list[i + 1]:
            kv_block_size = kv_block_size_list[i]
    return ctx_len // kv_block_size


def attention_configurator(
    bs: int,
    seq_len: int,
    ctx_len: int,
    num_heads: int,
    head_dim: int,
    num_socs: int,
    num_nsps: int,
    data_bytes: int,
    blocking_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Suggest attention blocking configuration based on model and device constraints.
    """
    mode = (blocking_mode or "hqkv").lower()

    num_kv_blocks_list = block_candidates_generator(ctx_len) if "kv" in mode else [1]
    num_q_blocks_list = block_candidates_generator(ctx_len) if "q" in mode else [1]

    head_block_size = num_socs if "h" in mode else num_heads
    num_head_blocks = math.ceil(num_heads / head_block_size)
    num_heads_per_iter = math.ceil(head_block_size / num_socs)

    best_config = {
        "head_block_size": head_block_size,
        "num_head_blocks": num_head_blocks,
        "head_blocking_enabled": num_head_blocks > 1,
        "num_q_blocks": None,
        "num_kv_blocks": None,
        "q_kv_ratio": None,
        "vtcm_footprint": None,
    }

    def update_best_config(num_q_blocks: int, num_kv_blocks: int, q_kv_ratio: float, footprint: float) -> None:
        best_config["num_q_blocks"] = num_q_blocks
        best_config["num_kv_blocks"] = num_kv_blocks
        best_config["q_kv_ratio"] = q_kv_ratio
        best_config["vtcm_footprint"] = footprint

    for num_q_blocks in num_q_blocks_list:
        for num_kv_blocks in num_kv_blocks_list:
            q_sl_per_nsp = math.ceil(seq_len / num_nsps / num_q_blocks)
            q_size_per_nsp = num_heads_per_iter * bs * q_sl_per_nsp * head_dim * data_bytes

            kv_cl_per_nsp = math.ceil(ctx_len / num_kv_blocks)
            kv_size_per_nsp = num_heads_per_iter * bs * kv_cl_per_nsp * head_dim * data_bytes

            qk_size_per_nsp = num_heads_per_iter * bs * q_sl_per_nsp * kv_cl_per_nsp * data_bytes
            vtcm_footprint = q_size_per_nsp + kv_size_per_nsp + qk_size_per_nsp
            q_kv_ratio = max(q_size_per_nsp / kv_size_per_nsp, kv_size_per_nsp / q_size_per_nsp)
            num_total_blocks = num_q_blocks * num_kv_blocks

            if vtcm_footprint < VTCM_SIZE_THRESHOLD:
                if best_config["num_q_blocks"] is None:
                    update_best_config(num_q_blocks, num_kv_blocks, q_kv_ratio, vtcm_footprint)
                elif best_config["num_q_blocks"] * best_config["num_kv_blocks"] > num_total_blocks:
                    update_best_config(num_q_blocks, num_kv_blocks, q_kv_ratio, vtcm_footprint)
                elif (
                    best_config["num_q_blocks"] * best_config["num_kv_blocks"] == num_total_blocks
                    and best_config["q_kv_ratio"] >= q_kv_ratio
                ):
                    update_best_config(num_q_blocks, num_kv_blocks, q_kv_ratio, vtcm_footprint)
                break

    return best_config


def check_ffn_block_config(
    num_token_blocks: int,
    num_weight_blocks: int,
    *,
    model_config: Any,
    specializations: Optional[List[Dict[str, int]]],
    compile_config: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Decide (split_for_soc, split_for_nsp) for a given FFN block configuration by running
    the VTCM-threshold checks (same 3 strategies as `ffn_configurator`).

    This keeps call-sites simple: they only need block counts + specialization + compile options,
    and we derive the rest (bs/seq_len/dims/devices/dtype size) here.

    Args:
        num_token_blocks: number of token blocks (> 0)
        num_weight_blocks: number of weight blocks (> 0)
        model_config: model config object/dict (for VLMs, pass text_config)
        specializations: specialization list; uses first entry for bs/seq_len
        compile_config: compile options dict (expects `mdp_ts_num_devices`, `aic_num_cores`,
            and dtype info via `data_bytes` or `convert_to_fp16`)

    Returns:
        (split_for_soc, split_for_nsp) where each is:
          - "weights"     : shard weights across that dimension
          - "activation"  : shard activations across that dimension
          - None          : no strategy fits VTCM threshold for the given block sizes
    """
    if specializations is None or len(specializations) == 0:
        return None, None

    bs = get_attr_or_key(specializations[0], ("batch_size", "batch"))
    seq_len = get_attr_or_key(specializations[0], ("cl", "seq_len", "sequence_length"))

    d_model = get_attr_or_key(model_config, ("hidden_size", "d_model", "model_dim"))
    intermediate = get_attr_or_key(model_config, ("intermediate_size", "ffn_dim", "mlp_dim"))

    if bs is None or seq_len is None or d_model is None or intermediate is None:
        return None, None

    num_socs = int(compile_config.get("mdp_ts_num_devices", 1))
    num_nsps = int(compile_config.get("aic_num_cores", 1))
    data_bytes = _infer_data_bytes(compile_config)

    sl_block = int(seq_len) / int(num_token_blocks)
    i_block = int(intermediate) / int(num_weight_blocks)

    # 1) split weights across all (soc*nsp)
    footprints = _op_footprints(
        bs=int(bs),
        d_model=int(d_model),
        data_bytes=int(data_bytes),
        sl_block=sl_block,
        i_block=i_block,
        split_act=1,
        split_w_up_gate=num_socs * num_nsps,
        split_w_down=num_socs * num_nsps,
    )
    if all(v < VTCM_SIZE_THRESHOLD for v in footprints.values()):
        return "weights", "weights"

    # 2) split weights across NSP, activations across SOC
    footprints = _op_footprints(
        bs=int(bs),
        d_model=int(d_model),
        data_bytes=int(data_bytes),
        sl_block=sl_block,
        i_block=i_block,
        split_act=num_socs,
        split_w_up_gate=num_nsps,
        split_w_down=num_nsps,
    )
    if all(v < VTCM_SIZE_THRESHOLD for v in footprints.values()):
        return "activation", "weights"

    # 3) split activations across all (soc*nsp)
    footprints = _op_footprints(
        bs=int(bs),
        d_model=int(d_model),
        data_bytes=int(data_bytes),
        sl_block=sl_block,
        i_block=i_block,
        split_act=num_socs * num_nsps,
        split_w_up_gate=1,
        split_w_down=1,
    )
    if all(v < VTCM_SIZE_THRESHOLD for v in footprints.values()):
        return "activation", "activation"

    return None, None


def _weights_vtcm(dim1: float, dim2: float, *, data_bytes: int, split_weights: int) -> float:
    # weights tensor is [dim1, dim2], scaled by min(1, 32 / dim2)
    w_size = dim1 * (dim2 / split_weights) * data_bytes
    cache_factor = min(1.0, 32.0 / (dim2 / split_weights))
    return w_size * cache_factor


def _op_footprints(
    *,
    bs: int,
    d_model: int,
    data_bytes: int,
    sl_block: float,
    i_block: float,
    split_act: int,
    split_w_up_gate: int,
    split_w_down: int,
) -> Dict[str, float]:
    # activations split on dim0 (token dimension)
    t = (bs * sl_block) / split_act

    # up/gate input: [t, d_model], output: [t, i_block]
    up_in = t * d_model * data_bytes
    up_out = t * i_block * data_bytes
    up_w = _weights_vtcm(d_model, i_block, data_bytes=data_bytes, split_weights=split_w_up_gate)
    up = up_in + up_out + up_w

    gate_in = up_in
    gate_out = up_out
    gate_w = up_w
    gate = gate_in + gate_out + gate_w

    # down input: [t, i_block], output: [t, d_model]
    down_in = t * i_block * data_bytes
    down_out = t * d_model * data_bytes
    down_w = _weights_vtcm(i_block, d_model, data_bytes=data_bytes, split_weights=split_w_down)
    down = down_in + down_out + down_w

    return {"up_proj": up, "gate_proj": gate, "down_proj": down}


def ffn_configurator(
    bs: int,
    seq_len: int,
    d_model: int,
    intermediate: int,
    num_socs: int,
    num_nsps: int,
    data_bytes: int,
) -> Dict[str, Any]:
    num_token_blocks_list = block_candidates_generator(seq_len)
    num_weight_blocks_list = block_candidates_generator(intermediate)

    best_config = {
        "num_token_blocks": seq_len,
        "num_weight_blocks": intermediate,
        "split_for_soc": None,
        "split_for_nsp": None,
        "vtcm_footprint": None,
    }

    for num_token_blocks in num_token_blocks_list:
        sl_block = seq_len / num_token_blocks

        for num_weight_blocks in num_weight_blocks_list:
            i_block = intermediate / num_weight_blocks

            # Strategy 1: split weights across all (soc*nsp)
            footprints = _op_footprints(
                bs=bs,
                d_model=d_model,
                data_bytes=data_bytes,
                sl_block=sl_block,
                i_block=i_block,
                split_act=1,
                split_w_up_gate=num_socs * num_nsps,
                split_w_down=num_socs * num_nsps,
            )
            if all(v < VTCM_SIZE_THRESHOLD for v in footprints.values()):
                if (
                    best_config["num_token_blocks"] * best_config["num_weight_blocks"]
                    > num_token_blocks * num_weight_blocks
                ):
                    best_config.update(
                        {
                            "num_token_blocks": num_token_blocks,
                            "num_weight_blocks": num_weight_blocks,
                            "split_for_soc": "weights",
                            "split_for_nsp": "weights",
                            "vtcm_footprint": max(footprints.values()),
                        }
                    )
                    break

            # Strategy 2: split weights across NSP, activations across SOC
            footprints = _op_footprints(
                bs=bs,
                d_model=d_model,
                data_bytes=data_bytes,
                sl_block=sl_block,
                i_block=i_block,
                split_act=num_socs,
                split_w_up_gate=num_nsps,
                split_w_down=num_nsps,
            )
            if all(v < VTCM_SIZE_THRESHOLD for v in footprints.values()):
                if (
                    best_config["num_token_blocks"] * best_config["num_weight_blocks"]
                    > num_token_blocks * num_weight_blocks
                ):
                    best_config.update(
                        {
                            "num_token_blocks": num_token_blocks,
                            "num_weight_blocks": num_weight_blocks,
                            "split_for_soc": "activation",
                            "split_for_nsp": "weights",
                            "vtcm_footprint": max(footprints.values()),
                        }
                    )
                    break

            # Strategy 3: split activations across all (soc*nsp)
            footprints = _op_footprints(
                bs=bs,
                d_model=d_model,
                data_bytes=data_bytes,
                sl_block=sl_block,
                i_block=i_block,
                split_act=num_socs * num_nsps,
                split_w_up_gate=1,
                split_w_down=1,
            )
            if all(v < VTCM_SIZE_THRESHOLD for v in footprints.values()):
                if (
                    best_config["num_token_blocks"] * best_config["num_weight_blocks"]
                    > num_token_blocks * num_weight_blocks
                ):
                    best_config.update(
                        {
                            "num_token_blocks": num_token_blocks,
                            "num_weight_blocks": num_weight_blocks,
                            "split_for_soc": "activation",
                            "split_for_nsp": "activation",
                            "vtcm_footprint": max(footprints.values()),
                        }
                    )
                    break

    return best_config


def build_transformer_blocking_config(
    model_config: Any,
    pipeline_config: Optional[Any] = None,
    module_name: str = "transformer",
    blocking_mode: Optional[str] = None,
    ctx_len: Optional[int] = None,
    seq_len: Optional[int] = None,
    bs: Optional[int] = 1,
    compile_config: Optional[Dict[str, Any]] = None,
) -> AttentionBlockingConfig:
    """
    Build attention blocking configuration based on model config + pipeline compile config.
    """
    if ctx_len is None:
        ctx_len = seq_len

    if seq_len is None and ctx_len is None:
        return AttentionBlockingConfig(mode="")

    # we only block on text configs in the case of VLMs
    if hasattr(model_config, "text_config"):
        model_config = model_config.text_config
    num_heads = require_value(
        get_attr_or_key(model_config, ("num_attention_heads", "num_heads", "attention_heads", "n_heads")),
        "num attention heads",
    )
    head_dim = _infer_head_dim(model_config, int(num_heads))

    num_socs = int(compile_config.get("mdp_ts_num_devices", 1))
    num_nsps = int(compile_config.get("aic_num_cores", 1))
    data_bytes = _infer_data_bytes(compile_config)

    attention_cfg = attention_configurator(
        int(bs),
        int(seq_len),
        int(ctx_len),
        int(num_heads),
        int(head_dim),
        int(num_socs),
        int(num_nsps),
        int(data_bytes),
        blocking_mode=blocking_mode,
    )

    if "DeepseekV3ForCausalLM" in (getattr(model_config, "architectures", None) or []):
        if "kv" in blocking_mode:
            attention_cfg["num_kv_blocks"] = get_num_kv_blocks_for_mla(seq_len, num_heads, ctx_len)

    resolved_mode = _normalize_attention_mode(blocking_mode or "hqkv")
    effective_mode = _resolve_effective_blocking_mode(attention_cfg, resolved_mode)

    return AttentionBlockingConfig(
        mode=effective_mode,
        num_kv_blocks=attention_cfg["num_kv_blocks"],
        num_q_blocks=attention_cfg["num_q_blocks"],
        head_block_size=attention_cfg["head_block_size"],
    )


def build_ffn_blocking_config(
    model_config: Any,
    pipeline_config: Optional[Any] = None,
    module_name: str = "transformer",
    blocking_mode: Optional[str] = None,
    ctx_len: Optional[int] = None,
    seq_len: Optional[int] = None,
    bs: Optional[int] = 1,
    compile_config: Optional[Dict[str, Any]] = None,
) -> Tuple[FFNBlockingConfig, Dict[str, Any]]:
    """
    Auto-derive FFN blocking config (and any required compiler flags) from model + compile config.

    Returns:
        (FFNBlockingConfig, compiler_flags)
    """
    if ctx_len is None:
        ctx_len = seq_len

    if seq_len is None and ctx_len is None:
        return FFNBlockingConfig(mode=FFNBlockingMode.NONE), {}

    # we only block on text configs in the case of VLMs
    if hasattr(model_config, "text_config"):
        model_config = model_config.text_config

    d_model = require_value(get_attr_or_key(model_config, ("hidden_size", "d_model", "model_dim")), "hidden size")
    intermediate = require_value(
        get_attr_or_key(model_config, ("intermediate_size", "ffn_dim", "mlp_dim")), "intermediate size"
    )

    num_socs = int(compile_config.get("mdp_ts_num_devices", 1))
    num_nsps = int(compile_config.get("aic_num_cores", 1))
    data_bytes = _infer_data_bytes(compile_config)

    ffn_cfg = ffn_configurator(
        bs=int(bs),
        seq_len=int(seq_len),
        d_model=int(d_model),
        intermediate=int(intermediate),
        num_socs=int(num_socs),
        num_nsps=int(num_nsps),
        data_bytes=int(data_bytes),
    )

    requested_mode = str(blocking_mode or "tw").lower()
    effective_mode = resolve_ffn_blocking_mode(ffn_cfg, requested_mode)

    ffn_blocking_config = FFNBlockingConfig(
        mode=FFNBlockingMode(effective_mode) if effective_mode != "" else FFNBlockingMode.NONE
    )

    if "t" in effective_mode:
        ffn_blocking_config.num_token_blocks = int(ffn_cfg["num_token_blocks"])
    if "w" in effective_mode:
        ffn_blocking_config.num_weight_blocks = int(ffn_cfg["num_weight_blocks"])

    return ffn_blocking_config


def build_transformer_blocking_config_for_transform(
    model_config: Any,
    ctx_len: Optional[int] = None,
    seq_len: Optional[int] = None,
    bs: Optional[int] = 1,
    num_devices: Optional[int] = 1,
    qaic_config: Optional[dict] = None,
    **compile_options,
) -> Dict[str, Any]:
    if qaic_config:
        blocking_mode = BlockingMode(qaic_config.get("blocking_mode", "hqkv"))
    else:
        blocking_mode = BlockingMode.HQKV
    enable_blocking = False if not qaic_config else qaic_config.get("enable_blocking", False)

    # ---------------- Attention blocking ----------------
    if qaic_config is None and enable_blocking:
        blocking_config = build_transformer_blocking_config(
            model_config,
            blocking_mode=blocking_mode,
            ctx_len=ctx_len,
            seq_len=seq_len,
            bs=bs,
            compile_config={"mdp_ts_num_devices": num_devices, **compile_options},
        )
    elif not enable_blocking:
        blocking_config = None
    else:
        blocking_config = AttentionBlockingConfig()
        mode_from_config = ""
        if qaic_config.get("num_kv_blocks", False) and enable_blocking and "kv" in blocking_mode:
            mode_from_config = "kv" + mode_from_config
            blocking_config.num_kv_blocks = _get_valid_num_blocks(qaic_config, "num_kv_blocks")
        if qaic_config.get("num_q_blocks", False) and enable_blocking and "q" in blocking_mode:
            mode_from_config = "q" + mode_from_config
            blocking_config.num_q_blocks = _get_valid_num_blocks(qaic_config, "num_q_blocks")
        if qaic_config.get("head_block_size", False) and enable_blocking and "h" in blocking_mode:
            mode_from_config = "h" + mode_from_config
            blocking_config.head_block_size = _get_valid_num_blocks(qaic_config, "head_block_size")
        if qaic_config.get("num_batch_blocks", False) and enable_blocking and "b" in blocking_mode:
            mode_from_config = "b" + mode_from_config
            blocking_config.num_batch_blocks = _get_valid_num_blocks(qaic_config, "num_batch_blocks")

        # check if qaic config did not provide any blocking details
        if mode_from_config == "":
            blocking_config = build_transformer_blocking_config(
                model_config,
                blocking_mode=blocking_mode,
                ctx_len=ctx_len,
                seq_len=seq_len,
                bs=bs,
                compile_config={"mdp_ts_num_devices": num_devices, **compile_options},
            )
        else:
            blocking_config.mode = BlockingMode(mode_from_config)

        if qaic_config.get("skip_kv", False) and enable_blocking:
            blocking_config.skip_kv = qaic_config.get("skip_kv")

    # ---------------- FFN blocking ----------------
    if qaic_config:
        ffn_blocking_mode = FFNBlockingMode(qaic_config.get("ffn_blocking_mode", "tw"))
    else:
        ffn_blocking_mode = FFNBlockingMode.TW
    enable_ffn_blocking = False if not qaic_config else qaic_config.get("enable_ffn_blocking", False)

    if not enable_ffn_blocking:
        ffn_blocking_config = None
    else:
        ffn_blocking_config = FFNBlockingConfig()
        mode_from_config = ""

        if qaic_config is not None:
            if qaic_config.get("num_token_blocks", False) and enable_ffn_blocking and "t" in ffn_blocking_mode:
                ffn_blocking_config.num_token_blocks = _get_valid_num_blocks(qaic_config, "num_token_blocks")
                mode_from_config += "t"
            if qaic_config.get("num_weight_blocks", False) and enable_ffn_blocking and "w" in ffn_blocking_mode:
                ffn_blocking_config.num_weight_blocks = _get_valid_num_blocks(qaic_config, "num_weight_blocks")
                mode_from_config += "w"
            # check if qaic config did not provide any blocking details
            if mode_from_config == "":
                # Auto derive from constraints (and collect compiler flags)
                ffn_blocking_config = build_ffn_blocking_config(
                    model_config,
                    blocking_mode=str(ffn_blocking_mode).lower(),
                    ctx_len=ctx_len,
                    seq_len=seq_len,
                    bs=bs,
                    compile_config={"mdp_ts_num_devices": num_devices, **compile_options},
                )
            else:
                ffn_blocking_config.mode = FFNBlockingMode(mode_from_config)

    return {"attention": blocking_config, "ffn": ffn_blocking_config}
