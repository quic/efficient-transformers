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
""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode
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


def build_transformer_blocking_config(
    model_config: Any,
    pipeline_config: Optional[Any] = None,
    module_name: str = "transformer",
    blocking_mode: Optional[str] = None,
    ctx_len: Optional[int] = None,
    seq_len: Optional[int] = None,
    bs: Optional[int] = 1,
    compile_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build blocking configuration based on model config + pipeline compile config.
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

    return blocking_config
