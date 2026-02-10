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

import json
import math
from typing import Any, Dict, List, Optional, Tuple

VTCM_SIZE_THRESHOLD = 8 * 1024 * 1024 * 0.75


def _get_attr_or_key(obj: Any, names: Tuple[str, ...], default: Any = None) -> Any:
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _require_value(value: Any, label: str) -> Any:
    if value is None:
        raise ValueError(f"Missing required {label} to compute blocking configuration.")
    return value


def _infer_head_dim(model_config: Any, num_heads: int) -> int:
    head_dim = _get_attr_or_key(model_config, ("attention_head_dim", "head_dim", "head_dim_per_head"))
    if head_dim is not None:
        return int(head_dim)
    hidden_size = _get_attr_or_key(model_config, ("hidden_size", "d_model", "model_dim", "attention_dim"))
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


def _infer_pipeline_config(pipeline_config: Any) -> Dict[str, Any]:
    if pipeline_config is None:
        return {}
    if isinstance(pipeline_config, dict):
        return pipeline_config
    if isinstance(pipeline_config, str):
        with open(pipeline_config, "r", encoding="utf-8") as f:
            return json.load(f)
    raise TypeError("pipeline_config must be a dict or a path to a JSON config.")


def _extract_module_configs(
    pipeline_config: Dict[str, Any],
    module_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    modules = pipeline_config.get("modules", {})
    module_config = modules.get(module_name, {})
    specializations = module_config.get("specializations", {})
    compile_config = module_config.get("compilation", {})
    return specializations, compile_config


def _normalize_attention_mode(raw_mode: str) -> str:
    mode = raw_mode.lower()
    if "q" in mode and "kv" in mode:
        return "qkv"
    if "kv" in mode:
        return "kv"
    if "q" in mode:
        return "q"
    return "default"


def _resolve_effective_blocking_mode(attention_cfg: Dict[str, Any], requested_mode: str) -> str:
    mode = _normalize_attention_mode(requested_mode)
    if mode == "default":
        return "default"
    num_q_blocks = attention_cfg.get("num_q_blocks") or 1
    num_kv_blocks = attention_cfg.get("num_kv_blocks") or 1
    if num_q_blocks > 1 and num_kv_blocks > 1:
        return "qkv"
    if num_q_blocks > 1:
        return "q"
    if num_kv_blocks > 1:
        return "kv"
    return "default"


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


def ffn_configurator(
    bs: int,
    seq_len: int,
    d_model: int,
    intermediate: int,
    num_socs: int,
    num_nsps: int,
    data_bytes: int,
) -> Dict[str, Any]:
    class FFN:
        def __init__(self, seq_len: float, d_model: float, intermediate: float, data_bytes: int):
            class Conv:
                def __init__(self, dim_0: float, dim_1: float, dim_2: float, data_bytes: int):
                    class Tensor2D:
                        def __init__(self, dim_0: float, dim_1: float, data_bytes: int):
                            self.data = [dim_0, dim_1]
                            self.data_bytes = data_bytes

                        def split(self, num_split: int, axis: int) -> None:
                            self.data[axis] /= num_split

                        def get_size(self) -> float:
                            return self.data[0] * self.data[1] * self.data_bytes

                    self.input = Tensor2D(dim_0, dim_1, data_bytes)
                    self.output = Tensor2D(dim_0, dim_2, data_bytes)
                    self.weights = Tensor2D(dim_1, dim_2, data_bytes)

                def split_weights(self, num_split: int) -> None:
                    self.output.split(num_split, 1)
                    self.weights.split(num_split, 1)

                def split_activation(self, num_split: int) -> None:
                    self.input.split(num_split, 0)
                    self.output.split(num_split, 0)

                def get_vtcm_size(self) -> float:
                    return (
                        self.input.get_size()
                        + self.output.get_size()
                        + self.weights.get_size() * min(1, 32 / self.weights.data[1])
                    )

            self.up_proj = Conv(bs * seq_len, d_model, intermediate, data_bytes)
            self.gate_proj = Conv(bs * seq_len, d_model, intermediate, data_bytes)
            self.down_proj = Conv(bs * seq_len, intermediate, d_model, data_bytes)

        def split_weights(self, num_split: int) -> None:
            self.up_proj.split_weights(num_split)
            self.gate_proj.split_weights(num_split)
            self.down_proj.split_weights(num_split)

        def split_activation(self, num_split: int) -> None:
            self.up_proj.split_activation(num_split)
            self.gate_proj.split_activation(num_split)
            self.down_proj.split_activation(num_split)

        def get_vtcm_size(self) -> Dict[str, float]:
            return {
                "up_proj": self.up_proj.get_vtcm_size(),
                "gate_proj": self.gate_proj.get_vtcm_size(),
                "down_proj": self.down_proj.get_vtcm_size(),
            }

    num_token_blocks_list = block_candidates_generator(seq_len)
    num_weights_blocks_list = block_candidates_generator(intermediate)

    best_config = {
        "num_token_blocks": seq_len,
        "num_weights_blocks": intermediate,
        "split_for_soc": None,
        "split_for_nsp": None,
        "vtcm_footprint": None,
    }

    for num_token_blocks in num_token_blocks_list:
        for num_weights_blocks in num_weights_blocks_list:
            ffn = FFN(seq_len / num_token_blocks, d_model, intermediate / num_weights_blocks, data_bytes)
            ffn.split_weights(num_socs * num_nsps)
            if all(footprint < VTCM_SIZE_THRESHOLD for footprint in ffn.get_vtcm_size().values()):
                if (
                    best_config["num_token_blocks"] * best_config["num_weights_blocks"]
                    > num_token_blocks * num_weights_blocks
                ):
                    best_config["num_token_blocks"] = num_token_blocks
                    best_config["num_weights_blocks"] = num_weights_blocks
                    best_config["split_for_soc"] = "weights"
                    best_config["split_for_nsp"] = "weights"
                    best_config["vtcm_footprint"] = max(ffn.get_vtcm_size().values())
                    break

            ffn = FFN(seq_len / num_token_blocks, d_model, intermediate / num_weights_blocks, data_bytes)
            ffn.split_weights(num_nsps)
            ffn.split_activation(num_socs)
            if all(footprint < VTCM_SIZE_THRESHOLD for footprint in ffn.get_vtcm_size().values()):
                if (
                    best_config["num_token_blocks"] * best_config["num_weights_blocks"]
                    > num_token_blocks * num_weights_blocks
                ):
                    best_config["num_token_blocks"] = num_token_blocks
                    best_config["num_weights_blocks"] = num_weights_blocks
                    best_config["split_for_soc"] = "activation"
                    best_config["split_for_nsp"] = "weights"
                    best_config["vtcm_footprint"] = max(ffn.get_vtcm_size().values())
                    break

            ffn = FFN(seq_len / num_token_blocks, d_model, intermediate / num_weights_blocks, data_bytes)
            ffn.split_activation(num_nsps * num_socs)
            if all(footprint < VTCM_SIZE_THRESHOLD for footprint in ffn.get_vtcm_size().values()):
                if (
                    best_config["num_token_blocks"] * best_config["num_weights_blocks"]
                    > num_token_blocks * num_weights_blocks
                ):
                    best_config["num_token_blocks"] = num_token_blocks
                    best_config["num_weights_blocks"] = num_weights_blocks
                    best_config["split_for_soc"] = "activation"
                    best_config["split_for_nsp"] = "activation"
                    best_config["vtcm_footprint"] = max(ffn.get_vtcm_size().values())
                    break

    return best_config


def build_transformer_blocking_config(
    model_config: Any,
    pipeline_config: Optional[Any] = None,
    module_name: str = "transformer",
    blocking_mode: Optional[str] = None,
    specializations: Optional[Dict[str, Any]] = None,
    compile_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build blocking configuration based on model config + pipeline compile config.
    """
    pipeline_config_dict = _infer_pipeline_config(pipeline_config)

    if specializations is None or compile_config is None:
        spec, comp = _extract_module_configs(pipeline_config_dict, module_name)
        specializations = specializations or spec
        compile_config = compile_config or comp
    if isinstance(specializations, list):
        if not specializations:
            raise ValueError("Missing specializations for blocking configuration.")
        specializations = specializations[0]

    bs = _require_value(_get_attr_or_key(specializations, ("batch_size", "batch")), "batch size")
    seq_len = _get_attr_or_key(specializations, ("cl", "seq_len", "sequence_length"))
    if seq_len is None:
        raise ValueError("Missing sequence length (cl/seq_len/sequence_length) to compute blocking configuration.")
    ctx_len = _get_attr_or_key(specializations, ("ctx_len", "context_length"))
    if ctx_len is None:
        ctx_len = _get_attr_or_key(model_config, ("context_length", "max_position_embeddings"))
    if ctx_len is None:
        ctx_len = seq_len

    num_heads = _require_value(
        _get_attr_or_key(model_config, ("num_attention_heads", "num_heads", "attention_heads", "n_heads")),
        "num attention heads",
    )
    head_dim = _infer_head_dim(model_config, int(num_heads))
    d_model = _require_value(
        _get_attr_or_key(model_config, ("hidden_size", "d_model", "model_dim", "attention_dim")), "hidden size"
    )
    intermediate = _require_value(
        _get_attr_or_key(model_config, ("intermediate_size", "ffn_dim", "mlp_dim")), "intermediate size"
    )

    num_socs = int(compile_config.get("mdp_ts_num_devices", 1))
    num_nsps = int(compile_config.get("aic_num_cores", 1))
    data_bytes = _infer_data_bytes(compile_config)

    # import ipdb; ipdb.set_trace()

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

    ffn_cfg = ffn_configurator(
        int(bs),
        int(seq_len),
        int(d_model),
        int(intermediate),
        int(num_socs),
        int(num_nsps),
        int(data_bytes),
    )

    resolved_mode = _normalize_attention_mode(blocking_mode or "hqkv")
    effective_mode = _resolve_effective_blocking_mode(attention_cfg, resolved_mode)
    compile_flags: Dict[str, Any] = {}
    if ffn_cfg["split_for_soc"] == "activation":
        compile_flags["mdts-mos"] = 1
    if ffn_cfg["split_for_nsp"] == "activation":
        compile_flags["mos"] = 1

    return {
        "blocking_mode": resolved_mode,
        "effective_blocking_mode": effective_mode,
        "attention": attention_cfg,
        "ffn": ffn_cfg,
        "compile_flags": compile_flags,
    }