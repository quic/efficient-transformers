# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Checkpoint preparation transforms for weight-free ONNX export.

Concrete transforms below are picked in priority order by CheckpointTransformPipeline
(QEfficient/base/checkpoint_transforms.py) — the first whose is_applicable() returns
True runs and the pipeline stops. Layout transforms rewrite HF checkpoint keys to
match QEff-derived parameters; DtypeConversionCheckpointTransform is only used when
the source floating-point dtype does not already match the exported ONNX input dtype.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from QEfficient.base.checkpoint_transforms import CHECKPOINT_PREPARED_SENTINEL, BaseCheckpointTransform
from QEfficient.transformers.quantizers.quantizer_utils import convert_moe_packed_tensors
from QEfficient.utils.checkpoint_utils import (
    atomic_save,
    available_ram_gb,
    copy_checkpoint_aux_files,
    cpu_count,
    read_weight_map,
    requires_dtype_conversion,
    write_index,
)
from QEfficient.utils.logging_utils import logger

# ---------------------------------------------------------------------------
# MoE-specific memory estimation — tied to _LayerStacker's tensor layout below,
# so it stays here rather than in the generic checkpoint_utils helpers.
# ---------------------------------------------------------------------------


def _estimate_layer_stack_gb(
    expert_entries: Dict[Tuple[int, int, str], Tuple[str, str]],
    layer_idx: int,
    num_experts: int,
    src: Path,
    target_dtype: torch.dtype = torch.float32,
) -> float:
    """Estimate peak RAM (GB) required to stack one MoE layer's experts.

    At the moment stacker.stack(target_dtype) runs, five tensors exist in RAM:

        Inputs  (checkpoint dtype, e.g. BF16):
          gate  [E, I, H]
          up    [E, I, H]
          down  [E, H, I]

        Outputs (target_dtype, e.g. FP32 — twice as large when converting BF16→FP32):
          gate_up  [E, 2I, H]   cat(gate, up).to(target_dtype)
          down_out [E,  H, I]   down.to(target_dtype)

    Using source dtype bytes for the outputs underestimates by ~45% when
    converting BF16→FP32, causing too many parallel workers and OOM.
    Returns 1.0 GB as a safe fallback if the shape cannot be read.
    """
    sample = next(
        (v for (li, ei, k), v in expert_entries.items() if li == layer_idx and k in ("gate_proj", "linear", "w1")),
        None,
    )
    if sample is None:
        return 1.0

    shard_name, orig_key = sample
    try:
        with safe_open(str(src / shard_name), framework="pt") as f:
            sl = f.get_slice(orig_key)
            shape = sl.get_shape()  # [I, H]
            dtype_str = sl.get_dtype()
    except Exception:
        return 1.0

    src_bytes = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1}.get(dtype_str, 2)
    tgt_bytes = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}.get(target_dtype, 4)
    ffn_dim, hidden_dim = shape

    # Three input accumulators in source dtype + two output tensors in target dtype
    input_elements = num_experts * (
        ffn_dim * hidden_dim + ffn_dim * hidden_dim + hidden_dim * ffn_dim
    )  # gate + up + down
    output_elements = num_experts * (2 * ffn_dim * hidden_dim + hidden_dim * ffn_dim)  # gate_up + down_out
    return (input_elements * src_bytes + output_elements * tgt_bytes) / 1024**3


# ---------------------------------------------------------------------------
# Sentinel marking a fully-prepared checkpoint directory
# ---------------------------------------------------------------------------
_SENTINEL = CHECKPOINT_PREPARED_SENTINEL


# ---------------------------------------------------------------------------
# Transform 1: dtype conversion only — dense model path
# ---------------------------------------------------------------------------


class DtypeConversionCheckpointTransform(BaseCheckpointTransform):
    """Convert all floating-point tensors to ``target_dtype``.

    One pass per shard, shards processed in parallel via ThreadPoolExecutor.
    Used as the dense-model fallback; for MoE checkpoints,
    MoEExpertStackingCheckpointTransform handles dtype conversion as part
    of its own single pass and this transform is never reached.
    """

    @classmethod
    def is_applicable(
        cls,
        weight_map: Dict[str, str],
        src: Optional[Path] = None,
        target_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> bool:
        """Return True when dtype conversion is required for this checkpoint."""
        return src is None or requires_dtype_conversion(Path(src), weight_map, target_dtype)

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """Convert checkpoint shards to ``target_dtype`` in a prepared output directory."""
        sentinel = out / _SENTINEL
        if sentinel.exists():
            logger.info("DtypeConversionCheckpointTransform: prepared checkpoint exists, skipping.")
            return False

        out.mkdir(parents=True, exist_ok=True)
        copy_checkpoint_aux_files(src, out)

        weight_map = read_weight_map(src)
        allowed_keys = kwargs.get("allowed_checkpoint_keys")
        selected_keys = {key for key in weight_map if allowed_keys is None or key in allowed_keys}
        shard_names = sorted({weight_map[key] for key in selected_keys})
        new_name_for = {
            shard: (f"model_{idx:04d}.safetensors" if len(shard_names) > 1 else "model.safetensors")
            for idx, shard in enumerate(shard_names)
        }

        # I/O-bound: one thread per shard, capped at 4× CPU count and hard-capped
        # at 256 — beyond that OS scheduling overhead outweighs I/O parallelism gains.
        n_workers = max_workers if max_workers is not None else min(len(shard_names), cpu_count() * 4, 256)

        def _process_shard(shard_name: str) -> None:
            tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(src / shard_name), framework="pt") as f:
                for key in f.keys():
                    if key not in selected_keys:
                        continue
                    t = f.get_tensor(key)
                    tensors[key] = t.to(target_dtype) if t.is_floating_point() else t
            if tensors:
                atomic_save(tensors, out / new_name_for[shard_name])

        logger.info(
            f"DtypeConversionCheckpointTransform: converting {len(shard_names)} shards "
            f"→ {target_dtype} | workers={n_workers} (cpus={cpu_count()})"
        )
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_shard, s) for s in shard_names]
            for fut in as_completed(futures):
                fut.result()

        new_weight_map = {k: new_name_for[v] for k, v in weight_map.items() if k in selected_keys}
        write_index(out, new_weight_map)
        sentinel.touch()
        logger.info(f"DtypeConversionCheckpointTransform: done → {out}")
        return True


class KimiK25PackQuantizedExpertsCheckpointTransform(BaseCheckpointTransform):
    """Convert Kimi K2.5 packed int4 expert tensors to QEff QuantLinearORT keys."""

    _PACKED_RE = re.compile(r"^(.+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj))\.weight_packed$")
    _EXPERT_RE = re.compile(r"^(.+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight_packed$")

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        return any(cls._PACKED_RE.match(key) for key in weight_map)

    @staticmethod
    def _convert_pack_quantized_weight(src: Path, weight_map: Dict[str, str], prefix: str):
        from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

        packed_key = f"{prefix}.weight_packed"
        scale_key = f"{prefix}.weight_scale"
        shape_key = f"{prefix}.weight_shape"
        with safe_open(str(src / weight_map[packed_key]), framework="pt", device="cpu") as f:
            packed = f.get_tensor(packed_key)
        with safe_open(str(src / weight_map[scale_key]), framework="pt", device="cpu") as f:
            scale = f.get_tensor(scale_key)
        with safe_open(str(src / weight_map[shape_key]), framework="pt", device="cpu") as f:
            shape = torch.Size(f.get_tensor(shape_key).tolist())

        bits = 4
        group_size = 32
        out_features, in_features = shape
        unpacked = unpack_from_int32(packed, bits, shape, packed_dim=1)
        int_weight = (unpacked + 2 ** (bits - 1)).to(torch.uint8)
        q_rows = in_features // group_size
        qweight = (
            (int_weight[:, 0::2] | (int_weight[:, 1::2] << 4))
            .reshape(out_features, q_rows, group_size // (8 // bits))
            .contiguous()
        )
        qzeros = torch.full((q_rows, out_features), 2 ** (bits - 1), dtype=torch.uint8)
        qzeros = (qzeros[:, 0::2] | (qzeros[:, 1::2] << 4)).reshape(-1).contiguous()
        g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
        return {
            f"{prefix}.qweight": qweight,
            f"{prefix}.scales": scale.reshape(-1).to(torch.float32).contiguous(),
            f"{prefix}.qzeros": qzeros,
            f"{prefix}.g_idx": g_idx,
        }

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> bool:
        sentinel = out / _SENTINEL
        if sentinel.exists():
            logger.info("KimiK25PackQuantizedExpertsCheckpointTransform: prepared checkpoint exists, skipping.")
            return False

        out.mkdir(parents=True, exist_ok=True)
        copy_checkpoint_aux_files(src, out)
        weight_map = read_weight_map(src)
        model_config = kwargs.get("model_config")
        text_config = getattr(model_config, "text_config", None)
        max_text_layers = getattr(text_config, "num_hidden_layers", None)
        allowed_keys = kwargs.get("allowed_checkpoint_keys")

        def _key_allowed(key: str) -> bool:
            return allowed_keys is None or key in allowed_keys

        def _layer_stack_allowed(layer_prefix: str, output_name: str) -> bool:
            if allowed_keys is None:
                return True
            return any(
                f"{layer_prefix}.all_{output_name}_{suffix}" in allowed_keys
                for suffix in ("qweight", "scales", "qzeros", "gidx")
            )

        def _mla_allowed(attn_prefix: str) -> bool:
            if allowed_keys is None:
                return True
            return any(
                f"{attn_prefix}.{suffix}" in allowed_keys
                for suffix in (
                    "q_up",
                    "q_rope",
                    "k_up",
                    "v_up",
                    "per_head_v_up",
                    "per_head_q_up",
                    "per_head_k_up",
                    "per_head_k_up_normal",
                    "fusedqk",
                )
            )

        base_keys = {
            key
            for key in weight_map
            if not (key.endswith((".weight_packed", ".weight_scale", ".weight_shape")) and ".mlp.experts." in key)
            and _key_allowed(key)
        }
        shard_names = sorted({weight_map[key] for key in base_keys})
        new_name_for = {
            shard: (f"model_{idx:04d}.safetensors" if len(shard_names) > 1 else "model.safetensors")
            for idx, shard in enumerate(shard_names)
        }
        packed_prefixes = {match.group(1) for key in weight_map if (match := cls._PACKED_RE.match(key))}
        layer_expert_prefixes: Dict[str, Dict[int, Dict[str, str]]] = {}
        attn_prefixes = sorted(
            {
                key[: -len(".q_b_proj.weight")]
                for key in weight_map
                if key.endswith(".self_attn.q_b_proj.weight") and _mla_allowed(key[: -len(".q_b_proj.weight")])
            }
        )
        for key in weight_map:
            match = cls._EXPERT_RE.match(key)
            if not match:
                continue
            layer_prefix, expert_idx, proj_name = match.group(1), int(match.group(2)), match.group(3)
            layer_match = re.search(r"\.layers\.(\d+)\.", layer_prefix)
            if max_text_layers is not None and layer_match and int(layer_match.group(1)) >= max_text_layers:
                continue
            if not any(_layer_stack_allowed(layer_prefix, output_name) for output_name in ("gate", "up", "down")):
                continue
            expert_prefix = key[: -len(".weight_packed")]
            layer_expert_prefixes.setdefault(layer_prefix, {}).setdefault(expert_idx, {})[proj_name] = expert_prefix

        def _process_shard(shard_name: str) -> None:
            tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(src / shard_name), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key not in base_keys:
                        continue
                    tensor = f.get_tensor(key)
                    tensors[key] = tensor.to(target_dtype) if tensor.is_floating_point() else tensor

            if tensors:
                atomic_save(tensors, out / new_name_for[shard_name])

        def _write_layer_stacks(layer_prefix: str, experts_by_idx: Dict[int, Dict[str, str]]) -> Dict[str, str]:
            layer_match = re.search(r"\.layers\.(\d+)\.", layer_prefix)
            layer_idx = layer_match.group(1) if layer_match else str(abs(hash(layer_prefix)))
            stacked_map: Dict[str, str] = {}
            proj_to_output = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
            for proj_name, output_name in proj_to_output.items():
                if not _layer_stack_allowed(layer_prefix, output_name):
                    continue
                expert_indices = sorted(experts_by_idx)
                expert_prefixes = [experts_by_idx[expert_idx][proj_name] for expert_idx in expert_indices]
                converted = [cls._convert_pack_quantized_weight(src, weight_map, prefix) for prefix in expert_prefixes]
                qweight = torch.stack(
                    [
                        entry[f"{prefix}.qweight"].reshape(entry[f"{prefix}.qweight"].shape[0], -1)
                        for entry, prefix in zip(converted, expert_prefixes)
                    ],
                    dim=0,
                ).contiguous()
                scales = torch.stack(
                    [entry[f"{prefix}.scales"].reshape(-1) for entry, prefix in zip(converted, expert_prefixes)],
                    dim=0,
                )
                qzeros = torch.stack(
                    [entry[f"{prefix}.qzeros"] for entry, prefix in zip(converted, expert_prefixes)],
                    dim=0,
                )
                g_idx = torch.stack(
                    [entry[f"{prefix}.g_idx"] for entry, prefix in zip(converted, expert_prefixes)],
                    dim=0,
                )

                out_features = qweight.shape[1]
                in_half = qweight.shape[2]
                group_size = 32
                in_features = in_half * 2
                num_groups = in_features // group_size
                qzeros_groups = in_features // (group_size * 2)
                tensors = {
                    f"{layer_prefix}.all_{output_name}_qweight": qweight.contiguous(),
                    f"{layer_prefix}.all_{output_name}_scales": scales.reshape(
                        qweight.shape[0], out_features, num_groups
                    )
                    .to(target_dtype)
                    .contiguous(),
                    f"{layer_prefix}.all_{output_name}_qzeros": qzeros.reshape(
                        qweight.shape[0], out_features, qzeros_groups
                    ).contiguous(),
                    f"{layer_prefix}.all_{output_name}_gidx": g_idx,
                }
                stack_name = f"kimi-k25-layer-{layer_idx}-{output_name}.safetensors"
                atomic_save(tensors, out / stack_name)
                for key in tensors:
                    stacked_map[key] = stack_name
            return stacked_map

        def _write_mla_tensors(attn_prefix: str) -> Dict[str, str]:
            config = json.loads((src / "config.json").read_text())["text_config"]
            num_heads = config["num_attention_heads"]
            qk_nope_head_dim = config["qk_nope_head_dim"]
            qk_rope_head_dim = config["qk_rope_head_dim"]
            v_head_dim = config["v_head_dim"]
            q_lora_rank = config["q_lora_rank"]
            kv_lora_rank = config["kv_lora_rank"]

            q_b_key = f"{attn_prefix}.q_b_proj.weight"
            kv_b_key = f"{attn_prefix}.kv_b_proj.weight"
            if q_b_key not in weight_map or kv_b_key not in weight_map:
                return {}
            with safe_open(str(src / weight_map[q_b_key]), framework="pt", device="cpu") as f:
                q_b_proj = f.get_tensor(q_b_key).to(target_dtype)
            with safe_open(str(src / weight_map[kv_b_key]), framework="pt", device="cpu") as f:
                kv_b_proj = f.get_tensor(kv_b_key).to(target_dtype)

            q_up, q_rope = q_b_proj.T.view(-1, num_heads, qk_nope_head_dim + qk_rope_head_dim).split(
                [qk_nope_head_dim, qk_rope_head_dim], dim=-1
            )
            q_up = q_up.reshape(-1, num_heads * qk_nope_head_dim).unsqueeze(0).contiguous()
            q_rope = q_rope.reshape(-1, num_heads * qk_rope_head_dim).unsqueeze(0).contiguous()
            k_up, v_up = kv_b_proj.T.view(-1, num_heads, qk_nope_head_dim + v_head_dim).split(
                [qk_nope_head_dim, v_head_dim], dim=-1
            )
            k_up = k_up.reshape(-1, num_heads * qk_nope_head_dim).unsqueeze(0).contiguous()
            v_up = v_up.reshape(-1, num_heads * v_head_dim).unsqueeze(0).contiguous()
            per_head_q_up = q_up.squeeze(0).view(-1, num_heads, qk_nope_head_dim).transpose(0, 1)
            per_head_k_up = k_up.squeeze(0).view(-1, num_heads, qk_nope_head_dim).transpose(0, 1).transpose(1, 2)
            per_head_v_up = v_up.squeeze(0).view(-1, num_heads, v_head_dim).transpose(0, 1)
            fusedqk = torch.bmm(per_head_q_up, per_head_k_up).reshape(-1, num_heads, q_lora_rank, kv_lora_rank)

            tensors = {
                f"{attn_prefix}.q_up": q_up,
                f"{attn_prefix}.q_rope": q_rope,
                f"{attn_prefix}.k_up": k_up,
                f"{attn_prefix}.v_up": v_up,
                f"{attn_prefix}.per_head_v_up": per_head_v_up.unsqueeze(0).contiguous(),
                f"{attn_prefix}.per_head_q_up": per_head_q_up.unsqueeze(0).contiguous(),
                f"{attn_prefix}.per_head_k_up": per_head_k_up.unsqueeze(0).contiguous(),
                f"{attn_prefix}.per_head_k_up_normal": per_head_k_up.transpose(1, 2).unsqueeze(0).contiguous(),
                f"{attn_prefix}.fusedqk": fusedqk.contiguous(),
            }
            layer_match = re.search(r"\.layers\.(\d+)\.", attn_prefix)
            layer_idx = layer_match.group(1) if layer_match else str(abs(hash(attn_prefix)))
            stack_name = f"kimi-k25-layer-{layer_idx}-mla.safetensors"
            atomic_save(tensors, out / stack_name)
            return {key: stack_name for key in tensors}

        n_workers = max_workers if max_workers is not None else min(len(shard_names), cpu_count() * 2, 128)
        logger.info(
            f"KimiK25PackQuantizedExpertsCheckpointTransform: preparing {len(packed_prefixes)} packed expert tensors "
            f"across {len(shard_names)} shards → {target_dtype} | workers={n_workers}"
        )
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_shard, shard) for shard in shard_names]
            for fut in as_completed(futures):
                fut.result()

        new_weight_map = {}
        for key, shard_name in weight_map.items():
            if key not in base_keys:
                continue
            new_weight_map[key] = new_name_for[shard_name]
        for layer_prefix, experts_by_idx in sorted(layer_expert_prefixes.items()):
            new_weight_map.update(_write_layer_stacks(layer_prefix, experts_by_idx))
        for attn_prefix in attn_prefixes:
            layer_match = re.search(r"\.layers\.(\d+)\.", attn_prefix)
            if max_text_layers is not None and layer_match and int(layer_match.group(1)) >= max_text_layers:
                continue
            new_weight_map.update(_write_mla_tensors(attn_prefix))

        write_index(out, new_weight_map)
        sentinel.touch()
        logger.info(f"KimiK25PackQuantizedExpertsCheckpointTransform: done → {out}")
        return True


# ---------------------------------------------------------------------------
# Internal stacker helper for MoE layers
# ---------------------------------------------------------------------------


class _LayerStacker:
    """Accumulates per-expert tensors for one MoE layer and produces batched output."""

    def __init__(self, prefix: str, num_experts: int):
        """Create an accumulator for all experts in one MoE layer."""
        self.prefix = prefix
        self.num_experts = num_experts
        self._gate: Optional[torch.Tensor] = None
        self._up: Optional[torch.Tensor] = None
        self._down: Optional[torch.Tensor] = None

    def add(self, expert_idx: int, kind: str, tensor: torch.Tensor) -> None:
        """Add one expert projection tensor to the layer accumulator."""
        # Accept qwen3-moe names (gate_proj/up_proj/down_proj),
        # grok-1 names (linear/linear_v/linear_1),
        # and Mixtral names (w1=gate, w3=up, w2=down) — map to the same accumulators.
        if kind in ("gate_proj", "linear", "w1"):
            ffn_dim, hidden_dim = tensor.shape
            if self._gate is None:
                self._gate = torch.empty(self.num_experts, ffn_dim, hidden_dim, dtype=tensor.dtype)
            self._gate[expert_idx] = tensor
        elif kind in ("up_proj", "linear_v", "w3"):
            ffn_dim, hidden_dim = tensor.shape
            if self._up is None:
                self._up = torch.empty(self.num_experts, ffn_dim, hidden_dim, dtype=tensor.dtype)
            self._up[expert_idx] = tensor
        else:  # down_proj / linear_1 / w2 — shape is [hidden_dim, ffn_dim]
            hidden_dim, ffn_dim = tensor.shape
            if self._down is None:
                self._down = torch.empty(self.num_experts, hidden_dim, ffn_dim, dtype=tensor.dtype)
            self._down[expert_idx] = tensor

    def stack(self, target_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Return stacked expert tensors in the derived QEff checkpoint layout."""
        # Output in the exact layout that model __qeff_init__ creates so
        # promote_initializers_and_build_spec finds an exact checkpoint key match.
        #   _gate [E, I, H] → transpose(1,2) → gate_proj  [E, H, I]
        #   _up   [E, I, H] → transpose(1,2) → up_proj    [E, H, I]
        #   _down [E, H, I] → transpose(1,2) → down_proj_t [E, I, H]
        gate_proj = self._gate.to(target_dtype).transpose(1, 2).contiguous()
        up_proj = self._up.to(target_dtype).transpose(1, 2).contiguous()
        down_proj_t = self._down.to(target_dtype).transpose(1, 2).contiguous()
        return {
            f"{self.prefix}.gate_proj": gate_proj,  # [E, H, I]
            f"{self.prefix}.up_proj": up_proj,  # [E, H, I]
            f"{self.prefix}.down_proj_t": down_proj_t,  # [E, I, H]
        }


# ---------------------------------------------------------------------------
# Transform 2: MoE expert stacking + dtype conversion — single pass
# ---------------------------------------------------------------------------


class MoEExpertStackingCheckpointTransform(BaseCheckpointTransform):
    """Stack per-expert checkpoint keys into batched tensors AND convert dtype.

    Detects the HuggingFace per-expert layout::

        *.experts.{E}.gate_proj.weight  [I, H]  x  num_experts
        *.experts.{E}.up_proj.weight    [I, H]  x  num_experts
        *.experts.{E}.down_proj.weight  [H, I]  x  num_experts

    and produces::

        *.experts.gate_proj   [E, H, I]   (gate weights, transposed)
        *.experts.up_proj     [E, H, I]   (up weights, transposed)
        *.experts.down_proj_t [E, I, H]   (down weights, transposed)

    matching the derived parameter layout that QEff MoE model __qeff_init__
    creates, so promote_initializers_and_build_spec finds an exact key match.
    Non-expert keys receive dtype conversion in the same pass.

    Parallelism:

    - Phase 1 (scan):  one thread per shard, reads keys only (I/O bound, cheap).
    - Phase 2 (stack): one thread per layer, loads and stacks its experts.
    - Phase 3 (base):  one thread per shard, converts non-expert keys.

    Phases 2 and 3 run concurrently once phase 1 completes.
    """

    EXPERT_RE = re.compile(
        r"^(.+\.layers\.(\d+)\..+?\.experts)\.(\d+)\."
        r"(gate_proj|up_proj|down_proj|linear|linear_v|linear_1|w1|w2|w3)\.weight$"
    )

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True when the checkpoint uses per-expert MoE tensor keys."""
        return any(cls.EXPERT_RE.match(k) for k in weight_map)

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        max_workers_scan: Optional[int] = None,
        max_workers_layers: Optional[int] = None,
        max_workers_base: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """Stack per-expert MoE tensors and convert remaining tensors to ``target_dtype``."""
        sentinel = out / _SENTINEL
        if sentinel.exists():
            logger.info("MoEExpertStackingCheckpointTransform: prepared checkpoint exists, skipping.")
            return False

        out.mkdir(parents=True, exist_ok=True)
        copy_checkpoint_aux_files(src, out)

        weight_map = read_weight_map(src)
        shard_names = sorted(set(weight_map.values()))

        # ── Phase 1: parallel key scan — no tensor data loaded ────────────────
        #
        # expert_entries[(layer_idx, expert_idx, kind)] = (shard_name, orig_key)
        # layer_prefix[layer_idx]                        = prefix up to .experts
        # base_entries[orig_key]                         = shard_name
        expert_entries: Dict[Tuple[int, int, str], Tuple[str, str]] = {}
        layer_prefix: Dict[int, str] = {}
        base_entries: Dict[str, str] = {}

        def _scan(shard_name: str) -> Tuple[Dict, Dict, Dict]:
            loc_e: Dict[Tuple[int, int, str], Tuple[str, str]] = {}
            loc_p: Dict[int, str] = {}
            loc_b: Dict[str, str] = {}
            with safe_open(str(src / shard_name), framework="pt") as f:
                for key in f.keys():
                    m = cls.EXPERT_RE.match(key)
                    if m:
                        loc_e[(int(m.group(2)), int(m.group(3)), m.group(4))] = (shard_name, key)
                        loc_p[int(m.group(2))] = m.group(1)
                    else:
                        loc_b[key] = shard_name
            return loc_e, loc_p, loc_b

        # Phase 1: I/O-bound — cap at 4× logical CPUs, no point exceeding shard count.
        # Hard cap at 256: beyond that, OS scheduling overhead outweighs I/O gains.
        n_workers_scan = (
            max_workers_scan if max_workers_scan is not None else min(len(shard_names), cpu_count() * 4, 256)
        )
        logger.info(
            f"MoEExpertStackingCheckpointTransform: scanning {len(shard_names)} shards "
            f"(workers={n_workers_scan}, cpus={cpu_count()}, ram_avail={available_ram_gb():.1f} GB)..."
        )
        with ThreadPoolExecutor(max_workers=n_workers_scan) as ex:
            for loc_e, loc_p, loc_b in ex.map(_scan, shard_names):
                expert_entries.update(loc_e)
                layer_prefix.update(loc_p)
                base_entries.update(loc_b)

        experts_per_layer: Dict[int, set] = {}
        for layer_idx, expert_idx, _ in expert_entries:
            experts_per_layer.setdefault(layer_idx, set()).add(expert_idx)
        layer_indices = sorted(experts_per_layer.keys())
        sample_n = len(next(iter(experts_per_layer.values()))) if experts_per_layer else 0
        logger.info(f"  {len(layer_indices)} MoE layers × {sample_n} experts each.")

        new_weight_map: Dict[str, str] = {}

        # ── Phase 2: parallel layer stacking ──────────────────────────────────
        # Each layer thread loads its own experts (grouped by shard to open each
        # shard at most once per layer), stacks, converts dtype, writes atomically.
        def _stack_layer(layer_idx: int) -> Tuple[str, List[str]]:
            num_exp = len(experts_per_layer[layer_idx])
            stacker = _LayerStacker(layer_prefix[layer_idx], num_exp)

            # Detect which kind names are present (qwen3-moe: gate_proj/up_proj/down_proj;
            # grok-1: linear/linear_v/linear_1).
            kinds_present = {k for (li, _, k) in expert_entries if li == layer_idx}

            by_shard: Dict[str, List[Tuple[int, str, str]]] = {}
            for exp_idx in range(num_exp):
                for kind in kinds_present:
                    shard_name, orig_key = expert_entries[(layer_idx, exp_idx, kind)]
                    by_shard.setdefault(shard_name, []).append((exp_idx, kind, orig_key))

            for shard_name, entries in by_shard.items():
                with safe_open(str(src / shard_name), framework="pt") as f:
                    for exp_idx, kind, orig_key in entries:
                        stacker.add(exp_idx, kind, f.get_tensor(orig_key))

            stacked = stacker.stack(target_dtype)
            out_name = f"experts-layer-{layer_idx:05d}.safetensors"
            atomic_save(stacked, out / out_name)
            return out_name, list(stacked.keys())

        # Phase 2: memory-bound — each layer holds all E×3 expert tensors + the
        # stacked output in RAM simultaneously. Derive the worker count from
        # available RAM so we never OOM: keep 20% headroom, compute RAM per layer
        # from the actual tensor shapes in the checkpoint.
        if max_workers_layers is not None:
            n_workers_layers = max_workers_layers
        elif layer_indices:
            sample_layer = layer_indices[0]
            layer_gb = _estimate_layer_stack_gb(
                expert_entries, sample_layer, len(experts_per_layer[sample_layer]), src, target_dtype
            )
            available_gb = available_ram_gb()
            usable_gb = available_gb * 0.8
            n_workers_layers = max(1, min(len(layer_indices), int(usable_gb / layer_gb)))
        else:
            n_workers_layers = 1
            layer_gb = 0.0

        logger.info(
            f"  Stacking {len(layer_indices)} layers → {target_dtype} | "
            f"workers={n_workers_layers} (~{layer_gb:.2f} GB/layer, "
            f"{available_ram_gb():.1f} GB available)..."
        )
        with ThreadPoolExecutor(max_workers=n_workers_layers) as ex:
            futures = {ex.submit(_stack_layer, li): li for li in layer_indices}
            for fut in as_completed(futures):
                li = futures[fut]
                out_name, out_keys = fut.result()
                for key in out_keys:
                    new_weight_map[key] = out_name
                logger.info(f"    layer {li:5d} → {out_name}")

        # ── Phase 3: parallel base shard conversion ────────────────────────────
        by_shard_base: Dict[str, List[str]] = {}
        for key, shard_name in base_entries.items():
            by_shard_base.setdefault(shard_name, []).append(key)

        base_shard_list = sorted(by_shard_base)
        new_base_name_for = {shard: f"base-{idx:04d}.safetensors" for idx, shard in enumerate(base_shard_list)}

        def _convert_base(shard_name: str, keys: List[str]) -> None:
            tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(src / shard_name), framework="pt") as f:
                for key in keys:
                    t = f.get_tensor(key)
                    tensors[key] = t.to(target_dtype) if t.is_floating_point() else t
            atomic_save(tensors, out / new_base_name_for[shard_name])

        # Phase 3: mixed I/O + memory — one thread per shard, capped at CPU count.
        n_workers_base = (
            max_workers_base if max_workers_base is not None else max(1, min(len(base_shard_list), cpu_count()))
        )
        logger.info(f"  Converting {len(base_shard_list)} base shards → {target_dtype} | workers={n_workers_base}...")
        if base_shard_list:
            with ThreadPoolExecutor(max_workers=n_workers_base) as ex:
                futures_base = [ex.submit(_convert_base, s, keys) for s, keys in by_shard_base.items()]
                for fut in as_completed(futures_base):
                    fut.result()

        for key, shard_name in base_entries.items():
            new_weight_map[key] = new_base_name_for[shard_name]

        write_index(out, new_weight_map)
        sentinel.touch()
        logger.info(f"MoEExpertStackingCheckpointTransform: done → {out}")
        return True


# ---------------------------------------------------------------------------
# Transform 3: GptOss MXFP4 dequantize + split fused projections
# ---------------------------------------------------------------------------


class GptOssMxfp4ExpertDequantSplitCheckpointTransform(BaseCheckpointTransform):
    """Dequantize MXFP4-packed stacked expert tensors and split fused gate_up_proj.

    Detects the GptOss MXFP4 checkpoint layout::

        *.experts.gate_up_proj_blocks  [E, 2*I, G, B]   U8
        *.experts.gate_up_proj_scales  [E, 2*I, G]       U8
        *.experts.gate_up_proj_bias    [E, 2*I]           BF16
        *.experts.down_proj_blocks     [E, I,   G, B]   U8
        *.experts.down_proj_scales     [E, I,   G]       U8
        *.experts.down_proj_bias       [E, H]             BF16

    and produces::

        *.experts.gate_proj      [E, H, I]   (dequant gate_up_proj, first half)
        *.experts.up_proj        [E, H, I]   (dequant gate_up_proj, second half)
        *.experts.gate_proj_bias [E, I]       (gate_up_proj_bias, first half)
        *.experts.up_proj_bias   [E, I]       (gate_up_proj_bias, second half)
        *.experts.down_proj      [E, H, I]   (dequant down_proj)
        *.experts.down_proj_bias [E, H]       (dtype-converted, unchanged key)

    matching the derived parameter layout that QEffGptOssExperts.__qeff_init__
    creates, so promote_initializers_and_build_spec finds an exact key match.
    Non-expert keys receive dtype conversion in the same pass.

    Parallelism mirrors MoEExpertStackingCheckpointTransform:
    - Phase 1 (scan):    one thread per shard — collect expert tensor locations.
    - Phase 2 (dequant): one thread per layer — dequant, split, write.
    - Phase 3 (base):    one thread per shard — dtype-convert non-expert keys.
    """

    _BLOCKS_RE = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_blocks$")

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True when the checkpoint contains GPT-OSS MXFP4 expert blocks."""
        return any(cls._BLOCKS_RE.match(k) for k in weight_map)

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        max_workers_scan: Optional[int] = None,
        max_workers_layers: Optional[int] = None,
        max_workers_base: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """Dequantize GPT-OSS MXFP4 experts and split fused expert projections."""
        sentinel = out / _SENTINEL
        if sentinel.exists():
            logger.info("GptOssMxfp4ExpertDequantSplitCheckpointTransform: prepared checkpoint exists, skipping.")
            return False

        out.mkdir(parents=True, exist_ok=True)
        copy_checkpoint_aux_files(src, out)

        weight_map = read_weight_map(src)
        shard_names = sorted(set(weight_map.values()))

        # ── Phase 1: scan — collect expert tensor locations ──────────────────
        # expert_locs[(layer_idx, kind)] = (blocks_shard, blocks_key, scales_shard, scales_key)
        # bias_locs[(layer_idx, kind)]   = (shard, key)   for gate_up_proj_bias / down_proj_bias
        # layer_prefix[layer_idx]        = prefix up to .experts
        # base_entries[orig_key]         = shard_name
        _SCALES_RE = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_scales$")
        _BIAS_RE = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_bias$")

        expert_locs: Dict[Tuple[int, str], Dict] = {}  # {(layer, kind): {blocks/scales: (shard, key)}}
        bias_locs: Dict[Tuple[int, str], Tuple[str, str]] = {}
        layer_prefix: Dict[int, str] = {}
        base_entries: Dict[str, str] = {}

        def _scan(shard_name: str):
            loc_e: Dict[Tuple[int, str], Dict] = {}
            loc_b: Dict[Tuple[int, str], Tuple[str, str]] = {}
            loc_p: Dict[int, str] = {}
            loc_base: Dict[str, str] = {}
            with safe_open(str(src / shard_name), framework="pt") as f:
                for key in f.keys():
                    m = cls._BLOCKS_RE.match(key)
                    if m:
                        li, kind = int(m.group(2)), m.group(3)
                        loc_e.setdefault((li, kind), {})["blocks"] = (shard_name, key)
                        loc_p[li] = m.group(1)
                        continue
                    m = _SCALES_RE.match(key)
                    if m:
                        li, kind = int(m.group(2)), m.group(3)
                        loc_e.setdefault((li, kind), {})["scales"] = (shard_name, key)
                        loc_p[li] = m.group(1)
                        continue
                    m = _BIAS_RE.match(key)
                    if m:
                        li, kind = int(m.group(2)), m.group(3)
                        loc_b[(li, kind)] = (shard_name, key)
                        loc_p[li] = m.group(1)
                        continue
                    loc_base[key] = shard_name
            return loc_e, loc_b, loc_p, loc_base

        n_scan = max_workers_scan if max_workers_scan is not None else min(len(shard_names), cpu_count() * 4, 256)
        logger.info(
            f"GptOssMxfp4ExpertDequantSplitCheckpointTransform: scanning {len(shard_names)} shards "
            f"(workers={n_scan})..."
        )
        with ThreadPoolExecutor(max_workers=n_scan) as ex:
            for loc_e, loc_b, loc_p, loc_base in ex.map(_scan, shard_names):
                for k, v in loc_e.items():
                    expert_locs.setdefault(k, {}).update(v)
                bias_locs.update(loc_b)
                layer_prefix.update(loc_p)
                base_entries.update(loc_base)

        layer_indices = sorted({li for li, _ in expert_locs})
        logger.info(f"  Found {len(layer_indices)} MoE layers.")

        new_weight_map: Dict[str, str] = {}

        # ── Phase 2: per-layer dequant + split ────────────────────────────────
        def _process_layer(layer_idx: int) -> Tuple[str, List[str]]:
            prefix = layer_prefix[layer_idx]
            tensors: Dict[str, torch.Tensor] = {}

            def _load(shard: str, key: str) -> torch.Tensor:
                with safe_open(str(src / shard), framework="pt") as f:
                    return f.get_tensor(key)

            # gate_up_proj: dequant → [E, H, 2*I], then split interleaved (gate=even cols, up=odd cols)
            # HF _apply_gate uses gate_up[..., ::2] for gate and gate_up[..., 1::2] for up,
            # so columns are interleaved: col 0=gate0, col 1=up0, col 2=gate1, col 3=up1, ...
            gu_blocks_shard, gu_blocks_key = expert_locs[(layer_idx, "gate_up_proj")]["blocks"]
            gu_scales_shard, gu_scales_key = expert_locs[(layer_idx, "gate_up_proj")]["scales"]
            gu_blocks = _load(gu_blocks_shard, gu_blocks_key)
            gu_scales = _load(gu_scales_shard, gu_scales_key)
            gate_up = convert_moe_packed_tensors(gu_blocks, gu_scales, dtype=target_dtype)
            tensors[f"{prefix}.gate_proj"] = gate_up[..., 0::2].contiguous()
            tensors[f"{prefix}.up_proj"] = gate_up[..., 1::2].contiguous()

            # gate_up_proj_bias: split [E, 2*I] → [E, I] + [E, I] (same interleaved convention)
            if (layer_idx, "gate_up_proj") in bias_locs:
                bias_shard, bias_key = bias_locs[(layer_idx, "gate_up_proj")]
                gu_bias = _load(bias_shard, bias_key).to(target_dtype)
                tensors[f"{prefix}.gate_proj_bias"] = gu_bias[..., 0::2].contiguous()
                tensors[f"{prefix}.up_proj_bias"] = gu_bias[..., 1::2].contiguous()

            # down_proj: dequant → [E, H, I]
            dp_blocks_shard, dp_blocks_key = expert_locs[(layer_idx, "down_proj")]["blocks"]
            dp_scales_shard, dp_scales_key = expert_locs[(layer_idx, "down_proj")]["scales"]
            dp_blocks = _load(dp_blocks_shard, dp_blocks_key)
            dp_scales = _load(dp_scales_shard, dp_scales_key)
            tensors[f"{prefix}.down_proj"] = convert_moe_packed_tensors(dp_blocks, dp_scales, dtype=target_dtype)

            # down_proj_bias: pass through with dtype conversion
            if (layer_idx, "down_proj") in bias_locs:
                dp_bias_shard, dp_bias_key = bias_locs[(layer_idx, "down_proj")]
                tensors[f"{prefix}.down_proj_bias"] = _load(dp_bias_shard, dp_bias_key).to(target_dtype)

            out_name = f"experts-layer-{layer_idx:05d}.safetensors"
            atomic_save(tensors, out / out_name)
            return out_name, list(tensors.keys())

        n_layers = (
            max_workers_layers if max_workers_layers is not None else max(1, min(len(layer_indices), cpu_count()))
        )
        logger.info(f"  Dequantizing {len(layer_indices)} layers | workers={n_layers}...")
        with ThreadPoolExecutor(max_workers=n_layers) as ex:
            futures = {ex.submit(_process_layer, li): li for li in layer_indices}
            for fut in as_completed(futures):
                li = futures[fut]
                out_name, out_keys = fut.result()
                for key in out_keys:
                    new_weight_map[key] = out_name
                logger.info(f"    layer {li:5d} → {out_name}")

        # ── Phase 3: base shard dtype conversion ──────────────────────────────
        by_shard_base: Dict[str, List[str]] = {}
        for key, shard_name in base_entries.items():
            by_shard_base.setdefault(shard_name, []).append(key)

        base_shard_list = sorted(by_shard_base)
        new_base_name_for = {shard: f"base-{idx:04d}.safetensors" for idx, shard in enumerate(base_shard_list)}

        def _convert_base(shard_name: str, keys: List[str]) -> None:
            tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(src / shard_name), framework="pt") as f:
                for key in keys:
                    t = f.get_tensor(key)
                    tensors[key] = t.to(target_dtype) if t.is_floating_point() else t
            atomic_save(tensors, out / new_base_name_for[shard_name])

        n_base = max_workers_base if max_workers_base is not None else max(1, min(len(base_shard_list), cpu_count()))
        logger.info(f"  Converting {len(base_shard_list)} base shards | workers={n_base}...")
        if base_shard_list:
            with ThreadPoolExecutor(max_workers=n_base) as ex:
                futures_base = [ex.submit(_convert_base, s, keys) for s, keys in by_shard_base.items()]
                for fut in as_completed(futures_base):
                    fut.result()

        for key, shard_name in base_entries.items():
            new_weight_map[key] = new_base_name_for[shard_name]

        write_index(out, new_weight_map)
        sentinel.touch()
        logger.info(f"GptOssMxfp4ExpertDequantSplitCheckpointTransform: done → {out}")
        return True


# ---------------------------------------------------------------------------
# Transform 4: split already-stacked fused MoE experts (Mixtral v5+ layout)
# ---------------------------------------------------------------------------


class MoEFusedExpertSplitCheckpointTransform(BaseCheckpointTransform):
    """Split already-stacked MoE expert weights into the derived layout.

    Some MoE checkpoints (e.g. Mixtral transformers >= 5.x) store experts
    as per-layer fused tensors rather than per-expert individual weights:

        *.experts.gate_up_proj  [E, 2*I, H]   (gate and up concatenated)
        *.experts.down_proj     [E, H, I]

    The QEff model wrappers create derived parameters that the ONNX
    initializer names refer to:

        *.experts.gate_proj     [E, H, I]  = gate_up_proj[:, :ffn_dim, :].T(1,2)
        *.experts.up_proj       [E, H, I]  = gate_up_proj[:, ffn_dim:, :].T(1,2)
        *.experts.down_proj_t   [E, I, H]  = down_proj.T(1,2)

    is_applicable returns True only when the fused format is detected.
    Old-format checkpoints with per-expert keys (e.g. experts.0.gate_proj.weight)
    are handled by MoEExpertStackingCheckpointTransform instead.
    Also handles dtype conversion in the same pass.
    """

    _FUSED_GATE_UP_RE = re.compile(r"^(.+\.experts)\.gate_up_proj$")
    _FUSED_DOWN_RE = re.compile(r"^(.+\.experts)\.down_proj$")

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True when already-stacked fused MoE expert tensors are present."""
        return any(cls._FUSED_GATE_UP_RE.match(k) for k in weight_map)

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> bool:
        """Split fused MoE expert tensors and write a prepared checkpoint."""
        sentinel = out / _SENTINEL
        if sentinel.exists():
            logger.info("MoEFusedExpertSplitCheckpointTransform: prepared checkpoint exists, skipping.")
            return False

        index_path = src / "model.safetensors.index.json"
        if index_path.exists():
            weight_map: Dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        else:
            shards = sorted(src.glob("*.safetensors"))
            if not shards:
                return False
            weight_map = {}
            for shard in shards:
                with safe_open(str(shard), framework="pt") as f:
                    for k in f.keys():
                        weight_map[k] = shard.name

        if not cls.is_applicable(weight_map):
            return False

        out.mkdir(parents=True, exist_ok=True)
        copy_checkpoint_aux_files(src, out)

        new_weight_map: Dict[str, str] = {}
        for shard_name in sorted(set(weight_map.values())):
            shard_src = src / shard_name
            if not shard_src.exists():
                continue

            out_tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(shard_src), framework="pt") as f:
                for key in f.keys():
                    raw_tensor = f.get_tensor(key)
                    tensor = raw_tensor.to(target_dtype) if raw_tensor.is_floating_point() else raw_tensor
                    gate_up_m = cls._FUSED_GATE_UP_RE.match(key)
                    down_m = cls._FUSED_DOWN_RE.match(key)

                    if gate_up_m:
                        prefix = gate_up_m.group(1)
                        ffn_dim = tensor.shape[1] // 2
                        # Split fused [E,2I,H] → gate/up each [E,H,I]
                        out_tensors[f"{prefix}.gate_proj"] = tensor[:, :ffn_dim, :].transpose(1, 2).contiguous()
                        out_tensors[f"{prefix}.up_proj"] = tensor[:, ffn_dim:, :].transpose(1, 2).contiguous()
                        new_weight_map[f"{prefix}.gate_proj"] = shard_name
                        new_weight_map[f"{prefix}.up_proj"] = shard_name
                        # Keep original for completeness
                        out_tensors[key] = tensor
                        new_weight_map[key] = shard_name
                    elif down_m:
                        prefix = down_m.group(1)
                        # Transpose [E,H,I] → [E,I,H]
                        out_tensors[f"{prefix}.down_proj_t"] = tensor.transpose(1, 2).contiguous()
                        new_weight_map[f"{prefix}.down_proj_t"] = shard_name
                        out_tensors[key] = tensor
                        new_weight_map[key] = shard_name
                    else:
                        out_tensors[key] = tensor
                        new_weight_map[key] = shard_name

            save_file({k: v.contiguous() for k, v in out_tensors.items()}, str(out / shard_name))

        (out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": new_weight_map}, indent=2)
        )
        sentinel.touch()
        return True
