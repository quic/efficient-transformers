# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Checkpoint preparation transforms for weight-free ONNX export.

Layout transforms rewrite HF checkpoint keys to match QEff-derived parameters.
Each transform declares TRANSFORM_ID, get_consumed_keys(), and apply() which
returns {new_key: shard_file}.  DtypeConversionCheckpointTransform always runs
last on remaining (non-expert) keys.

FusedExpertSplitCheckpointTransform uses a two-map approach so that
architecture-specific key names are handled by a CHECKPOINT_KEY_REMAP class
attribute rather than separate transform classes.  Architecture subclasses
(e.g. GraniteMoeFusedExpertSplitCheckpointTransform) only declare the remap.
"""

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
# Canonical key mapping helpers
# ---------------------------------------------------------------------------


def build_canonical_maps(
    weight_map: Dict[str, str],
    key_remap: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build canonical index and key translation from raw weight_map + KEY_REMAP.

    Parameters
    ----------
    weight_map
        Raw ``{actual_key: shard_file}`` from model.safetensors.index.json.
    key_remap
        ``{regex_pattern: canonical_suffix_replacement}`` declared by the
        transform class.  Empty dict means no remapping (Mixtral-style
        checkpoints already use canonical names).

    Returns
    -------
    canonical_index
        ``{canonical_key: shard_file}`` — WHERE to find each tensor.
    key_translation
        ``{canonical_key: actual_key_in_shard}`` — WHAT to ask the shard for.
        Only contains entries where the key was remapped; absent means
        canonical_key == actual_key.
    """
    canonical_index: Dict[str, str] = {}
    key_translation: Dict[str, str] = {}
    for actual_key, shard_file in weight_map.items():
        canonical_key = actual_key
        for pattern, replacement in key_remap.items():
            remapped = re.sub(pattern, replacement, actual_key)
            if remapped != actual_key:
                canonical_key = remapped
                key_translation[canonical_key] = actual_key
                break
        canonical_index[canonical_key] = shard_file
    return canonical_index, key_translation

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

        Outputs (target_dtype, e.g. FP32 — twice as large when converting BF16->FP32):
          gate [E, H, I]
          up   [E, H, I]
          down [E, I, H]

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

    # Three input accumulators in source dtype + three output tensors in target dtype
    input_elements = num_experts * (
        ffn_dim * hidden_dim + ffn_dim * hidden_dim + hidden_dim * ffn_dim
    )  # gate + up + down
    output_elements = num_experts * (2 * ffn_dim * hidden_dim + hidden_dim * ffn_dim)  # gate + up + down
    return (input_elements * src_bytes + output_elements * tgt_bytes) / 1024**3


# ---------------------------------------------------------------------------
# Sentinel marking a fully-prepared checkpoint directory
# ---------------------------------------------------------------------------
_SENTINEL = CHECKPOINT_PREPARED_SENTINEL


def _moe_weights_prefix_from_experts_prefix(prefix: str) -> str:
    """Return the parent ``moe_weights`` prefix for a checkpoint expert prefix."""
    if prefix.endswith(".experts"):
        return prefix[: -len(".experts")] + ".moe_weights"
    return f"{prefix}.moe_weights"


def _infer_fused_gate_up_split_dim(
    gate_up_shape: Tuple[int, ...],
    down_shape: Optional[Tuple[int, ...]],
    *,
    preferred_split_dim: Optional[int] = None,
) -> int:
    """Infer whether fused gate/up uses [E,2I,H] or [E,H,2I]."""
    if len(gate_up_shape) != 3 or down_shape is None or len(down_shape) != 3:
        return preferred_split_dim if preferred_split_dim in (1, 2) else 1

    split_dim_1_match = gate_up_shape[1] == 2 * down_shape[2] and gate_up_shape[2] == down_shape[1]
    split_dim_2_match = gate_up_shape[2] == 2 * down_shape[1] and gate_up_shape[1] == down_shape[2]
    if split_dim_1_match and not split_dim_2_match:
        return 1
    if split_dim_2_match and not split_dim_1_match:
        return 2
    return preferred_split_dim if preferred_split_dim in (1, 2) else 1


def _split_fused_gate_up_to_canonical(
    gate_up: torch.Tensor,
    down_shape: Optional[Tuple[int, ...]] = None,
    *,
    interleaved: bool = False,
    preferred_split_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split fused gate/up tensor into canonical gate/up [E,H,I] tensors."""
    split_dim = _infer_fused_gate_up_split_dim(
        tuple(gate_up.shape),
        down_shape,
        preferred_split_dim=preferred_split_dim,
    )
    if split_dim == 2:
        if interleaved:
            return gate_up[..., 0::2].contiguous(), gate_up[..., 1::2].contiguous()
        ffn_dim = gate_up.shape[2] // 2
        return gate_up[..., :ffn_dim].contiguous(), gate_up[..., ffn_dim:].contiguous()

    ffn_dim = gate_up.shape[1] // 2
    gate = gate_up[:, :ffn_dim, :].transpose(1, 2).contiguous()
    up = gate_up[:, ffn_dim:, :].transpose(1, 2).contiguous()
    return gate, up


def _down_to_canonical(
    down: torch.Tensor,
    gate_up_shape: Optional[Tuple[int, ...]] = None,
    *,
    preferred_split_dim: Optional[int] = None,
) -> torch.Tensor:
    """Return canonical down [E,I,H] for the fused source layout."""
    split_dim = _infer_fused_gate_up_split_dim(
        gate_up_shape or (),
        tuple(down.shape),
        preferred_split_dim=preferred_split_dim,
    )
    if split_dim == 2:
        return down.contiguous().clone()
    return down.transpose(1, 2).contiguous()


def _split_gate_up_bias(gate_up_bias: torch.Tensor, *, interleaved: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split fused gate/up bias into canonical gate/up bias [E,I] tensors."""
    if interleaved:
        return gate_up_bias[..., 0::2].contiguous(), gate_up_bias[..., 1::2].contiguous()
    ffn_dim = gate_up_bias.shape[-1] // 2
    return gate_up_bias[..., :ffn_dim].contiguous(), gate_up_bias[..., ffn_dim:].contiguous()


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

    TRANSFORM_ID = "dtype_conversion_v1"

    @classmethod
    def get_consumed_keys(cls, weight_map: Dict[str, str]) -> set:
        return set(weight_map.keys())

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
    ) -> Dict[str, str]:
        """Convert checkpoint shards to ``target_dtype`` in a prepared output directory."""
        out.mkdir(parents=True, exist_ok=True)

        weight_map = kwargs.pop("weight_map", None) or read_weight_map(src)
        shard_names = sorted(set(weight_map.values()))
        # Always use "base-XXXX.safetensors" naming to avoid colliding with layout
        # transform output shards.  Layout transforms write to original shard names
        # (e.g. "model.safetensors" for single-file checkpoints); using the same name
        # would cause DtypeConversion to overwrite the layout transform's output.
        new_name_for = {
            shard: f"base-{idx:04d}.safetensors"
            for idx, shard in enumerate(shard_names)
        }

        # I/O-bound: one thread per shard, capped at 4× CPU count and hard-capped
        # at 256 — beyond that OS scheduling overhead outweighs I/O parallelism gains.
        n_workers = max_workers if max_workers is not None else min(len(shard_names), cpu_count() * 4, 256)

        shard_keys: Dict[str, set] = {}
        for k, v in weight_map.items():
            shard_keys.setdefault(v, set()).add(k)

        def _process_shard(shard_name: str) -> None:
            allowed = shard_keys[shard_name]
            tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(src / shard_name), framework="pt") as f:
                for key in f.keys():
                    if key not in allowed:
                        continue
                    t = f.get_tensor(key)
                    tensors[key] = t.to(target_dtype) if t.is_floating_point() else t
            atomic_save(tensors, out / new_name_for[shard_name])

        logger.info(
            f"DtypeConversionCheckpointTransform: converting {len(shard_names)} shards "
            f"→ {target_dtype} | workers={n_workers} (cpus={cpu_count()})"
        )
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_shard, s) for s in shard_names]
            for fut in as_completed(futures):
                fut.result()

        new_weight_map = {k: new_name_for[v] for k, v in weight_map.items()}
        logger.info(f"DtypeConversionCheckpointTransform: done → {out}")
        return new_weight_map


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
        # Output in the canonical layout that OptimizedMoETransform creates so
        # promote_initializers_and_build_spec finds an exact checkpoint key match.
        #   _gate [E, I, H] -> transpose(1,2) -> moe_weights.gate [E, H, I]
        #   _up   [E, I, H] -> transpose(1,2) -> moe_weights.up   [E, H, I]
        #   _down [E, H, I] -> transpose(1,2) -> moe_weights.down [E, I, H]
        moe_prefix = _moe_weights_prefix_from_experts_prefix(self.prefix)
        gate = self._gate.to(target_dtype).transpose(1, 2).contiguous()
        up = self._up.to(target_dtype).transpose(1, 2).contiguous()
        down = self._down.to(target_dtype).transpose(1, 2).contiguous()
        return {
            f"{moe_prefix}.gate": gate,  # [E, H, I]
            f"{moe_prefix}.up": up,  # [E, H, I]
            f"{moe_prefix}.down": down,  # [E, I, H]
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

        *.moe_weights.gate [E, H, I]   (gate weights, transposed)
        *.moe_weights.up   [E, H, I]   (up weights, transposed)
        *.moe_weights.down [E, I, H]   (down weights, transposed)

    matching the derived parameter layout that OptimizedMoETransform
    creates, so promote_initializers_and_build_spec finds an exact key match.
    Non-expert keys receive dtype conversion in the same pass.

    Parallelism:

    - Phase 1 (scan):  one thread per shard, reads keys only (I/O bound, cheap).
    - Phase 2 (stack): one thread per layer, loads and stacks its experts.
    - Phase 3 (base):  one thread per shard, converts non-expert keys.

    Phases 2 and 3 run concurrently once phase 1 completes.
    """

    TRANSFORM_ID = "moe_expert_stacking_v1"
    EXPERT_RE = re.compile(
        r"^(.+\.layers\.(\d+)\..+?\.experts)\.(\d+)\."
        r"(gate_proj|up_proj|down_proj|linear|linear_v|linear_1|w1|w2|w3)\.weight$"
    )

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True when the checkpoint uses per-expert MoE tensor keys."""
        return any(cls.EXPERT_RE.match(k) for k in weight_map)

    @classmethod
    def get_consumed_keys(cls, weight_map: Dict[str, str]) -> set:
        return {k for k in weight_map if cls.EXPERT_RE.match(k)}

    @classmethod
    def resolve_onnx_key(cls, onnx_key: str, checkpoint_index: Dict[str, str]) -> Optional[str]:
        """Explicit ONNX → checkpoint key mapping for per-expert MoE models.

        Handles the Mixtral/Qwen3-MoE convention where the ONNX graph names
        the MoE block as ``.mlp.`` but the checkpoint stores it as
        ``.block_sparse_moe.``.
        """
        if onnx_key in checkpoint_index:
            return onnx_key
        candidate = onnx_key.replace(".mlp.", ".block_sparse_moe.")
        if candidate in checkpoint_index:
            return candidate
        return None

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        weight_map: Optional[Dict[str, str]] = None,
        max_workers_layers: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Stack per-expert MoE tensors and convert remaining tensors to ``target_dtype``."""
        out.mkdir(parents=True, exist_ok=True)

        if weight_map is None:
            weight_map = read_weight_map(src)

        # Build expert lookup tables from weight_map keys — no shard files opened.
        # weight_map comes from model.safetensors.index.json which already has
        # all key names; the old Phase 1 shard scan is eliminated.
        #
        # expert_entries[(layer_idx, expert_idx, kind)] = (shard_name, orig_key)
        # layer_prefix[layer_idx]                        = prefix up to .experts
        expert_entries: Dict[Tuple[int, int, str], Tuple[str, str]] = {}
        layer_prefix: Dict[int, str] = {}
        for key, shard_name in weight_map.items():
            m = cls.EXPERT_RE.match(key)
            if m:
                expert_entries[(int(m.group(2)), int(m.group(3)), m.group(4))] = (shard_name, key)
                layer_prefix[int(m.group(2))] = m.group(1)

        experts_per_layer: Dict[int, set] = {}
        for layer_idx, expert_idx, _ in expert_entries:
            experts_per_layer.setdefault(layer_idx, set()).add(expert_idx)
        layer_indices = sorted(experts_per_layer.keys())
        sample_n = len(next(iter(experts_per_layer.values()))) if experts_per_layer else 0
        logger.info(f"  {len(layer_indices)} MoE layers × {sample_n} experts each.")

        new_weight_map: Dict[str, str] = {}

        # Phase 2: parallel layer stacking.
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
            layer_gb = 0.0
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

        logger.info(f"MoEExpertStackingCheckpointTransform: done → {out}")
        return new_weight_map


# ---------------------------------------------------------------------------
# Transform 2b: per-expert stacking + expert-parallel repacking
# ---------------------------------------------------------------------------


class MoEExpertParallelStackingCheckpointTransform(MoEExpertStackingCheckpointTransform):
    """Per-expert stacking + expert-parallel weight repacking for prefill.

    Extends MoEExpertStackingCheckpointTransform by applying
    pack_moe_weights_for_expert_parallel() after stacking, producing the
    [E/P, P, H, I] layout required for the expert_parallel prefill flavour.

    ``P``   = num_pipeline_stages
    ``E/P`` = num_parallelized_experts
    """

    TRANSFORM_ID = "moe_expert_parallel_stacking_v1"

    # Populated by .configured() — never instantiated directly.
    _num_pipeline_stages: int = 1
    _num_parallelized_experts: int = 1

    @classmethod
    def configured(cls, num_pipeline_stages: int, num_parallelized_experts: int):
        """Return a configured subclass with P and E/P baked in."""
        return type(
            f"MoEExpertParallelStackingCheckpointTransform[P={num_pipeline_stages},E_P={num_parallelized_experts}]",
            (cls,),
            {
                "_num_pipeline_stages": num_pipeline_stages,
                "_num_parallelized_experts": num_parallelized_experts,
                "TRANSFORM_ID": "moe_expert_parallel_stacking_v1",
            },
        )

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        weight_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Stack per-expert tensors then repack for expert-parallel execution."""
        from QEfficient.transformers.moe.weights import _pack_expert_parallel_tensor  # noqa: PLC0415

        # Step 1: standard per-expert stacking → moe_weights.gate/up/down [E, H, I]
        new_weight_map = super().apply(src, out, target_dtype=target_dtype, weight_map=weight_map, **kwargs)

        # Step 2: repack [E, H, I] → [E/P, P, H, I] — parallel per output shard.
        # Each stacked layer shard is independent, so we repack in parallel.
        # Repacking is I/O-bound (load shard + write shard) with a small
        # in-memory compute step (view + transpose), so CPU count is the limit.
        unique_shards = sorted(set(new_weight_map.values()))

        def _repack_shard(shard_name: str) -> None:
            shard_path = out / shard_name
            if not shard_path.exists():
                return
            tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(shard_path), framework="pt") as f:
                for k in f.keys():
                    t = f.get_tensor(k)
                    if t.is_floating_point():
                        packed = _pack_expert_parallel_tensor(
                            t,
                            num_pipeline_stages=cls._num_pipeline_stages,
                            num_parallelized_experts=cls._num_parallelized_experts,
                        )
                        tensors[k] = packed.data if hasattr(packed, "data") else packed
                    else:
                        tensors[k] = t
            atomic_save(tensors, shard_path)

        n_workers = max(1, min(len(unique_shards), cpu_count()))
        logger.info(
            f"MoEExpertParallelStackingCheckpointTransform: repacking "
            f"{len(unique_shards)} shards | P={cls._num_pipeline_stages} "
            f"E/P={cls._num_parallelized_experts} | workers={n_workers}..."
        )
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_repack_shard, s) for s in unique_shards]
            for fut in as_completed(futures):
                fut.result()

        logger.info(f"MoEExpertParallelStackingCheckpointTransform: done → {out}")
        return new_weight_map


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

        *.moe_weights.gate      [E, H, I]   (dequant gate_up_proj, first half)
        *.moe_weights.up        [E, H, I]   (dequant gate_up_proj, second half)
        *.moe_weights.gate_bias [E, I]       (gate_up_proj_bias, first half)
        *.moe_weights.up_bias   [E, I]       (gate_up_proj_bias, second half)
        *.moe_weights.down      [E, I, H]   (dequant down_proj)
        *.moe_weights.down_bias [E, H]       (dtype-converted)

    matching the derived parameter layout that OptimizedMoETransform
    creates, so promote_initializers_and_build_spec finds an exact key match.
    Non-expert keys receive dtype conversion in the same pass.

    Parallelism mirrors MoEExpertStackingCheckpointTransform:
    - Phase 1 (scan):    one thread per shard — collect expert tensor locations.
    - Phase 2 (dequant): one thread per layer — dequant, split, write.
    - Phase 3 (base):    one thread per shard — dtype-convert non-expert keys.
    """

    TRANSFORM_ID = "gptoss_mxfp4_dequant_v1"
    _BLOCKS_RE = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_blocks$")

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True when the checkpoint contains GPT-OSS MXFP4 expert blocks."""
        return any(cls._BLOCKS_RE.match(k) for k in weight_map)

    @classmethod
    def get_consumed_keys(cls, weight_map: Dict[str, str]) -> set:
        _SCALES_RE = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_scales$")
        _BIAS_RE = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_bias$")
        return {k for k in weight_map
                if cls._BLOCKS_RE.match(k) or _SCALES_RE.match(k) or _BIAS_RE.match(k)}

    @classmethod
    def resolve_onnx_key(cls, onnx_key: str, checkpoint_index: Dict[str, str]) -> Optional[str]:
        """Direct lookup only — GptOss checkpoint keys match ONNX names directly."""
        return onnx_key if onnx_key in checkpoint_index else None

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        weight_map: Optional[Dict[str, str]] = None,
        max_workers_layers: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Dequantize GPT-OSS MXFP4 experts and split fused expert projections."""
        out.mkdir(parents=True, exist_ok=True)

        if weight_map is None:
            weight_map = read_weight_map(src)

        # Build expert lookup tables from weight_map keys — no shard files opened.
        # Phase 1 shard scan eliminated; model.safetensors.index.json already
        # has all key names so we just classify each key by regex.
        _SCALES_RE = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_scales$")
        _BIAS_RE = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_bias$")

        expert_locs: Dict[Tuple[int, str], Dict] = {}  # {(layer, kind): {blocks/scales: (shard, key)}}
        bias_locs: Dict[Tuple[int, str], Tuple[str, str]] = {}
        layer_prefix: Dict[int, str] = {}

        for key, shard_name in weight_map.items():
            m = cls._BLOCKS_RE.match(key)
            if m:
                li, kind = int(m.group(2)), m.group(3)
                expert_locs.setdefault((li, kind), {})["blocks"] = (shard_name, key)
                layer_prefix[li] = m.group(1)
                continue
            m = _SCALES_RE.match(key)
            if m:
                li, kind = int(m.group(2)), m.group(3)
                expert_locs.setdefault((li, kind), {})["scales"] = (shard_name, key)
                layer_prefix[li] = m.group(1)
                continue
            m = _BIAS_RE.match(key)
            if m:
                li, kind = int(m.group(2)), m.group(3)
                bias_locs[(li, kind)] = (shard_name, key)
                layer_prefix[li] = m.group(1)
                continue

        layer_indices = sorted({li for li, _ in expert_locs})
        logger.info(f"  Found {len(layer_indices)} MoE layers.")

        new_weight_map: Dict[str, str] = {}

        # Phase 2: per-layer dequant + split.
        def _process_layer(layer_idx: int) -> Tuple[str, List[str]]:
            prefix = layer_prefix[layer_idx]
            moe_prefix = _moe_weights_prefix_from_experts_prefix(prefix)
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
            tensors[f"{moe_prefix}.gate"] = gate_up[..., 0::2].contiguous()
            tensors[f"{moe_prefix}.up"] = gate_up[..., 1::2].contiguous()

            # gate_up_proj_bias: split [E, 2*I] → [E, I] + [E, I] (same interleaved convention)
            if (layer_idx, "gate_up_proj") in bias_locs:
                bias_shard, bias_key = bias_locs[(layer_idx, "gate_up_proj")]
                gu_bias = _load(bias_shard, bias_key).to(target_dtype)
                tensors[f"{moe_prefix}.gate_bias"] = gu_bias[..., 0::2].contiguous()
                tensors[f"{moe_prefix}.up_bias"] = gu_bias[..., 1::2].contiguous()

            # down_proj: dequant -> [E, I, H]
            dp_blocks_shard, dp_blocks_key = expert_locs[(layer_idx, "down_proj")]["blocks"]
            dp_scales_shard, dp_scales_key = expert_locs[(layer_idx, "down_proj")]["scales"]
            dp_blocks = _load(dp_blocks_shard, dp_blocks_key)
            dp_scales = _load(dp_scales_shard, dp_scales_key)
            tensors[f"{moe_prefix}.down"] = convert_moe_packed_tensors(dp_blocks, dp_scales, dtype=target_dtype)

            # down_proj_bias: pass through with dtype conversion
            if (layer_idx, "down_proj") in bias_locs:
                dp_bias_shard, dp_bias_key = bias_locs[(layer_idx, "down_proj")]
                tensors[f"{moe_prefix}.down_bias"] = _load(dp_bias_shard, dp_bias_key).to(target_dtype)

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

        logger.info(f"GptOssMxfp4ExpertDequantSplitCheckpointTransform: done → {out}")
        return new_weight_map


# ---------------------------------------------------------------------------
# Transform 3b: GptOss MXFP4 dequant + expert-parallel repacking
# ---------------------------------------------------------------------------


class GptOssMxfp4ExpertDequantExpertParallelCheckpointTransform(GptOssMxfp4ExpertDequantSplitCheckpointTransform):
    """GptOss MXFP4 dequant + expert-parallel weight repacking for prefill.

    Extends GptOssMxfp4ExpertDequantSplitCheckpointTransform by applying
    pack_moe_weights_for_expert_parallel() after dequantization, producing
    the [E/P, P, H, I] layout required for the expert_parallel prefill flavour.

    ``P``   = num_pipeline_stages
    ``E/P`` = num_parallelized_experts
    """

    TRANSFORM_ID = "gptoss_mxfp4_dequant_expert_parallel_v1"

    _num_pipeline_stages: int = 1
    _num_parallelized_experts: int = 1

    @classmethod
    def configured(cls, num_pipeline_stages: int, num_parallelized_experts: int):
        """Return a configured subclass with P and E/P baked in."""
        return type(
            f"GptOssMxfp4ExpertDequantExpertParallelCheckpointTransform"
            f"[P={num_pipeline_stages},E_P={num_parallelized_experts}]",
            (cls,),
            {
                "_num_pipeline_stages": num_pipeline_stages,
                "_num_parallelized_experts": num_parallelized_experts,
                "TRANSFORM_ID": "gptoss_mxfp4_dequant_expert_parallel_v1",
            },
        )

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        weight_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Dequantize GptOss MXFP4 experts then repack for expert-parallel."""
        from QEfficient.transformers.moe.weights import _pack_expert_parallel_tensor  # noqa: PLC0415

        # Step 1: standard dequant → moe_weights.gate/up/down [E, H, I]
        new_weight_map = super().apply(src, out, target_dtype=target_dtype, weight_map=weight_map, **kwargs)

        # Step 2: repack [E, H, I] → [E/P, P, H, I] — parallel per output shard.
        # Identical repacking logic as MoEExpertParallelStackingCheckpointTransform.
        unique_shards = sorted(set(new_weight_map.values()))

        def _repack_shard(shard_name: str) -> None:
            shard_path = out / shard_name
            if not shard_path.exists():
                return
            tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(shard_path), framework="pt") as f:
                for k in f.keys():
                    t = f.get_tensor(k)
                    if t.is_floating_point():
                        packed = _pack_expert_parallel_tensor(
                            t,
                            num_pipeline_stages=cls._num_pipeline_stages,
                            num_parallelized_experts=cls._num_parallelized_experts,
                        )
                        tensors[k] = packed.data if hasattr(packed, "data") else packed
                    else:
                        tensors[k] = t
            atomic_save(tensors, shard_path)

        n_workers = max(1, min(len(unique_shards), cpu_count()))
        logger.info(
            f"GptOssMxfp4ExpertDequantExpertParallelCheckpointTransform: repacking "
            f"{len(unique_shards)} shards | P={cls._num_pipeline_stages} "
            f"E/P={cls._num_parallelized_experts} | workers={n_workers}..."
        )
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_repack_shard, s) for s in unique_shards]
            for fut in as_completed(futures):
                fut.result()

        logger.info(f"GptOssMxfp4ExpertDequantExpertParallelCheckpointTransform: done → {out}")
        return new_weight_map


class FusedExpertSplitCheckpointTransform(BaseCheckpointTransform):
    """Split pre-stacked fused expert tensors into canonical moe_weights layout.

    Handles checkpoints where all experts are stored as one stacked tensor:

        *.experts.gate_up_proj  [E, 2*I, H]   → moe_weights.gate + up
        *.experts.down_proj     [E, H, I]      → moe_weights.down

    Architecture subclasses override CHECKPOINT_KEY_REMAP to translate
    architecture-specific key names to the canonical form above so a single
    apply() implementation handles all fused variants.
    """

    TRANSFORM_ID = "fused_expert_split_v1"

    # Canonical key patterns — all architectures map to these names.
    _FUSED_GATE_UP_RE = re.compile(r"^(.+\.experts)\.gate_up_proj$")
    _FUSED_DOWN_RE = re.compile(r"^(.+\.experts)\.down_proj$")
    _FUSED_GATE_UP_BIAS_RE = re.compile(r"^(.+\.experts)\.gate_up_proj_bias$")
    _FUSED_DOWN_BIAS_RE = re.compile(r"^(.+\.experts)\.down_proj_bias$")

    @classmethod
    def _get_key_remap(cls, weight_map: Dict[str, str]) -> Dict[str, str]:
        """Detect architecture from weight_map keys and return the right remap.

        Uses weight_map key patterns — consistent with the rest of the detection
        design and requires no config or model_type.

        GraniteMoE always uses input_linear/output_linear key names.
        Mixtral fused and others already use canonical experts.gate_up_proj names.
        """
        if any("input_linear.weight" in k for k in weight_map):
            return {
                r"\.input_linear\.weight$":  ".experts.gate_up_proj",
                r"\.output_linear\.weight$": ".experts.down_proj",
            }
        return {}

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        canonical_index, _ = build_canonical_maps(weight_map, cls._get_key_remap(weight_map))
        return any(cls._FUSED_GATE_UP_RE.match(k) for k in canonical_index)

    @classmethod
    def get_consumed_keys(cls, weight_map: Dict[str, str]) -> set:
        """Return the ORIGINAL weight_map keys this transform processes.

        Must return original keys (not canonical), so the pipeline's remaining
        computation correctly excludes them.  For GraniteMoE, this returns
        ``input_linear.weight`` / ``output_linear.weight`` (not the canonical
        ``experts.gate_up_proj`` names).
        """
        key_remap = cls._get_key_remap(weight_map)
        consumed = set()
        for actual_key in weight_map:
            canonical_key = actual_key
            for pattern, replacement in key_remap.items():
                remapped = re.sub(pattern, replacement, actual_key)
                if remapped != actual_key:
                    canonical_key = remapped
                    break
            if (cls._FUSED_GATE_UP_RE.match(canonical_key)
                    or cls._FUSED_DOWN_RE.match(canonical_key)
                    or cls._FUSED_GATE_UP_BIAS_RE.match(canonical_key)
                    or cls._FUSED_DOWN_BIAS_RE.match(canonical_key)):
                consumed.add(actual_key)
        return consumed

    @classmethod
    def resolve_onnx_key(cls, onnx_key: str, checkpoint_index: Dict[str, str]) -> Optional[str]:
        """Explicit ONNX → checkpoint key mapping for fused MoE models.

        Handles the Mixtral convention where the ONNX graph names the MoE
        block as ``.mlp.`` but the checkpoint stores it as ``.block_sparse_moe.``.
        Also handles GraniteMoE which uses canonical names after key remapping.
        """
        if onnx_key in checkpoint_index:
            return onnx_key
        candidate = onnx_key.replace(".mlp.", ".block_sparse_moe.")
        if candidate in checkpoint_index:
            return candidate
        return None

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        weight_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Split fused expert tensors using canonical key mapping."""
        if weight_map is None:
            weight_map = read_weight_map(src)

        # Build canonical maps — two-map approach.
        # For Mixtral (no remap): canonical_index = weight_map, key_translation = {}
        # For GraniteMoE: input_linear.weight → experts.gate_up_proj in canonical_index
        key_remap = cls._get_key_remap(weight_map)
        canonical_index, key_translation = build_canonical_maps(weight_map, key_remap)

        if not any(cls._FUSED_GATE_UP_RE.match(k) for k in canonical_index):
            return {}

        out.mkdir(parents=True, exist_ok=True)
        new_weight_map: Dict[str, str] = {}

        # Group consumed canonical keys by shard for minimal file opens.
        consumed = cls.get_consumed_keys(canonical_index)
        by_shard: Dict[str, List[str]] = {}
        for canonical_key in consumed:
            by_shard.setdefault(canonical_index[canonical_key], []).append(canonical_key)

        # Single pass: load, split, write.
        # Split dim is determined from canonical_index key presence (bias or not)
        # — no shape reads from shard files needed.
        for shard_name, canonical_keys in by_shard.items():
            shard_src = src / shard_name
            if not shard_src.exists():
                continue
            out_tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(shard_src), framework="pt") as f:
                for ck in canonical_keys:
                    actual = key_translation.get(ck, ck)
                    raw = f.get_tensor(actual)
                    tensor = raw.to(target_dtype) if raw.is_floating_point() else raw

                    gate_up_m = cls._FUSED_GATE_UP_RE.match(ck)
                    down_m = cls._FUSED_DOWN_RE.match(ck)
                    gate_up_bias_m = cls._FUSED_GATE_UP_BIAS_RE.match(ck)
                    down_bias_m = cls._FUSED_DOWN_BIAS_RE.match(ck)

                    if gate_up_m:
                        prefix = gate_up_m.group(1)
                        moe_prefix = _moe_weights_prefix_from_experts_prefix(prefix)
                        split_dim = cls._resolve_split_dim(prefix, canonical_index)
                        interleaved = split_dim == 2
                        gate, up = _split_fused_gate_up_to_canonical(
                            tensor, None, interleaved=interleaved, preferred_split_dim=split_dim
                        )
                        out_tensors[f"{moe_prefix}.gate"] = gate
                        out_tensors[f"{moe_prefix}.up"] = up
                        new_weight_map[f"{moe_prefix}.gate"] = shard_name
                        new_weight_map[f"{moe_prefix}.up"] = shard_name

                    elif down_m:
                        prefix = down_m.group(1)
                        moe_prefix = _moe_weights_prefix_from_experts_prefix(prefix)
                        split_dim = cls._resolve_split_dim(prefix, canonical_index)
                        out_tensors[f"{moe_prefix}.down"] = _down_to_canonical(
                            tensor, None, preferred_split_dim=split_dim
                        )
                        new_weight_map[f"{moe_prefix}.down"] = shard_name

                    elif gate_up_bias_m:
                        prefix = gate_up_bias_m.group(1)
                        moe_prefix = _moe_weights_prefix_from_experts_prefix(prefix)
                        gate_bias, up_bias = _split_gate_up_bias(tensor, interleaved=True)
                        out_tensors[f"{moe_prefix}.gate_bias"] = gate_bias
                        out_tensors[f"{moe_prefix}.up_bias"] = up_bias
                        new_weight_map[f"{moe_prefix}.gate_bias"] = shard_name
                        new_weight_map[f"{moe_prefix}.up_bias"] = shard_name

                    elif down_bias_m:
                        prefix = down_bias_m.group(1)
                        moe_prefix = _moe_weights_prefix_from_experts_prefix(prefix)
                        out_tensors[f"{moe_prefix}.down_bias"] = tensor.clone()
                        new_weight_map[f"{moe_prefix}.down_bias"] = shard_name

            if out_tensors:
                save_file({k: v.contiguous() for k, v in out_tensors.items()}, str(out / shard_name))

        return new_weight_map

    @classmethod
    def _resolve_split_dim(cls, prefix: str, canonical_index: Dict[str, str]) -> int:
        """Return split dimension from canonical_index key presence.

        GptOss-MXFP4 has its own transform so FusedExpertSplitCheckpointTransform
        only sees two cases:
          bias present → GptOss dense interleaved → dim=2
          bias absent  → Mixtral or GraniteMoE    → dim=1
        No shape reads needed — canonical_index (from index.json) is sufficient.
        """
        return 2 if f"{prefix}.gate_up_proj_bias" in canonical_index else 1


# Backward-compatible aliases — existing callers keep working.
GraniteMoeFusedExpertSplitCheckpointTransform = FusedExpertSplitCheckpointTransform
MoEFusedExpertSplitCheckpointTransform = FusedExpertSplitCheckpointTransform
