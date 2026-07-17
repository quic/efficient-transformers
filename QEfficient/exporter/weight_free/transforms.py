#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#

"""
Checkpoint transforms for weight-free ONNX export.

Two transforms are provided, selected in priority order by CheckpointTransformPipeline:

  MoEExpertStackingCheckpointTransform  — stacks per-expert HF keys into batched
    tensors AND converts dtype, all in a single read pass over the checkpoint.
    Skipped automatically (is_applicable=False) for dense models.

  DtypeConversionCheckpointTransform    — converts all floating-point tensors to
    target_dtype. Used as the fallback path for dense models.

Usage on model classes (mirrors _pytorch_transforms / _onnx_transforms)::

    _checkpoint_transforms = [
        MoEExpertStackingCheckpointTransform,   # no-op for dense (is_applicable=False)
        DtypeConversionCheckpointTransform,     # fallback for dense models
    ]
"""

import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import psutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from QEfficient.transformers.quantizers.quantizer_utils import convert_moe_packed_tensors
from QEfficient.utils.logging_utils import logger

# ---------------------------------------------------------------------------
# System-state helpers — used to derive worker counts at runtime
# ---------------------------------------------------------------------------


def _available_ram_gb() -> float:
    """Available (free + reclaimable) RAM on the current machine in GB."""
    return psutil.virtual_memory().available / 1024**3


def _cpu_count() -> int:
    """Logical CPU count with a safe fallback."""
    return os.cpu_count() or 8


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
# Auxiliary file names copied alongside the prepared checkpoint
# ---------------------------------------------------------------------------
_AUX_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
]

_SENTINEL = ".checkpoint_prepared"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _convert_bin_to_safetensors(src: Path) -> None:
    """Load a .bin checkpoint via transformers and re-save as safetensors in place.

    Uses save_pretrained(safe_serialization=True) which correctly handles tied
    weights and multi-shard layouts, writing model.safetensors (single file) or
    model-NNNNN-of-MMMMM.safetensors + model.safetensors.index.json (multi-shard).
    Idempotent — the caller already verified no safetensors files exist before calling.
    """
    import gc

    from transformers import AutoConfig, AutoModelForCausalLM

    logger.info(f"No safetensors files found in {src}. Auto-converting .bin → safetensors (one-time).")
    config = AutoConfig.from_pretrained(str(src), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(src),
        config=config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.save_pretrained(str(src), safe_serialization=True)
    del model
    gc.collect()
    logger.info(f"Conversion complete — safetensors files written to {src}")


def _read_weight_map(src: Path) -> Dict[str, str]:
    """Return {tensor_key: shard_filename} from model.safetensors.index.json,
    or by scanning all *.safetensors for single-file checkpoints."""
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())["weight_map"]
    shard_files = sorted(src.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files found in {src}")
    weight_map: Dict[str, str] = {}
    for sf in shard_files:
        with safe_open(str(sf), framework="pt") as f:
            for k in f.keys():
                weight_map[k] = sf.name
    return weight_map


def _atomic_save(tensors: Dict[str, torch.Tensor], dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp))
    tmp.replace(dst)


def _write_index(out: Path, weight_map: Dict[str, str]) -> None:
    files = set(weight_map.values())
    total_size = sum((out / f).stat().st_size for f in files if (out / f).exists())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def _copy_aux_files(src: Path, out: Path) -> None:
    for name in _AUX_FILES:
        src_file = src / name
        if src_file.exists() and not (out / name).exists():
            shutil.copy2(str(src_file), str(out / name))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseCheckpointTransform:
    """Base class for checkpoint file transforms. Not to be instantiated.

    Each subclass produces a *complete* prepared checkpoint directory in ``out``.
    The pipeline picks the first applicable transform and stops — no chaining.
    """

    def __init__(self):
        raise TypeError("Checkpoint transform classes are not to be instantiated.")

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> bool:
        """Transform checkpoint at ``src``, write result to ``out``.
        Returns True if the checkpoint was prepared, False if skipped (idempotent)."""
        raise NotImplementedError

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str]) -> bool:
        """Return True if this transform should run for the given checkpoint."""
        return True


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
            logger.info("DtypeConversionCheckpointTransform: prepared checkpoint exists, skipping.")
            return False

        out.mkdir(parents=True, exist_ok=True)
        _copy_aux_files(src, out)

        weight_map = _read_weight_map(src)
        shard_names = sorted(set(weight_map.values()))
        new_name_for = {
            shard: (f"model_{idx:04d}.safetensors" if len(shard_names) > 1 else "model.safetensors")
            for idx, shard in enumerate(shard_names)
        }

        # I/O-bound: one thread per shard, capped at 4× CPU count and hard-capped
        # at 256 — beyond that OS scheduling overhead outweighs I/O parallelism gains.
        n_workers = max_workers if max_workers is not None else min(len(shard_names), _cpu_count() * 4, 256)

        def _process_shard(shard_name: str) -> None:
            tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(src / shard_name), framework="pt") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    tensors[key] = t.to(target_dtype) if t.is_floating_point() else t
            _atomic_save(tensors, out / new_name_for[shard_name])

        logger.info(
            f"DtypeConversionCheckpointTransform: converting {len(shard_names)} shards "
            f"→ {target_dtype} | workers={n_workers} (cpus={_cpu_count()})"
        )
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_shard, s) for s in shard_names]
            for fut in as_completed(futures):
                fut.result()

        new_weight_map = {k: new_name_for[v] for k, v in weight_map.items()}
        _write_index(out, new_weight_map)
        sentinel.touch()
        logger.info(f"DtypeConversionCheckpointTransform: done → {out}")
        return True


# ---------------------------------------------------------------------------
# Internal stacker helper for MoE layers
# ---------------------------------------------------------------------------


class _LayerStacker:
    """Accumulates per-expert tensors for one MoE layer and produces batched output."""

    def __init__(self, prefix: str, num_experts: int):
        self.prefix = prefix
        self.num_experts = num_experts
        self._gate: Optional[torch.Tensor] = None
        self._up: Optional[torch.Tensor] = None
        self._down: Optional[torch.Tensor] = None

    def add(self, expert_idx: int, kind: str, tensor: torch.Tensor) -> None:
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
        # Output in the exact layout that model __qeff_init__ creates so
        # _promote_initializers_and_build_spec finds an exact checkpoint key match.
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
    creates, so _promote_initializers_and_build_spec finds an exact key match.
    Non-expert keys receive dtype conversion in the same pass.

    Parallelism:

    - Phase 1 (scan):  one thread per shard, reads keys only (I/O bound, cheap).
    - Phase 2 (stack): one thread per layer, loads and stacks its experts.
    - Phase 3 (base):  one thread per shard, converts non-expert keys.

    Phases 2 and 3 run concurrently once phase 1 completes.
    """

    EXPERT_RE = re.compile(
        r"^(.+\.layers\.(\d+)\..+?\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj|linear|linear_v|linear_1|w1|w2|w3)\.weight$"
    )

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str]) -> bool:
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
        sentinel = out / _SENTINEL
        if sentinel.exists():
            logger.info("MoEExpertStackingCheckpointTransform: prepared checkpoint exists, skipping.")
            return False

        out.mkdir(parents=True, exist_ok=True)
        _copy_aux_files(src, out)

        weight_map = _read_weight_map(src)
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
            max_workers_scan if max_workers_scan is not None else min(len(shard_names), _cpu_count() * 4, 256)
        )
        logger.info(
            f"MoEExpertStackingCheckpointTransform: scanning {len(shard_names)} shards "
            f"(workers={n_workers_scan}, cpus={_cpu_count()}, ram_avail={_available_ram_gb():.1f} GB)..."
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
            _atomic_save(stacked, out / out_name)
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
            available_gb = _available_ram_gb()
            usable_gb = available_gb * 0.8
            n_workers_layers = max(1, min(len(layer_indices), int(usable_gb / layer_gb)))
        else:
            n_workers_layers = 1
            layer_gb = 0.0

        logger.info(
            f"  Stacking {len(layer_indices)} layers → {target_dtype} | "
            f"workers={n_workers_layers} (~{layer_gb:.2f} GB/layer, "
            f"{_available_ram_gb():.1f} GB available)..."
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
            _atomic_save(tensors, out / new_base_name_for[shard_name])

        # Phase 3: mixed I/O + memory — one thread per shard, capped at CPU count.
        n_workers_base = max_workers_base if max_workers_base is not None else min(len(base_shard_list), _cpu_count())
        logger.info(f"  Converting {len(base_shard_list)} base shards → {target_dtype} | workers={n_workers_base}...")
        with ThreadPoolExecutor(max_workers=n_workers_base) as ex:
            futures_base = [ex.submit(_convert_base, s, keys) for s, keys in by_shard_base.items()]
            for fut in as_completed(futures_base):
                fut.result()

        for key, shard_name in base_entries.items():
            new_weight_map[key] = new_base_name_for[shard_name]

        _write_index(out, new_weight_map)
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
    creates, so _promote_initializers_and_build_spec finds an exact key match.
    Non-expert keys receive dtype conversion in the same pass.

    Parallelism mirrors MoEExpertStackingCheckpointTransform:
    - Phase 1 (scan):    one thread per shard — collect expert tensor locations.
    - Phase 2 (dequant): one thread per layer — dequant, split, write.
    - Phase 3 (base):    one thread per shard — dtype-convert non-expert keys.
    """

    _BLOCKS_RE = re.compile(r"^(.+\.layers\.(\d+)\..+?\.experts)\.(gate_up_proj|down_proj)_blocks$")

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str]) -> bool:
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
        sentinel = out / _SENTINEL
        if sentinel.exists():
            logger.info("GptOssMxfp4ExpertDequantSplitCheckpointTransform: prepared checkpoint exists, skipping.")
            return False

        out.mkdir(parents=True, exist_ok=True)
        _copy_aux_files(src, out)

        weight_map = _read_weight_map(src)
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

        n_scan = max_workers_scan if max_workers_scan is not None else min(len(shard_names), _cpu_count() * 4, 256)
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
            _atomic_save(tensors, out / out_name)
            return out_name, list(tensors.keys())

        n_layers = (
            max_workers_layers if max_workers_layers is not None else max(1, min(len(layer_indices), _cpu_count()))
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
            _atomic_save(tensors, out / new_base_name_for[shard_name])

        n_base = max_workers_base if max_workers_base is not None else min(len(base_shard_list), _cpu_count())
        logger.info(f"  Converting {len(base_shard_list)} base shards | workers={n_base}...")
        with ThreadPoolExecutor(max_workers=n_base) as ex:
            futures_base = [ex.submit(_convert_base, s, keys) for s, keys in by_shard_base.items()]
            for fut in as_completed(futures_base):
                fut.result()

        for key, shard_name in base_entries.items():
            new_weight_map[key] = new_base_name_for[shard_name]

        _write_index(out, new_weight_map)
        sentinel.touch()
        logger.info(f"GptOssMxfp4ExpertDequantSplitCheckpointTransform: done → {out}")
        return True


# ---------------------------------------------------------------------------
# Pipeline
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

    _FUSED_GATE_UP_RE = re.compile(
        r"^(.+\.experts)\.gate_up_proj$"
    )
    _FUSED_DOWN_RE = re.compile(
        r"^(.+\.experts)\.down_proj$"
    )

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str]) -> bool:
        return any(cls._FUSED_GATE_UP_RE.match(k) for k in weight_map)

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> bool:
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

        new_weight_map: Dict[str, str] = {}
        for shard_name in sorted(set(weight_map.values())):
            shard_src = src / shard_name
            if not shard_src.exists():
                continue

            out_tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(shard_src), framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key).to(target_dtype)
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
        return True


class CheckpointTransformPipeline:
    """Selects and runs the first applicable checkpoint transform.

    Transforms are priority-ordered. The first one whose ``is_applicable()``
    returns True is executed and the pipeline stops. Each transform produces a
    complete prepared checkpoint — there is no chaining between transforms.

    Example::

        pipeline = CheckpointTransformPipeline([
            MoEExpertStackingCheckpointTransform,   # MoE models: stacks + converts
            DtypeConversionCheckpointTransform,     # dense models: converts only
        ])
        prepared_dir = pipeline.apply(src, out, target_dtype=torch.float32)
    """

    def __init__(self, transforms: List[Type[BaseCheckpointTransform]]):
        self.transforms = transforms

    def apply(
        self,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> Path:
        src, out = Path(src), Path(out)

        # Auto-convert .bin checkpoints to safetensors on first use.
        # Idempotent: skipped on subsequent runs once safetensors files exist.
        has_safetensors = bool(list(src.glob("*.safetensors"))) or (src / "model.safetensors.index.json").exists()
        if not has_safetensors and list(src.glob("*.bin")):
            _convert_bin_to_safetensors(src)

        weight_map = _read_weight_map(src)
        for transform in self.transforms:
            if transform.is_applicable(weight_map):
                transform.apply(src, out, target_dtype=target_dtype, **kwargs)
                return out
        return src  # no transform applicable — source is already usable as-is
