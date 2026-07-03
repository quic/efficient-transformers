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

from QEfficient.utils.logging_utils import logger

# ---------------------------------------------------------------------------
# System-state helpers — used to derive worker counts at runtime
# ---------------------------------------------------------------------------

def _available_ram_gb() -> float:
    """Available (free + reclaimable) RAM on the current machine in GB."""
    return psutil.virtual_memory().available / 1024 ** 3


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
        (v for (li, ei, k), v in expert_entries.items() if li == layer_idx and k == "gate_proj"),
        None,
    )
    if sample is None:
        return 1.0

    shard_name, orig_key = sample
    try:
        with safe_open(str(src / shard_name), framework="pt") as f:
            sl = f.get_slice(orig_key)
            shape = sl.get_shape()   # [I, H]
            dtype_str = sl.get_dtype()
    except Exception:
        return 1.0

    src_bytes = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1}.get(dtype_str, 2)
    tgt_bytes = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}.get(target_dtype, 4)
    I, H = shape

    # Three input accumulators in source dtype + two output tensors in target dtype
    input_elements  = num_experts * (I * H + I * H + H * I)   # gate + up + down
    output_elements = num_experts * (2 * I * H + H * I)        # gate_up + down_out
    return (input_elements * src_bytes + output_elements * tgt_bytes) / 1024 ** 3

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
        if kind == "gate_proj":
            I, H = tensor.shape
            if self._gate is None:
                self._gate = torch.empty(self.num_experts, I, H, dtype=tensor.dtype)
            self._gate[expert_idx] = tensor
        elif kind == "up_proj":
            I, H = tensor.shape
            if self._up is None:
                self._up = torch.empty(self.num_experts, I, H, dtype=tensor.dtype)
            self._up[expert_idx] = tensor
        else:  # down_proj shape is [H, I]
            H, I = tensor.shape
            if self._down is None:
                self._down = torch.empty(self.num_experts, H, I, dtype=tensor.dtype)
            self._down[expert_idx] = tensor

    def stack(self, target_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        # Output in the exact layout that model __qeff_init__ creates so
        # _promote_initializers_and_build_spec finds an exact checkpoint key match.
        #   _gate [E, I, H] → transpose(1,2) → gate_proj  [E, H, I]
        #   _up   [E, I, H] → transpose(1,2) → up_proj    [E, H, I]
        #   _down [E, H, I] → transpose(1,2) → down_proj_t [E, I, H]
        gate_proj   = self._gate.to(target_dtype).transpose(1, 2).contiguous()
        up_proj     = self._up.to(target_dtype).transpose(1, 2).contiguous()
        down_proj_t = self._down.to(target_dtype).transpose(1, 2).contiguous()
        return {
            f"{self.prefix}.gate_proj":   gate_proj,    # [E, H, I]
            f"{self.prefix}.up_proj":     up_proj,      # [E, H, I]
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
        r"^(.+\.layers\.(\d+)\..+?\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
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
        n_workers_scan = max_workers_scan if max_workers_scan is not None else min(len(shard_names), _cpu_count() * 4, 256)
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

            by_shard: Dict[str, List[Tuple[int, str, str]]] = {}
            for exp_idx in range(num_exp):
                for kind in ("gate_proj", "up_proj", "down_proj"):
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
        new_base_name_for = {
            shard: f"base-{idx:04d}.safetensors"
            for idx, shard in enumerate(base_shard_list)
        }

        def _convert_base(shard_name: str, keys: List[str]) -> None:
            tensors: Dict[str, torch.Tensor] = {}
            with safe_open(str(src / shard_name), framework="pt") as f:
                for key in keys:
                    t = f.get_tensor(key)
                    tensors[key] = t.to(target_dtype) if t.is_floating_point() else t
            _atomic_save(tensors, out / new_base_name_for[shard_name])

        # Phase 3: mixed I/O + memory — one thread per shard, capped at CPU count.
        n_workers_base = max_workers_base if max_workers_base is not None else min(len(base_shard_list), _cpu_count())
        logger.info(
            f"  Converting {len(base_shard_list)} base shards → {target_dtype} | "
            f"workers={n_workers_base}..."
        )
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
# Pipeline
# ---------------------------------------------------------------------------

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
        weight_map = _read_weight_map(src)
        for transform in self.transforms:
            if transform.is_applicable(weight_map):
                transform.apply(src, out, target_dtype=target_dtype, **kwargs)
                return out
        return src  # no transform applicable — source is already usable as-is
