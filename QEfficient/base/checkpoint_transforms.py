# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Base mechanism for checkpoint file transforms.

Unlike PytorchTransform (base/pytorch_transforms.py), which mutates a live
nn.Module already in memory, a checkpoint transform operates on files on disk
before any model object exists - rewriting a source checkpoint directory into
a prepared one (dtype conversion, MoE expert stacking, etc.) for weight-free
ONNX export. Concrete transforms live alongside the feature that needs them,
e.g. QEfficient/exporter/weight_free/checkpoint_transforms.py.
"""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Type

import torch

from QEfficient.utils.checkpoint_utils import copy_checkpoint_aux_files, read_weight_map, write_index

# Marks a prepared checkpoint directory as complete, so re-runs can skip work.
CHECKPOINT_PREPARED_SENTINEL = ".checkpoint_prepared"
CHECKPOINT_PREPARED_MANIFEST = ".checkpoint_prepared.json"


def _checkpoint_files(root: Path) -> List[Path]:
    patterns = ("*.safetensors", "*.bin", "*.json")
    files = set()
    for pattern in patterns:
        files.update(root.glob(pattern))
    return sorted(files)


def _checkpoint_file_fingerprint(root: Path) -> List[dict]:
    return [
        {"path": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in _checkpoint_files(root)
    ]


def _checkpoint_manifest(
    src: Path,
    target_dtype: torch.dtype,
    transforms: List[Type["BaseCheckpointTransform"]],
    active_group_id: str = "none",
) -> dict:
    return {
        "version": 2,
        "source": str(src.resolve()),
        "target_dtype": str(target_dtype),
        "active_group": active_group_id,
        "transforms": [f"{t.__module__}.{t.__name__}" for t in transforms],
        "files": _checkpoint_file_fingerprint(src),
    }


def _manifest_matches(out: Path, expected: dict) -> bool:
    manifest_path = out / CHECKPOINT_PREPARED_MANIFEST
    if not manifest_path.is_file():
        return False
    try:
        return json.loads(manifest_path.read_text()) == expected
    except (OSError, json.JSONDecodeError):
        return False


def _write_manifest(out: Path, manifest: dict) -> None:
    (out / CHECKPOINT_PREPARED_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _clear_stale_prepared_dir(out: Path, src: Path) -> None:
    if not out.exists() or out == src:
        return
    if out.is_dir():
        shutil.rmtree(out)
    else:
        out.unlink()


def detect_group_transform(
    config,
    weight_map: Dict[str, str],
    hash_params: Optional[Dict] = None,
    transforms: Optional[List] = None,
) -> Optional[Type["BaseCheckpointTransform"]]:
    """Return the active layout transform class, or None for dense models.

    Scans weight_map key patterns (gated by config.num_experts) to identify
    which layout transform applies.  DtypeConversionCheckpointTransform is
    excluded — it always runs unconditionally and is not a layout transform.

    Parameters
    ----------
    config
        HuggingFace model config.  ``num_local_experts`` / ``num_experts``
        gates all MoE detection — absent means dense model.
    weight_map
        ``{tensor_key: shard_filename}`` from ``model.safetensors.index.json``.
    hash_params
        Model hash parameters (from ``qeff_model.hash_params``).
    transforms
        Registered transforms list — used to look up the class by TRANSFORM_ID.
        Falls back to direct imports when not provided.

    Returns
    -------
    Type[BaseCheckpointTransform] or None
        The active layout transform class, or ``None`` for dense models.
    """
    if hash_params is None:
        hash_params = {}

    num_experts = None
    if config is not None:
        num_experts = getattr(config, "num_local_experts", None) or getattr(config, "num_experts", None)
    if not num_experts:
        return None

    # Per-expert format: validate expert indices match config declaration.
    expert_indices_per_layer: Dict[int, set] = defaultdict(set)
    for k in weight_map:
        m = re.search(r"\.layers\.(\d+)\..*\.experts\.(\d+)\.", k)
        if m:
            expert_indices_per_layer[int(m.group(1))].add(int(m.group(2)))

    if expert_indices_per_layer:
        expected = set(range(num_experts))
        for layer_idx, found in expert_indices_per_layer.items():
            if found != expected:
                raise ValueError(
                    f"Layer {layer_idx}: config declares {num_experts} experts "
                    f"but checkpoint contains indices {sorted(found)}. "
                    "The checkpoint may be incomplete or corrupted."
                )
        if hash_params.get("moe_prefill_flavour") == "expert_parallel":
            p = hash_params.get("moe_prefill_num_pipeline_stages")
            e_p = hash_params.get("moe_prefill_num_parallelized_experts")
            if p is None or e_p is None:
                raise ValueError(
                    "expert_parallel flavour requires moe_prefill_num_pipeline_stages "
                    "and moe_prefill_num_parallelized_experts in hash_params."
                )
            from QEfficient.exporter.weight_free.checkpoint_transforms import (  # noqa: PLC0415
                MoEExpertParallelStackingCheckpointTransform,
            )
            return MoEExpertParallelStackingCheckpointTransform.configured(int(p), int(e_p))
        return _find_transform_by_id("moe_expert_stacking_v1", transforms)

    # Pre-stacked formats — delegate detection to each transform's is_applicable().
    # FusedExpertSplitCheckpointTransform handles both Mixtral fused and GraniteMoE
    # internally via _get_key_remap() — no hardcoded patterns needed here.
    quant_config = getattr(config, "quantization_config", None) if config else None
    if quant_config and any("_blocks" in k for k in weight_map):
        return _find_transform_by_id("gptoss_mxfp4_dequant_v1", transforms)

    fused_cls = _find_transform_by_id("fused_expert_split_v1", transforms)
    if fused_cls is not None and fused_cls.is_applicable(weight_map):
        return fused_cls

    return None


def _find_transform_by_id(
    transform_id: str,
    transforms: Optional[List],
) -> Optional[Type["BaseCheckpointTransform"]]:
    """Return the transform class with matching TRANSFORM_ID from the list."""
    if transforms:
        for t in transforms:
            if getattr(t, "TRANSFORM_ID", None) == transform_id:
                return t
    # Fallback: import directly when transforms list not provided
    from QEfficient.exporter.weight_free.checkpoint_transforms import (  # noqa: PLC0415
        DtypeConversionCheckpointTransform,
        FusedExpertSplitCheckpointTransform,
        GptOssMxfp4ExpertDequantSplitCheckpointTransform,
        MoEExpertStackingCheckpointTransform,
    )

    _ID_MAP = {
        "moe_expert_stacking_v1":          MoEExpertStackingCheckpointTransform,
        "gptoss_mxfp4_dequant_v1":         GptOssMxfp4ExpertDequantSplitCheckpointTransform,
        "fused_expert_split_v1":           FusedExpertSplitCheckpointTransform,
        "moe_fused_expert_split_v1":       FusedExpertSplitCheckpointTransform,
        "granite_moe_fused_split_v1":      FusedExpertSplitCheckpointTransform,
        "dtype_conversion_v1":             DtypeConversionCheckpointTransform,
    }
    return _ID_MAP.get(transform_id)


class BaseCheckpointTransform:
    """Base class for checkpoint file transforms. Not to be instantiated.

    Each subclass declares:
    * ``TRANSFORM_ID`` — stable string used in the cache hash and for detection.
    * ``get_consumed_keys()`` — which checkpoint keys this transform processes.
    * ``apply()`` — performs the transform, returns ``{new_key: shard_file}``.
    """

    TRANSFORM_ID: str = ""

    def __init__(self):
        """Prevent direct instantiation of transform marker classes."""
        raise TypeError("Checkpoint transform classes are not to be instantiated.")

    @classmethod
    def apply(
        cls,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        weight_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Transform checkpoint tensors, write output shards, return new weight map entries."""
        raise NotImplementedError

    @classmethod
    def get_consumed_keys(cls, weight_map: Dict[str, str]) -> set:
        """Return the set of weight_map keys this transform will process."""
        return set(weight_map.keys())

    @classmethod
    def is_applicable(cls, weight_map: Dict[str, str], **kwargs) -> bool:
        """Return True if this transform should run for the given checkpoint."""
        return True


class CheckpointTransformPipeline:
    """Selects and runs the first applicable checkpoint transform.

    Transforms are priority-ordered. The first one whose ``is_applicable()``
    returns True is executed and the pipeline stops. Each transform produces a
    complete prepared checkpoint — there is no chaining between transforms.

    TODO(wf): Current implementation is a selector but is named as pipeline.
    Correct design is to apply multiple transforms sequentially on the tensors that it applies to
    and we parallelize this processing across all tensors.
    Each transform should have single transformation responsibility.
    Currently all transforms copy multiple responsibitlies from each-other. This is not scalable.

    Example::

        pipeline = CheckpointTransformPipeline([
            MoEExpertStackingCheckpointTransform,   # MoE models: stacks + converts
            DtypeConversionCheckpointTransform,     # dense models: converts only
        ])
        prepared_dir = pipeline.apply(src, out, target_dtype=torch.float32)
    """

    def __init__(self, transforms: List[Type[BaseCheckpointTransform]]):
        """Create a priority-ordered checkpoint transform pipeline."""
        self.transforms = transforms

    def apply(
        self,
        src: Path,
        out: Path,
        target_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> Path:
        """Prepare checkpoint at ``src`` into ``out`` and return the usable directory."""
        src, out = Path(src), Path(out)

        # ① VALIDATE
        if list(src.glob("*.bin")):
            raise ValueError(
                f"Checkpoint at {src} contains .bin files. "
                "Weight-free export requires safetensors format. "
                "Convert the checkpoint to safetensors before exporting."
            )

        source_dir = src
        weight_map = read_weight_map(source_dir)

        # ② CACHE CHECK
        config = kwargs.pop("config", None)
        hash_params = kwargs.pop("hash_params", None) or {}
        active_transform = detect_group_transform(config, weight_map, hash_params, self.transforms)
        active_group_id = active_transform.TRANSFORM_ID if active_transform else "none"
        expected_manifest = _checkpoint_manifest(src, target_dtype, self.transforms, active_group_id)
        if (out / CHECKPOINT_PREPARED_SENTINEL).exists() and _manifest_matches(out, expected_manifest):
            return out
        _clear_stale_prepared_dir(out, src)
        out.mkdir(parents=True, exist_ok=True)

        # ④ EXECUTE — layout transform then dtype conversion
        from QEfficient.exporter.weight_free.checkpoint_transforms import (  # noqa: PLC0415
            DtypeConversionCheckpointTransform,
        )
        transforms_to_run = []
        if active_transform is not None:
            transforms_to_run.append(active_transform)
        transforms_to_run.append(DtypeConversionCheckpointTransform)

        new_weight_map: Dict[str, str] = {}
        consumed: set = set()
        for transform in transforms_to_run:
            remaining = {k: v for k, v in weight_map.items() if k not in consumed}
            result = transform.apply(
                source_dir, out, target_dtype=target_dtype, weight_map=remaining, **kwargs
            )
            if isinstance(result, dict):
                new_weight_map.update(result)
            consumed.update(transform.get_consumed_keys(weight_map))

        # ⑤ FINALISE
        copy_checkpoint_aux_files(source_dir, out)
        write_index(out, new_weight_map)
        _write_manifest(out, expected_manifest)
        (out / CHECKPOINT_PREPARED_SENTINEL).touch()
        return out
