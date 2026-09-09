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

from QEfficient.utils.checkpoint_utils import read_weight_map

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


def detect_group_transform_id(
    config,
    weight_map: Dict[str, str],
    hash_params: Optional[Dict] = None,
) -> str:
    """Return a stable string ID for the group transform that applies to this checkpoint.

    Used as part of the prepared-checkpoint cache key so that different model
    flavours (decode vs expert_parallel, dense vs MoE, different quantizations)
    always hash to different prepared directories and never overwrite each other.

    The IDs returned here match the TRANSFORM_ID strings that will be declared
    on each group transform class in the full pipeline redesign.

    Parameters
    ----------
    config
        HuggingFace model config.  ``num_local_experts`` / ``num_experts``
        gates all MoE detection — absent means dense model.
    weight_map
        ``{tensor_key: shard_filename}`` from ``model.safetensors.index.json``.
    hash_params
        Model hash parameters (from ``qeff_model.hash_params``).  Used to
        detect the ``expert_parallel`` prefill flavour which requires a
        different weight layout than standard decode.

    Returns
    -------
    str
        One of the stable transform ID strings, or ``"none"`` for dense models.
    """
    if hash_params is None:
        hash_params = {}

    num_experts = None
    if config is not None:
        num_experts = getattr(config, "num_local_experts", None) or getattr(config, "num_experts", None)
    if not num_experts:
        return "none"

    # Per-expert format: validate expert indices match config declaration.
    # Keys look like: model.layers.0.block_sparse_moe.experts.0.w1.weight
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
            return "moe_expert_parallel_stacking_v1"
        return "moe_expert_stacking_v1"

    # Pre-stacked formats detected purely from key patterns.
    quant_config = getattr(config, "quantization_config", None) if config else None
    if quant_config and any("_blocks" in k for k in weight_map):
        return "gptoss_mxfp4_dequant_v1"

    if any("input_linear.weight" in k and ".experts." not in k for k in weight_map):
        return "granite_moe_fused_split_v1"

    if any(".experts.gate_up_proj" in k for k in weight_map):
        return "moe_fused_expert_split_v1"

    # Config declares MoE but no known format found — treat as unknown.
    return "none"


class BaseCheckpointTransform:
    """Base class for checkpoint file transforms. Not to be instantiated.

    Each subclass produces a *complete* prepared checkpoint directory in ``out``.
    The pipeline picks the first applicable transform and stops — no chaining.
    """

    def __init__(self):
        """Prevent direct instantiation of transform marker classes."""
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
        """Apply the first matching transform and return the usable checkpoint directory."""
        src, out = Path(src), Path(out)

        if list(src.glob("*.bin")):
            raise ValueError(
                f"Checkpoint at {src} contains .bin files. "
                "Weight-free export requires safetensors format. "
                "Convert the checkpoint to safetensors before exporting."
            )

        source_dir = src
        weight_map = read_weight_map(source_dir)
        active_group_id = detect_group_transform_id(
            kwargs.pop("config", None), weight_map, kwargs.pop("hash_params", None)
        )
        expected_manifest = _checkpoint_manifest(src, target_dtype, self.transforms, active_group_id)
        if (out / CHECKPOINT_PREPARED_SENTINEL).exists() and _manifest_matches(out, expected_manifest):
            return out
        _clear_stale_prepared_dir(out, src)

        for transform in self.transforms:
            if transform.is_applicable(weight_map, src=source_dir, target_dtype=target_dtype):
                transform.apply(source_dir, out, target_dtype=target_dtype, weight_map=weight_map, **kwargs)
                if (out / CHECKPOINT_PREPARED_SENTINEL).exists():
                    _write_manifest(out, expected_manifest)
                return out
        return source_dir  # no transform applicable - source is already usable as-is
