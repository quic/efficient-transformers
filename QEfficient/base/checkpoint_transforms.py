# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Base mechanism for checkpoint file transforms.

Unlike PytorchTransform (base/pytorch_transforms.py), which mutates a live
nn.Module already in memory, a checkpoint transform operates on files on disk
before any model object exists — rewriting a source checkpoint directory into
a prepared one (dtype conversion, MoE expert stacking, etc.) for weight-free
ONNX export. Concrete transforms live alongside the feature that needs them,
e.g. QEfficient/exporter/weight_free/checkpoint_transforms.py.
"""

from pathlib import Path
from typing import Dict, List, Type

import torch

from QEfficient.utils.checkpoint_utils import convert_bin_to_safetensors, read_weight_map

# Marks a prepared checkpoint directory as complete, so re-runs can skip work.
CHECKPOINT_PREPARED_SENTINEL = ".checkpoint_prepared"


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
        if (out / CHECKPOINT_PREPARED_SENTINEL).exists():
            return out

        source_dir = src
        has_safetensors = bool(list(src.glob("*.safetensors"))) or (src / "model.safetensors.index.json").exists()
        if not has_safetensors and list(src.glob("*.bin")):
            source_dir = out.with_name(out.name + "-source-safetensors")
            convert_bin_to_safetensors(src, source_dir)

        weight_map = read_weight_map(source_dir)
        for transform in self.transforms:
            if transform.is_applicable(weight_map, src=source_dir, target_dtype=target_dtype):
                transform.apply(source_dir, out, target_dtype=target_dtype, **kwargs)
                return out
        return source_dir  # no transform applicable — source is already usable as-is
