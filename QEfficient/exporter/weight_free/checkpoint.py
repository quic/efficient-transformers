# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from safetensors import safe_open

from QEfficient.utils._utils import hf_download


@lru_cache(maxsize=None)
def resolve_checkpoint_dir(model_id_or_path: str) -> Path:
    """Resolve a local or remote model reference to a checkpoint directory.

    Parameters
    ----------
    model_id_or_path : str
        Local model path or Hugging Face Hub model id.

    Returns
    -------
    Path
        Directory containing the downloaded or local checkpoint files.
    """
    candidate = Path(model_id_or_path).expanduser()
    if candidate.exists():
        return candidate

    snapshot_dir = hf_download(
        repo_id=model_id_or_path,
        allow_patterns=["*.safetensors", "*.json"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.txt", "*.pdf", "*.msgpack", "*.h5", "*.pth"],
    )
    snapshot_path = Path(snapshot_dir)
    has_weights = (
        bool(list(snapshot_path.glob("*.safetensors"))) or (snapshot_path / "model.safetensors.index.json").exists()
    )

    if not has_weights:
        snapshot_dir = hf_download(
            repo_id=model_id_or_path,
            allow_patterns=["*.bin", "*.json"],
            ignore_patterns=[
                "*.onnx",
                "*.ot",
                "*.md",
                "*.txt",
                "*.pdf",
                "*.msgpack",
                "*.h5",
                "*.pth",
                "flax_model*",
                "tf_model*",
            ],
        )

    return Path(snapshot_dir)


def resolve_checkpoint_files(model_id_or_path: str) -> List[str]:
    """Return safetensors checkpoint files for a model reference.

    Parameters
    ----------
    model_id_or_path : str
        Local model path or Hugging Face Hub model id.

    Returns
    -------
    List[str]
        Sorted safetensors shard paths.
    """
    checkpoint_dir = resolve_checkpoint_dir(model_id_or_path)
    checkpoint_files = sorted(str(path) for path in checkpoint_dir.glob("*.safetensors"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No safetensors checkpoint files found for {model_id_or_path}")
    return checkpoint_files


def checkpoint_root(model_id_or_path: str, checkpoint_files: Sequence[str]) -> Optional[Path]:
    """Return the root directory used for relative checkpoint file paths.

    Parameters
    ----------
    model_id_or_path : str
        Local model path or Hugging Face Hub model id.
    checkpoint_files : Sequence[str]
        Checkpoint shard paths resolved for the model.

    Returns
    -------
    Optional[Path]
        Parent root for relative paths, or ``None`` when no checkpoint files exist.
    """
    if not checkpoint_files:
        return None

    candidate = Path(model_id_or_path).expanduser()
    if candidate.exists():
        return candidate.parent

    first_checkpoint = Path(checkpoint_files[0])
    for parent in first_checkpoint.parents:
        if parent.name.startswith("models--"):
            return parent.parent
    return first_checkpoint.parent


def load_checkpoint_index(checkpoint_files: List[str]) -> Dict[str, str]:
    """Build a tensor-to-shard map by scanning safetensors checkpoint files.

    Parameters
    ----------
    checkpoint_files : List[str]
        Safetensors shard paths.

    Returns
    -------
    Dict[str, str]
        Mapping from checkpoint tensor key to the shard path containing it.
    """
    tensor_to_file = {}
    for checkpoint_file in checkpoint_files:
        with safe_open(checkpoint_file, framework="pt") as handle:
            for key in handle.keys():
                tensor_to_file[key] = checkpoint_file
    return tensor_to_file


def prepare_checkpoint_for_weight_free_export(
    qeff_model,
    model_ref: str,
    target_dtype: torch.dtype,
) -> str:
    """Prepare a checkpoint directory for weight-free ONNX export.

    Parameters
    ----------
    qeff_model
        QEfficient model wrapper that provides checkpoint transform classes.
    model_ref : str
        Local model path or Hugging Face Hub model id.
    target_dtype : torch.dtype
        Floating-point dtype expected by the exported ONNX graph.

    Returns
    -------
    str
        Path to the prepared checkpoint directory.
    """
    from QEfficient.exporter.weight_free.checkpoint_transforms import CheckpointTransformPipeline

    source_dir = resolve_checkpoint_dir(model_ref)
    dtype_suffix = str(target_dtype).replace("torch.", "")
    prepared_out = source_dir.parent / (source_dir.name + f"-qeff-prepared-{dtype_suffix}")
    prep_pipeline = CheckpointTransformPipeline(transforms=qeff_model._checkpoint_transforms)
    return str(
        prep_pipeline.apply(
            src=source_dir,
            out=prepared_out,
            target_dtype=target_dtype,
        )
    )
