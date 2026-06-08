# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import gc
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from QEfficient.utils.constants import QEFF_MODELS_DIR
from QEfficient.utils.logging_utils import logger

QEFF_MATERIALIZED_CHECKPOINT_DIR_ENV = "QEFF_MATERIALIZED_CHECKPOINT_DIR"
QEFF_MATERIALIZE_FORCE_ENV = "QEFF_MATERIALIZE_FORCE"

_INDEX_FILE = "model.safetensors.index.json"
_SINGLE_FILE = "model.safetensors"
_MARKER_FILE = "qeff_materialized_checkpoint.json"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack")

_TORCH_DTYPE_TO_CONFIG = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}

_SAFETENSOR_FLOAT_DTYPES = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


@dataclass(frozen=True)
class MaterializedCheckpoint:
    path: Path
    source_path: Path
    target_dtype: torch.dtype
    materialized: bool
    dtype_aligned: bool


@dataclass(frozen=True)
class StreamingLoadResult:
    loaded_tensor_count: int
    loaded_tensor_bytes: int
    missing_parameter_names: List[str]
    missing_buffer_names: List[str]
    unexpected_tensor_names: List[str]


def env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_torch_dtype(dtype: Any) -> Optional[torch.dtype]:
    if dtype is None or dtype == "auto":
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        normalized = dtype.lower()
        if normalized.startswith("torch."):
            normalized = normalized[len("torch.") :]
        for torch_dtype, config_value in _TORCH_DTYPE_TO_CONFIG.items():
            if normalized == config_value:
                return torch_dtype
    raise ValueError(f"Unsupported torch dtype for checkpoint materialization: {dtype!r}")


def materialize_safetensor_checkpoint(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    target_dtype: torch.dtype,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    local_files_only: bool = False,
    force: bool = False,
    hub_cache_dir: Optional[Union[str, os.PathLike]] = None,
) -> MaterializedCheckpoint:
    source_path = _resolve_source_path(
        pretrained_model_name_or_path,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
        hub_cache_dir=hub_cache_dir,
    )
    layout = _load_safetensor_layout(source_path)
    if not _checkpoint_requires_conversion(source_path, layout["shards"], target_dtype):
        return MaterializedCheckpoint(
            path=source_path,
            source_path=source_path,
            target_dtype=target_dtype,
            materialized=False,
            dtype_aligned=True,
        )

    output_root_value = cache_dir or os.environ.get(QEFF_MATERIALIZED_CHECKPOINT_DIR_ENV)
    if output_root_value:
        output_root = Path(output_root_value).expanduser()
    else:
        output_root = Path(QEFF_MODELS_DIR) / "materialized_checkpoints"
    output_root.mkdir(parents=True, exist_ok=True)

    cache_key = _cache_key(source_path, layout["shards"], target_dtype, revision)
    target_path = output_root / f"{source_path.name}-{_TORCH_DTYPE_TO_CONFIG[target_dtype]}-{cache_key[:12]}"
    marker_path = target_path / _MARKER_FILE
    if marker_path.exists() and not force:
        return MaterializedCheckpoint(
            path=target_path,
            source_path=source_path,
            target_dtype=target_dtype,
            materialized=True,
            dtype_aligned=True,
        )
    if target_path.exists():
        shutil.rmtree(target_path)

    logger.info("Materializing %s checkpoint at %s into %s", target_dtype, source_path, target_path)
    with TemporaryDirectory(dir=output_root, prefix=f".{target_path.name}.") as temp_dir:
        temp_path = Path(temp_dir)
        _copy_non_weight_files(source_path, temp_path)
        converted_size = _convert_checkpoint_files(source_path, temp_path, layout, target_dtype)
        _write_materialized_config(temp_path, target_dtype)
        _write_marker(temp_path, source_path, target_dtype, layout["shards"], converted_size)
        shutil.move(str(temp_path), target_path)

    return MaterializedCheckpoint(
        path=target_path,
        source_path=source_path,
        target_dtype=target_dtype,
        materialized=True,
        dtype_aligned=True,
    )


def maybe_materialize_safetensor_checkpoint(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    target_dtype: Any,
    *,
    enabled: Optional[bool] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    local_files_only: bool = False,
    force: Optional[bool] = None,
    hub_cache_dir: Optional[Union[str, os.PathLike]] = None,
) -> Optional[MaterializedCheckpoint]:
    normalized_dtype = normalize_torch_dtype(target_dtype)
    should_materialize = normalized_dtype is not None if enabled is None else enabled
    if not should_materialize or normalized_dtype is None:
        return None
    if normalized_dtype not in _TORCH_DTYPE_TO_CONFIG:
        raise ValueError(f"Unsupported target dtype for QEfficient checkpoint materialization: {normalized_dtype}")

    return materialize_safetensor_checkpoint(
        pretrained_model_name_or_path,
        normalized_dtype,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
        force=env_flag(QEFF_MATERIALIZE_FORCE_ENV) if force is None else force,
        hub_cache_dir=hub_cache_dir,
    )


def load_safetensor_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: Union[str, os.PathLike],
    target_dtype: Any = None,
) -> StreamingLoadResult:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    layout = _load_safetensor_layout(checkpoint_path)
    normalized_dtype = normalize_torch_dtype(target_dtype)

    expected_parameters = dict(model.named_parameters())
    expected_buffers = dict(model.named_buffers())
    expected_names = set(expected_parameters) | set(expected_buffers)
    loaded_names = set()
    unexpected_names = []
    loaded_tensor_count = 0
    loaded_tensor_bytes = 0

    for shard in layout["shards"]:
        with safe_open(checkpoint_path / shard, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name not in expected_names:
                    unexpected_names.append(name)
                    continue
                tensor = reader.get_tensor(name)
                if normalized_dtype is not None and tensor.is_floating_point() and tensor.dtype != normalized_dtype:
                    tensor = tensor.to(dtype=normalized_dtype)
                _set_module_tensor(model, name, tensor)
                loaded_names.add(name)
                loaded_tensor_count += 1
                loaded_tensor_bytes += tensor.numel() * tensor.element_size()
        gc.collect()

    if hasattr(model, "tie_weights"):
        model.tie_weights()

    missing_parameter_names = _remaining_meta_names(model.named_parameters(), loaded_names)
    missing_buffer_names = _remaining_meta_names(model.named_buffers(), loaded_names)
    _materialize_missing_meta_buffers(model, missing_buffer_names, normalized_dtype)
    missing_buffer_names = _remaining_meta_names(model.named_buffers(), loaded_names)
    if missing_parameter_names:
        raise RuntimeError(
            "QEfficient streaming checkpoint load left parameters on meta device: "
            + ", ".join(missing_parameter_names[:10])
        )

    return StreamingLoadResult(
        loaded_tensor_count=loaded_tensor_count,
        loaded_tensor_bytes=loaded_tensor_bytes,
        missing_parameter_names=missing_parameter_names,
        missing_buffer_names=missing_buffer_names,
        unexpected_tensor_names=unexpected_names,
    )


def _remaining_meta_names(named_tensors, loaded_names: set[str]) -> List[str]:
    return [name for name, tensor in named_tensors if name not in loaded_names and tensor.device.type == "meta"]


def _materialize_missing_meta_buffers(
    model: torch.nn.Module,
    missing_buffer_names: List[str],
    target_dtype: Optional[torch.dtype],
) -> None:
    for name in missing_buffer_names:
        module, tensor_name = _get_parent_module(model, name)
        old_buffer = module._buffers[tensor_name]
        dtype = target_dtype if target_dtype is not None and old_buffer.is_floating_point() else old_buffer.dtype
        module._buffers[tensor_name] = torch.zeros(old_buffer.shape, dtype=dtype, device="cpu")


def _set_module_tensor(model: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    module, tensor_name = _get_parent_module(model, name)
    if tensor_name in module._parameters:
        old_parameter = module._parameters[tensor_name]
        requires_grad = old_parameter.requires_grad if old_parameter is not None else False
        module._parameters[tensor_name] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
        return
    if tensor_name in module._buffers:
        module._buffers[tensor_name] = tensor
        return
    setattr(module, tensor_name, tensor)


def _get_parent_module(model: torch.nn.Module, tensor_name: str) -> tuple[torch.nn.Module, str]:
    if "." not in tensor_name:
        return model, tensor_name
    module_name, child_name = tensor_name.rsplit(".", 1)
    return model.get_submodule(module_name), child_name


def _resolve_source_path(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    *,
    revision: Optional[str],
    token: Optional[Union[str, bool]],
    local_files_only: bool,
    hub_cache_dir: Optional[Union[str, os.PathLike]],
) -> Path:
    candidate = Path(pretrained_model_name_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return Path(
        snapshot_download(
            repo_id=str(pretrained_model_name_or_path),
            revision=revision,
            token=token,
            local_files_only=local_files_only,
            cache_dir=hub_cache_dir,
        )
    ).resolve()


def _load_safetensor_layout(source_path: Path) -> Dict[str, Any]:
    index_path = source_path / _INDEX_FILE
    if index_path.exists():
        with index_path.open() as handle:
            index = json.load(handle)
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise ValueError(f"Safetensors index has no weight_map entries: {index_path}")
        return {"index": index, "index_name": _INDEX_FILE, "shards": shards}

    model_file = source_path / _SINGLE_FILE
    if model_file.exists():
        return {"index": None, "index_name": None, "shards": [_SINGLE_FILE]}

    candidates = sorted(path.name for path in source_path.glob("*.safetensors"))
    if len(candidates) == 1:
        return {"index": None, "index_name": None, "shards": candidates}

    raise ValueError(
        "QEfficient checkpoint materialization requires a safetensors checkpoint with "
        f"{_SINGLE_FILE} or {_INDEX_FILE}; found {candidates or 'none'} in {source_path}"
    )


def _checkpoint_requires_conversion(source_path: Path, shards: List[str], target_dtype: torch.dtype) -> bool:
    for shard in shards:
        with safe_open(source_path / shard, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                safe_dtype = reader.get_slice(name).get_dtype()
                if _SAFETENSOR_FLOAT_DTYPES.get(safe_dtype) not in {None, target_dtype}:
                    return True
    return False


def _cache_key(source_path: Path, shards: List[str], target_dtype: torch.dtype, revision: Optional[str]) -> str:
    file_stats = []
    for shard in shards:
        path = source_path / shard
        stat = path.stat()
        file_stats.append({"name": shard, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    payload = {
        "source": str(source_path),
        "revision": revision,
        "target_dtype": _TORCH_DTYPE_TO_CONFIG[target_dtype],
        "files": file_stats,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _copy_non_weight_files(source_path: Path, target_path: Path) -> None:
    for item in source_path.rglob("*"):
        relative_path = item.relative_to(source_path)
        if item.is_dir():
            (target_path / relative_path).mkdir(parents=True, exist_ok=True)
            continue
        if item.suffix in _WEIGHT_SUFFIXES or item.name == _INDEX_FILE:
            continue
        destination = target_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)


def _convert_checkpoint_files(
    source_path: Path, target_path: Path, layout: Dict[str, Any], target_dtype: torch.dtype
) -> int:
    total_size = 0
    for shard in layout["shards"]:
        source_file = source_path / shard
        target_file = target_path / shard
        target_file.parent.mkdir(parents=True, exist_ok=True)
        tensors = {}
        with safe_open(source_file, framework="pt", device="cpu") as reader:
            metadata = reader.metadata() or {"format": "pt"}
            for name in reader.keys():
                tensor = reader.get_tensor(name)
                if tensor.is_floating_point() and tensor.dtype != target_dtype:
                    tensor = tensor.to(dtype=target_dtype)
                tensors[name] = tensor
        save_file(tensors, target_file, metadata=metadata)
        total_size += target_file.stat().st_size
        del tensors
        gc.collect()

    if layout["index"] is not None:
        index = dict(layout["index"])
        metadata = dict(index.get("metadata", {}))
        metadata["total_size"] = total_size
        index["metadata"] = metadata
        with (target_path / layout["index_name"]).open("w") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)

    return total_size


def _write_materialized_config(target_path: Path, target_dtype: torch.dtype) -> None:
    config_path = target_path / "config.json"
    if not config_path.exists():
        return
    with config_path.open() as handle:
        config = json.load(handle)
    _replace_torch_dtype(config, _TORCH_DTYPE_TO_CONFIG[target_dtype])
    with config_path.open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _replace_torch_dtype(value: Any, dtype_name: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "torch_dtype":
                value[key] = dtype_name
            else:
                _replace_torch_dtype(item, dtype_name)
    elif isinstance(value, list):
        for item in value:
            _replace_torch_dtype(item, dtype_name)


def _write_marker(
    target_path: Path,
    source_path: Path,
    target_dtype: torch.dtype,
    shards: List[str],
    converted_size: int,
) -> None:
    marker = {
        "source_path": str(source_path),
        "target_dtype": _TORCH_DTYPE_TO_CONFIG[target_dtype],
        "shards": shards,
        "converted_size": converted_size,
    }
    with (target_path / _MARKER_FILE).open("w") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
        handle.write("\n")
