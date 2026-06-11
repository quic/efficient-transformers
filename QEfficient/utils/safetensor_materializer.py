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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig

from QEfficient.utils import constants
from QEfficient.utils.constants import QEFF_MODELS_DIR
from QEfficient.utils.layer_scale_checkpoint import (
    TensorScaleSpec,
    apply_layer_scale_recipe_to_loaded_model,
    build_tensor_scale_specs,
    inject_layer_scale_metadata_to_loaded_model,
    load_layer_scale_recipe,
)
from QEfficient.utils.logging_utils import logger

QEFF_MATERIALIZED_CHECKPOINT_DIR_ENV = "QEFF_MATERIALIZED_CHECKPOINT_DIR"
QEFF_MATERIALIZE_FORCE_ENV = "QEFF_MATERIALIZE_FORCE"

_INDEX_FILE = "model.safetensors.index.json"
_SINGLE_FILE = "model.safetensors"
_MARKER_FILE = "qeff_materialized_checkpoint.json"
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".msgpack")
_QEFF_STREAMING_CHECKPOINT_PATH_KWARG = "_qeff_streaming_checkpoint_path"
_QEFF_STREAMING_CHECKPOINT_DTYPE_KWARG = "_qeff_streaming_checkpoint_dtype"
_QEFF_LAYER_SCALE_YAML_KWARG = "qeff_layer_scale_yaml"

_TORCH_DTYPE_TO_CONFIG = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}

_SAFETENSOR_PARALLEL_WORKERS = min(4, os.cpu_count() or 4)

_SAFETENSOR_FLOAT_DTYPES = {
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


@dataclass(frozen=True)
class MaterializedCheckpoint:
    """Path metadata for an explicitly materialized safetensor checkpoint copy."""

    path: Path
    source_path: Path
    target_dtype: torch.dtype
    materialized: bool
    dtype_aligned: bool


@dataclass(frozen=True)
class StreamingCheckpoint:
    """Path metadata for directly streaming a safetensor checkpoint."""

    path: Path
    target_dtype: torch.dtype
    dtype_aligned: bool


@dataclass(frozen=True)
class StreamingLoadResult:
    """Summary of tensors loaded into a meta-initialized model."""

    loaded_tensor_count: int
    loaded_tensor_bytes: int
    missing_parameter_names: List[str]
    missing_buffer_names: List[str]
    unexpected_tensor_names: List[str]


def env_flag(name: str) -> bool:
    """Return True when an environment variable uses a truthy value."""

    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_torch_dtype(dtype: Any) -> Optional[torch.dtype]:
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
    raise ValueError(f"Unsupported torch dtype for QEfficient safetensor loading: {dtype!r}")


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
    """Create or reuse a local safetensor checkpoint copy in the requested dtype."""

    target_dtype = _normalize_torch_dtype(target_dtype)
    if target_dtype not in _TORCH_DTYPE_TO_CONFIG:
        raise ValueError(f"Unsupported target dtype for QEfficient checkpoint materialization: {target_dtype}")
    force = force or env_flag(QEFF_MATERIALIZE_FORCE_ENV)

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


def resolve_safetensor_checkpoint_for_streaming(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    target_dtype: Any,
    *,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    local_files_only: bool = False,
    hub_cache_dir: Optional[Union[str, os.PathLike]] = None,
) -> Optional[StreamingCheckpoint]:
    """Resolve a safetensor checkpoint that can be streamed directly into a model."""

    normalized_dtype = _normalize_torch_dtype(target_dtype)
    if normalized_dtype is None:
        return None
    if normalized_dtype not in _TORCH_DTYPE_TO_CONFIG:
        raise ValueError(f"Unsupported target dtype for QEfficient checkpoint streaming: {normalized_dtype}")

    source_path = _resolve_source_path(
        pretrained_model_name_or_path,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
        hub_cache_dir=hub_cache_dir,
    )
    layout = _load_safetensor_layout(source_path)
    return StreamingCheckpoint(
        path=source_path,
        target_dtype=normalized_dtype,
        dtype_aligned=not _checkpoint_requires_conversion(source_path, layout["shards"], normalized_dtype),
    )


def prepare_safetensor_streaming_from_pretrained_source(pretrained_model_name_or_path: str, kwargs: dict) -> str:
    """Prepare ``from_pretrained`` kwargs for QEfficient direct safetensor streaming."""

    requested_torch_dtype = kwargs.get("torch_dtype") if "torch_dtype" in kwargs else kwargs.get("dtype")
    _resolve_qeff_torch_dtype(kwargs)
    if requested_torch_dtype is None:
        return pretrained_model_name_or_path

    try:
        streaming_checkpoint = resolve_safetensor_checkpoint_for_streaming(
            pretrained_model_name_or_path,
            kwargs.get("torch_dtype"),
            revision=kwargs.get("revision"),
            token=kwargs.get("token", kwargs.get("use_auth_token")),
            local_files_only=kwargs.get("local_files_only", False),
            hub_cache_dir=kwargs.get("cache_dir"),
        )
    except (OSError, ValueError) as exc:
        logger.warning("Skipping QEfficient streaming checkpoint load: %s", exc)
        return pretrained_model_name_or_path

    if streaming_checkpoint is None:
        return pretrained_model_name_or_path

    kwargs[_QEFF_STREAMING_CHECKPOINT_PATH_KWARG] = str(streaming_checkpoint.path)
    kwargs[_QEFF_STREAMING_CHECKPOINT_DTYPE_KWARG] = kwargs.get("torch_dtype")
    kwargs["low_cpu_mem_usage"] = True
    kwargs["use_safetensors"] = True
    if streaming_checkpoint.dtype_aligned:
        logger.info("Streaming safetensors checkpoint directly from %s", streaming_checkpoint.path)
    else:
        logger.info(
            "Streaming safetensors checkpoint directly from %s with %s casts",
            streaming_checkpoint.path,
            streaming_checkpoint.target_dtype,
        )
    return str(streaming_checkpoint.path)


def load_hf_model_with_optional_safetensor_streaming(
    auto_class, pretrained_model_name_or_path, args: tuple, kwargs: dict
):
    """Load a Hugging Face model, streaming safetensors into a meta model when prepared."""

    layer_scale_recipe_path = kwargs.pop(_QEFF_LAYER_SCALE_YAML_KWARG, None)
    if layer_scale_recipe_path is not None:
        layer_scale_recipe_path = str(Path(layer_scale_recipe_path).expanduser().resolve())

    streaming_checkpoint_path = kwargs.pop(_QEFF_STREAMING_CHECKPOINT_PATH_KWARG, None)
    streaming_dtype = kwargs.pop(_QEFF_STREAMING_CHECKPOINT_DTYPE_KWARG, None)
    if streaming_checkpoint_path is None:
        model = auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if layer_scale_recipe_path is not None:
            scale_audit = apply_layer_scale_recipe_to_loaded_model(
                model=model,
                recipe_path=layer_scale_recipe_path,
                strict=True,
                inject_config_metadata=True,
            )
            logger.info(
                "Applied layer-scale recipe %s to loaded model tensors: scaled=%d missing=%d",
                layer_scale_recipe_path,
                scale_audit["scaled_tensor_count"],
                len(scale_audit["missing_recipe_keys"]),
            )
        return model

    if args:
        logger.warning("QEfficient streaming checkpoint load does not support positional model args; using HF loader.")
        model = auto_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if layer_scale_recipe_path is not None:
            scale_audit = apply_layer_scale_recipe_to_loaded_model(
                model=model,
                recipe_path=layer_scale_recipe_path,
                strict=True,
                inject_config_metadata=True,
            )
            logger.info(
                "Applied layer-scale recipe %s to loaded model tensors: scaled=%d missing=%d",
                layer_scale_recipe_path,
                scale_audit["scaled_tensor_count"],
                len(scale_audit["missing_recipe_keys"]),
            )
        return model

    config = kwargs.get("config")
    if config is None:
        config_kwargs = {}
        for key in ("cache_dir", "revision", "token", "use_auth_token", "local_files_only", "trust_remote_code"):
            if key in kwargs:
                config_kwargs[key] = kwargs[key]
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)

    constructor_kwargs = {}
    for key in ("attn_implementation", "trust_remote_code", "torch_dtype", "dtype"):
        if key in kwargs:
            constructor_kwargs[key] = kwargs[key]

    normalized_streaming_dtype = _normalize_torch_dtype(streaming_dtype)
    if normalized_streaming_dtype is not None:
        _set_config_torch_dtype(config, normalized_streaming_dtype)

    logger.info("Streaming safetensors checkpoint into model from %s", streaming_checkpoint_path)
    with torch.device("meta"):
        model = auto_class.from_config(config, **constructor_kwargs)
    if normalized_streaming_dtype is not None:
        _set_config_torch_dtype(getattr(model, "config", None), normalized_streaming_dtype)
    load_summary = load_safetensor_checkpoint_into_model(
        model,
        streaming_checkpoint_path,
        target_dtype=streaming_dtype,
        layer_scale_recipe_path=layer_scale_recipe_path,
    )
    if layer_scale_recipe_path is not None:
        inject_layer_scale_metadata_to_loaded_model(model=model, recipe_path=layer_scale_recipe_path)
    logger.info(
        "Loaded %s tensors (%s bytes) through QEfficient streaming safetensor loader",
        load_summary.loaded_tensor_count,
        load_summary.loaded_tensor_bytes,
    )
    model.eval()
    return model


def _resolve_qeff_torch_dtype(kwargs: dict) -> None:
    aic_hw_version = constants.DEFAULT_AIC_HW_VERSION
    current_dtype = kwargs.get("torch_dtype", None)

    if (current_dtype is None or current_dtype == torch.bfloat16) and aic_hw_version != "ai200":
        if current_dtype == torch.bfloat16:
            logger.warning(
                "torch_dtype=bfloat16 is not supported on %s. Overriding to torch.float32.",
                aic_hw_version,
            )
        kwargs["torch_dtype"] = torch.float32


def _set_config_torch_dtype(config, torch_dtype: torch.dtype) -> None:
    if config is None:
        return
    if hasattr(config, "torch_dtype"):
        config.torch_dtype = torch_dtype
    for child_name in ("text_config", "vision_config", "llm_config"):
        if hasattr(config, child_name):
            _set_config_torch_dtype(getattr(config, child_name), torch_dtype)


def load_safetensor_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: Union[str, os.PathLike],
    target_dtype: Any = None,
    layer_scale_recipe_path: Union[str, os.PathLike, None] = None,
) -> StreamingLoadResult:
    """Stream safetensor shards into matching parameters and buffers on CPU."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    layout = _load_safetensor_layout(checkpoint_path)
    normalized_dtype = _normalize_torch_dtype(target_dtype)
    layer_scale_specs = _load_layer_scale_specs(layer_scale_recipe_path)

    expected_parameters = dict(model.named_parameters())
    expected_buffers = dict(model.named_buffers())
    expected_names = set(expected_parameters) | set(expected_buffers)
    missing_scale_keys = sorted(name for name in layer_scale_specs if name not in expected_names)
    if missing_scale_keys:
        raise KeyError(
            "Layer-scale recipe keys missing from model tensors. "
            f"Missing={missing_scale_keys[:8]} total_missing={len(missing_scale_keys)}"
        )
    checkpoint_name_to_shard = _index_safetensor_names(checkpoint_path, layout["shards"])
    legacy_expert_source_names = _collect_legacy_expert_source_names(expected_parameters, checkpoint_name_to_shard)
    loaded_names = set()
    unexpected_names = []
    loaded_tensor_count = 0
    loaded_tensor_bytes = 0

    def load_shard(shard: str) -> tuple[list[tuple[str, torch.Tensor]], list[str]]:
        loaded_tensors = []
        shard_unexpected_names = []
        with safe_open(checkpoint_path / shard, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name not in expected_names:
                    if name not in legacy_expert_source_names:
                        shard_unexpected_names.append(name)
                    continue
                tensor = reader.get_tensor(name)
                if normalized_dtype is not None and tensor.is_floating_point() and tensor.dtype != normalized_dtype:
                    tensor = tensor.to(dtype=normalized_dtype)
                tensor = _maybe_apply_layer_scale(tensor, layer_scale_specs.get(name))
                loaded_tensors.append((name, tensor))
        return loaded_tensors, shard_unexpected_names

    for loaded_tensors, shard_unexpected_names in _iter_shard_results(layout["shards"], load_shard):
        unexpected_names.extend(shard_unexpected_names)
        for name, tensor in loaded_tensors:
            _set_module_tensor(model, name, tensor)
            loaded_names.add(name)
            loaded_tensor_count += 1
            loaded_tensor_bytes += tensor.numel() * tensor.element_size()
        del loaded_tensors
        gc.collect()

    packed_count, packed_bytes, packed_names = _materialize_legacy_expert_parameters(
        model,
        expected_parameters,
        loaded_names,
        checkpoint_path,
        checkpoint_name_to_shard,
        normalized_dtype,
        layer_scale_specs,
    )
    loaded_tensor_count += packed_count
    loaded_tensor_bytes += packed_bytes
    loaded_names.update(packed_names)

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


def _index_safetensor_names(checkpoint_path: Path, shards: List[str]) -> Dict[str, str]:
    name_to_shard = {}
    for shard in shards:
        with safe_open(checkpoint_path / shard, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                name_to_shard[name] = shard
    return name_to_shard


def _collect_legacy_expert_source_names(
    expected_parameters: Dict[str, torch.nn.Parameter],
    checkpoint_name_to_shard: Dict[str, str],
) -> set[str]:
    source_names = set()
    for target_name, parameter in expected_parameters.items():
        if target_name.endswith(".experts.gate_up_proj"):
            source_names.update(
                _legacy_expert_source_names(
                    target_name,
                    tuple(parameter.shape),
                    (("gate_proj", "weight"), ("up_proj", "weight")),
                    checkpoint_name_to_shard,
                )
            )
        elif target_name.endswith(".experts.down_proj"):
            source_names.update(
                _legacy_expert_source_names(
                    target_name,
                    tuple(parameter.shape),
                    (("down_proj", "weight"),),
                    checkpoint_name_to_shard,
                )
            )
    return source_names


def _legacy_expert_source_names(
    target_name: str,
    target_shape: tuple[int, ...],
    projections: tuple[tuple[str, str], ...],
    checkpoint_name_to_shard: Dict[str, str],
) -> List[str]:
    if len(target_shape) != 3:
        return []
    base_name = target_name.rsplit(".", 1)[0]
    source_names = [
        f"{base_name}.{expert_idx}.{projection}.{weight_name}"
        for expert_idx in range(target_shape[0])
        for projection, weight_name in projections
    ]
    if all(name in checkpoint_name_to_shard for name in source_names):
        return source_names
    return []


def _materialize_legacy_expert_parameters(
    model: torch.nn.Module,
    expected_parameters: Dict[str, torch.nn.Parameter],
    loaded_names: set[str],
    checkpoint_path: Path,
    checkpoint_name_to_shard: Dict[str, str],
    target_dtype: Optional[torch.dtype],
    layer_scale_specs: Dict[str, TensorScaleSpec],
) -> tuple[int, int, set[str]]:
    loaded_tensor_count = 0
    loaded_tensor_bytes = 0
    packed_names = set()

    for target_name, parameter in expected_parameters.items():
        if target_name in loaded_names or parameter.device.type != "meta":
            continue
        if target_name.endswith(".experts.gate_up_proj"):
            tensor = _load_legacy_gate_up_experts(
                target_name, parameter, checkpoint_path, checkpoint_name_to_shard, target_dtype
            )
        elif target_name.endswith(".experts.down_proj"):
            tensor = _load_legacy_down_experts(
                target_name, parameter, checkpoint_path, checkpoint_name_to_shard, target_dtype
            )
        else:
            continue
        if tensor is None:
            continue
        tensor = _maybe_apply_layer_scale(tensor, layer_scale_specs.get(target_name))
        _set_module_tensor(model, target_name, tensor)
        packed_names.add(target_name)
        loaded_tensor_count += 1
        loaded_tensor_bytes += tensor.numel() * tensor.element_size()
        del tensor
        gc.collect()

    return loaded_tensor_count, loaded_tensor_bytes, packed_names


def _load_layer_scale_specs(
    recipe_path: Union[str, os.PathLike, None],
) -> Dict[str, TensorScaleSpec]:
    if recipe_path is None:
        return {}
    recipe = load_layer_scale_recipe(Path(recipe_path).expanduser().resolve())
    return {spec.tensor_key: spec for spec in build_tensor_scale_specs(recipe)}


def _maybe_apply_layer_scale(tensor: torch.Tensor, spec: Optional[TensorScaleSpec]) -> torch.Tensor:
    if spec is None:
        return tensor
    out = tensor.clone()
    scalar = out.new_tensor(float(spec.scale))
    if spec.operation == "scale_all":
        out.mul_(scalar)
        return out
    if spec.operation == "scale_second_half_dim1":
        if out.ndim < 2:
            raise ValueError(f"scale_second_half_dim1 requires rank>=2. Got rank={out.ndim}")
        dim1 = int(out.shape[1])
        if dim1 % 2 != 0:
            raise ValueError(f"scale_second_half_dim1 requires even dim1. Got dim1={dim1}")
        half = dim1 // 2
        index = (slice(None), slice(half, None), *([slice(None)] * (out.ndim - 2)))
        out[index].mul_(scalar)
        return out
    raise ValueError(f"Unsupported layer-scale operation={spec.operation!r}")


def _load_legacy_gate_up_experts(
    target_name: str,
    parameter: torch.nn.Parameter,
    checkpoint_path: Path,
    checkpoint_name_to_shard: Dict[str, str],
    target_dtype: Optional[torch.dtype],
) -> Optional[torch.Tensor]:
    target_shape = tuple(parameter.shape)
    source_names = _legacy_expert_source_names(
        target_name,
        target_shape,
        (("gate_proj", "weight"), ("up_proj", "weight")),
        checkpoint_name_to_shard,
    )
    if not source_names:
        return None

    output = torch.empty(target_shape, dtype=_target_parameter_dtype(parameter, target_dtype), device="cpu")
    base_name = target_name.rsplit(".", 1)[0]
    for expert_idx in range(target_shape[0]):
        gate = _read_safetensor_tensor(
            checkpoint_path, checkpoint_name_to_shard, f"{base_name}.{expert_idx}.gate_proj.weight", target_dtype
        )
        up = _read_safetensor_tensor(
            checkpoint_path, checkpoint_name_to_shard, f"{base_name}.{expert_idx}.up_proj.weight", target_dtype
        )
        packed = torch.cat((gate, up), dim=0)
        output[expert_idx].copy_(_align_legacy_expert_slice(packed, output[expert_idx].shape, target_name))
    return output


def _load_legacy_down_experts(
    target_name: str,
    parameter: torch.nn.Parameter,
    checkpoint_path: Path,
    checkpoint_name_to_shard: Dict[str, str],
    target_dtype: Optional[torch.dtype],
) -> Optional[torch.Tensor]:
    target_shape = tuple(parameter.shape)
    source_names = _legacy_expert_source_names(
        target_name,
        target_shape,
        (("down_proj", "weight"),),
        checkpoint_name_to_shard,
    )
    if not source_names:
        return None

    output = torch.empty(target_shape, dtype=_target_parameter_dtype(parameter, target_dtype), device="cpu")
    base_name = target_name.rsplit(".", 1)[0]
    for expert_idx in range(target_shape[0]):
        down = _read_safetensor_tensor(
            checkpoint_path, checkpoint_name_to_shard, f"{base_name}.{expert_idx}.down_proj.weight", target_dtype
        )
        output[expert_idx].copy_(_align_legacy_expert_slice(down, output[expert_idx].shape, target_name))
    return output


def _read_safetensor_tensor(
    checkpoint_path: Path,
    checkpoint_name_to_shard: Dict[str, str],
    tensor_name: str,
    target_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    with safe_open(checkpoint_path / checkpoint_name_to_shard[tensor_name], framework="pt", device="cpu") as reader:
        tensor = reader.get_tensor(tensor_name)
    if target_dtype is not None and tensor.is_floating_point() and tensor.dtype != target_dtype:
        tensor = tensor.to(dtype=target_dtype)
    return tensor


def _target_parameter_dtype(parameter: torch.nn.Parameter, target_dtype: Optional[torch.dtype]) -> torch.dtype:
    if target_dtype is not None and parameter.is_floating_point():
        return target_dtype
    return parameter.dtype


def _align_legacy_expert_slice(tensor: torch.Tensor, target_shape: torch.Size, target_name: str) -> torch.Tensor:
    expected_shape = tuple(target_shape)
    if tuple(tensor.shape) == expected_shape:
        return tensor
    if tensor.ndim == 2 and tuple(tensor.transpose(0, 1).shape) == expected_shape:
        return tensor.transpose(0, 1).contiguous()
    raise RuntimeError(
        f"Cannot map legacy expert tensor into {target_name}: checkpoint slice shape {tuple(tensor.shape)} "
        f"does not match target slice shape {expected_shape}"
    )


def _iter_shard_results(shards: List[str], worker):
    """Run shard I/O jobs through a small thread pool and yield completed results."""

    worker_count = min(len(shards), _SAFETENSOR_PARALLEL_WORKERS)
    if worker_count <= 1:
        for shard in shards:
            yield worker(shard)
        return

    with ThreadPoolExecutor(max_workers=worker_count) as thread_pool:
        futures = [thread_pool.submit(worker, shard) for shard in shards]
        for future in as_completed(futures):
            yield future.result()


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
        "QEfficient safetensor loading requires a checkpoint with "
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
    def convert_shard(shard: str) -> int:
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
        converted_size = target_file.stat().st_size
        del tensors
        gc.collect()
        return converted_size

    total_size = sum(_iter_shard_results(layout["shards"], convert_shard))

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
