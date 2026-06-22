# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Utilities for selectively loading Hugging Face checkpoint weights."""

from __future__ import annotations

import gc
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Pattern, Sequence

import torch
import transformers


_LAYER_INDEX_PATTERNS = (
    re.compile(r"(?:^|\.)model\.language_model\.layers\.(\d+)\."),
    re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\."),
    re.compile(r"(?:^|\.)model\.layers\.(\d+)\."),
    re.compile(r"^layers\.(\d+)\."),
)

_LAYER_MODULE_PATTERNS = (
    re.compile(r"(?:^|\.)model\.language_model\.layers\.(\d+)$"),
    re.compile(r"(?:^|\.)language_model\.layers\.(\d+)$"),
    re.compile(r"(?:^|\.)model\.layers\.(\d+)$"),
    re.compile(r"(?:^|\.)layers\.(\d+)$"),
)


@dataclass(frozen=True)
class WeightSelectionPolicy:
    """Predicate object deciding which checkpoint keys are visible to HF loading."""

    layer_indices: Optional[frozenset[int]] = None
    layer_key_patterns: Sequence[Pattern[str]] = _LAYER_INDEX_PATTERNS
    keep_key_patterns: Sequence[Pattern[str]] = field(default_factory=tuple)
    keep_non_layer_keys: bool = True

    @classmethod
    def from_layer_indices(
        cls,
        layer_indices: Optional[Iterable[int]],
        *,
        layer_key_patterns: Optional[Sequence[str | Pattern[str]]] = None,
        keep_key_patterns: Optional[Sequence[str | Pattern[str]]] = None,
        keep_non_layer_keys: bool = True,
    ) -> "WeightSelectionPolicy":
        return cls(
            layer_indices=None if layer_indices is None else frozenset(int(idx) for idx in layer_indices),
            layer_key_patterns=_compile_patterns(layer_key_patterns) or _LAYER_INDEX_PATTERNS,
            keep_key_patterns=_compile_patterns(keep_key_patterns),
            keep_non_layer_keys=keep_non_layer_keys,
        )

    def include_key(self, key: str) -> bool:
        if self.layer_indices is None:
            return True
        if any(pattern.search(key) for pattern in self.keep_key_patterns):
            return True
        layer_idx = self.layer_index_for_key(key)
        if layer_idx is None:
            return self.keep_non_layer_keys
        return layer_idx in self.layer_indices

    def layer_index_for_key(self, key: str) -> Optional[int]:
        for pattern in self.layer_key_patterns:
            match = pattern.search(key)
            if match:
                return int(match.group(1))
        return None

    def filter_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.layer_indices is None:
            return state_dict
        return {key: value for key, value in state_dict.items() if self.include_key(key)}

    def filter_weight_map(self, weight_map: Dict[str, str]) -> Dict[str, str]:
        if self.layer_indices is None:
            return dict(weight_map)
        return {key: value for key, value in weight_map.items() if self.include_key(key)}


def _compile_patterns(patterns: Optional[Sequence[str | Pattern[str]]]) -> tuple[Pattern[str], ...]:
    if not patterns:
        return ()
    return tuple(re.compile(pattern) if isinstance(pattern, str) else pattern for pattern in patterns)


class _FilteredSafeOpen:
    """Proxy around safetensors.safe_open that hides unselected keys."""

    def __init__(self, file_pointer, policy: WeightSelectionPolicy):
        self._file_pointer = file_pointer
        self._policy = policy

    def keys(self):
        return [key for key in self._file_pointer.keys() if self._policy.include_key(key)]

    def get_slice(self, key):
        return self._file_pointer.get_slice(key)

    def get_tensor(self, key):
        return self._file_pointer.get_tensor(key)

    def metadata(self):
        return self._file_pointer.metadata()

    def __enter__(self):
        entered = self._file_pointer.__enter__()
        if entered is not self._file_pointer:
            self._file_pointer = entered
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._file_pointer.__exit__(exc_type, exc_value, traceback)

    def __getattr__(self, name):
        return getattr(self._file_pointer, name)


class CustomLoader:
    """Selective loader that keeps Hugging Face's normal weight-loading machinery."""

    def __init__(
        self,
        model_id: str | Path,
        layer_indices: Optional[Iterable[int]] = None,
        *,
        layer_key_patterns: Optional[Sequence[str | Pattern[str]]] = None,
        keep_key_patterns: Optional[Sequence[str | Pattern[str]]] = None,
        load_kwargs: Optional[Dict[str, Any]] = None,
        keep_non_layer_keys: bool = True,
    ) -> None:
        self.model_id = str(model_id)
        self.load_kwargs = dict(load_kwargs or {})
        self.policy = WeightSelectionPolicy.from_layer_indices(
            layer_indices,
            layer_key_patterns=layer_key_patterns,
            keep_key_patterns=keep_key_patterns,
            keep_non_layer_keys=keep_non_layer_keys,
        )

    @contextmanager
    def scoped_loading(self):
        """Temporarily filter checkpoint visibility inside Transformers loading."""

        modeling_utils = transformers.modeling_utils
        original_get_checkpoint_shard_files = modeling_utils.get_checkpoint_shard_files
        original_load_state_dict = modeling_utils.load_state_dict
        original_safe_open = modeling_utils.safe_open

        def patched_get_checkpoint_shard_files(*args, **kwargs):
            shard_files, metadata = original_get_checkpoint_shard_files(*args, **kwargs)
            weight_map = metadata.get("weight_map") if metadata else None
            if not weight_map:
                return shard_files, metadata

            filtered_weight_map = self.policy.filter_weight_map(weight_map)
            if not filtered_weight_map:
                return shard_files, metadata

            shard_name_to_path = {path.split("/")[-1]: path for path in shard_files}
            filtered_shard_names = sorted(set(filtered_weight_map.values()))
            filtered_shard_files = [
                shard_name_to_path[name] for name in filtered_shard_names if name in shard_name_to_path
            ]
            if not filtered_shard_files:
                return shard_files, metadata

            filtered_metadata = dict(metadata)
            filtered_metadata["weight_map"] = filtered_weight_map
            filtered_metadata["all_checkpoint_keys"] = list(filtered_weight_map.keys())
            return filtered_shard_files, filtered_metadata

        def patched_safe_open(*args, **kwargs):
            return _FilteredSafeOpen(original_safe_open(*args, **kwargs), self.policy)

        def patched_load_state_dict(*args, **kwargs):
            state_dict = original_load_state_dict(*args, **kwargs)
            return self.policy.filter_state_dict(state_dict)

        modeling_utils.get_checkpoint_shard_files = patched_get_checkpoint_shard_files
        modeling_utils.safe_open = patched_safe_open
        modeling_utils.load_state_dict = patched_load_state_dict
        try:
            yield self
        finally:
            modeling_utils.get_checkpoint_shard_files = original_get_checkpoint_shard_files
            modeling_utils.safe_open = original_safe_open
            modeling_utils.load_state_dict = original_load_state_dict

    def load_model(self, hf_auto_class, **from_pretrained_kwargs):
        kwargs = dict(self.load_kwargs)
        kwargs.update(from_pretrained_kwargs)
        with self.scoped_loading():
            model = hf_auto_class.from_pretrained(self.model_id, **kwargs)
        self.to_meta_unselected_layers(model)
        return model

    def load_with(self, load_fn: Callable[..., Any], *args, **kwargs):
        with self.scoped_loading():
            model = load_fn(*args, **kwargs)
        self.to_meta_unselected_layers(model)
        return model

    def load_into(self, model, load_fn: Callable[..., Any], *args, **kwargs):
        del model
        return self.load_with(load_fn, *args, **kwargs)

    @staticmethod
    def build_meta_model(hf_auto_class, model_id: str | Path, *, config=None, **from_config_kwargs):
        if config is None:
            config_kwargs = {
                key: from_config_kwargs.pop(key)
                for key in ("trust_remote_code", "revision", "token", "subfolder", "cache_dir")
                if key in from_config_kwargs
            }
            config = transformers.AutoConfig.from_pretrained(model_id, **config_kwargs)
        torch_dtype = from_config_kwargs.pop("torch_dtype", from_config_kwargs.pop("dtype", torch.float32))
        with torch.device("meta"):
            if hasattr(hf_auto_class, "from_config"):
                return hf_auto_class.from_config(config, torch_dtype=torch_dtype, **from_config_kwargs)
            if hasattr(hf_auto_class, "_from_config"):
                return hf_auto_class._from_config(config, torch_dtype=torch_dtype, **from_config_kwargs)
            return hf_auto_class(config, **from_config_kwargs)

    @staticmethod
    def to_meta(model, *, recurse: bool = True, clear_cuda_cache: bool = False):
        model.to_empty(device="meta", recurse=recurse)
        gc.collect()
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    def to_meta_unselected_layers(self, model, *, clear_cuda_cache: bool = False):
        if self.policy.layer_indices is None or not hasattr(model, "named_modules"):
            return model
        moved_modules = set()
        for module_name, module in model.named_modules():
            layer_idx = _layer_index_for_module_name(module_name)
            if layer_idx is None or layer_idx in self.policy.layer_indices or id(module) in moved_modules:
                continue
            module.to_empty(device="meta", recurse=True)
            moved_modules.add(id(module))
        gc.collect()
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model


def _layer_index_for_module_name(module_name: str) -> Optional[int]:
    for pattern in _LAYER_MODULE_PATTERNS:
        match = pattern.search(module_name)
        if match:
            return int(match.group(1))
    return None
