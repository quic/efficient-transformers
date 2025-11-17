# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from abc import ABC
from collections.abc import Iterable
from typing import Any, Dict, Optional, Tuple

import torch

from QEfficient.customop import (
    CtxGatherFunc,
    CtxGatherFuncCB,
    CtxScatterFunc,
    CtxScatterFuncCB,
)
from QEfficient.utils.constants import INVALID_IDX


class QEffDynamicLayer(ABC):
    def __init__(self):
        self.keys, self.values = None, None

    def get_seq_length(self, cache_position=None) -> int:
        """Returns the sequence length of the cached states."""
        if self.keys is None or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    @classmethod
    def from_tensors(cls, keys: torch.Tensor, values: torch.Tensor) -> "QEffDynamicLayer":
        """
        Build a `QEffDynamicLayer` instance from pre-existing key/value tensors.

        Args:
            keys (`torch.Tensor`):
                Key cache tensor of shape ``[batch_size, num_heads, seq_len, head_dim]``.
            values (`torch.Tensor`):
                Value cache tensor of shape ``[batch_size, num_heads, seq_len, head_dim]``.

        Returns:
            `QEffDynamicLayer`: The newly constructed layer whose internal cache directly references
            the supplied tensors.
        """
        layer = cls()
        layer.keys = keys
        layer.values = values
        return layer

    def read_only(self, cache_kwargs):
        """
        Reads the `key_states` and `value_states` for the layer.

        Parameters:
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Gather
        k_out, v_out = self.keys, self.values
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)
        ctx_len = k_out.shape[2]
        ctx_indices = torch.arange(ctx_len)[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit

        if torch.onnx.is_in_onnx_export():
            invalid_idx_value = INVALID_IDX
        else:
            invalid_idx_value = 0

        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices)
            v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices)
        else:
            k_out = CtxGatherFunc.apply(k_out, ctx_indices)
            v_out = CtxGatherFunc.apply(v_out, ctx_indices)

        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
        return k_out, v_out

    def write_only(self, key_states, value_states, cache_kwargs):
        """
        Write in the cache with the new `key_states` and `value_states` for the layer.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
        """
        # Update the cache
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
        else:
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)

            # Scatter
            if batch_index is not None:
                if torch.onnx.is_in_onnx_export():
                    scatter_position_ids = torch.where(position_ids < 0, INVALID_IDX, position_ids)
                else:
                    scatter_position_ids = position_ids

                self.keys = CtxScatterFuncCB.apply(self.keys, batch_index, scatter_position_ids, key_states)
                self.values = CtxScatterFuncCB.apply(self.values, batch_index, scatter_position_ids, value_states)
            else:
                self.keys = CtxScatterFunc.apply(self.keys, position_ids, key_states)
                self.values = CtxScatterFunc.apply(self.values, position_ids, value_states)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            k_out, v_out = self.keys, self.values
        else:
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)

            # Scatter
            if batch_index is not None:
                if torch.onnx.is_in_onnx_export():
                    scatter_position_ids = torch.where(position_ids < 0, INVALID_IDX, position_ids)
                else:
                    scatter_position_ids = position_ids
                self.keys = CtxScatterFuncCB.apply(self.keys, batch_index, scatter_position_ids, key_states)
                self.values = CtxScatterFuncCB.apply(self.values, batch_index, scatter_position_ids, value_states)
            else:
                self.keys = CtxScatterFunc.apply(self.keys, position_ids, key_states)
                self.values = CtxScatterFunc.apply(self.values, position_ids, value_states)

            k_out, v_out = self.keys, self.values

            # Gather
            ctx_len = k_out.shape[2]
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit

            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = INVALID_IDX
            else:
                invalid_idx_value = 0

            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
            if batch_index is not None:
                k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices)
                v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices)
            else:
                k_out = CtxGatherFunc.apply(k_out, ctx_indices)
                v_out = CtxGatherFunc.apply(v_out, ctx_indices)
            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
        return k_out, v_out


class QEffDynamicCache:
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    - Optimized implementation for the Cloud AI 100 to reuse KV Cache.
    - get the position_ids input using kwargs.
    - Use custom Onnxscript ops to write optimized version to generate Onnx model.

    """

    def __init__(self, ddp_cache_data: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None, *args, **kwargs):
        self.layers = []
        self.layer_classes = QEffDynamicLayer
        if ddp_cache_data is not None:
            for key_states, value_states in ddp_cache_data:
                self.layers.append(QEffDynamicLayer.from_tensors(key_states, value_states))

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        # Best effort BC support for old-style caches like Mambas, Falcon, HybridChunked that rely on __len__
        if getattr(self, "layers", None) is None:
            if getattr(self, "key_cache", None) is not None:
                return len(self.key_cache)
            return 0
        # Empty dynamic caches initialize an empty layer to be ready for first update
        dynamic_empty = (
            getattr(self, "layers", None) is not None
            and len(self.layers) == 1
            and isinstance(self.layers[0], QEffDynamicLayer)
            and self.layers[0].keys is None
        )
        return len(self.layers) if not dynamic_empty else 0

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Converts the `Cache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility.
        """
        legacy_cache = ()
        for layer in self.layers:
            legacy_cache += ((layer.keys, layer.values),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: tuple[tuple[torch.FloatTensor, torch.FloatTensor], ...]
    ) -> "QEffDynamicCache":
        """
        Converts a cache in the legacy cache format into an equivalent `Cache`. Used for
        backward compatibility.
        """
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def append_new_layers(self, layer_idx: int) -> None:
        """
        Appends layers to the cache until the layer `layer_idx` is reached.
        Used for preallocation in static caches and on the fly in dynamic caches.

        Args:
            layer_idx (`int`):
                The index of the layer to append.
        """
        while len(self.layers) <= layer_idx:
            new_layer_class = (
                self.layer_classes[len(self.layers)] if isinstance(self.layer_classes, list) else self.layer_classes
            )
            new_layer = new_layer_class()
            self.layers.append(new_layer)

    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        """Returns the sequence length of the cache for the given layer. TODO: deprecate in favor of cache_position"""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length(cache_position)

    def read_only(self, layer_idx, cache_kwargs):
        """
        Reads the `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        return self.layers[layer_idx].read_only(cache_kwargs)

    def write_only(self, key_states, value_states, layer_idx, cache_kwargs):
        """
        Write in the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
        """
        self.append_new_layers(layer_idx)
        return self.layers[layer_idx].write_only(key_states, value_states, cache_kwargs)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        self.append_new_layers(layer_idx)
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)


class QEffEncoderDecoderCache:
    """
    Updated the `EncoderDecoderCache` to use the `QEffDynamicCache` for both self-attention and cross-attention caches.
    """

    def __init__(self, self_attention_cache, cross_attention_cache):
        # self.layers = []
        # self.layer_classes = QEffDynamicLayer
        self.self_attention_cache = self_attention_cache
        self.cross_attention_cache = cross_attention_cache

        self.is_updated = {}
        for layer_idx in range(len(cross_attention_cache)):
            self.is_updated[layer_idx] = bool(cross_attention_cache.get_seq_length(layer_idx) > 0)

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor]]:
        """Converts the `EncoderDecoderCache` instance into its equivalent in the legacy cache format."""
        legacy_cache = ()
        if len(self.cross_attention_cache) > 0:
            for self_attn, cross_attn in zip(
                self.self_attention_cache.to_legacy_cache(), self.cross_attention_cache.to_legacy_cache()
            ):
                legacy_cache += (self_attn + cross_attn,)
        else:
            legacy_cache = self.self_attention_cache.to_legacy_cache()
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "QEffEncoderDecoderCache":
        """Converts a cache in the legacy cache format into an equivalent `EncoderDecoderCache`."""
        cache = cls(
            self_attention_cache=QEffDynamicCache(),
            cross_attention_cache=QEffDynamicCache(),
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx][:2]
                cache.self_attention_cache.update(key_states, value_states, layer_idx)
                if len(past_key_values[layer_idx]) > 2:
                    key_states, value_states = past_key_values[layer_idx][2:]
                    cache.cross_attention_cache.update(key_states, value_states, layer_idx)
                    cache.is_updated[layer_idx] = True
        return cache

    def get_seq_length(self, layer_idx: Optional[int] = 0, cache_position=None) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # check if empty list because in case of static cache it will be a tensors and we can't check `if not torch.Tensor`
        return self.self_attention_cache.get_seq_length(layer_idx, cache_position)


# TODO:This function will be depercated in future.
class QEffHybridCache(QEffDynamicCache):
    def __init__(self, config, batch_size, max_cache_len):
        layer_classes = [QEffHybridCacheLayer] * config.num_hidden_layers
        self.layers = []
        self.layer_classes = layer_classes
        self.max_cache_len = max_cache_len

    @classmethod
    def from_legacy_cache(
        cls, config, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "QEffHybridCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls(config, batch_size=past_key_values[0][0].shape[0], max_cache_len=past_key_values[0][0].shape[2])
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


class QEffHybridCacheLayer(QEffDynamicLayer):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            k_out, v_out = self.keys, self.values
        else:
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)
            sliding_window_pattern = cache_kwargs.get("sliding_window_pattern")

            # remove layer_idx
            is_sliding_layer = torch.tensor(bool((layer_idx + 1) % sliding_window_pattern))

            layer_ctx_len = self.keys.shape[2]

            if is_sliding_layer:
                kv_position_ids = torch.where(position_ids == -1, position_ids, position_ids % (layer_ctx_len - 1))

                kv_position_ids = torch.where(
                    position_ids.max() >= (layer_ctx_len - 1) * 2, (position_ids + 1) % layer_ctx_len, kv_position_ids
                )
            else:
                kv_position_ids = position_ids

            valid_mask = (kv_position_ids != -1).unsqueeze(1).unsqueeze(-1)
            key_states = torch.where(valid_mask == 1, key_states, torch.zeros_like(key_states))
            value_states = torch.where(valid_mask == 1, value_states, torch.zeros_like(value_states))

            # Scatter
            if batch_index is not None:
                if torch.onnx.is_in_onnx_export():
                    scatter_position_ids = torch.where(kv_position_ids < 0, INVALID_IDX, kv_position_ids)
                else:
                    scatter_position_ids = kv_position_ids
                self.keys = CtxScatterFuncCB.apply(self.keys, batch_index, scatter_position_ids, key_states)
                self.values = CtxScatterFuncCB.apply(self.values, batch_index, scatter_position_ids, value_states)
            else:
                self.keys = CtxScatterFunc.apply(self.keys, kv_position_ids, key_states)
                self.values = CtxScatterFunc.apply(self.values, kv_position_ids, value_states)
            k_out, v_out = self.keys, self.values

            # Gather
            ctx_len = k_out.shape[2]
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit

            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = INVALID_IDX
            else:
                invalid_idx_value = 0

            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            if batch_index is not None:
                k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices)
                v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices)
            else:
                k_out = CtxGatherFunc.apply(k_out, ctx_indices)
                v_out = CtxGatherFunc.apply(v_out, ctx_indices)
            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
        return k_out, v_out


# TODO:This function will be depercated in future.
class QEffHybridChunkedCache(QEffDynamicCache):
    def __init__(self, config, max_batch_size, max_cache_len):
        layer_classes = [QEffHybridChunkedLayer] * config.num_hidden_layers
        self.layers = []
        self.layer_classes = layer_classes
        self.max_cache_len = max_cache_len

    @classmethod
    def from_legacy_cache(
        cls, config, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "QEffHybridChunkedCache":
        """Converts a cache in the legacy cache format into an equivalent `HybridChunkedCache`. Used for
        backward compatibility."""
        cache = cls(config, max_batch_size=past_key_values[0][0].shape[0], max_cache_len=past_key_values[0][0].shape[2])
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


class QEffHybridChunkedLayer(QEffDynamicLayer):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            k_out, v_out = self.keys, self.values
        else:
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)

            # handle layer_idx and self.is_sliding
            is_sliding_layer = torch.tensor(bool(self.is_sliding[layer_idx]))

            # Update the position_ids to handle the sliding window
            layer_ctx_len = self.keys.shape[2]

            if is_sliding_layer:
                kv_position_ids = torch.where(position_ids == -1, position_ids, position_ids % (layer_ctx_len - 1))

                kv_position_ids = torch.where(
                    position_ids.max() >= (layer_ctx_len - 1) * 2, (position_ids + 1) % layer_ctx_len, kv_position_ids
                )
            else:
                kv_position_ids = position_ids

            valid_mask = (kv_position_ids != -1).unsqueeze(1).unsqueeze(-1)
            key_states = torch.where(valid_mask == 1, key_states, torch.zeros_like(key_states))
            value_states = torch.where(valid_mask == 1, value_states, torch.zeros_like(value_states))

            # Scatter
            if batch_index is not None:
                if torch.onnx.is_in_onnx_export():
                    scatter_position_ids = torch.where(kv_position_ids < 0, INVALID_IDX, kv_position_ids)
                else:
                    scatter_position_ids = kv_position_ids
                self.keys = CtxScatterFuncCB.apply(self.keys, batch_index, scatter_position_ids, key_states)
                self.values = CtxScatterFuncCB.apply(self.values, batch_index, scatter_position_ids, value_states)
            else:
                self.keys = CtxScatterFunc.apply(self.keys, kv_position_ids, key_states)
                self.values = CtxScatterFunc.apply(self.values, kv_position_ids, value_states)
            k_out, v_out = self.keys, self.values

            # Gather
            ctx_len = min(layer_ctx_len, k_out.shape[2])
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit
            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = INVALID_IDX
            else:
                invalid_idx_value = 0
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            if batch_index is not None:
                k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices)
                v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices)
            else:
                k_out = CtxGatherFunc.apply(k_out, ctx_indices)
                v_out = CtxGatherFunc.apply(v_out, ctx_indices)
            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
        return k_out, v_out


# This is a hack for now, until we get to merging this code with HybridCache class,
# We don't really need to inherit transformers classes as their cache classes are made to work with pytorch and
# ours are made to work with AIC
class QEffHybridCacheForGPTOSS(QEffDynamicCache):
    def __init__(self, config, max_cache_len, sliding_window_len):
        layer_classes = [QEffGPTOSSLayer] * config.num_hidden_layers
        self.layers = []
        self.layer_classes = layer_classes
        self.max_cache_len = max_cache_len
        self.sliding_window_len = sliding_window_len

    @classmethod
    def from_legacy_cache(
        cls, config, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "QEffHybridCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls(
            config,
            max_cache_len=past_key_values[1][0].shape[2],
            sliding_window_len=past_key_values[0][0].shape[2],
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


class QEffGPTOSSLayer(QEffDynamicLayer):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            k_out, v_out = self.keys, self.values
        else:
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)
            is_sliding_layer = cache_kwargs.get("is_sliding")
            sliding_window = cache_kwargs.get("sliding_window")

            if is_sliding_layer:
                kv_position_ids = torch.where(position_ids == -1, position_ids, position_ids % sliding_window)
            else:
                kv_position_ids = position_ids

            # Scatter
            if batch_index is not None:
                if torch.onnx.is_in_onnx_export():
                    scatter_position_ids = torch.where(kv_position_ids < 0, INVALID_IDX, kv_position_ids)
                else:
                    scatter_position_ids = kv_position_ids
                self.keys = CtxScatterFuncCB.apply(self.keys, batch_index, scatter_position_ids, key_states)
                self.values = CtxScatterFuncCB.apply(self.values, batch_index, scatter_position_ids, value_states)
            else:
                self.keys = CtxScatterFunc.apply(self.keys, kv_position_ids, key_states)
                self.values = CtxScatterFunc.apply(self.values, kv_position_ids, value_states)
            k_out, v_out = self.keys, self.values

            # Gather
            ctx_len = k_out.shape[2]
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit
            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = INVALID_IDX
            else:
                invalid_idx_value = 0
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            if batch_index is not None:
                k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices)
                v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices)
            else:
                k_out = CtxGatherFunc.apply(k_out, ctx_indices)
                v_out = CtxGatherFunc.apply(v_out, ctx_indices)
            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
        return k_out, v_out
