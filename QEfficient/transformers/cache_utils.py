# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, EncoderDecoderCache, HybridCache, HybridChunkedCache

from QEfficient.customop import (
    CtxGatherFunc,
    CtxGatherFunc3D,
    CtxGatherFuncBlockedKV,
    CtxGatherFuncBlockedKVCB,
    CtxGatherFuncCB,
    CtxGatherFuncCB3D,
    CtxScatterFunc,
    CtxScatterFunc3D,
    CtxScatterFuncCB,
    CtxScatterFuncCB3D,
)


class InvalidIndexProvider:
    SUBFUNC_ENABLED = False

    @classmethod
    def enable_subfunc(cls):
        cls.SUBFUNC_ENABLED = True

    @classmethod
    def _get_invalid_idx_value(cls):
        """
        Get the appropriate invalid index value for CtxGather operations.

        For ONNX export with functions, we use 0 to avoid INT32_MAX constants
        that cause issues when functions are inlined at runtime.

        Returns:
            int: Invalid index value (0 for ONNX functions, INT32_MAX otherwise)
        """
        if torch.onnx.is_in_onnx_export():
            if cls.SUBFUNC_ENABLED:
                return 0
            else:
                return torch.iinfo(torch.int32).max
        else:
            return 0


class QEffDynamicLayer(DynamicLayer):
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
        ctx_len = cache_kwargs.get("CCL", k_out.shape[2])

        ctx_indices = torch.arange(ctx_len)[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit

        invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()

        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices, ctx_len)
            v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices, ctx_len)
        else:
            k_out = CtxGatherFunc.apply(k_out, ctx_indices, ctx_len)
            v_out = CtxGatherFunc.apply(v_out, ctx_indices, ctx_len)

        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)

    def read_only_blockedKV(self, start_index, end_index, cache_kwargs):
        """
        Reads the `key_states` and `value_states` for the layer for each KV block.

        Parameters:
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

            start_index (`int`):
                Start index of the K/V block to read

            end_index (`int`):
                End index of the K/V block to read

        Return:
            A tuple containing the updated key and value states.
        """
        # Gather
        k_out, v_out = self.keys, self.values
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)
        batch, num_kv_heads, _, _ = k_out.shape
        ctx_indices = torch.arange(start=start_index, end=end_index)[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit

        if torch.onnx.is_in_onnx_export():
            invalid_idx_value = torch.iinfo(torch.int32).max
        else:
            invalid_idx_value = 0

        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_out = CtxGatherFuncBlockedKVCB.apply(k_out, batch_index, ctx_indices)
            v_out = CtxGatherFuncBlockedKVCB.apply(v_out, batch_index, ctx_indices)
        else:
            ctx_indices = ctx_indices.expand(batch, num_kv_heads, ctx_indices.shape[-1])
            k_out = CtxGatherFuncBlockedKV.apply(k_out, ctx_indices)
            v_out = CtxGatherFuncBlockedKV.apply(v_out, ctx_indices)

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
            batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value form the kwargs

            # Scatter
            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

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
            batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value form the kwargs

            # Scatter
            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

                self.keys = CtxScatterFuncCB.apply(self.keys, batch_index, scatter_position_ids, key_states)

                self.values = CtxScatterFuncCB.apply(self.values, batch_index, scatter_position_ids, value_states)
            else:
                self.keys = CtxScatterFunc.apply(self.keys, position_ids, key_states)
                self.values = CtxScatterFunc.apply(self.values, position_ids, value_states)

            k_out, v_out = self.keys, self.values

            # Gather
            ctx_len = cache_kwargs.get("CCL", k_out.shape[2])
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit

            invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()

            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
            if batch_index is not None:
                k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices, ctx_len)
                v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices, ctx_len)
            else:
                k_out = CtxGatherFunc.apply(k_out, ctx_indices, ctx_len)
                v_out = CtxGatherFunc.apply(v_out, ctx_indices, ctx_len)
            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)

        return k_out, v_out

    # TODO:This function will be depercated in future.
    def update3D(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

                self.keys = CtxScatterFuncCB3D.apply(self.keys, batch_index, scatter_position_ids, key_states)

                self.values = CtxScatterFuncCB3D.apply(self.values, batch_index, scatter_position_ids, value_states)
            else:
                self.keys = CtxScatterFunc3D.apply(self.keys, position_ids, key_states)
                self.values = CtxScatterFunc3D.apply(self.values, position_ids, value_states)

            k_out, v_out = self.keys, self.values

            # Gather
            ctx_len = k_out.shape[1]
            ctx_indices = torch.arange(ctx_len)[None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values
            invalid_mask = ctx_indices > gather_limit
            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = torch.iinfo(torch.int32).max
            else:
                invalid_idx_value = 0
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
            if batch_index is not None:
                k_out = CtxGatherFuncCB3D.apply(k_out, batch_index, ctx_indices)
                v_out = CtxGatherFuncCB3D.apply(v_out, batch_index, ctx_indices)
            else:
                k_out = CtxGatherFunc3D.apply(k_out, ctx_indices)
                v_out = CtxGatherFunc3D.apply(v_out, ctx_indices)

            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)

        return k_out, v_out


class QEffDynamicCache(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    - Optimized implementation for the Cloud AI 100 to reuse KV Cache.
    - get the position_ids input using kwargs.
    - Use custom Onnxscript ops to write optimized version to generate Onnx model.

    """

    def __init__(self, ddp_cache_data: Optional[Iterable[tuple[torch.Tensor, torch.Tensor]]] = None, *args, **kwargs):
        # Remove layer_classes if present to avoid duplicate argument
        kwargs.pop("layer_classes", None)
        from transformers.cache_utils import Cache  # Import here to avoid circular import

        Cache.__init__(self, layer_classes=QEffDynamicLayer, *args, **kwargs)
        if ddp_cache_data is not None:
            for key_states, value_states in ddp_cache_data:
                self.layers.append(QEffDynamicLayer.from_tensors(key_states, value_states))

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

    def read_only_blockedKV(self, start_index, end_index, layer_idx, cache_kwargs):
        """
        Reads the `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            start_index (`int`):
                Start index of the K/V block to read
            end_index (`int`):
                End index of the K/V block to read
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        return self.layers[layer_idx].read_only_blockedKV(start_index, end_index, cache_kwargs)

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

    # TODO:This function will be depercated in future.
    def update3D(
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
        return self.layers[layer_idx].update3D(key_states, value_states, cache_kwargs)


class QEffEncoderDecoderCache(EncoderDecoderCache):
    """
    Updated the `EncoderDecoderCache` to use the `QEffDynamicCache` for both self-attention and cross-attention caches.
    """

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "EncoderDecoderCache":
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


# TODO:This function will be depercated in future.
class QEffHybridCache(HybridCache):
    def __init__(self, config, batch_size, max_cache_len):
        super().__init__(config, batch_size, max_cache_len=max_cache_len)
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    @classmethod
    def from_legacy_cache(
        cls, config, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "HybridCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls(config, batch_size=past_key_values[0][0].shape[0], max_cache_len=past_key_values[0][0].shape[2])
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states
        else:
            position_ids = cache_kwargs.get("position_ids")
            sliding_window_pattern = cache_kwargs.get("sliding_window_pattern")
            is_sliding_layer = torch.tensor(bool((layer_idx + 1) % sliding_window_pattern))
            layer_ctx_len = self.key_cache[layer_idx].shape[2]
            kv_position_ids = torch.where(
                (~is_sliding_layer | (position_ids == -1)), position_ids, position_ids % (layer_ctx_len - 1)
            )

            kv_position_ids = torch.where(
                is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1) * 2),
                (position_ids + 1) % layer_ctx_len,
                kv_position_ids,
            )

            valid_mask = (kv_position_ids != -1).unsqueeze(1).unsqueeze(-1)
            key_states = torch.where(valid_mask == 1, key_states, torch.zeros_like(key_states))
            value_states = torch.where(valid_mask == 1, value_states, torch.zeros_like(value_states))
            self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], kv_position_ids, key_states)
            self.value_cache[layer_idx] = CtxScatterFunc.apply(
                self.value_cache[layer_idx], kv_position_ids, value_states
            )
            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Original Gather
            ctx_len = cache_kwargs.get("CCL", self.key_cache[layer_idx].shape[2])
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = kv_position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit
            invalid_idx_value = InvalidIndexProvider._get_invalid_idx_value()
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            all_indices = torch.arange(layer_ctx_len) + kv_position_ids.max() + 1
            rolling_indices = torch.where(all_indices > layer_ctx_len - 1, all_indices % layer_ctx_len, all_indices)
            rolling_indices = rolling_indices[:ctx_len]
            final_indices = torch.where(
                (is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1))), rolling_indices, ctx_indices
            )
            k_out = CtxGatherFunc.apply(k_out, final_indices, ctx_len)
            v_out = CtxGatherFunc.apply(v_out, final_indices, ctx_len)
            ctx_v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
            v_out = torch.where((is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1))), v_out, ctx_v_out)
        return k_out, v_out


# TODO:This function will be depercated in future.
class QEffHybridChunkedCache(HybridChunkedCache):
    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `HybridChunkedCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, config, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "HybridChunkedCache":
        """Converts a cache in the legacy cache format into an equivalent `HybridChunkedCache`. Used for
        backward compatibility."""
        cache = cls(config, max_batch_size=past_key_values[0][0].shape[0], max_cache_len=past_key_values[0][0].shape[2])
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states

        else:
            position_ids = cache_kwargs.get("position_ids")
            is_sliding_layer = torch.tensor(bool(self.is_sliding[layer_idx]))

            # Update the position_ids to handle the sliding window
            layer_ctx_len = self.key_cache[layer_idx].shape[2]
            kv_position_ids = torch.where(
                (~is_sliding_layer | (position_ids == -1)), position_ids, position_ids % (layer_ctx_len - 1)
            )

            kv_position_ids = torch.where(
                is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1) * 2),
                (position_ids + 1) % layer_ctx_len,
                kv_position_ids,
            )

            valid_mask = (kv_position_ids != -1).unsqueeze(1).unsqueeze(-1)
            key_states = torch.where(valid_mask == 1, key_states, torch.zeros_like(key_states))
            value_states = torch.where(valid_mask == 1, value_states, torch.zeros_like(value_states))
            self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], kv_position_ids, key_states)
            self.value_cache[layer_idx] = CtxScatterFunc.apply(
                self.value_cache[layer_idx], kv_position_ids, value_states
            )
            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Original Gather
            ctx_len = cache_kwargs.get("CCL", k_out.shape[2])
            ctx_len = min(layer_ctx_len, ctx_len)
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = kv_position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit
            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = torch.iinfo(torch.int32).max
            else:
                invalid_idx_value = 0
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            # Rolling indices for sliding window
            all_indices = torch.arange(layer_ctx_len) + kv_position_ids.max() + 1
            rolling_indices = torch.where(all_indices > layer_ctx_len - 1, all_indices % layer_ctx_len, all_indices)
            rolling_indices = rolling_indices[:ctx_len]
            final_indices = torch.where(
                (is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1))), rolling_indices, ctx_indices
            )
            k_out = CtxGatherFunc.apply(k_out, final_indices, ctx_len)
            v_out = CtxGatherFunc.apply(v_out, final_indices, ctx_len)
            ctx_v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
            v_out = torch.where((is_sliding_layer & (position_ids.max() >= (layer_ctx_len - 1))), v_out, ctx_v_out)
        return k_out, v_out


# This is a hack for now, until we get to merging this code with HybridCache class,
# We don't really need to inherit transformers classes as their cache classes are made to work with pytorch and
# ours are made to work with AIC
class QEffHybridCacheForGPTOSS:
    def __init__(self, config, batch_size, max_cache_len, sliding_window_len):
        self.max_cache_len = max_cache_len
        self.batch_size = batch_size
        self.sliding_window_len = sliding_window_len
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    @classmethod
    def from_legacy_cache(
        cls, config, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "HybridCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls(
            config,
            batch_size=past_key_values[0][0].shape[0],
            max_cache_len=past_key_values[1][0].shape[2],
            sliding_window_len=past_key_values[0][0].shape[2],
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states
        else:
            position_ids = cache_kwargs.get("position_ids")
            is_sliding_layer = cache_kwargs.get("is_sliding")
            sliding_window = cache_kwargs.get("sliding_window")
            batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value from the kwargs

            if is_sliding_layer:
                kv_position_ids = torch.where(position_ids == -1, position_ids, position_ids % sliding_window)
            else:
                kv_position_ids = position_ids

            if batch_index is not None:
                if torch.onnx.is_in_onnx_export():
                    invalid_scatter_index = torch.iinfo(torch.int32).max
                    scatter_position_ids = torch.where(kv_position_ids < 0, invalid_scatter_index, kv_position_ids)
                else:
                    scatter_position_ids = kv_position_ids
                self.key_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
                )
                self.value_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
                )
            else:
                self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], kv_position_ids, key_states)
                self.value_cache[layer_idx] = CtxScatterFunc.apply(
                    self.value_cache[layer_idx], kv_position_ids, value_states
                )

            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Original Gather
            if is_sliding_layer:
                ctx_len = self.key_cache[layer_idx].shape[2]
            else:
                ctx_len = cache_kwargs.get("CCL", self.key_cache[layer_idx].shape[2])

            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit
            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = torch.iinfo(torch.int32).max
            else:
                invalid_idx_value = 0
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

            if batch_index is not None:
                k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices, ctx_len)
                v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices, ctx_len)
            else:
                k_out = CtxGatherFunc.apply(k_out, ctx_indices, ctx_len)
                v_out = CtxGatherFunc.apply(v_out, ctx_indices, ctx_len)

            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
        return k_out, v_out
